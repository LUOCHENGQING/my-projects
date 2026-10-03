"""引用可追溯性测试：结论 -> 证据 -> 资料编号 -> 文件，全链路可回溯。"""

from __future__ import annotations

import re
from pathlib import Path

from src.tools import PermissionLevel

QUESTION = "请分析示例科技股份有限公司 2024 年度的盈利能力、偿债能力和现金流质量，并提示主要风险。"


def _cited_numbers(report: str) -> set:
    return {int(m) for m in re.findall(r"\[(\d+)\]", report or "")}


# ---------------------------------------------------------------------------
# 1. 报告中的每个 [n] 都必须能回溯
# ---------------------------------------------------------------------------
def test_every_citation_number_in_report_resolves(pipeline):
    result = pipeline.run(QUESTION, run_id="run-cite")
    state = result["state"]
    report = state["report"]
    citations = state["citations"]

    used = _cited_numbers(report)
    known = {int(c["citation_no"]) for c in citations}
    assert used, "报告中至少应出现一处引用编号"
    assert used <= known, f"存在悬空引用：{sorted(used - known)}"


def test_every_conclusion_carries_a_citation(pipeline):
    """硬性要求：每条核心结论都必须带引用编号。"""
    result = pipeline.run(QUESTION, run_id="run-cite-conclusions")
    report = result["state"]["report"]

    conclusions = []
    in_section = False
    for line in report.splitlines():
        if line.startswith("## ") and "核心结论" in line:
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and re.match(r"^\d+\.\s", line.strip()):
            conclusions.append(line.strip())

    assert conclusions, "应当解析出核心结论条目"
    for item in conclusions:
        assert re.search(r"\[\d+\]", item), f"该结论缺少引用编号：{item}"


def test_every_finding_is_backed_by_existing_evidence(pipeline):
    result = pipeline.run(QUESTION, run_id="run-cite-evidence")
    state = result["state"]
    evidence_ids = {e["child_id"] for e in state["evidence"]}
    for finding in state["findings"]:
        assert finding["evidence_ids"], f"{finding['id']} 没有证据支撑"
        assert set(finding["evidence_ids"]) <= evidence_ids


# ---------------------------------------------------------------------------
# 2. 引用记录本身指向真实存在的文件与资料编号
# ---------------------------------------------------------------------------
def test_citations_point_to_real_documents(pipeline, documents):
    result = pipeline.run(QUESTION, run_id="run-cite-files")
    citations = result["state"]["citations"]
    assert citations, "应当产生引用记录"
    known_sources = set(documents.source_ids)
    for item in citations:
        assert item["source_id"] in known_sources
        assert item["section"], "引用应携带具体章节"
        assert Path(item["path"]).is_file(), f"引用指向的文件不存在：{item['path']}"
        assert item["company"] and item["period"]


def test_report_contains_citation_section_listing_all_sources(pipeline):
    result = pipeline.run(QUESTION, run_id="run-cite-section")
    state = result["state"]
    report = state["report"]
    assert "引用来源" in report
    for item in state["citations"]:
        assert f"[{item['citation_no']}]" in report
        assert item["source_id"] in report


# ---------------------------------------------------------------------------
# 3. cite_source 工具本身的边界
# ---------------------------------------------------------------------------
def test_cite_source_rejects_unknown_source_id(registry):
    result = registry.call("cite_source", {"source_id": "NOT-EXIST-2024"},
                           granted={PermissionLevel.WRITE})
    assert result.ok is False
    assert result.error["code"] == "EXECUTION_ERROR"


def test_cite_source_assigns_increasing_numbers(registry):
    first = registry.call("cite_source", {"source_id": "EX-BANK-2024-AR"},
                          granted={PermissionLevel.WRITE})
    second = registry.call("cite_source", {"source_id": "EX-TECH-2024-Q3"},
                           granted={PermissionLevel.WRITE})
    assert first.data["citation_no"] == 1
    assert second.data["citation_no"] == 2


def test_only_writer_agent_holds_write_permission(pipeline):
    """最小权限：只有 WriterAgent 持有 write 权限，其余 Agent 拿不到 cite_source。"""
    from src.agents import AnalystAgent, PlannerAgent, RetrieverAgent, RiskCheckerAgent, WriterAgent
    from src.agents.base import AgentContext
    from src.llm.client import LLMClient
    from src.tracing import TraceRecorder

    ctx = AgentContext(
        registry=pipeline.registry,
        llm=LLMClient(force_mock=True),
        recorder=TraceRecorder("perm-test", runs_dir=pipeline.runs_dir),
        config=pipeline.config,
    )
    for agent in (PlannerAgent(ctx), RetrieverAgent(ctx), AnalystAgent(ctx), RiskCheckerAgent(ctx)):
        assert "cite_source" not in agent.allowed_tools
        denied = agent.call_tool("cite_source", {"source_id": "EX-TECH-2024-AR"})
        assert denied.ok is False
        assert denied.error["code"] == "TOOL_NOT_ALLOWED"

    writer = WriterAgent(ctx)
    assert "cite_source" in writer.allowed_tools
    assert writer.call_tool("cite_source", {"source_id": "EX-TECH-2024-AR"}).ok is True
