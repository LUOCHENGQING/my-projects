"""引用可追溯性测试：结论 -> 证据 -> 资料编号 -> 文件，全链路可回溯。

被测行为（ResearchPipeline 端到端产物、cite_source 工具、各 Agent 的工具白名单）：
1. 报告正文里出现的每个 [n] 引用编号都必须能在 state["citations"] 中找到记录（不存在悬空引用）；
2. 每条核心结论、每条 finding 都必须有引用 / 证据，且 evidence_ids 指向真实存在的证据块；
3. 引用记录自身的 source_id、章节、磁盘文件路径、公司与期间必须真实存在，且报告带「引用来源」清单；
4. cite_source 工具的边界：未知资料编号被拒、引用序号单调递增、只有 WriterAgent 持有 write 权限。

覆盖策略：正常（真实管线在 mock LLM 下端到端产出报告）、边界（未知资料编号、序号连续性）、
对抗（其余四类 Agent 越权调用 cite_source 必须被拒，最小权限不得被绕过）。
"""

from __future__ import annotations

import re
from pathlib import Path

from src.tools import PermissionLevel

QUESTION = "请分析示例科技股份有限公司 2024 年度的盈利能力、偿债能力和现金流质量，并提示主要风险。"


def _cited_numbers(report: str) -> set:
    """从报告文本里抽出全部 [n] 形式的引用编号（返回 int 集合，便于做包含关系判断）。"""
    return {int(m) for m in re.findall(r"\[(\d+)\]", report or "")}


# ---------------------------------------------------------------------------
# 1. 报告中的每个 [n] 都必须能回溯
# ---------------------------------------------------------------------------
def test_every_citation_number_in_report_resolves(pipeline):
    """验证不变式：报告中出现的每个引用编号都必须能回溯到一条已登记的引用记录，不存在悬空编号。"""
    result = pipeline.run(QUESTION, run_id="run-cite")
    state = result["state"]
    report = state["report"]
    citations = state["citations"]

    # 用集合包含关系而非逐条断言：报告里“用到的编号”必须是“已登记编号”的子集
    used = _cited_numbers(report)
    known = {int(c["citation_no"]) for c in citations}
    assert used, "报告中至少应出现一处引用编号"
    assert used <= known, f"存在悬空引用：{sorted(used - known)}"


def test_every_conclusion_carries_a_citation(pipeline):
    """验证规则：报告「核心结论」小节里的每一条结论都必须带 [n] 引用编号。"""
    result = pipeline.run(QUESTION, run_id="run-cite-conclusions")
    report = result["state"]["report"]

    # 只在「核心结论」小节内按 “N. ” 序号行取样，避免把其它章节的列表误当作结论
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
    """验证不变式：每条分析结论的 evidence_ids 必须非空，且全部指向真实存在的证据块。"""
    result = pipeline.run(QUESTION, run_id="run-cite-evidence")
    state = result["state"]
    # 先把证据块 id 收成集合：后面既要判非空，也要判「子集」关系
    evidence_ids = {e["child_id"] for e in state["evidence"]}
    for finding in state["findings"]:
        assert finding["evidence_ids"], f"{finding['id']} 没有证据支撑"
        assert set(finding["evidence_ids"]) <= evidence_ids


# ---------------------------------------------------------------------------
# 2. 引用记录本身指向真实存在的文件与资料编号
# ---------------------------------------------------------------------------
def test_citations_point_to_real_documents(pipeline, documents):
    """验证不变式：引用记录必须指向真实存在的 source_id 与磁盘文件，且携带章节、公司与期间。"""
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
    """验证规则：报告必须含「引用来源」章节，并逐条列出全部引用的编号与 source_id。"""
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
    """验证边界：引用不存在的资料编号必须被拒绝，不允许凭空生成引用。"""
    # 编号格式合法但库里不存在，因此期望的是执行期校验失败（EXECUTION_ERROR）而非 schema 错误
    result = registry.call("cite_source", {"source_id": "NOT-EXIST-2024"},
                           granted={PermissionLevel.WRITE})
    assert result.ok is False
    assert result.error["code"] == "EXECUTION_ERROR"


def test_cite_source_assigns_increasing_numbers(registry):
    """验证规则：引用编号从 1 起按调用顺序单调递增，不按资料去重、不跳号。"""
    # 两份不同资料各引用一次，用来验证序号是按调用次数累加而非按资料复用
    first = registry.call("cite_source", {"source_id": "EX-BANK-2024-AR"},
                          granted={PermissionLevel.WRITE})
    second = registry.call("cite_source", {"source_id": "EX-TECH-2024-Q3"},
                           granted={PermissionLevel.WRITE})
    assert first.data["citation_no"] == 1
    assert second.data["citation_no"] == 2


def test_only_writer_agent_holds_write_permission(pipeline):
    """验证规则（最小权限）：只有 WriterAgent 持有 write 权限，其余 Agent 既看不到 cite_source，越权调用也必须被拒。"""
    # 就地导入：这些 Agent 只在「权限矩阵」这一个用例里用到
    from src.agents import AnalystAgent, PlannerAgent, RetrieverAgent, RiskCheckerAgent, WriterAgent
    from src.agents.base import AgentContext
    from src.llm.client import LLMClient
    from src.tracing import TraceRecorder

    # 复用 pipeline 的 registry / runs_dir / config：权限判定必须基于同一份工具注册表
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
