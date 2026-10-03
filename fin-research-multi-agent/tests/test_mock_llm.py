"""Mock LLM 测试：无 Key 必须自动降级、输出确定、全流程可跑通、trace 字段齐备。"""

from __future__ import annotations

import json

import pytest

from src.llm.client import LLMClient
from src.llm.mock import run_mock
from src.tracing import load_trace

QUESTION = "请分析示例科技股份有限公司 2024 年度的盈利能力、偿债能力和现金流质量，并提示主要风险。"


# ---------------------------------------------------------------------------
# 1. 无 Key 自动进入 mock 模式
# ---------------------------------------------------------------------------
def test_client_enters_mock_mode_without_api_key(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient()
    assert client.is_mock is True
    assert client.model_name.startswith("mock::")


def test_client_uses_real_mode_when_key_present(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-test")
    client = LLMClient()
    assert client.is_mock is False
    assert client.model_name != "mock::"
    # 但不应该真的发请求：只是实例化
    assert client.call_count == 0


def test_llm_degrades_to_mock_on_api_failure(monkeypatch):
    """真机调用失败时必须自动降级，而不是让整条流水线崩掉。"""
    monkeypatch.setenv("MOCK_LLM", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-test")
    client = LLMClient()

    def boom():
        raise RuntimeError("模拟网络故障")

    monkeypatch.setattr(client, "_ensure_client", boom)
    response = client.chat("plan", {"question": QUESTION, "companies": ["示例科技股份有限公司"], "latest_year": 2024})
    assert response.mocked is True
    assert response.degraded is True
    assert "模拟网络故障" in (response.error or "")
    assert response.data["route"]           # 降级后依然产出可用结果
    assert client.degraded_count == 1


# ---------------------------------------------------------------------------
# 2. 确定性
# ---------------------------------------------------------------------------
def test_mock_output_is_deterministic():
    payload = {
        "question": QUESTION,
        "companies": ["示例科技股份有限公司", "示例智造银行股份有限公司"],
        "latest_year": 2024,
    }
    first = run_mock("plan", payload)
    second = run_mock("plan", payload)
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_mock_plan_extracts_company_year_period_targets():
    data = run_mock(
        "plan",
        {
            "question": "示例科技股份有限公司 2024 年前三季度的成长性和现金流表现如何？",
            "companies": ["示例科技股份有限公司", "示例智造银行股份有限公司"],
            "latest_year": 2024,
        },
    )
    assert data["companies"] == ["示例科技股份有限公司"]
    assert data["year"] == 2024
    assert data["period"] == "三季度"
    assert "成长性" in data["targets"]
    assert "现金流质量" in data["targets"]
    assert data["route"] == ["retriever", "analyst", "risk_checker", "writer"]
    assert len(data["retrieval_queries"]) >= 2
    assert len(data["subtasks"]) == 4


def test_mock_plan_falls_back_when_question_is_vague():
    data = run_mock("plan", {"question": "帮我看看这家公司怎么样", "companies": ["示例科技股份有限公司"], "latest_year": 2024})
    assert data["companies"] == ["示例科技股份有限公司"]
    assert data["targets"]          # 一定有兜底的默认分析维度
    assert "writer" in data["route"]


def test_mock_retrieve_dedupes_by_parent_and_reports_coverage():
    candidates = [
        {"child_id": "c1", "parent_id": "p1", "score": 0.9, "text": "净利润同比增长",
         "section_title": "三、关键财务数据", "matched_terms": ["净利润"]},
        {"child_id": "c2", "parent_id": "p1", "score": 0.8, "text": "净利润同比下降",
         "section_title": "三、关键财务数据", "matched_terms": ["净利润"]},
        {"child_id": "c3", "parent_id": "p2", "score": 0.7, "text": "未决诉讼与担保",
         "section_title": "六、主要风险因素", "matched_terms": ["诉讼"]},
    ]
    data = run_mock("retrieve", {"candidates": candidates, "targets": ["盈利能力", "风险合规"], "limit": 5})
    assert data["selected"] == ["c1", "c3"]          # 同一父块只保留最高分
    assert data["coverage"]["风险合规"] == ["c3"]
    assert data["missing_data"] == [] or isinstance(data["missing_data"], list)


def test_mock_risk_review_follows_deterministic_rules():
    gaps = [{"code": "GAP-EVIDENCE", "problem": "缺少证据", "required_fix": "补证据"}]
    revise = run_mock("risk_review", {
        "gate": {"gaps": gaps, "round": 0, "max_rounds": 2},
        "risk": {"overall_level": "medium", "findings": []},
    })
    assert revise["verdict"] == "revise"

    escalate_by_limit = run_mock("risk_review", {
        "gate": {"gaps": gaps, "round": 2, "max_rounds": 2},
        "risk": {"overall_level": "medium", "findings": []},
    })
    assert escalate_by_limit["verdict"] == "escalate"

    escalate_by_risk = run_mock("risk_review", {
        "gate": {"gaps": [], "round": 0, "max_rounds": 2},
        "risk": {"overall_level": "high", "findings": [{"title": "净利润现金含量过低", "level": "high"}]},
    })
    assert escalate_by_risk["verdict"] == "escalate"

    passed = run_mock("risk_review", {
        "gate": {"gaps": [], "round": 0, "max_rounds": 2},
        "risk": {"overall_level": "low", "findings": []},
    })
    assert passed["verdict"] == "pass"


# ---------------------------------------------------------------------------
# 3. 全流程离线可跑通（mock 模式的核心承诺）
# ---------------------------------------------------------------------------
def test_full_pipeline_runs_offline_in_mock_mode(pipeline):
    assert pipeline.llm.is_mock is True
    result = pipeline.run(QUESTION, run_id="run-mock-e2e")
    state = result["state"]

    assert state["report"], "mock 模式下必须产出完整简报"
    assert len(state["report"]) > 800
    assert state["findings"], "mock 模式下必须产出分析结论"
    assert state["citations"], "mock 模式下必须产出引用"
    assert state["errors"] == []
    assert result["summary"]["status"] == "ok"
    assert result["llm_stats"]["calls"] >= 5   # plan/retrieve/analyze x2/risk_review x2/write
    assert result["llm_stats"]["mock_calls"] == result["llm_stats"]["calls"]


def test_mock_report_has_all_expected_sections(pipeline):
    result = pipeline.run(QUESTION, run_id="run-mock-sections")
    report = result["state"]["report"]
    for section in ("核心结论", "关键财务指标", "关键比率指标", "分析与论证", "风险提示", "引用来源", "免责声明"):
        assert section in report, f"报告缺少章节：{section}"


# ---------------------------------------------------------------------------
# 4. trace 的固定字段
# ---------------------------------------------------------------------------
def test_trace_jsonl_has_required_fields(pipeline):
    result = pipeline.run(QUESTION, run_id="run-mock-trace")
    trace = load_trace("run-mock-trace", runs_dir=pipeline.runs_dir)
    assert len(trace["steps"]) == len(result["state"]["steps"])

    required = {"step", "agent", "input_digest", "output_digest", "latency_ms", "status"}
    for entry in trace["entries"]:
        assert required <= set(entry), f"trace 行缺少字段：{required - set(entry)}"
        assert isinstance(entry["latency_ms"], (int, float))
        assert len(entry["input_digest"]) == 16
        assert len(entry["output_digest"]) == 16
        assert entry["status"] in {"ok", "error", "skipped"}

    # step 单调递增
    steps = [e["step"] for e in trace["entries"]]
    assert steps == sorted(steps)
    assert steps == list(range(1, len(steps) + 1))


def test_trace_records_tool_calls_per_step(pipeline):
    result = pipeline.run(QUESTION, run_id="run-mock-tools")
    trace = load_trace("run-mock-tools", runs_dir=pipeline.runs_dir)
    tool_names = {
        call["tool"]
        for entry in trace["steps"]
        for call in (entry.get("extra") or {}).get("tool_calls", [])
    }
    assert {"search_filings", "get_financial_metric", "calc_ratio", "check_risk_rules", "cite_source"} <= tool_names


def test_replay_can_read_the_trace(pipeline, capsys):
    from src.replay import render

    result = pipeline.run(QUESTION, run_id="run-mock-replay")
    code = render("run-mock-replay", runs_dir=pipeline.runs_dir)
    assert code == 0
    out = capsys.readouterr().out
    assert "执行回放" in out
    assert "PlannerAgent" in out
    assert "WriterAgent" in out
    assert result["run_id"] in out
