"""Mock LLM 测试：无 Key 必须自动降级、输出确定、全流程可跑通、trace 字段齐备。

被测行为（src.llm.client.LLMClient、src.llm.mock.run_mock、src.tracing、src.replay）：
1. 模式选择：无 Key（即便 MOCK_LLM=0）自动进 mock；有 Key 时进真实模式但实例化不发请求；真机调用失败自动降级且仍产出可用结果；
2. 确定性：同一 (route, payload) 的 mock 输出逐字节一致；plan 能抽取公司 / 年份 / 报告期 / 分析维度，问题笼统时回落到默认维度；
3. mock 子路由规则：retrieve 按父块去重并回报维度覆盖，risk_review 的 revise / escalate / pass 判定；
4. 端到端：mock 模式下离线跑通完整管线，报告章节完备、零错误、调用全部来自 mock；
5. 可观测性：trace JSONL 字段齐备且 step 连续、逐步骤记录工具调用、replay 能读取 trace 完成回放。

覆盖策略：正常（离线端到端）、边界（无 Key、笼统问题、未传参数走默认）、
异常（真机 API 故障必须降级而不是崩溃）、对抗（monkeypatch 篡改环境变量与 _ensure_client，验证降级路径真的生效）。
"""

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
    """验证规则：显式关闭 MOCK_LLM 且没有任何 API Key 时，客户端仍必须自动降级为 mock 模式。"""
    # 先关环境开关再删 Key：模拟「新机器尚未配置凭据」这一首要使用场景
    monkeypatch.setenv("MOCK_LLM", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient()
    assert client.is_mock is True
    assert client.model_name.startswith("mock::")


def test_client_uses_real_mode_when_key_present(monkeypatch):
    """验证规则：有 Key 且未强制 mock 时进入真实模式，但仅实例化不得发起任何调用。"""
    monkeypatch.setenv("MOCK_LLM", "0")
    # 假 Key 只用于让分支判定成立，用例不会真的发出请求
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-test")
    client = LLMClient()
    assert client.is_mock is False
    assert client.model_name != "mock::"
    # 但不应该真的发请求：只是实例化
    assert client.call_count == 0


def test_llm_degrades_to_mock_on_api_failure(monkeypatch):
    """验证规则：真机调用失败时必须自动降级为 mock（标记 degraded 并保留错误原因），而不是让整条流水线崩掉。"""
    monkeypatch.setenv("MOCK_LLM", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-test")
    client = LLMClient()

    def boom():
        """桩函数：模拟真实 LLM 调用抛异常，用于验证自动降级路径。"""
        raise RuntimeError("模拟网络故障")

    # 替换客户端构造：制造必然失败的真机调用，免去对真实网络的依赖
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
    """验证性质：同一 route 与 payload 的 mock 输出必须完全一致，保证用例可复现、可回归。"""
    # payload 固定（含两家公司）后连调两次，比较原始返回值本身而非序列化文本
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
    """验证规则：plan 抽出的公司、年份、报告期、分析维度，以及固定路由与子任务数都必须正确。"""
    # 问题同时出现两家公司与「前三季度」：用来验证只保留被问到的公司、并识别出报告期
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
    """验证边界：问题笼统、没给出分析维度时，plan 必须回落到默认维度而不是产出空计划。"""
    # 刻意不写年份与维度，模拟“帮我看看这家公司怎么样”这类模糊提问
    data = run_mock("plan", {"question": "帮我看看这家公司怎么样", "companies": ["示例科技股份有限公司"], "latest_year": 2024})
    assert data["companies"] == ["示例科技股份有限公司"]
    assert data["targets"]          # 一定有兜底的默认分析维度
    assert "writer" in data["route"]


def test_mock_retrieve_dedupes_by_parent_and_reports_coverage():
    """验证规则：retrieve 结果按父块去重（同父块只留最高分），并按分析维度回报覆盖情况。"""
    # c1/c2 同属父块 p1 且同为「净利润」命中，专门用来验证同父块只保留最高分那条
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
    """验证规则：risk_review 的裁决只由（有无缺口、已用轮次是否达上限、风险等级）决定——有缺口未超限即 revise，超限或高风险即 escalate，无缺口且低风险才 pass。"""
    # 四个场景共用同一组缺口，只改「轮次 / 风险等级」，用来验证裁决分支与优先级
    gaps = [{"code": "GAP-EVIDENCE", "problem": "缺少证据", "required_fix": "补证据"}]
    # 场景 1：有缺口且未超轮次 -> 打回重算
    revise = run_mock("risk_review", {
        "gate": {"gaps": gaps, "round": 0, "max_rounds": 2},
        "risk": {"overall_level": "medium", "findings": []},
    })
    assert revise["verdict"] == "revise"

    # 场景 2：轮次已用满（round == max_rounds）-> 超限升级人工
    escalate_by_limit = run_mock("risk_review", {
        "gate": {"gaps": gaps, "round": 2, "max_rounds": 2},
        "risk": {"overall_level": "medium", "findings": []},
    })
    assert escalate_by_limit["verdict"] == "escalate"

    # 场景 3：缺口已清空但风险等级为 high -> 同样升级人工
    escalate_by_risk = run_mock("risk_review", {
        "gate": {"gaps": [], "round": 0, "max_rounds": 2},
        "risk": {"overall_level": "high", "findings": [{"title": "净利润现金含量过低", "level": "high"}]},
    })
    assert escalate_by_risk["verdict"] == "escalate"

    # 场景 4：无缺口且低风险 -> 直接放行
    passed = run_mock("risk_review", {
        "gate": {"gaps": [], "round": 0, "max_rounds": 2},
        "risk": {"overall_level": "low", "findings": []},
    })
    assert passed["verdict"] == "pass"


# ---------------------------------------------------------------------------
# 3. 全流程离线可跑通（mock 模式的核心承诺）
# ---------------------------------------------------------------------------
def test_full_pipeline_runs_offline_in_mock_mode(pipeline):
    """验证承诺：mock 模式下整条管线离线可跑通——零错误、产出报告 / 结论 / 引用，且统计到的调用全部来自 mock。"""
    # 先确认 fixture 确实处于 mock 模式，避免用例在联网环境下“假通过”
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
    """验证规则：mock 简报必须包含约定的全部固定章节（核心结论到免责声明）。"""
    result = pipeline.run(QUESTION, run_id="run-mock-sections")
    report = result["state"]["report"]
    for section in ("核心结论", "关键财务指标", "关键比率指标", "分析与论证", "风险提示", "引用来源", "免责声明"):
        assert section in report, f"报告缺少章节：{section}"


# ---------------------------------------------------------------------------
# 4. trace 的固定字段
# ---------------------------------------------------------------------------
def test_trace_jsonl_has_required_fields(pipeline):
    """验证契约：trace 每条记录字段齐备、摘要定长 16、latency 为数值、status 属于枚举，且 step 从 1 连续递增并与状态步数一致。"""
    result = pipeline.run(QUESTION, run_id="run-mock-trace")
    trace = load_trace("run-mock-trace", runs_dir=pipeline.runs_dir)
    assert len(trace["steps"]) == len(result["state"]["steps"])

    # 这六个字段是审计与回放的最小集合，缺任意一个都无法复盘一次执行
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
    """验证规则：trace 必须在步骤维度记录实际调用过的工具，五个业务工具都要留痕。"""
    result = pipeline.run(QUESTION, run_id="run-mock-tools")
    trace = load_trace("run-mock-tools", runs_dir=pipeline.runs_dir)
    # extra.tool_calls 可能缺失，统一用 get 兜底后再对全部步骤取并集
    tool_names = {
        call["tool"]
        for entry in trace["steps"]
        for call in (entry.get("extra") or {}).get("tool_calls", [])
    }
    assert {"search_filings", "get_financial_metric", "calc_ratio", "check_risk_rules", "cite_source"} <= tool_names


def test_replay_can_read_the_trace(pipeline, capsys):
    """验证规则：replay 能读取落盘的 trace 并成功渲染一次回放（退出码 0，输出含关键 Agent 与 run_id）。"""
    # 就地导入：避免 replay 模块成为整份测试的导入期依赖
    from src.replay import render

    result = pipeline.run(QUESTION, run_id="run-mock-replay")
    code = render("run-mock-replay", runs_dir=pipeline.runs_dir)
    assert code == 0
    out = capsys.readouterr().out
    assert "执行回放" in out
    assert "PlannerAgent" in out
    assert "WriterAgent" in out
    assert result["run_id"] in out
