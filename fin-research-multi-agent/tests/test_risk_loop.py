"""RiskCheckerAgent 反思循环测试：打回、重算、超限升级人工确认。"""

from __future__ import annotations

from src.agents.risk_checker import RiskCheckerAgent
from src.state import new_state

QUESTION = "请分析示例科技股份有限公司 2024 年度的盈利能力和现金流质量，并提示主要风险。"


def _base_state(**overrides):
    state = new_state("测试问题", "run-risk", config={}, max_revision_rounds=2)
    state["plan"] = {
        "companies": ["示例科技股份有限公司"],
        "year": 2024,
        "period": "年度",
        "targets": ["盈利能力", "现金流质量"],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# 1. 端到端：打回一次后因高风险升级人工确认
# ---------------------------------------------------------------------------
def test_reflection_loop_revises_then_escalates(pipeline):
    result = pipeline.run(QUESTION, run_id="run-loop")
    state = result["state"]
    visited = [s["agent"] for s in state["steps"]]

    # 反思循环真实发生：analyst 被访问两次（初次 + 重算）
    assert visited.count("analyst") == 2, visited
    assert visited.count("risk_checker") == 2, visited
    assert state["revision_round"] == 1

    # 最终因高风险转人工
    assert state["risk_verdict"] == "escalate"
    assert state["needs_human"] is True
    assert state["human_decision"]["decision"] == "approved"  # auto 模式自动放行
    assert "human_review" in visited


def test_revision_requests_are_actionable(pipeline):
    """打回意见必须是可执行的整改要求，而不是空话。

    注意：核查缺口会被记入 risk_report.gate_history —— 最终收敛为 escalate 时
    当前缺口可能为空，但"曾经因为什么被打回"必须留痕，否则事后无法复盘。
    """
    result = pipeline.run(QUESTION, run_id="run-loop-actionable")
    history = result["state"]["risk_report"]["gate_history"]
    assert len(history) == 2, "本轮应当经历 2 次核查（初次 + 重算后）"
    assert history[0]["verdict"] == "revise"
    assert history[1]["verdict"] == "escalate"

    gaps = history[0]["gaps"]
    assert gaps, "第 1 轮应当至少产生一条核查缺口"
    for gap in gaps:
        assert gap["code"].startswith("GAP-")
        assert gap["problem"]
        assert len(gap["required_fix"]) > 5
        assert gap["severity"] in {"blocker", "high", "medium", "low"}
    # 重算后缺口确实被消除（循环收敛，而不是原地打转）
    assert history[1]["gaps"] == []


def test_analyst_responds_to_revision_with_cross_checks(pipeline):
    """被打回后 Analyst 必须补上交叉验证，否则循环不会收敛。"""
    result = pipeline.run(QUESTION, run_id="run-loop-xcheck")
    findings = result["state"]["findings"]
    assert any(f.get("cross_checks") for f in findings), "重算后应出现交叉验证内容"
    # 重算轮引入了上期对比口径
    assert "receivable_growth" in result["state"]["metrics"]


# ---------------------------------------------------------------------------
# 2. 核查门（gate）的确定性判定
# ---------------------------------------------------------------------------
def test_gate_flags_finding_without_evidence(agent_ctx):
    agent = RiskCheckerAgent(agent_ctx)
    state = _base_state()
    state["evidence"] = [{"child_id": "EX-TECH-2024-AR#2-c0", "source_id": "EX-TECH-2024-AR"}]
    findings = [
        {
            "id": "F1",
            "title": "无证据结论",
            "statement": "净利润大幅增长。",
            "ratio_refs": ["net_margin"],
            "evidence_ids": [],
            "cross_checks": [],
        }
    ]
    gaps = agent._gate(state, findings, {"net_margin": {"value": 0.075}}, {"net_margin": 0.075})
    codes = {g["code"] for g in gaps}
    assert "GAP-EVIDENCE" in codes


def test_gate_flags_finding_without_quantitative_support(agent_ctx):
    agent = RiskCheckerAgent(agent_ctx)
    state = _base_state()
    state["evidence"] = [{"child_id": "c1", "source_id": "EX-TECH-2024-AR"}]
    findings = [
        {"id": "F1", "title": "纯定性结论", "statement": "公司前景良好。",
         "ratio_refs": [], "evidence_ids": ["c1"], "cross_checks": []}
    ]
    gaps = agent._gate(state, findings, {}, {})
    assert "GAP-METRIC" in {g["code"] for g in gaps}


def test_gate_flags_outlier_without_cross_check(agent_ctx):
    """异常指标没做交叉验证时必须被打回 —— 这是反思循环的主要触发点。"""
    agent = RiskCheckerAgent(agent_ctx)
    state = _base_state()
    state["evidence"] = [{"child_id": "c1", "source_id": "EX-TECH-2024-AR"}]
    findings = [
        {"id": "F1", "title": "现金流分析", "statement": "现金流偏弱。",
         "ratio_refs": ["cash_conversion"], "evidence_ids": ["c1"], "cross_checks": []}
    ]
    metrics = {"cash_conversion": {"value": 0.44}}
    gaps = agent._gate(state, findings, metrics, {"cash_conversion": 0.44})
    xcheck = [g for g in gaps if g["code"] == "GAP-XCHECK"]
    assert xcheck and xcheck[0]["target"] == "cash_conversion"

    # 补上交叉验证后，同一缺口应消失（循环因此可收敛）
    findings[0]["cross_checks"] = ["上期为 0.89 倍，同比下降。"]
    gaps2 = agent._gate(state, findings, metrics, {"cash_conversion": 0.44})
    assert "GAP-XCHECK" not in {g["code"] for g in gaps2}


def test_gate_flags_empty_findings(agent_ctx):
    agent = RiskCheckerAgent(agent_ctx)
    gaps = agent._gate(_base_state(), [], {}, {})
    assert [g["code"] for g in gaps] == ["GAP-NO-FINDING"]


# ---------------------------------------------------------------------------
# 3. 循环上限：超限必须升级为「需人工确认」
# ---------------------------------------------------------------------------
def test_loop_limit_escalates_to_human(agent_ctx):
    agent = RiskCheckerAgent(agent_ctx)
    state = _base_state(
        revision_round=2,  # 已经用满 2 轮
        findings=[
            {"id": "F1", "title": "仍无证据的结论", "statement": "结论未获支撑。",
             "ratio_refs": [], "evidence_ids": [], "cross_checks": []}
        ],
        evidence=[],
    )
    out = agent.run(state)
    assert out["risk_verdict"] == "escalate"
    assert out["needs_human"] is True
    assert out["revision_round"] == 2  # 超限后不再递增
    assert out["revision_requests"] == []
    reason = out["risk_report"]["escalation_reason"]
    assert "上限" in reason or "人工" in reason


def test_decide_is_deterministic():
    """裁决函数是纯函数：同样的输入永远得到同样的裁决。"""
    gaps = [{"code": "GAP-EVIDENCE"}]
    assert RiskCheckerAgent._decide(gaps, 0, 2, "medium") == "revise"
    assert RiskCheckerAgent._decide(gaps, 1, 2, "medium") == "revise"
    assert RiskCheckerAgent._decide(gaps, 2, 2, "medium") == "escalate"
    assert RiskCheckerAgent._decide([], 0, 2, "high") == "escalate"
    assert RiskCheckerAgent._decide([], 0, 2, "medium") == "pass"
    assert RiskCheckerAgent._decide([], 0, 2, "info") == "pass"


# ---------------------------------------------------------------------------
# 4. 低风险场景不应打扰人工
# ---------------------------------------------------------------------------
def test_bank_case_passes_without_human_review(pipeline):
    result = pipeline.run("示例智造银行股份有限公司 2024 年的资产质量和监管指标如何？",
                          run_id="run-bank-pass")
    state = result["state"]
    assert state["risk_verdict"] == "pass"
    assert state["needs_human"] is False
    assert state["human_decision"] is None
    assert "human_review" not in [s["agent"] for s in state["steps"]]
    assert state["risk_report"]["entity_type"] == "financial"
