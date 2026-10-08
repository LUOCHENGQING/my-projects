"""RiskCheckerAgent 反思循环测试：打回、重算、超限升级人工确认。

被测行为（src.orchestrator 的条件边回环、src.agents.risk_checker 的 _gate / _decide / run）：
1. 端到端回环：核查缺口把流程打回 Analyst 重算一次，缺口消除后因高风险升级人工确认；
2. 打回意见可用性：缺口必须结构化（GAP- 前缀、problem、required_fix、severity）并留痕于 risk_report.gate_history；
3. gate 的确定性判定：无证据 -> GAP-EVIDENCE、缺量化支撑 -> GAP-METRIC、异常指标未交叉验证 -> GAP-XCHECK、无结论 -> GAP-NO-FINDING；
4. 循环上限：轮次用满后必须升级人工（needs_human=True）、不再递增轮次、清空整改请求；
5. 低风险场景：银行案例必须 pass 且完全不进入人工复核。

覆盖策略：正常（端到端两轮收敛）、边界（轮次恰好用满、缺口补上后消失、空结论集）、
异常（高风险与超限两条人工兜底路径）、对抗（打回意见必须可执行，防止「空话式整改」让循环原地打转）。
"""

from __future__ import annotations

from src.agents.risk_checker import RiskCheckerAgent
from src.state import new_state

QUESTION = "请分析示例科技股份有限公司 2024 年度的盈利能力和现金流质量，并提示主要风险。"


def _base_state(**overrides):
    """组装核查器单元测试用的基础状态：固定一家公司 / 2024 年度 / 两个分析维度，overrides 可覆盖任意字段。"""
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
    """验证不变式：打回一次后 Analyst 必须被重算且 revision_round 记为 1，最终因高风险升级人工并触发 human_review。"""
    result = pipeline.run(QUESTION, run_id="run-loop")
    state = result["state"]
    # 先取执行轨迹里的 agent 序列：比只断言终态更能证明「回环真的跑了一轮」
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
    """验证规则：打回意见必须是可执行的整改要求（GAP- 编号 + 问题 + 整改建议 + 严重级），而不是空话。

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
    """验证规则：被打回后 Analyst 必须补上交叉验证并引入上期对比口径，否则循环不会收敛。"""
    result = pipeline.run(QUESTION, run_id="run-loop-xcheck")
    findings = result["state"]["findings"]
    assert any(f.get("cross_checks") for f in findings), "重算后应出现交叉验证内容"
    # 重算轮引入了上期对比口径
    assert "receivable_growth" in result["state"]["metrics"]


# ---------------------------------------------------------------------------
# 2. 核查门（gate）的确定性判定
# ---------------------------------------------------------------------------
def test_gate_flags_finding_without_evidence(agent_ctx):
    """验证规则：evidence_ids 为空的结论必须被判 GAP-EVIDENCE。"""
    agent = RiskCheckerAgent(agent_ctx)
    state = _base_state()
    # 证据池里给一条证据，但下面的结论故意不引用它，用来单独隔离「结论没挂证据」这一条规则
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
    """验证规则：只有文字、既无 ratio_refs 又无指标值的结论必须被判 GAP-METRIC。"""
    agent = RiskCheckerAgent(agent_ctx)
    state = _base_state()
    state["evidence"] = [{"child_id": "c1", "source_id": "EX-TECH-2024-AR"}]
    # 结论确实引用了证据，但 ratio_refs 为空且 metrics 传空，专门触发「缺量化支撑」这一条
    findings = [
        {"id": "F1", "title": "纯定性结论", "statement": "公司前景良好。",
         "ratio_refs": [], "evidence_ids": ["c1"], "cross_checks": []}
    ]
    gaps = agent._gate(state, findings, {}, {})
    assert "GAP-METRIC" in {g["code"] for g in gaps}


def test_gate_flags_outlier_without_cross_check(agent_ctx):
    """验证规则：异常指标未做交叉验证时必须判 GAP-XCHECK，且缺口 target 指回该指标；补上交叉验证后缺口立即消失（这是反思循环的主要触发点）。"""
    agent = RiskCheckerAgent(agent_ctx)
    state = _base_state()
    state["evidence"] = [{"child_id": "c1", "source_id": "EX-TECH-2024-AR"}]
    findings = [
        {"id": "F1", "title": "现金流分析", "statement": "现金流偏弱。",
         "ratio_refs": ["cash_conversion"], "evidence_ids": ["c1"], "cross_checks": []}
    ]
    # 0.44 低于 0.50 的异常阈值，属于必须补交叉验证的区间
    metrics = {"cash_conversion": {"value": 0.44}}
    gaps = agent._gate(state, findings, metrics, {"cash_conversion": 0.44})
    xcheck = [g for g in gaps if g["code"] == "GAP-XCHECK"]
    assert xcheck and xcheck[0]["target"] == "cash_conversion"

    # 补上交叉验证后，同一缺口应消失（循环因此可收敛）
    findings[0]["cross_checks"] = ["上期为 0.89 倍，同比下降。"]
    gaps2 = agent._gate(state, findings, metrics, {"cash_conversion": 0.44})
    assert "GAP-XCHECK" not in {g["code"] for g in gaps2}


def test_gate_flags_empty_findings(agent_ctx):
    """验证边界：一条结论都没有时，gate 必须只报 GAP-NO-FINDING（唯一且首位的阻断缺口）。"""
    agent = RiskCheckerAgent(agent_ctx)
    # 直接传空结论集：验证「无产出」本身也是一种必须拦截的缺口
    gaps = agent._gate(_base_state(), [], {}, {})
    assert [g["code"] for g in gaps] == ["GAP-NO-FINDING"]


# ---------------------------------------------------------------------------
# 3. 循环上限：超限必须升级为「需人工确认」
# ---------------------------------------------------------------------------
def test_loop_limit_escalates_to_human(agent_ctx):
    """验证规则：轮次已用满而缺口仍在时，必须升级人工确认，且不再递增轮次、清空整改请求。"""
    agent = RiskCheckerAgent(agent_ctx)
    # 构造「仍然没有证据」的结论，保证第 3 次核查依旧有缺口，从而走到超限分支
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
    """验证性质：_decide 是纯函数真值表——同样的输入永远得到同样的裁决：有缺口未超限即 revise，超限或高风险即 escalate，无缺口且非高风险才 pass。"""
    # 固定三个入参维度逐行断言，等价于给裁决逻辑钉一张真值表
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
    """验证规则：低风险银行案例必须 pass，且完全不进入人工复核环节（不该无谓地占用人工确认）。"""
    # 换一个银行问题跑完整管线，与科技公司案例构成「高风险升级 / 低风险放行」的对照
    result = pipeline.run("示例智造银行股份有限公司 2024 年的资产质量和监管指标如何？",
                          run_id="run-bank-pass")
    state = result["state"]
    assert state["risk_verdict"] == "pass"
    assert state["needs_human"] is False
    assert state["human_decision"] is None
    assert "human_review" not in [s["agent"] for s in state["steps"]]
    assert state["risk_report"]["entity_type"] == "financial"
