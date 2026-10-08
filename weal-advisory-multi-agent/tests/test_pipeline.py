"""端到端流水线测试：拦截 → 打回重配 → 定稿、超限转人工、人机协同、版本链。

被测模块：`src.pipeline`（状态机与落盘）配合 `src.engine.native.run_native`
- 五 Agent 顺序：客户画像 → 产品筛选 → 组合优化 → 适当性复核 → 建议书生成；
- 闸门语义：block 命中打回重配（下发约束收紧）→ 重配 → 复核，轮次用尽转人工，veto 直接拒绝；
- 人工闸门形态：演示自动放行 / 非交互降级 / 交互批准 / 交互拒绝 / 规则豁免；
- 留痕：trace 步骤、变更日志、版本链与快照证据。

覆盖策略
- 正常路径：C001 一次通过定稿，C002 无需人工确认；
- 边界路径：C004 首轮 block 后第二轮定稿，C006 可行域为空直接拒绝，
  `max_repair_rounds=0` 首轮即用尽转人工；
- 异常/降级路径：非交互环境不得静默通过，必须显式降级并写入备注；
- 对抗探针：任何定稿组合都必须零硬约束违规；人工确认结论必须同时出现在结构化建议与正文中。

注：本模块全部为模块级测试函数，未定义测试类，故无类级 docstring。
"""

from __future__ import annotations

import builtins

import pytest

from src.constraints import check_portfolio
from src.engine.native import run_native
from src.pipeline import (
    STATUS_BLOCKED,
    STATUS_FINAL,
    STATUS_FINAL_DEGRADED,
    STATUS_FINAL_ESCALATED,
    STATUS_REJECTED,
    STATUS_REJECTED_BY_HUMAN,
    state_summary,
)


def _run(make_pipeline, client_id: str, **overrides):
    """用工厂夹具装配流水线并跑一次 native 引擎，返回 (state, tracer, pipeline) 三元组。

    三个返回值分别服务于不同断言面：状态机（state）、可观测性（tracer）、版本留痕（pipeline.store）。
    """
    pipeline, tracer = make_pipeline(**overrides)
    state = run_native(pipeline, client_id, tracer.run_id)
    return state, tracer, pipeline


def test_full_pipeline_produces_final_advice(make_pipeline):
    """不变式：C001 正常客户必须一轮定稿——状态 final、directive=pass、版本 1，
    3 个压力情景、≥3 个反事实变体、12 项要素齐全且正文非空。"""
    state, tracer, pipeline = _run(make_pipeline, "C001")
    summary = state_summary(state)
    assert summary["status"] == STATUS_FINAL
    assert summary["directive"] == "pass"
    assert summary["version"] == 1
    assert summary["stress_scenarios"] == 3
    assert summary["counterfactual_variants"] >= 3
    assert summary["elements_ok"] == 12
    assert state["advice"].narrative


def test_accepted_portfolio_has_zero_constraint_violations(make_pipeline):
    """不变式：任何进入定稿的组合都必须零硬约束违规（端到端复核算一遍）。"""
    for client_id in ("C001", "C002", "C003", "C004", "C005"):
        state, _, _ = _run(make_pipeline, client_id)
        portfolio = state["portfolio"]
        assert check_portfolio(portfolio, state["client"]) == [], client_id


def test_blocked_client_is_repaired_in_second_round(make_pipeline):
    """规则 S-ELDERLY：C004 高龄客户首轮必被 block 打回（directive=reoptimize），
    第二轮复核通过定稿；被打回的那一轮必须落盘为 blocked 版本，留痕不可跳过。"""
    state, _, pipeline = _run(make_pipeline, "C004")
    rounds = state["gate_rounds"]
    assert len(rounds) == 2
    assert set(rounds[0]["blocks"]) == {"S-ELDERLY"}
    assert rounds[0]["directive"] == "reoptimize"
    assert rounds[1]["directive"] == "pass"
    assert state["status"] == STATUS_FINAL
    assert state["round"] == 1
    # 被打回的一轮必须落盘为 blocked 版本
    chain = [item.status for item in pipeline.store.chain("C004")]
    assert STATUS_BLOCKED in chain


def test_tighten_instruction_reduces_risk_cap(make_pipeline):
    """不变式：打回时必须下发单调收紧——有效风险等级低于客户原值，单一产品上限压到 20% 以内。"""
    state, _, _ = _run(make_pipeline, "C004")
    client = state["client"]
    effective = state["effective_client"]
    assert effective.risk_capacity < client.risk_capacity
    assert effective.max_single_product_ratio <= 0.20 + 1e-9


def test_infeasible_client_is_rejected(make_pipeline):
    """规则 S-FEASIBLE-POOL（veto）：可行域为空时直接 reject，block_rules 仅此一条，组合为空仓。"""
    state, _, _ = _run(make_pipeline, "C006")
    summary = state_summary(state)
    assert summary["status"] == STATUS_REJECTED
    assert summary["directive"] == "reject"
    assert summary["block_rules"] == ["S-FEASIBLE-POOL"]
    assert state["portfolio"].weights == {}


def test_escalation_to_human_when_repair_rounds_exhausted(make_pipeline):
    """不变式：max_repair_rounds=0 时首轮即打回次数用尽 → 转人工（escalated / human_required 均为真）并带升级原因。"""
    state, _, _ = _run(make_pipeline, "C004", max_repair_rounds=0)
    summary = state_summary(state)
    assert summary["directive"] == "reject"
    assert summary["escalated"] is True
    assert summary["human_required"] is True
    assert state["escalation_reasons"]


def test_human_review_required_for_elderly(make_pipeline):
    """不变式：高龄客户必须进入人工确认（原因含"高龄"）；
    演示模式下可自动放行，但状态只能是 final 或 final_after_escalation，二者不得含糊。"""
    state, _, _ = _run(make_pipeline, "C004")
    review = state["human_review"]
    assert review.required is True
    assert any("高龄" in reason for reason in review.reasons)
    assert review.decision == "auto_approved"
    assert state["status"] == STATUS_FINAL_ESCALATED or state["status"] == STATUS_FINAL


def test_human_review_required_for_internal_warning_line(make_pipeline):
    """规则 S-CONCENTRATION-WARNING：命中内控预警线必须转人工，原因含"内控预警线"。"""
    state, _, _ = _run(make_pipeline, "C003")
    review = state["human_review"]
    assert review.required is True
    assert any("内控预警线" in reason for reason in review.reasons)


def test_human_gate_auto_degraded_in_non_interactive_environment(make_pipeline):
    """不变式：非交互且未开启 auto 时人工闸门不得静默通过，
    必须降级为 auto_degraded（degraded=True、状态 final_degraded、备注含"降级"）。"""
    state, _, _ = _run(make_pipeline, "C003", auto=False, interactive=False)
    review = state["human_review"]
    assert review.required is True
    assert review.decision == "auto_degraded"
    assert review.degraded is True
    assert state["status"] == STATUS_FINAL_DEGRADED
    assert "降级" in review.note


def test_human_gate_interactive_rejection(make_pipeline, monkeypatch):
    """交互路径：人工输入 n 表示否决 → 状态 rejected_by_human（人工否决优先于任何自动结论）。"""
    # 注入 input，避免测试真的阻塞等待键盘输入
    monkeypatch.setattr(builtins, "input", lambda *args, **kwargs: "n")
    state, _, _ = _run(make_pipeline, "C003", auto=False, interactive=True)
    review = state["human_review"]
    assert review.decision == "rejected"
    assert state["status"] == STATUS_REJECTED_BY_HUMAN


def test_human_gate_interactive_approval(make_pipeline, monkeypatch):
    """交互路径：人工输入 y 表示批准 → 状态 final 并记录 decision=approved。"""
    monkeypatch.setattr(builtins, "input", lambda *args, **kwargs: "y")
    state, _, _ = _run(make_pipeline, "C003", auto=False, interactive=True)
    review = state["human_review"]
    assert review.decision == "approved"
    assert state["status"] == STATUS_FINAL


def test_exemption_path_downgrades_block_to_warn(make_pipeline):
    """豁免路径：S-ELDERLY 经人工豁免后降级为 warn、首轮即通过（不再打回），
    但仍必须转人工并在原因中写明"豁免"——豁免不等于免确认。"""
    state, _, _ = _run(make_pipeline, "C004", exempt_rules=("S-ELDERLY",))
    rounds = state["gate_rounds"]
    assert len(rounds) == 1
    assert rounds[0]["directive"] == "pass"
    assert rounds[0]["exempted"] == ["S-ELDERLY"]
    review = state["human_review"]
    assert review.required is True
    assert any("豁免" in reason for reason in review.reasons)


def test_human_review_is_written_into_advice_and_narrative(make_pipeline):
    """人工确认结果必须同时出现在结构化建议与建议书正文中。"""
    state, _, _ = _run(make_pipeline, "C004")
    review = state["human_review"]
    advice = state["advice"]
    assert review.required is True
    assert advice.human_review.model_dump() == review.model_dump()
    assert advice.status == state["status"]
    assert "## 十一、人工确认记录" in advice.narrative
    assert "是否需人工确认：**是**" in advice.narrative
    assert "自动放行" in advice.narrative


def test_no_human_review_needed_for_clean_client(make_pipeline):
    """边界：无触发条件的 C002 必须 required=False、decision=not_required，且不影响定稿。"""
    state, _, _ = _run(make_pipeline, "C002")
    review = state["human_review"]
    assert review.required is False
    assert review.decision == "not_required"
    assert state["status"] == STATUS_FINAL


def test_trace_records_all_five_agents(make_pipeline):
    """不变式：五个 Agent 按固定顺序各出场一次；C004 走了两轮，适当性复核出现两次；全部步骤状态为 ok。

    用 dict.fromkeys 去重保序得到"唯一出场序列"，再单独断言重复计数——
    因为两轮复核会让 SuitabilityOfficerAgent 在原始序列中出现两次。
    """
    _, tracer, _ = _run(make_pipeline, "C004")
    agents = tracer.agent_sequence()
    assert list(dict.fromkeys(agents)) == [
        "ClientProfilingAgent",
        "ProductScreeningAgent",
        "PortfolioOptimizerAgent",
        "SuitabilityOfficerAgent",
        "AdvisorNarrativeAgent",
    ]
    # C004 走了两轮，适当性复核出现两次
    assert agents.count("SuitabilityOfficerAgent") == 2
    assert tracer.step_count() >= 6
    assert set(tracer.summary()["statuses"]) == {"ok"}


def test_version_chain_links_parent_for_repaired_client(make_pipeline):
    """不变式：版本链父指针必须指向上一版（v2.parent_version == 1），
    v2 状态为 final，且变更原因写明收紧或重配。"""
    state, _, pipeline = _run(make_pipeline, "C004")
    chain = pipeline.store.chain("C004")
    assert [item.version for item in chain] == [1, 2]
    assert chain[1].parent_version == 1
    assert chain[1].status == STATUS_FINAL
    assert "收紧" in chain[1].change_reason or "重配" in chain[1].change_reason
    assert state["version"] == 2


def test_version_snapshot_contains_constraint_and_rule_evidence(make_pipeline):
    """不变式：首个快照必须留存可审计证据——有效客户约束、持仓权重、规则命中、产品快照，且哈希校验通过。"""
    _, _, pipeline = _run(make_pipeline, "C004")
    snapshot = pipeline.store.chain("C004")[0]
    payload = snapshot.payload
    assert payload["client_constraints"]["risk_capacity"] == 4
    assert payload["portfolio"]["weights"]
    assert payload["rule_hits"]
    assert payload["product_snapshots"]
    assert snapshot.verify()


def test_change_log_records_repair(make_pipeline):
    """不变式：变更日志必须记录 repair 节点；两轮复核对应两次 suitability；末尾为 human_gate → narrative。"""
    state, _, _ = _run(make_pipeline, "C004")
    nodes = [item["node"] for item in state["change_log"]]
    assert "repair" in nodes
    assert nodes.count("suitability") == 2
    assert nodes[-2:] == ["human_gate", "narrative"]


def test_state_summary_shape(make_pipeline):
    """口径：state_summary 必须覆盖评估脚本依赖的全部字段（子集断言，允许扩展但不允许缺失）。"""
    state, _, _ = _run(make_pipeline, "C001")
    summary = state_summary(state)
    expected_keys = {
        "client_id",
        "engine",
        "status",
        "rounds",
        "candidates",
        "weights",
        "cash_weight",
        "metrics",
        "directive",
        "block_rules",
        "warn_rules",
        "escalated",
        "human_required",
        "human_decision",
        "version",
        "elements_ok",
        "stress_scenarios",
        "counterfactual_variants",
        "narrative_length",
        "advice_status",
    }
    assert expected_keys <= set(summary)


def test_pipeline_engine_field_defaults_to_configured_engine(make_pipeline):
    """口径：engine 字段透传配置值，并在建议书正文中标注所用引擎（native），保证结果可归因到引擎实现。"""
    state, _, _ = _run(make_pipeline, "C001", engine="native")
    assert state["engine"] == "native"
    assert "native" in state["advice"].narrative
