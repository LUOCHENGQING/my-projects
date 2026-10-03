"""端到端流水线测试：拦截 → 打回重配 → 定稿、超限转人工、人机协同、版本链。"""

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
    pipeline, tracer = make_pipeline(**overrides)
    state = run_native(pipeline, client_id, tracer.run_id)
    return state, tracer, pipeline


def test_full_pipeline_produces_final_advice(make_pipeline):
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
    for client_id in ("C001", "C002", "C003", "C004", "C005"):
        state, _, _ = _run(make_pipeline, client_id)
        portfolio = state["portfolio"]
        assert check_portfolio(portfolio, state["client"]) == [], client_id


def test_blocked_client_is_repaired_in_second_round(make_pipeline):
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
    state, _, _ = _run(make_pipeline, "C004")
    client = state["client"]
    effective = state["effective_client"]
    assert effective.risk_capacity < client.risk_capacity
    assert effective.max_single_product_ratio <= 0.20 + 1e-9


def test_infeasible_client_is_rejected(make_pipeline):
    state, _, _ = _run(make_pipeline, "C006")
    summary = state_summary(state)
    assert summary["status"] == STATUS_REJECTED
    assert summary["directive"] == "reject"
    assert summary["block_rules"] == ["S-FEASIBLE-POOL"]
    assert state["portfolio"].weights == {}


def test_escalation_to_human_when_repair_rounds_exhausted(make_pipeline):
    state, _, _ = _run(make_pipeline, "C004", max_repair_rounds=0)
    summary = state_summary(state)
    assert summary["directive"] == "reject"
    assert summary["escalated"] is True
    assert summary["human_required"] is True
    assert state["escalation_reasons"]


def test_human_review_required_for_elderly(make_pipeline):
    state, _, _ = _run(make_pipeline, "C004")
    review = state["human_review"]
    assert review.required is True
    assert any("高龄" in reason for reason in review.reasons)
    assert review.decision == "auto_approved"
    assert state["status"] == STATUS_FINAL_ESCALATED or state["status"] == STATUS_FINAL


def test_human_review_required_for_internal_warning_line(make_pipeline):
    state, _, _ = _run(make_pipeline, "C003")
    review = state["human_review"]
    assert review.required is True
    assert any("内控预警线" in reason for reason in review.reasons)


def test_human_gate_auto_degraded_in_non_interactive_environment(make_pipeline):
    state, _, _ = _run(make_pipeline, "C003", auto=False, interactive=False)
    review = state["human_review"]
    assert review.required is True
    assert review.decision == "auto_degraded"
    assert review.degraded is True
    assert state["status"] == STATUS_FINAL_DEGRADED
    assert "降级" in review.note


def test_human_gate_interactive_rejection(make_pipeline, monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda *args, **kwargs: "n")
    state, _, _ = _run(make_pipeline, "C003", auto=False, interactive=True)
    review = state["human_review"]
    assert review.decision == "rejected"
    assert state["status"] == STATUS_REJECTED_BY_HUMAN


def test_human_gate_interactive_approval(make_pipeline, monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda *args, **kwargs: "y")
    state, _, _ = _run(make_pipeline, "C003", auto=False, interactive=True)
    review = state["human_review"]
    assert review.decision == "approved"
    assert state["status"] == STATUS_FINAL


def test_exemption_path_downgrades_block_to_warn(make_pipeline):
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
    state, _, _ = _run(make_pipeline, "C002")
    review = state["human_review"]
    assert review.required is False
    assert review.decision == "not_required"
    assert state["status"] == STATUS_FINAL


def test_trace_records_all_five_agents(make_pipeline):
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
    state, _, pipeline = _run(make_pipeline, "C004")
    chain = pipeline.store.chain("C004")
    assert [item.version for item in chain] == [1, 2]
    assert chain[1].parent_version == 1
    assert chain[1].status == STATUS_FINAL
    assert "收紧" in chain[1].change_reason or "重配" in chain[1].change_reason
    assert state["version"] == 2


def test_version_snapshot_contains_constraint_and_rule_evidence(make_pipeline):
    _, _, pipeline = _run(make_pipeline, "C004")
    snapshot = pipeline.store.chain("C004")[0]
    payload = snapshot.payload
    assert payload["client_constraints"]["risk_capacity"] == 4
    assert payload["portfolio"]["weights"]
    assert payload["rule_hits"]
    assert payload["product_snapshots"]
    assert snapshot.verify()


def test_change_log_records_repair(make_pipeline):
    state, _, _ = _run(make_pipeline, "C004")
    nodes = [item["node"] for item in state["change_log"]]
    assert "repair" in nodes
    assert nodes.count("suitability") == 2
    assert nodes[-2:] == ["human_gate", "narrative"]


def test_state_summary_shape(make_pipeline):
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
    state, _, _ = _run(make_pipeline, "C001", engine="native")
    assert state["engine"] == "native"
    assert "native" in state["advice"].narrative
