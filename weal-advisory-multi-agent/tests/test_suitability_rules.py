"""适当性规则库与硬闸门测试：22 条规则逐条可触发、block 必拦、warn 不拦。"""

from __future__ import annotations

from typing import Callable

import pytest

from src.constraints import TightenSpec
from src.schemas import ClientProfile, Product
from src.suitability import (
    ALL_RULES,
    RULES_BY_ID,
    RuleContext,
    SuitabilityGate,
    build_tighten,
    evaluate_rules,
    format_rule_table,
    get_rule,
)
from src.suitability.rules import has_block_rule, rule_catalog
from tests.helpers import build_portfolio_obj, make_client, make_product, singleton_products

TriggerFactory = Callable[[], tuple[ClientProfile, dict[str, Product], dict[str, float], tuple[str, ...]]]


def _trigger_risk_match():
    product = make_product(risk_level=4)
    return make_client(risk_capacity=2), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_horizon():
    product = make_product(horizon_years=5.0)
    return make_client(investment_horizon_years=1.0), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_conc_single():
    product = make_product()
    return (
        make_client(max_single_product_ratio=0.3, max_single_class_ratio=0.95, max_single_issuer_ratio=0.95),
        singleton_products(product),
        {product.product_id: 0.5},
        (product.product_id,),
    )


def _trigger_conc_class():
    first = make_product(product_id="A", issuer="甲")
    second = make_product(product_id="B", issuer="乙")
    return (
        make_client(max_single_product_ratio=0.35, max_single_class_ratio=0.5, max_single_issuer_ratio=0.95),
        singleton_products(first, second),
        {"A": 0.3, "B": 0.3},
        ("A", "B"),
    )


def _trigger_conc_issuer():
    first = make_product(product_id="A", issuer="同一主体")
    second = make_product(product_id="B", issuer="同一主体")
    return (
        make_client(max_single_product_ratio=0.35, max_single_class_ratio=0.95, max_single_issuer_ratio=0.5),
        singleton_products(first, second),
        {"A": 0.3, "B": 0.3},
        ("A", "B"),
    )


def _trigger_liquidity():
    product = make_product(liquidity_ratio=0.0)
    return (
        make_client(liquidity_floor_ratio=0.5, max_single_product_ratio=0.95, max_single_class_ratio=0.95),
        singleton_products(product),
        {product.product_id: 0.9},
        (product.product_id,),
    )


def _trigger_entry():
    product = make_product(min_investment=5000000.0)
    return make_client(investable_amount=100000.0), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_elderly():
    product = make_product(risk_level=4)
    return make_client(age=70), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_experience():
    product = make_product(requires_experience=["权益"])
    return make_client(experienced_categories=["货币"]), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_derivative_ban():
    product = make_product(asset_class="混合", is_derivative=True)
    return (
        make_client(prohibited_categories=["衍生品"]),
        singleton_products(product),
        {product.product_id: 0.3},
        (product.product_id,),
    )


def _trigger_currency():
    product = make_product(currency="USD")
    return make_client(), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_qualified():
    product = make_product(qualified_investor_only=True)
    return make_client(qualified_investor=False), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_prohibited():
    product = make_product(asset_class="固收")
    return make_client(prohibited_categories=["固收"]), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_weight():
    first = make_product(product_id="A", issuer="甲")
    second = make_product(product_id="B", issuer="乙")
    return (
        make_client(max_single_product_ratio=0.95, max_single_class_ratio=0.95, max_single_issuer_ratio=0.95),
        singleton_products(first, second),
        {"A": 0.7, "B": 0.7},
        ("A", "B"),
    )


def _trigger_feasible_pool():
    product = make_product()
    return make_client(), singleton_products(product), {}, ()


def _trigger_dual_record():
    product = make_product(risk_level=4)
    return (
        make_client(dual_record_completed=False),
        singleton_products(product),
        {product.product_id: 0.3},
        (product.product_id,),
    )


def _trigger_fee():
    product = make_product(fee_rate=0.02)
    return (
        make_client(annual_fee_budget_ratio=0.001),
        singleton_products(product),
        {product.product_id: 0.3},
        (product.product_id,),
    )


def _trigger_tax():
    product = make_product(tax_advantaged=True)
    return (
        make_client(tax_advantaged_quota=1000.0),
        singleton_products(product),
        {product.product_id: 0.3},
        (product.product_id,),
    )


def _trigger_cooling():
    product = make_product(horizon_years=3.0, liquidity_ratio=0.1)
    return make_client(), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_concentration_warning():
    product = make_product()
    return (
        make_client(internal_single_product_warning_ratio=0.2),
        singleton_products(product),
        {product.product_id: 0.3},
        (product.product_id,),
    )


def _trigger_cash_reserve():
    product = make_product()
    return (
        make_client(max_single_product_ratio=1.0, max_single_class_ratio=1.0, max_single_issuer_ratio=1.0),
        singleton_products(product),
        {product.product_id: 1.0},
        (product.product_id,),
    )


def _trigger_diversification():
    product = make_product()
    return make_client(), singleton_products(product), {product.product_id: 0.5}, (product.product_id,)


TRIGGERS: dict[str, TriggerFactory] = {
    "S-RISK-MATCH": _trigger_risk_match,
    "S-HORIZON": _trigger_horizon,
    "S-CONC-SINGLE": _trigger_conc_single,
    "S-CONC-CLASS": _trigger_conc_class,
    "S-CONCENTRATION-ISSUER": _trigger_conc_issuer,
    "S-LIQUIDITY": _trigger_liquidity,
    "S-ENTRY": _trigger_entry,
    "S-ELDERLY": _trigger_elderly,
    "S-EXPERIENCE": _trigger_experience,
    "S-DERIVATIVE-BAN": _trigger_derivative_ban,
    "S-CURRENCY": _trigger_currency,
    "S-QUALIFIED": _trigger_qualified,
    "S-PROHIBITED": _trigger_prohibited,
    "S-WEIGHT": _trigger_weight,
    "S-FEASIBLE-POOL": _trigger_feasible_pool,
    "S-DUAL-RECORD": _trigger_dual_record,
    "S-FEE-DISCLOSE": _trigger_fee,
    "S-TAX": _trigger_tax,
    "S-COOLING": _trigger_cooling,
    "S-CONCENTRATION-WARNING": _trigger_concentration_warning,
    "S-CASH-RESERVE": _trigger_cash_reserve,
    "S-DIVERSIFICATION": _trigger_diversification,
}


def _context(factory: TriggerFactory) -> RuleContext:
    client, products, weights, candidates = factory()
    portfolio = build_portfolio_obj(products, weights)
    return RuleContext.build(client, portfolio, products, candidates)


# ---------------------------------------------------------------------------
# 规则库自身
# ---------------------------------------------------------------------------
def test_rule_catalog_has_at_least_fifteen_rules():
    assert len(ALL_RULES) >= 15


def test_rule_ids_are_unique():
    ids = [rule.id for rule in ALL_RULES]
    assert len(ids) == len(set(ids))


def test_triggers_cover_every_rule():
    """触发器必须覆盖全部规则，避免新增规则后测试悄悄失效。"""
    assert set(TRIGGERS) == {rule.id for rule in ALL_RULES}


def test_every_rule_has_required_fields():
    for rule in ALL_RULES:
        assert rule.id.startswith("S-")
        assert rule.name
        assert rule.category
        assert rule.basis
        assert rule.severity in {"block", "warn"}
        assert callable(rule.handler)


def test_required_rule_ids_present():
    """硬性要求中逐条点名的规则号必须齐全。"""
    required = {
        "S-RISK-MATCH",
        "S-HORIZON",
        "S-CONC-SINGLE",
        "S-CONC-CLASS",
        "S-LIQUIDITY",
        "S-ENTRY",
        "S-ELDERLY",
        "S-EXPERIENCE",
        "S-DERIVATIVE-BAN",
        "S-DUAL-RECORD",
        "S-FEE-DISCLOSE",
        "S-CURRENCY",
        "S-TAX",
        "S-CONCENTRATION-ISSUER",
        "S-COOLING",
    }
    assert required <= {rule.id for rule in ALL_RULES}


def test_block_rules_have_repair_strategy_except_veto():
    for rule in ALL_RULES:
        if rule.severity == "block" and not rule.veto:
            assert rule.repair, f"{rule.id} 是 block 级但未定义修复策略"


def test_only_feasible_pool_is_veto():
    veto_ids = {rule.id for rule in ALL_RULES if rule.veto}
    assert veto_ids == {"S-FEASIBLE-POOL"}


def test_rule_catalog_and_get_rule_agree():
    catalog = {item["id"] for item in rule_catalog()}
    assert catalog == set(RULES_BY_ID)
    assert get_rule("S-RISK-MATCH").severity == "block"
    assert has_block_rule("S-ELDERLY")
    assert not has_block_rule("S-COOLING")


def test_format_rule_table_lists_all_ids():
    table = format_rule_table()
    for rule in ALL_RULES:
        assert rule.id in table


def test_get_rule_unknown_raises():
    with pytest.raises(KeyError):
        get_rule("S-NOT-EXIST")


# ---------------------------------------------------------------------------
# 逐条规则可触发
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rule_id", [rule.id for rule in ALL_RULES])
def test_every_rule_is_triggerable(rule_id: str):
    evaluation = evaluate_rules(_context(TRIGGERS[rule_id]))
    assert rule_id in evaluation.hit_rule_ids


def test_rule_severity_classification_is_consistent():
    for rule_id, factory in TRIGGERS.items():
        evaluation = evaluate_rules(_context(factory))
        rule = RULES_BY_ID[rule_id]
        matched = [v for v in evaluation.hits if v.rule_id == rule_id]
        assert matched, rule_id
        assert all(v.severity == rule.severity for v in matched)


# ---------------------------------------------------------------------------
# 闸门行为
# ---------------------------------------------------------------------------
def test_gate_blocks_on_block_rule_and_requests_reoptimize():
    gate = SuitabilityGate(max_repair_rounds=2)
    decision = gate.review(_context(_trigger_risk_match), round_index=0)
    assert decision.passed is False
    assert decision.directive == "reoptimize"
    assert "S-RISK-MATCH" in [v.rule_id for v in decision.blocks]
    assert decision.tighten  # 必须下发收紧指令


def test_gate_escalates_when_repair_rounds_exhausted():
    gate = SuitabilityGate(max_repair_rounds=0)
    decision = gate.review(_context(_trigger_risk_match), round_index=0)
    assert decision.directive == "reject"
    assert decision.escalated is True
    assert decision.tighten == {}


def test_gate_passes_when_only_warn_rules_hit():
    gate = SuitabilityGate()
    decision = gate.review(_context(_trigger_cooling), round_index=0)
    assert decision.passed is True
    assert decision.directive == "pass"
    assert "S-COOLING" in [v.rule_id for v in decision.warns]


def test_gate_veto_rule_rejects_directly():
    gate = SuitabilityGate()
    decision = gate.review(_context(_trigger_feasible_pool), round_index=0)
    assert decision.directive == "reject"
    assert decision.veto_rules == ["S-FEASIBLE-POOL"]
    assert decision.escalated is False


def test_gate_exemption_downgrades_block_to_warn():
    gate = SuitabilityGate()
    decision = gate.review(_context(_trigger_risk_match), round_index=0, exempt_rules=["S-RISK-MATCH"])
    assert decision.passed is True
    assert decision.directive == "pass"
    assert decision.exempted_rules == ["S-RISK-MATCH"]
    assert any("经人工豁免" in v.detail for v in decision.warns)


def test_gate_comment_mentions_directive():
    gate = SuitabilityGate()
    decision = gate.review(_context(_trigger_conc_single), round_index=0)
    assert "打回" in decision.comment
    assert "S-CONC-SINGLE" in decision.comment


def test_evaluate_rules_is_deterministic():
    ctx = _context(_trigger_conc_class)
    first = [v.model_dump() for v in evaluate_rules(ctx).hits]
    second = [v.model_dump() for v in evaluate_rules(ctx).hits]
    assert first == second


def test_evaluate_rules_does_not_mutate_portfolio():
    client, products, weights, candidates = _trigger_conc_class()
    portfolio = build_portfolio_obj(products, weights)
    before = portfolio.model_dump()
    RuleContext.build(client, portfolio, products, candidates)
    assert portfolio.model_dump() == before


# ---------------------------------------------------------------------------
# 打回重配的约束收紧构造
# ---------------------------------------------------------------------------
def test_build_tighten_for_elderly_rule():
    ctx = _context(_trigger_elderly)
    evaluation = evaluate_rules(ctx)
    blocks = [v for v in evaluation.blocks if v.rule_id == "S-ELDERLY"]
    spec = build_tighten(blocks, ctx)
    assert spec.risk_cap == 3
    assert spec.max_single_product_ratio == pytest.approx(0.20)


def test_build_tighten_excludes_offending_products():
    ctx = _context(_trigger_experience)
    evaluation = evaluate_rules(ctx)
    blocks = [v for v in evaluation.blocks if v.rule_id == "S-EXPERIENCE"]
    spec = build_tighten(blocks, ctx)
    assert "T-PROD-01" in spec.excluded_product_ids


def test_build_tighten_for_concentration_scales_cap_down():
    ctx = _context(_trigger_conc_single)
    evaluation = evaluate_rules(ctx)
    blocks = [v for v in evaluation.blocks if v.rule_id == "S-CONC-SINGLE"]
    spec = build_tighten(blocks, ctx)
    assert spec.max_single_product_ratio == pytest.approx(0.3 * 0.8)


def test_build_tighten_is_empty_for_veto_rule():
    ctx = _context(_trigger_feasible_pool)
    evaluation = evaluate_rules(ctx)
    spec = build_tighten(evaluation.blocks, ctx)
    assert spec.is_empty()


def test_tighten_spec_from_gate_is_applicable():
    gate = SuitabilityGate()
    decision = gate.review(_context(_trigger_risk_match), round_index=0)
    client = make_client(risk_capacity=2)
    tightened = TightenSpec.from_dict(decision.tighten).apply(client)
    assert tightened.risk_capacity <= 2
