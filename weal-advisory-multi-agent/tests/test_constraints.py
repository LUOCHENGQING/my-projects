"""硬约束求解器测试：确定性、幂等、纯函数、反例必被拦下。"""

from __future__ import annotations

import copy

import pytest

from src.constraints import (
    TightenSpec,
    check_portfolio,
    check_product_admissibility,
    feasibility_report,
    is_feasible,
    screen_products,
    violation_count,
)
from src.optimizer import build_portfolio
from tests.helpers import build_portfolio_obj, make_client, make_product, singleton_products


# ---------------------------------------------------------------------------
# 单体产品准入
# ---------------------------------------------------------------------------
def test_risk_mismatch_excluded():
    client = make_client(risk_capacity=2)
    product = make_product(risk_level=4)
    assert "C-RISK" in check_product_admissibility(product, client)


def test_horizon_mismatch_excluded():
    client = make_client(investment_horizon_years=2.0)
    product = make_product(horizon_years=5.0)
    assert "C-HORIZON" in check_product_admissibility(product, client)


def test_prohibited_category_excluded():
    client = make_client(prohibited_categories=["衍生品"])
    product = make_product(asset_class="衍生品", risk_level=3)
    assert "C-PROHIBITED-CLASS" in check_product_admissibility(product, client)


def test_prohibited_product_excluded():
    client = make_client(prohibited_product_ids=["T-PROD-01"])
    assert "C-PROHIBITED-PRODUCT" in check_product_admissibility(make_product(), client)


def test_derivative_ban_excluded():
    client = make_client(prohibited_categories=["衍生品"])
    product = make_product(asset_class="混合", is_derivative=True)
    assert "C-DERIVATIVE-BAN" in check_product_admissibility(product, client)


def test_currency_mismatch_excluded():
    client = make_client(currency="CNY")
    product = make_product(currency="USD")
    assert "C-CURRENCY" in check_product_admissibility(product, client)


def test_experience_missing_excluded():
    client = make_client(experienced_categories=["货币"])
    product = make_product(requires_experience=["权益"])
    assert "C-EXPERIENCE" in check_product_admissibility(product, client)


def test_entry_amount_excluded():
    client = make_client(investable_amount=5000.0)
    product = make_product(min_investment=10000.0)
    assert "C-ENTRY" in check_product_admissibility(product, client)


def test_qualified_investor_required_excluded():
    client = make_client(qualified_investor=False)
    product = make_product(qualified_investor_only=True)
    assert "C-QUALIFIED" in check_product_admissibility(product, client)


def test_clean_product_is_admissible():
    client = make_client()
    assert check_product_admissibility(make_product(), client) == []


# ---------------------------------------------------------------------------
# 组合层面约束
# ---------------------------------------------------------------------------
def test_single_product_concentration_detected():
    client = make_client(max_single_product_ratio=0.3, max_single_class_ratio=0.95)
    product = make_product()
    portfolio = build_portfolio_obj(singleton_products(product), {product.product_id: 0.5})
    codes = {v.rule_id for v in check_portfolio(portfolio, client)}
    assert "C-CONC-SINGLE" in codes


def test_class_concentration_detected():
    client = make_client(max_single_product_ratio=0.35, max_single_class_ratio=0.5, max_single_issuer_ratio=0.95)
    first = make_product(product_id="A", issuer="主体甲")
    second = make_product(product_id="B", issuer="主体乙")
    portfolio = build_portfolio_obj(
        singleton_products(first, second), {"A": 0.3, "B": 0.3}
    )
    codes = {v.rule_id for v in check_portfolio(portfolio, client)}
    assert "C-CONC-CLASS" in codes


def test_issuer_concentration_detected():
    client = make_client(
        max_single_product_ratio=0.35, max_single_class_ratio=0.95, max_single_issuer_ratio=0.5
    )
    first = make_product(product_id="A", issuer="同一主体")
    second = make_product(product_id="B", issuer="同一主体")
    portfolio = build_portfolio_obj(singleton_products(first, second), {"A": 0.3, "B": 0.3})
    codes = {v.rule_id for v in check_portfolio(portfolio, client)}
    assert "C-CONC-ISSUER" in codes


def test_liquidity_floor_detected():
    client = make_client(liquidity_floor_ratio=0.5, max_single_product_ratio=0.95, max_single_class_ratio=0.95)
    product = make_product(liquidity_ratio=0.0)
    portfolio = build_portfolio_obj(
        singleton_products(product), {product.product_id: 0.9}, cash_weight=0.1
    )
    codes = {v.rule_id for v in check_portfolio(portfolio, client)}
    assert "C-LIQUIDITY" in codes


def test_entry_min_amount_detected():
    client = make_client(investable_amount=100000.0, max_single_product_ratio=0.95)
    product = make_product(min_investment=50000.0)
    portfolio = build_portfolio_obj(singleton_products(product), {product.product_id: 0.1})
    codes = {v.rule_id for v in check_portfolio(portfolio, client)}
    assert "C-ENTRY-MIN" in codes


def test_negative_weight_detected():
    client = make_client()
    product = make_product()
    portfolio = build_portfolio_obj(singleton_products(product), {product.product_id: 0.5}, cash_weight=0.5)
    portfolio.weights[product.product_id] = -0.1
    codes = {v.rule_id for v in check_portfolio(portfolio, client)}
    assert "C-WEIGHT-NEG" in codes


def test_weight_sum_over_one_detected():
    client = make_client(max_single_product_ratio=0.95, max_single_class_ratio=0.95)
    first = make_product(product_id="A", issuer="甲")
    second = make_product(product_id="B", issuer="乙")
    portfolio = build_portfolio_obj(
        singleton_products(first, second), {"A": 0.7, "B": 0.7}, cash_weight=0.0
    )
    codes = {v.rule_id for v in check_portfolio(portfolio, client)}
    assert "C-WEIGHT-SUM" in codes


def test_borderline_weights_exactly_at_cap_pass():
    """贴边（恰好等于上限）不算违反——浮点容差必须生效。"""
    client = make_client(
        max_single_product_ratio=0.3, max_single_class_ratio=0.6, max_single_issuer_ratio=0.9
    )
    first = make_product(product_id="A", issuer="甲")
    second = make_product(product_id="B", issuer="乙")
    portfolio = build_portfolio_obj(singleton_products(first, second), {"A": 0.3, "B": 0.3})
    assert check_portfolio(portfolio, client) == []


# ---------------------------------------------------------------------------
# 纯函数 / 确定性 / 幂等
# ---------------------------------------------------------------------------
def test_check_portfolio_does_not_mutate_inputs():
    client = make_client()
    product = make_product()
    portfolio = build_portfolio_obj(singleton_products(product), {product.product_id: 0.5})
    client_before = copy.deepcopy(client.model_dump())
    portfolio_before = copy.deepcopy(portfolio.model_dump())
    check_portfolio(portfolio, client)
    assert client.model_dump() == client_before
    assert portfolio.model_dump() == portfolio_before


def test_check_portfolio_is_deterministic():
    client = make_client(risk_capacity=1, prohibited_categories=["衍生品"])
    products = singleton_products(
        make_product(product_id="A", risk_level=4),
        make_product(product_id="B", asset_class="衍生品", is_derivative=True),
    )
    portfolio = build_portfolio_obj(products, {"A": 0.4, "B": 0.4})
    results = [tuple(v.rule_id for v in check_portfolio(portfolio, client)) for _ in range(5)]
    assert len(set(results)) == 1


def test_check_portfolio_is_idempotent():
    client = make_client(risk_capacity=2, max_single_product_ratio=0.2)
    product = make_product(risk_level=4)
    portfolio = build_portfolio_obj(singleton_products(product), {product.product_id: 0.5})
    first = check_portfolio(portfolio, client)
    second = check_portfolio(portfolio, client)
    assert [v.model_dump() for v in first] == [v.model_dump() for v in second]


def test_is_feasible_and_violation_count_agree():
    client = make_client()
    product = make_product()
    portfolio = build_portfolio_obj(singleton_products(product), {product.product_id: 0.5})
    assert is_feasible(portfolio, client)
    assert violation_count(portfolio, client) == 0


def test_feasibility_report_shape():
    client = make_client(risk_capacity=1)
    product = make_product(risk_level=5)
    portfolio = build_portfolio_obj(singleton_products(product), {product.product_id: 0.5})
    report = feasibility_report(portfolio, client)
    assert report["feasible"] is False
    assert report["violation_count"] >= 1


# ---------------------------------------------------------------------------
# TightenSpec：单调收紧、不改原对象
# ---------------------------------------------------------------------------
def test_tighten_spec_apply_does_not_mutate_original():
    client = make_client(risk_capacity=4, max_single_product_ratio=0.4)
    spec = TightenSpec(risk_cap=2, max_single_product_ratio=0.2)
    tightened = spec.apply(client)
    assert client.risk_capacity == 4
    assert client.max_single_product_ratio == 0.4
    assert tightened.risk_capacity == 2
    assert tightened.max_single_product_ratio == 0.2


def test_tighten_spec_is_monotone_and_never_loosens():
    """收紧指令只会更严：反向指令不能放宽已经收紧的约束。"""
    client = make_client(risk_capacity=2, liquidity_floor_ratio=0.3)
    spec = TightenSpec(risk_cap=5, liquidity_floor_ratio=0.05)
    tightened = spec.apply(client)
    assert tightened.risk_capacity == 2
    assert tightened.liquidity_floor_ratio == 0.3


def test_tighten_spec_merge_takes_stricter_side():
    first = TightenSpec(risk_cap=3, max_single_product_ratio=0.3, excluded_product_ids=("A",))
    second = TightenSpec(risk_cap=2, max_single_product_ratio=0.5, excluded_product_ids=("B",))
    merged = first.merge(second)
    assert merged.risk_cap == 2
    assert merged.max_single_product_ratio == 0.3
    assert merged.excluded_product_ids == ("A", "B")


def test_tighten_spec_roundtrip_dict():
    spec = TightenSpec(risk_cap=3, excluded_categories=("权益",), reasons=("测试",))
    assert TightenSpec.from_dict(spec.to_dict()) == spec


def test_empty_tighten_is_empty():
    assert TightenSpec().is_empty()
    assert not TightenSpec(risk_cap=3).is_empty()


# ---------------------------------------------------------------------------
# 与优化器联动：被接受的组合约束违反数必须为 0
# ---------------------------------------------------------------------------
def test_accepted_portfolios_from_optimizer_have_zero_violations(data):
    for client in data.all_clients():
        screening = screen_products(client, data.products, 0)
        portfolio = build_portfolio(client, screening.included, data.products)
        assert check_portfolio(portfolio, client) == [], f"{client.client_id} 组合违反硬约束"


def test_optimizer_output_stays_feasible_under_random_tightening(data):
    """随机收紧约束（固定种子）后，求解器输出仍然必须 0 违反。"""
    import random

    rng = random.Random(20261003)
    client = data.client("C005")
    for _ in range(12):
        spec = TightenSpec(
            risk_cap=rng.randint(1, 5),
            max_single_product_ratio=round(rng.uniform(0.08, 0.5), 4),
            max_single_class_ratio=round(rng.uniform(0.15, 0.8), 4),
            max_single_issuer_ratio=round(rng.uniform(0.2, 0.9), 4),
            liquidity_floor_ratio=round(rng.uniform(0.0, 0.6), 4),
            horizon_years=round(rng.uniform(0.5, 10.0), 4),
        )
        tightened = spec.apply(client)
        screening = screen_products(tightened, data.products, 0)
        portfolio = build_portfolio(tightened, screening.included, data.products)
        assert check_portfolio(portfolio, tightened) == [], f"收紧指令 {spec} 下组合违反约束"
