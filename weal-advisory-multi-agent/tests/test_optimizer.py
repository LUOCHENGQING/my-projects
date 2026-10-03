"""组合构建（多目标权衡）测试：确定性、约束感知、现金缓冲、指标口径。"""

from __future__ import annotations

import pytest

from src.constraints import check_portfolio, screen_products
from src.optimizer import (
    CASH_RESERVE_TARGET,
    binding_constraints,
    build_portfolio,
    compute_metrics,
    product_score,
    weight_diff,
)
from src.schemas import Portfolio
from tests.helpers import make_client, make_product, singleton_products


def test_empty_candidate_pool_yields_all_cash():
    client = make_client()
    portfolio = build_portfolio(client, [], singleton_products(make_product()))
    assert portfolio.weights == {}
    assert portfolio.cash_weight == pytest.approx(1.0)
    assert check_portfolio(portfolio, client) == []


def test_portfolio_is_feasible_for_every_sample_client(data):
    for client in data.all_clients():
        screening = screen_products(client, data.products, 0)
        portfolio = build_portfolio(client, screening.included, data.products)
        assert check_portfolio(portfolio, client) == []


def test_portfolio_weights_never_exceed_one(data):
    for client in data.sample_clients():
        screening = screen_products(client, data.products, 0)
        portfolio = build_portfolio(client, screening.included, data.products)
        assert sum(portfolio.weights.values()) + portfolio.cash_weight <= 1.0 + 1e-9


def test_cash_reserve_target_kept_when_pool_allows(data):
    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    assert portfolio.cash_weight >= CASH_RESERVE_TARGET - 1e-9


def test_build_portfolio_is_deterministic(data):
    client = data.client("C005")
    screening = screen_products(client, data.products, 0)
    first = build_portfolio(client, screening.included, data.products)
    second = build_portfolio(client, screening.included, data.products)
    assert first.model_dump() == second.model_dump()


def test_single_product_cap_respected(data):
    for client in data.sample_clients():
        screening = screen_products(client, data.products, 0)
        portfolio = build_portfolio(client, screening.included, data.products)
        for weight in portfolio.weights.values():
            assert weight <= client.max_single_product_ratio + 1e-9


def test_class_cap_respected(data):
    for client in data.sample_clients():
        screening = screen_products(client, data.products, 0)
        portfolio = build_portfolio(client, screening.included, data.products)
        for asset_class, total in portfolio.class_weights().items():
            if asset_class == "现金":
                continue
            assert total <= client.max_single_class_ratio + 1e-9


def test_min_investment_respected(data):
    for client in data.sample_clients():
        screening = screen_products(client, data.products, 0)
        portfolio = build_portfolio(client, screening.included, data.products)
        for pid, weight in portfolio.weights.items():
            assert weight * client.investable_amount >= portfolio.products[pid].min_investment - 1e-6


def test_only_candidate_products_are_held(data):
    client = data.client("C003")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    assert set(portfolio.weights) <= set(screening.included)


def test_metrics_are_consistent_with_weights(data):
    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    recomputed = compute_metrics(portfolio, client)
    assert recomputed == portfolio.metrics
    assert portfolio.metrics["holding_count"] == float(len(portfolio.held_ids()))


def test_product_score_prefers_lower_volatility_for_conservative_client():
    conservative = make_client(risk_capacity=1)
    aggressive = make_client(risk_capacity=5)
    steady = make_product(volatility=0.01, expected_return=0.03)
    wild = make_product(product_id="W", volatility=0.4, expected_return=0.03)
    assert product_score(steady, conservative) > product_score(wild, conservative)
    # 进取型客户对波动的惩罚更轻，两者差距应小于保守型客户
    assert (
        product_score(steady, aggressive) - product_score(wild, aggressive)
        < product_score(steady, conservative) - product_score(wild, conservative)
    )


def test_binding_constraints_reports_capped_position(data):
    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    binding = binding_constraints(portfolio, client)
    assert binding
    assert "单一产品上限" in "".join(binding)
    assert "示例货币基金 D" in "".join(binding)


def test_weight_diff_returns_nonzero_only():
    assert weight_diff({"A": 0.5}, {"A": 0.5, "B": 0.2}) == {"B": 0.2}
    assert weight_diff({"A": 0.5}, {"A": 0.5}) == {}


def test_portfolio_liquid_ratio_accounts_for_cash():
    product = make_product(liquidity_ratio=0.0)
    portfolio = Portfolio(
        weights={product.product_id: 0.5},
        products={product.product_id: product},
        cash_weight=0.5,
    )
    assert portfolio.liquid_ratio() == pytest.approx(0.5)


def test_holding_amounts_scale_with_investable_amount():
    product = make_product()
    portfolio = Portfolio(
        weights={product.product_id: 0.25},
        products={product.product_id: product},
        cash_weight=0.75,
    )
    assert portfolio.holding_amounts(100000.0) == {product.product_id: 25000.0}
