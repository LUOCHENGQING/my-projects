"""情景压力测试测试：情景数量、确定性、公式自洽、回撤判断。"""

from __future__ import annotations

import pytest

from src.constraints import screen_products
from src.optimizer import build_portfolio
from src.schemas import Portfolio
from src.stress import asset_class_impact, run_stress, stress_coverage, stress_to_rows
from tests.helpers import make_client, make_product


def _portfolio(products, weights, cash=0.0) -> Portfolio:
    held = {pid: w for pid, w in weights.items() if w > 0}
    return Portfolio(
        weights=held,
        products={pid: products[pid] for pid in held},
        cash_weight=cash,
    )


def test_three_scenarios_are_configured(data):
    assert len(data.stress_config["scenarios"]) >= 3


def test_stress_report_has_one_entry_per_scenario(data):
    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    report = run_stress(portfolio, client, data.stress_config)
    assert report.scenario_count == len(data.stress_config["scenarios"])
    assert report.worst_scenario_id in {item.scenario_id for item in report.scenarios}
    assert report.formula


def test_stress_is_deterministic(data):
    client = data.client("C005")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    first = run_stress(portfolio, client, data.stress_config)
    second = run_stress(portfolio, client, data.stress_config)
    assert first.model_dump() == second.model_dump()


def test_equity_drawdown_hits_equity_heavy_portfolio():
    client = make_client()
    product = make_product(asset_class="权益", risk_level=5, volatility=0.3)
    portfolio = _portfolio({product.product_id: product}, {product.product_id: 0.97}, cash=0.03)
    report = run_stress(portfolio, client, _config())
    scenario = next(item for item in report.scenarios if item.scenario_id == "SC-EQUITY-DD")
    assert scenario.portfolio_impact < -0.15
    assert scenario.estimated_drawdown > 0.15


def test_cash_only_portfolio_has_tiny_impact(data):
    client = data.client("C001")
    portfolio = Portfolio(weights={}, products={}, cash_weight=1.0)
    report = run_stress(portfolio, client, data.stress_config)
    for item in report.scenarios:
        assert abs(item.portfolio_impact) < 0.01


def test_by_product_sums_to_portfolio_impact(data):
    client = data.client("C003")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    report = run_stress(portfolio, client, data.stress_config)
    for item in report.scenarios:
        assert sum(item.by_product.values()) == pytest.approx(item.portfolio_impact, abs=1e-9)


def test_high_drawdown_marks_exceeds_tolerance():
    client = make_client(max_drawdown_tolerance=0.05)
    product = make_product(asset_class="权益", risk_level=5)
    portfolio = _portfolio({product.product_id: product}, {product.product_id: 0.97}, cash=0.03)
    report = run_stress(portfolio, client, _config())
    scenario = next(item for item in report.scenarios if item.scenario_id == "SC-EQUITY-DD")
    assert scenario.exceeds_tolerance is True


def test_rate_up_negative_for_bond_portfolio():
    client = make_client()
    product = make_product(asset_class="固收", risk_level=2)
    portfolio = _portfolio({product.product_id: product}, {product.product_id: 0.97}, cash=0.03)
    report = run_stress(portfolio, client, _config())
    scenario = next(item for item in report.scenarios if item.scenario_id == "SC-RATE-UP")
    assert scenario.portfolio_impact < 0


def test_asset_class_impact_uses_sensitivity_table():
    sensitivity = {"固收": {"rate": -3.5, "equity": -0.08, "credit_spread": -3.0}}
    shocks = {"rate": 0.01, "equity": 0.0, "credit_spread": 0.0}
    assert asset_class_impact("固收", sensitivity, shocks) == pytest.approx(-0.035)
    assert asset_class_impact("未知类别", sensitivity, shocks) == 0.0


def test_stress_rows_and_coverage(data):
    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    report = run_stress(portfolio, client, data.stress_config)
    rows = stress_to_rows(report)
    assert len(rows) == report.scenario_count
    assert stress_coverage(report) == 1.0
    assert stress_coverage(None) == 0.0


def test_stress_does_not_mutate_portfolio(data):
    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    before = portfolio.model_dump()
    run_stress(portfolio, client, data.stress_config)
    assert portfolio.model_dump() == before


def _config() -> dict:
    return {
        "scenarios": [
            {
                "scenario_id": "SC-RATE-UP",
                "name": "利率上行 +100bp",
                "description": "测试情景",
                "shocks": {"rate": 0.01, "equity": 0.0, "credit_spread": 0.0},
            },
            {
                "scenario_id": "SC-EQUITY-DD",
                "name": "权益市场回撤 -20%",
                "description": "测试情景",
                "shocks": {"rate": 0.0, "equity": -0.2, "credit_spread": 0.0},
            },
            {
                "scenario_id": "SC-CREDIT-WIDE",
                "name": "信用利差走阔 +150bp",
                "description": "测试情景",
                "shocks": {"rate": 0.0, "equity": 0.0, "credit_spread": 0.015},
            },
        ],
        "sensitivity": {
            "货币": {"rate": -0.2, "equity": 0.0, "credit_spread": -0.05},
            "固收": {"rate": -3.5, "equity": -0.08, "credit_spread": -3.0},
            "混合": {"rate": -2.0, "equity": 0.62, "credit_spread": -1.6},
            "权益": {"rate": -1.2, "equity": 0.95, "credit_spread": -0.7},
            "衍生品": {"rate": -4.0, "equity": 1.35, "credit_spread": -3.6},
            "现金": {"rate": 0.05, "equity": 0.0, "credit_spread": 0.0},
        },
        "cash_asset_class": "现金",
        "drawdown_amplification": 1.15,
    }
