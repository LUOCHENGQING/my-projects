"""情景压力测试测试：情景数量、确定性、公式自洽、回撤判断。

被测模块：`src.stress`
- `run_stress`：把组合暴露到多情景宏观冲击下，逐情景给出组合冲击、逐产品归因、
  估算回撤与是否超出客户回撤容忍度；
- `asset_class_impact`：查资产类别敏感度表把冲击换算成该类资产收益影响；
- `stress_to_rows` / `stress_coverage`：报告行渲染与覆盖度口径。

覆盖策略
- 正常路径：情景条目数与配置一致、worst_scenario_id 落在情景集合内、报告必须附公式；
- 边界路径：100% 现金组合冲击必须近似为 0；
- 公式自洽：逐产品冲击之和必须等于组合冲击（归因可加），单类冲击 = Σ 敏感度 × 冲击；
- 不变式：确定性、纯函数（不修改组合对象）、回撤超容忍度必须打标；
- 对抗/口径探针：未登记资产类别按零冲击处理而非猜测。

注：本模块全部为模块级测试函数，未定义测试类，故无类级 docstring。
"""

from __future__ import annotations

import pytest

from src.constraints import screen_products
from src.optimizer import build_portfolio
from src.schemas import Portfolio
from src.stress import asset_class_impact, run_stress, stress_coverage, stress_to_rows
from tests.helpers import make_client, make_product


def _portfolio(products, weights, cash=0.0) -> Portfolio:
    """由产品表与权重字典构造 Portfolio，自动丢弃零权重项（避免出现"持有 0%"这类噪声持仓）。"""
    held = {pid: w for pid, w in weights.items() if w > 0}
    return Portfolio(
        weights=held,
        products={pid: products[pid] for pid in held},
        cash_weight=cash,
    )


def test_three_scenarios_are_configured(data):
    """硬性要求：压力情景至少配置 3 个（单情景无法覆盖利率/权益/信用三类风险）。"""
    assert len(data.stress_config["scenarios"]) >= 3


def test_stress_report_has_one_entry_per_scenario(data):
    """不变式：情景条目数等于配置数，worst_scenario_id 必须落在情景集合内，且报告附公式说明（可解释性）。"""
    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    report = run_stress(portfolio, client, data.stress_config)
    assert report.scenario_count == len(data.stress_config["scenarios"])
    assert report.worst_scenario_id in {item.scenario_id for item in report.scenarios}
    assert report.formula


def test_stress_is_deterministic(data):
    """不变式：同一输入两次压力测试必须逐字段一致（不含随机抽样）。"""
    client = data.client("C005")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    first = run_stress(portfolio, client, data.stress_config)
    second = run_stress(portfolio, client, data.stress_config)
    assert first.model_dump() == second.model_dump()


def test_equity_drawdown_hits_equity_heavy_portfolio():
    """规则 SC-EQUITY-DD：权益占 97% 的组合在权益 -20% 情景下冲击必须超过 -15%，估算回撤同步放大。

    现金留 3% 使权重合计为 1，避免触发权重合法性之类的无关噪声。
    """
    client = make_client()
    product = make_product(asset_class="权益", risk_level=5, volatility=0.3)
    portfolio = _portfolio({product.product_id: product}, {product.product_id: 0.97}, cash=0.03)
    report = run_stress(portfolio, client, _config())
    scenario = next(item for item in report.scenarios if item.scenario_id == "SC-EQUITY-DD")
    assert scenario.portfolio_impact < -0.15
    assert scenario.estimated_drawdown > 0.15


def test_cash_only_portfolio_has_tiny_impact(data):
    """边界：100% 现金组合在所有情景下的冲击都必须逼近 0（|impact| < 1%）。"""
    client = data.client("C001")
    portfolio = Portfolio(weights={}, products={}, cash_weight=1.0)
    report = run_stress(portfolio, client, data.stress_config)
    for item in report.scenarios:
        assert abs(item.portfolio_impact) < 0.01


def test_by_product_sums_to_portfolio_impact(data):
    """公式自洽：逐产品冲击之和必须等于组合冲击（容差 1e-9），保证归因可加、不凭空多出损益。"""
    client = data.client("C003")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    report = run_stress(portfolio, client, data.stress_config)
    for item in report.scenarios:
        assert sum(item.by_product.values()) == pytest.approx(item.portfolio_impact, abs=1e-9)


def test_high_drawdown_marks_exceeds_tolerance():
    """不变式：估算回撤超过客户 max_drawdown_tolerance 时，情景必须标记 exceeds_tolerance=True。"""
    client = make_client(max_drawdown_tolerance=0.05)
    product = make_product(asset_class="权益", risk_level=5)
    portfolio = _portfolio({product.product_id: product}, {product.product_id: 0.97}, cash=0.03)
    report = run_stress(portfolio, client, _config())
    scenario = next(item for item in report.scenarios if item.scenario_id == "SC-EQUITY-DD")
    assert scenario.exceeds_tolerance is True


def test_rate_up_negative_for_bond_portfolio():
    """规则 SC-RATE-UP：利率 +100bp 下固收组合因负的利率敏感度必须产生负冲击。"""
    client = make_client()
    product = make_product(asset_class="固收", risk_level=2)
    portfolio = _portfolio({product.product_id: product}, {product.product_id: 0.97}, cash=0.03)
    report = run_stress(portfolio, client, _config())
    scenario = next(item for item in report.scenarios if item.scenario_id == "SC-RATE-UP")
    assert scenario.portfolio_impact < 0


def test_asset_class_impact_uses_sensitivity_table():
    """口径：单类资产冲击 = Σ(敏感度 × 冲击)；表中未登记的资产类别按 0 处理，不做任何外推猜测。

    固收利率敏感度 -3.5，利率冲击 +1%，故结果可直接手算为 -0.035。
    """
    sensitivity = {"固收": {"rate": -3.5, "equity": -0.08, "credit_spread": -3.0}}
    shocks = {"rate": 0.01, "equity": 0.0, "credit_spread": 0.0}
    assert asset_class_impact("固收", sensitivity, shocks) == pytest.approx(-0.035)
    assert asset_class_impact("未知类别", sensitivity, shocks) == 0.0


def test_stress_rows_and_coverage(data):
    """口径：stress_to_rows 行数等于情景数；覆盖度满覆盖为 1.0，报告缺失（None）为 0.0。"""
    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    report = run_stress(portfolio, client, data.stress_config)
    rows = stress_to_rows(report)
    assert len(rows) == report.scenario_count
    assert stress_coverage(report) == 1.0
    assert stress_coverage(None) == 0.0


def test_stress_does_not_mutate_portfolio(data):
    """不变式：压力测试是纯函数——运行后组合对象必须逐字段不变。"""
    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    before = portfolio.model_dump()
    run_stress(portfolio, client, data.stress_config)
    assert portfolio.model_dump() == before


def _config() -> dict:
    """测试专用三情景配置（含 6 类资产敏感度表）。

    与 data 中的样例配置解耦，使断言数值可以手工验算；三个 scenario_id
    （SC-RATE-UP / SC-EQUITY-DD / SC-CREDIT-WIDE）正是各用例中按 id 取情景的依据。
    """
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
