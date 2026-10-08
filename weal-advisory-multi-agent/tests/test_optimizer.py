"""组合构建（多目标权衡）测试：确定性、约束感知、现金缓冲、指标口径。

被测模块：`src.optimizer` 与 `src.schemas.Portfolio` 的派生属性
- `build_portfolio`：在客户硬约束下求解权重（收益 / 波动 / 集中度 / 流动性多目标权衡）；
- `product_score`：单产品打分（对波动的惩罚随客户风险承受能力变化）；
- `compute_metrics` / `binding_constraints`：指标口径与"哪些约束顶到上限"的归因；
- `weight_diff`：版本间权重差异（供 diff 与建议书使用）；
- `Portfolio.liquid_ratio` / `holding_amounts`：流动性占比与金额折算口径。

覆盖策略
- 正常路径：全部样例客户的输出都必须通过硬约束复核（零违规）；
- 边界路径：候选池为空时退化为 100% 现金而不是报错；权重合计 + 现金 ≤ 100% 的浮点容差；
- 不变式：单一产品上限、资产类别上限、起投金额、候选池子集、确定性、纯函数性（不污染入参）；
- 对抗/口径探针：指标可重算一致、现金是否计入流动性、weight_diff 只报非零差异。

注：本模块全部为模块级测试函数，未定义测试类，故无类级 docstring。
"""

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
    """边界：候选池为空时必须退化为 100% 现金且零约束违规，而不是抛异常或胡编持仓。"""
    client = make_client()
    portfolio = build_portfolio(client, [], singleton_products(make_product()))
    assert portfolio.weights == {}
    assert portfolio.cash_weight == pytest.approx(1.0)
    assert check_portfolio(portfolio, client) == []


def test_portfolio_is_feasible_for_every_sample_client(data):
    """不变式：对全部样例客户，优化器输出必须通过硬约束复核（违规列表为空）。"""
    for client in data.all_clients():
        screening = screen_products(client, data.products, 0)
        portfolio = build_portfolio(client, screening.included, data.products)
        assert check_portfolio(portfolio, client) == []


def test_portfolio_weights_never_exceed_one(data):
    """不变式：产品权重合计 + 现金权重 ≤ 100%（容差 1e-9，避免浮点误差误报）。"""
    for client in data.sample_clients():
        screening = screen_products(client, data.products, 0)
        portfolio = build_portfolio(client, screening.included, data.products)
        assert sum(portfolio.weights.values()) + portfolio.cash_weight <= 1.0 + 1e-9


def test_cash_reserve_target_kept_when_pool_allows(data):
    """不变式：候选池充足时现金缓冲不得低于 CASH_RESERVE_TARGET。

    选 C001（约束宽松、候选充足）才能成立"池子允许"这一前提，
    否则现金可能因其它约束被动抬高，断言会失去意义。
    """
    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    assert portfolio.cash_weight >= CASH_RESERVE_TARGET - 1e-9


def test_build_portfolio_is_deterministic(data):
    """不变式：同一输入两次求解必须逐字段一致（不得依赖随机数、时间或哈希顺序）。"""
    client = data.client("C005")
    screening = screen_products(client, data.products, 0)
    first = build_portfolio(client, screening.included, data.products)
    second = build_portfolio(client, screening.included, data.products)
    assert first.model_dump() == second.model_dump()


def test_single_product_cap_respected(data):
    """不变式：任一产品权重都不得超过客户约定的单一产品集中度上限。"""
    for client in data.sample_clients():
        screening = screen_products(client, data.products, 0)
        portfolio = build_portfolio(client, screening.included, data.products)
        for weight in portfolio.weights.values():
            assert weight <= client.max_single_product_ratio + 1e-9


def test_class_cap_respected(data):
    """不变式：同一资产类别合计权重不得超过客户约定的类别上限；现金单列，不计入类别校验。"""
    for client in data.sample_clients():
        screening = screen_products(client, data.products, 0)
        portfolio = build_portfolio(client, screening.included, data.products)
        for asset_class, total in portfolio.class_weights().items():
            if asset_class == "现金":
                continue
            assert total <= client.max_single_class_ratio + 1e-9


def test_min_investment_respected(data):
    """不变式：每只持仓的参考金额（权重 × 可投金额）都不低于该产品的起投金额。"""
    for client in data.sample_clients():
        screening = screen_products(client, data.products, 0)
        portfolio = build_portfolio(client, screening.included, data.products)
        for pid, weight in portfolio.weights.items():
            assert weight * client.investable_amount >= portfolio.products[pid].min_investment - 1e-6


def test_only_candidate_products_are_held(data):
    """不变式：持仓必须是候选池的子集——优化器不得"复活"被筛选剔除的产品。"""
    client = data.client("C003")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    assert set(portfolio.weights) <= set(screening.included)


def test_metrics_are_consistent_with_weights(data):
    """口径：compute_metrics 必须可由组合重算且与内置 metrics 完全相等；holding_count 为持仓只数（float）。"""
    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    recomputed = compute_metrics(portfolio, client)
    assert recomputed == portfolio.metrics
    assert portfolio.metrics["holding_count"] == float(len(portfolio.held_ids()))


def test_product_score_prefers_lower_volatility_for_conservative_client():
    """不变式：收益相同时低波动得分更高，且该波动惩罚对保守型客户强于进取型客户。"""
    conservative = make_client(risk_capacity=1)
    aggressive = make_client(risk_capacity=5)
    # 两只产品预期收益完全相同，唯一差异是波动率（0.01 vs 0.40），隔离波动惩罚这一单一变量
    steady = make_product(volatility=0.01, expected_return=0.03)
    wild = make_product(product_id="W", volatility=0.4, expected_return=0.03)
    assert product_score(steady, conservative) > product_score(wild, conservative)
    # 进取型客户对波动的惩罚更轻，两者差距应小于保守型客户
    assert (
        product_score(steady, aggressive) - product_score(wild, aggressive)
        < product_score(steady, conservative) - product_score(wild, conservative)
    )


def test_binding_constraints_reports_capped_position(data):
    """不变式：顶到上限的持仓必须被指出，并带上约束名与产品名（供建议书"多目标权衡"章节引用）。"""
    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    binding = binding_constraints(portfolio, client)
    assert binding
    assert "单一产品上限" in "".join(binding)
    assert "示例货币基金 D" in "".join(binding)


def test_weight_diff_returns_nonzero_only():
    """口径：weight_diff 只返回发生变化的标的，无差异时返回空字典（不返回 0 值项）。"""
    assert weight_diff({"A": 0.5}, {"A": 0.5, "B": 0.2}) == {"B": 0.2}
    assert weight_diff({"A": 0.5}, {"A": 0.5}) == {}


def test_portfolio_liquid_ratio_accounts_for_cash():
    """口径：现金按 100% 计入流动性，故 liquid_ratio = 现金权重 + Σ(产品权重 × 该产品流动性比例)。"""
    product = make_product(liquidity_ratio=0.0)
    portfolio = Portfolio(
        weights={product.product_id: 0.5},
        products={product.product_id: product},
        cash_weight=0.5,
    )
    assert portfolio.liquid_ratio() == pytest.approx(0.5)


def test_holding_amounts_scale_with_investable_amount():
    """口径：holding_amounts = 权重 × 可投金额（按分取整），现金不产生金额条目。"""
    product = make_product()
    portfolio = Portfolio(
        weights={product.product_id: 0.25},
        products={product.product_id: product},
        cash_weight=0.75,
    )
    assert portfolio.holding_amounts(100000.0) == {product.product_id: 25000.0}
