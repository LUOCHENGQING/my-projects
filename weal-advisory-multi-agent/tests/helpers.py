"""测试用合成数据工厂。

用于**精确构造单点触发**的场景：默认参数下客户"几乎不设限"，
测试只改需要触发规则的那一个字段，避免误伤其它规则。
"""

from __future__ import annotations

from src.schemas import ClientProfile, Portfolio, Product


def make_client(**overrides) -> ClientProfile:
    """构造一个"几乎不设限"的客户。"""
    payload = {
        "client_id": "T001",
        "display_name": "测试客户",
        "age": 40,
        "risk_capacity": 5,
        "investment_horizon_years": 10.0,
        "liquidity_floor_ratio": 0.0,
        "max_single_product_ratio": 0.6,
        "max_single_class_ratio": 0.9,
        "max_single_issuer_ratio": 0.9,
        "investable_amount": 1000000.0,
        "prohibited_categories": [],
        "prohibited_product_ids": [],
        "experienced_categories": ["货币", "固收", "混合", "权益", "衍生品"],
        "qualified_investor": True,
        "currency": "CNY",
        "tax_advantaged_quota": 1000000.0,
        "dual_record_completed": True,
        "annual_fee_budget_ratio": 0.05,
        "max_drawdown_tolerance": 0.3,
        "return_target": 0.06,
        "internal_single_product_warning_ratio": 0.9,
    }
    payload.update(overrides)
    return ClientProfile.model_validate(payload)


def make_product(**overrides) -> Product:
    """构造一只"标准固收产品"。"""
    payload = {
        "product_id": "T-PROD-01",
        "name": "测试产品",
        "asset_class": "固收",
        "risk_level": 2,
        "horizon_years": 1.0,
        "min_investment": 1000.0,
        "expected_return": 0.04,
        "volatility": 0.03,
        "liquidity_ratio": 0.9,
        "fee_rate": 0.005,
        "issuer": "测试发行主体",
        "currency": "CNY",
        "is_derivative": False,
        "requires_experience": [],
        "qualified_investor_only": False,
        "tax_advantaged": False,
        "principal_protected": False,
    }
    payload.update(overrides)
    return Product.model_validate(payload)


def build_portfolio_obj(
    products: dict[str, Product],
    weights: dict[str, float],
    cash_weight: float | None = None,
) -> Portfolio:
    """由权重字典构造 Portfolio（自动补现金）。"""
    held = {pid: weight for pid, weight in weights.items() if weight > 0}
    cash = round(1.0 - sum(held.values()), 12) if cash_weight is None else cash_weight
    return Portfolio(
        weights=held,
        products={pid: products[pid] for pid in held},
        cash_weight=cash,
    )


def singleton_products(*products: Product) -> dict[str, Product]:
    """把若干产品打包成产品表。"""
    return {product.product_id: product for product in products}
