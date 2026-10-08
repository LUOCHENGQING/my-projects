"""测试用合成数据工厂。

定位
----
用于**精确构造单点触发**的场景：默认参数下客户"几乎不设限"、产品"最标准"，
测试只改需要触发规则的那一个字段，避免误伤其它规则，也避免为了触发某条规则
而被迫构造一大串无关数据 —— 一旦断言失败，原因一定是单一且可定位的。

与 `data/*.json` 的分工
----------------------
样例数据用于端到端 / 评估类的整体断言；本模块的合成对象用于约束求解器这类
需要"只差一个字段"的单元断言。

本模块只构造 `src.schemas` 的 pydantic 对象，不做任何业务计算。
"""

from __future__ import annotations

from src.schemas import ClientProfile, Portfolio, Product


def make_client(**overrides) -> ClientProfile:
    """构造一个"几乎不设限"的客户，返回校验通过的 `ClientProfile`。

    默认取值的用意：
    - `risk_capacity=5`、`qualified_investor=True`，且 `experienced_categories`
      覆盖全部五个品类 —— 让风险等级、合格投资者、投资经验三类准入规则默认不触发；
    - 三项集中度上限（0.6 / 0.9 / 0.9）与 `liquidity_floor_ratio=0.0` 留足空间，
      使组合层面的集中度与流动性规则默认也不触发；
    - `prohibited_categories` / `prohibited_product_ids` 为空，即默认没有禁止项。

    调用方通过关键字覆盖**唯一**需要触发规则的那个字段
    （例如 `make_client(risk_capacity=2)`），保证用例的失败原因单一。
    """
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
    """构造一只"标准固收产品"（R2、1 年期、起投 1000 元），返回 `Product`。

    默认取值刻意保持"最无害"：风险等级低、期限短、非衍生品、无需投资经验、
    非合格投资者专属、无税收优惠、不保本、币种与客户一致，
    因此只有在某条约束被显式覆盖后才可能触发对应原因码
    （例如 `make_product(risk_level=4)` 用于触发 `C-RISK`）。
    """
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
    """由权重字典构造 `Portfolio`，`cash_weight` 缺省时自动补足剩余为现金。

    细节约定：
    - 只有 `weight > 0` 的条目会进入 `held`（进而进入 `products` 引用表），
      便于构造"只持有指定产品"的最小组合；
    - 未显式给 `cash_weight` 时按 `1 - sum(权重)` 反推，并用 `round(..., 12)`
      抵消二进制浮点误差（例如 0.3 + 0.3 会得到 0.39999999999999997，直接相减
      会让组合看起来"总权重不足/超出"）；显式传入 `cash_weight` 则可构造
      权重合计超过 100% 这类非法组合（如 `{"A": 0.7, "B": 0.7}, cash_weight=0.0`，
      用于触发 `C-WEIGHT-SUM`）。

    注意：本函数不做任何约束校验，负权重（会被 `held` 过滤）与超额权重都照单全收，
    正是为了让测试能构造出违例组合喂给 `check_portfolio`。
    """
    held = {pid: weight for pid, weight in weights.items() if weight > 0}
    cash = round(1.0 - sum(held.values()), 12) if cash_weight is None else cash_weight
    return Portfolio(
        weights=held,
        products={pid: products[pid] for pid in held},
        cash_weight=cash,
    )


def singleton_products(*products: Product) -> dict[str, Product]:
    """把若干产品打包成"产品表"（`product_id -> Product`），供 `Portfolio` 引用。

    名字取自 singleton（单元素集合）的直觉用法：最常见的是只传一只产品，
    表达"组合里只有它"；需要验证集中度类规则时也可一次传多只。
    """
    return {product.product_id: product for product in products}
