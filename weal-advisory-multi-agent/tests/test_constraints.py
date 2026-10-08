"""硬约束求解器测试（`src.constraints`）。

覆盖对象
--------
单体产品准入 `check_product_admissibility`；组合层面检查 `check_portfolio`
及其衍生接口 `is_feasible` / `violation_count` / `feasibility_report`；
候选池筛选 `screen_products`；约束收紧指令 `TightenSpec`。

覆盖策略
--------
- **正常路径**：无瑕疵的产品 / 组合必须 0 违反，确保规则"不误杀"。
- **边界**：权重恰好等于上限（贴边）必须判通过，用于验证浮点容差确实生效。
- **异常 / 违例**：每类原因码（`C-RISK`、`C-HORIZON`、`C-PROHIBITED-CLASS`、
  `C-PROHIBITED-PRODUCT`、`C-DERIVATIVE-BAN`、`C-CURRENCY`、`C-EXPERIENCE`、
  `C-ENTRY`、`C-ENTRY-MIN`、`C-QUALIFIED`、`C-CONC-SINGLE`、`C-CONC-CLASS`、
  `C-CONC-ISSUER`、`C-LIQUIDITY`、`C-WEIGHT-SUM`、`C-WEIGHT-NEG`）各有一个
  最小触发用例：靠 `tests.helpers` 的合成数据"只改一个字段"精确命中，
  并顺手把不相关维度的上限放宽，避免一次命中多条规则。
- **纯函数性质**：不修改入参、确定性（同输入同输出且顺序一致）、幂等。
- **与优化器联动（核心不变式）**：任何被接受的组合，其硬约束违反数恒为 0 ——
  既在全部样例客户上逐个验证，也在固定种子的随机收紧指令下反复验证。

数据来源：`tests.helpers` 的合成对象负责单点触发；`data` 夹具（全量样例数据）
负责跨模块的不变式断言。
"""

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
    """产品风险等级高于客户承受等级时必须剔除（C-RISK）。"""
    client = make_client(risk_capacity=2)
    product = make_product(risk_level=4)
    assert "C-RISK" in check_product_admissibility(product, client)


def test_horizon_mismatch_excluded():
    """产品期限超过客户投资期限时必须剔除（C-HORIZON），防止期限错配。"""
    client = make_client(investment_horizon_years=2.0)
    product = make_product(horizon_years=5.0)
    assert "C-HORIZON" in check_product_admissibility(product, client)


def test_prohibited_category_excluded():
    """客户明确排除的资产类别不得出现（C-PROHIBITED-CLASS），属不可突破的禁止项。"""
    client = make_client(prohibited_categories=["衍生品"])
    product = make_product(asset_class="衍生品", risk_level=3)
    assert "C-PROHIBITED-CLASS" in check_product_admissibility(product, client)


def test_prohibited_product_excluded():
    """客户明确排除的具体产品不得出现（C-PROHIBITED-PRODUCT）。"""
    client = make_client(prohibited_product_ids=["T-PROD-01"])
    assert "C-PROHIBITED-PRODUCT" in check_product_admissibility(make_product(), client)


def test_derivative_ban_excluded():
    """客户排除衍生品类时，仅带衍生品结构的资产同样被拦（C-DERIVATIVE-BAN），即使其类别标签不是"衍生品"。"""
    client = make_client(prohibited_categories=["衍生品"])
    product = make_product(asset_class="混合", is_derivative=True)
    assert "C-DERIVATIVE-BAN" in check_product_admissibility(product, client)


def test_currency_mismatch_excluded():
    """产品币种与客户偏好币种不一致时必须剔除（C-CURRENCY）。"""
    client = make_client(currency="CNY")
    product = make_product(currency="USD")
    assert "C-CURRENCY" in check_product_admissibility(product, client)


def test_experience_missing_excluded():
    """客户不具备某品类投资经验时该品类不得推荐（C-EXPERIENCE）。"""
    client = make_client(experienced_categories=["货币"])
    product = make_product(requires_experience=["权益"])
    assert "C-EXPERIENCE" in check_product_admissibility(product, client)


def test_entry_amount_excluded():
    """产品起投金额超过客户可投金额时无法完成配置（C-ENTRY）。"""
    client = make_client(investable_amount=5000.0)
    product = make_product(min_investment=10000.0)
    assert "C-ENTRY" in check_product_admissibility(product, client)


def test_qualified_investor_required_excluded():
    """合格投资者专属产品对非合格客户不可投（C-QUALIFIED）。"""
    client = make_client(qualified_investor=False)
    product = make_product(qualified_investor_only=True)
    assert "C-QUALIFIED" in check_product_admissibility(product, client)


def test_clean_product_is_admissible():
    """正常路径：无任何瑕疵的产品必须返回空原因码列表（不得误杀）。"""
    client = make_client()
    assert check_product_admissibility(make_product(), client) == []


# ---------------------------------------------------------------------------
# 组合层面约束
# ---------------------------------------------------------------------------
def test_single_product_concentration_detected():
    """单一产品权重超过上限时命中 C-CONC-SINGLE。"""
    # 类别上限放宽到 0.95，确保只命中"单一产品"这一条规则
    client = make_client(max_single_product_ratio=0.3, max_single_class_ratio=0.95)
    product = make_product()
    portfolio = build_portfolio_obj(singleton_products(product), {product.product_id: 0.5})
    codes = {v.rule_id for v in check_portfolio(portfolio, client)}
    assert "C-CONC-SINGLE" in codes


def test_class_concentration_detected():
    """同类资产合计权重超过上限时命中 C-CONC-CLASS。"""
    # 单产品上限放宽到 0.35 使两只 0.3 的持仓各自合规，从而隔离出"类别合计超标"
    client = make_client(max_single_product_ratio=0.35, max_single_class_ratio=0.5, max_single_issuer_ratio=0.95)
    first = make_product(product_id="A", issuer="主体甲")
    second = make_product(product_id="B", issuer="主体乙")
    portfolio = build_portfolio_obj(
        singleton_products(first, second), {"A": 0.3, "B": 0.3}
    )
    codes = {v.rule_id for v in check_portfolio(portfolio, client)}
    assert "C-CONC-CLASS" in codes


def test_issuer_concentration_detected():
    """同一发行人合计权重超过上限时命中 C-CONC-ISSUER。"""
    # 两只产品故意共用同一发行主体，而类别上限保持宽松，以确保只命中发行人维度
    client = make_client(
        max_single_product_ratio=0.35, max_single_class_ratio=0.95, max_single_issuer_ratio=0.5
    )
    first = make_product(product_id="A", issuer="同一主体")
    second = make_product(product_id="B", issuer="同一主体")
    portfolio = build_portfolio_obj(singleton_products(first, second), {"A": 0.3, "B": 0.3})
    codes = {v.rule_id for v in check_portfolio(portfolio, client)}
    assert "C-CONC-ISSUER" in codes


def test_liquidity_floor_detected():
    """流动性资产占比低于客户下限时命中 C-LIQUIDITY。"""
    # 产品本身流动性比率为 0，组合里现金只有 0.1，低于客户要求的 0.5
    client = make_client(liquidity_floor_ratio=0.5, max_single_product_ratio=0.95, max_single_class_ratio=0.95)
    product = make_product(liquidity_ratio=0.0)
    portfolio = build_portfolio_obj(
        singleton_products(product), {product.product_id: 0.9}, cash_weight=0.1
    )
    codes = {v.rule_id for v in check_portfolio(portfolio, client)}
    assert "C-LIQUIDITY" in codes


def test_entry_min_amount_detected():
    """持仓金额低于产品起投金额（不满足成立条件）时命中 C-ENTRY-MIN。"""
    # 10 万 × 10% = 1 万 < 起投 5 万：金额够"买得起"，但配置额不满足成立门槛
    client = make_client(investable_amount=100000.0, max_single_product_ratio=0.95)
    product = make_product(min_investment=50000.0)
    portfolio = build_portfolio_obj(singleton_products(product), {product.product_id: 0.1})
    codes = {v.rule_id for v in check_portfolio(portfolio, client)}
    assert "C-ENTRY-MIN" in codes


def test_negative_weight_detected():
    """组合出现负权重（变相做空）时命中 C-WEIGHT-NEG。"""
    client = make_client()
    product = make_product()
    portfolio = build_portfolio_obj(singleton_products(product), {product.product_id: 0.5}, cash_weight=0.5)
    # 构造器会过滤非正权重，因此先建好合法组合，再直接改写权重字段制造违例
    portfolio.weights[product.product_id] = -0.1
    codes = {v.rule_id for v in check_portfolio(portfolio, client)}
    assert "C-WEIGHT-NEG" in codes


def test_weight_sum_over_one_detected():
    """产品权重合计超过 100% 时命中 C-WEIGHT-SUM。"""
    # 两只各 0.7（各自未超单产品上限 0.95），合计 1.4 越界；现金流置 0 以免被自动补足掩盖
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
    # A=0.3 正好等于单一产品上限，类别合计 0.6 正好等于类别上限，均为临界通过
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
    """纯函数：检查过程不修改传入的客户与组合对象。"""
    client = make_client()
    product = make_product()
    portfolio = build_portfolio_obj(singleton_products(product), {product.product_id: 0.5})
    client_before = copy.deepcopy(client.model_dump())
    portfolio_before = copy.deepcopy(portfolio.model_dump())
    check_portfolio(portfolio, client)
    assert client.model_dump() == client_before
    assert portfolio.model_dump() == portfolio_before


def test_check_portfolio_is_deterministic():
    """确定性：同一输入重复检查 5 次，违反项的内容与顺序完全一致。"""
    # 同时制造两个不同维度的违例（风险等级 + 衍生品禁止），顺带验证输出顺序稳定
    client = make_client(risk_capacity=1, prohibited_categories=["衍生品"])
    products = singleton_products(
        make_product(product_id="A", risk_level=4),
        make_product(product_id="B", asset_class="衍生品", is_derivative=True),
    )
    portfolio = build_portfolio_obj(products, {"A": 0.4, "B": 0.4})
    results = [tuple(v.rule_id for v in check_portfolio(portfolio, client)) for _ in range(5)]
    assert len(set(results)) == 1


def test_check_portfolio_is_idempotent():
    """幂等：重复调用返回逐字段相同的违反项（不累积副作用）。"""
    client = make_client(risk_capacity=2, max_single_product_ratio=0.2)
    product = make_product(risk_level=4)
    portfolio = build_portfolio_obj(singleton_products(product), {product.product_id: 0.5})
    first = check_portfolio(portfolio, client)
    second = check_portfolio(portfolio, client)
    assert [v.model_dump() for v in first] == [v.model_dump() for v in second]


def test_is_feasible_and_violation_count_agree():
    """一致性：`is_feasible` 与 `violation_count == 0` 对同一组合给出相同结论。"""
    client = make_client()
    product = make_product()
    portfolio = build_portfolio_obj(singleton_products(product), {product.product_id: 0.5})
    assert is_feasible(portfolio, client)
    assert violation_count(portfolio, client) == 0


def test_feasibility_report_shape():
    """契约：`feasibility_report` 在不可行时给出 feasible=False 且违反数 ≥ 1。"""
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
    """不变式：`TightenSpec.apply` 返回收紧后的副本，原客户对象保持原值。"""
    client = make_client(risk_capacity=4, max_single_product_ratio=0.4)
    spec = TightenSpec(risk_cap=2, max_single_product_ratio=0.2)
    tightened = spec.apply(client)
    assert client.risk_capacity == 4
    assert client.max_single_product_ratio == 0.4
    assert tightened.risk_capacity == 2
    assert tightened.max_single_product_ratio == 0.2


def test_tighten_spec_is_monotone_and_never_loosens():
    """收紧指令只会更严：反向指令不能放宽已经收紧的约束。"""
    # 传入的 risk_cap=5 / liquidity_floor=0.05 都比客户现状宽松，必须被忽略
    client = make_client(risk_capacity=2, liquidity_floor_ratio=0.3)
    spec = TightenSpec(risk_cap=5, liquidity_floor_ratio=0.05)
    tightened = spec.apply(client)
    assert tightened.risk_capacity == 2
    assert tightened.liquidity_floor_ratio == 0.3


def test_tighten_spec_merge_takes_stricter_side():
    """合并语义：数值约束取更严格一侧（等级取小、上限取低），禁止项取并集。"""
    first = TightenSpec(risk_cap=3, max_single_product_ratio=0.3, excluded_product_ids=("A",))
    second = TightenSpec(risk_cap=2, max_single_product_ratio=0.5, excluded_product_ids=("B",))
    merged = first.merge(second)
    assert merged.risk_cap == 2
    assert merged.max_single_product_ratio == 0.3
    assert merged.excluded_product_ids == ("A", "B")


def test_tighten_spec_roundtrip_dict():
    """序列化往返：`to_dict` → `from_dict` 之后与原指令等价。"""
    spec = TightenSpec(risk_cap=3, excluded_categories=("权益",), reasons=("测试",))
    assert TightenSpec.from_dict(spec.to_dict()) == spec


def test_empty_tighten_is_empty():
    """契约：默认构造的收紧指令为空，带任一约束即视为非空。"""
    assert TightenSpec().is_empty()
    assert not TightenSpec(risk_cap=3).is_empty()


# ---------------------------------------------------------------------------
# 与优化器联动：被接受的组合约束违反数必须为 0
# ---------------------------------------------------------------------------
def test_accepted_portfolios_from_optimizer_have_zero_violations(data):
    """核心不变式：对每个样例客户，求解器接受的组合违反硬约束数恒为 0。"""
    for client in data.all_clients():
        screening = screen_products(client, data.products, 0)
        portfolio = build_portfolio(client, screening.included, data.products)
        assert check_portfolio(portfolio, client) == [], f"{client.client_id} 组合违反硬约束"


def test_optimizer_output_stays_feasible_under_random_tightening(data):
    """随机收紧约束（固定种子）后，求解器输出仍然必须 0 违反。"""
    import random

    # 固定种子：让"随机收紧"本身可复现，失败时能重放同一组指令定位问题
    rng = random.Random(20261003)
    client = data.client("C005")
    for _ in range(12):
        # 六项数值约束各自在合理区间内随机，兼顾"极紧"（几乎无可行域）与"较松"两侧
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
