"""产品筛选测试：可行域计算、剔除原因、确定性、与独立预言机一致。

被测模块：`src.constraints` 的可行域求解（`screen_products` / `check_product_admissibility`）
- `screen_products`：把全量产品池切成 included / excluded，并给出原因码与中文明细；
- 原因码来自 `CONSTRAINT_DEFS`（C-RISK / C-HORIZON / C-ENTRY / C-CURRENCY / C-QUALIFIED …），
  与适当性规则号的映射关系定义在 `src.suitability.rules.CONSTRAINT_TO_RULE`。

覆盖策略
- 正常路径：划分完备且互斥、纳入产品满足风险等级与期限约束、输出排序稳定；
- 边界路径：空产品池、无任何可行产品（C006）、round_index 透传、多因累积不短路；
- 异常/健壮性：筛选必须是不修改入参的纯函数；
- 对抗探针：与 `eval.run_eval.oracle_admissible` 这一独立实现逐产品交叉校验，
  并用 `CONSTRAINT_DEFS` 反向校验原因码封闭性，防止规则层自造码。

注：本模块全部为模块级测试函数，未定义测试类，故无类级 docstring。
"""

from __future__ import annotations

import pytest

from src.constraints import check_product_admissibility, screen_products
from tests.helpers import make_client, make_product, singleton_products


def test_screening_splits_universe(data):
    """不变式：可行域划分完备且互斥——included 与 excluded 之和等于全量，且两者无交集。"""
    client = data.client("C001")
    result = screen_products(client, data.products, 0)
    assert result.universe_size == len(data.products)
    assert len(result.included) + len(result.excluded) == result.universe_size
    assert set(result.included) & {item.product_id for item in result.excluded} == set()


def test_screening_records_reason_codes_and_detail(data):
    """不变式：每个剔除项都必须同时带原因码与可读明细，且原因码统一以 C- 前缀。"""
    client = data.client("C002")
    result = screen_products(client, data.products, 0)
    for item in result.excluded:
        assert item.reasons
        assert item.detail
        assert all(code.startswith("C-") for code in item.reasons)


def test_screening_excludes_high_risk_products(data):
    """规则 S-RISK-MATCH（对应原因码 C-RISK）：纳入产品的风险等级不得高于客户风险承受等级。"""
    client = data.client("C001")
    result = screen_products(client, data.products, 0)
    for pid in result.included:
        assert data.products[pid].risk_level <= client.risk_capacity


def test_screening_is_deterministic(data):
    """不变式：同一输入两次筛选结果逐字段一致（可行域求解不得依赖集合遍历顺序）。"""
    client = data.client("C005")
    first = screen_products(client, data.products, 0).model_dump()
    second = screen_products(client, data.products, 0).model_dump()
    assert first == second


def test_screening_sorted_output(data):
    """不变式：输出顺序稳定（included 与 excluded 均按产品号升序），保证建议书与版本 diff 可复现。"""
    client = data.client("C003")
    result = screen_products(client, data.products, 0)
    assert result.included == sorted(result.included)
    assert [item.product_id for item in result.excluded] == sorted(
        item.product_id for item in result.excluded
    )


def test_screening_round_index_is_passed_through(data):
    """口径：round_index 只是透传字段（标注这是第几轮重配的可行域），不参与筛选判定。"""
    client = data.client("C001")
    assert screen_products(client, data.products, 0).round_index == 0
    assert screen_products(client, data.products, 2).round_index == 2


def test_screening_included_ratio(data):
    """边界：C006 无任何可行产品时 included_ratio 必须为 0.0，且全量产品都被剔除。"""
    client = data.client("C006")
    result = screen_products(client, data.products, 0)
    assert result.included_ratio == 0.0
    assert len(result.excluded) == result.universe_size


def test_screening_empty_universe():
    """边界：空产品池不得抛异常，universe_size 与 included_ratio 都退化为 0。"""
    client = make_client()
    result = screen_products(client, {}, 0)
    assert result.universe_size == 0
    assert result.included == []
    assert result.included_ratio == 0.0


def test_screening_does_not_mutate_inputs(data):
    """不变式：筛选是纯函数——调用后客户档案与产品表必须逐字段不变。"""
    client = data.client("C004")
    before_client = client.model_dump()
    before_products = {pid: product.model_dump() for pid, product in data.products.items()}
    screen_products(client, data.products, 0)
    assert client.model_dump() == before_client
    assert {pid: product.model_dump() for pid, product in data.products.items()} == before_products


def test_screening_rejects_empty_experience_for_new_category():
    """规则 S-EXPERIENCE（原因码 C-EXPERIENCE）：客户无"权益"经验时该品类被剔除，仅保留有经验的货币类。

    单点构造：客户只移除"权益"经验，产品只声明 requires_experience=["权益"]，
    因此剔除原因应恰好只有 C-EXPERIENCE 这一条，用来验证规则不会被其它约束污染。
    """
    client = make_client(experienced_categories=["货币"])
    equity = make_product(product_id="EQ", asset_class="权益", risk_level=4, requires_experience=["权益"])
    cash = make_product(product_id="CS", asset_class="货币", risk_level=1)
    result = screen_products(client, singleton_products(equity, cash), 0)
    assert result.included == ["CS"]
    assert result.excluded[0].reasons == ["C-EXPERIENCE"]


def test_screening_multiple_reasons_accumulate():
    """不变式：多条硬约束同时命中时原因必须全量累积（不做"命中即停"的短路）。

    刻意把风险、期限、起投、币种、准入、禁止项六项冲突一次性堆满，
    验证六个原因码 C-RISK / C-HORIZON / C-PROHIBITED-CLASS / C-ENTRY / C-CURRENCY / C-QUALIFIED 全部出现。
    """
    client = make_client(
        risk_capacity=1,
        investable_amount=1000.0,
        investment_horizon_years=1.0,
        prohibited_categories=["固收"],
        experienced_categories=[],
        qualified_investor=False,
        currency="CNY",
    )
    product = make_product(
        risk_level=5,
        horizon_years=9.0,
        min_investment=100000.0,
        currency="USD",
        qualified_investor_only=True,
    )
    reasons = check_product_admissibility(product, client)
    assert {"C-RISK", "C-HORIZON", "C-PROHIBITED-CLASS", "C-ENTRY", "C-CURRENCY", "C-QUALIFIED"} <= set(reasons)
    result = screen_products(client, singleton_products(product), 0)
    assert result.included == []
    assert set(result.excluded[0].reasons) >= {
        "C-RISK",
        "C-HORIZON",
        "C-PROHIBITED-CLASS",
        "C-ENTRY",
        "C-CURRENCY",
        "C-QUALIFIED",
    }


def test_screening_matches_oracle_for_sample_clients(data):
    """与评估脚本中的独立预言机逐产品交叉校验。"""
    from eval.run_eval import oracle_admissible

    for client in data.sample_clients():
        included = set(screen_products(client, data.products, 0).included)
        for pid, product in data.products.items():
            assert (pid in included) == oracle_admissible(product, client), f"{client.client_id}/{pid}"


def test_screening_prunes_long_horizon_products(data):
    """规则 S-HORIZON（原因码 C-HORIZON）：纳入产品期限不得超过客户投资期限（含 1e-9 浮点容差）。"""
    client = data.client("C002")
    result = screen_products(client, data.products, 0)
    for pid in result.included:
        assert data.products[pid].horizon_years <= client.investment_horizon_years + 1e-9


def test_screening_reason_codes_are_known(data):
    """不变式：任何剔除原因码都必须存在于 CONSTRAINT_DEFS（原因码封闭，禁止规则层自造）。"""
    from src.constraints import CONSTRAINT_DEFS

    known = set(CONSTRAINT_DEFS)
    for client in data.all_clients():
        for item in screen_products(client, data.products, 0).excluded:
            assert set(item.reasons) <= known, item.reasons
