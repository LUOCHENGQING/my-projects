"""产品筛选测试：可行域计算、剔除原因、确定性、与独立预言机一致。"""

from __future__ import annotations

import pytest

from src.constraints import check_product_admissibility, screen_products
from tests.helpers import make_client, make_product, singleton_products


def test_screening_splits_universe(data):
    client = data.client("C001")
    result = screen_products(client, data.products, 0)
    assert result.universe_size == len(data.products)
    assert len(result.included) + len(result.excluded) == result.universe_size
    assert set(result.included) & {item.product_id for item in result.excluded} == set()


def test_screening_records_reason_codes_and_detail(data):
    client = data.client("C002")
    result = screen_products(client, data.products, 0)
    for item in result.excluded:
        assert item.reasons
        assert item.detail
        assert all(code.startswith("C-") for code in item.reasons)


def test_screening_excludes_high_risk_products(data):
    client = data.client("C001")
    result = screen_products(client, data.products, 0)
    for pid in result.included:
        assert data.products[pid].risk_level <= client.risk_capacity


def test_screening_is_deterministic(data):
    client = data.client("C005")
    first = screen_products(client, data.products, 0).model_dump()
    second = screen_products(client, data.products, 0).model_dump()
    assert first == second


def test_screening_sorted_output(data):
    client = data.client("C003")
    result = screen_products(client, data.products, 0)
    assert result.included == sorted(result.included)
    assert [item.product_id for item in result.excluded] == sorted(
        item.product_id for item in result.excluded
    )


def test_screening_round_index_is_passed_through(data):
    client = data.client("C001")
    assert screen_products(client, data.products, 0).round_index == 0
    assert screen_products(client, data.products, 2).round_index == 2


def test_screening_included_ratio(data):
    client = data.client("C006")
    result = screen_products(client, data.products, 0)
    assert result.included_ratio == 0.0
    assert len(result.excluded) == result.universe_size


def test_screening_empty_universe():
    client = make_client()
    result = screen_products(client, {}, 0)
    assert result.universe_size == 0
    assert result.included == []
    assert result.included_ratio == 0.0


def test_screening_does_not_mutate_inputs(data):
    client = data.client("C004")
    before_client = client.model_dump()
    before_products = {pid: product.model_dump() for pid, product in data.products.items()}
    screen_products(client, data.products, 0)
    assert client.model_dump() == before_client
    assert {pid: product.model_dump() for pid, product in data.products.items()} == before_products


def test_screening_rejects_empty_experience_for_new_category():
    client = make_client(experienced_categories=["货币"])
    equity = make_product(product_id="EQ", asset_class="权益", risk_level=4, requires_experience=["权益"])
    cash = make_product(product_id="CS", asset_class="货币", risk_level=1)
    result = screen_products(client, singleton_products(equity, cash), 0)
    assert result.included == ["CS"]
    assert result.excluded[0].reasons == ["C-EXPERIENCE"]


def test_screening_multiple_reasons_accumulate():
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
    client = data.client("C002")
    result = screen_products(client, data.products, 0)
    for pid in result.included:
        assert data.products[pid].horizon_years <= client.investment_horizon_years + 1e-9


def test_screening_reason_codes_are_known(data):
    from src.constraints import CONSTRAINT_DEFS

    known = set(CONSTRAINT_DEFS)
    for client in data.all_clients():
        for item in screen_products(client, data.products, 0).excluded:
            assert set(item.reasons) <= known, item.reasons
