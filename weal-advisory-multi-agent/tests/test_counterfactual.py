"""反事实解释测试：可复现、结构化差异、差异非空。"""

from __future__ import annotations

import pytest

from src.constraints import screen_products
from src.counterfactual import (
    DEFAULT_VARIANTS,
    analyze_counterfactual,
    counterfactual_to_rows,
    evaluate_variant,
    tighten_from_variant,
)
from src.optimizer import build_portfolio
from tests.helpers import make_client


def _baseline(data, client_id: str):
    client = data.client(client_id)
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    return client, screening, portfolio


def test_default_variants_cover_required_questions():
    ids = {spec.variant_id for spec in DEFAULT_VARIANTS}
    assert "CF-RISK-DOWN" in ids
    assert "CF-HORIZON-HALF" in ids
    assert "CF-CONC-TIGHTEN" in ids
    assert len(DEFAULT_VARIANTS) >= 3


def test_counterfactual_is_reproducible(data):
    client, screening, portfolio = _baseline(data, "C001")
    first = analyze_counterfactual(client, data.products, portfolio, ["S-COOLING"])
    second = analyze_counterfactual(client, data.products, portfolio, ["S-COOLING"])
    assert [v.model_dump() for v in first.variants] == [v.model_dump() for v in second.variants]


def test_counterfactual_has_non_empty_variant_for_every_sample_client(data):
    """硬性要求：每个方案都必须有非空反事实（差异非空）。"""
    for cid in [c.client_id for c in data.sample_clients()]:
        client, screening, portfolio = _baseline(data, cid)
        report = analyze_counterfactual(client, data.products, portfolio, [])
        assert report.variants, cid
        assert report.coverage > 0, f"{cid} 的反事实全部为空"
        assert any(not v.is_empty() for v in report.variants), cid


def test_risk_downgrade_changes_pool_for_high_risk_client(data):
    client, screening, portfolio = _baseline(data, "C005")
    spec = next(s for s in DEFAULT_VARIANTS if s.variant_id == "CF-RISK-DOWN")
    variant = evaluate_variant(spec, client, data.products, portfolio, [])
    assert variant.constraint_delta["to"] == client.risk_capacity - 1
    assert variant.products_removed, "风险等级下调一级应剔除 R5 产品"
    assert variant.explanation


def test_variant_diff_is_structured(data):
    client, screening, portfolio = _baseline(data, "C001")
    report = analyze_counterfactual(client, data.products, portfolio, ["S-CASH-RESERVE"])
    variant = report.variants[0]
    payload = variant.model_dump()
    assert set(payload) >= {
        "variant_id",
        "question",
        "constraint_delta",
        "products_added",
        "products_removed",
        "weight_changes",
        "metric_changes",
        "rule_changes",
        "explanation",
    }
    assert "resolved" in variant.rule_changes


def test_counterfactual_for_infeasible_client_marks_infeasible(data):
    client = data.client("C006")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    report = analyze_counterfactual(client, data.products, portfolio, ["S-FEASIBLE-POOL"])
    assert all(v.feasible is False for v in report.variants)
    assert all(not v.is_empty() for v in report.variants)


def test_counterfactual_rows_shape(data):
    client, screening, portfolio = _baseline(data, "C003")
    report = analyze_counterfactual(client, data.products, portfolio, [])
    rows = counterfactual_to_rows(report)
    assert len(rows) == len(report.variants)
    assert {"variant_id", "question", "status", "return_delta"} <= set(rows[0])


def test_tighten_from_variant_maps_fields():
    client = make_client(risk_capacity=4)
    spec = tighten_from_variant(client, {"field": "risk_capacity", "to": 3})
    assert spec.risk_cap == 3
    spec2 = tighten_from_variant(client, {"field": "max_single_product_ratio", "to": 0.2})
    assert spec2.max_single_product_ratio == pytest.approx(0.2)
    assert tighten_from_variant(client, {"field": "unknown", "to": 1}).is_empty()


def test_variant_constraint_delta_does_not_mutate_client(data):
    client, screening, portfolio = _baseline(data, "C005")
    before = client.model_dump()
    for spec in DEFAULT_VARIANTS:
        evaluate_variant(spec, client, data.products, portfolio, [])
    assert client.model_dump() == before


def test_counterfactual_supports_effective_baseline_and_real_rule_subject(data):
    """基线可用"生效约束"，而适当性判定仍针对客户真实档案。"""
    client = data.client("C004")
    assert client.is_elderly
    effective = client.model_copy(
        update={"risk_capacity": 3, "max_single_product_ratio": 0.20}, deep=True
    )
    screening = screen_products(effective, data.products, 0)
    portfolio = build_portfolio(effective, screening.included, data.products)

    report = analyze_counterfactual(
        effective,
        data.products,
        portfolio,
        baseline_rule_ids=[],
        rule_client=client,
    )
    risk_variant = next(v for v in report.variants if v.variant_id == "CF-RISK-DOWN")
    # 基线来自生效约束（R3 → R2），而不是档案登记等级 R4
    assert risk_variant.constraint_delta["from"] == 3
    assert risk_variant.constraint_delta["to"] == 2
    # 规则判定主体仍是真实的高龄客户，因此 S-ELDERLY 会出现在命中集合中
    assert any(v.rule_changes.get("introduced") for v in report.variants)


def test_counterfactual_rule_subject_defaults_to_baseline_client(data):
    client, screening, portfolio = _baseline(data, "C001")
    default_report = analyze_counterfactual(client, data.products, portfolio, [])
    explicit_report = analyze_counterfactual(
        client, data.products, portfolio, [], rule_client=client
    )
    assert [v.model_dump() for v in default_report.variants] == [
        v.model_dump() for v in explicit_report.variants
    ]
