"""反事实解释测试（`src.counterfactual`）。

覆盖对象
--------
`DEFAULT_VARIANTS` 默认问题集、`evaluate_variant`（改一条约束 → 重新求解 →
结构化 diff）、`analyze_counterfactual`（整份报告）、`counterfactual_to_rows`
（表格化输出）、`tighten_from_variant`（把差异还原成收紧指令）。

覆盖策略
--------
- **正常路径 / 结构契约**：差异表的字段必须齐全（新增/剔除产品、权重变动、
  指标变动、规则变动、解释文案），表格行数与非空变体数一一对应。
- **硬性要求**：每个样例客户都必须至少有一个"非空"反事实，即"如果……会怎样"
  必须真的产生差异；对"可行域为空"的反例客户则要如实标记 `feasible=False`，
  不允许伪造一个看似可行的方案。
- **确定性（可复现）**：同一输入求解两次，产出的变体逐字段一致。
- **纯函数**：评估变体的全过程不得修改客户档案。
- **语义边界**：风险下调取 `max(1, capacity - 1)`（不会跌破 R1）；无法识别的
  字段还原成空 `TightenSpec`；"生效约束（基线）"与"规则判定主体"两个角色可以
  分离——定稿方案可能基于闸门收紧后的约束，但适当性判定始终针对客户真实档案。

数据铺垫：多数用例先经 `_baseline` 跑出一份定稿基线方案，反事实才有 diff 的对象。
"""

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
    """为指定客户跑一遍"筛选 → 求解"，返回 `(客户, 筛选结果, 基线组合)`。

    这是绝大多数反事实用例的公共前置：反事实本质上是"相对基线做 diff"，
    所以必须先有一份定稿方案作为对照。
    客户对象由 `data.client()` 返回且已是深拷贝，测试可安全地在其上做收紧。
    """
    client = data.client(client_id)
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    return client, screening, portfolio


def test_default_variants_cover_required_questions():
    """契约：默认问题集必须覆盖风险 / 期限 / 集中度三类客户最关心的问题，且至少 3 条。"""
    ids = {spec.variant_id for spec in DEFAULT_VARIANTS}
    assert "CF-RISK-DOWN" in ids
    assert "CF-HORIZON-HALF" in ids
    assert "CF-CONC-TIGHTEN" in ids
    assert len(DEFAULT_VARIANTS) >= 3


def test_counterfactual_is_reproducible(data):
    """确定性：同一客户、同一基线求解两次，产出的变体逐字段一致。"""
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
        # coverage > 0 表示至少有一条变体产生了真实差异，而不是"四条问题全都无变化"
        assert report.coverage > 0, f"{cid} 的反事实全部为空"
        assert any(not v.is_empty() for v in report.variants), cid


def test_risk_downgrade_changes_pool_for_high_risk_client(data):
    """语义：风险等级下调一级后，可行域必须剔除原先靠高风险等级才入选的产品。"""
    # C005 档案等级为 R5（最高），降一级到 R4 才会真正切掉 R5 产品，因此用它验证"确实有变化"
    client, screening, portfolio = _baseline(data, "C005")
    spec = next(s for s in DEFAULT_VARIANTS if s.variant_id == "CF-RISK-DOWN")
    variant = evaluate_variant(spec, client, data.products, portfolio, [])
    assert variant.constraint_delta["to"] == client.risk_capacity - 1
    assert variant.products_removed, "风险等级下调一级应剔除 R5 产品"
    assert variant.explanation


def test_variant_diff_is_structured(data):
    """契约：变体差异是结构化字段（新增/剔除产品、权重/指标/规则变动），不是一段散文。"""
    client, screening, portfolio = _baseline(data, "C001")
    # 传入一条基线规则 id，用于验证 rule_changes 的三分类（resolved/introduced/common）结构
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
    """边界：可行域为空的客户（C006）其变体必须如实标记不可行，同时仍给出非空差异。"""
    client = data.client("C006")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    report = analyze_counterfactual(client, data.products, portfolio, ["S-FEASIBLE-POOL"])
    assert all(v.feasible is False for v in report.variants)
    assert all(not v.is_empty() for v in report.variants)


def test_counterfactual_rows_shape(data):
    """契约：`counterfactual_to_rows` 每行对应一个变体，且含渲染所需列（状态、收益变动等）。"""
    client, screening, portfolio = _baseline(data, "C003")
    report = analyze_counterfactual(client, data.products, portfolio, [])
    rows = counterfactual_to_rows(report)
    assert len(rows) == len(report.variants)
    assert {"variant_id", "question", "status", "return_delta"} <= set(rows[0])


def test_tighten_from_variant_maps_fields():
    """映射契约：反事实 delta 能还原成对应字段的 `TightenSpec`，无法识别的字段还原为空指令。"""
    client = make_client(risk_capacity=4)
    spec = tighten_from_variant(client, {"field": "risk_capacity", "to": 3})
    assert spec.risk_cap == 3
    spec2 = tighten_from_variant(client, {"field": "max_single_product_ratio", "to": 0.2})
    assert spec2.max_single_product_ratio == pytest.approx(0.2)
    # 未知字段必须安全降级为空指令，而不是抛异常或凭空造一条约束
    assert tighten_from_variant(client, {"field": "unknown", "to": 1}).is_empty()


def test_variant_constraint_delta_does_not_mutate_client(data):
    """纯函数：评估默认问题集的全过程不得修改客户档案。"""
    client, screening, portfolio = _baseline(data, "C005")
    before = client.model_dump()
    for spec in DEFAULT_VARIANTS:
        evaluate_variant(spec, client, data.products, portfolio, [])
    assert client.model_dump() == before


def test_counterfactual_supports_effective_baseline_and_real_rule_subject(data):
    """基线可用"生效约束"，而适当性判定仍针对客户真实档案。"""
    client = data.client("C004")
    # C004 是 68 岁高龄客户：规则判定主体必须保持其真实档案（才能命中 S-ELDERLY）
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
    """缺省语义：不传 `rule_client` 时，规则判定主体等同于基线客户，结果与显式传自身一致。"""
    client, screening, portfolio = _baseline(data, "C001")
    default_report = analyze_counterfactual(client, data.products, portfolio, [])
    explicit_report = analyze_counterfactual(
        client, data.products, portfolio, [], rule_client=client
    )
    assert [v.model_dump() for v in default_report.variants] == [
        v.model_dump() for v in explicit_report.variants
    ]
