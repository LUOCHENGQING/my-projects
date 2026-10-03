"""反事实解释（Counterfactual Explanation）。

回答的是投顾场景里客户最常问的一类问题：
> “如果我的风险等级低一级 / 投资期限短一点 / 集中度上限收紧一些，你还会这样配吗？”

实现方式（与「反思循环」架构的本质区别）
----------------------------------------
反事实**不是**让模型重新写一段话，而是**真的改一条约束 → 重新求解 → 结构化 diff**：
把客户约束改动后的可行域重新跑一遍「筛选 → 组合构建 → 适当性闸门」，
然后逐产品比较权重、逐指标比较预期值、逐规则比较命中集合，产出可复核的差异表。
因此结果既可复现，也能直接回答"为什么"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .constraints import TightenSpec, screen_products
from .optimizer import build_portfolio
from .schemas import (
    ClientProfile,
    CounterfactualReport,
    CounterfactualVariant,
    Portfolio,
    Product,
)
from .suitability import RuleContext, SuitabilityGate

#: 参与 diff 的指标集合（保持固定顺序，保证输出稳定）
DIFF_METRICS: tuple[str, ...] = (
    "expected_return",
    "expected_volatility",
    "liquidity_ratio",
    "max_single_weight",
    "weighted_risk_level",
    "holding_count",
    "cash_weight",
)

#: 集中度上限收紧比例
CONCENTRATION_TIGHTEN_FACTOR = 0.8
#: 流动性下限抬升幅度（百分点）
LIQUIDITY_STEP = 0.10


@dataclass(frozen=True)
class VariantSpec:
    """一条反事实问题的定义。"""

    variant_id: str
    question: str
    transform: Callable[[ClientProfile], tuple[ClientProfile, dict[str, Any]]]


def _risk_down(client: ClientProfile) -> tuple[ClientProfile, dict[str, Any]]:
    """风险承受等级下调一级。"""
    new_level = max(1, client.risk_capacity - 1)
    updated = client.model_copy(update={"risk_capacity": new_level}, deep=True)
    return updated, {
        "field": "risk_capacity",
        "from": client.risk_capacity,
        "to": new_level,
        "description": f"风险承受等级由 R{client.risk_capacity} 下调至 R{new_level}",
    }


def _horizon_half(client: ClientProfile) -> tuple[ClientProfile, dict[str, Any]]:
    """投资期限缩短为原来的一半。"""
    new_horizon = round(max(0.25, client.investment_horizon_years / 2), 6)
    updated = client.model_copy(update={"investment_horizon_years": new_horizon}, deep=True)
    return updated, {
        "field": "investment_horizon_years",
        "from": client.investment_horizon_years,
        "to": new_horizon,
        "description": (
            f"投资期限由 {client.investment_horizon_years:g} 年缩短至 {new_horizon:g} 年"
        ),
    }


def _concentration_tighten(client: ClientProfile) -> tuple[ClientProfile, dict[str, Any]]:
    """单一产品集中度上限收紧 20%。"""
    new_cap = round(max(0.01, client.max_single_product_ratio * CONCENTRATION_TIGHTEN_FACTOR), 6)
    updated = client.model_copy(update={"max_single_product_ratio": new_cap}, deep=True)
    return updated, {
        "field": "max_single_product_ratio",
        "from": client.max_single_product_ratio,
        "to": new_cap,
        "description": (
            f"单一产品集中度上限由 {client.max_single_product_ratio:.2%} 收紧至 {new_cap:.2%}"
        ),
    }


def _liquidity_up(client: ClientProfile) -> tuple[ClientProfile, dict[str, Any]]:
    """流动性下限抬升 10 个百分点（不超过 100%）。"""
    new_floor = round(min(1.0, client.liquidity_floor_ratio + LIQUIDITY_STEP), 6)
    updated = client.model_copy(update={"liquidity_floor_ratio": new_floor}, deep=True)
    return updated, {
        "field": "liquidity_floor_ratio",
        "from": client.liquidity_floor_ratio,
        "to": new_floor,
        "description": (
            f"流动性资产占比下限由 {client.liquidity_floor_ratio:.2%} 提高至 {new_floor:.2%}"
        ),
    }


#: 默认反事实问题集（三条硬性要求 + 一条流动性敏感性）
DEFAULT_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("CF-RISK-DOWN", "若客户风险承受等级下调一级，方案会如何变化？", _risk_down),
    VariantSpec("CF-HORIZON-HALF", "若客户投资期限缩短一半，方案会如何变化？", _horizon_half),
    VariantSpec("CF-CONC-TIGHTEN", "若单一产品集中度上限收紧 20%，方案会如何变化？", _concentration_tighten),
    VariantSpec("CF-LIQUIDITY-UP", "若客户流动性需求提高 10 个百分点，方案会如何变化？", _liquidity_up),
)


def _weights_of(portfolio: Portfolio) -> dict[str, float]:
    """权重字典（含现金占位，便于整体比较）。"""
    weights = {pid: portfolio.weights[pid] for pid in portfolio.held_ids()}
    weights["CASH"] = portfolio.cash_weight
    return weights


def _diff_rules(baseline: Sequence[str], variant: Sequence[str]) -> dict[str, list[str]]:
    """规则命中集合差异。"""
    base = set(baseline)
    other = set(variant)
    return {
        "resolved": sorted(base - other),
        "introduced": sorted(other - base),
        "common": sorted(base & other),
    }


def evaluate_variant(
    spec: VariantSpec,
    client: ClientProfile,
    products: Mapping[str, Product],
    baseline: Portfolio,
    baseline_rule_ids: Sequence[str],
    gate: SuitabilityGate | None = None,
    rule_client: ClientProfile | None = None,
) -> CounterfactualVariant:
    """对一条反事实问题做「改约束 → 重求解 → 结构化 diff」。

    - `client`：反事实的**基线约束**（即定稿方案实际依据的生效约束）；
    - `rule_client`：适当性判定主体。定稿方案可能是在闸门收紧后的约束下求解的，
      但规则判定始终针对**客户真实档案**，因此这两个角色需要分开。
    """
    gate = gate or SuitabilityGate()
    rule_subject = rule_client or client
    variant_client, delta = spec.transform(client)

    screening = screen_products(variant_client, products, 0)
    portfolio = build_portfolio(variant_client, screening.included, products)

    context = RuleContext.build(rule_subject, portfolio, products, screening.included)
    decision = gate.review(context, 0)

    base_weights = _weights_of(baseline)
    variant_weights = _weights_of(portfolio)

    added = sorted(pid for pid in variant_weights if pid not in base_weights)
    removed = sorted(pid for pid in base_weights if pid not in variant_weights)
    weight_changes = {
        pid: round(variant_weights.get(pid, 0.0) - base_weights.get(pid, 0.0), 12)
        for pid in sorted(set(base_weights) | set(variant_weights))
    }
    weight_changes = {pid: value for pid, value in weight_changes.items() if abs(value) > 1e-9}

    metric_changes: dict[str, float] = {}
    for key in DIFF_METRICS:
        before = baseline.metrics.get(key, 0.0)
        after = portfolio.metrics.get(key, 0.0)
        metric_changes[key] = round(after - before, 12)
    metric_changes = {key: value for key, value in metric_changes.items() if abs(value) > 1e-12}

    variant_rule_ids = [v.rule_id for v in decision.blocks] + [v.rule_id for v in decision.warns]
    rule_changes = _diff_rules(baseline_rule_ids, variant_rule_ids)

    feasible = bool(screening.included) and decision.directive != "reject"
    status = "可行" if feasible else "不可行"
    unchanged = not added and not removed and not weight_changes and not metric_changes
    explanation = _explain(
        delta, added, removed, metric_changes, rule_changes, status, decision.directive, unchanged
    )

    return CounterfactualVariant(
        variant_id=spec.variant_id,
        question=spec.question,
        constraint_delta=dict(delta),
        feasible=feasible,
        status=f"{status}（闸门结论：{decision.directive}）",
        products_added=added,
        products_removed=removed,
        weight_changes=weight_changes,
        metric_changes=metric_changes,
        rule_changes=rule_changes,
        explanation=explanation,
    )


def _explain(
    delta: Mapping[str, Any],
    added: Sequence[str],
    removed: Sequence[str],
    metric_changes: Mapping[str, float],
    rule_changes: Mapping[str, list[str]],
    status: str,
    directive: str,
    unchanged: bool = False,
) -> str:
    """生成反事实差异的中文解释（确定性模板，不依赖模型）。"""
    parts: list[str] = [str(delta.get("description", "约束变更"))]
    if unchanged and status == "可行":
        parts.append("重新求解后方案与基线完全一致，说明当前方案对该条约束不敏感（差异为空）")
        return "；".join(parts) + "。"
    if status != "可行":
        parts.append(f"在该假设下可行域为空或方案被闸门拒绝（{directive}），说明方案对这条约束高度敏感")
    if removed:
        parts.append(f"需剔除 {len(removed)} 只产品（{'、'.join(removed)}）")
    if added:
        parts.append(f"新增 {len(added)} 只产品（{'、'.join(added)}）")
    if not removed and not added:
        parts.append("持仓产品集合不变，仅权重与指标发生变化")
    if "expected_return" in metric_changes:
        parts.append(f"组合预期年化收益变动 {metric_changes['expected_return'] * 100:+.2f} 个百分点")
    if "max_single_weight" in metric_changes:
        parts.append(f"最大单一持仓变动 {metric_changes['max_single_weight'] * 100:+.2f} 个百分点")
    if "liquidity_ratio" in metric_changes:
        parts.append(f"流动性占比变动 {metric_changes['liquidity_ratio'] * 100:+.2f} 个百分点")
    resolved = rule_changes.get("resolved") or []
    introduced = rule_changes.get("introduced") or []
    if resolved:
        parts.append(f"不再命中的规则：{'、'.join(resolved)}")
    if introduced:
        parts.append(f"新增命中的规则：{'、'.join(introduced)}")
    return "；".join(parts) + "。"


def analyze_counterfactual(
    client: ClientProfile,
    products: Mapping[str, Product],
    baseline: Portfolio,
    baseline_rule_ids: Sequence[str] = (),
    gate: SuitabilityGate | None = None,
    variants: Sequence[VariantSpec] = DEFAULT_VARIANTS,
    baseline_version: str = "baseline",
    baseline_status: str = "",
    rule_client: ClientProfile | None = None,
) -> CounterfactualReport:
    """生成完整的反事实解释报告（对固定问题集逐条求解并 diff）。

    `client` 为基线生效约束，`rule_client` 为适当性判定主体（缺省同 `client`）。
    """
    results: list[CounterfactualVariant] = []
    for spec in variants:
        results.append(
            evaluate_variant(spec, client, products, baseline, baseline_rule_ids, gate, rule_client)
        )
    return CounterfactualReport(
        baseline_version=baseline_version,
        baseline_status=baseline_status or "已生成",
        variants=results,
    )


def counterfactual_to_rows(report: CounterfactualReport) -> list[dict[str, Any]]:
    """把反事实报告转成表格行，便于 demo / 建议书渲染。"""
    rows: list[dict[str, Any]] = []
    for variant in report.variants:
        rows.append(
            {
                "variant_id": variant.variant_id,
                "question": variant.question,
                "status": variant.status,
                "products_added": variant.products_added,
                "products_removed": variant.products_removed,
                "return_delta": variant.metric_changes.get("expected_return", 0.0),
                "max_weight_delta": variant.metric_changes.get("max_single_weight", 0.0),
                "rules_resolved": variant.rule_changes.get("resolved", []),
                "rules_introduced": variant.rule_changes.get("introduced", []),
            }
        )
    return rows


def tighten_from_variant(client: ClientProfile, delta: Mapping[str, Any]) -> TightenSpec:
    """把一条反事实 delta 还原成 `TightenSpec`（用于对外解释"约束怎么改"）。"""
    field = delta.get("field")
    value = delta.get("to")
    if field == "risk_capacity":
        return TightenSpec(risk_cap=int(value))
    if field == "investment_horizon_years":
        return TightenSpec(horizon_years=float(value))
    if field == "max_single_product_ratio":
        return TightenSpec(max_single_product_ratio=float(value))
    if field == "liquidity_floor_ratio":
        return TightenSpec(liquidity_floor_ratio=float(value))
    return TightenSpec()
