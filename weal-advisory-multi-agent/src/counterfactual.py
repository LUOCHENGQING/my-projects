"""反事实解释（Counterfactual Explanation）。

回答的是投顾场景里客户最常问的一类问题：
> “如果我的风险等级低一级 / 投资期限短一点 / 集中度上限收紧一些，你还会这样配吗？”

实现方式（与「反思循环」架构的本质区别）
----------------------------------------
反事实**不是**让模型重新写一段话，而是**真的改一条约束 → 重新求解 → 结构化 diff**：
把客户约束改动后的可行域重新跑一遍「筛选 → 组合构建 → 适当性闸门」，
然后逐产品比较权重、逐指标比较预期值、逐规则比较命中集合，产出可复核的差异表。
因此结果既可复现，也能直接回答"为什么"。

所属层次
--------
架构中的**解释层**：站在约束层（`screen_products`）与求解环节（`build_portfolio`）
之上，并在每个变体上调用一次适当性闸门（`SuitabilityGate`）复核；本身完全确定性，
不调用模型。与 `stress.py`（情景压力测试）、`versioning.py`（建议版本链）并列，
共同为建议书提供素材。

关键接口
--------
- `DEFAULT_VARIANTS`：默认 4 条反事实问题（风险下调一级 / 期限减半 /
  集中度上限收紧 20% / 流动性下限提高 10 个百分点）；
- `evaluate_variant(...)` -> 单条变体的结构化 diff（`CounterfactualVariant`）；
- `analyze_counterfactual(...)` -> 完整 `CounterfactualReport`；
- `counterfactual_to_rows(report)` -> 建议书 / demo 渲染用的表格行；
- `tighten_from_variant(client, delta)` -> 把变体 delta 还原成 `TightenSpec`。

输入 / 输出
-----------
- 输入：基线 `client`（生效约束）、`rule_client`（客户真实档案，用于规则判定）、
  `products` 产品表、基线 `Portfolio`、基线规则命中集合。
- 输出：`CounterfactualReport` / `CounterfactualVariant`（见 `src/schemas.py`），
  含产品增删、逐产品权重 delta、逐指标 delta、规则命中集合变化与中文解释。
- 副作用/异常：无（不写磁盘、不修改入参，`transform` 一律返回深拷贝副本）。

被谁调用
--------
- `src/agents/tools.py`（工具 `narrative.build_counterfactual`）→
  `AdvisorNarrativeAgent` 生成建议书的反事实章节；
- `src/narrative.py`、`src/demo.py`：用 `counterfactual_to_rows` 渲染表格；
- `tests/test_counterfactual.py`。
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
#: 共 7 项，取自 `optimizer.compute_metrics` 的 8 个键（不含 `expected_fee_rate`）；
#: 固定顺序让差异字典的键序在任何运行中都一致
DIFF_METRICS: tuple[str, ...] = (
    "expected_return",
    "expected_volatility",
    "liquidity_ratio",
    "max_single_weight",
    "weighted_risk_level",
    "holding_count",
    "cash_weight",
)

#: 集中度上限收紧比例：新上限 = 原上限 × 0.8，即收紧 20%（下界 0.01 见 `_concentration_tighten`）
CONCENTRATION_TIGHTEN_FACTOR = 0.8
#: 流动性下限抬升幅度：+0.10 即提高 10 个百分点（上界 1.0 见 `_liquidity_up`）
LIQUIDITY_STEP = 0.10


@dataclass(frozen=True)
class VariantSpec:
    """一条反事实问题的定义（问题文本 + 约束变换函数）。

    关键属性
    --------
    - `variant_id`：变体编号（如 `CF-RISK-DOWN`），写入报告与表格行；
    - `question`：面向客户的自然语言问题；
    - `transform`：纯函数 `ClientProfile -> (新的 ClientProfile, delta 字典)`；
      delta 至少含 `field` / `from` / `to` / `description` 四个键，其中 `description`
      会作为中文解释的开头（见 `_explain`）。
    本类为 frozen dataclass；默认问题集见 `DEFAULT_VARIANTS`，
    被 `evaluate_variant` / `analyze_counterfactual` 使用。
    """

    variant_id: str
    question: str
    transform: Callable[[ClientProfile], tuple[ClientProfile, dict[str, Any]]]


def _risk_down(client: ClientProfile) -> tuple[ClientProfile, dict[str, Any]]:
    """风险承受等级下调一级。

    参数：`client` —— 基线客户约束。
    返回：`(变体客户副本, delta)`；`max(1, ...)` 保证等级不低于 R1，
    因此 R1 客户的该变体与基线等价（此时「差异为空」本身就是有意义的结论）。
    副作用：无（`model_copy(deep=True)` 生成副本，不改入参）。
    """
    new_level = max(1, client.risk_capacity - 1)
    updated = client.model_copy(update={"risk_capacity": new_level}, deep=True)
    return updated, {
        "field": "risk_capacity",
        "from": client.risk_capacity,
        "to": new_level,
        "description": f"风险承受等级由 R{client.risk_capacity} 下调至 R{new_level}",
    }


def _horizon_half(client: ClientProfile) -> tuple[ClientProfile, dict[str, Any]]:
    """投资期限缩短为原来的一半。

    参数：`client` —— 基线客户约束。
    返回：`(变体客户副本, delta)`；下界 0.25 年（`ClientProfile` 要求期限 > 0），
    并四舍五入到 6 位小数以保证结果可复现。
    副作用：无（深拷贝副本）。
    """
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
    """单一产品集中度上限收紧 20%。

    参数：`client` —— 基线客户约束。
    返回：`(变体客户副本, delta)`；新上限 = 原上限 × `CONCENTRATION_TIGHTEN_FACTOR`，
    并保留下界 0.01（上限被压到 0 会让可行域变空，与 `optimizer.clamp_cap` 同口径）。
    副作用：无（深拷贝副本）。
    """
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
    """流动性下限抬升 10 个百分点（不超过 100%）。

    参数：`client` —— 基线客户约束。
    返回：`(变体客户副本, delta)`；新下限 = 原下限 + `LIQUIDITY_STEP`，
    并以 1.0 封顶（100% 流动性可被全现金组合满足，因此不会把可行域逼空）。
    副作用：无（深拷贝副本）。
    """
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


#: 默认反事实问题集（三条硬性要求 + 一条流动性敏感性，共 4 条）
#: 顺序固定：报告与表格行的顺序即此顺序（保证输出可复现）
DEFAULT_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("CF-RISK-DOWN", "若客户风险承受等级下调一级，方案会如何变化？", _risk_down),
    VariantSpec("CF-HORIZON-HALF", "若客户投资期限缩短一半，方案会如何变化？", _horizon_half),
    VariantSpec("CF-CONC-TIGHTEN", "若单一产品集中度上限收紧 20%，方案会如何变化？", _concentration_tighten),
    VariantSpec("CF-LIQUIDITY-UP", "若客户流动性需求提高 10 个百分点，方案会如何变化？", _liquidity_up),
)


def _weights_of(portfolio: Portfolio) -> dict[str, float]:
    """权重字典（含现金占位，便于整体比较）。

    参数：`portfolio` —— 任意组合。
    返回：`{product_id: 权重}`，另加一个特殊键 `"CASH"` 承载现金权重——
    现金不是产品，若不单独占位，现金比例的变化就无法出现在逐产品 diff 中。
    副作用/异常：无。
    """
    weights = {pid: portfolio.weights[pid] for pid in portfolio.held_ids()}
    weights["CASH"] = portfolio.cash_weight
    return weights


def _diff_rules(baseline: Sequence[str], variant: Sequence[str]) -> dict[str, list[str]]:
    """规则命中集合差异。

    参数：`baseline` / `variant` —— 两侧的规则号序列（基线 vs 变体）。
    返回：三个键 —— `resolved`（基线命中、变体不再命中）、`introduced`（变体新增命中）、
    `common`（两侧都命中）；三个列表都已排序，保证输出稳定。
    说明：规则号取自 `GateDecision.blocks + warns`，因此「命中」同时包含 block 与 warn 两级。
    副作用/异常：无。
    """
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

    参数：
    - `spec`：变体定义（问题文本 + 约束变换函数）；
    - `client`：反事实的**基线约束**（即定稿方案实际依据的生效约束）；
    - `products`：产品要素表；`baseline`：基线组合；`baseline_rule_ids`：基线规则命中集合；
    - `gate`：适当性闸门，缺省时新建 `SuitabilityGate()`；
    - `rule_client`：适当性判定主体。定稿方案可能是在闸门收紧后的约束下求解的，
      但规则判定始终针对**客户真实档案**，因此这两个角色需要分开。

    返回：`CounterfactualVariant` —— 产品增删、逐产品权重 delta（|delta| > 1e-9 才保留）、
    逐指标 delta（|delta| > 1e-12 才保留）、规则命中集合变化与中文解释。
    判定口径：`feasible = 变体候选池非空 且 闸门结论不是 reject`。
    副作用/异常：无（不修改入参；`spec.transform` 返回的是深拷贝副本）。
    """
    gate = gate or SuitabilityGate()
    rule_subject = rule_client or client
    variant_client, delta = spec.transform(client)

    # 变体一律按第 0 轮重新筛选：反事实考察的是「约束改动后的可行域」本身，
    # 与基线方案经历过几轮打回无关，这样结果才可独立复核
    screening = screen_products(variant_client, products, 0)
    portfolio = build_portfolio(variant_client, screening.included, products)

    context = RuleContext.build(rule_subject, portfolio, products, screening.included)
    decision = gate.review(context, 0)

    base_weights = _weights_of(baseline)
    variant_weights = _weights_of(portfolio)

    # 产品维度用「含 CASH 占位的完整视图」比较：现金从无到有同样算作新增项
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
    # 只保留绝对值大于 1e-12 的指标变化，过滤浮点噪声（否则「无变化」会被误判为有差异）
    metric_changes = {key: value for key, value in metric_changes.items() if abs(value) > 1e-12}

    variant_rule_ids = [v.rule_id for v in decision.blocks] + [v.rule_id for v in decision.warns]
    rule_changes = _diff_rules(baseline_rule_ids, variant_rule_ids)

    # 可行性口径：变体可行域非空，且闸门没有直接拒绝（reoptimize 仍算可行，只是会被打回）
    feasible = bool(screening.included) and decision.directive != "reject"
    status = "可行" if feasible else "不可行"
    # 「对该条约束完全不敏感」= 产品集合、权重、指标三者都没有变化
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
    """生成反事实差异的中文解释（确定性模板，不依赖模型）。

    参数：`delta` —— 约束变更描述（取 `description` 作为开头，缺失时退回「约束变更」）；
    `added` / `removed` —— 新增 / 剔除的产品；`metric_changes` —— 指标 delta；
    `rule_changes` —— 规则命中集合变化；`status` —— 「可行」/「不可行」；
    `directive` —— 闸门结论（pass / reoptimize / reject）；`unchanged` —— 差异是否为空的标记。
    返回：以「；」连接、句号收尾的中文说明，包含变动幅度（换算成百分点）与规则增删。
    设计意图：措辞完全由模板产生，因此同样输入永远得到同样文本（可复现、可单测）。
    副作用/异常：无。
    """
    parts: list[str] = [str(delta.get("description", "约束变更"))]
    if unchanged and status == "可行":
        # 提前返回：对某条约束「不敏感」本身就是一个有价值的结论，无需再罗列变化
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

    参数：`products` / `baseline` / `baseline_rule_ids` / `gate` 透传给 `evaluate_variant`；
    `variants` 缺省为 `DEFAULT_VARIANTS`（4 条）；`baseline_version` 与 `baseline_status`
    用于在报告中标注基线版本与状态（`baseline_status` 为空时记为「已生成」）。
    返回：`CounterfactualReport`，其 `variants` 顺序与入参 `variants` 顺序一致（可复现）。
    副作用/异常：无。
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
    """把反事实报告转成表格行，便于 demo / 建议书渲染。

    参数：`report` —— `analyze_counterfactual` 的产物。
    返回：每条变体一行、固定 9 个键的字典列表 —— `variant_id`、`question`、`status`、
    `products_added`、`products_removed`、`return_delta`、`max_weight_delta`、
    `rules_resolved`、`rules_introduced`；行顺序与 `report.variants` 一致。
    缺失的指标以 0.0、缺失的规则集合以空列表兜底，渲染端无需再判空。
    副作用/异常：无。
    """
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
    """把一条反事实 delta 还原成 `TightenSpec`（用于对外解释"约束怎么改"）。

    参数：`client` —— 客户约束；`delta` —— 变体 delta（读取 `field` 与 `to` 两个键）。
    返回：只含被改动那一项的 `TightenSpec`；当 `field` 不属于已知的 4 个字段
    （`risk_capacity` / `investment_horizon_years` / `max_single_product_ratio` /
    `liquidity_floor_ratio`）时返回空指令。
    注：实际实现中参数 `client` 未被读取，字段映射只依赖 `delta` 的 `field` 与 `to`。
    副作用/异常：无。
    """
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
