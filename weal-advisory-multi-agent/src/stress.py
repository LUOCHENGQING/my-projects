"""情景压力测试（确定性公式驱动，不调用模型）。

公式
----
对每种资产类别 c 与情景 s，定义敏感性系数 `beta[c][k]`（k ∈ {rate, equity, credit_spread}）：

    类别冲击(c, s) = Σ_k beta[c][k] × shock[s][k]
    组合冲击(s)     = Σ_c w_c × 类别冲击(c, s)
    估计最大回撤(s) = max(0, -组合冲击(s)) × 放大系数

系数含义为「冲击量为 1.0（100%）时该类别的估值变动比例」，全部来自
`data/stress_scenarios.json` 的敏感性系数表，改数据即可改情景，无需改代码。
"""

from __future__ import annotations

from typing import Any, Mapping

from .schemas import ClientProfile, Portfolio, ScenarioImpact, StressReport

#: 默认的冲击维度（与数据文件中的 shocks 键一致）
SHOCK_KEYS: tuple[str, ...] = ("rate", "equity", "credit_spread")


def _shock_value(shocks: Mapping[str, float], key: str) -> float:
    """读取某维度的冲击量（缺失视为 0）。"""
    return float(shocks.get(key, 0.0))


def asset_class_impact(asset_class: str, sensitivity: Mapping[str, Mapping[str, float]], shocks: Mapping[str, float]) -> float:
    """单一资产类别在给定情景下的估值冲击。"""
    betas = sensitivity.get(asset_class)
    if not betas:
        return 0.0
    return sum(float(betas.get(key, 0.0)) * _shock_value(shocks, key) for key in SHOCK_KEYS)


def scenario_impact(
    portfolio: Portfolio,
    client: ClientProfile,
    scenario: Mapping[str, Any],
    sensitivity: Mapping[str, Mapping[str, float]],
    cash_asset_class: str,
    amplification: float,
) -> ScenarioImpact:
    """计算单个情景下的组合冲击明细。"""
    shocks = scenario.get("shocks", {})

    by_class: dict[str, float] = {}
    for asset_class, weight in sorted(portfolio.class_weights().items()):
        by_class[asset_class] = round(asset_class_impact(asset_class, sensitivity, shocks), 12)

    by_product: dict[str, float] = {}
    for pid in portfolio.held_ids():
        product = portfolio.products[pid]
        impact = asset_class_impact(product.asset_class, sensitivity, shocks)
        by_product[pid] = round(portfolio.weights[pid] * impact, 12)
    if portfolio.cash_weight > 0:
        cash_impact = asset_class_impact(cash_asset_class, sensitivity, shocks)
        by_product["CASH"] = round(portfolio.cash_weight * cash_impact, 12)
        by_class[cash_asset_class] = round(cash_impact, 12)

    portfolio_impact = round(sum(by_product.values()), 12)
    estimated_drawdown = round(max(0.0, -portfolio_impact) * amplification, 12)

    return ScenarioImpact(
        scenario_id=str(scenario.get("scenario_id", "")),
        name=str(scenario.get("name", "")),
        description=str(scenario.get("description", "")),
        portfolio_impact=portfolio_impact,
        estimated_drawdown=estimated_drawdown,
        exceeds_tolerance=estimated_drawdown > client.max_drawdown_tolerance + 1e-12,
        by_asset_class=by_class,
        by_product=by_product,
    )


def run_stress(
    portfolio: Portfolio,
    client: ClientProfile,
    config: Mapping[str, Any],
) -> StressReport:
    """对组合执行全部情景的压力测试。"""
    sensitivity = config.get("sensitivity", {})
    cash_asset_class = str(config.get("cash_asset_class", "现金"))
    amplification = float(config.get("drawdown_amplification", 1.0))

    impacts = [
        scenario_impact(portfolio, client, scenario, sensitivity, cash_asset_class, amplification)
        for scenario in config.get("scenarios", [])
    ]

    worst = min(impacts, key=lambda item: item.portfolio_impact) if impacts else None
    return StressReport(
        scenarios=impacts,
        worst_scenario_id=worst.scenario_id if worst else "",
        worst_impact=worst.portfolio_impact if worst else 0.0,
        formula=(
            "组合冲击 = Σ_类别 权重 × Σ_k 敏感性系数 × 冲击量；"
            f"估计最大回撤 = max(0, -组合冲击) × {amplification:g}"
        ),
    )


def stress_to_rows(report: StressReport) -> list[dict[str, Any]]:
    """转成表格行，便于 demo / 建议书渲染。"""
    return [
        {
            "scenario_id": item.scenario_id,
            "name": item.name,
            "portfolio_impact": item.portfolio_impact,
            "estimated_drawdown": item.estimated_drawdown,
            "exceeds_tolerance": item.exceeds_tolerance,
            "by_asset_class": item.by_asset_class,
        }
        for item in report.scenarios
    ]


def stress_coverage(report: StressReport | None) -> float:
    """压力测试覆盖率（情景数 / 要求情景数，上限 1.0）。"""
    if report is None:
        return 0.0
    return round(min(1.0, report.scenario_count / 3), 6)
