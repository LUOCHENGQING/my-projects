"""情景压力测试（确定性公式驱动，不调用模型）。

公式
----
对每种资产类别 c 与情景 s，定义敏感性系数 `beta[c][k]`（k ∈ {rate, equity, credit_spread}）：

    类别冲击(c, s) = Σ_k beta[c][k] × shock[s][k]
    组合冲击(s)     = Σ_c w_c × 类别冲击(c, s)
    估计最大回撤(s) = max(0, -组合冲击(s)) × 放大系数

系数含义为「冲击量为 1.0（100%）时该类别的估值变动比例」，全部来自
`data/stress_scenarios.json` 的敏感性系数表，改数据即可改情景，无需改代码。

所属层次
--------
架构中的**解释层**（与 `counterfactual.py`、`versioning.py` 并列）：
输入是已经定稿的 `Portfolio`，输出是给人看的压力测试结论。
本模块不含随机数、不调用模型，因此同一输入永远得到同一结论。

输入 / 输出
-----------
- 输入：`Portfolio`（权重 + 产品要素快照）、`ClientProfile`（只用到
  `max_drawdown_tolerance`），以及由 `data/stress_scenarios.json` 解析出的 `config`
  （`scenarios` 情景列表、`sensitivity` 敏感性系数表、`cash_asset_class`、
  `drawdown_amplification`）。数据文件中配置了 3 个情景（利率上行 +100bp、
  权益回撤 -20%、信用利差走阔 +150bp）。
- 输出：`StressReport` / `ScenarioImpact`（见 `src/schemas.py`），含逐情景的组合冲击、
  估计最大回撤、是否超过客户回撤容忍度，以及按资产类别、按产品的冲击明细。
- 副作用/异常：无（不写磁盘、不修改入参、缺失配置走缺省值而不抛异常）。

被谁调用
--------
- `src/agents/tools.py`（工具 `narrative.run_stress`）→ `AdvisorNarrativeAgent`
  生成建议书的压力测试章节；
- `src/narrative.py`、`src/demo.py`：用 `stress_to_rows` 渲染表格；
- `eval/run_eval.py`：用 `stress_coverage` 计算「压力测试覆盖率」指标；
- `tests/test_stress.py`、`tests/test_narrative.py`。
"""

from __future__ import annotations

from typing import Any, Mapping

from .schemas import ClientProfile, Portfolio, ScenarioImpact, StressReport

#: 冲击维度（与数据文件中的 `shocks` 键一致，共 3 个）：利率 / 权益 / 信用利差。
#: 类别冲击按此顺序累加，保证浮点求和顺序固定（确定性）
SHOCK_KEYS: tuple[str, ...] = ("rate", "equity", "credit_spread")


def _shock_value(shocks: Mapping[str, float], key: str) -> float:
    """读取某维度的冲击量（缺失视为 0）。

    参数：`shocks` —— 某情景的冲击字典（如 `{"rate": 0.01, "equity": 0.0, ...}`）；
    `key` —— 维度名（`SHOCK_KEYS` 之一）。
    返回：该维度冲击量的 float 值；键缺失时返回 0.0，因此情景可以只写有冲击的维度。
    副作用/异常：无。
    """
    return float(shocks.get(key, 0.0))


def asset_class_impact(asset_class: str, sensitivity: Mapping[str, Mapping[str, float]], shocks: Mapping[str, float]) -> float:
    """单一资产类别在给定情景下的估值冲击。

    参数：`asset_class` —— 类别名（如「固收」「权益」「现金」）；
    `sensitivity` —— 敏感性系数表 `{类别: {维度: 系数}}`；`shocks` —— 情景冲击量。
    返回：`Σ_k 系数[类别][k] × 冲击量[k]`（k 遍历 `SHOCK_KEYS`）；类别不在系数表中
    （或系数为空）时返回 0.0，即保守地视为无冲击，而不是抛异常。
    副作用/异常：无。系数含义为「冲击量 = 1.0（100%）时该类别的估值变动比例」。
    """
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
    """计算单个情景下的组合冲击明细。

    参数：`portfolio` —— 待测试组合；`client` —— 提供最大回撤容忍度；
    `scenario` —— 单个情景定义（`shocks` 为冲击字典，另含 id / name / description）；
    `sensitivity` —— 敏感性系数表；`cash_asset_class` —— 现金所属类别名（用于取系数）；
    `amplification` —— 回撤放大系数（估计最大回撤 = 负向组合冲击 × 该系数）。
    返回：`ScenarioImpact`（组合冲击、估计最大回撤、是否超限、按类别与按产品的明细）。
    副作用/异常：无。所有数值四舍五入到 12 位小数，保证结果可复现。
    """
    shocks = scenario.get("shocks", {})

    by_class: dict[str, float] = {}
    # 遍历 class_weights 只为取到全部类别名（含「现金」）；类别冲击与权重无关，
    # 因此这里的 weight 不参与计算，权重在下面的 by_product 中才相乘
    for asset_class, weight in sorted(portfolio.class_weights().items()):
        by_class[asset_class] = round(asset_class_impact(asset_class, sensitivity, shocks), 12)

    by_product: dict[str, float] = {}
    for pid in portfolio.held_ids():
        product = portfolio.products[pid]
        impact = asset_class_impact(product.asset_class, sensitivity, shocks)
        by_product[pid] = round(portfolio.weights[pid] * impact, 12)
    if portfolio.cash_weight > 0:
        # 现金同样计入冲击（例如利率上行时现金系数为正，是正向贡献）；
        # 用 "CASH" 键占位，避免与真实 product_id 混淆
        cash_impact = asset_class_impact(cash_asset_class, sensitivity, shocks)
        by_product["CASH"] = round(portfolio.cash_weight * cash_impact, 12)
        by_class[cash_asset_class] = round(cash_impact, 12)

    # 组合冲击 = 逐产品冲击之和（现金已折算在内）；
    # 估计最大回撤只取负向冲击：组合上行时回撤为 0，不产生「负回撤」
    portfolio_impact = round(sum(by_product.values()), 12)
    estimated_drawdown = round(max(0.0, -portfolio_impact) * amplification, 12)

    return ScenarioImpact(
        scenario_id=str(scenario.get("scenario_id", "")),
        name=str(scenario.get("name", "")),
        description=str(scenario.get("description", "")),
        portfolio_impact=portfolio_impact,
        estimated_drawdown=estimated_drawdown,
        # 必须「超过」容忍度才算超限：恰好等于容忍度（在 1e-12 容差内）不标记超限
        exceeds_tolerance=estimated_drawdown > client.max_drawdown_tolerance + 1e-12,
        by_asset_class=by_class,
        by_product=by_product,
    )


def run_stress(
    portfolio: Portfolio,
    client: ClientProfile,
    config: Mapping[str, Any],
) -> StressReport:
    """对组合执行全部情景的压力测试。

    参数：`portfolio` —— 待测试组合；`client` —— 提供最大回撤容忍度；
    `config` —— `data/stress_scenarios.json` 的内容，用到 4 个键：`sensitivity`
    （敏感性系数表）、`cash_asset_class`（现金类别名，缺省「现金」）、
    `drawdown_amplification`（回撤放大系数，缺省 1.0）、`scenarios`（情景列表）。
    返回：`StressReport` —— 逐情景结果 + 最不利情景的 id 与冲击 + 公式说明文本。
    缺失键一律走缺省值（`sensitivity` 缺省为空表 → 全部冲击为 0），不抛异常。
    副作用/异常：无。
    """
    sensitivity = config.get("sensitivity", {})
    cash_asset_class = str(config.get("cash_asset_class", "现金"))
    amplification = float(config.get("drawdown_amplification", 1.0))

    impacts = [
        scenario_impact(portfolio, client, scenario, sensitivity, cash_asset_class, amplification)
        for scenario in config.get("scenarios", [])
    ]

    # 最不利情景 = 组合冲击最小（最负）的那个；没有情景时留空，避免 min() 抛异常
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
    """转成表格行，便于 demo / 建议书渲染。

    参数：`report` —— `run_stress` 的产物。
    返回：每条情景一行、固定 6 个键的字典列表 —— `scenario_id`、`name`、
    `portfolio_impact`、`estimated_drawdown`、`exceeds_tolerance`、`by_asset_class`；
    行顺序与 `report.scenarios` 一致。副作用/异常：无。
    """
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
    """压力测试覆盖率（情景数 / 要求情景数，上限 1.0）。

    参数：`report` —— 压力测试报告；传 `None`（未跑压力测试）时视为覆盖率 0。
    返回：`min(1.0, 情景数 / 3)`，四舍五入到 6 位小数。
    用途：`eval/run_eval.py` 的「压力测试覆盖率」指标（阈值 1.0，即每个方案都要有 3 个情景）。
    注：实际实现中分母的要求情景数是字面量 3，与数据文件里的 3 个情景及 README 的
    「拥有 ≥ 3 个情景结果」口径一致；若增减情景数，需要同步修改此处。
    副作用/异常：无。
    """
    if report is None:
        return 0.0
    # 分母 3 = 要求的情景数（固定口径，见上方 docstring 说明）
    return round(min(1.0, report.scenario_count / 3), 6)
