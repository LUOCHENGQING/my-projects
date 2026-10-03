"""投顾建议书撰写（含反事实解释、压力测试、风险揭示与双录留痕标记）。

本模块负责把确定性流水线算出的**全部结构化事实**组织成一份可交付的建议书，
并显式标注要素是否齐全（评估指标「建议书要素完整率」直接复用 `elements`）。

措辞部分（摘要、权衡说明、风险揭示）交给 LLM；无 Key 时由 mock 大脑生成确定性文案。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .counterfactual import counterfactual_to_rows
from .schemas import (
    ClientProfile,
    CounterfactualReport,
    GateDecision,
    HumanReview,
    Portfolio,
    ScreeningResult,
    StressReport,
    risk_label,
)
from .stress import stress_to_rows
from .utils import pct

#: 建议书必备要素（key -> 章节标题），评估指标按此计算完整率
NARRATIVE_ELEMENTS: dict[str, str] = {
    "constraints": "## 一、客户约束回顾",
    "screening": "## 二、候选产品池与剔除说明",
    "allocation": "## 三、配置建议",
    "tradeoff": "## 四、多目标权衡说明",
    "counterfactual": "## 五、反事实解释",
    "stress": "## 六、情景压力测试",
    "suitability": "## 七、适当性规则命中与合规说明",
    "risk": "## 八、风险揭示",
    "fee": "## 九、费率揭示",
    "dual_record": "## 十、双录留痕标记",
    "human_review": "## 十一、人工确认记录",
    "version": "## 十二、建议版本链",
}


def _holdings_table(portfolio: Portfolio, client: ClientProfile) -> list[str]:
    """持仓明细表。"""
    lines = ["| 产品代码 | 产品名称 | 类别 | 风险等级 | 权重 | 参考金额（元） |", "| --- | --- | --- | --- | --- | --- |"]
    amounts = portfolio.holding_amounts(client.investable_amount)
    for pid in portfolio.held_ids():
        product = portfolio.products[pid]
        lines.append(
            f"| {pid} | {product.name} | {product.asset_class} | {risk_label(product.risk_level)} | "
            f"{pct(portfolio.weights[pid])} | {amounts[pid]:,.0f} |"
        )
    if portfolio.cash_weight > 0:
        lines.append(
            f"| CASH | 现金/活期留存 | 现金 | R1-低风险 | {pct(portfolio.cash_weight)} | "
            f"{portfolio.cash_weight * client.investable_amount:,.0f} |"
        )
    return lines


def _metrics_line(portfolio: Portfolio) -> str:
    """核心指标一行。"""
    metrics = portfolio.metrics
    return (
        f"预期年化收益 {pct(metrics.get('expected_return', 0.0))}；"
        f"预期年化波动 {pct(metrics.get('expected_volatility', 0.0))}；"
        f"流动性资产占比 {pct(metrics.get('liquidity_ratio', 0.0))}；"
        f"最大单一持仓 {pct(metrics.get('max_single_weight', 0.0))}；"
        f"综合费率 {pct(metrics.get('expected_fee_rate', 0.0), 3)}；"
        f"持仓只数 {int(metrics.get('holding_count', 0))} 只"
    )


def dual_record_required(client: ClientProfile, portfolio: Portfolio) -> tuple[bool, list[str]]:
    """判断是否触发双录留痕要求，并给出触发原因。"""
    reasons: list[str] = []
    if client.is_elderly:
        reasons.append("客户为高龄客户")
    if any(portfolio.products[pid].risk_level >= 4 for pid in portfolio.held_ids()):
        reasons.append("配置了 R4 及以上风险等级产品")
    if any(portfolio.products[pid].is_derivative for pid in portfolio.held_ids()):
        reasons.append("配置了含衍生品结构的产品")
    return bool(reasons), reasons


def build_narrative(
    *,
    client: ClientProfile,
    portfolio: Portfolio,
    screening: ScreeningResult | None,
    gate: GateDecision | None,
    human_review: HumanReview | None,
    stress: StressReport | None,
    counterfactual: CounterfactualReport | None,
    version: int,
    engine: str,
    run_id: str,
    compose: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
    prior_versions: Sequence[Mapping[str, Any]] = (),
) -> tuple[str, dict[str, bool]]:
    """生成建议书正文与要素齐备情况。

    `compose(task, context)` 由调用方注入（Agent 通过白名单工具 `narrative.compose_text`
    走 LLM/mock），因此本函数本身不直接依赖任何模型客户端，便于单测。
    """
    context: Mapping[str, Any] = {
        "display_name": client.display_name,
        "holding_count": len(portfolio.held_ids()),
        "cash_weight": portfolio.cash_weight,
        "metrics": portfolio.metrics,
        "status": "rejected" if (gate is not None and gate.directive == "reject") else "final",
    }
    summary = str(compose("advisor_summary", context).get("summary", ""))

    tradeoff_ctx = {
        "metrics": portfolio.metrics,
        "binding": [],
        "class_targets": portfolio.class_weights(),
    }
    tradeoff = str(compose("optimizer_rationale", tradeoff_ctx).get("rationale", ""))

    worst = None
    if stress and stress.scenarios:
        worst = min(stress.scenarios, key=lambda item: item.portfolio_impact)
    disclosure = str(
        compose(
            "risk_disclosure",
            {
                "risk_level": client.risk_capacity,
                "drawdown_tolerance": client.max_drawdown_tolerance,
                "worst_scenario_name": worst.name if worst else "情景压力测试",
                "worst_drawdown": worst.estimated_drawdown if worst else 0.0,
            },
        ).get("disclosure", "")
    )

    lines: list[str] = []
    lines.append(f"# 投资顾问建议书（示例演示件）")
    lines.append("")
    lines.append(
        f"- 客户：{client.display_name}（{client.client_id}），{client.age} 周岁"
        f"{'，高龄客户特别保护适用' if client.is_elderly else ''}"
    )
    lines.append(f"- 运行标识：`{run_id}`　推理引擎：`{engine}`　建议版本：v{version}")
    lines.append(f"- 结论：{summary}")
    lines.append("")

    # 一、客户约束回顾
    lines.append(NARRATIVE_ELEMENTS["constraints"])
    lines.append("")
    snapshot = client.constraint_snapshot()
    lines.append(
        f"风险承受等级 **R{client.risk_capacity}**（{risk_label(client.risk_capacity)}）；"
        f"投资期限 **{client.investment_horizon_years:g} 年**；"
        f"流动性资产占比下限 **{pct(client.liquidity_floor_ratio)}**；"
        f"可投金额 **{client.investable_amount:,.0f} 元**。"
    )
    lines.append(
        f"集中度上限：单一产品 **{pct(client.max_single_product_ratio)}**、"
        f"单一资产类别 **{pct(client.max_single_class_ratio)}**、"
        f"同一发行人 **{pct(client.max_single_issuer_ratio)}**；"
        f"内控预警线 **{pct(client.warning_ratio)}**。"
    )
    lines.append(
        f"禁止项：{'、'.join(client.prohibited_categories) or '无'}"
        f"{'；禁止产品：' + '、'.join(client.prohibited_product_ids) if client.prohibited_product_ids else ''}；"
        f"已具备经验品类：{'、'.join(client.experienced_categories) or '无'}；"
        f"币种偏好：{client.currency}；"
        f"合格投资者：{'是' if client.qualified_investor else '否'}。"
    )
    lines.append(
        f"软偏好：收益目标 {pct(client.return_target)}、最大回撤容忍 {pct(client.max_drawdown_tolerance)}、"
        f"费率预算 {pct(client.annual_fee_budget_ratio, 3)}。"
    )
    lines.append("")

    # 二、候选产品池与剔除说明
    lines.append(NARRATIVE_ELEMENTS["screening"])
    lines.append("")
    if screening is not None:
        note = str(
            compose(
                "screening_note",
                {
                    "universe_size": screening.universe_size,
                    "included_count": len(screening.included),
                    "top_exclusions": [
                        f"{item.product_name}（{'/'.join(item.reasons)}）" for item in screening.excluded[:3]
                    ],
                },
            ).get("note", "")
        )
        lines.append(note)
        lines.append("")
        if screening.excluded:
            lines.append("| 被剔除产品 | 原因码 | 说明 |")
            lines.append("| --- | --- | --- |")
            for item in screening.excluded:
                lines.append(f"| {item.product_name} | {'/'.join(item.reasons)} | {item.detail} |")
    lines.append("")

    # 三、配置建议
    lines.append(NARRATIVE_ELEMENTS["allocation"])
    lines.append("")
    if portfolio.held_ids():
        lines.extend(_holdings_table(portfolio, client))
        lines.append("")
        lines.append(f"组合指标：{_metrics_line(portfolio)}")
    else:
        lines.append("硬约束下不存在可行组合，本次不出具配置建议。")
    lines.append("")

    # 四、多目标权衡说明
    lines.append(NARRATIVE_ELEMENTS["tradeoff"])
    lines.append("")
    lines.append(tradeoff)
    lines.append("")
    for item in portfolio.rationale:
        lines.append(f"- {item}")
    lines.append("")

    # 五、反事实解释
    lines.append(NARRATIVE_ELEMENTS["counterfactual"])
    lines.append("")
    if counterfactual is not None:
        note = str(
            compose(
                "counterfactual_note",
                {
                    "variants": [v.variant_id for v in counterfactual.variants],
                    "non_empty": sum(1 for v in counterfactual.variants if not v.is_empty()),
                },
            ).get("note", "")
        )
        lines.append(note)
        lines.append("")
        lines.append(
            "> 说明：反事实以**定稿方案实际依据的生效约束**为基线逐条改动后重新求解；"
            "适当性判定始终针对客户真实档案。差异为空表示方案对该条约束不敏感。"
        )
        lines.append("")
        lines.append("| 假设 | 结论 | 产品增 | 产品减 | 收益变动 | 规则变化 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in counterfactual_to_rows(counterfactual):
            lines.append(
                f"| {row['question']} | {row['status']} | {'、'.join(row['products_added']) or '-'} | "
                f"{'、'.join(row['products_removed']) or '-'} | {row['return_delta'] * 100:+.2f}pp | "
                f"{'、'.join(row['rules_introduced']) or '-'} |"
            )
        lines.append("")
        for variant in counterfactual.variants:
            lines.append(f"- **{variant.question}** {variant.explanation}")
    lines.append("")

    # 六、情景压力测试
    lines.append(NARRATIVE_ELEMENTS["stress"])
    lines.append("")
    if stress is not None:
        lines.append(f"测算方式：{stress.formula}")
        lines.append("")
        lines.append("| 情景 | 组合估值冲击 | 估计最大回撤 | 是否超出客户回撤容忍度 |")
        lines.append("| --- | --- | --- | --- |")
        for row in stress_to_rows(stress):
            lines.append(
                f"| {row['name']} | {row['portfolio_impact'] * 100:+.2f}% | "
                f"{row['estimated_drawdown'] * 100:.2f}% | {'是' if row['exceeds_tolerance'] else '否'} |"
            )
        lines.append("")
        lines.append(f"最不利情景：{stress.worst_scenario_id}，组合估值冲击 {stress.worst_impact * 100:+.2f}%。")
    lines.append("")

    # 七、适当性规则命中与合规说明
    lines.append(NARRATIVE_ELEMENTS["suitability"])
    lines.append("")
    if gate is not None:
        lines.append(gate.comment)
        lines.append("")
        if gate.blocks:
            lines.append("**block 级命中（必须拦截）**")
            lines.append("")
            for violation in gate.blocks:
                lines.append(f"- `{violation.rule_id}` {violation.detail}（依据：{violation.basis}）")
            lines.append("")
        if gate.warns:
            lines.append("**warn 级命中（须揭示）**")
            lines.append("")
            for violation in gate.warns:
                lines.append(f"- `{violation.rule_id}` {violation.detail}（依据：{violation.basis}）")
            lines.append("")
        if not gate.blocks and not gate.warns:
            lines.append("本轮未命中任何适当性规则。")
    lines.append("")

    # 八、风险揭示
    lines.append(NARRATIVE_ELEMENTS["risk"])
    lines.append("")
    lines.append(disclosure)
    lines.append("")

    # 九、费率揭示
    lines.append(NARRATIVE_ELEMENTS["fee"])
    lines.append("")
    fee_rate = portfolio.metrics.get("expected_fee_rate", 0.0)
    lines.append(
        f"组合综合费率（示例口径，按权重加权）：**{pct(fee_rate, 3)}**，"
        f"客户费率预算上限 {pct(client.annual_fee_budget_ratio, 3)}，"
        f"对应年度费用约 **{fee_rate * client.investable_amount:,.0f} 元**。"
    )
    lines.append("各产品费率已在产品要素表中逐项列示，申购费、赎回费、管理费等以产品法律文件为准。")
    lines.append("")

    # 十、双录留痕标记
    lines.append(NARRATIVE_ELEMENTS["dual_record"])
    lines.append("")
    required, reasons = dual_record_required(client, portfolio)
    if required:
        lines.append(
            f"- 双录留痕要求：**触发**（{'；'.join(reasons)}）\n"
            f"- 双录完成状态：{'已完成，记录已归档' if client.dual_record_completed else '**尚未完成，定稿前必须补录**'}"
        )
    else:
        lines.append("- 双录留痕要求：本次未触发（无高龄、无 R4 及以上产品、无衍生品结构）。")
    lines.append("")

    # 十一、人工确认记录
    lines.append(NARRATIVE_ELEMENTS["human_review"])
    lines.append("")
    if human_review is not None and human_review.required:
        lines.append(
            f"- 是否需人工确认：**是**\n"
            f"- 触发原因：{'；'.join(human_review.reasons)}\n"
            f"- 确认结论：{human_review.decision}（操作人：{human_review.operator}）\n"
            f"- 确认时间：{human_review.decided_at}\n"
            f"- 说明：{human_review.note}"
        )
    else:
        lines.append("- 是否需人工确认：否（未触发豁免、内控预警线与高龄客户三类情形）。")
    lines.append("")

    # 十二、建议版本链
    lines.append(NARRATIVE_ELEMENTS["version"])
    lines.append("")
    lines.append(f"- 当前版本：**v{version}**（推理引擎 `{engine}`，运行 `{run_id}`）")
    if prior_versions:
        for item in prior_versions:
            lines.append(
                f"- 历史版本 v{item.get('version')}：{item.get('status')}，"
                f"变更原因「{item.get('change_reason')}」，快照哈希 `{str(item.get('hash', ''))[:12]}`"
            )
    else:
        lines.append("- 历史版本：无（本次为该客户首个建议版本）")
    lines.append("")
    lines.append("> 本建议书为开源演示件，全部客户与产品均为虚构，不构成任何投资建议。")

    text = "\n".join(lines)
    elements = {key: (marker in text) for key, marker in NARRATIVE_ELEMENTS.items()}
    return text, elements


def narrative_completeness(elements: Mapping[str, bool]) -> float:
    """建议书要素完整率。"""
    if not elements:
        return 0.0
    return round(sum(1 for value in elements.values() if value) / len(NARRATIVE_ELEMENTS), 6)
