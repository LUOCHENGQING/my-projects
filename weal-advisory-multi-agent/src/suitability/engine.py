"""适当性规则引擎与硬闸门。

`SuitabilityGate` 是本项目的核心合规闸门：
- 规则求值完全确定（不调用模型），命中 `block` 必须打回重配或拒绝；
- 打回时下发 `TightenSpec`（约束收紧指令），让组合在**更小的可行域**里重新求解，
  而不是让模型"再想想"——这是与反思循环（reflection loop）式架构的本质区别；
- 收紧次数用尽仍不合规则转人工，`veto` 规则命中则直接拒绝。
"""

from __future__ import annotations

from typing import Iterable, Sequence

from pydantic import BaseModel, Field

from ..constraints import EMPTY_TIGHTEN, TightenSpec
from ..schemas import ClientProfile, GateDecision, Violation
from ..utils import pct
from .rules import ALL_RULES, RULES_BY_ID, Rule, RuleContext

#: 默认最多打回重配轮次
DEFAULT_MAX_REPAIR_ROUNDS = 2
#: 每轮收紧比例（集中度类上限）
CAP_TIGHTEN_FACTOR = 0.8
#: 流动性下限每轮抬升幅度
LIQUIDITY_TIGHTEN_STEP = 0.05
#: 高龄客户保护等级与集中度上限
ELDERLY_RISK_CAP = 3
ELDERLY_SINGLE_PRODUCT_CAP = 0.20


class RuleEvaluation(BaseModel):
    """一次规则求值的完整结果。"""

    hits: list[Violation] = Field(default_factory=list)
    blocks: list[Violation] = Field(default_factory=list)
    warns: list[Violation] = Field(default_factory=list)
    hit_rule_ids: list[str] = Field(default_factory=list)

    @property
    def block_rule_ids(self) -> list[str]:
        """命中的 block 级规则号（去重、保序）。"""
        return list(dict.fromkeys(v.rule_id for v in self.blocks))

    @property
    def warn_rule_ids(self) -> list[str]:
        """命中的 warn 级规则号（去重、保序）。"""
        return list(dict.fromkeys(v.rule_id for v in self.warns))


def evaluate_rules(ctx: RuleContext, rules: Sequence[Rule] = ALL_RULES) -> RuleEvaluation:
    """按固定顺序求值全部规则，输出稳定排序的命中列表。"""
    hits: list[Violation] = []
    for rule in rules:
        produced = rule.handler(ctx)
        for violation in produced:
            # 统一以规则定义为准，防止 handler 内外不一致
            violation.rule_id = rule.id
            if not violation.basis:
                violation.basis = rule.basis
            if not violation.severity:
                violation.severity = rule.severity
            hits.append(violation)

    order = {rule.id: index for index, rule in enumerate(rules)}
    hits.sort(key=lambda v: (order.get(v.rule_id, 10_000), v.key()))

    blocks = [v for v in hits if v.severity == "block"]
    warns = [v for v in hits if v.severity == "warn"]
    return RuleEvaluation(
        hits=hits,
        blocks=blocks,
        warns=warns,
        hit_rule_ids=list(dict.fromkeys(v.rule_id for v in hits)),
    )


def _offending_products(blocks: Iterable[Violation], ctx: RuleContext) -> list[str]:
    """收集命中条目涉及的产品号。"""
    return sorted({pid for violation in blocks for pid in violation.product_ids})


def build_tighten(blocks: Sequence[Violation], ctx: RuleContext) -> TightenSpec:
    """把 block 命中翻译成**单调收紧**的约束指令（打回重配的核心）。"""
    client = ctx.client
    spec = EMPTY_TIGHTEN
    reasons: list[str] = []

    for violation in blocks:
        rule = RULES_BY_ID.get(violation.rule_id)
        hint = rule.repair if rule else ""
        step = TightenSpec()

        if hint == "risk_cap_down":
            offenders = [ctx.portfolio.products[pid] for pid in violation.product_ids if pid in ctx.portfolio.products]
            if offenders:
                level = min(p.risk_level for p in offenders) - 1
            else:
                level = client.risk_capacity - 1
            step = TightenSpec(risk_cap=max(1, level))
        elif hint == "horizon_down":
            offenders = [ctx.portfolio.products[pid] for pid in violation.product_ids if pid in ctx.portfolio.products]
            if offenders:
                horizon = max(0.1, min(p.horizon_years for p in offenders) * 0.5)
            else:
                horizon = max(0.1, client.investment_horizon_years * 0.5)
            step = TightenSpec(horizon_years=horizon)
        elif hint == "single_cap_down":
            step = TightenSpec(max_single_product_ratio=max(0.01, client.max_single_product_ratio * CAP_TIGHTEN_FACTOR))
        elif hint == "class_cap_down":
            step = TightenSpec(max_single_class_ratio=max(0.01, client.max_single_class_ratio * CAP_TIGHTEN_FACTOR))
        elif hint == "issuer_cap_down":
            step = TightenSpec(max_single_issuer_ratio=max(0.01, client.max_single_issuer_ratio * CAP_TIGHTEN_FACTOR))
        elif hint == "liquidity_up":
            step = TightenSpec(liquidity_floor_ratio=client.liquidity_floor_ratio + LIQUIDITY_TIGHTEN_STEP)
        elif hint == "exclude_products":
            step = TightenSpec(excluded_product_ids=tuple(violation.product_ids))
        elif hint == "exclude_categories":
            classes = {
                ctx.portfolio.products[pid].asset_class
                for pid in violation.product_ids
                if pid in ctx.portfolio.products
            }
            step = TightenSpec(
                excluded_product_ids=tuple(violation.product_ids),
                excluded_categories=tuple(sorted(classes)),
            )
        elif hint == "elderly_protect":
            step = TightenSpec(
                risk_cap=ELDERLY_RISK_CAP,
                max_single_product_ratio=ELDERLY_SINGLE_PRODUCT_CAP,
            )
        else:
            # 无修复策略（例如 veto 规则）或 normalize：不产生收紧
            step = EMPTY_TIGHTEN

        if not step.is_empty():
            reasons.append(f"{violation.rule_id}: {violation.detail}")
            spec = spec.merge(step)

    if reasons:
        spec = spec.merge(TightenSpec(reasons=tuple(reasons)))
    return spec


class SuitabilityGate:
    """适当性合规硬闸门。"""

    def __init__(self, max_repair_rounds: int = DEFAULT_MAX_REPAIR_ROUNDS) -> None:
        if max_repair_rounds < 0:
            raise ValueError("max_repair_rounds 不能为负数")
        self.max_repair_rounds = max_repair_rounds

    # ------------------------------------------------------------------
    def review(
        self,
        ctx: RuleContext,
        round_index: int = 0,
        exempt_rules: Iterable[str] = (),
    ) -> GateDecision:
        """对候选组合执行适当性复核，返回闸门结论。"""
        evaluation = evaluate_rules(ctx)
        exempt = {rid for rid in exempt_rules}

        exempted = [v for v in evaluation.blocks if v.rule_id in exempt]
        effective_blocks = [v for v in evaluation.blocks if v.rule_id not in exempt]
        warns = list(evaluation.warns)
        for violation in exempted:
            warns.append(
                Violation(
                    rule_id=violation.rule_id,
                    severity="warn",
                    detail=f"【经人工豁免】{violation.detail}",
                    basis=violation.basis,
                    product_ids=list(violation.product_ids),
                    code=violation.rule_id,
                )
            )

        veto_rules = [v.rule_id for v in effective_blocks if RULES_BY_ID.get(v.rule_id, None) and RULES_BY_ID[v.rule_id].veto]
        veto_rules = list(dict.fromkeys(veto_rules))

        escalated = False
        if veto_rules:
            directive = "reject"
        elif effective_blocks:
            if round_index >= self.max_repair_rounds:
                directive = "reject"
                escalated = True
            else:
                directive = "reoptimize"
        else:
            directive = "pass"

        tighten = (
            build_tighten(effective_blocks, ctx).to_dict()
            if directive == "reoptimize"
            else {}
        )

        return GateDecision(
            round_index=round_index,
            passed=not effective_blocks,
            directive=directive,  # type: ignore[arg-type]
            blocks=effective_blocks,
            warns=warns,
            veto_rules=veto_rules,
            exempted_rules=sorted(exempt),
            escalated=escalated,
            tighten=tighten,
            comment=self._comment(directive, effective_blocks, warns, round_index, veto_rules, exempted),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _comment(
        directive: str,
        blocks: Sequence[Violation],
        warns: Sequence[Violation],
        round_index: int,
        veto_rules: Sequence[str],
        exempted: Sequence[Violation],
    ) -> str:
        """生成确定性的闸门结论说明（模型的措辞另由 Agent 补充）。"""
        parts: list[str] = []
        if directive == "pass":
            parts.append(f"第 {round_index + 1} 轮复核通过：未命中 block 级适当性规则")
        elif directive == "reoptimize":
            parts.append(
                f"第 {round_index + 1} 轮复核不通过：命中 {len(blocks)} 条 block 级规则，"
                "已下发约束收紧指令，打回重新配置"
            )
        elif veto_rules:
            parts.append(f"复核直接拒绝：命中不可修复规则 {'、'.join(veto_rules)}")
        else:
            parts.append(f"第 {round_index + 1} 轮复核仍不通过：打回重配次数用尽，转人工处理")
        if exempted:
            parts.append(f"其中 {len(exempted)} 条经人工豁免降级为告警")
        if warns:
            parts.append(f"另有 {len(warns)} 条告警级提示需在建议书中揭示")
        block_rules = list(dict.fromkeys(v.rule_id for v in blocks))
        if block_rules:
            parts.append(f"命中规则：{'、'.join(block_rules)}")
        return "；".join(parts) + "。"


def gate_summary(decision: GateDecision) -> dict[str, object]:
    """把闸门结论压缩成便于评估统计的摘要。"""
    return {
        "round_index": decision.round_index,
        "passed": decision.passed,
        "directive": decision.directive,
        "block_rules": list(dict.fromkeys(v.rule_id for v in decision.blocks)),
        "warn_rules": list(dict.fromkeys(v.rule_id for v in decision.warns)),
        "escalated": decision.escalated,
    }


def format_rule_table() -> str:
    """规则清单的纯文本表格（demo 打印用）。"""
    lines = [f"{'规则号':<26}{'级别':<7}{'分类':<10}规则名称"]
    for rule in ALL_RULES:
        lines.append(f"{rule.id:<26}{rule.severity:<8}{rule.category:<11}{rule.name}")
    return "\n".join(lines)


def liquidity_line(client: ClientProfile, liquid: float) -> str:
    """流动性一行摘要。"""
    return f"流动性资产占比 {pct(liquid)}（下限 {pct(client.liquidity_floor_ratio)}）"
