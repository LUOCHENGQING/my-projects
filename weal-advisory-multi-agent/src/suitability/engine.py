"""适当性规则引擎与硬闸门。

所属层次
--------
领域规则层（`src/suitability/`）：位于 Agent 层（`SuitabilityOfficerAgent`）
之下、纯工具层（`src/constraints.py`、`src/schemas.py`）之上。
本模块**不调用模型**，也不依赖任何 Agent——它只是一个可被反复调用的
确定性判定器，这样"合规"就不受模型波动影响。

解决什么问题
------------
把「规则命中」翻译成「流程动作」：命中 block 不是重写措辞，而是产出
`TightenSpec`（约束收紧指令）让组合在更小的可行域里重新求解；
打回次数用尽转人工；命中 veto 规则直接拒绝。

`SuitabilityGate` 是本项目的核心合规闸门：
- 规则求值完全确定（不调用模型），命中 `block` 必须打回重配或拒绝；
- 打回时下发 `TightenSpec`（约束收紧指令），让组合在**更小的可行域**里重新求解，
  而不是让模型"再想想"——这是与反思循环（reflection loop）式架构的本质区别；
- 收紧次数用尽仍不合规则转人工，`veto` 规则命中则直接拒绝。

对外暴露
--------
- `RuleEvaluation`：一次规则求值的完整结果（含去重的 block/warn 规则号）
- `evaluate_rules(ctx, rules=ALL_RULES)`：纯函数求值入口
- `build_tighten(blocks, ctx)`：把 block 命中翻译成单调收紧指令
- `SuitabilityGate`：闸门对象（`review` 出结论）
- 常量 `DEFAULT_MAX_REPAIR_ROUNDS` / `CAP_TIGHTEN_FACTOR` /
  `LIQUIDITY_TIGHTEN_STEP` / `ELDERLY_RISK_CAP` / `ELDERLY_SINGLE_PRODUCT_CAP`
- 展示辅助 `gate_summary` / `format_rule_table` / `liquidity_line`

被谁调用
--------
`src/suitability/__init__.py`（再导出，`liquidity_line` 不在其列）、
`src/agents/tools.py` 的 `suitability.*` 工具封装（`SuitabilityOfficerAgent` 因此间接受益）、
`eval/run_eval.py` 与 `tests/test_suitability_rules.py`。
"""

from __future__ import annotations

from typing import Iterable, Sequence

from pydantic import BaseModel, Field

from ..constraints import EMPTY_TIGHTEN, TightenSpec
from ..schemas import ClientProfile, GateDecision, Violation
from ..utils import pct
from .rules import ALL_RULES, RULES_BY_ID, Rule, RuleContext

#: 默认最多打回重配轮次
#: 与 `src/pipeline.py` 的 `PipelineConfig.max_repair_rounds` 默认值保持一致；
#: 到第 N 轮仍命中 block 就从"打回"转为"拒绝 + 转人工"。
DEFAULT_MAX_REPAIR_ROUNDS = 2
#: 每轮收紧比例（集中度类上限）
#: 单一产品 / 单一类别 / 同一发行人三类上限每次打回都乘以本系数（×0.8）。
CAP_TIGHTEN_FACTOR = 0.8
#: 流动性下限每轮抬升幅度
#: 流动性下限是"越高越严"，因此每轮按此步长**加**（+5 个百分点）。
LIQUIDITY_TIGHTEN_STEP = 0.05
#: 高龄客户保护等级与集中度上限
#: `S-ELDERLY` 命中后的修复目标：等级上限压到 R3、单一产品上限压到 20%。
ELDERLY_RISK_CAP = 3
ELDERLY_SINGLE_PRODUCT_CAP = 0.20


class RuleEvaluation(BaseModel):
    """一次规则求值的完整结果。

    所有字段都由 `evaluate_rules` 填充，且 `hits` 已按"规则表顺序 + 命中键"
    稳定排序，因此同样的输入必然得到同样的列表顺序（可复现、可 diff）。
    """

    hits: list[Violation] = Field(default_factory=list)
    blocks: list[Violation] = Field(default_factory=list)
    warns: list[Violation] = Field(default_factory=list)
    hit_rule_ids: list[str] = Field(default_factory=list)

    @property
    def block_rule_ids(self) -> list[str]:
        """命中的 block 级规则号（去重、保序）。

        参数：无（属性）。
        返回：规则号列表，顺序为首次命中顺序（`dict.fromkeys` 去重）。
        副作用/异常：无。
        """
        return list(dict.fromkeys(v.rule_id for v in self.blocks))

    @property
    def warn_rule_ids(self) -> list[str]:
        """命中的 warn 级规则号（去重、保序）。

        参数：无（属性）。
        返回：规则号列表，顺序为首次命中顺序。
        副作用/异常：无。
        """
        return list(dict.fromkeys(v.rule_id for v in self.warns))


def evaluate_rules(ctx: RuleContext, rules: Sequence[Rule] = ALL_RULES) -> RuleEvaluation:
    """按固定顺序求值全部规则，输出稳定排序的命中列表。

    参数：
        ctx: `RuleContext`，一次求值的全部输入（客户 / 组合 / 产品池 / 候选池 /
            预先算好的硬约束违反清单）。
        rules: 规则序列，默认全量 22 条（`ALL_RULES`）；传入子集即可只跑部分规则
            （此时输出顺序仍按传入顺序，排序基准也随之改变）。

    返回：
        `RuleEvaluation`：hits（全量命中）、blocks（severity == "block"）、
        warns（severity == "warn"）、hit_rule_ids（去重规则号）。

    副作用：
        会**原地补写** handler 返回的 `Violation` 对象上的 `rule_id` / `basis` /
        `severity`（以规则表定义为准，防止 handler 内外不一致）；
        这些对象是 handler 当场新建的，因此不会污染上下文里的既有数据。
        本函数不写文件、不调模型、不写状态。

    异常：
        无显式抛出；若某条规则 handler 抛错则原样向上传播（规则应保持纯函数）。
    """
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

    # 先按规则表顺序、再按命中自身的稳定键排序，保证输出可复现
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
    """收集命中条目涉及的产品号。

    参数：
        blocks: 命中条目序列（只读取其中的 `product_ids`）。
        ctx: 规则上下文；本实现只用于保持签名一致，**未读取任何字段**。

    返回：
        去重并排序后的产品号列表。

    副作用/异常：无。

    注：实际实现为——`build_tighten` 目前直接使用 `violation.product_ids`，
    本函数在当前仓库内**没有任何调用方**（保留为后续扩展/对齐用的辅助函数）。
    """
    return sorted({pid for violation in blocks for pid in violation.product_ids})


def build_tighten(blocks: Sequence[Violation], ctx: RuleContext) -> TightenSpec:
    """把 block 命中翻译成**单调收紧**的约束指令（打回重配的核心）。

    映射依据是规则定义里的 `repair` 提示字段（见 `rules.py` 的规则表），
    各分支对应的规则如下（与 `RULES_BY_ID` 一致，共 10 种提示取值）：

    ==================== ========================================== ================================
    repair 提示          对应规则                                     产生的收紧动作
    ==================== ========================================== ================================
    `risk_cap_down`      S-RISK-MATCH                                等级上限降到违规产品最低等级 -1
    `horizon_down`       S-HORIZON                                   期限上限降到违规产品最短期限的一半
    `single_cap_down`    S-CONC-SINGLE                               单一产品上限 ×0.8（下限 1%）
    `class_cap_down`     S-CONC-CLASS                                单一类别上限 ×0.8（下限 1%）
    `issuer_cap_down`    S-CONCENTRATION-ISSUER                      同一发行人上限 ×0.8（下限 1%）
    `liquidity_up`       S-LIQUIDITY                                 流动性下限 +5pp
    `exclude_products`   S-ENTRY / S-EXPERIENCE / S-CURRENCY /       把违规产品加入禁止清单
                         S-QUALIFIED / S-PROHIBITED
    `exclude_categories` S-DERIVATIVE-BAN                            违规产品及其所属类别一并排除
    `elderly_protect`    S-ELDERLY                                   等级上限压到 R3、单一产品压到 20%
    `normalize`          S-WEIGHT                                    本条**无专门分支**（落入 else），
                                                                    不产生收紧动作
    `""`（空）           S-FEASIBLE-POOL（veto）及 7 条 warn 规则     不产生收紧动作
    ==================== ========================================== ================================

    参数：
        blocks: 需要修复的违规条目（通常来自 `GateDecision.blocks`，
            即已剔除人工豁免后的 block 命中）。
        ctx: 规则上下文，提供客户档案（作为收紧基准）与组合（用于回查
            违规产品的等级/期限/类别）。

    返回：
        `TightenSpec`：多条命中的收紧动作经 `merge` 合并后的**单调收紧**指令
        （数值取更严一侧、禁止项取并集）；无任何动作时等于 `EMPTY_TIGHTEN`。
        若产生了实际动作，还会附上 `reasons`（形如 `"S-XXX: 违规事实"`），
        用于在建议书与运行日志里说明"为什么收紧"。

    副作用/异常：
        无副作用（不修改 `ctx`，也不写状态）；产品号不在 `ctx.portfolio.products`
        内时按"无违规产品信息"处理（退化为按客户现有约束收紧），不抛异常。
    """
    client = ctx.client
    spec = EMPTY_TIGHTEN
    reasons: list[str] = []

    for violation in blocks:
        # repair 提示来自规则定义；未知规则号（或规则已被移除）视为无提示
        rule = RULES_BY_ID.get(violation.rule_id)
        hint = rule.repair if rule else ""
        step = TightenSpec()

        if hint == "risk_cap_down":
            # 等级上限：取违规产品中最低风险等级再降一级；查不到产品则按客户等级降一级
            offenders = [ctx.portfolio.products[pid] for pid in violation.product_ids if pid in ctx.portfolio.products]
            if offenders:
                level = min(p.risk_level for p in offenders) - 1
            else:
                level = client.risk_capacity - 1
            step = TightenSpec(risk_cap=max(1, level))
        elif hint == "horizon_down":
            # 期限上限：取违规产品最短期限的一半（下限 0.1 年，避免压到 0）
            offenders = [ctx.portfolio.products[pid] for pid in violation.product_ids if pid in ctx.portfolio.products]
            if offenders:
                horizon = max(0.1, min(p.horizon_years for p in offenders) * 0.5)
            else:
                horizon = max(0.1, client.investment_horizon_years * 0.5)
            step = TightenSpec(horizon_years=horizon)
        elif hint == "single_cap_down":
            # 单一产品集中度上限按比例压缩（下限 1%，避免上限被压到 0）
            step = TightenSpec(max_single_product_ratio=max(0.01, client.max_single_product_ratio * CAP_TIGHTEN_FACTOR))
        elif hint == "class_cap_down":
            # 单一资产类别上限按比例压缩
            step = TightenSpec(max_single_class_ratio=max(0.01, client.max_single_class_ratio * CAP_TIGHTEN_FACTOR))
        elif hint == "issuer_cap_down":
            # 同一发行人上限按比例压缩
            step = TightenSpec(max_single_issuer_ratio=max(0.01, client.max_single_issuer_ratio * CAP_TIGHTEN_FACTOR))
        elif hint == "liquidity_up":
            # 流动性下限是"越高越严"，因此这里做加法（上限由 TightenSpec.apply 截到 1.0）
            step = TightenSpec(liquidity_floor_ratio=client.liquidity_floor_ratio + LIQUIDITY_TIGHTEN_STEP)
        elif hint == "exclude_products":
            # 直接排除违规产品（起投/币种/经验/准入/禁止项类规则的统一修法）
            step = TightenSpec(excluded_product_ids=tuple(violation.product_ids))
        elif hint == "exclude_categories":
            # 衍生品禁令：产品与"该产品所属类别"一起排除，避免同类换个产品再犯
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
            # 高龄保护：等级上限与单一产品上限同时收紧到保护档位
            step = TightenSpec(
                risk_cap=ELDERLY_RISK_CAP,
                max_single_product_ratio=ELDERLY_SINGLE_PRODUCT_CAP,
            )
        else:
            # 无修复策略（例如 veto 规则）或 normalize：不产生收紧
            step = EMPTY_TIGHTEN

        if not step.is_empty():
            reasons.append(f"{violation.rule_id}: {violation.detail}")
            # merge 保证多轮/多条命中叠加后依然单调收紧（不会来回震荡）
            spec = spec.merge(step)

    if reasons:
        spec = spec.merge(TightenSpec(reasons=tuple(reasons)))
    return spec


class SuitabilityGate:
    """适当性合规硬闸门。

    纯判定对象：不调用模型、不写状态、不依赖 Agent；输入 `RuleContext`，
    输出 `GateDecision`。它把"规则命中"翻译成三种流程指令：
    `pass`（无未豁免 block）/ `reoptimize`（有 block 且还有重配机会）/
    `reject`（命中 veto 规则，或打回次数用尽）。
    """

    def __init__(self, max_repair_rounds: int = DEFAULT_MAX_REPAIR_ROUNDS) -> None:
        """设置允许打回重配的最大轮次。

        参数：
            max_repair_rounds: 非负整数；`review` 中当
                `round_index >= max_repair_rounds` 且仍有 block 命中时，
                指令从 `reoptimize` 转为 `reject` 并标记 `escalated`。

        返回：None。

        副作用/异常：
            无副作用；为负数时抛 ValueError（轮次语义上不允许为负，
            否则首轮就会直接拒绝，容易掩盖配置错误）。
        """
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
        """对候选组合执行适当性复核，返回闸门结论。

        判定顺序（与实现一致）：
        1. 求值全量规则得到 hits / blocks / warns；
        2. 把 `exempt_rules` 里的 block 命中降级为 warn 并留痕（"【经人工豁免】"）；
        3. 剩余 block 中若含 `veto=True` 的规则 → `reject`；
        4. 否则若有 block：轮次未用尽 → `reoptimize`（附收紧指令），
           轮次用尽 → `reject` + `escalated=True`；
        5. 否则 → `pass`。

        参数：
            ctx: 规则上下文（客户 / 组合 / 产品池 / 候选池 / 硬约束违反清单）。
            round_index: 本轮是第几轮重配（0 为首轮）；只有"已用满
                `max_repair_rounds`"的语义，不看真实时间或次数。
            exempt_rules: 经人工批准的豁免规则号；只对 severity == "block"
                的命中生效，命中的条目会被改写成 severity="warn" 的副本。

        返回：
            `GateDecision`：
            - `passed`：**只看未被豁免的 block**（warn 与豁免项都不影响它）；
            - `directive`：`pass` / `reoptimize` / `reject`；
            - `blocks` / `warns`：有效 block 与全部 warn（含降级后的豁免项）；
            - `veto_rules`：命中的不可修复规则号（去重）；
            - `exempted_rules`：本次被豁免的规则号（排序）；
            - `escalated`：是否因打回次数用尽而转人工；
            - `tighten`：仅 `reoptimize` 时为 `build_tighten(...).to_dict()`，
              其余情况为 `{}`；
            - `comment`：确定性结论说明（模型措辞由 Agent 另外补充）。

        副作用/异常：
            无副作用（不修改 `ctx`，也不写文件）；实现上是纯函数式判定。
        """
        evaluation = evaluate_rules(ctx)
        exempt = {rid for rid in exempt_rules}

        # 豁免只对 block 生效：命中项降级为 warn 副本，原始命中不进入 blocks
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

        # veto 规则：命中即拒，不接受"再收紧一次"的修复路径
        veto_rules = [v.rule_id for v in effective_blocks if RULES_BY_ID.get(v.rule_id, None) and RULES_BY_ID[v.rule_id].veto]
        veto_rules = list(dict.fromkeys(veto_rules))

        escalated = False
        if veto_rules:
            directive = "reject"
        elif effective_blocks:
            if round_index >= self.max_repair_rounds:
                # 打回额度用尽：不再收紧，转人工处理（reject + escalated 双标记）
                directive = "reject"
                escalated = True
            else:
                directive = "reoptimize"
        else:
            directive = "pass"

        # 只在打回时携带收紧指令；其余情况给空 dict，避免下游误用
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
        """生成确定性的闸门结论说明（模型的措辞另由 Agent 补充）。

        参数：
            directive: 闸门指令（`pass` / `reoptimize` / `reject`）。
            blocks: 未被豁免的 block 命中（用于统计条数与列规则号）。
            warns: 全部 warn 条目（含由豁免降级而来的条目）。
            round_index: 本轮轮次（说明文字里按 1 基展示）。
            veto_rules: 命中的 veto 规则号（用于区分"直接拒绝"与"超限转人工"）。
            exempted: 被人工豁免的条目（用于附加"经人工豁免"说明）。

        返回：
            单条中文说明字符串（用「；」连接各片段，以「。」结尾）；
            同样的输入必然得到同样的文字，可直接作为留痕依据。

        副作用/异常：无（纯格式化）。
        """
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
    """把闸门结论压缩成便于评估统计的摘要。

    参数：
        decision: `GateDecision`（闸门结论）。

    返回：
        扁平字典：`round_index` / `passed` / `directive` / `block_rules` /
        `warn_rules`（后两者已去重）/ `escalated`；
        刻意**不含** tighten 与 comment 等长字段，方便写进评估报告 JSON。

    副作用/异常：无。
    """
    return {
        "round_index": decision.round_index,
        "passed": decision.passed,
        "directive": decision.directive,
        "block_rules": list(dict.fromkeys(v.rule_id for v in decision.blocks)),
        "warn_rules": list(dict.fromkeys(v.rule_id for v in decision.warns)),
        "escalated": decision.escalated,
    }


def format_rule_table() -> str:
    """规则清单的纯文本表格（demo 打印用）。

    参数：无（读取 `rules.ALL_RULES` 的当前顺序）。

    返回：
        多行字符串：首行表头，其后每行一条规则（规则号 / 级别 / 分类 / 名称），
        用固定宽度左对齐（中文按字符宽度近似，仅用于终端展示）。

    副作用/异常：无。
    """
    lines = [f"{'规则号':<26}{'级别':<7}{'分类':<10}规则名称"]
    for rule in ALL_RULES:
        lines.append(f"{rule.id:<26}{rule.severity:<8}{rule.category:<11}{rule.name}")
    return "\n".join(lines)


def liquidity_line(client: ClientProfile, liquid: float) -> str:
    """流动性一行摘要。

    参数：
        client: 客户档案（只取 `liquidity_floor_ratio` 作为下限展示）。
        liquid: 实际流动性资产占比（小数，如 0.35 表示 35%）。

    返回：
        形如 `流动性资产占比 35.0%（下限 20.0%）` 的单行中文摘要。

    副作用/异常：无。

    注：实际实现为——本函数**未**在 `src/suitability/__init__.py` 的
    `__all__` 中导出，当前仓库内也没有任何调用方（保留为展示辅助）。
    """
    return f"流动性资产占比 {pct(liquid)}（下限 {pct(client.liquidity_floor_ratio)}）"
