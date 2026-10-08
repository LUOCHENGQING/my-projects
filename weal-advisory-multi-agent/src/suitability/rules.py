"""适当性规则库（Deterministic Suitability Rule Library）。

所属层次
--------
领域规则层（`src/suitability/`）的最底层：不依赖 Agent、不调用模型、不写状态，
只把「客户 + 组合 + 产品池」映射成「带规则号与依据的命中清单」。
上层是 `engine.py`（求值 + 闸门），再上层是 `SuitabilityOfficerAgent`。

解决什么问题
------------
让合规判定**可枚举、可单测、可打印、可追溯**：把"适当性"从散落在 if 里的
经验判断，变成一张有编号、有依据、有修复策略的规则表。

规则条数（与代码严格一致，勿凭印象改动）
----------------------------------------
`ALL_RULES` 共 **22 条**：
- **15 条 block**（命中必须打回重配或拒绝）：
  S-RISK-MATCH、S-HORIZON、S-CONC-SINGLE、S-CONC-CLASS、S-CONCENTRATION-ISSUER、
  S-LIQUIDITY、S-ENTRY、S-ELDERLY、S-EXPERIENCE、S-DERIVATIVE-BAN、S-CURRENCY、
  S-QUALIFIED、S-PROHIBITED、S-WEIGHT、S-FEASIBLE-POOL；
- **7 条 warn**（只揭示与人工提示，不作为拦截理由）：
  S-DUAL-RECORD、S-FEE-DISCLOSE、S-TAX、S-COOLING、S-CONCENTRATION-WARNING、
  S-CASH-RESERVE、S-DIVERSIFICATION。
其中 **仅 1 条带 `veto=True`**：S-FEASIBLE-POOL（可行域为空，不可修复，直接拒绝）。

设计原则
--------
1. **规则即数据 + 纯函数**：每条规则是 `Rule(id/name/severity/category/basis/handler/repair)`，
   没有隐式状态，可单测、可枚举、可打印成规则清单。
2. **不与硬约束求解器重复造轮子**：凡是数值型硬约束（等级、期限、集中度、起投、
   流动性）一律复用 `src/constraints.py` 的判定结果，规则层只负责把它翻译成带
   规则号与依据说明的适当性条目，并叠加适当性专有规则（高龄保护、经验、双录、
   冷静期、税收优惠、费率揭示、内控预警线等）。
3. **block 必须拦**：`severity == "block"` 的规则一旦命中，闸门必须打回重配或直接拒绝，
   不允许「带病通过」。`severity == "warn"` 只做揭示与人工提示。

对外暴露
--------
- `RuleContext`（求值上下文，`build()` 构造）、`Rule`（规则定义）
- 常量 `ALL_RULES`（22 条，顺序即评估顺序）、`RULES_BY_ID`
- 函数 `get_rule` / `rule_catalog` / `has_block_rule`
- 阈值常量（`ELDERLY_RISK_LIMIT` 等）与 `CONSTRAINT_TO_RULE` 对照表

被谁调用
--------
`src/suitability/engine.py`（`evaluate_rules` / `build_tighten` / `format_rule_table`）、
`src/suitability/__init__.py`（再导出）、`src/agents/tools.py`（间接）、
`tests/test_suitability_rules.py`、`eval/run_eval.py`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from ..constraints import admissibility_detail, check_portfolio, check_product_admissibility
from ..schemas import ClientProfile, Portfolio, Product, Severity, Violation
from ..utils import pct

# ---------------------------------------------------------------------------
# 适当性专有阈值（示例口径，均可在数据层调整）
# ---------------------------------------------------------------------------
ELDERLY_RISK_LIMIT = 4          # 高龄客户禁止持有 R4 及以上产品
ELDERLY_SINGLE_PRODUCT_CAP = 0.20  # 高龄客户单一产品集中度进一步压降到 20%
CASH_RESERVE_FLOOR = 0.03       # 建议保留的现金缓冲下限
MIN_HOLDINGS = 3                # 组合分散度下限（产品只数）
COOLING_HORIZON_YEARS = 1.0     # 触发冷静期/封闭期提示的期限
COOLING_LIQUIDITY = 0.3         # 触发冷静期/封闭期提示的流动性比例

# 硬约束原因码 -> 适当性规则号（单一事实来源：数值判定只在 constraints.py 里写一次）
# 说明：数值是否越界只由 `src/constraints.py` 判定一次，这里只做"原因码 → 规则号"的
# 翻译，`_from_constraints()` 生成的 handler 会按此表筛出对应命中。
# 注：实际实现为——各规则的 handler 是**内联**原因码列表
# （如 `_from_constraints(RULE_RISK_MATCH, ["C-RISK"])`），本表在 `rules.py`
# 内部**未被引用**，当前仅作为对照/文档用途（供阅读者与外部分析脚本核对）。
CONSTRAINT_TO_RULE: dict[str, str] = {
    "C-RISK": "S-RISK-MATCH",
    "C-HORIZON": "S-HORIZON",
    "C-CONC-SINGLE": "S-CONC-SINGLE",
    "C-CONC-CLASS": "S-CONC-CLASS",
    "C-CONC-ISSUER": "S-CONCENTRATION-ISSUER",
    "C-LIQUIDITY": "S-LIQUIDITY",
    "C-ENTRY": "S-ENTRY",
    "C-ENTRY-MIN": "S-ENTRY",
    "C-EXPERIENCE": "S-EXPERIENCE",
    "C-DERIVATIVE-BAN": "S-DERIVATIVE-BAN",
    "C-CURRENCY": "S-CURRENCY",
    "C-QUALIFIED": "S-QUALIFIED",
    "C-PROHIBITED-CLASS": "S-PROHIBITED",
    "C-PROHIBITED-PRODUCT": "S-PROHIBITED",
    "C-WEIGHT-SUM": "S-WEIGHT",
    "C-WEIGHT-NEG": "S-WEIGHT",
}


@dataclass(frozen=True)
class RuleContext:
    """规则求值上下文：一次求值的全部输入都是冻结的。

    冻结（`frozen=True`）是刻意的：规则 handler 只能读、不能改输入，
    因此同一次求值的所有规则看到的是同一份事实，闸门结论可复现。
    字段含义：
    - client：生效客户档案（约束基准）；
    - portfolio：待复核组合（权重、现金、指标）；
    - universe：全量产品池（算类别/发行人集中度、可行域是否为空都要它）；
    - candidates：当前候选池产品号（已排序去重；为空 → 触发 veto 规则）；
    - constraint_violations：`constraints.check_portfolio` 预先算好的硬约束违反清单，
      供"复用型"规则直接翻译，避免同一套数值判定写两遍。
    """

    client: ClientProfile
    portfolio: Portfolio
    universe: Mapping[str, Product]
    candidates: tuple[str, ...]
    constraint_violations: tuple[Violation, ...]

    @classmethod
    def build(
        cls,
        client: ClientProfile,
        portfolio: Portfolio,
        universe: Mapping[str, Product],
        candidates: Iterable[str],
    ) -> "RuleContext":
        """构造上下文，并复用硬约束求解器的判定结果。

        参数：
            client: 生效客户档案。
            portfolio: 待复核组合。
            universe: 全量产品池。
            candidates: 候选池产品号（本方法内排序去重后再冻结）。

        返回：
            `RuleContext` 实例，其 `constraint_violations` 为
            `check_portfolio(portfolio, client)` 的元组化结果。

        副作用/异常：
            无副作用（只做纯计算与排序）；`check_portfolio` 内部不写状态。
        """
        violations = tuple(check_portfolio(portfolio, client))
        return cls(
            client=client,
            portfolio=portfolio,
            universe=universe,
            candidates=tuple(sorted(candidates)),
            constraint_violations=violations,
        )

    def held_products(self) -> list[Product]:
        """持有产品（按产品号排序）。

        参数：无。
        返回：组合中权重对应的产品对象列表，顺序为 `Portfolio.held_ids()` 的顺序；
            产品号在 `portfolio.products` 中缺失时会被静默跳过。
        副作用/异常：无。
        """
        return [self.portfolio.products[pid] for pid in self.portfolio.held_ids() if pid in self.portfolio.products]

    def weight_of(self, product_id: str) -> float:
        """某产品权重。

        参数：
            product_id: 产品号。
        返回：该产品在组合中的权重（小数）；未持有时返回 0.0（不抛 KeyError）。
        副作用/异常：无。
        """
        return self.portfolio.weights.get(product_id, 0.0)


@dataclass(frozen=True)
class Rule:
    """一条适当性规则。

    字段含义（构造后不可修改，保证规则表在运行期是常量）：
    - `id`：规则号（如 `"S-RISK-MATCH"`），命中记录与豁免都以此为准；
    - `name`：中文规则名（打印规则清单、写复核意见用）；
    - `severity`：`"block"`（必须拦）或 `"warn"`（只揭示），取值受 `Severity` 约束；
    - `category`：分类（适当性匹配 / 集中度管理 / 流动性管理 / 投资者保护 /
      禁止项 / 准入管理 / 信息披露 / 销售留痕 / 交易可行性 / 组合合法性 / 组合管理）；
    - `basis`：监管/内控依据说明，写入命中记录供监管问询复核；
    - `handler`：`RuleContext -> list[Violation]` 的**纯函数**，命中即返回条目，
      未命中返回空列表；
    - `repair`：修复提示，由 `engine.build_tighten` 解释成具体收紧动作
      （如 `"risk_cap_down"` / `"exclude_products"` / `"normalize"`）；
      空字符串表示"无修复策略"；
    - `veto`：`True` 表示命中即拒、**不可**通过收紧约束修复（目前仅
      S-FEASIBLE-POOL 一条）；
    - `description`：给人看的一句话说明。
    """

    id: str
    name: str
    severity: Severity
    category: str
    basis: str
    handler: Callable[[RuleContext], list[Violation]]
    repair: str = ""
    veto: bool = False
    description: str = ""


def _violation(
    rule: "Rule | str",
    detail: str,
    product_ids: Iterable[str] = (),
    basis: str = "",
    severity: Severity | None = None,
) -> Violation:
    """按规则定义生成命中记录。

    参数：
        rule: 规则对象（推荐，自动带出 id/severity/basis）或规则号字符串。
        detail: 违规事实的中文说明（含具体数值，便于向客户解释）。
        product_ids: 涉及的产品号（会自动去重排序）。
        basis: 依据说明；传空则回落到规则自带的 `basis`。
        severity: 覆盖级别；传 None 用规则自身级别。例外的用法见
            `_dual_record_handler`（它显式传 `"warn"` 以强调只提示不拦截）。

    返回：
        `Violation`（`code` 与 `rule_id` 同值，便于按原因码检索）。

    副作用/异常：
        无副作用；`rule` 为字符串且未给 severity 时默认按 `"block"` 处理
        （保守取值：宁可拦也不放过），不抛异常。
    """
    if isinstance(rule, Rule):
        rule_id, rule_severity, rule_basis = rule.id, rule.severity, rule.basis
    else:
        rule_id, rule_severity, rule_basis = rule, (severity or "block"), basis
    return Violation(
        rule_id=rule_id,
        severity=severity or rule_severity,
        detail=detail,
        basis=basis or rule_basis,
        product_ids=sorted(set(product_ids)),
        code=rule_id,
    )


def _from_constraints(rule: "Rule", codes: Sequence[str]) -> Callable[[RuleContext], list[Violation]]:
    """把一个或多个硬约束原因码映射成该规则的命中列表。

    这是"规则层不重复造轮子"的落地方式：数值判定只在 `constraints.py` 做一次，
    这里只做原因码筛选 + 包装成带规则号的命中。

    参数：
        rule: 目标规则（新命中的 rule_id/severity/basis 以它为准）。
        codes: 该规则认领的硬约束原因码集合，如 `["C-ENTRY", "C-ENTRY-MIN"]`。

    返回：
        闭包 handler：`(ctx) -> list[Violation]`，逐条把 `ctx.constraint_violations`
        中 `code` 属于 `codes` 的条目翻译成该规则的命中；没有匹配时返回空列表。

    副作用/异常：
        无副作用（只读 `ctx.constraint_violations`）；`codes` 为空时永远返回空列表。
    """

    wanted = set(codes)

    def handler(ctx: RuleContext) -> list[Violation]:
        """从上下文里筛出属于本规则原因码的硬约束违反并包装成命中。

        参数：
            ctx: 规则上下文（只读 `constraint_violations`）。

        返回：
            命中列表（可能为空）；每条沿用硬约束给出的 detail 与 product_ids。
        副作用/异常：无。
        """
        hits: list[Violation] = []
        for violation in ctx.constraint_violations:
            if violation.code in wanted:
                hits.append(
                    _violation(rule, violation.detail, violation.product_ids, basis=rule.basis)
                )
        return hits

    return handler


# ---------------------------------------------------------------------------
# 规则实现
# ---------------------------------------------------------------------------
def _elderly_handler(ctx: RuleContext) -> list[Violation]:
    """高龄客户特别保护：禁止 R4 及以上产品，并对集中度进一步压降。

    对应规则 `S-ELDERLY`（block，repair=`elderly_protect`）——这是本库中
    唯一需要"两条判据合并到一条规则"的 handler：

    参数：
        ctx: 规则上下文（读客户年龄/是否高龄、持仓产品、权重）。

    返回：
        命中列表，可能同时包含两类命中：
        1. 高龄客户持有 `risk_level >= ELDERLY_RISK_LIMIT`（即 R4 及以上）的产品；
        2. 单一产品权重超过 `min(客户上限, ELDERLY_SINGLE_PRODUCT_CAP)` 的产品。

    副作用/异常：
        无副作用；客户非高龄时直接返回空列表（不产生任何命中）。

    命中后如何收紧（repair=`elderly_protect`）：
        `engine.build_tighten` 会下发 `risk_cap=3` + `max_single_product_ratio=0.20`
        的收紧指令，打回重配；注意它同时收紧等级与集中度两项。
    """
    client = ctx.client
    if not client.is_elderly:
        return []
    # 保护上限取"客户约定上限"与"高龄保护档位 20%"中更严的一个
    cap = min(client.max_single_product_ratio, ELDERLY_SINGLE_PRODUCT_CAP)
    hits: list[Violation] = []
    for product in ctx.held_products():
        if product.risk_level >= ELDERLY_RISK_LIMIT:
            hits.append(
                _violation(
                    RULE_ELDERLY,
                    f"客户年龄 {client.age} 岁属于高龄客户，不得配置 R{product.risk_level} 产品「{product.name}」",
                    [product.product_id],
                )
            )
        weight = ctx.weight_of(product.product_id)
        # 1e-9 容差：浮点误差不算越界（与四处比较口径一致）
        if weight > cap + 1e-9:
            hits.append(
                _violation(
                    RULE_ELDERLY,
                    f"高龄客户单一产品集中度 {pct(weight)} 超过保护上限 {pct(cap)}（产品「{product.name}」）",
                    [product.product_id],
                )
            )
    return hits


def _dual_record_handler(ctx: RuleContext) -> list[Violation]:
    """双录留痕：高风险产品、衍生品结构、高龄客户等情形必须留痕。

    对应规则 `S-DUAL-RECORD`（**warn**，无 repair）：只提示、不拦截，
    因此无论是否已完成双录都不会产生 block 命中，只是说明文字不同。

    参数：
        ctx: 规则上下文（读客户年龄、双录完成标记、持仓产品的风险等级与衍生品标记）。

    返回：
        空列表（未触发任何情形）或**单条** warn 命中；两种文案分别对应
        "已完成双录（已留痕）"与"未完成双录（定稿前必须补录）"。

    副作用/异常：无副作用；触发情形为高龄客户、持有 R4 及以上产品、
    持有含衍生品结构的产品三者之一。
    """
    client = ctx.client
    triggers: list[str] = []
    if client.is_elderly:
        triggers.append("客户为高龄客户")
    risky = [p.name for p in ctx.held_products() if p.risk_level >= 4]
    if risky:
        triggers.append(f"配置了 R4 及以上产品（{'、'.join(sorted(risky))}）")
    derivative = [p.name for p in ctx.held_products() if p.is_derivative]
    if derivative:
        triggers.append(f"配置了含衍生品结构的产品（{'、'.join(sorted(derivative))}）")
    if not triggers:
        return []
    # 已留痕 → 提示型文案（告知双录要求已被满足）
    if client.dual_record_completed:
        return [
            _violation(
                RULE_DUAL_RECORD,
                f"已触发双录留痕要求（{'；'.join(triggers)}），系统中已留存双录记录标识",
                severity="warn",
            )
        ]
    # 未留痕 → 提示型文案（定稿前必须补录），注意仍是 warn，不拦截流程
    return [
        _violation(
            RULE_DUAL_RECORD,
            f"已触发双录留痕要求（{'；'.join(triggers)}），但客户档案显示尚未完成双录，定稿前必须补录",
            severity="warn",
        )
    ]


def _fee_handler(ctx: RuleContext) -> list[Violation]:
    """费率揭示：组合综合费率不得超过客户费率预算上限。

    对应规则 `S-FEE-DISCLOSE`（**warn**，无 repair）：超出预算不是违规，
    但必须在建议书中显著揭示并取得客户确认。

    参数：
        ctx: 规则上下文（读组合指标与持仓产品费率、客户费率预算）。

    返回：
        空列表（未超预算，容差 1e-9）或单条 warn 命中（含综合费率与预算对比）。

    副作用/异常：
        无副作用。实现上优先取组合指标里的 `expected_fee_rate`；
        指标缺失（为 None）时**自行按权重加权求和**产品费率作为兜底口径。
    """
    client = ctx.client
    total_fee = ctx.portfolio.metrics.get("expected_fee_rate")
    if total_fee is None:
        # 兜底口径：Σ(权重 × 产品费率)；与求解器写入指标的口径保持一致
        total_fee = sum(ctx.weight_of(p.product_id) * p.fee_rate for p in ctx.held_products())
    if total_fee > client.annual_fee_budget_ratio + 1e-9:
        return [
            _violation(
                RULE_FEE_DISCLOSE,
                f"组合综合费率 {pct(total_fee, 3)} 超过客户费率预算 {pct(client.annual_fee_budget_ratio, 3)}，"
                "须在建议书中显著揭示并取得客户确认",
                severity="warn",
            )
        ]
    return []


def _tax_handler(ctx: RuleContext) -> list[Violation]:
    """税收优惠额度：税收优惠型产品配置金额不得超过客户可用额度。

    对应规则 `S-TAX`（**warn**，无 repair）：超出额度的部分只是无法享受税收优惠，
    因此提示而不拦截。

    参数：
        ctx: 规则上下文（读持仓产品的 `tax_advantaged` 标记、权重与客户税额上限）。

    返回：
        空列表（未配置税优产品，或未超额）或单条 warn 命中
        （含实际配置金额与可用额度的对比）。

    副作用/异常：
        无副作用。金额口径为 `Σ(权重 × 客户可投金额)`；
        超额判断容差为 1e-6（比权重口径更松，避免分位舍入误报）。
    """
    client = ctx.client
    total = 0.0
    product_ids: list[str] = []
    for product in ctx.held_products():
        if product.tax_advantaged:
            total += ctx.weight_of(product.product_id) * client.investable_amount
            product_ids.append(product.product_id)
    if product_ids and total > client.tax_advantaged_quota + 1e-6:
        return [
            _violation(
                RULE_TAX,
                f"税收优惠型产品配置金额 {total:,.0f} 元超过客户可用税收优惠额度 "
                f"{client.tax_advantaged_quota:,.0f} 元，超出部分无法享受税收优惠",
                product_ids,
                severity="warn",
            )
        ]
    return []


def _cooling_handler(ctx: RuleContext) -> list[Violation]:
    """冷静期提示：存在封闭期/低流动性且期限较长的产品时须提示冷静期安排。

    对应规则 `S-COOLING`（**warn**，无 repair）：这是销售适当性上的告知义务，
    不是准入条件，故不做拦截。

    参数：
        ctx: 规则上下文（读持仓产品的期限与可即时变现比例）。

    返回：
        命中列表，**一只产品至多一条**：期限 >= `COOLING_HORIZON_YEARS`（1 年）
        且流动性 < `COOLING_LIQUIDITY`（30%）时生成一条 warn 命中；
        全部不满足时返回空列表。

    副作用/异常：无副作用。
    """
    hits: list[Violation] = []
    for product in ctx.held_products():
        if product.horizon_years >= COOLING_HORIZON_YEARS and product.liquidity_ratio < COOLING_LIQUIDITY:
            hits.append(
                _violation(
                    RULE_COOLING,
                    f"产品「{product.name}」期限 {product.horizon_years:g} 年、可即时变现比例 "
                    f"{pct(product.liquidity_ratio)}，须向客户提示封闭期与冷静期安排",
                    [product.product_id],
                    severity="warn",
                )
            )
    return hits


def _concentration_warning_handler(ctx: RuleContext) -> list[Violation]:
    """内控预警线：单一产品集中度超过内部预警线时提示人工复核。

    对应规则 `S-CONCENTRATION-WARNING`（**warn**，无 repair）：预警线严于
    客户约定的集中度上限（block 级由 S-CONC-SINGLE 管），因此这里只提示
    "需要理财经理人工确认"，不拦截。

    参数：
        ctx: 规则上下文（读客户 `warning_ratio` 与各持仓权重）。

    返回：
        命中列表，**超线产品各一条** warn；无持仓或均未超线时返回空列表。
        比较带 1e-9 容差。

    副作用/异常：无副作用。
    """
    client = ctx.client
    threshold = client.warning_ratio
    hits: list[Violation] = []
    for product in ctx.held_products():
        weight = ctx.weight_of(product.product_id)
        if weight > threshold + 1e-9:
            hits.append(
                _violation(
                    RULE_CONCENTRATION_WARNING,
                    f"产品「{product.name}」集中度 {pct(weight)} 超过内控预警线 {pct(threshold)}，"
                    "须理财经理人工确认后方可定稿",
                    [product.product_id],
                    severity="warn",
                )
            )
    return hits


def _cash_reserve_handler(ctx: RuleContext) -> list[Violation]:
    """现金缓冲：建议保留一定比例现金以应对临时流动性需求。

    对应规则 `S-CASH-RESERVE`（**warn**，无 repair）：现金少说明"钱都投出去了"，
    属配置风格问题而非合规问题，因此只提示。

    参数：
        ctx: 规则上下文（只读组合的现金权重）。

    返回：
        空列表（现金权重 >= `CASH_RESERVE_FLOOR` - 1e-9）或单条 warn 命中
        （含实际比例与建议下限）。注意**不区分**"没有持仓"与"全现金"之外的
        情形，只看 `portfolio.cash_weight`。

    副作用/异常：无副作用。
    """
    if ctx.portfolio.cash_weight < CASH_RESERVE_FLOOR - 1e-9:
        return [
            _violation(
                RULE_CASH_RESERVE,
                f"现金及活期留存比例 {pct(ctx.portfolio.cash_weight)} 低于建议下限 {pct(CASH_RESERVE_FLOOR)}",
                severity="warn",
            )
        ]
    return []


def _diversification_handler(ctx: RuleContext) -> list[Violation]:
    """分散度：持仓产品只数过少时提示集中风险。

    对应规则 `S-DIVERSIFICATION`（**warn**，无 repair）：只数少不等于违规
    （例如客户可投金额小、可行域本来就窄），故提示而不拦截。

    参数：
        ctx: 规则上下文（读 `portfolio.held_ids()`）。

    返回：
        空列表；或**仅当** `0 < 持仓只数 < MIN_HOLDINGS`（3）时返回单条 warn 命中
        （附全部持仓产品号，便于排查是哪几只）。
        注：0 只（全现金组合）**不触发**本规则，也不触发上限比较。

    副作用/异常：无副作用。
    """
    held = ctx.portfolio.held_ids()
    if 0 < len(held) < MIN_HOLDINGS:
        return [
            _violation(
                RULE_DIVERSIFICATION,
                f"组合仅持有 {len(held)} 只产品，低于建议分散度下限 {MIN_HOLDINGS} 只",
                held,
                severity="warn",
            )
        ]
    return []


def _feasible_pool_handler(ctx: RuleContext) -> list[Violation]:
    """可行域非空：硬约束下必须至少存在一个可投产品，否则无法给出建议。

    对应规则 `S-FEASIBLE-POOL`（**block + veto=True**，repair 为空）：
    这是全库唯一不可修复的规则——收紧约束只会让可行域更空，
    所以命中即 `reject`（`SuitabilityGate` 会把它算进 `veto_rules`），
    不会产生任何 `TightenSpec`。

    参数：
        ctx: 规则上下文（读 `candidates` 与全量产品池 `universe`）。

    返回：
        空列表（候选池非空）；或候选池为空时返回**单条** block 命中，
        其 detail 会附带前 3 个产品的不可投原因摘要，便于向客户解释
        "为什么一个都买不了"。

    副作用/异常：
        无副作用；内部对全量产品逐个调用 `check_product_admissibility`
        与 `admissibility_detail` 做纯计算（产品很多时会遍历全池）。
    """
    if ctx.candidates:
        return []
    reasons: list[str] = []
    for product in sorted(ctx.universe.values(), key=lambda p: p.product_id):
        codes = check_product_admissibility(product, ctx.client)
        reasons.append(f"{product.name}：{admissibility_detail(product, ctx.client, codes)}")
    preview = "；".join(reasons[:3])
    return [
        _violation(
            RULE_FEASIBLE_POOL,
            f"客户硬约束下不存在任何可投产品，建议无法生成（示例：{preview}）",
            severity="block",
        )
    ]


# ---------------------------------------------------------------------------
# 规则注册表
# ---------------------------------------------------------------------------
# 说明：下面 22 条 Rule 的 handler 分两种来源：
#   (a) 复用硬约束判定 —— 先用 `lambda ctx: []` 占位，待全部 Rule 定义完成后
#       再用 `_from_constraints(...)` 回填 handler（见文件后半段"补上…"区块），
#       以避免 `_from_constraints` 引用尚未定义的全局变量造成前向引用问题；
#   (b) 适当性专有 —— 直接指向本文件中的 `_xxx_handler` 纯函数。
# 每条规则后的注释给出：规则意图 / 级别 / repair（命中后如何收紧）。

# S-RISK-MATCH｜风险等级匹配｜block｜repair=risk_cap_down
#   意图：产品风险等级不得高于客户风险承受等级（对应硬约束原因码 C-RISK）。
#   命中后：等级上限降到"违规产品中最低等级 - 1"（下限 R1），打回重配。
RULE_RISK_MATCH = Rule(
    id="S-RISK-MATCH",
    name="风险等级匹配",
    severity="block",
    category="适当性匹配",
    basis="适当性管理要求：产品风险等级不得高于客户风险承受等级",
    handler=lambda ctx: [],
    repair="risk_cap_down",
    description="禁止向客户推荐风险等级高于其风险承受等级的产品。",
)
# S-HORIZON｜投资期限匹配｜block｜repair=horizon_down
#   意图：产品期限不得超过客户投资期限，防范期限错配（对应 C-HORIZON）。
#   命中后：期限上限降到"违规产品最短期限的一半"（下限 0.1 年），打回重配。
RULE_HORIZON = Rule(
    id="S-HORIZON",
    name="投资期限匹配",
    severity="block",
    category="适当性匹配",
    basis="适当性管理要求：产品期限不得超过客户投资期限，防范期限错配",
    handler=lambda ctx: [],
    repair="horizon_down",
    description="产品期限必须落在客户投资期限之内。",
)
# S-CONC-SINGLE｜单一产品集中度上限｜block｜repair=single_cap_down
#   意图：单一产品权重不得超过客户约定上限（对应 C-CONC-SINGLE）。
#   命中后：该上限 ×0.8（下限 1%），打回重配。
RULE_CONC_SINGLE = Rule(
    id="S-CONC-SINGLE",
    name="单一产品集中度上限",
    severity="block",
    category="集中度管理",
    basis="集中度管理要求：单一产品权重不得超过客户约定上限",
    handler=lambda ctx: [],
    repair="single_cap_down",
    description="单一产品权重不得超过客户档案约定的集中度上限。",
)
# S-CONC-CLASS｜单一资产类别上限｜block｜repair=class_cap_down
#   意图：同一资产类别合计权重不得超过约定上限（对应 C-CONC-CLASS）。
#   命中后：该上限 ×0.8（下限 1%），打回重配。
RULE_CONC_CLASS = Rule(
    id="S-CONC-CLASS",
    name="单一资产类别上限",
    severity="block",
    category="集中度管理",
    basis="集中度管理要求：单一资产类别权重不得超过客户约定上限",
    handler=lambda ctx: [],
    repair="class_cap_down",
    description="同一资产类别合计权重不得超过约定上限。",
)
# S-CONCENTRATION-ISSUER｜同一发行人集中度上限｜block｜repair=issuer_cap_down
#   意图：同一发行主体下产品合计权重不得超过约定上限（对应 C-CONC-ISSUER）。
#   命中后：该上限 ×0.8（下限 1%），打回重配。
RULE_CONC_ISSUER = Rule(
    id="S-CONCENTRATION-ISSUER",
    name="同一发行人集中度上限",
    severity="block",
    category="集中度管理",
    basis="集中度管理要求：同一发行主体合计权重不得超过约定上限",
    handler=lambda ctx: [],
    repair="issuer_cap_down",
    description="同一发行主体下的产品合计权重不得超过约定上限。",
)
# S-LIQUIDITY｜流动性需求满足｜block｜repair=liquidity_up
#   意图：组合流动性资产占比不得低于客户流动性需求下限（对应 C-LIQUIDITY）。
#   命中后：流动性下限每轮 +5 个百分点（TightenSpec.apply 会截到 100%），打回重配。
RULE_LIQUIDITY = Rule(
    id="S-LIQUIDITY",
    name="流动性需求满足",
    severity="block",
    category="流动性管理",
    basis="适当性管理要求：组合流动性资产占比不得低于客户流动性需求下限",
    handler=lambda ctx: [],
    repair="liquidity_up",
    description="流动性资产占比必须覆盖客户约定的流动性下限。",
)
# S-ENTRY｜起投金额匹配｜block｜repair=exclude_products
#   意图：起投金额 > 可投金额，或配置金额 < 起投金额时不得纳入组合
#         （对应 C-ENTRY 与 C-ENTRY-MIN 两个原因码）。
#   命中后：把涉及的违规产品加入客户禁止清单，打回重配。
RULE_ENTRY = Rule(
    id="S-ENTRY",
    name="起投金额匹配",
    severity="block",
    category="交易可行性",
    basis="交易可行性要求：产品起投金额与客户可投金额、配置金额必须匹配",
    handler=lambda ctx: [],
    repair="exclude_products",
    description="起投金额超过可投金额，或配置金额低于起投金额时不得纳入组合。",
)
# S-ELDERLY｜高龄客户特别保护｜block｜repair=elderly_protect
#   意图：高龄客户执行更审慎的风险等级与集中度限制（专有规则，无硬约束原因码）。
#   命中后：下发 risk_cap=R3 + 单一产品上限 20%（两项同时收紧），打回重配。
RULE_ELDERLY = Rule(
    id="S-ELDERLY",
    name="高龄客户特别保护",
    severity="block",
    category="投资者保护",
    basis="投资者保护要求：对高龄客户应执行更审慎的风险等级与集中度限制",
    handler=_elderly_handler,
    repair="elderly_protect",
    description=f"高龄客户不得持有 R{ELDERLY_RISK_LIMIT} 及以上产品，单一产品上限压降至 {ELDERLY_SINGLE_PRODUCT_CAP:.0%}。",
)
# S-EXPERIENCE｜投资经验匹配｜block｜repair=exclude_products
#   意图：不具备相关投资经验的品类不得推荐（对应 C-EXPERIENCE）。
#   命中后：把该产品加入禁止清单，打回重配。
RULE_EXPERIENCE = Rule(
    id="S-EXPERIENCE",
    name="投资经验匹配",
    severity="block",
    category="适当性匹配",
    basis="适当性管理要求：不具备相关投资经验的品类不得推荐",
    handler=lambda ctx: [],
    repair="exclude_products",
    description="客户须具备产品所要求品类的投资经验。",
)
# S-DERIVATIVE-BAN｜衍生品禁止项｜block｜repair=exclude_categories
#   意图：客户约定排除衍生品时，含衍生品结构的产品不得进入组合（对应 C-DERIVATIVE-BAN）。
#   命中后：**产品 + 其所属资产类别**一并排除（防同类换只再犯），打回重配。
RULE_DERIVATIVE_BAN = Rule(
    id="S-DERIVATIVE-BAN",
    name="衍生品禁止项",
    severity="block",
    category="禁止项",
    basis="客户约定禁止项：不得推荐含衍生品结构的资产",
    handler=lambda ctx: [],
    repair="exclude_categories",
    description="客户明确排除衍生品时，任何含衍生品结构的产品都不得进入组合。",
)
# S-CURRENCY｜币种匹配｜block｜repair=exclude_products
#   意图：产品币种须与客户币种偏好一致（对应 C-CURRENCY）。
#   命中后：把该产品加入禁止清单，打回重配。
RULE_CURRENCY = Rule(
    id="S-CURRENCY",
    name="币种匹配",
    severity="block",
    category="适当性匹配",
    basis="币种偏好约束：产品币种需与客户偏好币种一致",
    handler=lambda ctx: [],
    repair="exclude_products",
    description="产品币种须与客户币种偏好一致。",
)
# S-QUALIFIED｜合格投资者准入｜block｜repair=exclude_products
#   意图：仅面向合格投资者的产品不得销售给非合格投资者（对应 C-QUALIFIED）。
#   命中后：把该产品加入禁止清单，打回重配。
RULE_QUALIFIED = Rule(
    id="S-QUALIFIED",
    name="合格投资者准入",
    severity="block",
    category="准入管理",
    basis="准入管理要求：仅面向合格投资者的产品不得销售给非合格投资者",
    handler=lambda ctx: [],
    repair="exclude_products",
    description="合格投资者专属产品需要客户满足准入资格。",
)
# S-PROHIBITED｜禁止项清单｜block｜repair=exclude_products
#   意图：客户禁止清单内的类别或具体产品不得配置
#         （对应 C-PROHIBITED-CLASS 与 C-PROHIBITED-PRODUCT 两个原因码）。
#   命中后：把涉及的违规产品加入禁止清单，打回重配。
RULE_PROHIBITED = Rule(
    id="S-PROHIBITED",
    name="禁止项清单",
    severity="block",
    category="禁止项",
    basis="客户约定禁止项：禁止清单内的类别与产品不得纳入组合",
    handler=lambda ctx: [],
    repair="exclude_products",
    description="客户禁止清单（类别或具体产品）内的标的不得配置。",
)
# S-WEIGHT｜权重合法性｜block｜repair=normalize
#   意图：组合权重必须合法——合计 ≤ 100% 且不允许负权重
#         （对应 C-WEIGHT-SUM 与 C-WEIGHT-NEG）。
#   命中后：repair 提示为 `normalize`，但注：实际实现为——`engine.build_tighten`
#   没有 `normalize` 分支（落入 else），因此**不产生任何收紧动作**；
#   权重非法应由求解器的归一化/可行性修复负责，闸门只负责拦下并留痕。
RULE_WEIGHT = Rule(
    id="S-WEIGHT",
    name="权重合法性",
    severity="block",
    category="组合合法性",
    basis="组合合法性要求：权重合计不得超过 100%，且不允许负权重",
    handler=lambda ctx: [],
    repair="normalize",
    description="组合权重必须合法（合计 ≤ 100%、非负）。",
)
# S-FEASIBLE-POOL｜可行域非空｜block + veto｜repair=""（无修复策略）
#   意图：客户硬约束下至少要有一个可投产品；一个都没有时无法给出建议。
#   命中后：**不可修复**——`veto=True`，闸门直接 reject（不走打回重配），
#   也不产生 TightenSpec（收紧只会让可行域更空）。
RULE_FEASIBLE_POOL = Rule(
    id="S-FEASIBLE-POOL",
    name="可行域非空",
    severity="block",
    category="准入管理",
    basis="适当性管理要求：无法在客户约束下构建任何组合时应拒绝出具建议",
    handler=_feasible_pool_handler,
    repair="",
    veto=True,
    description="硬约束下可行域为空时直接拒绝，不允许通过收紧约束修复。",
)
# S-DUAL-RECORD｜双录留痕｜warn（以下 7 条均为 warn，无 repair，不拦截）
#   意图：高龄 / R4 及以上 / 衍生品结构等情形触发录音录像留痕要求。
#   命中后仅提示：未完成双录的提示"定稿前必须补录"，不产生收紧动作。
RULE_DUAL_RECORD = Rule(
    id="S-DUAL-RECORD",
    name="双录留痕",
    severity="warn",
    category="销售留痕",
    basis="销售留痕要求：特定情形下须完成录音录像并留存",
    handler=_dual_record_handler,
    description="高风险产品、衍生品结构、高龄客户等情形触发双录留痕要求。",
)
# S-FEE-DISCLOSE｜费率揭示｜warn
#   意图：组合综合费率超过客户费率预算时须显著揭示（不是禁止）。
#   命中后仅提示：建议书必须披露并取得客户确认。
RULE_FEE_DISCLOSE = Rule(
    id="S-FEE-DISCLOSE",
    name="费率揭示",
    severity="warn",
    category="信息披露",
    basis="信息披露要求：应向客户充分揭示产品费率与综合成本",
    handler=_fee_handler,
    description="组合综合费率超客户预算时须显著揭示。",
)
# S-TAX｜税收优惠额度｜warn
#   意图：税收优惠型产品配置金额超过客户可用额度时，超出部分不享受优惠。
#   命中后仅提示：告知超出额度，不拦截。
RULE_TAX = Rule(
    id="S-TAX",
    name="税收优惠额度",
    severity="warn",
    category="信息披露",
    basis="税收政策口径：税收优惠型产品存在额度上限，超出部分不享受优惠",
    handler=_tax_handler,
    description="税收优惠型产品配置金额不得超过客户可用额度。",
)
# S-COOLING｜冷静期提示｜warn
#   意图：期限 >= 1 年且可即时变现比例 < 30% 的产品，须提示封闭期与冷静期安排。
#   命中后仅提示：逐产品揭示赎回限制。
RULE_COOLING = Rule(
    id="S-COOLING",
    name="冷静期提示",
    severity="warn",
    category="投资者保护",
    basis="投资者保护要求：封闭期产品应提示冷静期与赎回限制",
    handler=_cooling_handler,
    description="封闭期/低流动性产品须提示冷静期安排。",
)
# S-CONCENTRATION-WARNING｜内控集中度预警线｜warn
#   意图：单一产品集中度超过客户内控预警线（严于约定上限）时须人工复核。
#   命中后仅提示：转理财经理人工确认，不拦截（block 级由 S-CONC-SINGLE 负责）。
RULE_CONCENTRATION_WARNING = Rule(
    id="S-CONCENTRATION-WARNING",
    name="内控集中度预警线",
    severity="warn",
    category="集中度管理",
    basis="内控管理要求：单一产品集中度超过内部预警线时须人工复核",
    handler=_concentration_warning_handler,
    description="集中度超过内控预警线时转人工确认。",
)
# S-CASH-RESERVE｜现金缓冲建议｜warn
#   意图：现金及活期留存低于建议下限（3%）时提示保留缓冲。
#   命中后仅提示：属配置风格建议，不影响放行。
RULE_CASH_RESERVE = Rule(
    id="S-CASH-RESERVE",
    name="现金缓冲建议",
    severity="warn",
    category="流动性管理",
    basis="流动性管理建议：组合宜保留一定比例的现金或活期留存",
    handler=_cash_reserve_handler,
    description="现金缓冲低于建议下限时提示。",
)
# S-DIVERSIFICATION｜分散度建议｜warn
#   意图：持仓只数少于建议下限（3 只，且非全现金）时提示集中风险。
#   命中后仅提示：建议增加分散度，不做强制。
RULE_DIVERSIFICATION = Rule(
    id="S-DIVERSIFICATION",
    name="分散度建议",
    severity="warn",
    category="组合管理",
    basis="组合管理建议：持仓产品只数过少时集中风险偏高",
    handler=_diversification_handler,
    description="持有产品只数低于建议下限时提示。",
)

# 补上「复用硬约束判定」的规则 handler（在 Rule 定义之后再绑定，避免前向引用）
# 注：这里用 `Rule(**{**RULE.__dict__, "handler": ...})` 重建**新的** frozen 实例
# （frozen dataclass 不能直接改字段），只替换 handler，其余字段原样继承；
# 下方 13 次重绑定对应的原因码即上表 CONSTRAINT_TO_RULE 的取值。
RULE_RISK_MATCH = Rule(**{**RULE_RISK_MATCH.__dict__, "handler": _from_constraints(RULE_RISK_MATCH, ["C-RISK"])})
RULE_HORIZON = Rule(**{**RULE_HORIZON.__dict__, "handler": _from_constraints(RULE_HORIZON, ["C-HORIZON"])})
RULE_CONC_SINGLE = Rule(**{**RULE_CONC_SINGLE.__dict__, "handler": _from_constraints(RULE_CONC_SINGLE, ["C-CONC-SINGLE"])})
RULE_CONC_CLASS = Rule(**{**RULE_CONC_CLASS.__dict__, "handler": _from_constraints(RULE_CONC_CLASS, ["C-CONC-CLASS"])})
RULE_CONC_ISSUER = Rule(
    **{**RULE_CONC_ISSUER.__dict__, "handler": _from_constraints(RULE_CONC_ISSUER, ["C-CONC-ISSUER"])}
)
RULE_LIQUIDITY = Rule(**{**RULE_LIQUIDITY.__dict__, "handler": _from_constraints(RULE_LIQUIDITY, ["C-LIQUIDITY"])})
RULE_ENTRY = Rule(**{**RULE_ENTRY.__dict__, "handler": _from_constraints(RULE_ENTRY, ["C-ENTRY", "C-ENTRY-MIN"])})
RULE_EXPERIENCE = Rule(**{**RULE_EXPERIENCE.__dict__, "handler": _from_constraints(RULE_EXPERIENCE, ["C-EXPERIENCE"])})
RULE_DERIVATIVE_BAN = Rule(
    **{**RULE_DERIVATIVE_BAN.__dict__, "handler": _from_constraints(RULE_DERIVATIVE_BAN, ["C-DERIVATIVE-BAN"])}
)
RULE_CURRENCY = Rule(**{**RULE_CURRENCY.__dict__, "handler": _from_constraints(RULE_CURRENCY, ["C-CURRENCY"])})
RULE_QUALIFIED = Rule(**{**RULE_QUALIFIED.__dict__, "handler": _from_constraints(RULE_QUALIFIED, ["C-QUALIFIED"])})
RULE_PROHIBITED = Rule(
    **{
        **RULE_PROHIBITED.__dict__,
        "handler": _from_constraints(RULE_PROHIBITED, ["C-PROHIBITED-CLASS", "C-PROHIBITED-PRODUCT"]),
    }
)
RULE_WEIGHT = Rule(**{**RULE_WEIGHT.__dict__, "handler": _from_constraints(RULE_WEIGHT, ["C-WEIGHT-SUM", "C-WEIGHT-NEG"])})

#: 全量规则表（顺序即评估顺序，保证输出稳定）
#: 共 22 条：前 15 条为 block 级（其中最后一条 S-FEASIBLE-POOL 带 veto），
#: 后 7 条为 warn 级。引擎按本元组顺序求值并据此排序命中，因此**追加新规则只应
#: 加在末尾**，改动中间顺序会改变 `evaluate_rules` 的输出顺序（影响留痕与快照 diff）。
ALL_RULES: tuple[Rule, ...] = (
    RULE_RISK_MATCH,
    RULE_HORIZON,
    RULE_CONC_SINGLE,
    RULE_CONC_CLASS,
    RULE_CONC_ISSUER,
    RULE_LIQUIDITY,
    RULE_ENTRY,
    RULE_ELDERLY,
    RULE_EXPERIENCE,
    RULE_DERIVATIVE_BAN,
    RULE_CURRENCY,
    RULE_QUALIFIED,
    RULE_PROHIBITED,
    RULE_WEIGHT,
    RULE_FEASIBLE_POOL,
    RULE_DUAL_RECORD,
    RULE_FEE_DISCLOSE,
    RULE_TAX,
    RULE_COOLING,
    RULE_CONCENTRATION_WARNING,
    RULE_CASH_RESERVE,
    RULE_DIVERSIFICATION,
)

#: 规则号 -> 规则对象（`build_tighten` 靠它把命中反查回 `repair` 提示）
RULES_BY_ID: dict[str, Rule] = {rule.id: rule for rule in ALL_RULES}


def get_rule(rule_id: str) -> Rule:
    """按规则号取规则；不存在时抛 KeyError。

    参数：
        rule_id: 规则号（如 `"S-ELDERLY"`）。
    返回：对应的 `Rule` 对象。
    副作用/异常：无副作用；规则号不存在时抛 KeyError（不返回 None，
        避免调用方把"写错规则号"静默当成"没有这条规则"）。
    """
    return RULES_BY_ID[rule_id]


def rule_catalog() -> list[dict[str, str]]:
    """规则清单（demo / README / 评估报告使用）。

    参数：无。
    返回：按 `ALL_RULES` 顺序排列的字典列表，每项含
        `id` / `name` / `severity` / `category` / `basis`
        （刻意不含 handler 与 repair，保证可 JSON 序列化）。
    副作用/异常：无。
    """
    return [
        {
            "id": rule.id,
            "name": rule.name,
            "severity": rule.severity,
            "category": rule.category,
            "basis": rule.basis,
        }
        for rule in ALL_RULES
    ]


def has_block_rule(rule_id: str) -> bool:
    """该规则是否为 block 级。

    参数：
        rule_id: 规则号。
    返回：规则存在且 `severity == "block"` 时为 True，否则 False
        （未知规则号一律返回 False）。
    副作用/异常：无副作用；不抛异常，便于调用方直接用于条件判断。
    """
    return rule_id in RULES_BY_ID and RULES_BY_ID[rule_id].severity == "block"
