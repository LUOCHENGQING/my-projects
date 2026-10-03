"""适当性规则库（Deterministic Suitability Rule Library）。

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
    """规则求值上下文：一次求值的全部输入都是冻结的。"""

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
        """构造上下文，并复用硬约束求解器的判定结果。"""
        violations = tuple(check_portfolio(portfolio, client))
        return cls(
            client=client,
            portfolio=portfolio,
            universe=universe,
            candidates=tuple(sorted(candidates)),
            constraint_violations=violations,
        )

    def held_products(self) -> list[Product]:
        """持有产品（按产品号排序）。"""
        return [self.portfolio.products[pid] for pid in self.portfolio.held_ids() if pid in self.portfolio.products]

    def weight_of(self, product_id: str) -> float:
        """某产品权重。"""
        return self.portfolio.weights.get(product_id, 0.0)


@dataclass(frozen=True)
class Rule:
    """一条适当性规则。"""

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
    """按规则定义生成命中记录。"""
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
    """把一个或多个硬约束原因码映射成该规则的命中列表。"""

    wanted = set(codes)

    def handler(ctx: RuleContext) -> list[Violation]:
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
    """高龄客户特别保护：禁止 R4 及以上产品，并对集中度进一步压降。"""
    client = ctx.client
    if not client.is_elderly:
        return []
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
    """双录留痕：高风险产品、衍生品结构、高龄客户等情形必须留痕。"""
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
    if client.dual_record_completed:
        return [
            _violation(
                RULE_DUAL_RECORD,
                f"已触发双录留痕要求（{'；'.join(triggers)}），系统中已留存双录记录标识",
                severity="warn",
            )
        ]
    return [
        _violation(
            RULE_DUAL_RECORD,
            f"已触发双录留痕要求（{'；'.join(triggers)}），但客户档案显示尚未完成双录，定稿前必须补录",
            severity="warn",
        )
    ]


def _fee_handler(ctx: RuleContext) -> list[Violation]:
    """费率揭示：组合综合费率不得超过客户费率预算上限。"""
    client = ctx.client
    total_fee = ctx.portfolio.metrics.get("expected_fee_rate")
    if total_fee is None:
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
    """税收优惠额度：税收优惠型产品配置金额不得超过客户可用额度。"""
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
    """冷静期提示：存在封闭期/低流动性且期限较长的产品时须提示冷静期安排。"""
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
    """内控预警线：单一产品集中度超过内部预警线时提示人工复核。"""
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
    """现金缓冲：建议保留一定比例现金以应对临时流动性需求。"""
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
    """分散度：持仓产品只数过少时提示集中风险。"""
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
    """可行域非空：硬约束下必须至少存在一个可投产品，否则无法给出建议。"""
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
RULE_DUAL_RECORD = Rule(
    id="S-DUAL-RECORD",
    name="双录留痕",
    severity="warn",
    category="销售留痕",
    basis="销售留痕要求：特定情形下须完成录音录像并留存",
    handler=_dual_record_handler,
    description="高风险产品、衍生品结构、高龄客户等情形触发双录留痕要求。",
)
RULE_FEE_DISCLOSE = Rule(
    id="S-FEE-DISCLOSE",
    name="费率揭示",
    severity="warn",
    category="信息披露",
    basis="信息披露要求：应向客户充分揭示产品费率与综合成本",
    handler=_fee_handler,
    description="组合综合费率超客户预算时须显著揭示。",
)
RULE_TAX = Rule(
    id="S-TAX",
    name="税收优惠额度",
    severity="warn",
    category="信息披露",
    basis="税收政策口径：税收优惠型产品存在额度上限，超出部分不享受优惠",
    handler=_tax_handler,
    description="税收优惠型产品配置金额不得超过客户可用额度。",
)
RULE_COOLING = Rule(
    id="S-COOLING",
    name="冷静期提示",
    severity="warn",
    category="投资者保护",
    basis="投资者保护要求：封闭期产品应提示冷静期与赎回限制",
    handler=_cooling_handler,
    description="封闭期/低流动性产品须提示冷静期安排。",
)
RULE_CONCENTRATION_WARNING = Rule(
    id="S-CONCENTRATION-WARNING",
    name="内控集中度预警线",
    severity="warn",
    category="集中度管理",
    basis="内控管理要求：单一产品集中度超过内部预警线时须人工复核",
    handler=_concentration_warning_handler,
    description="集中度超过内控预警线时转人工确认。",
)
RULE_CASH_RESERVE = Rule(
    id="S-CASH-RESERVE",
    name="现金缓冲建议",
    severity="warn",
    category="流动性管理",
    basis="流动性管理建议：组合宜保留一定比例的现金或活期留存",
    handler=_cash_reserve_handler,
    description="现金缓冲低于建议下限时提示。",
)
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

RULES_BY_ID: dict[str, Rule] = {rule.id: rule for rule in ALL_RULES}


def get_rule(rule_id: str) -> Rule:
    """按规则号取规则；不存在时抛 KeyError。"""
    return RULES_BY_ID[rule_id]


def rule_catalog() -> list[dict[str, str]]:
    """规则清单（demo / README / 评估报告使用）。"""
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
    """该规则是否为 block 级。"""
    return rule_id in RULES_BY_ID and RULES_BY_ID[rule_id].severity == "block"
