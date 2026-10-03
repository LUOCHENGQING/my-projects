"""硬约束求解器（确定性，绝不交给模型）。

定位
----
本模块是「约束驱动」路线的地基：把客户档案里的**硬约束**翻译成可执行的
纯函数检查，先在可行域上做求解，再让 Agent 在可行域内做多目标权衡。

三条不可动摇的性质（均有单测覆盖）
----------------------------------
1. **纯函数**：只读输入，不修改传入对象，不依赖随机数、时间、全局状态。
2. **确定性**：同样的输入永远返回同样的输出（连顺序都一致）。
3. **幂等**：`f(f(x)) == f(x)`；重复调用不会累积副作用。

关键接口
--------
- `check_product_admissibility(product, client)` -> 单体产品不可投原因码列表
- `check_portfolio(portfolio, client)` -> 组合层面的违反项列表
- `screen_products(client, products)` -> 候选池 + 剔除原因
- `TightenSpec(...).apply(client)` -> 按闸门指令收紧客户约束（打回重配用）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .schemas import ClientProfile, Exclusion, Portfolio, Product, ScreeningResult, Violation
from .utils import RATIO_TOL, WEIGHT_TOL, gt, le

# ---------------------------------------------------------------------------
# 约束字典：原因码 -> 依据说明（监管口径统一使用泛称）
# ---------------------------------------------------------------------------
CONSTRAINT_DEFS: dict[str, str] = {
    "C-RISK": "适当性管理要求：产品风险等级不得高于客户风险承受等级",
    "C-HORIZON": "适当性管理要求：产品期限不得超过客户投资期限，避免期限错配",
    "C-PROHIBITED-CLASS": "客户明确排除的资产类别，属于不可突破的禁止项",
    "C-PROHIBITED-PRODUCT": "客户明确排除的具体产品，属于不可突破的禁止项",
    "C-DERIVATIVE-BAN": "客户禁止项：不得推荐任何含衍生品结构的资产",
    "C-CURRENCY": "币种偏好约束：产品币种需与客户偏好币种一致",
    "C-EXPERIENCE": "适当性管理要求：客户不具备相关投资经验的品类不得推荐",
    "C-ENTRY": "起投金额超过客户可投金额，无法完成配置",
    "C-ENTRY-MIN": "该产品配置金额低于其起投金额，不满足成立条件",
    "C-QUALIFIED": "该产品仅面向合格投资者，客户不满足准入资格",
    "C-CONC-SINGLE": "单一产品集中度上限约束",
    "C-CONC-CLASS": "单一资产类别集中度上限约束",
    "C-CONC-ISSUER": "同一发行人集中度上限约束",
    "C-LIQUIDITY": "流动性需求约束：流动性资产占比不得低于客户下限",
    "C-WEIGHT-SUM": "组合权重合计不得超过 100%",
    "C-WEIGHT-NEG": "组合中不允许出现负权重（不做空）",
}

# 集中度类原因码：用于候选池「预言机」交叉校验时排除组合层面的干扰
CONCENTRATION_CODES: frozenset[str] = frozenset({"C-CONC-SINGLE", "C-CONC-CLASS", "C-CONC-ISSUER"})

# 产品层面的准入原因码（只在候选池阶段判定）
PRODUCT_LEVEL_CODES: tuple[str, ...] = (
    "C-RISK",
    "C-HORIZON",
    "C-PROHIBITED-CLASS",
    "C-PROHIBITED-PRODUCT",
    "C-DERIVATIVE-BAN",
    "C-CURRENCY",
    "C-EXPERIENCE",
    "C-ENTRY",
    "C-QUALIFIED",
)


# ---------------------------------------------------------------------------
# 约束收紧指令
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TightenSpec:
    """适当性闸门下发的「约束收紧」指令。

    语义统一为**单调收紧**：所有数值型约束只会取更严格的一侧
    （等级取更小、期限取更短、上限取更低、流动性下限取更高），
    因此多轮打回一定是收敛的，不会来回震荡。
    """

    risk_cap: int | None = None
    horizon_years: float | None = None
    max_single_product_ratio: float | None = None
    max_single_class_ratio: float | None = None
    max_single_issuer_ratio: float | None = None
    liquidity_floor_ratio: float | None = None
    excluded_product_ids: tuple[str, ...] = ()
    excluded_categories: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def apply(self, client: ClientProfile) -> ClientProfile:
        """返回收紧后的客户约束副本（原对象不被修改）。"""
        updates: dict[str, Any] = {}

        if self.risk_cap is not None:
            updates["risk_capacity"] = min(client.risk_capacity, int(self.risk_cap))
        if self.horizon_years is not None:
            updates["investment_horizon_years"] = min(client.investment_horizon_years, float(self.horizon_years))
        if self.max_single_product_ratio is not None:
            updates["max_single_product_ratio"] = min(
                client.max_single_product_ratio, float(self.max_single_product_ratio)
            )
        if self.max_single_class_ratio is not None:
            updates["max_single_class_ratio"] = min(
                client.max_single_class_ratio, float(self.max_single_class_ratio)
            )
        if self.max_single_issuer_ratio is not None:
            updates["max_single_issuer_ratio"] = min(
                client.max_single_issuer_ratio, float(self.max_single_issuer_ratio)
            )
        if self.liquidity_floor_ratio is not None:
            updates["liquidity_floor_ratio"] = max(
                client.liquidity_floor_ratio, min(1.0, float(self.liquidity_floor_ratio))
            )
        if self.excluded_categories:
            merged = sorted(set(client.prohibited_categories) | set(self.excluded_categories))
            updates["prohibited_categories"] = merged
        if self.excluded_product_ids:
            merged_ids = sorted(set(client.prohibited_product_ids) | set(self.excluded_product_ids))
            updates["prohibited_product_ids"] = merged_ids

        if not updates:
            return client.model_copy(deep=True)
        return client.model_copy(update=updates, deep=True)

    def merge(self, other: "TightenSpec") -> "TightenSpec":
        """按单调收紧语义合并两条指令（数值取更严格一侧，禁止项取并集）。"""
        def _tighten_min(a: float | None, b: float | None) -> float | None:
            if a is None:
                return b
            if b is None:
                return a
            return min(a, b)

        def _tighten_max(a: float | None, b: float | None) -> float | None:
            if a is None:
                return b
            if b is None:
                return a
            return max(a, b)

        risk_cap = _tighten_min(
            None if self.risk_cap is None else float(self.risk_cap),
            None if other.risk_cap is None else float(other.risk_cap),
        )
        return TightenSpec(
            risk_cap=None if risk_cap is None else int(risk_cap),
            horizon_years=_tighten_min(self.horizon_years, other.horizon_years),
            max_single_product_ratio=_tighten_min(self.max_single_product_ratio, other.max_single_product_ratio),
            max_single_class_ratio=_tighten_min(self.max_single_class_ratio, other.max_single_class_ratio),
            max_single_issuer_ratio=_tighten_min(self.max_single_issuer_ratio, other.max_single_issuer_ratio),
            liquidity_floor_ratio=_tighten_max(self.liquidity_floor_ratio, other.liquidity_floor_ratio),
            excluded_product_ids=tuple(sorted(set(self.excluded_product_ids) | set(other.excluded_product_ids))),
            excluded_categories=tuple(sorted(set(self.excluded_categories) | set(other.excluded_categories))),
            reasons=tuple(dict.fromkeys(self.reasons + other.reasons)),
        )

    def is_empty(self) -> bool:
        """是否为空指令（未做任何收紧）。"""
        return self == TightenSpec()

    def to_dict(self) -> dict[str, Any]:
        """转成可写入 trace / state 的普通字典。"""
        return {
            "risk_cap": self.risk_cap,
            "horizon_years": self.horizon_years,
            "max_single_product_ratio": self.max_single_product_ratio,
            "max_single_class_ratio": self.max_single_class_ratio,
            "max_single_issuer_ratio": self.max_single_issuer_ratio,
            "liquidity_floor_ratio": self.liquidity_floor_ratio,
            "excluded_product_ids": list(self.excluded_product_ids),
            "excluded_categories": list(self.excluded_categories),
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "TightenSpec":
        """从 state / trace 还原指令。"""
        if not payload:
            return cls()
        return cls(
            risk_cap=payload.get("risk_cap"),
            horizon_years=payload.get("horizon_years"),
            max_single_product_ratio=payload.get("max_single_product_ratio"),
            max_single_class_ratio=payload.get("max_single_class_ratio"),
            max_single_issuer_ratio=payload.get("max_single_issuer_ratio"),
            liquidity_floor_ratio=payload.get("liquidity_floor_ratio"),
            excluded_product_ids=tuple(payload.get("excluded_product_ids") or ()),
            excluded_categories=tuple(payload.get("excluded_categories") or ()),
            reasons=tuple(payload.get("reasons") or ()),
        )


EMPTY_TIGHTEN = TightenSpec()


# ---------------------------------------------------------------------------
# 单体产品准入检查
# ---------------------------------------------------------------------------
def check_product_admissibility(product: Product, client: ClientProfile) -> list[str]:
    """检查单个产品是否落在客户硬约束可行域内。

    返回**不可投原因码**列表（空列表表示可投）。纯函数，不修改入参。
    """
    reasons: list[str] = []

    if product.risk_level > client.risk_capacity:
        reasons.append("C-RISK")

    if gt(product.horizon_years, client.investment_horizon_years, RATIO_TOL):
        reasons.append("C-HORIZON")

    if product.asset_class in client.prohibited_categories:
        reasons.append("C-PROHIBITED-CLASS")

    if product.product_id in client.prohibited_product_ids:
        reasons.append("C-PROHIBITED-PRODUCT")

    if product.is_derivative and "衍生品" in client.prohibited_categories:
        reasons.append("C-DERIVATIVE-BAN")

    if product.currency != client.currency:
        reasons.append("C-CURRENCY")

    missing = sorted(set(product.requires_experience) - set(client.experienced_categories))
    if missing:
        reasons.append("C-EXPERIENCE")

    if gt(product.min_investment, client.investable_amount, RATIO_TOL):
        reasons.append("C-ENTRY")

    if product.qualified_investor_only and not client.qualified_investor:
        reasons.append("C-QUALIFIED")

    return reasons


def admissibility_detail(product: Product, client: ClientProfile, reasons: Iterable[str]) -> str:
    """把原因码翻译成可读中文，并带上具体数值，便于客户经理解释。"""
    texts: list[str] = []
    for code in reasons:
        if code == "C-RISK":
            texts.append(
                f"产品风险等级 R{product.risk_level} 高于客户风险承受等级 R{client.risk_capacity}"
            )
        elif code == "C-HORIZON":
            texts.append(
                f"产品期限 {product.horizon_years:g} 年超过客户投资期限 {client.investment_horizon_years:g} 年"
            )
        elif code == "C-PROHIBITED-CLASS":
            texts.append(f"客户禁止投资「{product.asset_class}」类别")
        elif code == "C-PROHIBITED-PRODUCT":
            texts.append("该产品在客户禁止清单内")
        elif code == "C-DERIVATIVE-BAN":
            texts.append("该产品含衍生品结构，客户已明确排除衍生品")
        elif code == "C-CURRENCY":
            texts.append(f"产品币种 {product.currency} 与客户偏好币种 {client.currency} 不一致")
        elif code == "C-EXPERIENCE":
            missing = sorted(set(product.requires_experience) - set(client.experienced_categories))
            texts.append(f"客户缺少投资经验：{'、'.join(missing)}")
        elif code == "C-ENTRY":
            texts.append(
                f"起投金额 {product.min_investment:,.0f} 元超过客户可投金额 {client.investable_amount:,.0f} 元"
            )
        elif code == "C-QUALIFIED":
            texts.append("该产品仅面向合格投资者，客户不满足准入资格")
        else:
            texts.append(CONSTRAINT_DEFS.get(code, code))
    return "；".join(texts)


def screen_products(
    client: ClientProfile,
    products: Mapping[str, Product],
    round_index: int = 0,
) -> ScreeningResult:
    """在硬约束可行域内筛选候选产品池，并记录每一个剔除原因。"""
    included: list[str] = []
    excluded: list[Exclusion] = []

    for pid in sorted(products):
        product = products[pid]
        reasons = check_product_admissibility(product, client)
        if reasons:
            excluded.append(
                Exclusion(
                    product_id=pid,
                    product_name=product.name,
                    reasons=reasons,
                    detail=admissibility_detail(product, client, reasons),
                )
            )
        else:
            included.append(pid)

    return ScreeningResult(
        client_id=client.client_id,
        round_index=round_index,
        included=included,
        excluded=excluded,
        universe_size=len(products),
    )


# ---------------------------------------------------------------------------
# 组合层面约束检查
# ---------------------------------------------------------------------------
def _violation(
    code: str,
    detail: str,
    product_ids: Iterable[str] = (),
    severity: str = "block",
) -> Violation:
    return Violation(
        rule_id=code,
        severity=severity,  # type: ignore[arg-type]
        detail=detail,
        basis=CONSTRAINT_DEFS.get(code, ""),
        product_ids=sorted(set(product_ids)),
        code=code,
    )


def check_portfolio(
    portfolio: Portfolio,
    client: ClientProfile,
    tol: float = WEIGHT_TOL,
) -> list[Violation]:
    """组合层面的硬约束检查，返回违反项列表（按规则号+产品稳定排序）。

    `portfolio` 内嵌产品要素快照，因此本函数是自洽纯函数：
    只依赖两个入参，可以在无数据文件的情况下离线重放与单测。
    """
    violations: list[Violation] = []
    weights = portfolio.weights

    # ---- 权重合法性 ----
    for pid in sorted(weights):
        if weights[pid] < -tol:
            violations.append(_violation("C-WEIGHT-NEG", f"产品 {pid} 权重为负：{weights[pid]:.6f}", [pid]))

    total = sum(weights.get(pid, 0.0) for pid in weights)
    if total > 1.0 + tol:
        violations.append(
            _violation("C-WEIGHT-SUM", f"产品权重合计 {total:.6f} 超过 100%")
        )

    # ---- 逐产品准入检查（风险/期限/禁止项/币种/经验/起投/合格投资者）----
    for pid in sorted(weights):
        weight = weights[pid]
        if le(weight, 0.0, tol):
            continue
        product = portfolio.products.get(pid)
        if product is None:
            violations.append(_violation("C-PROHIBITED-PRODUCT", f"组合包含未登记要素的产品 {pid}", [pid]))
            continue
        for code in check_product_admissibility(product, client):
            violations.append(
                _violation(code, admissibility_detail(product, client, [code]), [pid])
            )
        # 起投金额与实际配置金额匹配
        amount = weight * client.investable_amount
        if amount + tol < product.min_investment:
            violations.append(
                _violation(
                    "C-ENTRY-MIN",
                    f"{product.name} 配置金额 {amount:,.2f} 元低于起投金额 {product.min_investment:,.2f} 元",
                    [pid],
                )
            )

    # ---- 集中度：单一产品 ----
    for pid in sorted(weights):
        weight = weights[pid]
        if gt(weight, client.max_single_product_ratio, tol):
            name = portfolio.products[pid].name if pid in portfolio.products else pid
            violations.append(
                _violation(
                    "C-CONC-SINGLE",
                    f"{name} 权重 {weight:.4%} 超过单一产品上限 {client.max_single_product_ratio:.2%}",
                    [pid],
                )
            )

    # ---- 集中度：单一资产类别 ----
    class_totals: dict[str, float] = {}
    class_members: dict[str, list[str]] = {}
    for pid in sorted(weights):
        if le(weights[pid], 0.0, tol):
            continue
        product = portfolio.products.get(pid)
        if product is None:
            continue
        class_totals[product.asset_class] = class_totals.get(product.asset_class, 0.0) + weights[pid]
        class_members.setdefault(product.asset_class, []).append(pid)
    for asset_class in sorted(class_totals):
        if gt(class_totals[asset_class], client.max_single_class_ratio, tol):
            violations.append(
                _violation(
                    "C-CONC-CLASS",
                    f"「{asset_class}」类别权重 {class_totals[asset_class]:.4%} "
                    f"超过单一类别上限 {client.max_single_class_ratio:.2%}",
                    class_members[asset_class],
                )
            )

    # ---- 集中度：同一发行人 ----
    issuer_totals: dict[str, float] = {}
    issuer_members: dict[str, list[str]] = {}
    for pid in sorted(weights):
        if le(weights[pid], 0.0, tol):
            continue
        product = portfolio.products.get(pid)
        if product is None:
            continue
        issuer_totals[product.issuer] = issuer_totals.get(product.issuer, 0.0) + weights[pid]
        issuer_members.setdefault(product.issuer, []).append(pid)
    for issuer in sorted(issuer_totals):
        if gt(issuer_totals[issuer], client.max_single_issuer_ratio, tol):
            violations.append(
                _violation(
                    "C-CONC-ISSUER",
                    f"发行人「{issuer}」合计权重 {issuer_totals[issuer]:.4%} "
                    f"超过同一发行人上限 {client.max_single_issuer_ratio:.2%}",
                    issuer_members[issuer],
                )
            )

    # ---- 流动性下限 ----
    liquid = portfolio.liquid_ratio()
    if liquid + tol < client.liquidity_floor_ratio:
        violations.append(
            _violation(
                "C-LIQUIDITY",
                f"流动性资产占比 {liquid:.4%} 低于客户流动性下限 {client.liquidity_floor_ratio:.2%}",
                portfolio.held_ids(),
            )
        )

    violations.sort(key=lambda v: v.key())
    return violations


def violation_count(portfolio: Portfolio, client: ClientProfile, tol: float = WEIGHT_TOL) -> int:
    """约束违反数（评估指标之一：被接受组合必须为 0）。"""
    return len(check_portfolio(portfolio, client, tol=tol))


def is_feasible(portfolio: Portfolio, client: ClientProfile, tol: float = WEIGHT_TOL) -> bool:
    """组合是否落在硬约束可行域内。"""
    return violation_count(portfolio, client, tol=tol) == 0


def assert_feasible(portfolio: Portfolio, client: ClientProfile, tol: float = WEIGHT_TOL) -> None:
    """断言组合可行；不可行时抛出带明细的异常（供流水线与测试使用）。"""
    violations = check_portfolio(portfolio, client, tol=tol)
    if violations:
        detail = "；".join(f"[{v.rule_id}] {v.detail}" for v in violations)
        raise AssertionError(f"组合未通过硬约束检查：{detail}")


def feasibility_report(portfolio: Portfolio, client: ClientProfile) -> dict[str, Any]:
    """可行域体检报告（demo / 评估用）。"""
    violations = check_portfolio(portfolio, client)
    return {
        "client_id": client.client_id,
        "feasible": not violations,
        "violation_count": len(violations),
        "violations": [v.model_dump() for v in violations],
        "liquid_ratio": portfolio.liquid_ratio(),
        "liquidity_floor": client.liquidity_floor_ratio,
        "max_single_weight": max(portfolio.weights.values(), default=0.0),
    }
