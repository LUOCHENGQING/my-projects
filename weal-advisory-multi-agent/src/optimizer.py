"""候选组合构建（约束感知的多目标权衡）。

定位
----
Optimizer **不做**扣减约束的"聪明"决策：
- 可行域由 `src/constraints.py` 决定（硬约束），优化器只在可行域内做权重的多目标权衡；
- 目标是四目标折中：期望收益 ↑、波动 ↓、流动性 ↑、集中度 ↓（并按客户风险承受调整权重）；
- 输出永远满足硬约束：内部有一个**确定性的可行性修复循环**，
  最后兜底手段是把剩余权重全部放进现金（现金不存在任何约束违反），因此一定收敛。

为什么不用均值-方差凸优化？
--------------------------
演示数据集只有 17 只产品且约束是**分段的业务约束**（起投金额、集中度、准入、
流动性比例），用确定性打分 + 贪心补位 + 修复循环更容易做到「结果完全可复现、
每一分钱权重都能解释为什么」，也更契合适当性场景。已知局限见 README。
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from .constraints import PRODUCT_LEVEL_CODES, check_portfolio
from .schemas import ClientProfile, Portfolio, Product
from .utils import WEIGHT_TOL, clamp

#: 现金/活期留存的示例收益（用于计算组合期望收益）
CASH_RETURN = 0.015

#: 组合保留的现金缓冲下限（同时是规则 S-CASH-RESERVE 的建议阈值）
CASH_RESERVE_TARGET = 0.03

#: 按风险承受等级给出的资产类别基准配置（示例口径）
CLASS_TARGETS: dict[int, dict[str, float]] = {
    1: {"货币": 0.35, "固收": 0.55, "混合": 0.10, "权益": 0.00, "衍生品": 0.00},
    2: {"货币": 0.20, "固收": 0.50, "混合": 0.20, "权益": 0.10, "衍生品": 0.00},
    3: {"货币": 0.12, "固收": 0.38, "混合": 0.30, "权益": 0.20, "衍生品": 0.00},
    4: {"货币": 0.08, "固收": 0.27, "混合": 0.30, "权益": 0.35, "衍生品": 0.00},
    5: {"货币": 0.05, "固收": 0.20, "混合": 0.30, "权益": 0.45, "衍生品": 0.00},
}

#: 衍生品仅在「合格投资者 + 进取型 + 具备衍生品经验」时才纳入目标配置
DERIVATIVE_TARGET = 0.06
#: 可行性修复循环的最大迭代次数（防御性上限，正常 1-4 次即可收敛）
MAX_REPAIR_ITERATIONS = 64


def product_score(product: Product, client: ClientProfile) -> float:
    """产品打分：收益 - 风险惩罚 - 费用惩罚 + 流动性奖励。

    风险惩罚系数随客户风险承受能力下降而上升（保守客户更厌恶波动）。
    """
    risk_aversion = 1.0 + (5 - client.risk_capacity) * 0.5
    return (
        product.expected_return
        - risk_aversion * 0.5 * product.volatility
        - 0.5 * product.fee_rate
        + 0.02 * product.liquidity_ratio
    )


def _class_targets(
    client: ClientProfile,
    candidates: Sequence[str],
    products: Mapping[str, Product],
) -> dict[str, float]:
    """按可用候选池收缩类别目标权重。"""
    base = dict(CLASS_TARGETS[client.risk_capacity])
    available = {products[pid].asset_class for pid in candidates}

    if (
        "衍生品" in available
        and client.qualified_investor
        and client.risk_capacity >= 5
        and "衍生品" in client.experienced_categories
    ):
        base = {cls: value * (1.0 - DERIVATIVE_TARGET) for cls, value in base.items()}
        base["衍生品"] = DERIVATIVE_TARGET

    targets = {cls: (value if cls in available else 0.0) for cls, value in base.items()}
    return {cls: min(value, client.max_single_class_ratio) for cls, value in targets.items()}


def _shifted_scores(members: Sequence[str], raw: Mapping[str, float]) -> dict[str, float]:
    """类内打分平移：保证非负且保序，用于按分数分配权重。"""
    lowest = min(raw[pid] for pid in members)
    return {pid: max(raw[pid] - lowest, 0.0) + 0.01 for pid in members}


def _class_sum(weights: Mapping[str, float], products: Mapping[str, Product], asset_class: str) -> float:
    return sum(w for pid, w in weights.items() if w > 0 and products[pid].asset_class == asset_class)


def _issuer_sum(weights: Mapping[str, float], products: Mapping[str, Product], issuer: str) -> float:
    return sum(w for pid, w in weights.items() if w > 0 and products[pid].issuer == issuer)


def _headroom(
    pid: str,
    weights: Mapping[str, float],
    products: Mapping[str, Product],
    client: ClientProfile,
) -> float:
    """该产品还能加多少权重（受单一产品 / 类别 / 发行人三重上限约束）。"""
    product = products[pid]
    current = weights.get(pid, 0.0)
    return max(
        0.0,
        min(
            client.max_single_product_ratio - current,
            client.max_single_class_ratio - _class_sum(weights, products, product.asset_class),
            client.max_single_issuer_ratio - _issuer_sum(weights, products, product.issuer),
        ),
    )


def _fill(
    weights: dict[str, float],
    raw_scores: Mapping[str, float],
    products: Mapping[str, Product],
    client: ClientProfile,
    budget: float,
    banned: Iterable[str] = (),
) -> float:
    """贪心补位：把 budget 按打分从高到低分配到产品上，返回未分配完的余额。

    遵守：单一产品 / 单一类别 / 同一发行人上限、起投金额门槛。
    """
    banned_set = set(banned)
    order = sorted(
        (pid for pid in weights if pid not in banned_set),
        key=lambda pid: (-raw_scores[pid], pid),
    )
    guard = 0
    while budget > WEIGHT_TOL and guard < MAX_REPAIR_ITERATIONS:
        guard += 1
        progressed = False
        for pid in order:
            if budget <= WEIGHT_TOL:
                break
            product = products[pid]
            need = product.min_investment / client.investable_amount
            current = weights.get(pid, 0.0)
            room = _headroom(pid, weights, products, client)
            if room <= WEIGHT_TOL:
                continue
            addition = min(budget, room)
            if current <= 0 and budget + WEIGHT_TOL < need:
                continue
            if current + addition + WEIGHT_TOL < need:
                continue
            weights[pid] = current + addition
            budget -= addition
            progressed = True
        if not progressed:
            break
    return budget


def _clip_class(
    weights: dict[str, float],
    products: Mapping[str, Product],
    asset_class: str,
    limit: float,
) -> float:
    """把某类别权重整体缩放到上限内，返回释放出的权重。"""
    total = _class_sum(weights, products, asset_class)
    if total <= limit + WEIGHT_TOL or total <= 0:
        return 0.0
    factor = limit / total
    released = 0.0
    for pid in sorted(weights):
        if weights[pid] <= 0:
            continue
        if products[pid].asset_class != asset_class:
            continue
        old = weights[pid]
        weights[pid] = old * factor
        released += old - weights[pid]
    return released


def _clip_issuer(
    weights: dict[str, float],
    products: Mapping[str, Product],
    issuer: str,
    limit: float,
) -> float:
    """把某发行人权重整体缩放到上限内，返回释放出的权重。"""
    total = _issuer_sum(weights, products, issuer)
    if total <= limit + WEIGHT_TOL or total <= 0:
        return 0.0
    factor = limit / total
    released = 0.0
    for pid in sorted(weights):
        if weights[pid] <= 0 or products[pid].issuer != issuer:
            continue
        old = weights[pid]
        weights[pid] = old * factor
        released += old - weights[pid]
    return released


def _enforce_feasible(
    weights: dict[str, float],
    products: Mapping[str, Product],
    client: ClientProfile,
    raw_scores: Mapping[str, float],
) -> tuple[float, list[str]]:
    """确定性可行性修复循环：保证最终组合 0 违反（兜底为全现金）。

    返回 (现金权重, 修复说明列表)。

    终止性说明：每次迭代只会「释放权重」（剔除违规产品、压缩超限类别/发行人），
    释放出的权重先进入现金再按上限补位；由于压缩后相关上限的 headroom 变为 0，
    补位不会把权重加回违规标的，因此不存在震荡。极端情况下全部权重落到现金，
    而全现金组合必然满足所有硬约束，故算法一定在有限步内收敛。
    """
    notes: list[str] = []
    banned: set[str] = set()
    cash = max(0.0, 1.0 - sum(weights.values()))

    for _ in range(MAX_REPAIR_ITERATIONS):
        provisional = _portfolio_of(weights, products, cash, banned)
        violations = check_portfolio(provisional, client)
        if not violations:
            break

        released = 0.0
        for violation in violations:
            code = violation.code
            if code in PRODUCT_LEVEL_CODES or code == "C-ENTRY-MIN":
                for pid in violation.product_ids:
                    if weights.get(pid, 0.0) > 0:
                        released += weights[pid]
                        weights[pid] = 0.0
                        banned.add(pid)
                notes.append(f"剔除不合规产品以消除 {code}")
            elif code == "C-CONC-SINGLE":
                for pid in violation.product_ids:
                    if weights.get(pid, 0.0) > client.max_single_product_ratio:
                        old = weights[pid]
                        weights[pid] = client.max_single_product_ratio
                        released += old - weights[pid]
                notes.append("按单一产品集中度上限压缩权重")
            elif code == "C-CONC-CLASS":
                for asset_class in sorted({products[pid].asset_class for pid in violation.product_ids if pid in products}):
                    released += _clip_class(weights, products, asset_class, client.max_single_class_ratio)
                notes.append("按单一资产类别上限压缩权重")
            elif code == "C-CONC-ISSUER":
                for issuer in sorted({products[pid].issuer for pid in violation.product_ids if pid in products}):
                    released += _clip_issuer(weights, products, issuer, client.max_single_issuer_ratio)
                notes.append("按同一发行人上限压缩权重")
            elif code == "C-WEIGHT-SUM":
                total = sum(weights.values())
                if total > 1.0:
                    for pid in weights:
                        weights[pid] = weights[pid] / total
                    released += total - 1.0
                notes.append("按 100% 上限归一化权重")
            elif code == "C-LIQUIDITY":
                # 从流动性最低的持仓开始抽离权重转入现金，直到补足流动性缺口
                held_ids = sorted(
                    (pid for pid in weights if weights[pid] > WEIGHT_TOL and pid not in banned),
                    key=lambda pid: (products[pid].liquidity_ratio, pid),
                )
                for pid in held_ids:
                    liquid = (cash + released) + sum(
                        weights[p] * products[p].liquidity_ratio
                        for p in weights
                        if weights[p] > WEIGHT_TOL and p not in banned
                    )
                    if liquid + WEIGHT_TOL >= client.liquidity_floor_ratio:
                        break
                    if products[pid].liquidity_ratio >= 1.0:
                        continue
                    release = weights[pid]
                    weights[pid] = 0.0
                    banned.add(pid)
                    released += release
                if released > 0:
                    notes.append("提升现金与高流动性资产比例以满足流动性下限")
            else:
                notes.append(f"未识别的违反项 {code}，交由兜底策略处理")

        cash += released
        fill_budget = max(0.0, cash - CASH_RESERVE_TARGET)
        leftover = _fill(weights, raw_scores, products, client, fill_budget, banned=banned)
        cash = cash - fill_budget + leftover

        if released <= WEIGHT_TOL:
            # 无法再通过局部调整改善：兜底为全现金，保证可行性与终止性
            notes.append("剩余权重全部转入现金以保证组合可行（确定性兜底策略）")
            cash += sum(weights.values())
            for pid in weights:
                weights[pid] = 0.0
            break

    return cash, list(dict.fromkeys(notes))


def _portfolio_of(
    weights: Mapping[str, float],
    products: Mapping[str, Product],
    cash: float,
    banned: Iterable[str] = (),
) -> Portfolio:
    """由权重字典快速构造 Portfolio（内部使用）。"""
    banned_set = set(banned)
    held = {pid: w for pid, w in weights.items() if w > WEIGHT_TOL and pid not in banned_set}
    return Portfolio(
        weights=dict(held),
        products={pid: products[pid] for pid in held},
        cash_weight=cash,
    )


def compute_metrics(portfolio: Portfolio, client: ClientProfile) -> dict[str, float]:
    """组合预期指标（确定性公式）。"""
    held = portfolio.held_ids()
    expected_return = portfolio.cash_weight * CASH_RETURN
    variance = 0.0
    fee = 0.0
    weighted_risk = portfolio.cash_weight * 1.0
    for pid in held:
        weight = portfolio.weights[pid]
        product = portfolio.products[pid]
        expected_return += weight * product.expected_return
        variance += (weight * product.volatility) ** 2  # 简化：忽略资产间相关性
        fee += weight * product.fee_rate
        weighted_risk += weight * product.risk_level

    return {
        "expected_return": round(expected_return, 12),
        "expected_volatility": round(variance ** 0.5, 12),
        "expected_fee_rate": round(fee, 12),
        "liquidity_ratio": round(portfolio.liquid_ratio(), 12),
        "max_single_weight": round(max(portfolio.weights.values(), default=0.0), 12),
        "weighted_risk_level": round(weighted_risk, 12),
        "holding_count": float(len(held)),
        "cash_weight": round(portfolio.cash_weight, 12),
    }


def build_portfolio(
    client: ClientProfile,
    candidates: Sequence[str],
    products: Mapping[str, Product],
) -> Portfolio:
    """在候选池内构建满足全部硬约束的候选组合。

    步骤：类别目标 -> 类内按打分分配 -> 贪心补位 -> 可行性修复 -> 指标计算。
    """
    candidate_ids = sorted(set(candidates))
    if not candidate_ids:
        portfolio = Portfolio(weights={}, products={}, cash_weight=1.0, rationale=["可行域为空，无法构建组合"])
        portfolio.metrics = compute_metrics(portfolio, client)
        return portfolio

    raw_scores = {pid: product_score(products[pid], client) for pid in candidate_ids}
    targets = _class_targets(client, candidate_ids, products)

    # 类别目标合计超过「1 - 现金缓冲」时按比例收缩，保证组合始终留有现金缓冲
    target_total = sum(targets.values())
    if target_total > 1.0 - CASH_RESERVE_TARGET:
        scale = (1.0 - CASH_RESERVE_TARGET) / target_total
        targets = {cls: value * scale for cls, value in targets.items()}

    weights: dict[str, float] = {pid: 0.0 for pid in candidate_ids}
    rationale: list[str] = []

    for asset_class in sorted(targets):
        target = targets[asset_class]
        members = [pid for pid in candidate_ids if products[pid].asset_class == asset_class]
        if not members or target <= 0:
            continue
        shifted = _shifted_scores(members, raw_scores)
        total = sum(shifted[pid] for pid in members)
        for pid in members:
            weights[pid] = target * shifted[pid] / total
        rationale.append(
            f"{asset_class}类别目标权重 {target:.2%}，在 {len(members)} 只候选中按多目标打分分配"
        )

    available = max(0.0, 1.0 - sum(weights.values()))
    fill_budget = max(0.0, available - CASH_RESERVE_TARGET)
    if fill_budget > WEIGHT_TOL:
        rationale.append(f"未被类别目标覆盖的 {fill_budget:.2%} 权重按打分补位")
    leftover = _fill(weights, raw_scores, products, client, fill_budget)

    # 起投金额校验前置：低于起投金额的小额头寸先归零，避免修复循环反复剔除
    for pid in sorted(candidate_ids):
        if 0 < weights[pid] * client.investable_amount < products[pid].min_investment:
            weights[pid] = 0.0
    available = max(0.0, 1.0 - sum(weights.values()))
    fill_budget = max(0.0, available - CASH_RESERVE_TARGET)
    if fill_budget > WEIGHT_TOL:
        leftover = _fill(weights, raw_scores, products, client, fill_budget)
        rationale.append("为低于起投金额的头寸重新补位")
    del leftover  # 余额由 _enforce_feasible 依据实际权重重新推导现金

    cash, repair_notes = _enforce_feasible(weights, products, client, raw_scores)
    cash = round(max(0.0, cash), 12)
    rationale.extend(dict.fromkeys(repair_notes))

    held = {pid: round(w, 12) for pid, w in weights.items() if w > WEIGHT_TOL}
    portfolio = Portfolio(
        weights=held,
        products={pid: products[pid] for pid in held},
        cash_weight=cash,
        rationale=rationale,
    )
    portfolio.metrics = compute_metrics(portfolio, client)
    return portfolio


def binding_constraints(portfolio: Portfolio, client: ClientProfile) -> list[str]:
    """识别"紧约束"（贴边的上限），用于向客户解释权衡边界。"""
    binding: list[str] = []
    limit = client.max_single_product_ratio
    for pid in portfolio.held_ids():
        if abs(portfolio.weights[pid] - limit) < 1e-6:
            binding.append(f"单一产品上限（{portfolio.products[pid].name} 已顶格 {limit:.2%}）")
    for asset_class, total in sorted(portfolio.class_weights().items()):
        if asset_class == "现金":
            continue
        if abs(total - client.max_single_class_ratio) < 1e-6:
            binding.append(f"单一类别上限（{asset_class} 已顶格 {client.max_single_class_ratio:.2%}）")
    for issuer, total in sorted(portfolio.issuer_weights().items()):
        if abs(total - client.max_single_issuer_ratio) < 1e-6:
            binding.append(f"同一发行人上限（{issuer} 已顶格 {client.max_single_issuer_ratio:.2%}）")
    if portfolio.liquid_ratio() - client.liquidity_floor_ratio < 1e-6:
        binding.append(f"流动性下限（{client.liquidity_floor_ratio:.2%}）")
    return binding


def weight_diff(baseline: Mapping[str, float], target: Mapping[str, float], tol: float = 1e-9) -> dict[str, float]:
    """两份权重字典的逐产品差异（target - baseline），只保留非零差异。"""
    keys = sorted(set(baseline) | set(target))
    diff = {key: round(target.get(key, 0.0) - baseline.get(key, 0.0), 12) for key in keys}
    return {key: value for key, value in diff.items() if abs(value) > tol}


def clamp_cap(value: float) -> float:
    """把上限裁剪到 [0.01, 1.0]，避免打回重配时上限被压到 0。"""
    return clamp(value, 0.01, 1.0)
