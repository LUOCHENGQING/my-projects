"""候选组合构建（约束感知的多目标权衡）。

定位
----
Optimizer **不做**扣减约束的"聪明"决策：
- 可行域由 `src/constraints.py` 决定（硬约束），优化器只在可行域内做权重的多目标权衡；
- 目标是四目标折中：期望收益 ↑、波动 ↓、流动性 ↑、集中度 ↓（并按客户风险承受调整权重）；
- 输出永远满足硬约束：内部有一个**确定性的可行性修复循环**，
  最后兜底手段是把剩余权重全部放进现金（现金不存在任何约束违反），因此一定收敛。

所属层次
--------
架构中的**求解环节**：位于约束层（`src/constraints.py`）之上、适当性闸门
（`src/suitability/`）之下。上游是筛选后的候选池，下游是被闸门复核的候选组合，
因此本模块必须保证「被接受的组合约束违反数为 0」——这是结构性保证，不是概率事件。

注：实际实现为——`product_score` 的打分项只有 4 项：
`+ 预期收益 − 风险惩罚（波动率）− 费用惩罚（fee_rate）+ 流动性奖励`。
上面四目标中的「集中度 ↓」**不是**打分项，而是作为硬约束由单一产品 / 单一资产类别 /
同一发行人三重上限在 `_headroom`、`_clip_class`、`_clip_issuer`、`_enforce_feasible`
中强制生效（也因此本节四目标与打分项并非一一对应）。

被谁调用
--------
- `src/agents/tools.py`（工具 `optimize.score_products` / `optimize.solve_weights`）
  → `PortfolioOptimizerAgent` 的产品打分与组合求解；
- `src/agents/portfolio_optimizer.py`：用 `binding_constraints` 生成「紧约束」说明；
- `src/counterfactual.py`：对每一条反事实假设重新 `build_portfolio` 做结构化 diff；
- `src/demo.py` 与 `eval/run_eval.py` 通过上述 Agent 工具链**间接**调用；
  `tests/test_optimizer.py` 等直接调用打分、求解与 `binding_constraints`。

输入 / 输出
-----------
- 输入：`ClientProfile`（生效约束）、候选 `product_id` 序列、
  `{product_id: Product}` 产品要素表。
- 输出：`Portfolio`（持仓权重 + 产品要素快照 + `cash_weight` + `metrics` +
  `rationale` 权衡说明列表）。
- 副作用/异常：无（不写磁盘、不调模型、不抛异常），因此同一输入永远得到同一组合。

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

#: 现金/活期留存的示例收益（用于计算组合期望收益；现金在风险与流动性上按最保守处理）
CASH_RETURN = 0.015

#: 组合保留的现金缓冲下限（同时是规则 S-CASH-RESERVE 的建议阈值，规则层用同值 0.03
#: 的独立常量 CASH_RESERVE_FLOOR 表达）：类别目标合计与补位预算都要先扣掉这 3%，
#: 保证任何组合都留有现金缓冲，也让「全现金」兜底始终是一个合规解。
CASH_RESERVE_TARGET = 0.03

#: 按风险承受等级给出的资产类别基准配置（示例口径）。
#: 风险等级 1-5 各一套，每套权重合计均为 1.00；衍生品基准为 0.00，
#: 只有在满足「候选池含衍生品 + 合格投资者 + R5 + 具备衍生品投资经验」时，
#: 才由 `_class_targets` 从各类别中按比例腾出 `DERIVATIVE_TARGET` 注入。
CLASS_TARGETS: dict[int, dict[str, float]] = {
    1: {"货币": 0.35, "固收": 0.55, "混合": 0.10, "权益": 0.00, "衍生品": 0.00},
    2: {"货币": 0.20, "固收": 0.50, "混合": 0.20, "权益": 0.10, "衍生品": 0.00},
    3: {"货币": 0.12, "固收": 0.38, "混合": 0.30, "权益": 0.20, "衍生品": 0.00},
    4: {"货币": 0.08, "固收": 0.27, "混合": 0.30, "权益": 0.35, "衍生品": 0.00},
    5: {"货币": 0.05, "固收": 0.20, "混合": 0.30, "权益": 0.45, "衍生品": 0.00},
}

#: 衍生品仅在「合格投资者 + 进取型 + 具备衍生品经验」时才纳入目标配置
DERIVATIVE_TARGET = 0.06
#: 可行性修复循环的最大迭代次数（防御性上限，正常 1-4 次即可收敛）；
#: 同一个值也用作 `_fill` 贪心补位的循环护栏，防止极端数据下空转
MAX_REPAIR_ITERATIONS = 64


def product_score(product: Product, client: ClientProfile) -> float:
    """产品打分：收益 - 风险惩罚 - 费用惩罚 + 流动性奖励。

    风险惩罚系数随客户风险承受能力下降而上升（保守客户更厌恶波动）。

    参数：`product` —— 产品要素；`client` —— 生效客户约束（只用到 `risk_capacity`）。
    返回：分数（浮点，可正可负），仅用于**同类内排序与权重分配**，不是收益预测。
    注意：分数不直接比较跨类别的高低（各类别目标权重由 `CLASS_TARGETS` 决定，
    分数只在同一资产类别内部决定相对分配）。
    """
    # 风险厌恶系数：R5 = 1.0、R4 = 1.5、R3 = 2.0、R2 = 2.5、R1 = 3.0（每降一级 +0.5）；
    # 再乘 0.5 与波动率组合，使惩罚项与收益项保持可比量纲。数值为示例口径。
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
    """按可用候选池收缩类别目标权重。

    参数：`client` —— 生效客户约束（用 `risk_capacity` 选基准表、`qualified_investor`
    与 `experienced_categories` 判衍生品资格）；`candidates` —— 候选 product_id 序列；
    `products` —— 产品要素表。
    返回：`{资产类别: 目标权重}`，键与 `CLASS_TARGETS[risk_capacity]` 一致（5 个类别）。
    规则：① 候选池中不存在的类别目标置 0；② 满足衍生品四项条件时，先把其他类别按
    `(1 - DERIVATIVE_TARGET)` 等比缩放再注入衍生品，保证目标合计仍为 1.00；
    ③ 最后用客户单一类别上限夹断，避免目标本身就越界。
    副作用/异常：无。
    """
    base = dict(CLASS_TARGETS[client.risk_capacity])
    available = {products[pid].asset_class for pid in candidates}

    # 四项条件同时满足才纳入衍生品：候选池确实有、合格投资者、R5、且具备该品类经验
    if (
        "衍生品" in available
        and client.qualified_investor
        and client.risk_capacity >= 5
        and "衍生品" in client.experienced_categories
    ):
        base = {cls: value * (1.0 - DERIVATIVE_TARGET) for cls, value in base.items()}
        base["衍生品"] = DERIVATIVE_TARGET

    # 候选池没有的类别目标归零：其份额留给后续贪心补位，而不是强行塞给其他类别
    targets = {cls: (value if cls in available else 0.0) for cls, value in base.items()}
    # 再用客户的单一类别上限夹断目标本身，避免「目标权重」一出生就违反集中度
    return {cls: min(value, client.max_single_class_ratio) for cls, value in targets.items()}


def _shifted_scores(members: Sequence[str], raw: Mapping[str, float]) -> dict[str, float]:
    """类内打分平移：保证非负且保序，用于按分数分配权重。

    参数：`members` —— 同一资产类别内的候选 product_id；`raw` —— 全量原始打分。
    返回：`{product_id: 平移后分数}`，其中类内最低分恰好等于 0.01。
    做法：先减去类内最低分（让负分也能参与比例分配），再加 0.01（让**类内最低分也拿到
    非零权重**，避免「最低分产品恒为 0 权重」这种不可解释的硬性排除）。
    副作用：无（返回新字典）。
    """
    lowest = min(raw[pid] for pid in members)
    return {pid: max(raw[pid] - lowest, 0.0) + 0.01 for pid in members}


def _class_sum(weights: Mapping[str, float], products: Mapping[str, Product], asset_class: str) -> float:
    """某资产类别当前合计权重（只统计正权重持仓，负权重不参与抵减）。"""
    return sum(w for pid, w in weights.items() if w > 0 and products[pid].asset_class == asset_class)


def _issuer_sum(weights: Mapping[str, float], products: Mapping[str, Product], issuer: str) -> float:
    """某发行人当前合计权重（只统计正权重持仓，跨类别合并计算）。"""
    return sum(w for pid, w in weights.items() if w > 0 and products[pid].issuer == issuer)


def _headroom(
    pid: str,
    weights: Mapping[str, float],
    products: Mapping[str, Product],
    client: ClientProfile,
) -> float:
    """该产品还能加多少权重（受单一产品 / 类别 / 发行人三重上限约束）。

    参数：`pid` —— 目标产品；`weights` —— 当前权重快照（只读）；
    `products` —— 产品要素表；`client` —— 生效约束（提供三重集中度上限）。
    返回：非负余量 = min(单一产品余量, 所属类别余量, 所属发行人余量)；
    外层 `max(0.0, ...)` 用于兜住「历史权重已越界」的情形（此时余量为 0 而非负数）。
    这是 `_fill` 补位时始终遵守集中度的关键：每次加仓都以本函数为封顶。
    """
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

    参数：`weights` —— **会被就地修改**的权重字典（每个产品的当前权重，0 表示未持仓）；
    `raw_scores` —— 产品原始打分；`products` —— 产品要素表；`client` —— 生效约束；
    `budget` —— 本次可分派的权重预算；`banned` —— 禁止再买入的产品（修复循环中剔除过）。
    返回：未能分配出去的余额（调用方据此决定留在现金还是继续修复）。
    副作用：就地修改 `weights`（增仓）。异常：无。
    终止性：`guard` 上限为 `MAX_REPAIR_ITERATIONS`，且每轮要求至少有一个产品被加仓
    （`progressed`），否则立即退出，因此最坏情况也是有界循环。
    """
    banned_set = set(banned)
    # 排序键：打分降序、product_id 升序——同分产品的先后仍完全确定（可复现）
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
            # 起投金额换算到权重口径：need = 起投金额 / 客户可投金额
            need = product.min_investment / client.investable_amount
            current = weights.get(pid, 0.0)
            room = _headroom(pid, weights, products, client)
            if room <= WEIGHT_TOL:
                continue
            addition = min(budget, room)
            # 新建仓位但预算连起投门槛都不够：不开仓（否则会立刻违反 C-ENTRY-MIN）
            if current <= 0 and budget + WEIGHT_TOL < need:
                continue
            # 已有仓位的加仓量太小、加完仍低于起投门槛：跳过，避免权重卡在无效仓位
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
    """把某类别权重整体缩放到上限内，返回释放出的权重。

    参数：`weights` —— 会被就地修改的权重字典；`products` —— 产品要素表；
    `asset_class` —— 目标类别；`limit` —— 该类别允许的上限。
    返回：释放出的权重（未超限时为 0.0）。
    副作用：就地按同一比例缩放该类别全部正权重持仓。
    """
    total = _class_sum(weights, products, asset_class)
    if total <= limit + WEIGHT_TOL or total <= 0:
        return 0.0
    # 等比缩放而不是只砍某一只：不破坏类内相对排序（保留打分歧序的可解释性），
    # 缩放后该类别合计恰好等于 limit
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
    """把某发行人权重整体缩放到上限内，返回释放出的权重。

    参数：`weights` —— 会被就地修改的权重字典；`products` —— 产品要素表；
    `issuer` —— 目标发行主体；`limit` —— 同一发行人允许的上限。
    返回：释放出的权重（未超限时为 0.0）。
    副作用：就地缩放该发行人全部正权重持仓；与 `_clip_class` 同口径，
    因此若某产品同时属于超限类别与超限发行人，会被两次压缩（只减不增，仍然收敛）。
    """
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

    参数：`weights` —— **会被就地修改**的权重字典（初始由类别目标 + 贪心补位得到）；
    `products` —— 产品要素表；`client` —— 生效客户约束；`raw_scores` —— 产品打分，
    供补位时按分数分配释放出来的权重。
    返回：`(cash, notes)` —— `cash` 为最终现金权重，`notes` 为去重后的修复说明（人可读，
    最终进入 `Portfolio.rationale`）。
    副作用：就地修改 `weights`。异常：无。
    终止保证：每次迭代要么消除若干违反项、要么在无法改善时直接落全现金并跳出，
    迭代次数上限为 `MAX_REPAIR_ITERATIONS`。

    终止性说明：每次迭代只会「释放权重」（剔除违规产品、压缩超限类别/发行人），
    释放出的权重先进入现金再按上限补位；由于压缩后相关上限的 headroom 变为 0，
    补位不会把权重加回违规标的，因此不存在震荡。极端情况下全部权重落到现金，
    而全现金组合必然满足所有硬约束，故算法一定在有限步内收敛。
    """
    notes: list[str] = []
    banned: set[str] = set()
    # 初始现金 = 未被产品占用的权重（入参本身可能已含现金，故不强制从 0 开始）
    cash = max(0.0, 1.0 - sum(weights.values()))

    for _ in range(MAX_REPAIR_ITERATIONS):
        # 每轮都用当前权重重建临时组合送检；banned 中的产品不再计入组合
        provisional = _portfolio_of(weights, products, cash, banned)
        violations = check_portfolio(provisional, client)
        if not violations:
            # 0 违反即收敛，这是本函数的正常出口
            break

        released = 0.0
        for violation in violations:
            code = violation.code
            if code in PRODUCT_LEVEL_CODES or code == "C-ENTRY-MIN":
                # 产品级违反（准入不合格 / 配置金额不足起投）只能靠剔除产品消除；
                # 同时加入 banned，防止后续补位把这些钱又买回同一只产品（否则会震荡）
                for pid in violation.product_ids:
                    if weights.get(pid, 0.0) > 0:
                        released += weights[pid]
                        weights[pid] = 0.0
                        banned.add(pid)
                notes.append(f"剔除不合规产品以消除 {code}")
            elif code == "C-CONC-SINGLE":
                # 单品超限：直接压到上限，多出的权重释放进现金（不在此处加仓别处）
                for pid in violation.product_ids:
                    if weights.get(pid, 0.0) > client.max_single_product_ratio:
                        old = weights[pid]
                        weights[pid] = client.max_single_product_ratio
                        released += old - weights[pid]
                notes.append("按单一产品集中度上限压缩权重")
            elif code == "C-CONC-CLASS":
                # 类别超限：按涉及产品的 asset_class 去重后整体等比压缩（可能涉及多只）
                for asset_class in sorted({products[pid].asset_class for pid in violation.product_ids if pid in products}):
                    released += _clip_class(weights, products, asset_class, client.max_single_class_ratio)
                notes.append("按单一资产类别上限压缩权重")
            elif code == "C-CONC-ISSUER":
                # 发行人超限：跨类别合并后压缩，因此按 issuer 而不是 asset_class 处理
                for issuer in sorted({products[pid].issuer for pid in violation.product_ids if pid in products}):
                    released += _clip_issuer(weights, products, issuer, client.max_single_issuer_ratio)
                notes.append("按同一发行人上限压缩权重")
            elif code == "C-WEIGHT-SUM":
                # 合计超 100%：对全部产品权重整体归一化（现金权重在此不受影响）
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

        # 释放出来的权重先全部进现金，再按上限重新补位（先收回、再分配）
        cash += released
        # 只把超出 3% 现金缓冲的部分拿去补位，保证补位之后组合仍然留有现金
        fill_budget = max(0.0, cash - CASH_RESERVE_TARGET)
        leftover = _fill(weights, raw_scores, products, client, fill_budget, banned=banned)
        # 未能补出去的余额留在现金：因此现金权重 = 原有现金 − 预算 + 余额
        cash = cash - fill_budget + leftover

        if released <= WEIGHT_TOL:
            # 无法再通过局部调整改善：兜底为全现金，保证可行性与终止性
            notes.append("剩余权重全部转入现金以保证组合可行（确定性兜底策略）")
            cash += sum(weights.values())
            for pid in weights:
                weights[pid] = 0.0
            break

    # 说明列表去重保序：同一修复原因（如多轮剔除产品）只保留首次出现的表述
    return cash, list(dict.fromkeys(notes))


def _portfolio_of(
    weights: Mapping[str, float],
    products: Mapping[str, Product],
    cash: float,
    banned: Iterable[str] = (),
) -> Portfolio:
    """由权重字典快速构造 Portfolio（内部使用）。

    参数：`weights` —— 当前权重字典；`products` —— 产品要素表；`cash` —— 现金权重；
    `banned` —— 需要排除的产品（可能因修复被剔除，但字典里仍留有 0 值条目）。
    返回：仅包含「权重 > `WEIGHT_TOL` 且未被 banned」的持仓及其实例快照的 `Portfolio`；
    这是修复循环送检用的临时组合，不带 `metrics` 与 `rationale`。
    副作用/异常：无。
    """
    banned_set = set(banned)
    held = {pid: w for pid, w in weights.items() if w > WEIGHT_TOL and pid not in banned_set}
    return Portfolio(
        weights=dict(held),
        products={pid: products[pid] for pid in held},
        cash_weight=cash,
    )


def compute_metrics(portfolio: Portfolio, client: ClientProfile) -> dict[str, float]:
    """组合预期指标（确定性公式）。

    参数：`portfolio` —— 待计算的组合（用其权重、产品要素与现金权重）。
    返回：8 个键的字典 —— `expected_return`（含现金收益的年化期望）、
    `expected_volatility`、`expected_fee_rate`、`liquidity_ratio`、
    `max_single_weight`、`weighted_risk_level`、`holding_count`（浮点只数）、`cash_weight`；
    所有数值四舍五入到 12 位小数，保证快照与摘要可复现。
    简化口径：波动按 `sqrt(Σ(w_i·σ_i)²)`，**忽略资产间相关性**；现金收益率取
    `CASH_RETURN`、风险等级按 1.0 计入加权风险（已知局限见 README）。
    注：实际实现中参数 `client` 未被读取，指标完全由 `portfolio` 决定（签名保留以统一调用方式）。
    副作用/异常：无。
    """
    held = portfolio.held_ids()
    expected_return = portfolio.cash_weight * CASH_RETURN
    variance = 0.0
    fee = 0.0
    # 现金按风险等级 1.0 计入加权风险，与产品风险等级同量纲（简化口径）
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

    参数：`client` —— 生效客户约束；`candidates` —— 候选 product_id 序列（内部会去重排序）；
    `products` —— 全量产品要素表。
    返回：`Portfolio`（含 `weights`、产品要素快照、`cash_weight`、`metrics` 与
    `rationale` 权衡/修复说明）。**返回值必然 0 约束违反**（由 `_enforce_feasible` 保证）。
    副作用：无（不修改入参）。异常：无。

    步骤：类别目标 -> 类内按打分分配 -> 贪心补位 -> 可行性修复 -> 指标计算。
    """
    candidate_ids = sorted(set(candidates))
    if not candidate_ids:
        # 可行域为空（如客户禁止了全部可投类别）：返回全现金组合并写明理由，
        # 让下游闸门的 S-FEASIBLE-POOL 与建议书能直接引用这条解释
        portfolio = Portfolio(weights={}, products={}, cash_weight=1.0, rationale=["可行域为空，无法构建组合"])
        portfolio.metrics = compute_metrics(portfolio, client)
        return portfolio

    raw_scores = {pid: product_score(products[pid], client) for pid in candidate_ids}
    targets = _class_targets(client, candidate_ids, products)

    # 类别目标合计超过「1 - 现金缓冲」时按比例收缩，保证组合始终留有现金缓冲。
    # 注意：收缩只缩放目标权重、不改类别之间的相对比例（保持基准配置的形状）
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

    # 起投金额校验前置：低于起投金额的小额头寸先归零（按产品自身起投金额判定），
    # 避免这些头寸进入修复循环后被反复剔除、打乱补位预算
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

    # 落库前统一四舍五入到 12 位小数：与 compute_metrics / 快照摘要的口径一致，
    # 保证同一输入在不同环境下产生完全相同的组合与哈希
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
    """识别"紧约束"（贴边的上限），用于向客户解释权衡边界。

    参数：`portfolio` —— 最终组合；`client` —— 生效客户约束。
    返回：中文说明列表，逐条指出已被顶格的上限：单一产品上限（逐个持仓判定）、
    单一类别上限（跳过「现金」类别，现金不受该上限约束）、同一发行人上限、
    流动性下限。空列表表示没有贴边的上限。
    说明：产品/类别/发行人三类用 `abs(差值) < 1e-6` 做对称贴边判定；
    注：实际实现为——流动性下限用的是 `liquid_ratio() - liquidity_floor_ratio < 1e-6`
    （未取绝对值），因此「低于下限」时同样会被标记；正常流程中组合必然可行，
    故实际效果等同于贴边判定。
    本函数只产出解释文本，不参与任何准入或拦截。副作用/异常：无。
    """
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
    """两份权重字典的逐产品差异（target - baseline），只保留非零差异。

    参数：`baseline` / `target` —— 两份 `{product_id: 权重}`；`tol` —— 差异过滤阈值。
    返回：只含 `|差异| > tol` 的条目，值四舍五入到 12 位小数，键按 product_id 排序。
    说明：只出现在一侧的产品按 0 参与计算，因此「新增 / 剔除」也表现为一条差异。
    副作用/异常：无。主要供测试与版本对比使用。
    """
    keys = sorted(set(baseline) | set(target))
    diff = {key: round(target.get(key, 0.0) - baseline.get(key, 0.0), 12) for key in keys}
    return {key: value for key, value in diff.items() if abs(value) > tol}


def clamp_cap(value: float) -> float:
    """把上限裁剪到 [0.01, 1.0]，避免打回重配时上限被压到 0。

    参数：`value` —— 原始上限（例如被收紧后的集中度上限）。
    返回：落在 [0.01, 1.0] 区间内的值（低于 0.01 抬到 0.01，高于 1.0 压到 1.0）。
    边界含义：上限一旦被压到 0，可行域会直接变空（没有任何产品可配置），
    因此这里保留 1% 的最小可配额度。
    注：实际实现中本函数在当前仓库内没有调用点（仅在定义处出现），属于对外保留的工具函数。
    """
    return clamp(value, 0.01, 1.0)
