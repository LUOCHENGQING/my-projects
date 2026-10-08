"""硬约束求解器（确定性，绝不交给模型）。

定位
----
本模块是「约束驱动」路线的地基：把客户档案里的**硬约束**翻译成可执行的
纯函数检查，先在可行域上做求解，再让 Agent 在可行域内做多目标权衡。

所属层次
--------
架构中的**约束层**（见 README「架构总览」）：
数据层（`data/clients.json` / `data/products.json`）→ **约束层**（本模块 +
`src/suitability/`）→ Agent 层（`src/agents/`）→ 解释层
（`counterfactual` / `stress` / `versioning`）。
本层不调用任何模型：所有判定结果完全由输入决定，可单测、可重放、可逐条解释。

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

输入 / 输出
-----------
- 输入：`ClientProfile`（客户硬约束）、`Product`（产品要素）、`Portfolio`
  （权重 + 产品要素快照），三者均定义在 `src/schemas.py`。
- 输出：`ScreeningResult`（候选池与逐条剔除原因）、`list[Violation]`（违反项）、
  收紧后的 `ClientProfile` 副本（原对象不被修改）。返回值均为可序列化数据模型，
  可直接写入 trace 与建议版本快照。
- 异常：仅 `assert_feasible` 会在不可行时抛 `AssertionError`；其余函数不抛异常。

被谁调用
--------
- `src/agents/tools.py`（工具 `screening.filter_universe` / `suitability.*`）→
  `ProductScreeningAgent`、`SuitabilityOfficerAgent` 的过滤与收紧指令；
- `src/suitability/rules.py`：把数值判定翻译成带依据说明的适当性条目
  （`check_portfolio` / `check_product_admissibility` / `admissibility_detail`）；
- `src/suitability/engine.py` 与 `src/pipeline.py`：打回重配时构造/应用 `TightenSpec`；
- `src/optimizer.py`：可行性修复循环复用 `check_portfolio` 与 `PRODUCT_LEVEL_CODES`；
- `src/counterfactual.py`：反事实变体重新筛选，并把 delta 还原成 `TightenSpec`；
- `eval/run_eval.py`（候选池「预言机」交叉校验）、`src/demo.py`（可行域体检）与 `tests/`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .schemas import ClientProfile, Exclusion, Portfolio, Product, ScreeningResult, Violation
from .utils import RATIO_TOL, WEIGHT_TOL, gt, le

# ---------------------------------------------------------------------------
# 约束字典：原因码 -> 依据说明（监管口径统一使用泛称）
#
# 共 16 个原因码键，覆盖「产品准入（9 条）+ 集中度（3 条）+ 流动性（1 条）+
# 组合权重合法性（2 条）+ 配置金额不足起投（1 条）」。
# 说明文本一律使用泛称（「适当性管理要求」「客户禁止项」等），不出现具体监管
# 机构名称，便于对客户解释与合规留痕。
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
# （集中度是组合层面约束，单产品准入阶段不判定，所以交叉校验必须显式剔除这 3 个码；
#  被 eval/run_eval.py 的「候选池过滤正确率」指标使用。）
CONCENTRATION_CODES: frozenset[str] = frozenset({"C-CONC-SINGLE", "C-CONC-CLASS", "C-CONC-ISSUER"})

# 产品层面的准入原因码（只在候选池阶段判定）
# 共 9 条，与 `check_product_admissibility` 追加的原因码一一对应（不含 C-ENTRY-MIN，
# 后者依赖具体配置金额，只在组合层判定）。`src/optimizer.py` 的可行性修复循环用它
# 判断「该违反项只能靠剔除产品消除」。
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

    职责
    ----
    把「适当性闸门为什么打回」翻译成一条**可应用到客户约束上的确定性指令**：
    由 `SuitabilityGate.build_tighten`（`src/suitability/engine.py`）产生，
    由 `AdvisoryPipeline.node_suitability`（`src/pipeline.py`）应用到
    `effective_client`（生效约束）。它**不修改客户真实档案**，因此适当性判定与
    反事实解释仍可针对真实档案进行（见 `counterfactual.evaluate_variant`）。

    关键属性
    --------
    - `risk_cap` / `horizon_years`：风险等级上限、投资期限上限（取更小一侧）；
    - `max_single_product_ratio` / `max_single_class_ratio` /
      `max_single_issuer_ratio`：三重集中度上限（取更低一侧）；
    - `liquidity_floor_ratio`：流动性资产占比下限（取更高一侧）；
    - `excluded_product_ids` / `excluded_categories`：新增禁止项（取并集）；
    - `reasons`：收紧原因，合并时保序去重，最终写进版本快照的变更原因。
    字段默认 `None` / 空元组，故 `TightenSpec()` 即「不做任何收紧」（见 `EMPTY_TIGHTEN`）。

    语义统一为**单调收紧**：所有数值型约束只会取更严格的一侧
    （等级取更小、期限取更短、上限取更低、流动性下限取更高），
    因此多轮打回一定是收敛的，不会来回震荡。

    状态流转
    --------
    `闸门命中 block → build_tighten 构造 → apply() 得到收紧后的 effective_client →
    重新筛选与求解 → 再次送闸门`；重复打回时可先 `merge()` 合并多条指令。
    `to_dict()` / `from_dict()` 负责写进 `GateDecision.tighten` 与共享状态并原样还原。
    本类为 frozen dataclass 且字段均为不可变类型，可安全共享、比较与重复使用。

    被谁使用：`SuitabilityGate`（构造）、`AdvisoryPipeline`（应用与序列化）、
    `counterfactual.tighten_from_variant`（由反事实 delta 还原）。
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
        """把本条收紧指令应用到客户约束上，返回**副本**（原对象不被修改）。

        参数：`client` —— 当前生效约束；流水线中传入的是 `effective_client`
        （可能已被上一轮收紧过），而不是客户真实档案。
        返回：深拷贝后的 `ClientProfile`；本次未涉及的字段保持原值。
        副作用：无（不写磁盘、不改传入对象）。
        边界：`liquidity_floor_ratio` 额外用 `min(1.0, ...)` 兜住上界，避免指令把下限
        推到 100% 以上而永远无法满足；禁止项取并集后排序，保证同一组约束产生的
        快照与摘要稳定可复现。
        """
        updates: dict[str, Any] = {}

        # 以下数值型约束统一取「更严格的一侧」：上限取 min、下限取 max，
        # 这是「可行域单调收缩」的实现基础，也是多轮打回必然收敛的原因。
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
            # 下限取 max（越大越严），并以 1.0 封顶：100% 流动性一定可被全现金组合满足
            updates["liquidity_floor_ratio"] = max(
                client.liquidity_floor_ratio, min(1.0, float(self.liquidity_floor_ratio))
            )
        if self.excluded_categories:
            # 禁止项只增不减（并集 + 排序）：收紧单调，且结果与输入顺序无关
            merged = sorted(set(client.prohibited_categories) | set(self.excluded_categories))
            updates["prohibited_categories"] = merged
        if self.excluded_product_ids:
            merged_ids = sorted(set(client.prohibited_product_ids) | set(self.excluded_product_ids))
            updates["prohibited_product_ids"] = merged_ids

        if not updates:
            # 空指令：也必须返回新对象，避免调用方误改共享的客户档案
            return client.model_copy(deep=True)
        return client.model_copy(update=updates, deep=True)

    def merge(self, other: "TightenSpec") -> "TightenSpec":
        """按单调收紧语义合并两条指令（数值取更严格一侧，禁止项取并集）。

        参数：`other` —— 另一条收紧指令（例如新一轮闸门下发的指令）。
        返回：新的 `TightenSpec`（self 与 other 都不被修改）。
        说明：`None` 表示「本条没有对该项提出要求」，因此让位于另一侧的非空值；
        两侧都为空则该项保持为空。`reasons` 用 `dict.fromkeys` 保序去重，保证同样的
        多轮打回得到同样的原因文本顺序（可复现）。
        """
        def _tighten_min(a: float | None, b: float | None) -> float | None:
            """取更小的一侧；`None` 视作无要求，直接退化为另一侧。"""
            if a is None:
                return b
            if b is None:
                return a
            return min(a, b)

        def _tighten_max(a: float | None, b: float | None) -> float | None:
            """取更大的一侧（用于流动性下限这类「数值越大越严」的约束）。"""
            if a is None:
                return b
            if b is None:
                return a
            return max(a, b)

        # risk_cap 是 int，这里先统一转 float 以复用 _tighten_min，再取整回去；
        # 等级本身是 1-5 的整数，取整不会丢失语义。
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
        """是否为空指令（未做任何收紧）。

        返回：`True` 表示全部字段都是默认值。
        用途：合并多轮指令时判断「本轮是否真的收紧了约束」。
        """
        return self == TightenSpec()

    def to_dict(self) -> dict[str, Any]:
        """转成可写入 trace / state 的普通字典。

        返回：键名与 `TightenSpec` 字段一致的新字典（元组转列表，便于 JSON 序列化）。
        该结构即 `GateDecision.tighten` 的存储格式，与 `from_dict` 构成可逆的一对。
        """
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
        """从 state / trace 还原指令。

        参数：`payload` —— `to_dict()` 的产物，或 `GateDecision.tighten`。
        返回：对应的 `TightenSpec`；`payload` 为 `None` / 空字典时返回空指令
        （语义为「未下发收紧要求」），因此调用方无需再判空。
        """
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


#: 空收紧指令：闸门「无需收紧」与多轮合并的初始值共用同一不可变实例（可直接比较相等）
EMPTY_TIGHTEN = TightenSpec()


# ---------------------------------------------------------------------------
# 单体产品准入检查
# ---------------------------------------------------------------------------
def check_product_admissibility(product: Product, client: ClientProfile) -> list[str]:
    """检查单个产品是否落在客户硬约束可行域内。

    参数：`product` —— 待检查产品要素；`client` —— 生效客户约束。
    返回：**不可投原因码**列表，按固定判定顺序追加；空列表表示该产品可投。
    副作用/异常：无（纯函数，不修改入参，不抛异常）。
    用途：候选池筛选（`screen_products`）与组合层面逐产品复核（`check_portfolio`）共用，
    保证「筛选阶段剔除的」与「组合阶段拦截的」口径完全一致。
    """
    reasons: list[str] = []

    # 风险等级是 1-5 的整数档位，直接比较即可；期限与金额是浮点数，改用带容差的 gt()
    if product.risk_level > client.risk_capacity:
        reasons.append("C-RISK")

    if gt(product.horizon_years, client.investment_horizon_years, RATIO_TOL):
        reasons.append("C-HORIZON")

    if product.asset_class in client.prohibited_categories:
        reasons.append("C-PROHIBITED-CLASS")

    if product.product_id in client.prohibited_product_ids:
        reasons.append("C-PROHIBITED-PRODUCT")

    # 衍生品禁令单独成一个原因码：客户禁止清单里写的是「衍生品」这一品类，而含衍生品
    # 结构的产品可能以其他类别（如「混合」）登记，故再按产品结构判一次（从严原则）
    if product.is_derivative and "衍生品" in client.prohibited_categories:
        reasons.append("C-DERIVATIVE-BAN")

    if product.currency != client.currency:
        reasons.append("C-CURRENCY")

    # 集合差 = 客户缺少的投资经验品类；先排序，保证原因码集合与中文说明的确定性
    missing = sorted(set(product.requires_experience) - set(client.experienced_categories))
    if missing:
        reasons.append("C-EXPERIENCE")

    # C-ENTRY 只判断「产品门槛 vs 客户可投总额」这一产品层面事实；
    # 「已配置金额是否够起投」属于组合层面问题，由 C-ENTRY-MIN 在 check_portfolio 判定
    if gt(product.min_investment, client.investable_amount, RATIO_TOL):
        reasons.append("C-ENTRY")

    if product.qualified_investor_only and not client.qualified_investor:
        reasons.append("C-QUALIFIED")

    return reasons


def admissibility_detail(product: Product, client: ClientProfile, reasons: Iterable[str]) -> str:
    """把原因码翻译成可读中文，并带上具体数值，便于客户经理解释。

    参数：`product` / `client` —— 与 `check_product_admissibility` 同一组入参；
    `reasons` —— 待翻译的原因码集合（通常就是该函数的返回值）。
    返回：以「；」连接的中文说明，例如「产品风险等级 R4 高于客户风险承受等级 R3」。
    兜底：未在内置分支中的原因码取 `CONSTRAINT_DEFS` 的通用依据文本，仍未知则原样输出
    原因码，保证任何情况下都不会丢信息。
    用途：候选池剔除记录 `Exclusion.detail`（`screen_products`）与适当性条目
    `Violation.detail`（`check_portfolio`、`src/suitability/rules.py`）。
    """
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
    """在硬约束可行域内筛选候选产品池，并记录每一个剔除原因。

    参数：`client` —— 生效客户约束；`products` —— 全量产品要素表（key 为 product_id）；
    `round_index` —— 打回重配的轮次，原样写入结果用于版本留痕。
    返回：`ScreeningResult`（`included` 可投池、`excluded` 逐条剔除记录、
    `universe_size` 全量只数）。
    副作用/异常：无（只读入参、不抛异常）。
    """
    included: list[str] = []
    excluded: list[Exclusion] = []

    # 按 product_id 排序遍历：保证 included / excluded 的顺序与输入字典序无关（确定性）
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
    """构造一条约束违反记录（内部工厂，统一补齐依据说明与稳定排序）。

    参数：`code` —— 原因码（须是 `CONSTRAINT_DEFS` 的键，否则 `basis` 为空串）；
    `detail` —— 中文明细，通常由 `admissibility_detail` 生成；
    `product_ids` —— 涉及的产品，内部排序去重，保证同一条违反的 `key()` 稳定；
    `severity` —— 级别，本模块内一律使用默认的 `block`（`Violation` 结构同时服务于
    适当性规则层的 `warn` 级条目，但那部分由 `src/suitability/` 自行构造）。
    返回：填充好 `rule_id` / `basis` / `code` 的 `Violation`。
    """
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

    参数：`portfolio` —— 待检查组合（权重 + 产品要素快照）；`client` —— 生效客户约束；
    `tol` —— 浮点比较容差，默认 `WEIGHT_TOL`（1e-9，覆盖双精度累加误差）。
    返回：`Violation` 列表，空列表代表组合落在可行域内；已按 `Violation.key()` 排序。
    副作用/异常：无；本函数不修改 `portfolio`。

    `portfolio` 内嵌产品要素快照，因此本函数是自洽纯函数：
    只依赖两个入参，可以在无数据文件的情况下离线重放与单测。
    """
    violations: list[Violation] = []
    weights = portfolio.weights

    # ---- 权重合法性 ----
    # 本产品不做空：任何负权重都按 block 处理（不允许用负权重绕过集中度上限）
    for pid in sorted(weights):
        if weights[pid] < -tol:
            violations.append(_violation("C-WEIGHT-NEG", f"产品 {pid} 权重为负：{weights[pid]:.6f}", [pid]))

    # 只校验上界：允许权重合计 < 100%，差额即现金/活期留存（现金不受集中度约束）
    total = sum(weights.get(pid, 0.0) for pid in weights)
    if total > 1.0 + tol:
        violations.append(
            _violation("C-WEIGHT-SUM", f"产品权重合计 {total:.6f} 超过 100%")
        )

    # ---- 逐产品准入检查（风险/期限/禁止项/币种/经验/起投/合格投资者）----
    for pid in sorted(weights):
        weight = weights[pid]
        # 只复核真正持有的头寸；零权重/负权重不构成实际暴露
        if le(weight, 0.0, tol):
            continue
        product = portfolio.products.get(pid)
        if product is None:
            # 组合里出现未登记要素的产品：无从核对要素，按禁止项从严处理，绝不静默放行
            violations.append(_violation("C-PROHIBITED-PRODUCT", f"组合包含未登记要素的产品 {pid}", [pid]))
            continue
        for code in check_product_admissibility(product, client):
            violations.append(
                _violation(code, admissibility_detail(product, client, [code]), [pid])
            )
        # 起投金额与实际配置金额匹配：金额低于起投金额则该头寸在产品层面不成立
        # 边界：在容差内恰好等于起投金额视为满足（不制造假阳性）
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
    # 零权重不会触发 gt()，负权重已由 C-WEIGHT-NEG 单独报出，故此处无需再过滤
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
    # 只累加正权重持仓；`product is None`（未登记要素）的产品跳过——它已在上面报过
    # C-PROHIBITED-PRODUCT，这里再报一次只会重复计违反项
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
    # 与类别集中度同口径：跨类别持有同一发行人的产品必须合并计算（关联风险不因
    # 资产类别不同而分散），因此这里按 issuer 而不是 asset_class 汇总
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
    # 流动性占比由组合自算：现金视为 100% 流动，产品按 liquidity_ratio 折算；
    # 用 liquid + tol < floor 判定，贴边（在容差内）不算违反，避免浮点噪声误报
    liquid = portfolio.liquid_ratio()
    if liquid + tol < client.liquidity_floor_ratio:
        violations.append(
            _violation(
                "C-LIQUIDITY",
                f"流动性资产占比 {liquid:.4%} 低于客户流动性下限 {client.liquidity_floor_ratio:.2%}",
                portfolio.held_ids(),
            )
        )

    # 稳定排序（先规则号、后涉及产品）：同一组合的违反项顺序在任何环境下都一致
    violations.sort(key=lambda v: v.key())
    return violations


def violation_count(portfolio: Portfolio, client: ClientProfile, tol: float = WEIGHT_TOL) -> int:
    """约束违反数（评估指标之一：被接受组合必须为 0）。

    参数：`portfolio` / `client` / `tol` —— 同 `check_portfolio`。
    返回：违反项条数（含 warn 级；本层 `_violation` 默认 block）。
    """
    return len(check_portfolio(portfolio, client, tol=tol))


def is_feasible(portfolio: Portfolio, client: ClientProfile, tol: float = WEIGHT_TOL) -> bool:
    """组合是否落在硬约束可行域内。

    参数：同 `check_portfolio`。返回：违反数为 0 时为 `True`。
    """
    return violation_count(portfolio, client, tol=tol) == 0


def assert_feasible(portfolio: Portfolio, client: ClientProfile, tol: float = WEIGHT_TOL) -> None:
    """断言组合可行；不可行时抛出带明细的异常（供流水线与测试使用）。

    参数：同 `check_portfolio`。返回：`None`（可行时静默通过）。
    Raises: `AssertionError` —— 把每条违反拼成「[规则号] 明细」后一次抛出，
    便于定位是哪条约束、哪个产品导致不可行。
    """
    violations = check_portfolio(portfolio, client, tol=tol)
    if violations:
        detail = "；".join(f"[{v.rule_id}] {v.detail}" for v in violations)
        raise AssertionError(f"组合未通过硬约束检查：{detail}")


def feasibility_report(portfolio: Portfolio, client: ClientProfile) -> dict[str, Any]:
    """可行域体检报告（demo / 评估用）。

    参数：`portfolio` / `client` —— 待体检的组合与客户约束（使用默认容差 `WEIGHT_TOL`）。
    返回：7 个键的字典 —— `client_id`、`feasible`、`violation_count`、
    `violations`（`Violation.model_dump()` 列表）、`liquid_ratio`、`liquidity_floor`、
    `max_single_weight`（最大单一持仓权重，空组合取 0.0）。
    用途：demo 打印与评估留痕，不参与任何准入判定。
    """
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
