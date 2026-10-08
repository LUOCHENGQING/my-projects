"""领域数据模型。

层次定位
--------
本模块是**最底层的共享领域模型层**：只依赖 pydantic，被 `src/constraints.py`、
`src/suitability/`、`src/optimizer.py`、`src/narrative.py`、`src/pipeline.py`、
`src/versioning.py`、`eval/run_eval.py` 与全部测试模块共同引用，
是流水线各节点之间传递数据的统一「数据契约」。

本模块只负责**结构化建模**与**轻量自洽校验**，不含任何业务判断：
- 「能不能配」→ `src/constraints.py`（确定性硬约束求解器）
- 「该不该拦」→ `src/suitability/rules.py`（适当性规则库）
这样保证约束与规则都是可单测、可重放的纯函数，而模型只承担数据载体职责。

对外暴露的关键对象
------------------
- 客户侧：`ClientProfile`（硬约束 / 软偏好 / 适当性属性 + 派生属性与约束快照）
- 产品侧：`Product`（要素表 + 不可变要素快照 `snapshot()`）
- 过程与结论：`Violation`、`Portfolio`、`Exclusion`、`ScreeningResult`、
  `GateDecision`、`HumanReview`
- 解释与留痕：`ScenarioImpact`、`StressReport`、`CounterfactualVariant`、
  `CounterfactualReport`、`AdviceRecord`
- 枚举与工具：`RiskLevel` 相关的中文标签表 `RISK_LEVEL_LABELS`、`risk_label()`、
  类型别名 `Severity`（"block" / "warn"）与 `Directive`（"pass" / "reoptimize" / "reject"）

主要输入 / 输出
---------------
输入：各 Agent 与求解器构造的字段值，以及从 `data/*.json` 反序列化的原始字典
（由 `src/dataset.py` 用 `model_validate` 装载）。
输出：可作为流水线共享状态成员、可 `model_dump()` 落进版本快照的 pydantic 对象。

约定：多数模型设置了 `ConfigDict(extra="forbid")`，即输入出现未声明字段时**直接报错**
（宁可失败也不静默丢字段）；少数纯结果模型未设置该约束。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# 风险等级中文标签（1 最低、5 最高）
RISK_LEVEL_LABELS: dict[int, str] = {
    1: "R1-低风险",
    2: "R2-中低风险",
    3: "R3-中风险",
    4: "R4-中高风险",
    5: "R5-高风险",
}

Severity = Literal["block", "warn"]
Directive = Literal["pass", "reoptimize", "reject"]


def risk_label(level: int) -> str:
    """风险等级 -> 中文标签。

    参数：
        level：风险等级整数，约定取值 1-5（1 最低、5 最高）。

    返回：`RISK_LEVEL_LABELS` 中的中文标签；越界等级返回 `f"R{level}-未知"`
    （注：实际实现对任意不在表内的整数都走该兜底分支，不会抛异常）。
    """
    return RISK_LEVEL_LABELS.get(level, f"R{level}-未知")


class ClientProfile(BaseModel):
    """客户档案：硬约束 + 软偏好 + 适当性相关属性。

    职责：作为全流程**唯一的客户约束来源**——`constraints.py` 用它判定可行域，
    `suitability/rules.py` 用它作为适当性基准，`optimizer.py` 用它确定目标权重，
    `pipeline.node_profile` 产出的 `effective_client` 也是本模型的深拷贝（仅风险等级被收紧）。

    关键字段分组：
    - 硬约束：`risk_capacity`、`investment_horizon_years`、`liquidity_floor_ratio`、
      `max_single_product_ratio` / `max_single_class_ratio` / `max_single_issuer_ratio`、
      `investable_amount`、`prohibited_categories` / `prohibited_product_ids`、
      `experienced_categories`、`qualified_investor`、`currency`、`tax_advantaged_quota`；
    - 适当性与留痕：`dual_record_completed`、`annual_fee_budget_ratio`、
      `internal_single_product_warning_ratio`；
    - 软偏好：`max_drawdown_tolerance`、`return_target`；
    - 问卷与档案：`risk_questionnaire_score`、`notes`、`eval_sample`。

    派生属性（`is_elderly` / `warning_ratio`）与 `constraint_snapshot()` 是给下游直接消费的
    只读出口，避免各处重复实现同一口径。

    被谁使用：`src/dataset.py`（装载）、`src/agents/client_profiling.py`（收紧等级）、
    `src/constraints.py` / `src/suitability/` / `src/optimizer.py`（判据）、
    `src/pipeline.py`（共享状态）、`src/versioning.py`（写进版本快照）。
    """

    model_config = ConfigDict(extra="forbid")

    client_id: str
    display_name: str
    age: int = Field(ge=18, le=100)

    # ---- 硬约束 ----
    risk_capacity: int = Field(ge=1, le=5, description="风险承受等级上限（1-5）")
    investment_horizon_years: float = Field(gt=0, description="投资期限（年）")
    liquidity_floor_ratio: float = Field(ge=0, le=1, description="流动性资产占比下限")
    max_single_product_ratio: float = Field(gt=0, le=1, description="单一产品集中度上限")
    max_single_class_ratio: float = Field(gt=0, le=1, description="单一资产类别上限")
    max_single_issuer_ratio: float = Field(gt=0, le=1, description="同一发行人上限")
    investable_amount: float = Field(gt=0, description="可投金额（元）")
    prohibited_categories: list[str] = Field(default_factory=list, description="禁止项（资产类别/品类）")
    prohibited_product_ids: list[str] = Field(default_factory=list, description="禁止项（具体产品）")
    experienced_categories: list[str] = Field(default_factory=list, description="已具备投资经验的品类")
    qualified_investor: bool = Field(default=False, description="是否合格投资者")
    currency: str = Field(default="CNY", description="币种偏好")
    tax_advantaged_quota: float = Field(default=0.0, ge=0, description="税收优惠额度（元）")

    # ---- 适当性与留痕 ----
    dual_record_completed: bool = Field(default=False, description="是否已完成双录留痕")
    annual_fee_budget_ratio: float = Field(default=0.02, ge=0, description="年度综合费率预算上限")
    internal_single_product_warning_ratio: float | None = Field(
        default=None, description="内控单一产品集中度预警线（缺省取集中度上限的 80%）"
    )

    # ---- 软偏好 ----
    max_drawdown_tolerance: float = Field(default=0.15, ge=0, le=1, description="最大回撤容忍度")
    return_target: float = Field(default=0.04, description="年化收益目标（软偏好）")

    # ---- 问卷与档案 ----
    risk_questionnaire_score: int | None = Field(default=None, ge=0, le=100, description="风险测评问卷总分")
    notes: str = ""
    eval_sample: bool = True

    @field_validator("prohibited_categories", "experienced_categories", "prohibited_product_ids")
    @classmethod
    def _strip_items(cls, value: list[str]) -> list[str]:
        """去除空白项并保持稳定顺序，避免同一语义出现两种快照。

        参数：
            value：三个「字符串列表」字段中任意一个的原始取值。

        返回：逐项 `strip()` 后的新列表；空串与纯空白项被丢弃，**相对顺序不变**
        （不是排序，只做过滤与去空白）。

        副作用：无（返回新列表，不改动入参对象）。
        """
        return [item.strip() for item in value if item and item.strip()]

    # ------------------------------------------------------------------
    # 派生属性
    # ------------------------------------------------------------------
    @property
    def is_elderly(self) -> bool:
        """高龄客户判定（示例阈值：年满 65 周岁）。

        返回：`age >= 65` 为 True。该标记会触发两处保护动作：
        `pipeline.human_review_reasons` 追加「特别保护确认」，
        `narrative.dual_record_required` 参与双录留痕判定。
        """
        return self.age >= 65

    @property
    def warning_ratio(self) -> float:
        """内控单一产品集中度预警线（未显式配置时取集中度上限的 80%）。

        返回：显式配置值时返回 `internal_single_product_warning_ratio`；
        否则返回 `round(max_single_product_ratio * 0.8, 12)`（保留 12 位以规避浮点误差）。

        说明：该值是**内控预警**而非硬约束——超过它不会拦截组合，
        但会要求理财经理人工确认（见 `pipeline.human_review_reasons`）。
        """
        if self.internal_single_product_warning_ratio is not None:
            return float(self.internal_single_product_warning_ratio)
        return round(self.max_single_product_ratio * 0.8, 12)

    def constraint_snapshot(self) -> dict[str, Any]:
        """客户硬约束快照：写入建议版本链，用于事后回溯「当时按什么约束配的」。

        返回：扁平字典，键为字段名（含派生的 `is_elderly`、`risk_capacity_label`，
        以及 `internal_single_product_warning_ratio` → 实为 `warning_ratio` 的计算结果）。
        列表类字段均做浅拷贝，调用方修改不会回写模型。

        副作用：无。被 `pipeline.node_optimize` 写入 `portfolio_rounds`、
        被 `versioning.build_payload` 写入版本快照。
        """
        return {
            "client_id": self.client_id,
            "display_name": self.display_name,
            "age": self.age,
            "is_elderly": self.is_elderly,
            "risk_capacity": self.risk_capacity,
            "risk_capacity_label": risk_label(self.risk_capacity),
            "investment_horizon_years": self.investment_horizon_years,
            "liquidity_floor_ratio": self.liquidity_floor_ratio,
            "max_single_product_ratio": self.max_single_product_ratio,
            "max_single_class_ratio": self.max_single_class_ratio,
            "max_single_issuer_ratio": self.max_single_issuer_ratio,
            "internal_single_product_warning_ratio": self.warning_ratio,
            "investable_amount": self.investable_amount,
            "prohibited_categories": list(self.prohibited_categories),
            "prohibited_product_ids": list(self.prohibited_product_ids),
            "experienced_categories": list(self.experienced_categories),
            "qualified_investor": self.qualified_investor,
            "currency": self.currency,
            "tax_advantaged_quota": self.tax_advantaged_quota,
            "dual_record_completed": self.dual_record_completed,
            "annual_fee_budget_ratio": self.annual_fee_budget_ratio,
            "max_drawdown_tolerance": self.max_drawdown_tolerance,
            "return_target": self.return_target,
            "risk_questionnaire_score": self.risk_questionnaire_score,
        }


class Product(BaseModel):
    """产品要素表（全部为虚构产品）。

    职责：描述单个可投产品的要素，是筛选（准入判定）、优化（收益/风险/流动性打分）
    与建议书（持仓表）共同的输入。**注意**：仓库不得出现真实机构/产品名称，`issuer` 亦为虚构。

    关键字段：`risk_level`（1-5，与客户 `risk_capacity` 比较）、`horizon_years`（0 表示无固定期限）、
    `min_investment`、`expected_return`、`volatility`、`liquidity_ratio`、`fee_rate`、
    `is_derivative`、`requires_experience`、`qualified_investor_only`、`tax_advantaged`、
    `principal_protected`。

    被谁使用：`src/dataset.py`（装载）、`src/constraints.py`（准入与组合约束）、
    `src/agents/product_screening.py`、`src/agents/portfolio_optimizer.py`、`src/optimizer.py`。
    """

    model_config = ConfigDict(extra="forbid")

    product_id: str
    name: str
    asset_class: str
    risk_level: int = Field(ge=1, le=5)
    horizon_years: float = Field(ge=0, description="产品期限（年），0 表示无固定期限")
    min_investment: float = Field(ge=0, description="起投金额（元）")
    expected_return: float = Field(description="示例预期年化收益")
    volatility: float = Field(ge=0, description="示例年化波动率")
    liquidity_ratio: float = Field(ge=0, le=1, description="可即时变现比例")
    fee_rate: float = Field(ge=0, description="综合费率")
    issuer: str = Field(description="虚构发行主体")
    currency: str = "CNY"
    is_derivative: bool = False
    requires_experience: list[str] = Field(default_factory=list)
    qualified_investor_only: bool = False
    tax_advantaged: bool = False
    principal_protected: bool = False

    def snapshot(self) -> dict[str, Any]:
        """产品要素快照：写入建议版本链，防止事后要素变更污染历史建议。

        返回：扁平字典；在字段值之外额外附带 `risk_level_label`（由 `risk_label()` 生成），
        便于版本链与报告直接展示中文风险等级。

        副作用：无。被 `versioning.build_payload` 写入版本快照的
        `product_snapshots` 部分（测试 `test_version_snapshot_contains_constraint_and_rule_evidence`
        会断言该键存在）。
        """
        return {
            "product_id": self.product_id,
            "name": self.name,
            "asset_class": self.asset_class,
            "risk_level": self.risk_level,
            "risk_level_label": risk_label(self.risk_level),
            "horizon_years": self.horizon_years,
            "min_investment": self.min_investment,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "liquidity_ratio": self.liquidity_ratio,
            "fee_rate": self.fee_rate,
            "issuer": self.issuer,
            "currency": self.currency,
            "is_derivative": self.is_derivative,
            "tax_advantaged": self.tax_advantaged,
        }


class Violation(BaseModel):
    """一条约束违反 / 规则命中记录（约束与适当性规则共用同一结构）。

    职责：统一「硬约束违反」与「适当性规则命中」的表达，使拦截理由可复核、可引述。

    关键字段：
    - `rule_id`：规则号（硬约束为 `C-*`、适当性为 `S-*`），是人工豁免与断言的主要标识；
    - `severity`：`"block"`（拦截）或 `"warn"`（仅揭示）；
    - `detail` / `basis`：违规事实与规则依据（供监管问询）；
    - `product_ids`：涉及的产品；
    - `code`：约束侧用于机器判别的代码（如 `CONCENTRATION_CODES`、`C-LIQUIDITY`）。

    被谁使用：`GateDecision.blocks` / `warns`、`src/constraints.check_portfolio`、
    `src/suitability/rules.py`、`eval/run_eval.py`（按 `code` 过滤集中度类结论）。
    """

    rule_id: str
    severity: Severity = "block"
    detail: str
    basis: str = ""
    product_ids: list[str] = Field(default_factory=list)
    code: str = ""

    def key(self) -> tuple[str, str]:
        """稳定排序键：先规则号、后涉及产品。

        返回：`(rule_id, 逗号连接的已排序 product_ids)`；对产品号先排序再拼接，
        因此与传入顺序无关，可用于跨轮次/跨引擎的确定性去重与比对。
        """
        return (self.rule_id, ",".join(sorted(self.product_ids)))


class Portfolio(BaseModel):
    """候选组合：持仓权重 + 产品要素快照 + 预期指标。

    职责：承载「一份配置方案」，并提供权重聚合类派生计算，是所有下游判定的输入。
    设计要点：`products` 存的是**要素快照**而非产品号引用，
    因此产品表事后变更不会影响历史结论。

    关键字段：`weights`（产品号 -> 权重）、`products`（产品号 -> `Product` 快照）、
    `cash_weight`（现金权重）、`metrics`（预期收益/波动等指标）、`rationale`（权衡说明）。

    被谁使用：`src/optimizer.py`（求解产出）、`src/constraints.check_portfolio`（约束判定）、
    `src/suitability/`（规则判定）、`src/pipeline.py`（共享状态）、
    `eval/run_eval.py`（构造对抗探针组合）。
    """

    weights: dict[str, float] = Field(default_factory=dict)
    products: dict[str, Product] = Field(default_factory=dict, description="持仓产品的要素快照")
    cash_weight: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)

    # ------------------------------------------------------------------
    def held_ids(self) -> list[str]:
        """持有权重 > 0 的产品，按 product_id 排序（保证确定性）。

        返回：排序后的产品号列表；权重恰为 0（或负）的不计入。
        几乎所有聚合计算都先经过本方法，这是组合层面确定性的基础。
        """
        return sorted(pid for pid, w in self.weights.items() if w > 0)

    def total_weight(self) -> float:
        """产品权重合计（不含现金）。

        返回：`held_ids()` 各产品权重之和，四舍五入到 12 位小数以规避浮点误差。
        """
        return round(sum(self.weights.get(pid, 0.0) for pid in self.held_ids()), 12)

    def class_weights(self) -> dict[str, float]:
        """按资产类别汇总权重（含现金）。

        返回：资产类别 -> 权重；现金以固定键 `"现金"` 单列（`cash_weight > 0` 时）。
        供单一类别集中度约束与优化器目标使用。
        """
        result: dict[str, float] = {}
        for pid in self.held_ids():
            product = self.products[pid]
            result[product.asset_class] = round(result.get(product.asset_class, 0.0) + self.weights[pid], 12)
        if self.cash_weight > 0:
            result["现金"] = round(result.get("现金", 0.0) + self.cash_weight, 12)
        return result

    def issuer_weights(self) -> dict[str, float]:
        """按发行主体汇总权重。

        返回：发行人 -> 权重（不含现金：现金没有发行主体）。
        供同一发行人集中度上限约束使用。
        """
        result: dict[str, float] = {}
        for pid in self.held_ids():
            issuer = self.products[pid].issuer
            result[issuer] = round(result.get(issuer, 0.0) + self.weights[pid], 12)
        return result

    def liquid_ratio(self) -> float:
        """流动性资产占比（现金视为 100% 流动）。

        返回：`cash_weight + Σ(权重 × 产品 liquidity_ratio)`，保留 12 位小数。
        与客户 `liquidity_floor_ratio` 比较，判定流动性下限是否满足。
        """
        liquid = self.cash_weight
        for pid in self.held_ids():
            liquid += self.weights[pid] * self.products[pid].liquidity_ratio
        return round(liquid, 12)

    def holding_amounts(self, investable_amount: float) -> dict[str, float]:
        """按可投金额折算各产品持有金额（元）。

        参数：
            investable_amount：可投金额（元），通常取 `ClientProfile.investable_amount`。

        返回：产品号 -> 金额，保留 2 位小数（分）；品种与 `held_ids()` 一致，不含现金。
        """
        return {
            pid: round(self.weights[pid] * investable_amount, 2)
            for pid in self.held_ids()
        }

    def to_row(self) -> list[dict[str, Any]]:
        """转成表格行，便于 demo / 建议书直接渲染。

        返回：按 `held_ids()` 顺序的行列表（`product_id` / `name` / `asset_class` /
        `risk_level` / `weight`）；若 `cash_weight > 0`，在末尾追加一行
        「现金/活期留存」，其 `product_id` 固定为 `"CASH"`、`risk_level` 记为 1。
        """
        rows: list[dict[str, Any]] = []
        for pid in self.held_ids():
            product = self.products[pid]
            rows.append(
                {
                    "product_id": pid,
                    "name": product.name,
                    "asset_class": product.asset_class,
                    "risk_level": product.risk_level,
                    "weight": self.weights[pid],
                }
            )
        if self.cash_weight > 0:
            rows.append(
                {
                    "product_id": "CASH",
                    "name": "现金/活期留存",
                    "asset_class": "现金",
                    "risk_level": 1,
                    "weight": self.cash_weight,
                }
            )
        return rows


class Exclusion(BaseModel):
    """候选池剔除记录：产品 + 剔除原因码 + 中文说明。

    职责：让每一个被剔除的产品都有可逐条向客户解释的理由（合规留痕要求）。

    关键字段：`product_id` / `product_name` 标识产品，`reasons` 为原因**码**列表
    （可机器聚合，`src/constraints.PRODUCT_LEVEL_CODES` 即其取值域），
    `detail` 为面向人的中文说明。

    被谁使用：`ScreeningResult.excluded`、`src/agents/product_screening.py`（生成说明）。
    """

    product_id: str
    product_name: str
    reasons: list[str] = Field(default_factory=list)
    detail: str = ""


class ScreeningResult(BaseModel):
    """产品筛选结果。

    关键字段：`included`（入选产品号）、`excluded`（剔除记录）、`universe_size`（全市场产品数）、
    `round_index`（第几轮筛选，0 为初始轮，打回重配时递增）。

    被谁使用：`src/constraints.screen_products`（产出）、`src/agents/product_screening.py`、
    `src/pipeline.node_screen` / `node_optimize`、`src/narrative.py`（建议书披露筛选口径）。
    """

    client_id: str
    round_index: int = 0
    included: list[str] = Field(default_factory=list)
    excluded: list[Exclusion] = Field(default_factory=list)
    universe_size: int = 0

    @property
    def included_ratio(self) -> float:
        """候选池留存率。

        返回：`len(included) / universe_size`，保留 6 位小数；
        `universe_size` 为 0 时返回 0.0（避免除零）。
        """
        return round(len(self.included) / self.universe_size, 6) if self.universe_size else 0.0


class GateDecision(BaseModel):
    """适当性闸门结论。

    职责：全流程的**唯一否决出口**所产出的结构化结论，决定条件路由走向。

    关键字段（directive 的三态是路由依据）：
    - `passed`：是否无 block 级命中（注：`reject` 时也可能为 False，二者需分别判断）；
    - `directive`：`"pass"` → 人工闸门；`"reoptimize"` → 打回重配；`"reject"` → 转人工/拒绝；
    - `blocks` / `warns`：拦截项与揭示项（豁免后的 block 会被降级挪入 `warns`，
      并在 `detail` 前加「【经人工豁免】」）；
    - `veto_rules`：命中即拒、**不可通过收紧约束修复**的规则（如可行域为空）；
    - `exempted_rules`：经人工豁免的 block 级规则（豁免必须留痕并在建议书披露）；
    - `escalated`：打回重配次数用尽后转人工；
    - `tighten`：打回时下发的约束收紧指令（`TightenSpec.to_dict()` 的字典）；
    - `round_index`：本轮是第几轮复核（0 起）。

    被谁使用：`src/suitability/engine.py:SuitabilityGate.review`（产出）、
    `src/pipeline.node_suitability` / `node_escalate` / `node_human_gate` /
    `route_after_suitability`、`src/narrative.py`、`src/versioning.py`。
    """

    round_index: int = 0
    passed: bool = False
    directive: Directive = "pass"
    blocks: list[Violation] = Field(default_factory=list)
    warns: list[Violation] = Field(default_factory=list)
    veto_rules: list[str] = Field(default_factory=list, description="命中即拒、不可通过收紧约束修复的规则")
    exempted_rules: list[str] = Field(default_factory=list, description="经人工豁免的 block 级规则")
    escalated: bool = Field(default=False, description="打回重配次数用尽后转人工")
    tighten: dict[str, Any] = Field(default_factory=dict, description="打回重配时下发的约束收紧指令")
    comment: str = ""


class HumanReview(BaseModel):
    """人工（理财经理）确认记录。

    职责：把「是否必须人工确认 / 谁确认的 / 结论是什么」固化成可审计留痕，
    并直接决定终态（见 `pipeline.node_human_gate` 的状态映射）。

    关键字段：`required`（是否触发确认）、`reasons`（全部触发原因，可多条）、
    `decision` 五态见下、`operator`、`degraded`（是否为非交互降级放行）、
    `exempted_rules`（本次确认涉及的豁免规则）、`note`、`decided_at`。

    `decision` 取值：`approved`（交互同意）｜`auto_approved`（--auto 演示放行）｜
    `auto_degraded`（非交互环境自动降级放行）｜`rejected`（人工否决）｜`not_required`（无需确认）。

    被谁使用：`src/pipeline.decide_human_review`（产出）、`node_human_gate`、`node_narrative`
    （写入建议书「人工确认记录」章节）、`src/agents/advisor_narrative.py`、`src/versioning.py`。
    """

    required: bool = False
    reasons: list[str] = Field(default_factory=list)
    decision: str = "not_required"  # approved | auto_approved | auto_degraded | rejected | not_required
    operator: str = "system"
    degraded: bool = False
    exempted_rules: list[str] = Field(default_factory=list)
    note: str = ""
    decided_at: str = ""


class ScenarioImpact(BaseModel):
    """单情景压力测试结果。

    职责：记录「一个情景打在组合上会发生什么」，是风险揭示的量化依据。

    关键字段：`portfolio_impact`（组合层面冲击）、`estimated_drawdown`（估算回撤，
    与客户 `max_drawdown_tolerance` 比较）、`exceeds_tolerance`（是否超容忍度，即是否触发提示）、
    `by_asset_class` / `by_product`（下钻明细）。

    被谁使用：`StressReport.scenarios`、`src/stress.py`（产出）。
    """

    scenario_id: str
    name: str
    description: str
    portfolio_impact: float
    estimated_drawdown: float
    exceeds_tolerance: bool
    by_asset_class: dict[str, float] = Field(default_factory=dict)
    by_product: dict[str, float] = Field(default_factory=dict)


class StressReport(BaseModel):
    """压力测试报告（多情景）。

    关键字段：`scenarios`（各情景结果）、`worst_scenario_id` / `worst_impact`（最差情景）、
    `formula`（所用确定性公式说明，保证可复核）。

    被谁使用：`src/stress.py`（产出）、`src/agents/advisor_narrative.py`、
    `AdvisorNarrativeAgent.run` 内断言情景数、`eval/run_eval.py` 统计 `scenario_count >= 3`。
    """

    scenarios: list[ScenarioImpact] = Field(default_factory=list)
    worst_scenario_id: str = ""
    worst_impact: float = 0.0
    formula: str = ""

    @property
    def scenario_count(self) -> int:
        """情景数量。

        返回：`len(scenarios)`；评估指标 stress_coverage 要求每份方案 >= 3。
        """
        return len(self.scenarios)


class CounterfactualVariant(BaseModel):
    """单条反事实变体。

    职责：回答「如果改一条约束，结果会怎么变」——差异全部来自**重新求解**的确定性计算，
    而非模型臆测。

    关键字段：`question`（反事实设问）、`constraint_delta`（相对基线的约束改动）、
    `feasible`（改动后是否仍有可行解）、`status`（该变体的复核结论）、
    `products_added` / `products_removed` / `weight_changes` / `metric_changes` / `rule_changes`
    （结构化差异）、`explanation`（说明）。

    被谁使用：`CounterfactualReport.variants`、`src/counterfactual.py`（产出）。
    """

    variant_id: str
    question: str
    constraint_delta: dict[str, Any] = Field(default_factory=dict)
    feasible: bool = True
    status: str = ""
    products_added: list[str] = Field(default_factory=list)
    products_removed: list[str] = Field(default_factory=list)
    weight_changes: dict[str, float] = Field(default_factory=dict)
    metric_changes: dict[str, float] = Field(default_factory=dict)
    rule_changes: dict[str, list[str]] = Field(default_factory=dict)
    explanation: str = ""

    def is_empty(self) -> bool:
        """差异是否为空（无产品增删、无权重变化、无指标变化）。

        返回：True 表示该变体没有产生任何有意义差异；判定阈值为 `abs(变化量) > 1e-9`
        （小于等于该值视为无变化）。
        注：实际实现中 `not self.feasible`（不可行）也算作**非空**——不可行本身就是有效结论，
        因此会被 `CounterfactualReport.coverage` 计入覆盖率。
        """
        return not (
            self.products_added
            or self.products_removed
            or any(abs(v) > 1e-9 for v in self.weight_changes.values())
            or any(abs(v) > 1e-9 for v in self.metric_changes.values())
            or not self.feasible
        )


class CounterfactualReport(BaseModel):
    """反事实解释报告。

    关键字段：`baseline_version` / `baseline_status`（基线版本标识）、`variants`（各变体）。

    被谁使用：`src/counterfactual.py`（产出）、`src/agents/advisor_narrative.py`、
    `eval/run_eval.py`（要求 `coverage > 0`）。
    """

    baseline_version: str = ""
    baseline_status: str = ""
    variants: list[CounterfactualVariant] = Field(default_factory=list)

    @property
    def coverage(self) -> float:
        """非空变体占比。

        返回：`非空变体数 / 变体总数`，保留 6 位小数；`variants` 为空时返回 0.0。
        「非空」的判定见 `CounterfactualVariant.is_empty`。
        """
        if not self.variants:
            return 0.0
        non_empty = sum(1 for v in self.variants if not v.is_empty())
        return round(non_empty / len(self.variants), 6)


class AdviceRecord(BaseModel):
    """最终投顾建议（结构化部分，正文另见 narrative）。

    职责：把一次运行的全部结论**聚合为单一可交付对象**：配置 + 筛选 + 闸门 +
    人工确认 + 压力测试 + 反事实 + 正文 + 要素清单，是「结构化建议」与「建议书正文」的对应物。

    关键字段：`version` / `status` / `directive` / `engine`、`portfolio`、`screening`、
    `gate`、`human_review`、`stress`、`counterfactual`、`narrative`、`elements`、`created_at`。
    其中 `status` 与流水线终态保持一致（`pipeline.node_narrative` 把 `state["status"]` 传入）。

    被谁使用：`src/agents/advisor_narrative.py`（构造）、`src/pipeline.node_narrative`
    （写入共享状态）、`src/demo.py`（展示）。
    """

    client_id: str
    run_id: str
    version: int = 0
    status: str = "draft"
    directive: str = "pass"
    engine: str = ""
    portfolio: Portfolio = Field(default_factory=Portfolio)
    screening: ScreeningResult | None = None
    gate: GateDecision | None = None
    human_review: HumanReview = Field(default_factory=HumanReview)
    stress: StressReport | None = None
    counterfactual: CounterfactualReport | None = None
    narrative: str = ""
    elements: dict[str, bool] = Field(default_factory=dict)
    created_at: str = ""
