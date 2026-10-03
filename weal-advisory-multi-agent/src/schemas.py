"""领域数据模型。

本模块只负责**结构化建模**与**轻量自洽校验**，不含任何业务判断：
- 「能不能配」→ `src/constraints.py`（确定性硬约束求解器）
- 「该不该拦」→ `src/suitability/rules.py`（适当性规则库）
这样保证约束与规则都是可单测、可重放的纯函数，而模型只承担数据载体职责。
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
    """风险等级 -> 中文标签。"""
    return RISK_LEVEL_LABELS.get(level, f"R{level}-未知")


class ClientProfile(BaseModel):
    """客户档案：硬约束 + 软偏好 + 适当性相关属性。"""

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
        """去除空白项并保持稳定顺序，避免同一语义出现两种快照。"""
        return [item.strip() for item in value if item and item.strip()]

    # ------------------------------------------------------------------
    # 派生属性
    # ------------------------------------------------------------------
    @property
    def is_elderly(self) -> bool:
        """高龄客户判定（示例阈值：年满 65 周岁）。"""
        return self.age >= 65

    @property
    def warning_ratio(self) -> float:
        """内控单一产品集中度预警线（未显式配置时取集中度上限的 80%）。"""
        if self.internal_single_product_warning_ratio is not None:
            return float(self.internal_single_product_warning_ratio)
        return round(self.max_single_product_ratio * 0.8, 12)

    def constraint_snapshot(self) -> dict[str, Any]:
        """客户硬约束快照：写入建议版本链，用于事后回溯「当时按什么约束配的」。"""
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
    """产品要素表（全部为虚构产品）。"""

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
        """产品要素快照：写入建议版本链，防止事后要素变更污染历史建议。"""
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
    """一条约束违反 / 规则命中记录（约束与适当性规则共用同一结构）。"""

    rule_id: str
    severity: Severity = "block"
    detail: str
    basis: str = ""
    product_ids: list[str] = Field(default_factory=list)
    code: str = ""

    def key(self) -> tuple[str, str]:
        """稳定排序键：先规则号、后涉及产品。"""
        return (self.rule_id, ",".join(sorted(self.product_ids)))


class Portfolio(BaseModel):
    """候选组合：持仓权重 + 产品要素快照 + 预期指标。"""

    weights: dict[str, float] = Field(default_factory=dict)
    products: dict[str, Product] = Field(default_factory=dict, description="持仓产品的要素快照")
    cash_weight: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)

    # ------------------------------------------------------------------
    def held_ids(self) -> list[str]:
        """持有权重 > 0 的产品，按 product_id 排序（保证确定性）。"""
        return sorted(pid for pid, w in self.weights.items() if w > 0)

    def total_weight(self) -> float:
        """产品权重合计（不含现金）。"""
        return round(sum(self.weights.get(pid, 0.0) for pid in self.held_ids()), 12)

    def class_weights(self) -> dict[str, float]:
        """按资产类别汇总权重（含现金）。"""
        result: dict[str, float] = {}
        for pid in self.held_ids():
            product = self.products[pid]
            result[product.asset_class] = round(result.get(product.asset_class, 0.0) + self.weights[pid], 12)
        if self.cash_weight > 0:
            result["现金"] = round(result.get("现金", 0.0) + self.cash_weight, 12)
        return result

    def issuer_weights(self) -> dict[str, float]:
        """按发行主体汇总权重。"""
        result: dict[str, float] = {}
        for pid in self.held_ids():
            issuer = self.products[pid].issuer
            result[issuer] = round(result.get(issuer, 0.0) + self.weights[pid], 12)
        return result

    def liquid_ratio(self) -> float:
        """流动性资产占比（现金视为 100% 流动）。"""
        liquid = self.cash_weight
        for pid in self.held_ids():
            liquid += self.weights[pid] * self.products[pid].liquidity_ratio
        return round(liquid, 12)

    def holding_amounts(self, investable_amount: float) -> dict[str, float]:
        """按可投金额折算各产品持有金额（元）。"""
        return {
            pid: round(self.weights[pid] * investable_amount, 2)
            for pid in self.held_ids()
        }

    def to_row(self) -> list[dict[str, Any]]:
        """转成表格行，便于 demo / 建议书直接渲染。"""
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
    """候选池剔除记录：产品 + 剔除原因码 + 中文说明。"""

    product_id: str
    product_name: str
    reasons: list[str] = Field(default_factory=list)
    detail: str = ""


class ScreeningResult(BaseModel):
    """产品筛选结果。"""

    client_id: str
    round_index: int = 0
    included: list[str] = Field(default_factory=list)
    excluded: list[Exclusion] = Field(default_factory=list)
    universe_size: int = 0

    @property
    def included_ratio(self) -> float:
        return round(len(self.included) / self.universe_size, 6) if self.universe_size else 0.0


class GateDecision(BaseModel):
    """适当性闸门结论。"""

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
    """人工（理财经理）确认记录。"""

    required: bool = False
    reasons: list[str] = Field(default_factory=list)
    decision: str = "not_required"  # approved | auto_approved | auto_degraded | rejected | not_required
    operator: str = "system"
    degraded: bool = False
    exempted_rules: list[str] = Field(default_factory=list)
    note: str = ""
    decided_at: str = ""


class ScenarioImpact(BaseModel):
    """单情景压力测试结果。"""

    scenario_id: str
    name: str
    description: str
    portfolio_impact: float
    estimated_drawdown: float
    exceeds_tolerance: bool
    by_asset_class: dict[str, float] = Field(default_factory=dict)
    by_product: dict[str, float] = Field(default_factory=dict)


class StressReport(BaseModel):
    """压力测试报告（多情景）。"""

    scenarios: list[ScenarioImpact] = Field(default_factory=list)
    worst_scenario_id: str = ""
    worst_impact: float = 0.0
    formula: str = ""

    @property
    def scenario_count(self) -> int:
        return len(self.scenarios)


class CounterfactualVariant(BaseModel):
    """单条反事实变体。"""

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
        """差异是否为空（无产品增删、无权重变化、无指标变化）。"""
        return not (
            self.products_added
            or self.products_removed
            or any(abs(v) > 1e-9 for v in self.weight_changes.values())
            or any(abs(v) > 1e-9 for v in self.metric_changes.values())
            or not self.feasible
        )


class CounterfactualReport(BaseModel):
    """反事实解释报告。"""

    baseline_version: str = ""
    baseline_status: str = ""
    variants: list[CounterfactualVariant] = Field(default_factory=list)

    @property
    def coverage(self) -> float:
        """非空变体占比。"""
        if not self.variants:
            return 0.0
        non_empty = sum(1 for v in self.variants if not v.is_empty())
        return round(non_empty / len(self.variants), 6)


class AdviceRecord(BaseModel):
    """最终投顾建议（结构化部分，正文另见 narrative）。"""

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
