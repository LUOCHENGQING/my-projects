"""Agent 工具白名单与权限集合。

每个 Agent 有：
- **独立 system prompt**（角色与输出边界）
- **独立工具白名单**（`tools`）：只能调用白名单内的工具
- **独立权限集合**（`permissions`）：工具本身还带权限标签，双重校验
- **可写 state 键白名单**（`can_write_state`）：最小权限原则，越权写共享状态直接报错

这样做的目的是让"谁能做什么"变成可断言的数据，而不是散落在代码里的约定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from ..constraints import TightenSpec, screen_products
from ..counterfactual import analyze_counterfactual
from ..dataset import level_label, score_to_level
from ..optimizer import binding_constraints, build_portfolio, product_score
from ..schemas import (
    ClientProfile,
    GateDecision,
    Portfolio,
    Product,
    ScreeningResult,
    StressReport,
)
from ..stress import run_stress
from ..suitability import RuleContext, RuleEvaluation, SuitabilityGate, build_tighten, evaluate_rules

# ---------------------------------------------------------------------------
# 权限标签
# ---------------------------------------------------------------------------
PERM_PROFILE_READ = "profile:read"
PERM_PROFILE_DERIVE = "profile:derive"
PERM_SCREEN_EXECUTE = "screening:execute"
PERM_SCREEN_READ = "screening:read"
PERM_OPTIMIZE_COMPUTE = "optimize:compute"
PERM_OPTIMIZE_READ = "optimize:read"
PERM_SUIT_JUDGE = "suitability:judge"
PERM_SUIT_VETO = "suitability:veto"
PERM_NARRATIVE_WRITE = "narrative:write"
PERM_NARRATIVE_ANALYZE = "narrative:analyze"
PERM_TRACE_WRITE = "trace:write"


@dataclass(frozen=True)
class Tool:
    """一个可被 Agent 调用的工具。"""

    name: str
    permission: str
    description: str
    handler: Callable[..., Any]


class ToolRegistry:
    """工具注册表：按名字调用，未登记即报错。"""

    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {tool.name: tool for tool in tools}

    def get(self, name: str) -> Tool:
        """取工具；不存在时抛 KeyError。"""
        if name not in self._tools:
            raise KeyError(f"未登记的工具：{name}")
        return self._tools[name]

    def invoke(self, name: str, **kwargs: Any) -> Any:
        """调用工具。"""
        return self.get(name).handler(**kwargs)

    def names(self) -> list[str]:
        """全部工具名（排序）。"""
        return sorted(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


# ---------------------------------------------------------------------------
# 工具实现：全部是对确定性能力的薄封装（工具不是装饰）
# ---------------------------------------------------------------------------
def _tool_read_client(*, client: ClientProfile) -> ClientProfile:
    """读取客户档案（返回深拷贝，避免下游误改缓存）。"""
    return client.model_copy(deep=True)


def _tool_parse_questionnaire(
    *, client: ClientProfile, questionnaire: Mapping[str, Any]
) -> dict[str, Any]:
    """解析风险测评问卷，折算等级。"""
    responses = questionnaire.get("responses", {}).get(client.client_id) or {}
    score = int(sum(responses.values())) if responses else None
    level = score_to_level(score, dict(questionnaire))
    return {
        "client_id": client.client_id,
        "item_count": len(questionnaire.get("items", [])),
        "answers": dict(responses),
        "score": score,
        "level": level,
        "level_label": level_label(level, dict(questionnaire)),
    }


def _tool_derive_constraints(
    *, client: ClientProfile, questionnaire: Mapping[str, Any]
) -> dict[str, Any]:
    """抽取硬约束与软偏好，并按从严原则给出有效风险等级。"""
    parsed = _tool_parse_questionnaire(client=client, questionnaire=questionnaire)
    scored = parsed["level"]
    effective = client.risk_capacity if scored is None else min(client.risk_capacity, int(scored))
    return {
        "questionnaire": parsed,
        "archived_level": client.risk_capacity,
        "effective_level": effective,
        "tightened_by_questionnaire": bool(scored is not None and int(scored) < client.risk_capacity),
        "hard_constraints": {
            "risk_capacity": effective,
            "investment_horizon_years": client.investment_horizon_years,
            "liquidity_floor_ratio": client.liquidity_floor_ratio,
            "max_single_product_ratio": client.max_single_product_ratio,
            "max_single_class_ratio": client.max_single_class_ratio,
            "max_single_issuer_ratio": client.max_single_issuer_ratio,
            "investable_amount": client.investable_amount,
            "prohibited_categories": list(client.prohibited_categories),
            "prohibited_product_ids": list(client.prohibited_product_ids),
            "experienced_categories": list(client.experienced_categories),
            "qualified_investor": client.qualified_investor,
            "currency": client.currency,
            "tax_advantaged_quota": client.tax_advantaged_quota,
        },
        "soft_preferences": {
            "return_target": client.return_target,
            "max_drawdown_tolerance": client.max_drawdown_tolerance,
            "annual_fee_budget_ratio": client.annual_fee_budget_ratio,
        },
        "protection_flags": {
            "is_elderly": client.is_elderly,
            "dual_record_completed": client.dual_record_completed,
            "internal_warning_ratio": client.warning_ratio,
        },
    }


def _tool_filter_universe(
    *, client: ClientProfile, products: Mapping[str, Product], round_index: int = 0
) -> ScreeningResult:
    """在硬约束可行域内筛选候选池。"""
    return screen_products(client, products, round_index)


def _tool_explain_exclusion(*, screening: ScreeningResult, limit: int = 3) -> list[str]:
    """给出主要剔除原因（用于向客户解释）。"""
    return [f"{item.product_name}：{item.detail}" for item in screening.excluded[:limit]]


def _tool_score_products(
    *, client: ClientProfile, products: Mapping[str, Product]
) -> dict[str, float]:
    """按多目标口径给产品打分。"""
    return {pid: round(product_score(products[pid], client), 12) for pid in sorted(products)}


def _tool_solve_weights(
    *, client: ClientProfile, candidates: Iterable[str], products: Mapping[str, Product]
) -> Portfolio:
    """在候选池内求解权重。"""
    return build_portfolio(client, tuple(candidates), products)


def _tool_evaluate_rules(
    *,
    client: ClientProfile,
    portfolio: Portfolio,
    universe: Mapping[str, Product],
    candidates: Iterable[str],
) -> RuleEvaluation:
    """求值全部适当性规则。"""
    context = RuleContext.build(client, portfolio, universe, tuple(candidates))
    return evaluate_rules(context)


def _tool_issue_directive(
    *,
    client: ClientProfile,
    portfolio: Portfolio,
    universe: Mapping[str, Product],
    candidates: Iterable[str],
    round_index: int = 0,
    exempt_rules: Iterable[str] = (),
    max_repair_rounds: int = 2,
) -> GateDecision:
    """出具适当性闸门结论（含打回指令）。"""
    context = RuleContext.build(client, portfolio, universe, tuple(candidates))
    gate = SuitabilityGate(max_repair_rounds=max_repair_rounds)
    return gate.review(context, round_index=round_index, exempt_rules=exempt_rules)


def _tool_build_tighten(
    *,
    client: ClientProfile,
    portfolio: Portfolio,
    universe: Mapping[str, Product],
    candidates: Iterable[str],
    rule_ids: Iterable[str],
) -> dict[str, Any]:
    """由指定规则命中构造约束收紧指令。"""
    context = RuleContext.build(client, portfolio, universe, tuple(candidates))
    evaluation = evaluate_rules(context)
    wanted = set(rule_ids)
    blocks = [v for v in evaluation.blocks if v.rule_id in wanted]
    return build_tighten(blocks, context).to_dict()


def _tool_run_stress(
    *, portfolio: Portfolio, client: ClientProfile, config: Mapping[str, Any]
) -> StressReport:
    """执行情景压力测试。"""
    return run_stress(portfolio, client, config)


def _tool_build_counterfactual(
    *,
    client: ClientProfile,
    portfolio: Portfolio,
    products: Mapping[str, Product],
    rule_ids: Iterable[str],
    version_label: str,
    status: str,
    rule_client: ClientProfile | None = None,
):
    """生成反事实解释报告（基线为生效约束，规则判定针对真实客户档案）。"""
    return analyze_counterfactual(
        client,
        products,
        portfolio,
        baseline_rule_ids=tuple(rule_ids),
        baseline_version=version_label,
        baseline_status=status,
        rule_client=rule_client,
    )


def _tool_compose_text(*, llm: Any, task: str, context: Mapping[str, Any]) -> dict[str, Any]:
    """调用 LLM（或 mock 大脑）生成结构化文案。"""
    result = llm.compose(task, context)
    return {"task": result.task, "mode": result.mode, "data": result.data}


#: 全量工具表
ALL_TOOLS: tuple[Tool, ...] = (
    Tool("profile.read_client", PERM_PROFILE_READ, "读取客户档案", _tool_read_client),
    Tool("profile.parse_questionnaire", PERM_PROFILE_READ, "解析风险测评问卷", _tool_parse_questionnaire),
    Tool("profile.derive_constraints", PERM_PROFILE_DERIVE, "抽取硬约束与软偏好", _tool_derive_constraints),
    Tool("profile.compose_profile_text", PERM_PROFILE_DERIVE, "生成客户画像文案（LLM/mock）", _tool_compose_text),
    Tool("screening.filter_universe", PERM_SCREEN_EXECUTE, "在硬约束内筛选候选池", _tool_filter_universe),
    Tool("screening.explain_exclusion", PERM_SCREEN_READ, "解释产品剔除原因", _tool_explain_exclusion),
    Tool("screening.compose_screening_text", PERM_NARRATIVE_WRITE, "生成筛选说明文案", _tool_compose_text),
    Tool("optimize.score_products", PERM_OPTIMIZE_COMPUTE, "多目标打分", _tool_score_products),
    Tool("optimize.solve_weights", PERM_OPTIMIZE_COMPUTE, "求解组合权重", _tool_solve_weights),
    Tool("optimize.compose_rationale", PERM_NARRATIVE_WRITE, "生成权衡说明文案", _tool_compose_text),
    Tool("suitability.evaluate_rules", PERM_SUIT_JUDGE, "求值适当性规则", _tool_evaluate_rules),
    Tool("suitability.issue_directive", PERM_SUIT_VETO, "出具闸门结论与打回指令", _tool_issue_directive),
    Tool("suitability.build_tighten", PERM_SUIT_JUDGE, "构造约束收紧指令", _tool_build_tighten),
    Tool("suitability.compose_comment", PERM_SUIT_JUDGE, "生成复核意见文案", _tool_compose_text),
    Tool("narrative.run_stress", PERM_NARRATIVE_ANALYZE, "执行情景压力测试", _tool_run_stress),
    Tool("narrative.build_counterfactual", PERM_NARRATIVE_ANALYZE, "生成反事实解释", _tool_build_counterfactual),
    Tool("narrative.compose_text", PERM_NARRATIVE_WRITE, "生成文案（LLM/mock）", _tool_compose_text),
)

#: 默认工具注册表
TOOL_REGISTRY = ToolRegistry(ALL_TOOLS)


@dataclass(frozen=True)
class AgentSpec:
    """Agent 的能力契约。"""

    name: str
    role: str
    system_prompt: str
    tools: tuple[str, ...]
    permissions: frozenset[str]
    can_write_state: tuple[str, ...] = field(default=())

    def describe(self) -> dict[str, Any]:
        """能力清单（demo / README 使用）。"""
        return {
            "name": self.name,
            "role": self.role,
            "tools": list(self.tools),
            "permissions": sorted(self.permissions),
            "can_write_state": list(self.can_write_state),
        }


def tool_names() -> list[str]:
    """全部工具名。"""
    return TOOL_REGISTRY.names()


def registry_tools() -> tuple[Tool, ...]:
    """全部工具对象。"""
    return ALL_TOOLS


def tighten_from_payload(payload: Mapping[str, Any] | None) -> TightenSpec:
    """便捷函数：从 state 中的收紧指令字典还原对象。"""
    return TightenSpec.from_dict(payload)


__all__ = [
    "AgentSpec",
    "ALL_TOOLS",
    "PERM_NARRATIVE_ANALYZE",
    "PERM_NARRATIVE_WRITE",
    "PERM_OPTIMIZE_COMPUTE",
    "PERM_OPTIMIZE_READ",
    "PERM_PROFILE_DERIVE",
    "PERM_PROFILE_READ",
    "PERM_SCREEN_EXECUTE",
    "PERM_SCREEN_READ",
    "PERM_SUIT_JUDGE",
    "PERM_SUIT_VETO",
    "PERM_TRACE_WRITE",
    "TOOL_REGISTRY",
    "Tool",
    "ToolRegistry",
    "binding_constraints",
    "registry_tools",
    "tighten_from_payload",
    "tool_names",
]
