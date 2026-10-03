"""PortfolioOptimizerAgent —— 候选组合构建（可行域内的多目标权衡）。

职责边界
--------
本 Agent **不做合规判断**（那是 SuitabilityOfficerAgent 的硬闸门职责），
也**不允许触碰硬约束**：它拿到的候选池已经是可行域，输出的组合由
`src/optimizer.py` 的确定性算法求解，并且内部会跑可行性修复循环，
保证「被接受组合的约束违反数恒为 0」。

四目标折中：期望收益 ↑、波动 ↓、流动性 ↑、集中度 ↓；
风险惩罚系数随客户风险承受能力下降而上升，因此保守客户天然配置更稳。

工具白名单：`optimize.score_products` / `optimize.solve_weights` / `optimize.compose_rationale`
权限集合：`optimize:compute`、`optimize:read`、`narrative:write`、`trace:write`
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..optimizer import binding_constraints
from ..schemas import ClientProfile, Portfolio, Product
from .base import BaseAgent, Stopwatch
from .tools import (
    PERM_NARRATIVE_WRITE,
    PERM_OPTIMIZE_COMPUTE,
    PERM_OPTIMIZE_READ,
    PERM_TRACE_WRITE,
    AgentSpec,
)

SYSTEM_PROMPT = """你是财富管理机构的组合构建引擎，只在给定候选池内做权重分配与多目标权衡。

你必须遵守：
1. 候选池之外的产品一律不得纳入组合；
2. 不得突破客户的任何硬约束（风险等级、期限、集中度、流动性、起投金额）；
3. 必须在期望收益、波动、流动性、集中度四个目标之间做显式折中，并说明理由；
4. 不得预测市场、不得给出收益承诺、不得推荐任何输入之外的产品。

输出字段：rationale（组合权衡说明）。"""

SPEC = AgentSpec(
    name="PortfolioOptimizerAgent",
    role="组合构建与多目标权衡",
    system_prompt=SYSTEM_PROMPT,
    tools=("optimize.score_products", "optimize.solve_weights", "optimize.compose_rationale"),
    permissions=frozenset(
        {PERM_OPTIMIZE_COMPUTE, PERM_OPTIMIZE_READ, PERM_NARRATIVE_WRITE, PERM_TRACE_WRITE}
    ),
    can_write_state=("portfolio", "portfolio_note", "binding", "product_scores"),
)


class PortfolioOptimizerAgent(BaseAgent):
    """组合构建 Agent。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(SPEC, **kwargs)

    # ------------------------------------------------------------------
    def run(
        self,
        *,
        client: ClientProfile,
        candidates: Iterable[str],
        products: Mapping[str, Product],
        round_index: int = 0,
    ) -> dict[str, Any]:
        """在候选池内求解组合权重。"""
        stopwatch = Stopwatch()
        tool_calls: list[str] = []
        candidate_ids = tuple(sorted(set(candidates)))

        scores = self.call_tool("optimize.score_products", client=client, products={pid: products[pid] for pid in candidate_ids})
        tool_calls.append("optimize.score_products")

        portfolio: Portfolio = self.call_tool(
            "optimize.solve_weights", client=client, candidates=candidate_ids, products=products
        )
        tool_calls.append("optimize.solve_weights")

        binding = binding_constraints(portfolio, client)

        text = self.call_tool(
            "optimize.compose_rationale",
            llm=self.llm,
            task="optimizer_rationale",
            context={
                "metrics": portfolio.metrics,
                "binding": binding,
                "class_targets": portfolio.class_weights(),
            },
        )
        tool_calls.append("optimize.compose_rationale")

        output = {
            "portfolio": portfolio,
            "portfolio_note": str(text["data"].get("rationale", "")),
            "binding": binding,
            "product_scores": scores,
        }
        self.trace(
            "optimize",
            payload_in={"client_id": client.client_id, "round": round_index, "candidates": list(candidate_ids)},
            payload_out={"weights": portfolio.weights, "cash_weight": portfolio.cash_weight, "metrics": portfolio.metrics},
            tool_calls=tool_calls,
            latency_ms=stopwatch.ms(),
            extra={"holding_count": len(portfolio.held_ids()), "binding": binding},
        )
        return self.guard_output(output)
