"""PortfolioOptimizerAgent —— 候选组合构建（可行域内的多目标权衡）。

所属层次
--------
Agent 层（`src/agents/`），流水线第 3 环（`optimize` 节点）。
被 `src/pipeline.py` 的 `AdvisoryPipeline.node_optimize` 调用；
每次闸门打回重配都会再进一次本 Agent（约束已收紧，可行域更小）。

解决什么问题
------------
在"已经合规可选"的候选池内回答"各买多少"：不是让模型拍权重，
而是用确定性算法做多目标折中，并保证产出的组合**约束违反数为 0**。

职责边界
--------
本 Agent **不做合规判断**（那是 SuitabilityOfficerAgent 的硬闸门职责），
也**不允许触碰硬约束**：它拿到的候选池已经是可行域，输出的组合由
`src/optimizer.py` 的确定性算法求解，并且内部会跑可行性修复循环，
保证「被接受组合的约束违反数恒为 0」。

四目标折中：期望收益 ↑、波动 ↓、流动性 ↑、集中度 ↓；
风险惩罚系数随客户风险承受能力下降而上升，因此保守客户天然配置更稳。

与其他 Agent 的互斥关系
-----------------------
- 只写 `portfolio` / `portfolio_note` / `binding` / `product_scores`
  （`can_write_state`），**不写** `candidates`（候选池归筛选 Agent，
  本 Agent 只读）、**不写** `gate`（合规结论归复核 Agent）、
  **不写** `client`（约束由画像 Agent 与闸门收紧指令共同决定）；
- 没有 `suitability:judge` / `suitability:veto` 权限，因此**无法**自我判定合规，
  也无法拒绝流程——组合是否放行由闸门说了算，这正是"构建与复核分离"的设计意图；
- 模型只用于生成权衡说明（`optimize.compose_rationale`），
  权重与指标全部来自确定性计算，模型改不了任何一个数字。

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

#: 本 Agent 的能力契约
#: `optimize.read` 只有读权限语义，`optimize:compute` 才允许求解权重；
#: `can_write_state` 四个键与其它 Agent 不相交。
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
    """组合构建 Agent。

    职责：候选池打分 → 求解权重（含可行性修复）→ 识别紧约束 → 生成权衡说明，
    写出 `portfolio` / `portfolio_note` / `binding` / `product_scores`。

    边界：不做合规判断、不改硬约束、不改候选池、不预测收益走势；
    无 `suitability:*` 权限，所以它给出的组合**必须**经过闸门复核才算通过
    （这也是"生成即通过"这类风险被结构性挡住的原因）。
    """

    def __init__(self, **kwargs: Any) -> None:
        """以固定契约构造组合构建 Agent。

        参数：
            **kwargs: 透传给 `BaseAgent.__init__`（常用 `llm=`、`tracer=`）。

        返回：None。
        副作用/异常：无。
        """
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
        """在候选池内求解组合权重。

        参数（keyword-only）：
            client: **生效**客户档案（可能已被闸门 `TightenSpec` 收紧过，
                求解器直接以它作为约束来源）。
            candidates: 可投产品号集合（来自筛选 Agent）；本方法会先去重再排序，
                保证同样的输入得到同样的权重（可复现）。
            products: 全量产品映射 `{product_id: Product}`。
                注：这里传入的是全量池，**候选范围由 `candidates` 限定**，
                求解器内部只对候选池取数；全量映射是为了让类别/发行人
                等计算能查到产品元数据。
            round_index: 当前重配轮次（0 为首轮），用于 trace 留痕。

        返回：
            经 `guard_output` 校验的字典，含四个键：
            - `portfolio`：`Portfolio`（权重、现金权重、指标、求解元信息），
              保证硬约束违反数为 0；
            - `portfolio_note`：组合权衡说明文案（`rationale`）；
            - `binding`：紧约束原因码列表（哪些上限正贴边），用于解释权衡边界；
            - `product_scores`：候选产品的多目标打分明细。

        副作用：
            - `optimize.compose_rationale` 会触发 LLM（无 Key 时自动 mock）；
            - 写一条 `node="optimize"` 的 trace，`extra` 里带持仓只数与紧约束列表。

        异常：
            PermissionError: 越权调用工具或越权写共享状态。
            KeyError: `candidates` 里出现 `products` 中不存在的产品号。
        """
        stopwatch = Stopwatch()
        tool_calls: list[str] = []
        # 去重 + 排序：让权重求解对输入顺序不敏感（可复现性的前提）
        candidate_ids = tuple(sorted(set(candidates)))

        # 打分只针对候选池（用字典推导裁掉池外产品）
        scores = self.call_tool("optimize.score_products", client=client, products={pid: products[pid] for pid in candidate_ids})
        tool_calls.append("optimize.score_products")

        # 权重求解：内部含确定性可行性修复循环，被接受的组合违反数恒为 0
        portfolio: Portfolio = self.call_tool(
            "optimize.solve_weights", client=client, candidates=candidate_ids, products=products
        )
        tool_calls.append("optimize.solve_weights")

        # 紧约束识别：指出哪些上限"贴边"，供建议书向客户解释权衡代价
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
