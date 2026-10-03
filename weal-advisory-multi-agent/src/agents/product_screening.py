"""ProductScreeningAgent —— 产品筛选（在硬约束可行域内筛出候选池）。

职责边界
--------
本 Agent **不做收益预测、不做权重分配**，只回答一个问题：
「在这位客户的硬约束下，哪些产品**可以**进入候选池，哪些必须剔除，为什么？」

准入维度：风险等级上限、期限匹配、起投金额与可投金额、币种、禁止项
（类别 / 具体产品 / 衍生品结构）、投资经验、合格投资者准入。
每一个被剔除的产品都会保留**原因码 + 中文说明**，可直接向客户逐条解释。

工具白名单：`screening.filter_universe` / `screening.explain_exclusion` / `screening.compose_screening_text`
权限集合：`screening:execute`、`screening:read`、`trace:write`
"""

from __future__ import annotations

from typing import Any, Mapping

from ..schemas import ClientProfile, Product
from .base import BaseAgent, Stopwatch
from .tools import (
    PERM_NARRATIVE_WRITE,
    PERM_SCREEN_EXECUTE,
    PERM_SCREEN_READ,
    PERM_TRACE_WRITE,
    AgentSpec,
)

SYSTEM_PROMPT = """你是财富管理机构的产品筛选引擎，只负责在客户硬约束可行域内筛选候选产品池。

你必须遵守：
1. 硬约束（风险等级、期限、起投金额、币种、禁止项、投资经验、合格投资者准入）必须逐条校验；
2. 任何不可投产品必须被剔除，并且必须记录可复核的剔除原因，禁止"软性放行"；
3. 不得因为产品收益高就放宽任何准入条件；
4. 不得推荐任何输入产品池之外的产品，也不得编造产品名称或发行主体。

输出字段：note（筛选口径与主要剔除原因的说明）。"""

SPEC = AgentSpec(
    name="ProductScreeningAgent",
    role="产品筛选与剔除留痕",
    system_prompt=SYSTEM_PROMPT,
    tools=(
        "screening.filter_universe",
        "screening.explain_exclusion",
        "screening.compose_screening_text",
    ),
    permissions=frozenset({PERM_SCREEN_EXECUTE, PERM_SCREEN_READ, PERM_NARRATIVE_WRITE, PERM_TRACE_WRITE}),
    can_write_state=("screening", "candidates", "screening_note"),
)


class ProductScreeningAgent(BaseAgent):
    """产品筛选 Agent。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(SPEC, **kwargs)

    # ------------------------------------------------------------------
    def run(
        self,
        *,
        client: ClientProfile,
        products: Mapping[str, Product],
        round_index: int = 0,
    ) -> dict[str, Any]:
        """执行筛选并生成说明。"""
        stopwatch = Stopwatch()
        tool_calls: list[str] = []

        screening = self.call_tool(
            "screening.filter_universe", client=client, products=products, round_index=round_index
        )
        tool_calls.append("screening.filter_universe")

        explanations = self.call_tool("screening.explain_exclusion", screening=screening, limit=3)
        tool_calls.append("screening.explain_exclusion")

        text = self.call_tool(
            "screening.compose_screening_text",
            llm=self.llm,
            task="screening_note",
            context={
                "universe_size": screening.universe_size,
                "included_count": len(screening.included),
                "top_exclusions": explanations,
            },
        )
        tool_calls.append("screening.compose_screening_text")
        note = str(text["data"].get("note", ""))

        output = {"screening": screening, "candidates": list(screening.included), "screening_note": note}
        self.trace(
            "screen",
            payload_in={"client_id": client.client_id, "round": round_index, "universe": len(products)},
            payload_out={
                "included": screening.included,
                "excluded": [item.product_id for item in screening.excluded],
            },
            tool_calls=tool_calls,
            latency_ms=stopwatch.ms(),
            extra={
                "included_count": len(screening.included),
                "excluded_count": len(screening.excluded),
                "reason_codes": sorted({code for item in screening.excluded for code in item.reasons}),
            },
        )
        return self.guard_output(output)
