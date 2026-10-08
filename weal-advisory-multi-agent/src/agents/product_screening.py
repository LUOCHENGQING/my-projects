"""ProductScreeningAgent —— 产品筛选（在硬约束可行域内筛出候选池）。

所属层次
--------
Agent 层（`src/agents/`），流水线第 2 环（`screen` 节点）。
被 `src/pipeline.py` 的 `AdvisoryPipeline.node_screen` 调用，
`round_index > 0`（闸门打回后重配）时也会被再次调用。

解决什么问题
------------
把"这位客户能买什么"变成一个**可复核、可解释**的候选池：
不是让模型凭感觉挑产品，而是对全量产品池逐条跑硬约束判定，
留下"可投清单 + 剔除原因"，让后续构建只在这个可行域内做优化。

职责边界
--------
本 Agent **不做收益预测、不做权重分配**，只回答一个问题：
「在这位客户的硬约束下，哪些产品**可以**进入候选池，哪些必须剔除，为什么？」

准入维度：风险等级上限、期限匹配、起投金额与可投金额、币种、禁止项
（类别 / 具体产品 / 衍生品结构）、投资经验、合格投资者准入。
每一个被剔除的产品都会保留**原因码 + 中文说明**，可直接向客户逐条解释。

与其他 Agent 的互斥关系
-----------------------
- 只写 `screening` / `candidates` / `screening_note`（`can_write_state`），
  **不写** `client`（约束由画像 Agent 定义，本 Agent 无权改）、
  **不写** `portfolio`（权重归构建 Agent）、**不写** `gate`（合规归复核 Agent）；
- 硬约束数值全部来自 `src/constraints.py` 的确定性判定，
  模型只用于生成筛选说明文案，因此本 Agent 不会"因为收益高就放宽准入"，
  也无否决权——它只能缩小可选集合，不能否决流程。

工具白名单：`screening.filter_universe` / `screening.explain_exclusion` / `screening.compose_screening_text`
权限集合：`screening:execute`、`screening:read`、`trace:write`

注：实际实现为——`SPEC.permissions` 还包含 `narrative:write`：筛选说明文案复用
了统一的文案工具 `screening.compose_screening_text`，而该工具在注册表里声明的
权限标签是 `narrative:write`（见 `src/agents/tools.py` 的 `ALL_TOOLS`）。
工具白名单三项与 `SPEC.tools` 一致。
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

#: 本 Agent 的能力契约
#: 注意 `screening.compose_screening_text` 声明的是 `narrative:write` 权限
#: （它复用统一的文案工具），但产出只写入 `screening_note` 这一个键。
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
    """产品筛选 Agent。

    职责：在全量产品池上跑硬约束准入 → 输出候选池与逐条剔除原因 →
    生成筛选口径说明，写出 `screening` / `candidates` / `screening_note`。

    边界：不做收益预测、不做权重分配、不做合规闸门判定；
    没有 `optimize:*` 与 `suitability:*` 权限，因此无法直接改组合或改闸门结论；
    也没有 `profile:*` 权限，无法为了让产品通过而放宽客户约束。
    """

    def __init__(self, **kwargs: Any) -> None:
        """以固定契约构造筛选 Agent。

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
        products: Mapping[str, Product],
        round_index: int = 0,
    ) -> dict[str, Any]:
        """执行筛选并生成说明。

        参数（keyword-only）：
            client: **生效**客户档案（闸门打回后会带更紧的约束，
                例如更低的等级上限、更低的集中度上限、新增禁止项）。
            products: 全量产品池，`{product_id: Product}`；筛选只在该池内进行，
                不会引入池外产品。
            round_index: 当前重配轮次（0 表示首轮），用于在筛选结果中留痕
                "这是第几轮可行域"。

        返回：
            经 `guard_output` 校验的字典，含三个键：
            - `screening`：`ScreeningResult`，含全池数量、可投清单与剔除清单
              （每条带原因码与中文说明）；
            - `candidates`：可投产品号列表（`screening.included` 的副本）；
            - `screening_note`：筛选口径与主要剔除原因的说明文案。

        副作用：
            - `screening.compose_screening_text` 会触发 LLM（无 Key 时自动 mock）；
            - 写一条 `node="screen"` 的 trace，`extra` 里带可投/剔除数量与
              出现过的全部原因码集合（便于审计"哪些约束在起作用"）。

        异常：
            PermissionError: 越权调用工具或越权写共享状态。
            KeyError: 工具返回值缺少约定字段。
        """
        stopwatch = Stopwatch()
        tool_calls: list[str] = []

        # 硬约束筛选：结果同时携带 included（可投）与 excluded（含原因码）
        screening = self.call_tool(
            "screening.filter_universe", client=client, products=products, round_index=round_index
        )
        tool_calls.append("screening.filter_universe")

        # 只取前若干条剔除原因用于对外说明（全量仍在 screening.excluded 里）
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

        # candidates 与 screening.included 内容一致，便于下游（构建 Agent）直接消费
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
