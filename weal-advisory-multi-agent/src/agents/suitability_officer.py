"""SuitabilityOfficerAgent —— 适当性合规复核（**硬闸门**）。

所属层次
--------
Agent 层（`src/agents/`），流水线第 4 环（`suitability` 节点）——
它是唯一能改变流程走向的节点，被 `src/pipeline.py` 的
`AdvisoryPipeline.node_suitability` 调用。

解决什么问题
------------
把"能不能卖给这位客户"从模型判断变成**规则判断**：
命中 `block` 不是让模型"再想想措辞"，而是下发约束收紧指令，
让组合在更小的可行域里重新求解；打回用尽则转人工，不可修复则直接拒绝。

职责边界
--------
这是全流程唯一有**否决权**的 Agent，也是与「反思循环」架构差别最大的一环：

- 判定完全由 `src/suitability/rules.py` 的 22 条确定性规则完成，**不调用模型**；
- 命中 `block` 级规则 → 不是让模型"再想想"，而是下发**约束收紧指令**
  （`TightenSpec`：降等级 / 缩期限 / 压集中度 / 提流动性 / 排除产品与类别），
  让组合在更小的可行域里重新求解；
- 打回次数用尽仍不合规 → 转人工；命中 `veto` 规则（如可行域为空）→ 直接拒绝；
- 经人工豁免的 block 规则会降级为 warn 并**留痕**，但豁免必须在建议书里显式披露。

与其他 Agent 的互斥关系
-----------------------
- 只写 `gate` / `suitability_comment` / `tighten`（`can_write_state`），
  **不写** `portfolio`（它只判定、不改权重）、**不写** `client`
  （它只能通过 `TightenSpec` 让流程去收紧约束，而不是自己改档案）、
  **不写** `narrative`（披露由撰写 Agent 落笔）；
- `suitability:veto` 权限**仅本 Agent 拥有**（见 `src/agents/tools.py` 的
  `PERM_SUIT_VETO` 只授予 `suitability.issue_directive`），因此构建 Agent
  即使想"自我放行"也调不动闸门工具；
- 反过来，本 Agent 也**没有** `optimize:*` 权限，无法直接改权重来"修好"组合——
  它的手段只有「通过 / 打回并收紧约束 / 拒绝」三种。

工具白名单：`suitability.evaluate_rules` / `suitability.issue_directive` /
`suitability.build_tighten` / `suitability.compose_comment`
权限集合：`suitability:judge`、`suitability:veto`（**仅本 Agent 拥有**）、`trace:write`
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..schemas import ClientProfile, GateDecision, Portfolio, Product
from .base import BaseAgent, Stopwatch
from .tools import (
    PERM_SUIT_JUDGE,
    PERM_SUIT_VETO,
    PERM_TRACE_WRITE,
    AgentSpec,
)

SYSTEM_PROMPT = """你是财富管理机构的适当性合规复核岗，对组合拥有一票否决权。

你必须遵守：
1. 只依据给定的适当性规则做判断，不得引入规则之外的偏好，也不得因为业绩好看而放宽标准；
2. 命中 block 级规则必须拦截：要么打回重新配置，要么直接拒绝，禁止"带病通过"；
3. 每条命中必须给出规则号、违规事实与依据说明，便于事后复核与监管问询；
4. warn 级规则只做揭示与人工提示，不得当作拦截理由；
5. 不得修改客户档案、不得直接调整组合权重——你的手段只有"通过 / 打回并收紧约束 / 拒绝"。

输出字段：comment（复核意见）。"""

#: 本 Agent 的能力契约
#: `suitability:veto` 是全项目最敏感的一个权限标签，只出现在这里；
#: `can_write_state` 只有三个键，保证它无法顺手改掉组合或客户档案。
SPEC = AgentSpec(
    name="SuitabilityOfficerAgent",
    role="适当性合规复核（硬闸门）",
    system_prompt=SYSTEM_PROMPT,
    tools=(
        "suitability.evaluate_rules",
        "suitability.issue_directive",
        "suitability.build_tighten",
        "suitability.compose_comment",
    ),
    permissions=frozenset({PERM_SUIT_JUDGE, PERM_SUIT_VETO, PERM_TRACE_WRITE}),
    can_write_state=("gate", "suitability_comment", "tighten"),
)


class SuitabilityOfficerAgent(BaseAgent):
    """适当性复核 Agent。

    职责：规则求值 → 出具闸门指令（pass / reoptimize / reject）→
    打回时构造 `TightenSpec` → 生成复核意见，写出 `gate` /
    `suitability_comment` / `tighten`。

    边界：不改权重、不改客户档案、不筛选产品、不撰写建议书；
    判定输入来自 `RuleContext`（客户 + 组合 + 产品池 + 候选池），
    判定规则来自 `rules.py`，因此结论可复现、可逐条追溯到规则号。

    注：本 Agent 的 `run` 里会出现**两次**规则求值——一次是本 Agent 为统计命中
    显式调用的 `suitability.evaluate_rules`，一次在 `suitability.issue_directive`
    工具内部（由 `SuitabilityGate.review` 执行）。两次输入相同、规则表相同，
    结论必然一致，第二次才是真正决定 `directive` 的那一次。
    """

    def __init__(self, **kwargs: Any) -> None:
        """以固定契约构造复核 Agent。

        参数：
            **kwargs: 透传给 `BaseAgent.__init__`（常用 `llm=`、`tracer=`）。
                注意 `llm` 只影响复核意见的措辞，不影响任何判定。

        返回：None。
        副作用/异常：无。
        """
        super().__init__(SPEC, **kwargs)

    # ------------------------------------------------------------------
    def run(
        self,
        *,
        client: ClientProfile,
        portfolio: Portfolio,
        universe: Mapping[str, Product],
        candidates: Iterable[str],
        round_index: int = 0,
        exempt_rules: Iterable[str] = (),
        max_repair_rounds: int = 2,
    ) -> dict[str, Any]:
        """执行规则求值与闸门判定。

        参数（keyword-only）：
            client: 生效客户档案（已被画像与历次收紧指令修订过）。
            portfolio: 待复核组合（构建 Agent 的产出）。
            universe: 全量产品池（`{product_id: Product}`），
                用于计算类别/发行人集中度与"可行域是否为空"。
            candidates: 当前候选池产品号（会被去重排序）；
                为空即触发 `S-FEASIBLE-POOL`（veto）→ 直接拒绝。
            round_index: 已发生的重配轮次（0 为首轮）；达到
                `max_repair_rounds` 仍命中 block 时转为拒绝并标记 `escalated`。
            exempt_rules: 经人工批准的豁免规则号集合；命中这些规则的 block
                会被降级为 warn（说明文字加"【经人工豁免】"前缀）并记录在
                `gate.exempted_rules` 里，但仍须在建议书中披露。
            max_repair_rounds: 允许打回重配的最大轮次（默认 2）；
                直接透传给 `SuitabilityGate`。

        返回：
            经 `guard_output` 校验的字典，含三个键：
            - `gate`：`GateDecision`（directive / passed / blocks / warns /
              veto_rules / exempted_rules / escalated / tighten / comment）；
            - `suitability_comment`：复核意见（模型措辞；模型返回空串时
              回退为 `gate.comment` 的确定性说明）；
            - `tighten`：`TightenSpec.to_dict()`，仅当 directive == "reoptimize"
              时非空，其余情况为 `{}`。

        副作用：
            - 当 `decision.directive == "reoptimize"` 时，**原地修改**了由工具
              返回的 `GateDecision` 对象的 `tighten` 字段（补写收紧指令），
              因此 `gate.tighten` 与 `output["tighten"]` 始终同源同值；
            - `suitability.compose_comment` 会触发 LLM（无 Key 时自动 mock）；
            - 写一条 `node="suitability"` 的 trace，`extra` 里带命中规则总数、
              veto 规则号与是否转人工。

        异常：
            PermissionError: 越权调用工具或越权写共享状态。
            KeyError: 工具返回值缺少约定字段。
        """
        stopwatch = Stopwatch()
        tool_calls: list[str] = []
        candidate_ids = tuple(sorted(set(candidates)))

        # 第一次求值：本 Agent 自己取一份完整命中清单（用于 trace 统计与展示）
        evaluation = self.call_tool(
            "suitability.evaluate_rules",
            client=client,
            portfolio=portfolio,
            universe=universe,
            candidates=candidate_ids,
        )
        tool_calls.append("suitability.evaluate_rules")

        # 第二次求值在工具内部，并据此产出真正的闸门指令（pass / reoptimize / reject）
        decision: GateDecision = self.call_tool(
            "suitability.issue_directive",
            client=client,
            portfolio=portfolio,
            universe=universe,
            candidates=candidate_ids,
            round_index=round_index,
            exempt_rules=tuple(exempt_rules),
            max_repair_rounds=max_repair_rounds,
        )
        tool_calls.append("suitability.issue_directive")

        # 仅打回时构造收紧指令（工具内部会重新求值并按规则号过滤 block 命中）
        if decision.directive == "reoptimize":
            tighten = self.call_tool(
                "suitability.build_tighten",
                client=client,
                portfolio=portfolio,
                universe=universe,
                candidates=candidate_ids,
                rule_ids=[v.rule_id for v in decision.blocks],
            )
            tool_calls.append("suitability.build_tighten")
            decision.tighten = tighten  # 注：这是对工具返回对象字段的原地写入

        text = self.call_tool(
            "suitability.compose_comment",
            llm=self.llm,
            task="suitability_comment",
            context={
                "directive": decision.directive,
                "block_rules": [v.rule_id for v in decision.blocks],
                "warn_rules": list(dict.fromkeys(v.rule_id for v in decision.warns)),
                "round_index": round_index,
            },
        )
        tool_calls.append("suitability.compose_comment")
        # 模型只改措辞：返回空串时回退到闸门自带的确定性说明
        comment = str(text["data"].get("comment", "")) or decision.comment

        output = {"gate": decision, "suitability_comment": comment, "tighten": decision.tighten}
        self.trace(
            "suitability",
            payload_in={
                "client_id": client.client_id,
                "round": round_index,
                "portfolio": portfolio.weights,
                "candidates": list(candidate_ids),
            },
            payload_out={
                "directive": decision.directive,
                "passed": decision.passed,
                "blocks": [v.rule_id for v in decision.blocks],
                "warns": [v.rule_id for v in decision.warns],
                "tighten": decision.tighten,
            },
            tool_calls=tool_calls,
            latency_ms=stopwatch.ms(),
            extra={
                "hit_rule_count": len(evaluation.hits),
                "veto_rules": decision.veto_rules,
                "escalated": decision.escalated,
            },
        )
        return self.guard_output(output)

    # ------------------------------------------------------------------
    def evaluate_only(
        self,
        *,
        client: ClientProfile,
        portfolio: Portfolio,
        universe: Mapping[str, Product],
        candidates: Iterable[str],
    ) -> Any:
        """只做规则求值、不出具指令（供评估脚本统计命中情况）。

        与 `run` 的区别：不产生 `GateDecision`、不构造收紧指令、不调用模型、
        不写共享状态（因此也不经 `guard_output`），只把命中清单交给调用方。
        评估脚本用它统计"每条规则被命中多少次"，而不会推动流水线。

        参数（keyword-only）：
            client: 生效客户档案。
            portfolio: 待求值的组合。
            universe: 全量产品池。
            candidates: 当前候选池产品号（会被去重排序）。

        返回：
            `RuleEvaluation`（hits / blocks / warns / hit_rule_ids 及两个去重属性）。

        副作用：
            无：本方法不调用 `self.trace`，也不写共享状态，
            只执行纯计算（规则 handler 均为纯函数）。

        异常：
            PermissionError: 工具不在白名单或权限不足（正常构造下不会发生）。
        """
        return self.call_tool(
            "suitability.evaluate_rules",
            client=client,
            portfolio=portfolio,
            universe=universe,
            candidates=tuple(sorted(set(candidates))),
        )
