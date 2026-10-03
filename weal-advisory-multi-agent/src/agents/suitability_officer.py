"""SuitabilityOfficerAgent —— 适当性合规复核（**硬闸门**）。

职责边界
--------
这是全流程唯一有**否决权**的 Agent，也是与「反思循环」架构差别最大的一环：

- 判定完全由 `src/suitability/rules.py` 的 22 条确定性规则完成，**不调用模型**；
- 命中 `block` 级规则 → 不是让模型"再想想"，而是下发**约束收紧指令**
  （`TightenSpec`：降等级 / 缩期限 / 压集中度 / 提流动性 / 排除产品与类别），
  让组合在更小的可行域里重新求解；
- 打回次数用尽仍不合规 → 转人工；命中 `veto` 规则（如可行域为空）→ 直接拒绝；
- 经人工豁免的 block 规则会降级为 warn 并**留痕**，但豁免必须在建议书里显式披露。

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
    """适当性复核 Agent。"""

    def __init__(self, **kwargs: Any) -> None:
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
        """执行规则求值与闸门判定。"""
        stopwatch = Stopwatch()
        tool_calls: list[str] = []
        candidate_ids = tuple(sorted(set(candidates)))

        evaluation = self.call_tool(
            "suitability.evaluate_rules",
            client=client,
            portfolio=portfolio,
            universe=universe,
            candidates=candidate_ids,
        )
        tool_calls.append("suitability.evaluate_rules")

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
            decision.tighten = tighten

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
        """只做规则求值、不出具指令（供评估脚本统计命中情况）。"""
        return self.call_tool(
            "suitability.evaluate_rules",
            client=client,
            portfolio=portfolio,
            universe=universe,
            candidates=tuple(sorted(set(candidates))),
        )
