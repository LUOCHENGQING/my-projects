"""AdvisorNarrativeAgent —— 建议书撰写（含反事实解释、压力测试、风险揭示、双录留痕）。

职责边界
--------
本 Agent 是**唯一的产出方**，负责把上游已经算好的结构化事实组织成一份可交付建议书：

- 配置建议（持仓表 + 权重 + 参考金额 + 组合指标）
- **反事实解释**（改约束 → 重新求解 → 结构化差异，非模型臆测）
- **情景压力测试**（利率上行 / 权益回撤 / 信用利差走阔，确定性公式）
- 风险揭示、费率揭示
- **双录留痕标记**（高龄客户 / R4 及以上 / 衍生品结构等情形）
- 适当性规则命中说明与人工确认记录
- 建议版本链信息

措辞通过白名单工具 `narrative.compose_text` 走 LLM（无 Key 时自动 mock），
但**所有数字与结论都来自确定性计算结果**，模型只负责组织语言。

工具白名单：`narrative.run_stress` / `narrative.build_counterfactual` / `narrative.compose_text`
权限集合：`narrative:analyze`、`narrative:write`、`trace:write`
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..narrative import dual_record_required, narrative_completeness
from ..narrative import build_narrative as compose_narrative
from ..schemas import (
    AdviceRecord,
    ClientProfile,
    CounterfactualReport,
    GateDecision,
    HumanReview,
    Portfolio,
    Product,
    ScreeningResult,
    StressReport,
)
from ..utils import now_iso
from .base import BaseAgent, Stopwatch
from .tools import (
    PERM_NARRATIVE_ANALYZE,
    PERM_NARRATIVE_WRITE,
    PERM_TRACE_WRITE,
    AgentSpec,
)

SYSTEM_PROMPT = """你是财富管理机构的投顾建议书撰写岗，负责把已确定的配置与合规结论写成客户可读的建议书。

你必须遵守：
1. 只能使用给定的结构化数据，所有数字必须原样引用，禁止自行计算、四舍五入或改写；
2. 必须完整包含：配置建议、反事实解释、情景压力测试、风险揭示、费率揭示、双录留痕标记；
3. 禁止出现"保证收益""稳赚""无风险"等表述，禁止承诺未来业绩；
4. 禁止提及任何真实机构、真实产品或真实评级机构名称；
5. 语言克制、客观，面向普通投资者，避免过度专业的术语堆砌。

输出字段：summary（建议书摘要）、rationale（权衡说明）、disclosure（风险揭示）。"""

SPEC = AgentSpec(
    name="AdvisorNarrativeAgent",
    role="建议书撰写与留痕",
    system_prompt=SYSTEM_PROMPT,
    tools=("narrative.run_stress", "narrative.build_counterfactual", "narrative.compose_text"),
    permissions=frozenset({PERM_NARRATIVE_ANALYZE, PERM_NARRATIVE_WRITE, PERM_TRACE_WRITE}),
    can_write_state=("narrative", "elements", "advice", "stress", "counterfactual"),
)


class AdvisorNarrativeAgent(BaseAgent):
    """建议书撰写 Agent。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(SPEC, **kwargs)

    # ------------------------------------------------------------------
    def run(
        self,
        *,
        client: ClientProfile,
        portfolio: Portfolio,
        products: Mapping[str, Product],
        stress_config: Mapping[str, Any],
        screening: ScreeningResult | None = None,
        gate: GateDecision | None = None,
        human_review: HumanReview | None = None,
        version: int = 1,
        engine: str = "native",
        run_id: str = "",
        status: str = "",
        prior_versions: Sequence[Mapping[str, Any]] = (),
        counterfactual_client: ClientProfile | None = None,
    ) -> dict[str, Any]:
        """生成压力测试、反事实解释与建议书正文。"""
        stopwatch = Stopwatch()
        tool_calls: list[str] = []

        stress: StressReport = self.call_tool(
            "narrative.run_stress", portfolio=portfolio, client=client, config=stress_config
        )
        tool_calls.append("narrative.run_stress")

        rule_ids = []
        if gate is not None:
            rule_ids = [v.rule_id for v in gate.blocks] + [v.rule_id for v in gate.warns]
        counterfactual: CounterfactualReport = self.call_tool(
            "narrative.build_counterfactual",
            client=counterfactual_client or client,
            portfolio=portfolio,
            products=products,
            rule_ids=rule_ids,
            version_label=f"v{version}",
            status=(gate.directive if gate is not None else "unknown"),
            rule_client=client,
        )
        tool_calls.append("narrative.build_counterfactual")

        def compose(task: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
            """通过白名单工具生成文案（LLM 或 mock）。"""
            result = self.call_tool(
                "narrative.compose_text", llm=self.llm, task=task, context=context
            )
            return result["data"]

        text, elements = compose_narrative(
            client=client,
            portfolio=portfolio,
            screening=screening,
            gate=gate,
            human_review=human_review,
            stress=stress,
            counterfactual=counterfactual,
            version=version,
            engine=engine,
            run_id=run_id,
            compose=compose,
            prior_versions=prior_versions,
        )
        tool_calls.append("narrative.compose_text")

        required, reasons = dual_record_required(client, portfolio)
        final_status = status or ("final" if (gate is None or gate.directive != "reject") else "rejected")
        advice = AdviceRecord(
            client_id=client.client_id,
            run_id=run_id,
            version=version,
            status=final_status,
            directive=(gate.directive if gate is not None else "pass"),
            engine=engine,
            portfolio=portfolio,
            screening=screening,
            gate=gate,
            human_review=human_review or HumanReview(),
            stress=stress,
            counterfactual=counterfactual,
            narrative=text,
            elements=elements,
            created_at=now_iso(),
        )

        output = {
            "narrative": text,
            "elements": elements,
            "advice": advice,
            "stress": stress,
            "counterfactual": counterfactual,
        }
        self.trace(
            "narrative",
            payload_in={
                "client_id": client.client_id,
                "portfolio": portfolio.weights,
                "directive": gate.directive if gate else None,
            },
            payload_out={
                "elements": elements,
                "completeness": narrative_completeness(elements),
                "stress_scenarios": [item.scenario_id for item in stress.scenarios],
                "counterfactual_variants": [item.variant_id for item in counterfactual.variants],
                "dual_record_required": required,
            },
            tool_calls=list(dict.fromkeys(tool_calls)),
            latency_ms=stopwatch.ms(),
            extra={
                "narrative_length": len(text),
                "worst_scenario": stress.worst_scenario_id,
                "worst_impact": stress.worst_impact,
                "dual_record_reasons": reasons,
            },
        )
        return self.guard_output(output)
