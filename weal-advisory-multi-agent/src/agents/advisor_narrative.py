"""AdvisorNarrativeAgent —— 建议书撰写（含反事实解释、压力测试、风险揭示、双录留痕）。

所属层次
--------
Agent 层（`src/agents/`），流水线第 5 环（`narrative` 节点，位于 `human_gate` 之后）。
被 `src/pipeline.py` 的 `AdvisoryPipeline.node_narrative` 调用，
是整条流水线唯一产出"可交付物"的环节。

解决什么问题
------------
把上游已经算好的结构化事实（组合、闸门结论、压力测试、反事实）组织成
一份**客户可读、监管可查**的建议书，并同时留下 `AdviceRecord` 供版本链归档；
模型只负责措辞，所有数字都来自确定性计算结果。

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

与其他 Agent 的互斥关系
-----------------------
- 只写 `narrative` / `elements` / `advice` / `stress` / `counterfactual`
  （`can_write_state`），**不写** `portfolio`（权重来自构建 Agent）、
  **不写** `gate`（合规结论来自复核 Agent，本 Agent 只做披露与复述）、
  **不写** `client`（不能为了行文好看而改客户约束）；
- 没有 `suitability:*` 与 `optimize:*` 权限，因此**无法**自行改判合规结论
  或调整权重——它只能"如实转述"，不能"修饰结论"；
- 它也不参与流程路由：`verdict/directive` 早已由闸门定下，
  本 Agent 的产出无论好坏都不会改变 `pass` / `reject` 的既有事实。

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

#: 本 Agent 的能力契约
#: `narrative:analyze` 对应压力测试与反事实（"算事实"），
#: `narrative:write` 对应文案生成（"写文字"），两者分开以便审计。
SPEC = AgentSpec(
    name="AdvisorNarrativeAgent",
    role="建议书撰写与留痕",
    system_prompt=SYSTEM_PROMPT,
    tools=("narrative.run_stress", "narrative.build_counterfactual", "narrative.compose_text"),
    permissions=frozenset({PERM_NARRATIVE_ANALYZE, PERM_NARRATIVE_WRITE, PERM_TRACE_WRITE}),
    can_write_state=("narrative", "elements", "advice", "stress", "counterfactual"),
)


class AdvisorNarrativeAgent(BaseAgent):
    """建议书撰写 Agent。

    职责：跑压力测试 → 生成反事实解释 → 组装建议书正文与完整性要素 →
    生成 `AdviceRecord`（含双录标记），写出 `narrative` / `elements` /
    `advice` / `stress` / `counterfactual` 五个键。

    边界：不自算数字、不改组合、不改闸门结论、不做路由决策；
    所有对外表述都受 `narrative_completeness` 检查与 `dual_record_required`
    标记约束，保证"该揭示的一定在建议书里"。
    """

    def __init__(self, **kwargs: Any) -> None:
        """以固定契约构造撰写 Agent。

        参数：
            **kwargs: 透传给 `BaseAgent.__init__`（常用 `llm=`、`tracer=`）；
                `llm` 为 None 或未配置 Key 时，`narrative.compose_text` 走 mock 文案。

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
        """生成压力测试、反事实解释与建议书正文。

        参数（keyword-only）：
            client: 生效客户档案（建议书面向的"这位客户"）。
            portfolio: 待披露的组合（权重与指标原样引用，不做二次计算）。
            products: 产品映射，用于反事实重解与持仓明细展示。
            stress_config: 压力测试情景配置（情景 ID、冲击幅度等，
                来自 `data/stress_scenarios.json`）。
            screening: 筛选结果，用于在建议书中说明候选池与剔除情况；
                为 None 时相应小节按缺失处理。
            gate: 闸门结论；为 None 时视为无合规命中（`directive` 记为 `"pass"`）。
                其 `blocks` + `warns` 的规则号会作为反事实分析的基线规则。
            human_review: 人工复核记录（谁确认、何时、豁免了什么）；
                为 None 时构造一个空 `HumanReview()`，保证字段恒存在。
            version: 建议版本号（进版本链，默认 1）。
            engine: 实际生效的编排引擎名（`"langgraph"` / `"native"`），
                写入建议记录以实现"结论可追溯到引擎"。
            run_id: 本次运行追踪号；空串时仍会写入记录（便于离线调用）。
            status: 显式指定的建议状态；为空时按闸门结论推导
                （`directive != "reject"` → `"final"`，否则 `"rejected"`）。
            prior_versions: 历史版本摘要序列，用于在正文中生成"版本变更"小节。
            counterfactual_client: 反事实分析使用的客户档案（通常传收紧前的
                原始客户，以便对比"改约束会怎样"）；为 None 时退回 `client`。

        返回：
            经 `guard_output` 校验的字典，含五个键：
            - `narrative`：建议书正文（字符串）；
            - `elements`：完整性要素布尔字典（配置/反事实/压力/揭示/留痕等）；
            - `advice`：`AdviceRecord`（可直接归档进版本链）；
            - `stress`：`StressReport`（各情景结果与最差情景）；
            - `counterfactual`：`CounterfactualReport`（变体与结构化差异）。

        副作用：
            - `narrative.compose_text` 会触发 LLM（无 Key 时自动 mock）；
            - 读取当前时间填充 `AdviceRecord.created_at`（因此同一输入在不同
              时刻运行，记录里的时间戳不同，属预期行为）；
            - 写一条 `node="narrative"` 的 trace，`extra` 带正文长度、最差情景
              与双录原因，`tool_calls` 去重后上报。

        异常：
            PermissionError: 越权调用工具或越权写共享状态。
            KeyError: 工具返回值缺少约定字段。
        """
        stopwatch = Stopwatch()
        tool_calls: list[str] = []

        # 压力测试：确定性公式，情景与最差冲击都写进 StressReport
        stress: StressReport = self.call_tool(
            "narrative.run_stress", portfolio=portfolio, client=client, config=stress_config
        )
        tool_calls.append("narrative.run_stress")

        # 反事实基线规则号 = 闸门命中的 block + warn 规则（无闸门时为空列表）
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
            """通过白名单工具生成文案（LLM 或 mock）。

            作为回调传入 `compose_narrative`，让建议书组装逻辑无需知道
            LLM 的存在（便于离线 mock 与单测替换）。

            参数：
                task: 任务名（决定 system prompt 与输出字段，如
                    `"narrative_summary"`）。
                context: 结构化事实上下文（数字与结论，模型不得改写）。

            返回：
                工具返回体里的 `data` 字典（含 summary / rationale / disclosure 等字段）。

            副作用：
                触发 LLM 或 mock 调用（无网络、无 Key 时走 mock）。

            异常：
                PermissionError: 越权调用 `narrative.compose_text`。
            """
            result = self.call_tool(
                "narrative.compose_text", llm=self.llm, task=task, context=context
            )
            return result["data"]

        # 组装正文：数字来自 stress / counterfactual / portfolio，compose 只产措辞
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

        # 双录留痕判定：返回 (是否需要双录, 触发原因列表)
        required, reasons = dual_record_required(client, portfolio)
        # 未显式指定 status 时按闸门结论推导：被拒 → rejected，其余 → final
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
