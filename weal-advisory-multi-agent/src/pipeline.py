"""投顾流水线：节点实现 + 共享状态 + 人机协同。

共享状态 `AdvisoryState`（LangGraph 与降级引擎共用同一份 schema 与同一批节点函数，
因此两条路径的**状态流转与结果必然一致**，并有单测做路径一致性对照）。

流转：
    profile → screen → optimize → suitability
                          ↑            │
                          └── 打回重配 ─┤（最多 max_repair_rounds 轮）
                                       ├─ 通过 ─────→ narrative → human_gate → END
                                       └─ 超限 → escalate → narrative → human_gate → END

人机协同：命中 block 后被豁免、单一产品集中度超过内控预警线、高龄客户三类情形，
必须经理财经理确认才能定稿；`--auto` 用于演示自动放行，非交互环境自动降级并留痕。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, TypedDict

from .agents import (
    AdvisorNarrativeAgent,
    ClientProfilingAgent,
    PortfolioOptimizerAgent,
    ProductScreeningAgent,
    SuitabilityOfficerAgent,
)
from .constraints import TightenSpec
from .dataset import DataBundle, load_data
from .llm import BaseLLM, build_llm, load_dotenv
from .observability import Tracer, new_run_id
from .schemas import (
    AdviceRecord,
    ClientProfile,
    GateDecision,
    HumanReview,
    Portfolio,
    Product,
    ScreeningResult,
)
from .utils import now_iso, pct
from .versioning import (
    AdviceSnapshot,
    VersionStore,
    build_payload,
    chain_rows,
)

#: 流程状态取值
STATUS_FINAL = "final"
STATUS_FINAL_DEGRADED = "final_degraded"
STATUS_FINAL_ESCALATED = "final_after_escalation"
STATUS_REJECTED = "rejected"
STATUS_REJECTED_BY_HUMAN = "rejected_by_human"
STATUS_BLOCKED = "blocked"


class AdvisoryState(TypedDict, total=False):
    """投顾流水线共享状态（两个引擎共用）。"""

    run_id: str
    client_id: str
    engine: str
    client: ClientProfile
    effective_client: ClientProfile
    profile: dict[str, Any]
    screening: ScreeningResult
    candidates: list[str]
    screening_note: str
    screening_rounds: list[dict[str, Any]]
    portfolio: Portfolio
    portfolio_note: str
    binding: list[str]
    product_scores: dict[str, float]
    portfolio_rounds: list[dict[str, Any]]
    round: int
    gate: GateDecision
    suitability_comment: str
    tighten: dict[str, Any]
    gate_rounds: list[dict[str, Any]]
    escalation_reasons: list[str]
    stress: Any
    counterfactual: Any
    narrative: str
    elements: dict[str, bool]
    advice: AdviceRecord
    human_review: HumanReview
    version: int
    parent_version: int | None
    status: str
    change_log: list[dict[str, Any]]


@dataclass(frozen=True)
class PipelineConfig:
    """流水线配置。"""

    engine: str = "langgraph"
    auto: bool = False
    max_repair_rounds: int = 2
    exempt_rules: tuple[str, ...] = ()
    interactive: bool | None = None
    operator: str = "理财经理（示例工号 A001）"

    def resolve_interactive(self) -> bool:
        """是否具备交互式人工确认条件（未显式指定时按 stdin 是否 TTY 判断）。"""
        if self.interactive is not None:
            return bool(self.interactive)
        try:
            return bool(sys.stdin and sys.stdin.isatty())
        except Exception:  # noqa: BLE001 - 某些运行环境没有可用的 stdin
            return False


class AdvisoryPipeline:
    """约束驱动的投顾流水线（节点函数同时服务两个引擎）。"""

    def __init__(
        self,
        data: DataBundle,
        *,
        llm: BaseLLM | None = None,
        tracer: Tracer | None = None,
        store: VersionStore | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.data = data
        self.config = config or PipelineConfig()
        self.llm = llm or build_llm()
        self.tracer = tracer
        self.store = store or VersionStore()
        self.agents: dict[str, Any] = {
            "ClientProfilingAgent": ClientProfilingAgent(llm=self.llm, tracer=tracer),
            "ProductScreeningAgent": ProductScreeningAgent(llm=self.llm, tracer=tracer),
            "PortfolioOptimizerAgent": PortfolioOptimizerAgent(llm=self.llm, tracer=tracer),
            "SuitabilityOfficerAgent": SuitabilityOfficerAgent(llm=self.llm, tracer=tracer),
            "AdvisorNarrativeAgent": AdvisorNarrativeAgent(llm=self.llm, tracer=tracer),
        }
        self.engine_note: str = ""

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @property
    def products(self) -> dict[str, Product]:
        """产品要素表。"""
        return self.data.products

    def initial_state(self, client_id: str, run_id: str) -> AdvisoryState:
        """构造初始共享状态。"""
        return AdvisoryState(
            run_id=run_id,
            client_id=client_id,
            engine=self.config.engine,
            round=0,
            screening_rounds=[],
            portfolio_rounds=[],
            gate_rounds=[],
            change_log=[],
            tighten={},
            escalation_reasons=[],
            status="running",
        )

    # ------------------------------------------------------------------
    # 节点：客户画像
    # ------------------------------------------------------------------
    def node_profile(self, state: AdvisoryState) -> dict[str, Any]:
        """客户画像与硬约束提取。"""
        client = self.data.client(state["client_id"])
        result = self.agents["ClientProfilingAgent"].run(
            client=client, questionnaire=self.data.questionnaire
        )
        return {
            "client": result["client"],
            "effective_client": result["effective_client"],
            "profile": result["profile"],
        }

    # ------------------------------------------------------------------
    # 节点：产品筛选（初始可行域）
    # ------------------------------------------------------------------
    def node_screen(self, state: AdvisoryState) -> dict[str, Any]:
        """按有效客户约束筛选初始候选池。"""
        effective_client = state["effective_client"]
        result = self.agents["ProductScreeningAgent"].run(
            client=effective_client, products=self.products, round_index=0
        )
        screening: ScreeningResult = result["screening"]
        return {
            "screening": screening,
            "candidates": result["candidates"],
            "screening_note": result["screening_note"],
            "screening_rounds": [screening.model_dump()],
        }

    # ------------------------------------------------------------------
    # 节点：组合构建（打回重配时会再次进入）
    # ------------------------------------------------------------------
    def node_optimize(self, state: AdvisoryState) -> dict[str, Any]:
        """在（可能已被收紧的）可行域内求解组合。"""
        round_index = int(state.get("round", 0))
        effective_client = state["effective_client"]

        screening_rounds = list(state.get("screening_rounds") or [])
        if round_index == 0 and state.get("screening") is not None:
            screening: ScreeningResult = state["screening"]
        else:
            screening = self.agents["ProductScreeningAgent"].run(
                client=effective_client, products=self.products, round_index=round_index
            )["screening"]
            screening_rounds.append(screening.model_dump())

        result = self.agents["PortfolioOptimizerAgent"].run(
            client=effective_client,
            candidates=screening.included,
            products=self.products,
            round_index=round_index,
        )
        portfolio: Portfolio = result["portfolio"]
        portfolio_rounds = list(state.get("portfolio_rounds") or [])
        portfolio_rounds.append(
            {
                "round": round_index,
                "constraints": effective_client.constraint_snapshot(),
                "weights": dict(portfolio.weights),
                "cash_weight": portfolio.cash_weight,
                "metrics": dict(portfolio.metrics),
            }
        )

        change_log = list(state.get("change_log") or [])
        change_log.append(
            {
                "node": "optimize",
                "round": round_index,
                "candidates": len(screening.included),
                "holdings": len(portfolio.held_ids()),
                "violations": 0,
            }
        )

        return {
            "screening": screening,
            "candidates": list(screening.included),
            "screening_rounds": screening_rounds,
            "portfolio": portfolio,
            "portfolio_note": result["portfolio_note"],
            "binding": result["binding"],
            "product_scores": result["product_scores"],
            "portfolio_rounds": portfolio_rounds,
            "change_log": change_log,
        }

    # ------------------------------------------------------------------
    # 节点：适当性闸门
    # ------------------------------------------------------------------
    def node_suitability(self, state: AdvisoryState) -> dict[str, Any]:
        """适当性复核；不通过时下发收紧指令并记录被拦截版本。"""
        round_index = int(state.get("round", 0))
        client: ClientProfile = state["client"]
        portfolio: Portfolio = state["portfolio"]

        result = self.agents["SuitabilityOfficerAgent"].run(
            client=client,
            portfolio=portfolio,
            universe=self.products,
            candidates=state.get("candidates") or [],
            round_index=round_index,
            exempt_rules=self.config.exempt_rules,
            max_repair_rounds=self.config.max_repair_rounds,
        )
        gate: GateDecision = result["gate"]

        gate_rounds = list(state.get("gate_rounds") or [])
        gate_rounds.append(
            {
                "round": round_index,
                "directive": gate.directive,
                "blocks": [v.rule_id for v in gate.blocks],
                "warns": [v.rule_id for v in gate.warns],
                "exempted": gate.exempted_rules,
            }
        )

        updates: dict[str, Any] = {
            "gate": gate,
            "suitability_comment": result["suitability_comment"],
            "tighten": result["tighten"],
            "gate_rounds": gate_rounds,
        }

        change_log = list(state.get("change_log") or [])
        change_log.append(
            {
                "node": "suitability",
                "round": round_index,
                "directive": gate.directive,
                "blocks": [v.rule_id for v in gate.blocks],
            }
        )
        updates["change_log"] = change_log

        if gate.directive == "reoptimize":
            # 记录被拦截版本（建议版本链的第一个节点）
            self._save_snapshot(
                client=client,
                portfolio=portfolio,
                run_id=state["run_id"],
                status=STATUS_BLOCKED,
                directive=gate.directive,
                gate=gate,
                screening=state.get("screening"),
                change_reason=(
                    "适当性闸门打回：命中 "
                    + "、".join(dict.fromkeys(v.rule_id for v in gate.blocks))
                    + "，已下发约束收紧指令"
                ),
            )
            tighten = TightenSpec.from_dict(gate.tighten)
            updates["effective_client"] = tighten.apply(state["effective_client"])
            updates["round"] = round_index + 1
            change_log.append(
                {
                    "node": "repair",
                    "round": round_index,
                    "tighten": gate.tighten,
                    "next_round": round_index + 1,
                }
            )
            updates["change_log"] = change_log

        return updates

    # ------------------------------------------------------------------
    # 节点：超限转人工
    # ------------------------------------------------------------------
    def node_escalate(self, state: AdvisoryState) -> dict[str, Any]:
        """打回重配次数用尽：标记转人工并保留失败原因。"""
        gate: GateDecision = state["gate"]
        if gate.veto_rules:
            reasons = [
                "命中不可修复的 veto 规则，闸门直接拒绝："
                + "、".join(gate.veto_rules)
            ]
        else:
            reasons = [
                f"第 {gate.round_index + 1} 轮仍命中 block 级规则（打回重配次数用尽）："
                + "、".join(dict.fromkeys(v.rule_id for v in gate.blocks))
            ]
        change_log = list(state.get("change_log") or [])
        change_log.append({"node": "escalate", "reasons": reasons})
        return {
            "status": "escalated",
            "escalation_reasons": reasons,
            "change_log": change_log,
        }

    # ------------------------------------------------------------------
    # 节点：建议书
    # ------------------------------------------------------------------
    def node_narrative(self, state: AdvisoryState) -> dict[str, Any]:
        """生成压力测试、反事实解释与建议书正文，并落盘最终版本快照。"""
        client: ClientProfile = state["client"]
        portfolio: Portfolio = state["portfolio"]
        review: HumanReview = state.get("human_review") or HumanReview()
        status: str = state.get("status") or STATUS_FINAL
        version, parent_version = self.store.next_version(client.client_id)
        prior = chain_rows(self.store.chain(client.client_id))

        result = self.agents["AdvisorNarrativeAgent"].run(
            client=client,
            portfolio=portfolio,
            products=self.products,
            stress_config=self.data.stress_config,
            screening=state.get("screening"),
            gate=state.get("gate"),
            human_review=review,
            status=status,
            version=version,
            engine=state.get("engine", self.config.engine),
            run_id=state["run_id"],
            prior_versions=prior,
            counterfactual_client=state.get("effective_client"),
        )

        change_log = list(state.get("change_log") or [])
        change_log.append(
            {
                "node": "narrative",
                "version": version,
                "elements_ok": sum(1 for ok in result["elements"].values() if ok),
                "elements_total": len(result["elements"]),
            }
        )

        self._save_snapshot(
            client=client,
            portfolio=portfolio,
            run_id=state["run_id"],
            status=status,
            directive=(state["gate"].directive if state.get("gate") else "pass"),
            gate=state.get("gate"),
            screening=state.get("screening"),
            human_review=review,
            stress=result["stress"],
            counterfactual=result["counterfactual"],
            narrative=result["narrative"],
            change_reason=self._final_change_reason(client.client_id, state),
            version=version,
        )

        return {
            "narrative": result["narrative"],
            "elements": result["elements"],
            "advice": result["advice"],
            "stress": result["stress"],
            "counterfactual": result["counterfactual"],
            "version": version,
            "parent_version": parent_version,
            "change_log": change_log,
        }

    # ------------------------------------------------------------------
    # 节点：人机协同闸门
    # ------------------------------------------------------------------
    def node_human_gate(self, state: AdvisoryState) -> dict[str, Any]:
        """人工确认（或按配置放行）。

        闸门在建议书定稿**之前**执行：未通过人工确认（或被否决）时，
        最终状态不会是正常定稿，从而保证「必须人工确认后才能定稿」。
        """
        gate: GateDecision | None = state.get("gate")
        review = self.decide_human_review(state)

        status_map = {
            "not_required": STATUS_FINAL,
            "approved": STATUS_FINAL,
            "auto_approved": STATUS_FINAL,
            "auto_degraded": STATUS_FINAL_DEGRADED,
            "rejected": STATUS_REJECTED_BY_HUMAN,
        }
        if gate is not None and gate.directive == "reject":
            status = STATUS_REJECTED
        else:
            status = status_map.get(review.decision, STATUS_FINAL)
        if state.get("escalation_reasons") and status == STATUS_FINAL:
            status = STATUS_FINAL_ESCALATED

        change_log = list(state.get("change_log") or [])
        change_log.append(
            {
                "node": "human_gate",
                "required": review.required,
                "decision": review.decision,
                "status": status,
            }
        )
        return {"human_review": review, "status": status, "change_log": change_log}

    # ------------------------------------------------------------------
    # 人机协同判定
    # ------------------------------------------------------------------
    def human_review_reasons(self, state: AdvisoryState) -> list[str]:
        """判断是否需要人工确认，并给出全部触发原因。"""
        reasons: list[str] = []
        gate: GateDecision | None = state.get("gate")
        client: ClientProfile = state["client"]
        portfolio: Portfolio = state["portfolio"]

        if gate is not None and gate.exempted_rules:
            reasons.append(
                "命中 block 级适当性规则并经人工豁免：" + "、".join(gate.exempted_rules)
            )
        if state.get("escalation_reasons"):
            reasons.extend(state["escalation_reasons"])
        for pid in portfolio.held_ids():
            weight = portfolio.weights[pid]
            if weight > client.warning_ratio + 1e-9:
                reasons.append(
                    f"单一产品集中度 {pct(weight)} 超过内控预警线 {pct(client.warning_ratio)}"
                    f"（{portfolio.products[pid].name}）"
                )
        if client.is_elderly:
            reasons.append(f"客户为高龄客户（{client.age} 周岁），须执行特别保护确认")
        return list(dict.fromkeys(reasons))

    def decide_human_review(self, state: AdvisoryState) -> HumanReview:
        """执行人工确认流程：`--auto` 自动放行；非交互环境自动降级并留痕。"""
        reasons = self.human_review_reasons(state)
        if not reasons:
            return HumanReview(required=False, decision="not_required", operator="system")

        if self.config.auto:
            return HumanReview(
                required=True,
                reasons=reasons,
                decision="auto_approved",
                operator="auto(--auto)",
                note="演示模式（--auto）自动放行，已记录确认留痕",
                decided_at=now_iso(),
                exempted_rules=list((state.get("gate").exempted_rules if state.get("gate") else []) or []),
            )

        if self.config.resolve_interactive():
            answer = input("检测到需人工确认事项，是否放行？(y/N) ").strip().lower()
            approved = answer in {"y", "yes"}
            return HumanReview(
                required=True,
                reasons=reasons,
                decision="approved" if approved else "rejected",
                operator=self.config.operator,
                note="交互式人工确认" + ("：同意放行" if approved else "：否决"),
                decided_at=now_iso(),
                exempted_rules=list((state.get("gate").exempted_rules if state.get("gate") else []) or []),
            )

        return HumanReview(
            required=True,
            reasons=reasons,
            decision="auto_degraded",
            operator=self.config.operator,
            degraded=True,
            note="非交互环境无法完成人工确认，按配置自动降级放行并留痕，需后续人工复核",
            decided_at=now_iso(),
            exempted_rules=list((state.get("gate").exempted_rules if state.get("gate") else []) or []),
        )

    # ------------------------------------------------------------------
    # 版本链
    # ------------------------------------------------------------------
    def _final_change_reason(self, client_id: str, state: AdvisoryState) -> str:
        """生成最终版本的变更原因。"""
        chain = self.store.chain(client_id)
        if not chain:
            return "首次生成建议"
        last = chain[-1]
        if last.status == STATUS_BLOCKED:
            return "按适当性闸门下发的约束收紧指令重新配置后通过并定稿"
        if state.get("escalation_reasons"):
            return "超限转人工处理后定稿"
        return "重新生成建议（重跑流水线）"

    def _save_snapshot(
        self,
        *,
        client: ClientProfile,
        portfolio: Portfolio,
        run_id: str,
        status: str,
        directive: str,
        gate: GateDecision | None,
        screening: ScreeningResult | None = None,
        human_review: HumanReview | None = None,
        stress: Any = None,
        counterfactual: Any = None,
        narrative: str = "",
        change_reason: str = "",
        version: int | None = None,
    ) -> AdviceSnapshot:
        """保存一条不可变的建议版本快照。"""
        if version is None:
            version, parent = self.store.next_version(client.client_id)
        else:
            chain = self.store.chain(client.client_id)
            parent = chain[-1].version if chain else None
        payload = build_payload(
            client,
            portfolio,
            run_id=run_id,
            engine=self.config.engine,
            status=status,
            directive=directive,
            screening=screening,
            gate=gate,
            human_review=human_review,
            stress=stress,
            counterfactual=counterfactual,
            narrative=narrative,
        )
        snapshot = AdviceSnapshot.create(
            version=version,
            parent_version=parent,
            client_id=client.client_id,
            run_id=run_id,
            change_reason=change_reason,
            payload=payload,
            status=status,
        )
        self.store.append(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------
    def route_after_suitability(self, state: AdvisoryState) -> str:
        """闸门后的条件路由（两个引擎共用同一判定）。"""
        gate: GateDecision = state["gate"]
        if gate.directive == "pass":
            return "human_gate"
        if gate.directive == "reoptimize":
            return "optimize"
        return "escalate"


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------
def build_pipeline(
    *,
    data_dir: Any = None,
    engine: str = "langgraph",
    auto: bool = False,
    max_repair_rounds: int = 2,
    exempt_rules: tuple[str, ...] = (),
    interactive: bool | None = None,
    runs_dir: Any = None,
    store: VersionStore | None = None,
    llm: BaseLLM | None = None,
) -> tuple[AdvisoryPipeline, Tracer]:
    """构建流水线与 tracer。"""
    load_dotenv()
    from .engine import resolve_engine

    resolved_engine, engine_note = resolve_engine(engine)
    data = load_data(data_dir)
    config = PipelineConfig(
        engine=resolved_engine,
        auto=auto,
        max_repair_rounds=max_repair_rounds,
        exempt_rules=tuple(exempt_rules),
        interactive=interactive,
    )
    pipeline = AdvisoryPipeline(data, llm=llm, tracer=None, store=store, config=config)
    pipeline.engine_note = engine_note
    tracer = Tracer(run_id=new_run_id("advisory"), runs_dir=runs_dir)
    pipeline.tracer = tracer
    for agent in pipeline.agents.values():
        agent.tracer = tracer
    return pipeline, tracer


def run_pipeline(
    client_id: str,
    *,
    engine: str = "langgraph",
    auto: bool = True,
    data_dir: Any = None,
    runs_dir: Any = None,
    store: VersionStore | None = None,
    max_repair_rounds: int = 2,
    exempt_rules: tuple[str, ...] = (),
    run_id: str | None = None,
    interactive: bool | None = False,
) -> tuple[AdvisoryState, Tracer, AdvisoryPipeline]:
    """按指定引擎跑完一位客户的完整投顾流水线。"""
    pipeline, tracer = build_pipeline(
        data_dir=data_dir,
        engine=engine,
        auto=auto,
        max_repair_rounds=max_repair_rounds,
        exempt_rules=exempt_rules,
        interactive=interactive,
        runs_dir=runs_dir,
        store=store,
    )
    if run_id:
        tracer.run_id = run_id

    if engine == "native":
        state = run_native(pipeline, client_id, tracer.run_id)
    else:
        from .engine.langgraph_engine import run_langgraph

        state = run_langgraph(pipeline, client_id, tracer.run_id)
    return state, tracer, pipeline


def run_native(pipeline: AdvisoryPipeline, client_id: str, run_id: str) -> AdvisoryState:
    """降级引擎（零第三方依赖）：手动状态机，与 LangGraph 路径完全一致。"""
    from .engine.native import run_native as _run

    return _run(pipeline, client_id, run_id)


def state_summary(state: AdvisoryState) -> dict[str, Any]:
    """把最终状态压缩成可比较、可展示的摘要（评估与双引擎对照共用）。"""
    gate: GateDecision | None = state.get("gate")
    portfolio: Portfolio | None = state.get("portfolio")
    advice: AdviceRecord | None = state.get("advice")
    return {
        "client_id": state.get("client_id"),
        "engine": state.get("engine"),
        "status": state.get("status"),
        "rounds": int(state.get("round", 0)) + 1,
        "candidates": list(state.get("candidates") or []),
        "weights": dict(portfolio.weights) if portfolio else {},
        "cash_weight": portfolio.cash_weight if portfolio else 0.0,
        "metrics": dict(portfolio.metrics) if portfolio else {},
        "directive": gate.directive if gate else None,
        "block_rules": [v.rule_id for v in gate.blocks] if gate else [],
        "warn_rules": list(dict.fromkeys(v.rule_id for v in gate.warns)) if gate else [],
        "escalated": bool(gate.escalated) if gate else False,
        "human_required": bool(state.get("human_review") and state["human_review"].required),
        "human_decision": state["human_review"].decision if state.get("human_review") else None,
        "version": state.get("version"),
        "elements_ok": sum(1 for ok in (state.get("elements") or {}).values() if ok),
        "stress_scenarios": len(state["stress"].scenarios) if state.get("stress") else 0,
        "counterfactual_variants": (
            len(state["counterfactual"].variants) if state.get("counterfactual") else 0
        ),
        "narrative_length": len(state.get("narrative") or ""),
        "advice_status": advice.status if advice else None,
        "trace_steps": None,
    }
