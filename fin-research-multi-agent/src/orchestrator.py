"""编排层：把语料、工具、Agent 和引擎装配成一张可运行的图。

Agent 拓扑（与 README 的 mermaid 图一致）：

                    ┌──────────┐
                    │ planner  │
                    └────┬─────┘
              route[] ┌──┴───────────────┐
                      v                  v
                ┌───────────┐      ┌───────────┐
                │ retriever │      │  writer   │  (仅寒暄类问题)
                └─────┬─────┘      └─────┬─────┘
                      v                  │
                ┌───────────┐            │
                │  analyst  │<─────┐     │
                └─────┬─────┘      │     │
                      v            │     │
                ┌─────────────┐    │     │
                │risk_checker │────┘ revise（最多 2 轮）
                └──────┬──────┘          │
              escalate │ pass            │
                       v                 │
              ┌────────────────┐         │
              │  human_review  │─────────┤ approved
              └────────┬───────┘         │
                   rejected               │
                       v                  v
                     [END] <──────── writer
                                        │
                                       END

三条条件边（这是"多智能体"和"一条链"的本质区别）：
    1. planner  -> 根据 route 决定是否需要检索；
    2. risk_checker -> pass / revise（回到 analyst）/ escalate（转人工）；
    3. human_review -> approved（继续写简报）/ rejected（终止）。

反思循环的防死循环三保险：
    (a) revision_round 计数，超过 max_revision_rounds 一律 escalate；
    (b) 循环只允许 analyst <-> risk_checker 之间往返，拓扑上不会绕回 planner；
    (c) 引擎层 recursion_limit 兜底。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .agents import build_agents
from .agents.base import AgentContext
from .config import GRAPH_RECURSION_LIMIT, RUNS_DIR, RuntimeConfig, runtime_config
from .engine_langgraph import LANGGRAPH_AVAILABLE, build_langgraph_runner
from .engine_native import END, StateGraph
from .hitl import HumanReviewer, render_payload
from .llm.client import LLMClient
from .rag import HybridRetriever, build_chunks, build_fact_store, load_documents
from .state import ResearchState, new_state, snapshot
from .tools import build_default_registry
from .tracing import TraceRecorder
from .utils.digest import digest

__all__ = ["GraphSpec", "ResearchPipeline", "build_engine", "run_research"]


# ---------------------------------------------------------------------------
# 图规格（引擎无关的声明式描述）
# ---------------------------------------------------------------------------
@dataclass
class GraphSpec:
    """一张图的完整声明，可被 LangGraph 或自研引擎编译执行。"""

    state_schema: Any
    nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]
    entry: str
    edges: List[Tuple[str, str]] = field(default_factory=list)
    conditional_edges: List[Tuple[str, Callable[[Dict[str, Any]], str], Dict[str, str]]] = field(
        default_factory=list
    )
    end: str = END


# ---------------------------------------------------------------------------
# 引擎选择
# ---------------------------------------------------------------------------
def build_engine(spec: GraphSpec, preference: str = "auto", thread_id: str = "default") -> Any:
    """按偏好构建执行引擎；auto 优先 LangGraph，失败自动降级到自研引擎。"""
    pref = (preference or "auto").lower()

    if pref in {"auto", "langgraph"} and LANGGRAPH_AVAILABLE:
        try:
            return build_langgraph_runner(spec, thread_id=thread_id)
        except Exception:  # noqa: BLE001 - 版本不兼容时静默降级
            if pref == "langgraph":
                raise
            # auto 模式下继续走自研引擎

    graph = StateGraph(spec.state_schema)
    for name, fn in spec.nodes.items():
        graph.add_node(name, fn)
    graph.set_entry_point(spec.entry)
    for src, dst in spec.edges:
        graph.add_edge(src, dst)
    for src, router, mapping in spec.conditional_edges:
        graph.add_conditional_edges(src, router, mapping)
    from .engine_native import InMemoryCheckpointer

    return graph.compile(checkpointer=InMemoryCheckpointer())


# ---------------------------------------------------------------------------
# 管线
# ---------------------------------------------------------------------------
class ResearchPipeline:
    """投研多智能体管线。"""

    def __init__(
        self,
        *,
        auto: bool = False,
        engine: Optional[str] = None,
        runs_dir: Optional[Path] = None,
        config: Optional[RuntimeConfig] = None,
        data_dir: Optional[Path] = None,
        reviewer: Optional[HumanReviewer] = None,
        quiet: bool = False,
    ) -> None:
        self.config = config or runtime_config()
        if engine:
            self.config.engine = engine
        self.runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
        self.quiet = quiet

        # ---- 1) 语料 -> 父子块 -> 混合检索器 + 事实库 ----
        self.documents = load_documents(data_dir)
        self.parents, self.children = build_chunks(
            list(self.documents),
            child_chars=self.config.child_chunk_chars,
            overlap=self.config.child_chunk_overlap,
        )
        self.retriever = HybridRetriever(
            self.parents, self.children, weights=self.config.rerank_weights, embed_dim=self.config.embed_dim
        )
        self.fact_store = build_fact_store(list(self.documents))

        # ---- 2) 工具注册中心 ----
        self.registry = build_default_registry(self.retriever, self.fact_store, self.documents)

        # ---- 3) LLM（无 key 自动 mock） ----
        self.llm = LLMClient(self.config)

        # ---- 4) 人机协同 ----
        self.reviewer = reviewer or HumanReviewer(auto=auto, print_fn=print if not quiet else (lambda *a, **k: None))

        self.engine_name = "native"
        self.last_spec: Optional[GraphSpec] = None

    # ------------------------------------------------------------------
    # 图装配
    # ------------------------------------------------------------------
    def _make_routers(self) -> Dict[str, Callable[[Dict[str, Any]], str]]:
        """三条条件边的路由函数。"""

        def route_after_planner(state: Dict[str, Any]) -> str:
            """按 Planner 给出的 route 决定第一个下游 Agent。"""
            route = state.get("route") or []
            for candidate in ("retriever", "analyst", "risk_checker"):
                if candidate in route:
                    return candidate
            return "writer"

        def route_after_risk(state: Dict[str, Any]) -> str:
            """反思循环的核心分支：打回重算 / 转人工 / 通过。"""
            verdict = str(state.get("risk_verdict") or "pass")
            if verdict == "revise":
                return "analyst"
            if verdict == "escalate":
                return "human_review"
            return "writer"

        def route_after_human(state: Dict[str, Any]) -> str:
            decision = (state.get("human_decision") or {}).get("decision", "rejected")
            return "writer" if decision == "approved" else "end"

        return {
            "after_planner": route_after_planner,
            "after_risk": route_after_risk,
            "after_human": route_after_human,
        }

    def _build_spec(self, recorder: TraceRecorder) -> GraphSpec:
        """为一次运行装配图（Agent 与 recorder 绑定）。"""
        ctx = AgentContext(
            registry=self.registry,
            llm=self.llm,
            recorder=recorder,
            config=self.config,
            extras={"pipeline": self},
        )
        agents = build_agents(ctx)
        routers = self._make_routers()

        def human_review_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """人机协同节点：高风险 / 循环超限时暂停等待人工确认。"""
            started = time.perf_counter()
            new_state = dict(state)
            risk_report = state.get("risk_report") or {}
            reason = str(risk_report.get("escalation_reason") or "")
            gaps = risk_report.get("gaps") or []
            if not reason:
                if gaps:
                    reason = (
                        f"反思循环已达上限（{state.get('max_revision_rounds')} 轮），"
                        f"仍有 {len(gaps)} 项数据缺口未消除。"
                    )
                else:
                    reason = "风险等级为 high，按流程需要人工确认。"

            payload = render_payload(state, reason)
            decision = self.reviewer.review(payload)
            new_state["human_decision"] = decision.to_dict()

            entry = recorder.record(
                agent="human_review",
                input_obj=payload,
                output_obj=decision.to_dict(),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                status="ok",
                extra={"needs_human": True, "risk_level": state.get("risk_level", "")},
            )
            steps = list(new_state.get("steps") or [])
            steps.append({"step": entry["step"], "agent": "human_review", "status": "ok",
                          "latency_ms": entry["latency_ms"], "tool_calls": []})
            new_state["steps"] = steps
            return new_state

        nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "planner": agents["planner"].run,
            "retriever": agents["retriever"].run,
            "analyst": agents["analyst"].run,
            "risk_checker": agents["risk_checker"].run,
            "human_review": human_review_node,
            "writer": agents["writer"].run,
        }

        return GraphSpec(
            state_schema=ResearchState,
            nodes=nodes,
            entry="planner",
            edges=[
                ("retriever", "analyst"),
                ("analyst", "risk_checker"),
                ("writer", END),
            ],
            conditional_edges=[
                (
                    "planner",
                    routers["after_planner"],
                    {"retriever": "retriever", "analyst": "analyst",
                     "risk_checker": "risk_checker", "writer": "writer"},
                ),
                (
                    "risk_checker",
                    routers["after_risk"],
                    {"analyst": "analyst", "human_review": "human_review", "writer": "writer"},
                ),
                (
                    "human_review",
                    routers["after_human"],
                    {"writer": "writer", "end": END},
                ),
            ],
        )

    # ------------------------------------------------------------------
    # 运行
    # ------------------------------------------------------------------
    def run(self, question: str, run_id: Optional[str] = None) -> Dict[str, Any]:
        """执行一次完整投研流程。"""
        run_id = run_id or make_run_id(question)
        recorder = TraceRecorder(run_id, runs_dir=self.runs_dir)
        spec = self._build_spec(recorder)
        self.last_spec = spec

        thread_id = run_id
        runner = build_engine(spec, self.config.engine, thread_id=thread_id)
        self.engine_name = getattr(runner, "engine_name", "unknown")

        state: ResearchState = new_state(
            question=question,
            run_id=run_id,
            started_at=datetime.now().isoformat(timespec="seconds"),
            config=self.config.to_dict(),
            max_revision_rounds=self.config.max_revision_rounds,
        )
        graph_config = {
            "recursion_limit": GRAPH_RECURSION_LIMIT,
            "configurable": {"thread_id": thread_id},
        }

        started = time.perf_counter()
        try:
            final = runner.invoke(state, config=graph_config)
        except Exception as exc:  # noqa: BLE001 - LangGraph 运行期异常时降级重跑
            if self.engine_name == "langgraph":
                recorder.record(
                    agent="engine_fallback",
                    input_obj={"engine": "langgraph"},
                    output_obj={"error": f"{type(exc).__name__}: {exc}"},
                    latency_ms=0.0,
                    status="error",
                )
                self.config.engine = "native"
                runner = build_engine(spec, "native", thread_id=thread_id)
                self.engine_name = getattr(runner, "engine_name", "native")
                final = runner.invoke(state, config=graph_config)
            else:
                raise
        total_ms = (time.perf_counter() - started) * 1000.0

        summary = self._summary(final, total_ms, run_id)
        recorder.close(summary)

        return {
            "run_id": run_id,
            "engine": self.engine_name,
            "trace_path": str(recorder.path),
            "state": final,
            "report": final.get("report", "") or "",
            "summary": summary,
            "llm_stats": self.llm.stats(),
        }

    # ------------------------------------------------------------------
    def _summary(self, state: Dict[str, Any], total_ms: float, run_id: str) -> Dict[str, Any]:
        """运行汇总（写入 trace 最后一行，也是 eval 的主要数据来源）。"""
        findings = state.get("findings") or []
        citations = state.get("citations") or []
        report = state.get("report") or ""
        return {
            "run_id": run_id,
            "status": "ok" if report else "no_report",
            "engine": getattr(self, "engine_name", "unknown"),
            "mode": "mock" if self.llm.is_mock else "openai-compatible",
            "question": state.get("question", ""),
            "total_latency_ms": round(total_ms, 3),
            "steps": len(state.get("steps") or []),
            "agents_visited": [s.get("agent") for s in (state.get("steps") or [])],
            "revision_round": state.get("revision_round", 0),
            "risk_verdict": state.get("risk_verdict", ""),
            "risk_level": state.get("risk_level", ""),
            "needs_human": state.get("needs_human", False),
            "human_decision": (state.get("human_decision") or {}).get("decision", ""),
            "findings": len(findings),
            "citations": len(citations),
            "report_chars": len(report),
            "evidence": len(state.get("evidence") or []),
            "errors": state.get("errors") or [],
            "snapshot": snapshot(state),  # type: ignore[arg-type]
        }


def make_run_id(question: str) -> str:
    """生成可读且唯一的运行编号：run-时间戳-问题指纹。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"run-{stamp}-{digest(question, 6)}"


def run_research(
    question: str,
    *,
    auto: bool = False,
    engine: Optional[str] = None,
    run_id: Optional[str] = None,
    runs_dir: Optional[Path] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """一次性执行投研流程的便捷入口（供 demo / eval / 测试复用）。"""
    pipeline = ResearchPipeline(auto=auto, engine=engine, runs_dir=runs_dir, quiet=quiet)
    return pipeline.run(question, run_id=run_id)
