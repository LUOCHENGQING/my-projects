"""LangGraph 编排引擎。

使用 `StateGraph` 构建带**条件边**的投顾流水线，共享状态为 `AdvisoryState`：

    START → profile → screen → optimize → suitability
                              ↑              │
                              └── reoptimize ┘
                                             ├── pass ──────→ human_gate → narrative → END
                                             └── reject ────→ escalate ──→ human_gate → narrative → END

条件边由 `AdvisoryPipeline.route_after_suitability` 决定，与降级引擎共用同一判定，
保证两条路径结果一致。
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ..pipeline import AdvisoryState

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from ..pipeline import AdvisoryPipeline

#: 条件边映射表（路由键 -> 节点名）
ROUTE_MAP: dict[str, str] = {
    "optimize": "optimize",
    "human_gate": "human_gate",
    "escalate": "escalate",
}


def langgraph_available() -> bool:
    """当前环境是否可用 langgraph。"""
    try:
        import langgraph.graph  # noqa: F401
    except Exception:  # noqa: BLE001 - 缺依赖时静默降级
        return False
    return True


def build_graph(pipeline: "AdvisoryPipeline") -> Any:
    """构建并编译 LangGraph 应用。"""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AdvisoryState)
    graph.add_node("profile", pipeline.node_profile)
    graph.add_node("screen", pipeline.node_screen)
    graph.add_node("optimize", pipeline.node_optimize)
    graph.add_node("suitability", pipeline.node_suitability)
    graph.add_node("escalate", pipeline.node_escalate)
    graph.add_node("narrative", pipeline.node_narrative)
    graph.add_node("human_gate", pipeline.node_human_gate)

    graph.add_edge(START, "profile")
    graph.add_edge("profile", "screen")
    graph.add_edge("screen", "optimize")
    graph.add_edge("optimize", "suitability")
    graph.add_conditional_edges("suitability", pipeline.route_after_suitability, ROUTE_MAP)
    graph.add_edge("escalate", "human_gate")
    graph.add_edge("human_gate", "narrative")
    graph.add_edge("narrative", END)
    return graph.compile()


def run_langgraph(pipeline: "AdvisoryPipeline", client_id: str, run_id: str) -> AdvisoryState:
    """用 LangGraph 引擎跑完一位客户。"""
    app = build_graph(pipeline)
    result = app.invoke(pipeline.initial_state(client_id, run_id))
    return dict(result)  # type: ignore[return-value]


def graph_topology() -> dict[str, list[str]]:
    """导出拓扑（README 与测试用）。"""
    return {
        "profile": ["screen"],
        "screen": ["optimize"],
        "optimize": ["suitability"],
        "suitability": ["optimize", "human_gate", "escalate"],
        "escalate": ["human_gate"],
        "human_gate": ["narrative"],
        "narrative": ["END"],
    }
