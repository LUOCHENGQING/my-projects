"""LangGraph 主引擎适配层。

把统一的 `GraphSpec` 编译成 LangGraph 的 StateGraph。之所以再包一层适配，
是为了让 orchestrator 不必关心底层是 LangGraph 还是自研引擎：
两者都只暴露 `invoke(state, config)`。

用到的 LangGraph 能力：
    * `StateGraph(ResearchState)` —— 以 TypedDict 作为共享状态（blackboard）schema
    * `add_conditional_edges`     —— 条件边（Planner 路由 / RiskChecker 反思回边 / HITL 分支）
    * `MemorySaver` checkpointer  —— 按 thread_id 保存每一步状态
    * `recursion_limit`           —— 防死循环保险丝
"""

from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = ["LANGGRAPH_AVAILABLE", "LangGraphRunner", "build_langgraph_runner"]

try:  # pragma: no cover - 取决于运行环境
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END as _LG_END
    from langgraph.graph import StateGraph as _LGStateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:  # noqa: BLE001
    LANGGRAPH_AVAILABLE = False
    _LGStateGraph = None  # type: ignore[assignment]
    _LG_END = "__end__"
    MemorySaver = None  # type: ignore[assignment]


class LangGraphRunner:
    """对 LangGraph 编译产物的薄封装，接口与自研引擎一致。"""

    engine_name = "langgraph"

    def __init__(self, compiled: Any, thread_default: str = "default") -> None:
        self._compiled = compiled
        self._thread_default = thread_default

    @property
    def raw(self) -> Any:
        """底层 LangGraph 对象（调试用）。"""
        return self._compiled

    def invoke(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        config = dict(config or {})
        config.setdefault("configurable", {"thread_id": self._thread_default})
        if "recursion_limit" not in config:
            config["recursion_limit"] = 40
        result = self._compiled.invoke(dict(state), config=config)
        return dict(result)

    def stream(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """以 updates 模式逐步产出 (节点名, 增量)。"""
        config = dict(config or {})
        config.setdefault("configurable", {"thread_id": self._thread_default})
        if "recursion_limit" not in config:
            config["recursion_limit"] = 40
        for chunk in self._compiled.stream(dict(state), config=config, stream_mode="updates"):
            if isinstance(chunk, dict):
                for node, update in chunk.items():
                    yield node, update

    def get_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = self._compiled.get_state(config)
        except Exception:  # noqa: BLE001
            return None
        return dict(snapshot.values) if snapshot and snapshot.values else None

    def history(self, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        try:
            return list(self._compiled.get_state_history(config))
        except Exception:  # noqa: BLE001
            return []


def build_langgraph_runner(spec: Any, thread_id: str = "default") -> LangGraphRunner:
    """按 GraphSpec 构建 LangGraph 编译产物。"""
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("当前环境未安装 langgraph，无法使用 LangGraph 引擎")

    graph = _LGStateGraph(spec.state_schema)
    for name, fn in spec.nodes.items():
        graph.add_node(name, fn)
    graph.set_entry_point(spec.entry)
    for src, dst in spec.edges:
        graph.add_edge(src, dst if dst != spec.end else _LG_END)
    for src, router, mapping in spec.conditional_edges:
        resolved = {
            key: (dst if dst != spec.end else _LG_END) for key, dst in mapping.items()
        }
        graph.add_conditional_edges(src, router, resolved)

    checkpointer = MemorySaver() if MemorySaver is not None else None
    compiled = graph.compile(checkpointer=checkpointer)
    return LangGraphRunner(compiled, thread_default=thread_id)
