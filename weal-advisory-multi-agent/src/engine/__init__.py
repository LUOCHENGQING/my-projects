"""编排引擎包。

- `langgraph_engine`：正式引擎（LangGraph StateGraph + 条件边）
- `native`：零依赖降级引擎（手写状态机，`--engine native` 可强切）
- `resolve_engine`：按可用性解析最终使用的引擎

两个引擎共用 `src/pipeline.py` 的节点函数与路由函数，因此结果一致（有单测对照）。
"""

from __future__ import annotations

from .langgraph_engine import ROUTE_MAP, build_graph, graph_topology, langgraph_available, run_langgraph
from .native import MAX_STEPS, run_native

#: 受支持的引擎名
SUPPORTED_ENGINES: tuple[str, ...] = ("langgraph", "native")


def resolve_engine(requested: str) -> tuple[str, str]:
    """解析实际使用的引擎，返回 (引擎名, 说明)。

    `langgraph` 不可用时自动降级为 `native` 并给出说明，绝不静默失败。
    """
    engine = (requested or "langgraph").strip().lower()
    if engine not in SUPPORTED_ENGINES:
        raise ValueError(f"不支持的引擎 {requested}，可选：{'、'.join(SUPPORTED_ENGINES)}")
    if engine == "langgraph" and not langgraph_available():
        return "native", "未检测到 langgraph，已自动降级到 native 引擎"
    return engine, ""


__all__ = [
    "MAX_STEPS",
    "ROUTE_MAP",
    "SUPPORTED_ENGINES",
    "build_graph",
    "graph_topology",
    "langgraph_available",
    "resolve_engine",
    "run_langgraph",
    "run_native",
]
