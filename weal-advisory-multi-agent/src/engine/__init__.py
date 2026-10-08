"""编排引擎包。

所属层次
--------
编排层（`src/engine/`）：位于入口层（`src/pipeline.py`、`src/demo.py`）与
Agent 层（`src/agents/`）之间，只决定"流程怎么走"，不做任何投顾业务判定。

解决的问题
----------
同一套投顾流程需要两种跑法：有第三方编排框架时用正式引擎，
没有依赖（离线 / 精简环境）时也要能完整跑通，且**结果必须一致**。
本包把「引擎选择」收敛成一个纯函数 `resolve_engine`，绝不静默失败。

双引擎分工
----------
- `langgraph_engine`：正式引擎（LangGraph StateGraph + 条件边）
- `native`：零依赖降级引擎（手写状态机，`--engine native` 可强切）
- `resolve_engine`：按可用性解析最终使用的引擎

一致性要求：两个引擎共用 `src/pipeline.py` 的节点函数与路由函数，
因此结果一致（`tests/test_engine_parity.py` 做路径与拓扑对照）；
新增节点或改路由时必须同时改两个引擎，否则 `graph_topology`、
`ROUTE_MAP` 与 `native.run_native` 的调用顺序会对不上。

对外暴露
--------
`resolve_engine`（引擎解析）、`SUPPORTED_ENGINES`（受支持引擎名）、
以及两个引擎的实现入口 `run_langgraph` / `run_native`、图工具
`build_graph` / `graph_topology` / `ROUTE_MAP` / `langgraph_available` / `MAX_STEPS`。

被谁调用
--------
`src/pipeline.py`（`run_pipeline` 先 `resolve_engine` 再分发）、
`src/demo.py`（CLI `--engine`）、`tests/test_engine_parity.py`。
"""

from __future__ import annotations

from .langgraph_engine import ROUTE_MAP, build_graph, graph_topology, langgraph_available, run_langgraph
from .native import MAX_STEPS, run_native

#: 受支持的引擎名
#: `langgraph` = 正式引擎；`native` = 零依赖降级引擎。
SUPPORTED_ENGINES: tuple[str, ...] = ("langgraph", "native")


def resolve_engine(requested: str) -> tuple[str, str]:
    """解析实际使用的引擎，返回 (引擎名, 说明)。

    `langgraph` 不可用时自动降级为 `native` 并给出说明，绝不静默失败。

    参数：
        requested: 请求的引擎名（大小写与首尾空白不敏感）；
            空串 / None 视为默认值 `"langgraph"`。

    返回：
        `(engine, note)` 二元组：
        - engine: 实际生效的引擎名，取值为 `SUPPORTED_ENGINES` 之一；
        - note: 降级说明（未降级时为空字符串）。

    副作用/异常：
        无副作用（不导入 langgraph，只调用 `langgraph_available()` 做探测）；
        requested 不在 `SUPPORTED_ENGINES` 内时抛 ValueError。
    """
    engine = (requested or "langgraph").strip().lower()
    if engine not in SUPPORTED_ENGINES:
        raise ValueError(f"不支持的引擎 {requested}，可选：{'、'.join(SUPPORTED_ENGINES)}")
    # 只在请求 langgraph 时探测依赖；探测失败即降级并给出可展示的说明
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
