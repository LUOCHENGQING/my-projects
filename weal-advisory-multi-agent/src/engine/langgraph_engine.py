"""LangGraph 编排引擎。

所属层次
--------
编排层（`src/engine/`）。上接入口层（`src/pipeline.py`、`src/demo.py`），
下用 Agent 层提供的节点函数（`src/agents/`）；本模块**不含任何业务判定**，
只描述"图怎么连、边怎么走"。

解决什么问题
------------
把投顾流水线表达成一张显式的有向图（`StateGraph`），使流程拓扑可枚举、
可测试、可画出来（`graph_topology`），并把「适当性闸门打回重配」这一
反馈环表达成**条件边**而不是嵌套 if。

双引擎分工与一致性要求
----------------------
分工：本模块是正式引擎（依赖 langgraph，支持条件边与可视化）；
`native.py` 是零依赖降级引擎（手写状态机，`--engine native` 可强切）。
两者共用 `src/pipeline.py` 的同一批节点函数与同一个路由函数，
因此结果必须逐字段一致（`tests/test_engine_parity.py` 做一致性对照）。

一致性契约：本模块的节点集合与连边必须与 `native.run_native` 的调用顺序一致；
条件边的路由键必须与 `AdvisoryPipeline.route_after_suitability` 的返回值
以及 `ROUTE_MAP` 的键完全对应，不得在引擎侧新增/改写路由判定。

使用 `StateGraph` 构建带**条件边**的投顾流水线，共享状态为 `AdvisoryState`：

    START → profile → screen → optimize → suitability
                              ↑              │
                              └── reoptimize ┘
                                             ├── pass ──────→ human_gate → narrative → END
                                             └── reject ────→ escalate ──→ human_gate → narrative → END

条件边由 `AdvisoryPipeline.route_after_suitability` 决定，与降级引擎共用同一判定，
保证两条路径结果一致。
（注：上图即 `build_graph` 的实际连边；`src/pipeline.py` 模块 docstring 里的
  示意图与降级引擎 `native.py` 的调用顺序三者一致。）

对外暴露
--------
- `langgraph_available()`：探测可选依赖是否可用（供 `resolve_engine` 决策）
- `ROUTE_MAP`：条件边映射表（路由键 → 节点名）
- `build_graph(pipeline)`：构建并编译 LangGraph 应用
- `run_langgraph(pipeline, client_id, run_id)`：跑完一位客户
- `graph_topology()`：导出邻接表（README 与测试用）

被谁调用
--------
`src/engine/__init__.py`（再导出；`resolve_engine` 依赖 `langgraph_available`）、
`src/pipeline.py` 的 `run_pipeline`（`langgraph` 引擎分支）、
`tests/test_engine_parity.py`。
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ..pipeline import AdvisoryState

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from ..pipeline import AdvisoryPipeline

#: 条件边映射表（路由键 -> 节点名）
#: 键必须是 `AdvisoryPipeline.route_after_suitability` 的**全部**可能返回值，
#: 且与 `graph_topology()["suitability"]` 的三个后继一一对应
#: （`tests/test_engine_parity.py` 会断言这一组键）。
ROUTE_MAP: dict[str, str] = {
    "optimize": "optimize",
    "human_gate": "human_gate",
    "escalate": "escalate",
}


def langgraph_available() -> bool:
    """当前环境是否可用 langgraph。

    参数：无。

    返回：
        bool —— 能 import `langgraph.graph` 为 True，否则 False。

    副作用/异常：
        吞掉一切导入异常（`noqa: BLE001`），只返回布尔值，绝不抛出；
        这样缺依赖时上层可以安全降级到 native。
    """
    try:
        import langgraph.graph  # noqa: F401
    except Exception:  # noqa: BLE001 - 缺依赖时静默降级
        return False
    return True


def build_graph(pipeline: "AdvisoryPipeline") -> Any:
    """构建并编译 LangGraph 应用。

    参数：
        pipeline: 提供 7 个节点方法（`node_profile` / `node_screen` /
            `node_optimize` / `node_suitability` / `node_escalate` /
            `node_human_gate` / `node_narrative`）与条件路由方法
            `route_after_suitability` 的流水线对象。本函数只做**方法引用绑定**，
            不读取其内部数据。

    返回：
        已 `compile()` 的 LangGraph 应用对象（可 `invoke(初始状态)`），
        类型为 Any —— 以便在未安装 langgraph 的环境里也能通过类型检查。

    副作用：
        仅构造图对象，不写文件、不调模型。

    异常：
        ImportError / ModuleNotFoundError: 未安装 langgraph
        （`from langgraph.graph import ...` 在函数内部执行，导入失败即抛出；
        上层应先调 `langgraph_available()` 判断）。
    """
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AdvisoryState)
    # 节点绑定：节点名与 pipeline 方法一一对应，节点内部行为全在 pipeline 里
    graph.add_node("profile", pipeline.node_profile)
    graph.add_node("screen", pipeline.node_screen)
    graph.add_node("optimize", pipeline.node_optimize)
    graph.add_node("suitability", pipeline.node_suitability)
    graph.add_node("escalate", pipeline.node_escalate)
    graph.add_node("narrative", pipeline.node_narrative)
    graph.add_node("human_gate", pipeline.node_human_gate)

    # 主干连边（与 native.run_native 的调用顺序保持一致）
    graph.add_edge(START, "profile")
    graph.add_edge("profile", "screen")
    graph.add_edge("screen", "optimize")
    graph.add_edge("optimize", "suitability")
    # 唯一的条件边：闸门结论决定 → optimize（打回重配）/ human_gate（通过）/ escalate（拒绝或超限）
    graph.add_conditional_edges("suitability", pipeline.route_after_suitability, ROUTE_MAP)
    # 汇合：escalate 与 pass 都经 human_gate 到 narrative，最后结束
    graph.add_edge("escalate", "human_gate")
    graph.add_edge("human_gate", "narrative")
    graph.add_edge("narrative", END)
    return graph.compile()


def run_langgraph(pipeline: "AdvisoryPipeline", client_id: str, run_id: str) -> AdvisoryState:
    """用 LangGraph 引擎跑完一位客户。

    参数：
        pipeline: 已构建的 `AdvisoryPipeline`（提供节点、路由与 `initial_state`）。
        client_id: 目标客户号。
        run_id: 本次运行的追踪号（trace / 版本链留痕用）。

    返回：
        终态共享状态；实现上是把 LangGraph 的最终 state 复制成普通 dict，
        再按 `AdvisoryState` 返回（`type: ignore` 处即此转换），
        调用方可直接用 `state["portfolio"]` 等键读取。

    副作用：
        每次调用都会重新 `build_graph`（重新编译图，不复用缓存），
        并执行各节点，从而写 trace / 版本链快照，还可能触发人工确认交互。
    """
    app = build_graph(pipeline)
    result = app.invoke(pipeline.initial_state(client_id, run_id))
    return dict(result)  # type: ignore[return-value]


def graph_topology() -> dict[str, list[str]]:
    """导出拓扑（README 与测试用）。

    参数：无。

    返回：
        邻接表：节点名 -> 后继节点名列表；终点用字符串 `"END"` 表示。
        `"suitability"` 的三个后继即 `ROUTE_MAP` 的取值（顺序与键无关）。

    副作用：无（纯常量视图，不读取 pipeline，也不导入 langgraph）。
    """
    return {
        "profile": ["screen"],
        "screen": ["optimize"],
        "optimize": ["suitability"],
        "suitability": ["optimize", "human_gate", "escalate"],
        "escalate": ["human_gate"],
        "human_gate": ["narrative"],
        "narrative": ["END"],
    }
