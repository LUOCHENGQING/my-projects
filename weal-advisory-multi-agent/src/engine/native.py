"""零依赖降级引擎：手写状态机（`--engine native` 可强切）。

所属层次
--------
编排层（`src/engine/`）。上接入口层（`src/pipeline.py` 的 `run_pipeline`、
`src/demo.py` 的 CLI），下用 Agent 层提供的节点函数（`src/agents/`）。

解决什么问题
------------
当运行环境没有安装 `langgraph`，或用户显式指定 `--engine native` 时，
仍要用**完全相同的节点函数与路由判定**把一位客户的投顾流程跑完，
使「合规闸门等核心能力不依赖任何第三方编排框架」——
即使离线环境也能完整复现同一份投顾结论。

双引擎分工与一致性要求
----------------------
分工：`langgraph_engine` 是正式引擎（StateGraph + 条件边，便于可视化与扩展）；
本模块是零依赖降级引擎（手写 while 循环），二者**只负责"怎么走"**，
所有业务判定都在 `AdvisoryPipeline` 的节点里，引擎不得内联任何业务逻辑。

与 LangGraph 路径共用**同一批节点函数**与**同一个路由函数**
（`node_profile` / `node_screen` / `node_optimize` / `node_suitability` /
`node_escalate` / `node_human_gate` / `node_narrative` 与 `route_after_suitability`），
因此两条路径的状态流转与最终结果完全一致（有单测做路径一致性对照）。
当环境里没有安装 langgraph 时，本引擎保证项目依然可以完整跑通。

一致性契约（改动任一引擎时必须同时满足）：
1. 节点集合与调用顺序必须与 `langgraph_engine.build_graph` 的连边逐一对应：
   `profile → screen → optimize → suitability →
   （pass: human_gate / reoptimize: optimize 重配 / reject: escalate → human_gate）
   → narrative`；
2. 路由只认 `route_after_suitability` 的返回值（`human_gate` / `optimize` / `escalate`），
   引擎自己**不解释**闸门结论，也不判断 `GateDecision`；
3. 每个节点返回的增量更新都要经同一套合并方式写回共享状态
   （本引擎即 `state.update(updates)`，与 LangGraph 的状态合并语义在
   节点全部返回"整键覆盖"型更新的前提下等价）。

对外暴露
--------
- `run_native(pipeline, client_id, run_id)`：跑完一位客户，返回最终 `AdvisoryState`
- `MAX_STEPS`：单次运行的节点步数上限（防御死循环）

被谁调用
--------
`src/pipeline.py`（`run_pipeline` 按 `resolve_engine` 的结果分发；
`src/pipeline.py` 里另有一个同名薄封装）、`src/engine/__init__.py`（再导出）、
`tests/test_pipeline.py`、`tests/test_engine_parity.py`。
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from ..pipeline import AdvisoryPipeline, AdvisoryState

#: 单次运行的最大节点步数（防御死循环）
#: 计数口径是**节点执行次数**（含打回重配时的 optimize 重复执行），
#: 并非闸门轮次；超出即抛 RuntimeError，避免状态机不收敛时静默挂死。
MAX_STEPS = 64


def run_native(pipeline: "AdvisoryPipeline", client_id: str, run_id: str) -> "AdvisoryState":
    """按与 LangGraph 相同的拓扑手动推进状态机。

    参数：
        pipeline: 已构建好的 `AdvisoryPipeline`，提供全部节点函数、路由函数与
            `initial_state`（引擎只调用它，不关心其内部依赖）。
        client_id: 目标客户号；由 `pipeline.initial_state` 解析成完整客户档案。
        run_id: 本次运行的追踪号，用于 trace / 版本链留痕。

    返回：
        跑完全流程后的共享状态 `AdvisoryState`（原地累积的同一份 dict），
        其 `status` 字段为 `final*` / `rejected*` / `blocked` 等终态取值。

    副作用：
        会调用各节点函数，进而写 trace、写版本链快照（由 pipeline 负责），
        并可能执行人工确认交互（`node_human_gate`）。

    异常：
        RuntimeError: 节点执行次数超过 `MAX_STEPS`（疑似状态机未收敛）。
    """
    state: "AdvisoryState" = pipeline.initial_state(client_id, run_id)
    steps = 0

    def apply(updates: dict[str, Any]) -> None:
        """执行一次步数记账并把节点返回的增量合并进共享状态。

        参数：
            updates: 某个节点函数返回的状态增量字典。

        返回：
            None（原样原地更新外层 `state`）。

        副作用：
            `steps` 自增；超过 `MAX_STEPS` 时抛 RuntimeError。

        异常：
            RuntimeError: 步数超限。
        """
        nonlocal steps
        steps += 1
        if steps > MAX_STEPS:
            raise RuntimeError(f"降级引擎步数超过上限 {MAX_STEPS}，疑似状态机未收敛")
        state.update(updates)  # type: ignore[typeddict-item]

    # 前四步与 LangGraph 的连边一致：profile → screen → optimize → suitability
    apply(pipeline.node_profile(state))
    apply(pipeline.node_screen(state))
    apply(pipeline.node_optimize(state))

    # 闸门循环：pass 跳出到 human_gate；reoptimize 回到 optimize 重配；
    # 其余（reject，含 veto 与打回超限）先走 escalate 再汇入 human_gate。
    while True:
        apply(pipeline.node_suitability(state))
        route = pipeline.route_after_suitability(state)
        if route == "human_gate":
            break
        if route == "optimize":
            apply(pipeline.node_optimize(state))
            continue
        apply(pipeline.node_escalate(state))
        break

    # 注：顺序为 human_gate → narrative，与 langgraph_engine 的连边
    # add_edge("human_gate", "narrative") 以及 pipeline.py 的示意图一致。
    apply(pipeline.node_human_gate(state))
    apply(pipeline.node_narrative(state))
    return state
