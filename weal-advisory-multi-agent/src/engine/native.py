"""零依赖降级引擎：手写状态机（`--engine native` 可强切）。

与 LangGraph 路径共用**同一批节点函数**与**同一个路由函数**，
因此两条路径的状态流转与最终结果完全一致（有单测做路径一致性对照）。
当环境里没有安装 langgraph 时，本引擎保证项目依然可以完整跑通。
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from ..pipeline import AdvisoryPipeline, AdvisoryState

#: 单次运行的最大节点步数（防御死循环）
MAX_STEPS = 64


def run_native(pipeline: "AdvisoryPipeline", client_id: str, run_id: str) -> "AdvisoryState":
    """按与 LangGraph 相同的拓扑手动推进状态机。"""
    state: "AdvisoryState" = pipeline.initial_state(client_id, run_id)
    steps = 0

    def apply(updates: dict[str, Any]) -> None:
        nonlocal steps
        steps += 1
        if steps > MAX_STEPS:
            raise RuntimeError(f"降级引擎步数超过上限 {MAX_STEPS}，疑似状态机未收敛")
        state.update(updates)  # type: ignore[typeddict-item]

    apply(pipeline.node_profile(state))
    apply(pipeline.node_screen(state))
    apply(pipeline.node_optimize(state))

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

    apply(pipeline.node_human_gate(state))
    apply(pipeline.node_narrative(state))
    return state
