"""自研极简图引擎（LangGraph 接口兼容的降级方案）。

当 `pip install langgraph` 失败时，编排层会自动切换到这里的实现，保证项目
在任何环境下都能跑通。它实现了 LangGraph 里本项目用到的全部能力：

    Node / Edge / ConditionalEdge / State / Checkpointer
    条件边（add_conditional_edges）、共享状态（dict blackboard）、
    入口点、编译、invoke、stream、内存检查点、递归步数上限。

接口刻意与 LangGraph 对齐（StateGraph / add_node / add_edge /
add_conditional_edges / set_entry_point / compile / invoke / stream），
所以 orchestrator 只需换一个 builder，业务代码零改动。

与 LangGraph 的差异（README 中如实说明）：
    * 不支持并行分支与 fan-in（本项目的拓扑是线性的 + 一条反思回边，用不到）；
    * 状态通道只有"整体覆盖"一种语义，不支持 Annotated reducer；
    * checkpointer 只在本地内存，不落盘、不支持时间旅行恢复到任意历史快照。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

__all__ = [
    "END",
    "Node",
    "Edge",
    "ConditionalEdge",
    "StateGraph",
    "CompiledGraph",
    "InMemoryCheckpointer",
    "GraphExecutionError",
]

#: 终止节点标记，与 LangGraph 的 END 常量保持同值
END = "__end__"

State = Dict[str, Any]
NodeFn = Callable[[State], State]
Router = Callable[[State], str]


class GraphExecutionError(RuntimeError):
    """图执行异常（步数超限 / 路由到未知节点等）。"""


@dataclass
class Node:
    name: str
    fn: NodeFn


@dataclass
class Edge:
    src: str
    dst: str


@dataclass
class ConditionalEdge:
    """条件边：src 执行完后由 router 决定去哪。"""

    src: str
    router: Router
    mapping: Dict[str, str] = field(default_factory=dict)

    def resolve(self, state: State) -> str:
        key = self.router(state)
        target = self.mapping.get(key, key)
        return target


class InMemoryCheckpointer:
    """极简内存检查点：按 thread_id 保存每一步的状态快照。"""

    def __init__(self, max_snapshots: int = 200) -> None:
        self._store: Dict[str, List[Dict[str, Any]]] = {}
        self.max_snapshots = max_snapshots

    def save(self, thread_id: str, step: int, node: str, state: State) -> None:
        history = self._store.setdefault(thread_id, [])
        history.append({"step": step, "node": node, "state": dict(state)})
        if len(history) > self.max_snapshots:
            del history[0 : len(history) - self.max_snapshots]

    def list(self, thread_id: str) -> List[Dict[str, Any]]:
        return list(self._store.get(thread_id, []))

    def latest(self, thread_id: str) -> Optional[Dict[str, Any]]:
        history = self._store.get(thread_id)
        return history[-1] if history else None

    def clear(self, thread_id: Optional[str] = None) -> None:
        if thread_id is None:
            self._store.clear()
        else:
            self._store.pop(thread_id, None)


class StateGraph:
    """图定义与编译。"""

    def __init__(self, state_schema: Optional[type] = None) -> None:
        self.state_schema = state_schema
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.conditional_edges: List[ConditionalEdge] = []
        self.entry: Optional[str] = None

    # ---------------- 定义 ----------------
    def add_node(self, name: str, fn: NodeFn) -> "StateGraph":
        if name in self.nodes:
            raise ValueError(f"节点重复定义：{name}")
        self.nodes[name] = Node(name=name, fn=fn)
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        self.edges.append(Edge(src=src, dst=dst))
        return self

    def add_conditional_edges(
        self,
        src: str,
        router: Router,
        mapping: Optional[Dict[str, str]] = None,
    ) -> "StateGraph":
        self.conditional_edges.append(
            ConditionalEdge(src=src, router=router, mapping=dict(mapping or {}))
        )
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self.entry = name
        return self

    # ---------------- 编译 ----------------
    def compile(self, checkpointer: Optional[InMemoryCheckpointer] = None) -> "CompiledGraph":
        if self.entry is None:
            raise ValueError("未设置入口节点：请调用 set_entry_point()")
        if self.entry not in self.nodes:
            raise ValueError(f"入口节点不存在：{self.entry}")

        targets: Dict[str, List[str]] = {name: [] for name in self.nodes}
        for edge in self.edges:
            self._check_node(edge.src)
            if edge.dst != END:
                self._check_node(edge.dst)
            targets[edge.src].append(edge.dst)
        for cond in self.conditional_edges:
            self._check_node(cond.src)
            for dst in cond.mapping.values():
                if dst != END:
                    self._check_node(dst)
            targets[cond.src].append(END)

        for name, outs in targets.items():
            if not outs and name != END:
                raise ValueError(f"节点 {name} 没有任何出边，会导致流程卡死")

        return CompiledGraph(
            nodes={n: node.fn for n, node in self.nodes.items()},
            edges={e.src: e.dst for e in self.edges},
            conditional={c.src: c for c in self.conditional_edges},
            entry=self.entry,
            checkpointer=checkpointer,
        )

    def _check_node(self, name: str) -> None:
        if name not in self.nodes:
            raise ValueError(f"引用了未定义的节点：{name}")


class CompiledGraph:
    """可执行的图。"""

    engine_name = "native"

    def __init__(
        self,
        nodes: Dict[str, NodeFn],
        edges: Dict[str, str],
        conditional: Dict[str, ConditionalEdge],
        entry: str,
        checkpointer: Optional[InMemoryCheckpointer] = None,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.conditional = conditional
        self.entry = entry
        self.checkpointer = checkpointer

    # ---------------- 执行 ----------------
    def _next_node(self, current: str, state: State) -> str:
        cond = self.conditional.get(current)
        if cond is not None:
            return cond.resolve(state)
        return self.edges.get(current, END)

    def stream(
        self,
        state: State,
        config: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Tuple[str, State]]:
        """逐步执行，逐个 yield (节点名, 状态)。"""
        config = config or {}
        recursion_limit = int(config.get("recursion_limit") or 40)
        thread_id = str((config.get("configurable") or {}).get("thread_id") or "default")

        current_state: State = dict(state)
        current = self.entry
        step = 0

        if self.checkpointer is not None:
            self.checkpointer.save(thread_id, step, "__start__", current_state)

        while current != END:
            if step >= recursion_limit:
                raise GraphExecutionError(
                    f"图执行步数超过上限 {recursion_limit}（thread_id={thread_id}），"
                    "疑似条件边形成死循环"
                )
            node_fn = self.nodes.get(current)
            if node_fn is None:
                raise GraphExecutionError(f"路由到未定义的节点：{current}")

            current_state = node_fn(current_state)
            step += 1

            if self.checkpointer is not None:
                self.checkpointer.save(thread_id, step, current, current_state)

            yield current, current_state
            current = self._next_node(current, current_state)

    def invoke(self, state: State, config: Optional[Dict[str, Any]] = None) -> State:
        """执行到终止节点，返回最终状态。"""
        final: State = dict(state)
        for _, snapshot in self.stream(state, config):
            final = snapshot
        return final

    # ---------------- 检查点 ----------------
    def get_state(self, thread_id: str) -> Optional[State]:
        if self.checkpointer is None:
            return None
        latest = self.checkpointer.latest(thread_id)
        return latest["state"] if latest else None

    def history(self, thread_id: str) -> List[Dict[str, Any]]:
        if self.checkpointer is None:
            return []
        return self.checkpointer.list(thread_id)
