"""自研极简图引擎（LangGraph 接口兼容的降级方案）。

层次与职责
----------
编排层的备用引擎：当 langgraph 装不上、或 LangGraph 在运行期抛异常时接管执行。
它只实现"图怎么跑"，不含任何投研业务逻辑。

关键类：
    StateGraph        —— 图定义（add_node / add_edge / add_conditional_edges /
                         set_entry_point / compile）
    CompiledGraph     —— 可执行图（invoke / stream / get_state / history）
    InMemoryCheckpointer —— 按 thread_id 记录每一步状态快照的内存检查点
    GraphExecutionError  —— 图执行异常（步数超限 / 路由到未定义节点）
    Node / Edge / ConditionalEdge —— 内部数据结构；END 为终止标记 "__end__"

主要输入：dict 形式的共享状态（blackboard）+ graph_config（configurable.thread_id、
    recursion_limit）。
主要输出：invoke 返回最终状态 dict；stream 逐步产出 (节点名, 全量状态)。
被谁调用：src/orchestrator.py 的 build_engine（native 直接构建，或 LangGraph 降级后
    重建）；tests/test_state_flow.py 也直接用它测条件边与死循环保护。

一致性契约（与 engine_langgraph 的等价关系）
------------------------------------------
    * engine_name = "native"（LangGraph 侧为 "langgraph"）；
    * invoke(state, config) -> State：跑完整图；
    * stream(state, config) -> Iterator[(节点名, 状态)]；
    * get_state(thread_id) / history(thread_id)：读检查点（无 checkpointer 时分别返回
      None / []，与 LangGraph 侧"读不到就返回 None/[]"的行为对齐）；
    * 终点标记 END == "__end__"，与 LangGraph 的 END 同值；
    * 同一个 GraphSpec 在两个引擎下必须走出同一条节点序列——由
      tests/test_state_flow.py::test_both_engines_produce_the_same_execution_path 守护。
    唯一差异：本引擎 stream 产出的是**累积后的全量状态**，LangGraph 的 updates 模式产出
    的是**该节点的增量更新**。

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

降级语义（谁在什么时候切到这里）
--------------------------------
    * 构建期：config.GRAPH_ENGINE_PREF == "auto"（默认）且 LANGGRAPH_AVAILABLE 为 False
      -> 直接用本引擎；preference == "langgraph" 但未安装 -> 由 orchestrator 抛错，
      不静默替换。
    * 运行期：LangGraph runner.invoke 抛异常 -> orchestrator 记一条 engine_fallback
      trace，把 config.engine 改成 "native"，用同一个 state 新建本引擎重跑。
    因此本引擎的正确性直接决定"降级之后结论是否还可信"，它与主引擎的等价性由测试兜底。
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
    """图执行异常（步数超限 / 路由到未知节点等）。

    继承 RuntimeError，便于 orchestrator 与 LangGraph 的运行时异常一起兜住；
    消息里带 thread_id 与步数上限，方便定位是哪次运行、哪条边绕成了环。
    """


@dataclass
class Node:
    """一个节点：名字 + 执行函数。

    fn 的契约是 State -> State（**整体覆盖**语义：返回的必须是完整状态，
    本项目里各个 Agent 都是先 copy 再改字段）。
    """

    name: str
    fn: NodeFn


@dataclass
class Edge:
    """一条普通边：src 执行完固定走 dst（dst 可以是 END）。"""

    src: str
    dst: str


@dataclass
class ConditionalEdge:
    """条件边：src 执行完后由 router 决定去哪。

    关键属性：src 起点节点名；router State -> str 的路由函数（返回 mapping 的键，
        也允许直接返回节点名）；mapping 路由键 -> 目标节点名。
    """

    src: str
    router: Router
    mapping: Dict[str, str] = field(default_factory=dict)

    def resolve(self, state: State) -> str:
        """按当前状态决定下一个节点名。

        参数：state 走到 src 之后的共享状态。
        返回：目标节点名（mapping 未命中时把 router 的返回值当节点名直接用——
            这既支持"key -> node"映射，也支持 router 直接返回节点名）。
        副作用：无。router 里抛的异常会向上冒泡。
        """
        key = self.router(state)
        target = self.mapping.get(key, key)
        return target


class InMemoryCheckpointer:
    """极简内存检查点：按 thread_id 保存每一步的状态快照。

    关键属性：
        _store         —— thread_id -> [{"step", "node", "state"}, ...] 的字典
        max_snapshots  —— 单线程保留的快照上限（默认 200），超出后丢弃最旧的
    状态流转：save 追加 -> 超过上限裁剪 -> latest 取最后一条；clear 可按线程或整体清空。
    注：进程内存储，不落盘。save 里对 state 做的是**浅拷贝**（dict(state)），嵌套的
    list/dict 仍与当时的运行状态共享引用——本项目各 Agent 都遵守"浅拷贝后再改"的约定，
    所以已存快照在实践中不会被后续步骤改写。
    """

    def __init__(self, max_snapshots: int = 200) -> None:
        """初始化空检查点。

        参数：max_snapshots 每个 thread_id 最多保留的快照数（<=0 时每条新快照都会把
            历史裁到只剩最近 max_snapshots 条，即 0 条，属边界用法，未做校验）。
        返回：None。副作用：无（仅建一个空字典）。
        """
        self._store: Dict[str, List[Dict[str, Any]]] = {}
        self.max_snapshots = max_snapshots

    def save(self, thread_id: str, step: int, node: str, state: State) -> None:
        """记录一步快照。

        参数：thread_id 线程名；step 步号（__start__ 为 0，之后每执行完一个节点 +1）；
            node 刚执行完的节点名（起始快照固定为 "__start__"）；state 该步之后的状态。
        返回：None。
        副作用：向内存追加一条记录；超过 max_snapshots 时删除最旧的若干条——
            为什么要有上限：长跑或高频调用时快照会随步数线性增长，不设上限就是内存泄漏。
        """
        history = self._store.setdefault(thread_id, [])
        history.append({"step": step, "node": node, "state": dict(state)})
        if len(history) > self.max_snapshots:
            del history[0 : len(history) - self.max_snapshots]

    def list(self, thread_id: str) -> List[Dict[str, Any]]:
        """取某线程的全部快照（按时间顺序）。

        参数：thread_id 线程名。返回：快照列表的浅拷贝（外层新列表、元素仍是原字典）；
            线程不存在时返回 []。副作用：无。
        """
        return list(self._store.get(thread_id, []))

    def latest(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """取某线程的最后一条快照。

        参数：thread_id 线程名。返回：快照字典；无记录时返回 None。副作用：无。
        """
        history = self._store.get(thread_id)
        return history[-1] if history else None

    def clear(self, thread_id: Optional[str] = None) -> None:
        """清理快照。

        参数：thread_id 为 None 时清空全部线程，否则只清指定线程。
        返回：None。副作用：丢弃内存中的历史状态（不可恢复）。
        """
        if thread_id is None:
            self._store.clear()
        else:
            self._store.pop(thread_id, None)


class StateGraph:
    """图定义与编译（接口与 LangGraph 的 StateGraph 对齐）。

    关键属性：
        state_schema     —— 状态 schema（本项目传 ResearchState）；注：实际实现里只是
                            为对齐 LangGraph 接口而保存，编译与执行都不使用它，
                            状态就是一个普通 dict
        nodes            —— name -> Node
        edges            —— List[Edge]（普通边）
        conditional_edges —— List[ConditionalEdge]（条件边）
        entry            —— 入口节点名，未设置时 compile 会报错

    状态流转：定义期（add_node / add_edge / ... 全部返回 self，可链式书写）
    -> compile() 校验拓扑并产出 CompiledGraph；StateGraph 自身不再参与执行。
    """

    def __init__(self, state_schema: Optional[type] = None) -> None:
        """创建一张空图。

        参数：state_schema 状态类型（可选，仅作接口对齐与文档用途）。
        返回：None。副作用：无。
        """
        self.state_schema = state_schema
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.conditional_edges: List[ConditionalEdge] = []
        self.entry: Optional[str] = None

    # ---------------- 定义 ----------------
    def add_node(self, name: str, fn: NodeFn) -> "StateGraph":
        """注册一个节点。

        参数：name 节点名（全局唯一）；fn State -> State 的执行函数（须返回完整状态）。
        返回：self（支持链式调用）。
        副作用：写入 nodes。异常：重复定义同名节点抛 ValueError。
        """
        if name in self.nodes:
            raise ValueError(f"节点重复定义：{name}")
        self.nodes[name] = Node(name=name, fn=fn)
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        """注册一条固定边。

        参数：src 起点节点名；dst 目标节点名（可以是 END）。
        返回：self（链式）。副作用：追加到 edges。
        注：这里不校验节点是否存在，留到 compile 统一检查（LangGraph 亦然）。
        """
        self.edges.append(Edge(src=src, dst=dst))
        return self

    def add_conditional_edges(
        self,
        src: str,
        router: Router,
        mapping: Optional[Dict[str, str]] = None,
    ) -> "StateGraph":
        """注册一条条件边。

        参数：src 起点节点名；router State -> str 的路由函数；mapping 路由键 -> 目标
            节点名（可省略，此时 router 的返回值被直接当作节点名）。
        返回：self（链式）。副作用：追加到 conditional_edges（mapping 会被复制一份）。
        """
        self.conditional_edges.append(
            ConditionalEdge(src=src, router=router, mapping=dict(mapping or {}))
        )
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        """设置入口节点（执行从这里开始）。

        参数：name 节点名。返回：self（链式）。副作用：覆盖 self.entry。
        """
        self.entry = name
        return self

    # ---------------- 编译 ----------------
    def compile(self, checkpointer: Optional[InMemoryCheckpointer] = None) -> "CompiledGraph":
        """校验拓扑并把定义固化成可执行图。

        参数：checkpointer 可选的内存检查点；传入后每次执行都会按 thread_id 记快照。
        返回：CompiledGraph（节点函数、边表、条件边表、入口、检查点全部拷贝进去）。
        副作用：无（不改动 StateGraph 自身）。
        异常：
            * 未设置入口节点 -> ValueError("未设置入口节点…")；
            * 入口/边/条件边引用了不存在的节点 -> ValueError("引用了未定义的节点…")；
            * 有节点没有任何出边 -> ValueError("节点 … 没有任何出边，会导致流程卡死")。
              注：实际实现为——真跑起来时 _next_node 的 edges.get(current, END) 会退化成
              "直接结束"，即静默提前终止而不是卡死；编译期拦下来是为了把这种沉默的
              提前收尾变成显式错误。
        """
        if self.entry is None:
            raise ValueError("未设置入口节点：请调用 set_entry_point()")
        if self.entry not in self.nodes:
            raise ValueError(f"入口节点不存在：{self.entry}")

        # 为什么先建一张出边表：把"每个节点能去哪"摊平之后，缺出边的节点能在下面被
        # 一次性揪出来，而不是等运行到那一步才发现流程莫名结束。
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

        # 为什么只把 fn 拷进 CompiledGraph：执行期只需要函数本体，不必再留 Node 包装。
        # 注：edges 用字典推导，同一 src 若注册了多条普通边，只有最后一条会生效
        # （与"不支持并行分支"的定位一致）。
        return CompiledGraph(
            nodes={n: node.fn for n, node in self.nodes.items()},
            edges={e.src: e.dst for e in self.edges},
            conditional={c.src: c for c in self.conditional_edges},
            entry=self.entry,
            checkpointer=checkpointer,
        )

    def _check_node(self, name: str) -> None:
        """校验节点名已定义（编译期的一道集中校验）。

        参数：name 被引用的节点名。返回：None。
        副作用：无。异常：节点不存在时抛 ValueError（消息里带节点名，便于定位拼写错误）。
        """
        if name not in self.nodes:
            raise ValueError(f"引用了未定义的节点：{name}")


class CompiledGraph:
    """可执行的图（由 StateGraph.compile 产出，接口与 LangGraph 的编译产物对齐）。

    关键属性：
        nodes        —— name -> NodeFn（只留函数，执行期不需要 Node 包装）
        edges        —— src -> dst 的普通边表（同一 src 多条边时只保留最后一条）
        conditional  —— src -> ConditionalEdge
        entry        —— 入口节点名
        checkpointer —— 可选的内存检查点，None 时 get_state / history 返回空值
        engine_name  —— 固定为 "native"，orchestrator 靠它标记 trace 与降级决策

    状态流转：编译后只读；每次 stream / invoke 都是独立的执行过程，
    执行状态只体现在传入的 state 与 checkpointer 里。
    """

    engine_name = "native"

    def __init__(
        self,
        nodes: Dict[str, NodeFn],
        edges: Dict[str, str],
        conditional: Dict[str, ConditionalEdge],
        entry: str,
        checkpointer: Optional[InMemoryCheckpointer] = None,
    ) -> None:
        """组装可执行图（一般由 StateGraph.compile 调用，不建议手工构造）。

        参数：nodes 节点函数表；edges 普通边表；conditional 条件边表；entry 入口节点名；
            checkpointer 可选检查点。
        返回：None。副作用：无（只保存引用，不复制传入的字典）。
        """
        self.nodes = nodes
        self.edges = edges
        self.conditional = conditional
        self.entry = entry
        self.checkpointer = checkpointer

    # ---------------- 执行 ----------------
    def _next_node(self, current: str, state: State) -> str:
        """决定 current 执行完之后去哪。

        参数：current 刚执行完的节点名；state 该节点产出的状态。
        返回：下一个节点名，或 END。
        副作用：无（会调用条件边的 router，router 自身的异常向上冒泡）。
        注：条件边优先于普通边——同一节点若两者都配了，以条件边为准；
        没有任何出边的节点会退化为 END（编译期本应已被拦下）。
        """
        cond = self.conditional.get(current)
        if cond is not None:
            return cond.resolve(state)
        return self.edges.get(current, END)

    def stream(
        self,
        state: State,
        config: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Tuple[str, State]]:
        """逐步执行，逐个 yield (节点名, 状态)。

        参数：
            state: 初始共享状态（内部会 dict(state) 复制一份，不改调用方对象）。
            config: 可选；configurable.thread_id 决定快照归到哪个线程（默认 "default"），
                recursion_limit 决定最多执行多少个节点（默认 40，与 LangGraph 侧一致）。
        返回：生成器，每执行完一个节点 yield (节点名, **累积后的全量状态**)。
        副作用：有 checkpointer 时按 thread_id 写快照（起始快照节点名为 "__start__"）。
        异常：步数达到 recursion_limit 仍在循环时抛 GraphExecutionError；路由到未定义
            节点同样抛 GraphExecutionError（带 thread_id，便于定位是哪次运行绕成了环）。
        注：与 engine_langgraph.LangGraphRunner.stream（updates 模式，yield 增量）不同，
            这里 yield 的是全量状态。
        """
        config = config or {}
        # 为什么用 `or 40`：None / 0 / 空串一律退回默认上限（40，与 LangGraph 侧同值），
        # 免得把 0 理解成"一步都不许走"。
        recursion_limit = int(config.get("recursion_limit") or 40)
        thread_id = str((config.get("configurable") or {}).get("thread_id") or "default")

        current_state: State = dict(state)
        current = self.entry
        step = 0

        # 为什么要存起始快照：让 history[0] 就是初始状态，回溯"哪一步把状态改坏了"
        # 时有一个干净的基线可比。
        if self.checkpointer is not None:
            self.checkpointer.save(thread_id, step, "__start__", current_state)

        while current != END:
            # 为什么在节点执行**之前**判上限：这样"最多执行 recursion_limit 个节点"就是
            # 硬保证，条件边即使被写成自环也只会报错，不会把进程挂死。
            if step >= recursion_limit:
                raise GraphExecutionError(
                    f"图执行步数超过上限 {recursion_limit}（thread_id={thread_id}），"
                    "疑似条件边形成死循环"
                )
            node_fn = self.nodes.get(current)
            if node_fn is None:
                raise GraphExecutionError(f"路由到未定义的节点：{current}")

            # 整体覆盖语义：节点必须返回完整 state，这里不做 merge、也没有 reducer
            # （见模块 docstring 与 LangGraph 的差异说明）。
            current_state = node_fn(current_state)
            step += 1

            if self.checkpointer is not None:
                self.checkpointer.save(thread_id, step, current, current_state)

            # 先 yield 再算下一步：调用方拿到的是"该节点做完之后"的状态，与节点名严格配对。
            yield current, current_state
            current = self._next_node(current, current_state)

    def invoke(self, state: State, config: Optional[Dict[str, Any]] = None) -> State:
        """执行到终止节点，返回最终状态。

        参数：state 初始状态；config 同 stream（thread_id / recursion_limit）。
        返回：最后一个节点产出的完整状态；图一步都没走时返回初始状态的副本。
        副作用：同 stream（写检查点）。
        实现说明：直接消费 stream 的最后一个快照，因此 invoke 的语义与
        "for _, s in stream(...): last = s" 完全一致。
        """
        final: State = dict(state)
        for _, snapshot in self.stream(state, config):
            final = snapshot
        return final

    # ---------------- 检查点 ----------------
    def get_state(self, thread_id: str) -> Optional[State]:
        """取某线程最新一步的状态（对齐 LangGraph 的 get_state）。

        参数：thread_id 线程名。返回：状态 dict；无 checkpointer 或无记录时返回 None。
        副作用：无。
        """
        if self.checkpointer is None:
            return None
        latest = self.checkpointer.latest(thread_id)
        return latest["state"] if latest else None

    def history(self, thread_id: str) -> List[Dict[str, Any]]:
        """取某线程的全部快照（对齐 LangGraph 的 get_state_history）。

        参数：thread_id 线程名。返回：快照列表；无 checkpointer 时返回 []。
        副作用：无。用途：定位"哪一步把状态改坏了"（含 __start__ 起始快照）。
        """
        if self.checkpointer is None:
            return []
        return self.checkpointer.list(thread_id)
