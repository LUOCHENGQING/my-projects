"""LangGraph 主引擎适配层。

层次与职责
----------
编排层的引擎适配器：把 orchestrator 声明的 `GraphSpec`（引擎无关的图描述）翻译成
LangGraph 的 StateGraph 并编译执行。本模块不含任何业务判断。

关键类 / 函数：
    LANGGRAPH_AVAILABLE    —— 模块级可用性标志（import 失败即为 False）
    LangGraphRunner        —— 对编译产物的薄封装，对外只暴露 invoke / stream /
                              get_state / history 四个动作
    build_langgraph_runner —— 按 GraphSpec 构图并 compile

主要输入：GraphSpec（state_schema / nodes / entry / edges / conditional_edges / end）。
主要输出：已 compile 的 LangGraphRunner；其 invoke 返回最终状态 dict。
被谁调用：src/orchestrator.py 的 build_engine（auto 模式下优先走这里）。

一致性契约（与 engine_native 的等价关系）
----------------------------------------
两个引擎必须提供同一组能力，orchestrator 才能"只换 builder、业务代码零改动"：
    * 同名类属性 engine_name（本模块为 "langgraph"，自研引擎为 "native"），
      orchestrator 靠它判断运行期异常时能否降级重跑；
    * invoke(state, config) -> dict：跑完整图并返回最终状态；
    * stream(state, config) -> 迭代 (节点名, 增量)；
    * get_state(thread_id) / history(thread_id)：读取检查点。
唯一的语义差异在 stream 的粒度（见该方法说明）。tests/test_state_flow.py 的
test_both_engines_produce_the_same_execution_path 用同一个问题分别跑两个引擎并比对
执行路径，这是该契约的硬证据。

降级语义
--------
    1. 构建期：本模块 import 失败时 LANGGRAPH_AVAILABLE=False，orchestrator 直接改用
       engine_native；若 preference 显式写 "langgraph"，则 build_langgraph_runner 抛
       RuntimeError——显式要求就不静默降级，免得用户以为自己跑的是 LangGraph。
    2. 运行期：orchestrator 捕获 invoke 的异常，记一条 engine_fallback trace 后，
       用 native 引擎对同一个 state 重跑一遍。

把统一的 `GraphSpec` 编译成 LangGraph 的 StateGraph。之所以再包一层适配，
是为了让 orchestrator 不必关心底层是 LangGraph 还是自研引擎：
两者都只暴露 `invoke(state, config)`。

用到的 LangGraph 能力：
    * `StateGraph(ResearchState)` —— 以 TypedDict 作为共享状态（blackboard）schema
    * `add_conditional_edges`     —— 条件边（Planner 路由 / RiskChecker 反思回边 / HITL 分支）
    * `MemorySaver` checkpointer  —— 按 thread_id 保存每一步状态
    * `recursion_limit`           —— 防死循环保险丝

关于 END 常量：它在 langgraph 未安装时并不存在，所以这里用 `_LG_END` 别名承载——
GraphSpec 里的 `spec.end` 会被映射到它；import 失败时退回同值字符串 "__end__"
（与 engine_native.END 同值，这是两端终点语义能对齐的前提之一）。
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
    # 为什么把 ImportError 之外的所有异常也一起兜住：不同 langgraph 版本的模块路径、
    # 依赖冲突都可能抛别的异常，而这里的语义始终是"装不上就降级"，不该让启动挂掉。
    LANGGRAPH_AVAILABLE = False
    _LGStateGraph = None  # type: ignore[assignment]
    _LG_END = "__end__"
    MemorySaver = None  # type: ignore[assignment]


class LangGraphRunner:
    """对 LangGraph 编译产物的薄封装，接口与自研引擎一致。

    关键属性：
        _compiled        —— graph.compile() 的产物（含 checkpointer），一般不直接用
        _thread_default  —— 未显式传 thread_id 时使用的默认线程名
    状态流转：由 build_langgraph_runner 构造；构造后本对象不再改变，状态本身保存在
    LangGraph 的 checkpointer 里、按 thread_id 隔离（本项目每个 run_id 一个 thread）。
    """

    engine_name = "langgraph"

    def __init__(self, compiled: Any, thread_default: str = "default") -> None:
        """包住一个已编译的 LangGraph 对象。

        参数：compiled —— graph.compile() 的返回值；thread_default 默认线程名。
        返回：None。
        副作用：无（不触发任何执行）。
        """
        self._compiled = compiled
        self._thread_default = thread_default

    @property
    def raw(self) -> Any:
        """底层 LangGraph 对象（调试用）。

        参数：无（属性访问）。返回：编译产物本身。副作用：无。
        """
        return self._compiled

    def invoke(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """跑完整张图，返回最终状态。

        参数：
            state: 初始共享状态（本项目由 state.new_state 构造，会被复制一份传下去）。
            config: 可选 LangGraph config；未提供 configurable.thread_id 时用
                _thread_default，未提供 recursion_limit 时用 40（与
                config.GRAPH_RECURSION_LIMIT 的默认值一致）。
        返回：最终状态 dict（已复制，调用方改它不会写回 checkpointer）。
        副作用：会写 checkpointer（按 thread_id 记录每一步）。
        异常：图内节点抛的异常、超出 recursion_limit 时的 GraphRecursionError 都会向上抛，
            由 orchestrator 决定是否降级到 native 引擎重跑。
        """
        config = dict(config or {})
        config.setdefault("configurable", {"thread_id": self._thread_default})
        if "recursion_limit" not in config:
            config["recursion_limit"] = 40
        result = self._compiled.invoke(dict(state), config=config)
        return dict(result)

    def stream(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """以 updates 模式逐步产出 (节点名, 增量)。

        参数：state 初始状态；config 同 invoke（thread_id / recursion_limit 默认值一致）。
        返回：生成器，逐项 yield (node_name, update_dict)。
        副作用：写 checkpointer。
        注：实际实现用的是 stream_mode="updates"，所以 yield 出来的是**该节点的增量
            更新**；engine_native.CompiledGraph.stream 则 yield 累积后的全量状态。
            两者只保证"节点名序列一致"，增量/全量的语义差异由调用方自行对齐。
        """
        config = dict(config or {})
        config.setdefault("configurable", {"thread_id": self._thread_default})
        if "recursion_limit" not in config:
            config["recursion_limit"] = 40
        for chunk in self._compiled.stream(dict(state), config=config, stream_mode="updates"):
            # 为什么再拆一层：updates 模式的每个 chunk 形如 {节点名: 更新}，这里统一成
            # (节点名, 更新) 的二元组，才能与自研引擎的 yield 形状对齐。
            if isinstance(chunk, dict):
                for node, update in chunk.items():
                    yield node, update

    def get_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """读取某个线程的最新状态快照。

        参数：thread_id 线程名（本项目 = run_id）。
        返回：状态 dict；线程不存在或 checkpointer 不支持时返回 None。
        副作用：无。异常：底层异常被吞掉并降级为 None——读快照失败不该让主流程崩。
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = self._compiled.get_state(config)
        except Exception:  # noqa: BLE001
            return None
        return dict(snapshot.values) if snapshot and snapshot.values else None

    def history(self, thread_id: str):
        """读取某个线程的全部历史快照（LangGraph 的"时间旅行"能力）。

        参数：thread_id 线程名。
        返回：快照列表；出错时返回空列表 []（调用方按"没有历史"处理）。
        副作用：无。
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            return list(self._compiled.get_state_history(config))
        except Exception:  # noqa: BLE001
            return []


def build_langgraph_runner(spec: Any, thread_id: str = "default") -> LangGraphRunner:
    """按 GraphSpec 构建 LangGraph 编译产物。

    参数：
        spec: GraphSpec —— 引擎无关的图声明（nodes / entry / edges /
            conditional_edges / end / state_schema）。
        thread_id: 默认线程名，写进 runner 供未显式传 config 时兜底。
    返回：LangGraphRunner（已 compile，含 MemorySaver checkpointer）。
    副作用：无外部副作用（只在内存建图）。
    异常：当前环境无 langgraph 时抛 RuntimeError；构图/编译失败（如节点名重复、
        state_schema 不合法）时 LangGraph 自己的异常直接向上抛，本函数不吞。
    """
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("当前环境未安装 langgraph，无法使用 LangGraph 引擎")

    graph = _LGStateGraph(spec.state_schema)
    for name, fn in spec.nodes.items():
        graph.add_node(name, fn)
    graph.set_entry_point(spec.entry)
    # 为什么在构图时翻译终点标记：GraphSpec 用引擎无关的 spec.end，翻译只发生在这里，
    # 业务侧（orchestrator / agents）完全不需要知道 LangGraph 的 END 常量。
    for src, dst in spec.edges:
        graph.add_edge(src, dst if dst != spec.end else _LG_END)
    for src, router, mapping in spec.conditional_edges:
        resolved = {
            key: (dst if dst != spec.end else _LG_END) for key, dst in mapping.items()
        }
        graph.add_conditional_edges(src, router, resolved)

    # 为什么要判 None：MemorySaver 是在 import 块里拿到的，某些版本若模块路径变动会为 None；
    # 这时退化编译成"无检查点"的图，invoke 依然能跑通，只是 get_state / history 取不到东西。
    checkpointer = MemorySaver() if MemorySaver is not None else None
    compiled = graph.compile(checkpointer=checkpointer)
    return LangGraphRunner(compiled, thread_default=thread_id)
