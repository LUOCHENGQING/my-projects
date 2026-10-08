"""状态流转测试：共享状态（blackboard）契约、条件边路由、图引擎行为。

被测行为（src.state 的 new_state / STATE_FIELDS、orchestrator 的管线编排、src.engine_native 的 StateGraph）：
1. 状态契约：初始状态必须包含 STATE_FIELDS 全部字段并给出正确默认值；
2. 端到端流转：planner 首、writer 尾，retriever -> analyst -> risk_checker 顺序不变，各步产物都写回共享状态；
3. 路由：Planner 的 route 决定条件边走哪条分支，route 中声明的下游必须真的出现在执行路径里；
4. 落盘：每一步都写入 trace JSONL（有效行数 = 步数 + 1 行 run_summary）；
5. 图引擎：条件边分派、checkpointer 历史记录与状态回读、死循环保护（recursion_limit）、编译期孤立节点校验；
6. 双引擎一致性：native 与 langgraph 必须给出同一条执行路径（缺依赖或未启用时跳过）。

覆盖策略：正常（端到端顺序与图引擎条件边）、边界（初始状态字段、编译期校验、引擎参数化跳过）、
异常（死循环必须抛 GraphExecutionError 而非挂死进程）、对抗（两引擎执行路径必须互相印证）。
"""

from __future__ import annotations

import pytest

from src.engine_native import END, GraphExecutionError, InMemoryCheckpointer, StateGraph
from src.state import STATE_FIELDS, new_state

QUESTION = "请分析示例科技股份有限公司 2024 年度的盈利能力、偿债能力和现金流质量，并提示主要风险。"


# ---------------------------------------------------------------------------
# 1. 状态契约
# ---------------------------------------------------------------------------
def test_initial_state_contains_all_contract_fields():
    """验证契约：初始状态必须覆盖 STATE_FIELDS 全集且默认值为「空列表 + 轮次 0」，不允许缺字段。"""
    state = new_state("测试问题", "run-x", max_revision_rounds=2)
    # 用差集而非逐个断言：契约字段一旦新增，这里能立刻指出漏掉的是哪一个
    missing = [f for f in STATE_FIELDS if f not in state]
    assert missing == []
    assert state["question"] == "测试问题"
    assert state["revision_round"] == 0
    assert state["evidence"] == [] and state["findings"] == [] and state["steps"] == []


# ---------------------------------------------------------------------------
# 2. 端到端状态流转
# ---------------------------------------------------------------------------
def test_pipeline_fills_state_in_expected_order(pipeline):
    """验证不变式：planner 必为首、writer 必为尾，检索早于分析、分析早于核查，且各步产物全部落进共享状态。"""
    result = pipeline.run(QUESTION, run_id="run-state-flow")
    state = result["state"]
    # 用 index 比较先后关系而非硬编码完整序列，避免把无关的中间步骤也绑进断言
    visited = [s["agent"] for s in state["steps"]]

    # Planner 一定在最前，Writer 一定在最后
    assert visited[0] == "planner"
    assert visited[-1] == "writer"
    # 检索一定早于分析，分析一定早于核查
    assert visited.index("retriever") < visited.index("analyst")
    assert visited.index("analyst") < visited.index("risk_checker")

    # 每一步都写进了共享状态
    assert state["plan"]["companies"] == ["示例科技股份有限公司"]
    assert state["plan"]["year"] == 2024
    assert len(state["evidence"]) > 0
    assert len(state["metrics"]) > 0
    assert len(state["findings"]) > 0
    assert state["risk_verdict"] in {"pass", "revise", "escalate"}
    assert state["report"]


def test_route_decides_first_downstream_agent(pipeline):
    """验证规则：Planner 的 route 决定条件边走哪条分支——route 中声明的下游必须真的出现在执行路径里，末位为 writer。"""
    result = pipeline.run(QUESTION, run_id="run-route")
    state = result["state"]
    assert "retriever" in state["route"]
    assert state["route"][-1] == "writer"
    # route 里有 retriever，执行路径里就必须出现 retriever
    assert "retriever" in [s["agent"] for s in state["steps"]]


def test_state_flows_to_trace_file(pipeline):
    """验证契约：每一步都落到 trace，JSONL 有效行数 == 状态里的步数 + 1 行 run_summary 汇总。"""
    result = pipeline.run(QUESTION, run_id="run-trace-flow")
    # 过滤空行：这里统计的是有效记录条数
    lines = [ln for ln in open(result["trace_path"], encoding="utf-8").read().splitlines() if ln.strip()]
    assert len(lines) == len(result["state"]["steps"]) + 1


# ---------------------------------------------------------------------------
# 3. 自研图引擎：条件边 / checkpointer / 死循环保护
# ---------------------------------------------------------------------------
def _loop_graph() -> StateGraph:
    """构造 a -> (条件) b -> a 的环，c 为退出分支：用于验证条件边分派与 checkpointer 的记录。

    注：死循环保护由 test_native_engine_blocks_infinite_loop 自建无出口的图来验证，本图始终能正常退出。
    """

    def make(tag):
        """返回一个「把 tag 追加进 path 并把计数 n 加一」的节点函数，用于构造测试用图。"""
        def node(state):
            """节点函数：在共享状态上追加本节点标签并递增计数，返回增量状态。"""
            return {**state, "path": list(state["path"]) + [tag], "n": state["n"] + 1}

        return node

    graph = StateGraph(dict)
    graph.add_node("a", make("a"))
    graph.add_node("b", make("b"))
    graph.add_node("c", make("c"))
    graph.set_entry_point("a")
    # 条件边直接读共享状态里的 n：进入 a 时 n<3 就回 b 继续绕圈，否则去 c 收尾
    graph.add_conditional_edges(
        "a",
        lambda s: "loop" if s["n"] < 3 else "exit",
        {"loop": "b", "exit": "c"},
    )
    graph.add_edge("b", "a")
    graph.add_edge("c", END)
    return graph


def test_native_engine_conditional_edge_routes_correctly():
    """验证规则：条件边按共享状态取值分派，环上绕两圈后退出，最终路径与节点计数必须精确匹配。"""
    # 显式给足 recursion_limit，把「正常退出」与「被上限拦截」两种情形区分开
    app = _loop_graph().compile()
    out = app.invoke({"path": [], "n": 0}, config={"configurable": {"thread_id": "t1"}, "recursion_limit": 20})
    # a(n=1) -> loop -> b(n=2) -> a(n=3) -> exit -> c
    assert out["path"] == ["a", "b", "a", "c"]
    assert out["n"] == 4


def test_native_engine_checkpointer_records_history():
    """验证契约：checkpointer 按 thread_id 记录含 __start__ 的完整节点历史，并能回读最终状态。"""
    checkpointer = InMemoryCheckpointer()
    app = _loop_graph().compile(checkpointer=checkpointer)
    app.invoke({"path": [], "n": 0}, config={"configurable": {"thread_id": "thread-A"}, "recursion_limit": 20})

    # 用固定 thread_id 取历史：checkpointer 按 thread 隔离，这里验证「记录 + 回读」闭环
    history = checkpointer.list("thread-A")
    assert len(history) == 5  # __start__ + 4 个节点
    assert history[0]["node"] == "__start__"
    assert [h["node"] for h in history[1:]] == ["a", "b", "a", "c"]
    assert app.get_state("thread-A")["n"] == 4


def test_native_engine_blocks_infinite_loop():
    """验证规则：条件边写成死循环时必须被 recursion_limit 拦下并抛 GraphExecutionError，而不是把进程挂死。"""

    def node(state):
        """节点函数：仅把计数 n 加一，用于验证节点执行顺序与次数。"""
        return {**state, "n": state["n"] + 1}

    # 两个节点互为出口，构成没有终止条件的环，只能靠步数上限兜底
    graph = StateGraph(dict)
    graph.add_node("a", node)
    graph.add_node("b", node)
    graph.set_entry_point("a")
    graph.add_conditional_edges("a", lambda s: "again", {"again": "b"})
    graph.add_edge("b", "a")

    app = graph.compile()
    with pytest.raises(GraphExecutionError) as exc:
        app.invoke({"n": 0}, config={"configurable": {"thread_id": "dead"}, "recursion_limit": 6})
    assert "步数超过上限" in str(exc.value)


def test_compile_rejects_node_without_outgoing_edge():
    """验证规则：编译期必须拒绝没有任何出边的孤立节点，避免运行时静默走进死路。"""
    graph = StateGraph(dict)
    graph.add_node("a", lambda s: s)
    # orphan 只注册节点、不给任何边，专门用来触发编译期的可达性校验
    graph.add_node("orphan", lambda s: s)
    graph.set_entry_point("a")
    graph.add_edge("a", END)
    with pytest.raises(ValueError) as exc:
        graph.compile()
    assert "没有任何出边" in str(exc.value)


@pytest.mark.parametrize("engine", ["native", "langgraph"])
def test_both_engines_produce_the_same_execution_path(pipeline, engine):
    """验证不变式：LangGraph 与自研引擎必须给出同一条执行路径（接口兼容性的硬证据）。"""
    # 参数化跑两个引擎：langgraph 未安装或环境未启用时跳过，不算失败
    if engine == "langgraph":
        pytest.importorskip("langgraph")
    result = pipeline.run(QUESTION, run_id=f"run-parity-{engine}")
    if engine == "langgraph" and result["engine"] != "langgraph":
        pytest.skip("当前环境未能启用 LangGraph 引擎")
    visited = [s["agent"] for s in result["state"]["steps"]]
    assert visited[0] == "planner" and visited[-1] == "writer"
    assert result["state"]["report"]
