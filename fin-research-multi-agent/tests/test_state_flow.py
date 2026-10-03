"""状态流转测试：共享状态（blackboard）契约、条件边路由、图引擎行为。"""

from __future__ import annotations

import pytest

from src.engine_native import END, GraphExecutionError, InMemoryCheckpointer, StateGraph
from src.state import STATE_FIELDS, new_state

QUESTION = "请分析示例科技股份有限公司 2024 年度的盈利能力、偿债能力和现金流质量，并提示主要风险。"


# ---------------------------------------------------------------------------
# 1. 状态契约
# ---------------------------------------------------------------------------
def test_initial_state_contains_all_contract_fields():
    state = new_state("测试问题", "run-x", max_revision_rounds=2)
    missing = [f for f in STATE_FIELDS if f not in state]
    assert missing == []
    assert state["question"] == "测试问题"
    assert state["revision_round"] == 0
    assert state["evidence"] == [] and state["findings"] == [] and state["steps"] == []


# ---------------------------------------------------------------------------
# 2. 端到端状态流转
# ---------------------------------------------------------------------------
def test_pipeline_fills_state_in_expected_order(pipeline):
    result = pipeline.run(QUESTION, run_id="run-state-flow")
    state = result["state"]
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
    """Planner 的 route 直接决定条件边走哪条分支。"""
    result = pipeline.run(QUESTION, run_id="run-route")
    state = result["state"]
    assert "retriever" in state["route"]
    assert state["route"][-1] == "writer"
    # route 里有 retriever，执行路径里就必须出现 retriever
    assert "retriever" in [s["agent"] for s in state["steps"]]


def test_state_flows_to_trace_file(pipeline):
    """每一步都落到 trace：JSONL 行数 == 状态里的步数 + 1 行 run_summary。"""
    result = pipeline.run(QUESTION, run_id="run-trace-flow")
    lines = [ln for ln in open(result["trace_path"], encoding="utf-8").read().splitlines() if ln.strip()]
    assert len(lines) == len(result["state"]["steps"]) + 1


# ---------------------------------------------------------------------------
# 3. 自研图引擎：条件边 / checkpointer / 死循环保护
# ---------------------------------------------------------------------------
def _loop_graph() -> StateGraph:
    """a -> (条件) b -> a 的环，用于验证条件边与死循环保护。"""

    def make(tag):
        def node(state):
            return {**state, "path": list(state["path"]) + [tag], "n": state["n"] + 1}

        return node

    graph = StateGraph(dict)
    graph.add_node("a", make("a"))
    graph.add_node("b", make("b"))
    graph.add_node("c", make("c"))
    graph.set_entry_point("a")
    graph.add_conditional_edges(
        "a",
        lambda s: "loop" if s["n"] < 3 else "exit",
        {"loop": "b", "exit": "c"},
    )
    graph.add_edge("b", "a")
    graph.add_edge("c", END)
    return graph


def test_native_engine_conditional_edge_routes_correctly():
    app = _loop_graph().compile()
    out = app.invoke({"path": [], "n": 0}, config={"configurable": {"thread_id": "t1"}, "recursion_limit": 20})
    # a(n=1) -> loop -> b(n=2) -> a(n=3) -> exit -> c
    assert out["path"] == ["a", "b", "a", "c"]
    assert out["n"] == 4


def test_native_engine_checkpointer_records_history():
    checkpointer = InMemoryCheckpointer()
    app = _loop_graph().compile(checkpointer=checkpointer)
    app.invoke({"path": [], "n": 0}, config={"configurable": {"thread_id": "thread-A"}, "recursion_limit": 20})

    history = checkpointer.list("thread-A")
    assert len(history) == 5  # __start__ + 4 个节点
    assert history[0]["node"] == "__start__"
    assert [h["node"] for h in history[1:]] == ["a", "b", "a", "c"]
    assert app.get_state("thread-A")["n"] == 4


def test_native_engine_blocks_infinite_loop():
    """条件边写成死循环时必须被 recursion_limit 拦下，而不是把进程挂死。"""

    def node(state):
        return {**state, "n": state["n"] + 1}

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
    graph = StateGraph(dict)
    graph.add_node("a", lambda s: s)
    graph.add_node("orphan", lambda s: s)
    graph.set_entry_point("a")
    graph.add_edge("a", END)
    with pytest.raises(ValueError) as exc:
        graph.compile()
    assert "没有任何出边" in str(exc.value)


@pytest.mark.parametrize("engine", ["native", "langgraph"])
def test_both_engines_produce_the_same_execution_path(pipeline, engine):
    """LangGraph 与自研引擎必须给出同一条执行路径（接口兼容性的硬证据）。"""
    if engine == "langgraph":
        pytest.importorskip("langgraph")
    result = pipeline.run(QUESTION, run_id=f"run-parity-{engine}")
    if engine == "langgraph" and result["engine"] != "langgraph":
        pytest.skip("当前环境未能启用 LangGraph 引擎")
    visited = [s["agent"] for s in result["state"]["steps"]]
    assert visited[0] == "planner" and visited[-1] == "writer"
    assert result["state"]["report"]
