"""双引擎路径一致性对照测试（LangGraph 正式引擎 vs 零依赖降级引擎）。

覆盖对象
--------
`src.engine`（`resolve_engine` / `run_native` / `run_langgraph` / `build_graph` /
`graph_topology` / `langgraph_available` / `SUPPORTED_ENGINES` / `ROUTE_MAP`）
以及 `src.pipeline` 的 `build_pipeline` / `state_summary` 与 `src.versioning`
的版本链落盘结果。

对照方法
--------
两个引擎共用 `src/pipeline.py` 的节点函数与路由函数，因此对同一客户、同一
mock 大脑、同一 `run_id` 各跑一遍，逐项比对应当完全一致：状态摘要、建议书正文
（归一化后）、要素集合，以及版本链上每一条快照。

比对前必须做两处归一化（见 `_normalize_engine_tokens`），否则会因无关差异误报：
- 引擎名 `langgraph` / `native` 统一替换为 `ENGINE`；
- 12 位十六进制快照哈希替换为 `HASH`（该哈希的输入包含引擎名，天然不同）。

跳过策略：`langgraph` 未安装时，对照类用例整体 `skip`（而不是失败）；
但 native 引擎可用性、路由解析、拓扑常量等用例始终执行，保证降级路径也被覆盖。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.engine import (
    SUPPORTED_ENGINES,
    build_graph,
    graph_topology,
    langgraph_available,
    resolve_engine,
    run_langgraph,
    run_native,
)
from src.engine.langgraph_engine import ROUTE_MAP
from src.observability import Tracer
from src.pipeline import build_pipeline, state_summary
from src.versioning import VersionStore

#: 三个对照客户分别代表：常规合规客户(C001) / 高龄且双录缺失、必须人工复核的客户(C004) /
#: 可行域为空、应被直接拒绝的反例客户(C006) —— 覆盖流水线的主要分支
PARITY_CLIENTS = ("C001", "C004", "C006")

#: 双引擎对照时使用的统一 run_id（避免运行标识长度差异污染文本比对）
PARITY_RUN_ID = "parity-run"


def _normalize_engine_tokens(text: str) -> str:
    """遮蔽引擎名与快照哈希，只比对业务内容（快照哈希包含引擎名，天然不同）。"""
    normalized = text.replace("langgraph", "ENGINE").replace("native", "ENGINE")
    # 快照哈希为 12 位十六进制，被反引号包裹地嵌在正文中
    return re.sub(r"`[0-9a-f]{12}`", "`HASH`", normalized)


def _run_engine(data, mock_llm, tmp_path: Path, client_id: str, engine: str):
    """在隔离的版本链与 trace 目录下真正跑一遍指定引擎，返回 `(state, tracer, pipeline)`。

    隔离手段：每个 `(引擎, 客户)` 组合使用独立的 `chain-*.jsonl` 与 `runs-*` 目录，
    避免两引擎互相读到对方的版本快照或 trace，导致对照结论失真。
    """
    store = VersionStore(tmp_path / f"{engine}-{client_id}.jsonl")
    tracer = Tracer(run_id=PARITY_RUN_ID, runs_dir=tmp_path / f"runs-{engine}")
    pipeline = build_pipeline(
        engine=engine,
        auto=True,
        interactive=False,
        runs_dir=tmp_path / f"runs-{engine}",
        store=store,
        llm=mock_llm,
    )[0]
    # build_pipeline 会自建 tracer，这里统一覆写为本次对照专用的 tracer，
    # 并把同一实例下发给所有 Agent，保证两条路径的 run_id 与 trace 口径一致
    pipeline.tracer = tracer
    for agent in pipeline.agents.values():
        agent.tracer = tracer
    if engine == "native":
        state = run_native(pipeline, client_id, tracer.run_id)
    else:
        state = run_langgraph(pipeline, client_id, tracer.run_id)
    return state, tracer, pipeline


@pytest.mark.skipif(not langgraph_available(), reason="环境未安装 langgraph")
@pytest.mark.parametrize("client_id", PARITY_CLIENTS)
def test_engines_produce_identical_summary(data, mock_llm, tmp_path, client_id):
    """对照不变式：两引擎的状态摘要逐字段一致（引擎名等天然不同的项先剔除）。"""
    lg_state, lg_tracer, _ = _run_engine(data, mock_llm, tmp_path, client_id, "langgraph")
    nt_state, nt_tracer, _ = _run_engine(data, mock_llm, tmp_path, client_id, "native")

    lg_summary = state_summary(lg_state)
    nt_summary = state_summary(nt_state)
    # 引擎名属于实现差异，不参与业务对照
    lg_summary.pop("engine")
    nt_summary.pop("engine")
    # trace_steps 的内含文本含引擎/路径信息，改为单独对照 tracer 的步数与 Agent 序列
    lg_summary.pop("trace_steps", None)
    nt_summary.pop("trace_steps", None)
    # 正文长度改用归一化后的文本重新计算，消除快照哈希长度带来的差异
    lg_summary["narrative_length"] = len(_normalize_engine_tokens(lg_state["advice"].narrative))
    nt_summary["narrative_length"] = len(_normalize_engine_tokens(nt_state["advice"].narrative))
    assert lg_summary == nt_summary

    assert lg_tracer.agent_sequence() == nt_tracer.agent_sequence()
    assert lg_tracer.step_count() == nt_tracer.step_count()


@pytest.mark.skipif(not langgraph_available(), reason="环境未安装 langgraph")
@pytest.mark.parametrize("client_id", PARITY_CLIENTS)
def test_engines_produce_identical_narrative_digest(data, mock_llm, tmp_path, client_id):
    """对照不变式：两引擎生成的建议书正文（归一化后）与要素集合完全一致。"""
    lg_state, _, _ = _run_engine(data, mock_llm, tmp_path, client_id, "langgraph")
    nt_state, _, _ = _run_engine(data, mock_llm, tmp_path, client_id, "native")
    assert _normalize_engine_tokens(lg_state["advice"].narrative) == _normalize_engine_tokens(
        nt_state["advice"].narrative
    )
    assert lg_state["advice"].elements == nt_state["advice"].elements


@pytest.mark.skipif(not langgraph_available(), reason="环境未安装 langgraph")
def test_engines_produce_identical_version_chain(data, mock_llm, tmp_path):
    """对照不变式：两引擎写出的版本链条数、版本号、状态、权重、指标与约束证据逐项一致。"""
    # 固定用 C004：它会经历"阻断 → 收紧 → 重配"的多版本链路，比对信息量最大
    lg_state, _, lg_pipeline = _run_engine(data, mock_llm, tmp_path, "C004", "langgraph")
    nt_state, _, nt_pipeline = _run_engine(data, mock_llm, tmp_path, "C004", "native")
    lg_chain = lg_pipeline.store.chain("C004")
    nt_chain = nt_pipeline.store.chain("C004")
    assert len(lg_chain) == len(nt_chain)
    for left, right in zip(lg_chain, nt_chain):
        assert left.version == right.version
        assert left.status == right.status
        assert left.change_reason == right.change_reason
        assert left.portfolio_weights == right.portfolio_weights
        assert left.cash_weight == right.cash_weight
        assert left.metrics == right.metrics
        assert left.client_constraints == right.client_constraints
        # verify() 校验快照自身的完整性哈希，两侧都必须自洽
        assert left.verify() and right.verify()


def test_native_engine_is_always_available(data, mock_llm, tmp_path):
    """保证：不依赖任何第三方图引擎的 native 路径必须能跑到终态。"""
    state, _, _ = _run_engine(data, mock_llm, tmp_path, "C001", "native")
    assert state["status"] == "final"


def test_resolve_engine_falls_back_when_langgraph_missing(monkeypatch):
    """降级契约：langgraph 不可用时自动降为 native，并给出「降级」说明（绝不静默失败）。"""
    monkeypatch.setattr("src.engine.langgraph_available", lambda: False)
    engine, note = resolve_engine("langgraph")
    assert engine == "native"
    assert "降级" in note


def test_resolve_engine_keeps_langgraph_when_available(monkeypatch):
    """正常路径：langgraph 可用时按请求保留，且不产生任何说明文案。"""
    monkeypatch.setattr("src.engine.langgraph_available", lambda: True)
    engine, note = resolve_engine("langgraph")
    assert engine == "langgraph"
    assert note == ""


def test_resolve_engine_rejects_unknown_engine():
    """异常路径：不支持的引擎名报 ValueError，而不是悄悄降级到 native。"""
    with pytest.raises(ValueError):
        resolve_engine("tensorflow")


def test_supported_engines_constant():
    """契约：受支持引擎常量恰好是 langgraph 与 native 两个。"""
    assert set(SUPPORTED_ENGINES) == {"langgraph", "native"}


def test_graph_topology_matches_route_map():
    """拓扑契约：条件边的目标节点集合与 `ROUTE_MAP` 一致，且各节点出边固定。"""
    topology = graph_topology()
    assert topology["suitability"] == ["optimize", "human_gate", "escalate"]
    assert set(ROUTE_MAP) == {"optimize", "human_gate", "escalate"}
    assert topology["escalate"] == ["human_gate"]
    assert topology["human_gate"] == ["narrative"]
    assert topology["narrative"] == ["END"]


@pytest.mark.skipif(not langgraph_available(), reason="环境未安装 langgraph")
def test_build_graph_compiles(data, mock_llm, tmp_path):
    """可编译性：LangGraph 图能被真正编译出可 `invoke` 的对象（而非仅有拓扑声明）。"""
    pipeline, _ = build_pipeline(engine="langgraph", auto=True, runs_dir=tmp_path, llm=mock_llm)
    app = build_graph(pipeline)
    assert hasattr(app, "invoke")


def test_langgraph_availability_matches_import():
    """探针一致性：`langgraph_available()` 的结论必须与真实 import 结果一致（探测逻辑不许说谎）。"""
    try:
        import langgraph.graph  # noqa: F401
    except Exception:  # noqa: BLE001
        assert langgraph_available() is False
    else:
        assert langgraph_available() is True
