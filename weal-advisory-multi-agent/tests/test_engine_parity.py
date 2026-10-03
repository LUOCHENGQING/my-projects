"""双引擎路径一致性对照测试（LangGraph vs 零依赖降级引擎）。"""

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

PARITY_CLIENTS = ("C001", "C004", "C006")

#: 双引擎对照时使用的统一 run_id（避免运行标识长度差异污染文本比对）
PARITY_RUN_ID = "parity-run"


def _normalize_engine_tokens(text: str) -> str:
    """遮蔽引擎名与快照哈希，只比对业务内容（快照哈希包含引擎名，天然不同）。"""
    normalized = text.replace("langgraph", "ENGINE").replace("native", "ENGINE")
    return re.sub(r"`[0-9a-f]{12}`", "`HASH`", normalized)


def _run_engine(data, mock_llm, tmp_path: Path, client_id: str, engine: str):
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
    lg_state, lg_tracer, _ = _run_engine(data, mock_llm, tmp_path, client_id, "langgraph")
    nt_state, nt_tracer, _ = _run_engine(data, mock_llm, tmp_path, client_id, "native")

    lg_summary = state_summary(lg_state)
    nt_summary = state_summary(nt_state)
    lg_summary.pop("engine")
    nt_summary.pop("engine")
    lg_summary.pop("trace_steps", None)
    nt_summary.pop("trace_steps", None)
    lg_summary["narrative_length"] = len(_normalize_engine_tokens(lg_state["advice"].narrative))
    nt_summary["narrative_length"] = len(_normalize_engine_tokens(nt_state["advice"].narrative))
    assert lg_summary == nt_summary

    assert lg_tracer.agent_sequence() == nt_tracer.agent_sequence()
    assert lg_tracer.step_count() == nt_tracer.step_count()


@pytest.mark.skipif(not langgraph_available(), reason="环境未安装 langgraph")
@pytest.mark.parametrize("client_id", PARITY_CLIENTS)
def test_engines_produce_identical_narrative_digest(data, mock_llm, tmp_path, client_id):
    lg_state, _, _ = _run_engine(data, mock_llm, tmp_path, client_id, "langgraph")
    nt_state, _, _ = _run_engine(data, mock_llm, tmp_path, client_id, "native")
    assert _normalize_engine_tokens(lg_state["advice"].narrative) == _normalize_engine_tokens(
        nt_state["advice"].narrative
    )
    assert lg_state["advice"].elements == nt_state["advice"].elements


@pytest.mark.skipif(not langgraph_available(), reason="环境未安装 langgraph")
def test_engines_produce_identical_version_chain(data, mock_llm, tmp_path):
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
        assert left.verify() and right.verify()


def test_native_engine_is_always_available(data, mock_llm, tmp_path):
    state, _, _ = _run_engine(data, mock_llm, tmp_path, "C001", "native")
    assert state["status"] == "final"


def test_resolve_engine_falls_back_when_langgraph_missing(monkeypatch):
    monkeypatch.setattr("src.engine.langgraph_available", lambda: False)
    engine, note = resolve_engine("langgraph")
    assert engine == "native"
    assert "降级" in note


def test_resolve_engine_keeps_langgraph_when_available(monkeypatch):
    monkeypatch.setattr("src.engine.langgraph_available", lambda: True)
    engine, note = resolve_engine("langgraph")
    assert engine == "langgraph"
    assert note == ""


def test_resolve_engine_rejects_unknown_engine():
    with pytest.raises(ValueError):
        resolve_engine("tensorflow")


def test_supported_engines_constant():
    assert set(SUPPORTED_ENGINES) == {"langgraph", "native"}


def test_graph_topology_matches_route_map():
    topology = graph_topology()
    assert topology["suitability"] == ["optimize", "human_gate", "escalate"]
    assert set(ROUTE_MAP) == {"optimize", "human_gate", "escalate"}
    assert topology["escalate"] == ["human_gate"]
    assert topology["human_gate"] == ["narrative"]
    assert topology["narrative"] == ["END"]


@pytest.mark.skipif(not langgraph_available(), reason="环境未安装 langgraph")
def test_build_graph_compiles(data, mock_llm, tmp_path):
    pipeline, _ = build_pipeline(engine="langgraph", auto=True, runs_dir=tmp_path, llm=mock_llm)
    app = build_graph(pipeline)
    assert hasattr(app, "invoke")


def test_langgraph_availability_matches_import():
    try:
        import langgraph.graph  # noqa: F401
    except Exception:  # noqa: BLE001
        assert langgraph_available() is False
    else:
        assert langgraph_available() is True
