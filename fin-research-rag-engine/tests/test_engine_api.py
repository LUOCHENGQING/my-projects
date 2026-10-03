"""引擎、接口、轨迹与兜底服务的集成测试。"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from src.engine import RAGEngine, question_entities, unknown_entities
from src.tracing import TraceRecorder, new_run_id, read_trace, summarize_trace


# ---------------------------------------------------------------------------
# 引擎装配
# ---------------------------------------------------------------------------
def test_engine_stats_structure(engine):
    stats = engine.stats()
    assert stats["corpus"]["documents"] > 0
    assert stats["chunks"]["parents"] > 0
    assert stats["chunks"]["children"] > stats["chunks"]["parents"]
    assert stats["index"]["vocabulary"] > 0
    assert stats["llm"]["mode"] == "mock"
    assert stats["cache"]["backend"] in ("memory", "redis")
    assert "config" in stats


def test_engine_known_entities_cover_document_subjects(engine):
    assert "示例监管机构" in engine.known_entities
    assert any("示例集团" in name for name in engine.known_entities)


def test_engine_masking_counted(engine):
    assert engine.masked_documents >= 0


def test_engine_quality_reports_available(engine):
    assert engine.quality_scores
    assert all(0.0 <= v <= 1.0 for v in engine.quality_scores.values())


# ---------------------------------------------------------------------------
# 问答主流程
# ---------------------------------------------------------------------------
def test_ask_returns_traceable_answer(engine):
    result = engine.ask("合格投资者的金融资产门槛是多少？", use_cache=False)
    assert result.mode == "rag"
    assert result.traceable
    assert result.citations
    assert result.retrieval is not None
    assert result.timings["total_ms"] > 0


def test_ask_answer_contains_citation_marks(engine):
    result = engine.ask("合格投资者的金融资产门槛是多少？", use_cache=False)
    assert "[1]" in result.answer
    assert "【出处与时效】" in result.answer


def test_ask_uses_faq_for_high_frequency_question(engine):
    result = engine.ask("开户需要准备哪些材料？", use_cache=False)
    assert result.faq_hit
    assert result.mode == "faq"
    assert result.traceable


def test_ask_can_disable_faq(engine):
    result = engine.ask("开户需要准备哪些材料？", use_cache=False, use_faq=False)
    assert not result.faq_hit
    assert result.mode in ("rag", "refused")


def test_ask_refuses_unknown_entity(engine):
    result = engine.ask("示例科技股份有限公司 2024 年的净利润是多少？", use_cache=False)
    assert result.mode == "refused"
    assert result.citations == []
    assert "不在本资料库的主体范围内" in result.answer


def test_unknown_entities_matching_rules():
    known = ["示例银行股份有限公司", "示例监管机构"]
    assert unknown_entities("示例科技股份有限公司怎么样？", known)
    assert not unknown_entities("示例银行股份有限公司怎么样？", known)
    assert not unknown_entities("示例银行怎么样？", known)
    assert not unknown_entities("合格投资者的门槛是多少？", known)


def test_question_entities_deduplicates():
    entities = question_entities("示例银行和示例银行股份有限公司")
    assert len(entities) == len(set(entities))


def test_ask_cache_hit_second_time(engine):
    question = "双录资料需要保存多久？"
    first = engine.ask(question, use_cache=True, use_faq=False)
    second = engine.ask(question, use_cache=True, use_faq=False)
    assert second.cache_hit
    assert second.answer == first.answer
    assert second.timings["total_ms"] <= first.timings["total_ms"] + 5


def test_ask_no_cache_never_hits(engine):
    question = "冷静期是多久？"
    engine.ask(question, use_cache=True, use_faq=False)
    fresh = engine.ask(question, use_cache=False, use_faq=False)
    assert not fresh.cache_hit


def test_ask_with_metadata_filter(engine):
    result = engine.ask("合格投资者的金融资产门槛是多少？", expr='year = 2023', use_cache=False, use_faq=False)
    assert result.retrieval.plan.filter_expr == "year = 2023"
    assert all(e.source_id == "POL-2023-11" for e in result.evidence)


def test_ask_soft_filter_reported_in_notes(engine):
    result = engine.ask("示例集团 2023 年应收账款增速如何？", use_cache=False, use_faq=False)
    assert any("软过滤" in note for note in result.notes)


def test_answer_result_render_includes_citation_list(engine):
    rendered = engine.ask("合格投资者的金融资产门槛是多少？", use_cache=False).render()
    assert "【引用明细】" in rendered
    assert "POL-2024-07" in rendered


def test_answer_result_to_dict_is_jsonable(engine):
    payload = engine.ask("合格投资者的金融资产门槛是多少？", use_cache=False).to_dict()
    json.dumps(payload, ensure_ascii=False)
    assert payload["source_ids"]


def test_search_only_does_not_generate(engine):
    result = engine.search("合格投资者")
    assert result.evidence
    assert result.mode == "hybrid"


def test_compare_strategies_returns_three_modes(engine):
    payload = engine.compare_strategies("年满 65 周岁的客户购买 R3 产品有哪些额外要求？", top_k=3)
    assert set(payload["strategies"]) == {"dense_only", "bm25_only", "hybrid"}
    assert payload["strategies"]["hybrid"]["source_ids"]


# ---------------------------------------------------------------------------
# 轨迹
# ---------------------------------------------------------------------------
def test_new_run_id_format():
    run_id = new_run_id("ask")
    assert run_id.startswith("ask-")
    assert len(run_id.split("-")) == 4


def test_trace_recorder_writes_jsonl(tmp_path):
    recorder = TraceRecorder(run_id="unit-run", runs_dir=tmp_path)
    recorder.step("recall", "retriever", "输入", "输出", latency_ms=1.5, extra_field=1)
    rows = read_trace("unit-run", runs_dir=tmp_path)
    assert len(rows) == 1
    assert rows[0]["step"] == "recall"
    assert rows[0]["input_digest"]
    assert rows[0]["extra"]["extra_field"] == 1


def test_trace_recorder_disabled_keeps_records_in_memory(tmp_path):
    recorder = TraceRecorder(run_id="off-run", runs_dir=tmp_path, enabled=False)
    recorder.step("x", "y", "a", "b")
    assert recorder.records
    assert not recorder.path.exists()


def test_read_trace_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_trace("nope", runs_dir=tmp_path)


def test_summarize_trace_aggregates(tmp_path):
    recorder = TraceRecorder(run_id="sum-run", runs_dir=tmp_path)
    recorder.step("a", "x", "", "", latency_ms=2.0)
    recorder.step("a", "x", "", "", latency_ms=3.0, status="warn")
    summary = summarize_trace([r.to_dict() for r in recorder.records])
    assert summary["steps"] == 2
    assert summary["by_step"]["a"]["count"] == 2
    assert summary["errors"] == 1


def test_engine_writes_trace_file(tmp_path, cleaned_documents):
    engine = RAGEngine.build(runs_dir=tmp_path, quiet=True)
    result = engine.ask("冷静期是多久？", use_cache=False, use_faq=False)
    rows = read_trace(result.run_id, runs_dir=tmp_path)
    steps = {row["step"] for row in rows}
    assert {"recall", "generate", "validate", "done"} <= steps


# ---------------------------------------------------------------------------
# 接口层（纯函数）
# ---------------------------------------------------------------------------
def test_handle_health(engine):
    payload = __import__("src.api", fromlist=["handle_health"]).handle_health(engine)
    assert payload["status"] == "ok"
    assert payload["children"] > 0


def test_handle_stats(engine):
    from src.api import handle_stats

    assert handle_stats(engine)["stats"]["corpus"]["documents"] > 0


def test_handle_ask_ok(engine):
    from src.api import handle_ask

    payload = handle_ask(engine, {"question": "冷静期是多久？", "use_cache": False})
    assert payload["status"] == "ok"
    assert payload["answer"]
    assert payload["traceable"]


def test_handle_ask_missing_question(engine):
    from src.api import handle_ask

    payload = handle_ask(engine, {})
    assert payload["status"] == "error"
    assert payload["code"] == 400


def test_handle_search_returns_evidence(engine):
    from src.api import handle_search

    payload = handle_search(engine, {"question": "合格投资者", "top_k": 3})
    assert payload["status"] == "ok"
    assert len(payload["evidence"]) <= 3


def test_handle_search_missing_question(engine):
    from src.api import handle_search

    assert handle_search(engine, {})["status"] == "error"


def test_handle_feedback_records(engine):
    from src.api import FEEDBACK_LOG, handle_feedback

    before = len(FEEDBACK_LOG)
    payload = handle_feedback(engine, {"run_id": "r1", "verdict": "bad", "stage": "召回", "comment": "漏召回"})
    assert payload["status"] == "ok"
    assert len(FEEDBACK_LOG) == before + 1


def test_create_app_requires_fastapi(engine):
    try:
        import fastapi  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError):
            from src.api import create_app

            create_app(engine)
    else:
        from src.api import create_app

        assert create_app(engine) is not None


# ---------------------------------------------------------------------------
# 兜底 HTTP 服务
# ---------------------------------------------------------------------------
@pytest.fixture
def live_server(engine):
    from src.serve import make_handler

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}"
    server.shutdown()
    server.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 - 仅本地测试
        return json.loads(resp.read().decode("utf-8"))


def _post(url: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as resp:  # noqa: S310 - 仅本地测试
        return json.loads(resp.read().decode("utf-8"))


def test_serve_health_endpoint(live_server):
    payload = _get(f"{live_server}/health")
    assert payload["status"] == "ok"


def test_serve_stats_endpoint(live_server):
    assert _get(f"{live_server}/stats")["stats"]["corpus"]["documents"] > 0


def test_serve_ask_endpoint(live_server):
    payload = _post(f"{live_server}/ask", {"question": "冷静期是多久？", "use_cache": False, "use_faq": False})
    assert payload["status"] == "ok"
    assert payload["answer"]


def test_serve_search_endpoint(live_server):
    payload = _post(f"{live_server}/search", {"question": "合格投资者", "top_k": 2})
    assert payload["status"] == "ok"
    assert payload["evidence"]


def test_serve_unknown_path_returns_404(live_server):
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{live_server}/nope")
    assert exc.value.code == 404
