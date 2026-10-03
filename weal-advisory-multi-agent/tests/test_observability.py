"""可观测性测试：trace 落盘字段、回放读取、摘要统计。"""

from __future__ import annotations

import json

from src.observability import (
    TRACE_FIELDS,
    Tracer,
    format_trace_table,
    list_runs,
    new_run_id,
    read_trace,
    trace_summary,
)
from src.utils import canonical_json, digest


def test_tracer_writes_required_fields(tmp_path):
    tracer = Tracer(run_id="unit-run", runs_dir=tmp_path)
    tracer.step(
        "TestAgent",
        node="test",
        input_payload={"a": 1},
        output_payload={"b": 2},
        status="ok",
        tool_calls=["tool.x"],
        latency_ms=12.5,
    )
    tracer.step("TestAgent", node="test2", input_payload={"c": 3}, output_payload={"d": 4}, latency_ms=1.0)

    path = tmp_path / "unit-run.jsonl"
    assert path.exists()
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2
    for record in lines:
        assert set(TRACE_FIELDS) <= set(record)
    assert lines[0]["step"] == 1
    assert lines[1]["step"] == 2
    assert lines[0]["agent"] == "TestAgent"
    assert lines[0]["input_digest"] == digest({"a": 1})
    assert lines[0]["output_digest"] == digest({"b": 2})
    assert lines[0]["tool_calls"] == ["tool.x"]
    assert lines[0]["latency_ms"] == 12.5


def test_tracer_disabled_does_not_write_but_keeps_events(tmp_path):
    tracer = Tracer(run_id="off-run", runs_dir=tmp_path, enabled=False)
    tracer.step("A", input_payload=1, output_payload=2)
    assert not (tmp_path / "off-run.jsonl").exists()
    assert tracer.step_count() == 1


def test_read_trace_roundtrip(tmp_path):
    tracer = Tracer(run_id="read-run", runs_dir=tmp_path)
    tracer.step("A", input_payload={"x": 1}, output_payload={"y": 2}, latency_ms=3.0)
    tracer.step("B", input_payload={"x": 3}, output_payload={"y": 4}, latency_ms=4.0)
    records = read_trace("read-run", tmp_path)
    assert [item["agent"] for item in records] == ["A", "B"]


def test_read_trace_missing_file_raises(tmp_path):
    try:
        read_trace("nope", tmp_path)
    except FileNotFoundError as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应当抛出 FileNotFoundError")


def test_list_runs_and_summary(tmp_path):
    Tracer(run_id="r1", runs_dir=tmp_path).step("A", input_payload=1, output_payload=2, latency_ms=5.0)
    Tracer(run_id="r2", runs_dir=tmp_path).step("B", input_payload=1, output_payload=2, latency_ms=7.0)
    assert list_runs(tmp_path) == ["r1", "r2"]
    summary = trace_summary(read_trace("r1", tmp_path))
    assert summary["steps"] == 1
    assert summary["agents"] == ["A"]
    assert summary["total_latency_ms"] == 5.0
    assert trace_summary([])["steps"] == 0


def test_format_trace_table_contains_key_columns(tmp_path):
    tracer = Tracer(run_id="fmt-run", runs_dir=tmp_path)
    tracer.step("AgentX", node="nodex", input_payload=1, output_payload=2, tool_calls=["t"], latency_ms=1.5)
    table = format_trace_table(read_trace("fmt-run", tmp_path))
    assert "AgentX" in table
    assert "nodex" in table
    assert "latency(ms)" in table
    assert "t" in table


def test_digest_is_stable_and_order_insensitive():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    assert canonical_json({"a": 1.0}) == canonical_json({"a": 1})
    assert digest({"a": 1}) != digest({"a": 2})


def test_new_run_id_is_unique_enough():
    ids = {new_run_id("t") for _ in range(5)}
    assert len(ids) >= 1
    assert all(item.startswith("t-") for item in ids)
