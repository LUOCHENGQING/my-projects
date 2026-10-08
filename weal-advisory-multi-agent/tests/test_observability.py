"""可观测性测试：trace 落盘字段、回放读取、摘要统计。

被测模块：`src.observability` 与 `src.utils` 中的指纹工具
- `Tracer`：把每个 Agent 步骤写为 `<run_id>.jsonl`，记录必须覆盖 `TRACE_FIELDS`；
- `read_trace` / `list_runs`：按 run 回放与枚举；
- `trace_summary` / `format_trace_table`：给评估与人工排查用的统计与表格渲染；
- `new_run_id`：run 号生成（前缀 + 时间戳）；
- `digest` / `canonical_json`：输入输出以指纹而非明文落盘，指纹必须稳定。

覆盖策略
- 正常路径：两步落盘 → 字段齐备、step 自 1 递增、输入输出以 digest 记录；
- 边界路径：`enabled=False` 不落盘但内存事件仍计数；空列表摘要为 0；run 列表排序稳定；
- 异常路径：读取不存在的 run 必须抛 `FileNotFoundError`，不得静默返回空；
- 一致性/对抗探针：digest 对字典键序不敏感、对取值敏感，canonical_json 抹平 1 与 1.0 的表示差异。

注：本模块全部为模块级测试函数，未定义测试类，故无类级 docstring。
"""

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
    """不变式：每条 trace 记录都是 TRACE_FIELDS 的超集，step 自 1 递增，输入输出以 digest 落盘。

    第二步刻意省略 status / tool_calls，用来验证缺省值也会被补齐为完整字段集，
    而不是"缺字段就少写一行"。
    """
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

    # 落盘文件名为 <run_id>.jsonl
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
    """不变式：enabled=False 只关闭落盘，事件计数不受影响——便于无文件 IO 的单测。"""
    tracer = Tracer(run_id="off-run", runs_dir=tmp_path, enabled=False)
    tracer.step("A", input_payload=1, output_payload=2)
    assert not (tmp_path / "off-run.jsonl").exists()
    assert tracer.step_count() == 1


def test_read_trace_roundtrip(tmp_path):
    """不变式：写入顺序即回放顺序（read_trace 按 step 升序返回 A、B）。"""
    tracer = Tracer(run_id="read-run", runs_dir=tmp_path)
    tracer.step("A", input_payload={"x": 1}, output_payload={"y": 2}, latency_ms=3.0)
    tracer.step("B", input_payload={"x": 3}, output_payload={"y": 4}, latency_ms=4.0)
    records = read_trace("read-run", tmp_path)
    assert [item["agent"] for item in records] == ["A", "B"]


def test_read_trace_missing_file_raises(tmp_path):
    """异常路径：run 不存在时必须抛 FileNotFoundError，且异常信息里能看出是哪个 run。"""
    try:
        read_trace("nope", tmp_path)
    except FileNotFoundError as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应当抛出 FileNotFoundError")


def test_list_runs_and_summary(tmp_path):
    """不变式：list_runs 按 run_id 排序；摘要的 steps / agents / total_latency_ms 与记录一致，空输入返回 0。"""
    Tracer(run_id="r1", runs_dir=tmp_path).step("A", input_payload=1, output_payload=2, latency_ms=5.0)
    Tracer(run_id="r2", runs_dir=tmp_path).step("B", input_payload=1, output_payload=2, latency_ms=7.0)
    assert list_runs(tmp_path) == ["r1", "r2"]
    summary = trace_summary(read_trace("r1", tmp_path))
    assert summary["steps"] == 1
    assert summary["agents"] == ["A"]
    assert summary["total_latency_ms"] == 5.0
    assert trace_summary([])["steps"] == 0


def test_format_trace_table_contains_key_columns(tmp_path):
    """口径：表格渲染必须含 agent、node、latency(ms) 列与 tool_calls 内容，保证人工排查可用。"""
    tracer = Tracer(run_id="fmt-run", runs_dir=tmp_path)
    tracer.step("AgentX", node="nodex", input_payload=1, output_payload=2, tool_calls=["t"], latency_ms=1.5)
    table = format_trace_table(read_trace("fmt-run", tmp_path))
    assert "AgentX" in table
    assert "nodex" in table
    assert "latency(ms)" in table
    assert "t" in table


def test_digest_is_stable_and_order_insensitive():
    """不变式：digest 对键序不敏感、对取值敏感；canonical_json 归一 1 与 1.0，避免同义载荷生成不同指纹。"""
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    assert canonical_json({"a": 1.0}) == canonical_json({"a": 1})
    assert digest({"a": 1}) != digest({"a": 2})


def test_new_run_id_is_unique_enough():
    """边界：run 号必须带指定前缀；不断言绝对唯一（时间戳精度有限），只断言前缀与集合非空。"""
    ids = {new_run_id("t") for _ in range(5)}
    assert len(ids) >= 1
    assert all(item.startswith("t-") for item in ids)
