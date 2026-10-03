"""可观测性：每步 trace 落盘 + 回放。

`runs/<run_id>.jsonl` 每行一个步骤，字段固定为：
`step / agent / node / input_digest / output_digest / latency_ms / status / tool_calls`
（另附 timestamp 与可选的 extra，便于排查）。

`python -m src.replay <run_id>` 读取该文件并打印步骤表。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .utils import digest, now_iso

#: runs 目录（相对仓库根定位）
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

#: trace 行的固定字段（顺序即打印顺序）
TRACE_FIELDS: tuple[str, ...] = (
    "step",
    "agent",
    "node",
    "input_digest",
    "output_digest",
    "latency_ms",
    "status",
    "tool_calls",
)


def new_run_id(prefix: str = "run") -> str:
    """生成一个按时间可排序的 run_id。"""
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"


@dataclass
class Tracer:
    """逐步记录流水线执行痕迹。"""

    run_id: str
    runs_dir: Path | None = None
    enabled: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.runs_dir = Path(self.runs_dir) if self.runs_dir is not None else RUNS_DIR
        self._step = 0

    # ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        """trace 文件路径。"""
        return self.runs_dir / f"{self.run_id}.jsonl"

    def step(
        self,
        agent: str,
        *,
        node: str = "",
        input_payload: Any = None,
        output_payload: Any = None,
        status: str = "ok",
        tool_calls: Sequence[str] = (),
        latency_ms: float | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """记录一步执行，返回该条 trace（同时写入 jsonl）。"""
        self._step += 1
        record: dict[str, Any] = {
            "step": self._step,
            "agent": agent,
            "node": node or agent,
            "input_digest": digest(input_payload) if input_payload is not None else "",
            "output_digest": digest(output_payload) if output_payload is not None else "",
            "latency_ms": round(float(latency_ms), 3) if latency_ms is not None else 0.0,
            "status": status,
            "tool_calls": list(tool_calls),
            "timestamp": now_iso(),
        }
        if extra:
            record["extra"] = dict(extra)
        self.events.append(record)
        if self.enabled:
            self._write(record)
        return record

    def _write(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    def step_count(self) -> int:
        """已记录的步骤数。"""
        return self._step

    def total_latency_ms(self) -> float:
        """累计耗时（毫秒）。"""
        return round(sum(float(item.get("latency_ms", 0.0)) for item in self.events), 3)

    def agent_sequence(self) -> list[str]:
        """按顺序返回参与执行的 Agent 名（用于双引擎路径一致性比对）。"""
        return [str(item["agent"]) for item in self.events]

    def summary(self) -> dict[str, Any]:
        """trace 摘要。"""
        return {
            "run_id": self.run_id,
            "steps": self._step,
            "total_latency_ms": self.total_latency_ms(),
            "agents": list(dict.fromkeys(self.agent_sequence())),
            "statuses": sorted({str(item["status"]) for item in self.events}),
        }


def read_trace(run_id: str, runs_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """读取一个 run 的全部 trace 行。"""
    base = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    path = base / f"{run_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"未找到 trace 文件：{path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def list_runs(runs_dir: Path | str | None = None) -> list[str]:
    """列出全部 run_id（按文件名排序）。"""
    base = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    if not base.exists():
        return []
    return sorted(path.stem for path in base.glob("*.jsonl"))


def _ellipsis(text: str, width: int) -> str:
    """按显示宽度截断（超长时以省略号结尾）。"""
    return text if len(text) <= width else text[: width - 1] + "…"


def format_trace_table(records: Iterable[Mapping[str, Any]]) -> str:
    """把 trace 渲染成等宽表格（CLI 回放用）。"""
    header = (
        f"{'step':>4}  {'agent':<26}{'node':<14}{'latency(ms)':>12}  "
        f"{'status':<7}{'tool_calls':<44}{'input→output'}"
    )
    lines = [header, "-" * len(header)]
    for record in records:
        tools = _ellipsis(",".join(record.get("tool_calls") or []) or "-", 43)
        lines.append(
            f"{record.get('step', 0):>4}  {_ellipsis(str(record.get('agent', '')), 25):<26}"
            f"{_ellipsis(str(record.get('node', '')), 13):<14}"
            f"{float(record.get('latency_ms', 0.0)):>12.2f}  "
            f"{str(record.get('status', '')):<7}{tools:<44}"
            f"{record.get('input_digest', '')[:8]}→{record.get('output_digest', '')[:8]}"
        )
    return "\n".join(lines)


def trace_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """对一批 trace 行做统计汇总。"""
    if not records:
        return {"steps": 0, "total_latency_ms": 0.0, "agents": [], "statuses": []}
    return {
        "steps": len(records),
        "total_latency_ms": round(sum(float(item.get("latency_ms", 0.0)) for item in records), 3),
        "agents": list(dict.fromkeys(str(item.get("agent", "")) for item in records)),
        "statuses": sorted({str(item.get("status", "")) for item in records}),
        "run_id": str(records[0].get("run_id", "")) or None,
    }
