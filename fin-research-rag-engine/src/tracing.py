"""每一步一行 JSONL 的可观测记录。

RAG 系统的线上问题几乎都长这样：「同一个问题，昨天答得对，今天答错了」。
如果没有逐步轨迹，能查的只有最终答案，而答案错了可能是路由变了、召回少了、
重排顺序变了、缓存返回了旧结果——**每一种原因的修法完全不同**。

因此这里把每一步落成一行 JSONL：

    step / agent / input_digest / output_digest / latency_ms / status / extra

用 digest（短哈希 + 截断预览）而不是全文，是因为轨迹文件会被长期保留，
全文落盘既占空间又会把敏感数据（即使已脱敏）反复复制。

`read_trace()` 让一次运行可以被完整回放：召回了几条、哪条被去重、重排前后名次怎么变、
引用校验剔除了哪几个编号——面试时被追问"你怎么定位一个答错的 case"，
这就是答案。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import RUNS_DIR
from .utils.jsonable import to_plain
from .utils.text import digest

__all__ = ["TraceRecord", "TraceRecorder", "read_trace", "summarize_trace", "new_run_id"]


def new_run_id(prefix: str = "run") -> str:
    """生成可排序、可读的运行编号：run-20261003-012345-ab12cd。"""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = format(int(time.time() * 1000) % 0xFFFFFF, "06x")
    return f"{prefix}-{stamp}-{suffix}"


@dataclass
class TraceRecord:
    """一行轨迹。"""

    step: str
    agent: str
    run_id: str
    status: str = "ok"
    latency_ms: float = 0.0
    input_digest: str = ""
    output_digest: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "ts": round(self.ts, 3),
            "run_id": self.run_id,
            "step": self.step,
            "agent": self.agent,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 3),
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
        }
        if self.extra:
            payload["extra"] = to_plain(self.extra)
        return payload


class TraceRecorder:
    """把每一步追加写到 runs/<run_id>.jsonl。"""

    def __init__(self, run_id: Optional[str] = None, runs_dir: Optional[Path] = None, enabled: bool = True) -> None:
        self.run_id = run_id or new_run_id()
        self.runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
        self.enabled = enabled
        self.records: List[TraceRecord] = []
        self._started = time.perf_counter()

    # ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self.runs_dir / f"{self.run_id}.jsonl"

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._started) * 1000.0

    def step(
        self,
        step: str,
        agent: str,
        input_obj: Any = "",
        output_obj: Any = "",
        status: str = "ok",
        latency_ms: float = 0.0,
        **extra: Any,
    ) -> TraceRecord:
        """记录一步。input/output 会被压成 digest，避免轨迹里出现全文。"""
        record = TraceRecord(
            step=step,
            agent=agent,
            run_id=self.run_id,
            status=status,
            latency_ms=latency_ms,
            input_digest=digest(_as_text(input_obj)),
            output_digest=digest(_as_text(output_obj)),
            extra=dict(extra),
        )
        self.records.append(record)
        if self.enabled:
            self._append(record)
        return record

    def _append(self, record: TraceRecord) -> None:
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            # 落盘失败不应影响主流程（只读文件系统 / 磁盘满等），内存记录仍在
            pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "path": str(self.path),
            "steps": [r.to_dict() for r in self.records],
            "elapsed_ms": round(self.elapsed_ms, 3),
        }

    def summary(self) -> Dict[str, Any]:
        return summarize_trace([r.to_dict() for r in self.records])


def _as_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(to_plain(obj), ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(obj)


def read_trace(run_id: str, runs_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """读取一次运行的完整轨迹。"""
    directory = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    path = directory / (run_id if run_id.endswith(".jsonl") else f"{run_id}.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"轨迹文件不存在：{path}")
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize_trace(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """把轨迹汇总成「一次运行的画像」，用于快速定位性能与失败点。"""
    items = list(rows)
    by_step: Dict[str, Dict[str, Any]] = {}
    total = 0.0
    for row in items:
        step = str(row.get("step", ""))
        bucket = by_step.setdefault(step, {"count": 0, "latency_ms": 0.0, "errors": 0})
        bucket["count"] += 1
        latency = float(row.get("latency_ms", 0.0) or 0.0)
        bucket["latency_ms"] += latency
        total += latency
        if row.get("status") not in (None, "ok"):
            bucket["errors"] += 1
    for bucket in by_step.values():
        bucket["latency_ms"] = round(bucket["latency_ms"], 3)
    return {
        "steps": len(items),
        "total_latency_ms": round(total, 3),
        "by_step": by_step,
        "errors": sum(b["errors"] for b in by_step.values()),
    }
