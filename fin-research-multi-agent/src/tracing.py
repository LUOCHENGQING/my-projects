"""可观测：每一步一行 JSONL 的执行轨迹。

落盘位置：`runs/<run_id>.jsonl`
每行固定字段（题目要求的最小集合）：
    step            步序号（从 1 开始，全局单调递增）
    agent           执行者（planner / retriever / analyst / risk_checker / writer / human_review）
    input_digest    输入摘要（sha256 前 16 位）
    output_digest   输出摘要（sha256 前 16 位）
    latency_ms      本步耗时（毫秒）
    status          ok / error / skipped

额外附带（便于 replay 与排障，不影响固定字段）：
    ts, input_chars, output_chars, input_preview, output_preview, extra

设计要点：每写一行立即 flush，保证进程被强杀时轨迹依然可用；
preview 只截前 240 字符，既有可读性又不会让日志爆炸。
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import RUNS_DIR
from .utils.digest import digest

__all__ = ["TraceRecorder", "load_trace"]

_PREVIEW_CHARS = 240


def _preview(value: Any) -> str:
    """生成可读的一行预览（只压空白，不动标点，保证中文可读）。"""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            text = repr(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_PREVIEW_CHARS]


class TraceRecorder:
    """把每一步执行写成一行 JSONL。"""

    def __init__(self, run_id: str, runs_dir: Optional[Path] = None, echo: bool = False) -> None:
        self.run_id = run_id
        self.runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.runs_dir / f"{run_id}.jsonl"
        self.echo = echo
        self._step = 0
        self._lock = threading.Lock()
        self._started = time.perf_counter()
        # 新 run 覆盖同名文件，保证 replay 看到的是一次完整运行
        self.path.write_text("", encoding="utf-8")

    # ------------------------------------------------------------------
    @property
    def step_count(self) -> int:
        return self._step

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._started) * 1000.0

    def record(
        self,
        *,
        agent: str,
        input_obj: Any,
        output_obj: Any,
        latency_ms: float,
        status: str = "ok",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """写入一行轨迹，返回该行内容。"""
        with self._lock:
            self._step += 1
            entry = {
                "step": self._step,
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "agent": agent,
                "input_digest": digest(input_obj),
                "output_digest": digest(output_obj),
                "latency_ms": round(float(latency_ms), 3),
                "status": status,
                "input_chars": len(_preview(input_obj)) if not isinstance(input_obj, str) else len(input_obj),
                "output_chars": len(output_obj) if isinstance(output_obj, str) else len(_preview(output_obj)),
                "input_preview": _preview(input_obj),
                "output_preview": _preview(output_obj),
                "extra": extra or {},
            }
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                fh.flush()
            if self.echo:
                print(f"[trace] step={entry['step']} agent={agent} status={status} {entry['latency_ms']}ms")
            return entry

    def close(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """写入结束汇总行。"""
        return self.record(
            agent="run_summary",
            input_obj={"run_id": self.run_id},
            output_obj=summary,
            latency_ms=self.elapsed_ms,
            status=summary.get("status", "ok"),
            extra=summary,
        )


def load_trace(run_id: str, runs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """读取一次运行的轨迹。返回 {run_id, path, entries, summary}。"""
    directory = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    path = directory / f"{run_id}.jsonl"
    if not path.is_file():
        # 允许传入带 .jsonl 后缀或部分匹配
        candidates = sorted(directory.glob(f"*{run_id}*.jsonl")) if directory.is_dir() else []
        if not candidates:
            raise FileNotFoundError(f"未找到运行轨迹：{path}")
        path = candidates[-1]

    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    summary_entry = next((e for e in reversed(entries) if e.get("agent") == "run_summary"), None)
    # 汇总行的业务字段挂在 extra 里，这里拍平，方便调用方直接 summary["question"] 取值
    summary = None
    if summary_entry is not None:
        summary = {**summary_entry, **(summary_entry.get("extra") or {})}
    return {
        "run_id": path.stem,
        "path": str(path),
        "entries": entries,
        "summary": summary,
        "steps": [e for e in entries if e.get("agent") != "run_summary"],
    }
