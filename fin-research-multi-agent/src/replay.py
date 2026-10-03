"""回放入口：`python -m src.replay <run_id>`

读取 `runs/<run_id>.jsonl`，把整次执行过程按时间顺序还原出来：
每一步的执行者、耗时、输入/输出摘要、状态、工具调用、以及关键业务留痕，
并在末尾给出汇总与「从轨迹直接重建出来的报告」的引用可追溯性自检。

设计要点：轨迹是**自包含**的。即使原进程已经退出、资料库发生变化，
只要 jsonl 还在，就能复盘当时究竟发生了什么（每步的 input/output digest 可用于比对）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import RUNS_DIR
from .tracing import load_trace
from .utils.console import ensure_utf8_console

LINE = "=" * 78
SUB = "-" * 78

#: 每个 Agent 在回放时额外展示的业务留痕字段
HIGHLIGHT_FIELDS: Dict[str, List[str]] = {
    "planner": ["intent", "companies", "year", "targets", "route", "retrieval_queries"],
    "retriever": ["selected", "sources", "candidate_count", "coverage", "missing_data"],
    "analyst": ["metrics", "summary"],
    "risk_checker": ["verdict", "risk_level", "narrative", "escalation_reason", "gate_history"],
    "human_review": ["decision", "reason", "source"],
    "writer": ["citations", "report_chars"],
    "engine_fallback": ["error"],
}

AGENT_LABEL = {
    "planner": "PlannerAgent（任务分解与路由）",
    "retriever": "RetrieverAgent（多路检索）",
    "analyst": "AnalystAgent（指标计算与分析）",
    "risk_checker": "RiskCheckerAgent（风险核查 / 反思循环）",
    "human_review": "HumanReview（人工确认）",
    "writer": "WriterAgent（投研简报）",
    "run_summary": "运行汇总",
    "engine_fallback": "引擎降级",
}


def _short(value: Any, limit: int = 160) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def _status_mark(status: str) -> str:
    return {"ok": "✔", "error": "✘", "skipped": "·"}.get(status, "?")


def render(run_id: str, runs_dir: Optional[Path] = None, show_preview: bool = True) -> int:
    """打印一次运行的完整回放。"""
    try:
        trace = load_trace(run_id, runs_dir)
    except FileNotFoundError as exc:
        print(f"[replay] {exc}")
        directory = Path(runs_dir) if runs_dir else RUNS_DIR
        available = sorted(p.stem for p in directory.glob("*.jsonl")) if directory.is_dir() else []
        if available:
            print("[replay] 可回放的运行：")
            for name in available[-20:]:
                print(f"  - {name}")
        return 2

    entries: List[Dict[str, Any]] = trace["entries"]
    summary: Dict[str, Any] = trace["summary"] or {}
    steps = trace["steps"]

    print("")
    print(LINE)
    print(f"执行回放  ·  run_id = {trace['run_id']}")
    print(LINE)
    print(f"轨迹文件   : {trace['path']}")
    print(f"总步数     : {len(steps)}")
    if summary:
        print(f"状态       : {summary.get('status')}    引擎: {summary.get('engine')}    "
              f"模式: {summary.get('mode')}")
        print(f"研究问题   : {summary.get('question', '')}")
        print(f"执行路径   : {' -> '.join(summary.get('agents_visited') or [])}")
        print(f"总耗时     : {summary.get('total_latency_ms')} ms")
        print(f"反思轮次   : {summary.get('revision_round')}    "
              f"风险裁决: {summary.get('risk_verdict')} ({summary.get('risk_level')})    "
              f"人工确认: {summary.get('human_decision') or '未触发'}")
        print(f"证据/结论/引用: {summary.get('evidence')} / {summary.get('findings')} / {summary.get('citations')}")

    print("")
    print(SUB)
    print(f"{'步':>3}  {'执行者':<14}{'状态':<5}{'耗时(ms)':>10}  输入->输出摘要")
    print(SUB)
    for entry in steps:
        agent = str(entry.get("agent", ""))
        print(
            f"{entry.get('step', 0):>3}  {agent:<14}{_status_mark(str(entry.get('status'))):<5}"
            f"{float(entry.get('latency_ms') or 0):>10.2f}  "
            f"{entry.get('input_digest')} -> {entry.get('output_digest')}"
        )

    for entry in steps:
        agent = str(entry.get("agent", ""))
        print("")
        print(SUB)
        print(f"step {entry.get('step')} · {AGENT_LABEL.get(agent, agent)}"
              f"  [{entry.get('status')}]  {entry.get('latency_ms')} ms  @{entry.get('ts')}")
        print(SUB)
        print(f"  输入摘要 : {entry.get('input_digest')}  ({entry.get('input_chars')} chars)")
        print(f"  输出摘要 : {entry.get('output_digest')}  ({entry.get('output_chars')} chars)")
        if show_preview:
            print(f"  输入预览 : {entry.get('input_preview', '')}")
            print(f"  输出预览 : {entry.get('output_preview', '')}")

        for field in HIGHLIGHT_FIELDS.get(agent, []):
            payload = entry.get(field)
            if payload is None:
                payload = (entry.get("extra") or {}).get(field)
            if payload in (None, "", [], {}):
                continue
            print(f"  · {field:<22}: {_short(payload)}")

        tool_calls = (entry.get("extra") or {}).get("tool_calls") or []
        if tool_calls:
            print(f"  · 工具调用 ({len(tool_calls)})：")
            for call in tool_calls:
                flag = "缓存" if call.get("cached") else ("重试" if (call.get("attempts") or 1) > 1 else "实时")
                print(
                    f"      - {call.get('tool'):<20} 权限={call.get('permission_level'):<16}"
                    f" 耗时={call.get('latency_ms')}ms 尝试={call.get('attempts')} [{flag}]"
                    f"{'  错误=' + str(call.get('error_code')) if call.get('error_code') else ''}"
                )

    print("")
    print(LINE)
    print("回放结束")
    print(LINE)
    if summary.get("errors"):
        print("错误记录：")
        for err in summary["errors"]:
            print(f"  - {err}")
    print(f"提示：用 `python -m src.demo --run-id {trace['run_id']} --auto` 可用同一编号重跑比对。")
    print("")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.replay",
        description="回放一次多智能体投研运行的 JSONL 轨迹",
    )
    parser.add_argument("run_id", help="运行编号（runs/<run_id>.jsonl）")
    parser.add_argument("--runs-dir", default=None, help="轨迹目录（默认 runs/）")
    parser.add_argument("--no-preview", action="store_true", help="不显示输入/输出预览")
    return parser


def main(argv: List[str] | None = None) -> int:
    ensure_utf8_console()
    args = build_parser().parse_args(argv)
    runs_dir = Path(args.runs_dir) if args.runs_dir else None
    return render(args.run_id, runs_dir=runs_dir, show_preview=not args.no_preview)


if __name__ == "__main__":
    sys.exit(main())
