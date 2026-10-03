"""运行痕迹回放：`python -m src.replay <run_id>`。

读取 `runs/<run_id>.jsonl`，按步骤打印 `step / agent / node / latency / status /
tool_calls / input_digest → output_digest`，并给出汇总（步数、总耗时、参与 Agent）。

`--list` 列出全部可回放的 run_id；`--json` 输出原始 trace。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .observability import RUNS_DIR, format_trace_table, list_runs, read_trace, trace_summary


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(
        prog="python -m src.replay",
        description="回放某次投顾流水线的逐步 trace",
    )
    parser.add_argument("run_id", nargs="?", help="运行标识，例如 advisory-20261003-010203-456")
    parser.add_argument("--runs-dir", default=str(RUNS_DIR), help="trace 目录")
    parser.add_argument("--list", action="store_true", help="列出全部可回放的 run_id")
    parser.add_argument("--json", action="store_true", help="输出原始 trace JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(argv)
    base = Path(args.runs_dir)

    if args.list or not args.run_id:
        runs = list_runs(base)
        if not runs:
            print(f"目录 {base} 下暂无 trace 文件。")
            print("提示：先执行 `python -m src.demo --auto`。")
            return 1
        print(f"共 {len(runs)} 个可回放运行（目录：{base}）：")
        for name in runs:
            print(f"  {name}")
        return 0

    try:
        records = read_trace(args.run_id, base)
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return 0

    print(f"回放运行 {args.run_id}（{len(records)} 步）｜目录：{base}")
    print(format_trace_table(records))
    summary = trace_summary(records)
    print("-" * 60)
    print(
        f"汇总：步数 {summary['steps']}｜总耗时 {summary['total_latency_ms']:.2f} ms｜"
        f"参与 Agent {len(summary['agents'])} 个｜状态 {summary['statuses']}"
    )
    print(f"Agent 顺序：{' → '.join(summary['agents'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
