"""运行痕迹回放：`python -m src.replay <run_id>`。

所处层次
--------
本模块是 **CLI 展示层**（薄封装，不含业务逻辑）：只把 `observability` 读出的
落盘 trace 渲染成表格与汇总。调用方式是命令行（`python -m src.replay ...`），
也被 `demo.py` / `history.py` 的提示文案指向；`observability` 负责写入，本模块
负责读取，两者通过 `runs/<run_id>.jsonl` 这一约定耦合。

功能
----
读取 `runs/<run_id>.jsonl`，按步骤打印 `step / agent / node / latency / status /
tool_calls / input_digest → output_digest`，并给出汇总（步数、总耗时、参与 Agent）。

`--list` 列出全部可回放的 run_id；`--json` 输出原始 trace。

对外暴露
--------
- `build_parser()`：构造 argparse 解析器
- `main(argv=None)`：CLI 入口，返回进程退出码（0 成功；1 表示 trace 目录无内容或
  指定 run_id 不存在）

主要输入输出
------------
输入为命令行参数与 `--runs-dir` 指定的目录；输出全部通过 `print` 写 stdout，
不写文件。`read_trace` 抛出的 `FileNotFoundError` 在此被捕获并转成退出码 1。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .observability import RUNS_DIR, format_trace_table, list_runs, read_trace, trace_summary


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。

    返回:
        配置好的 `argparse.ArgumentParser`，参数为：
        `run_id`（位置参数，可省略）、`--runs-dir`（默认 `RUNS_DIR`）、
        `--list`（列出全部 run_id）、`--json`（输出原始 trace JSON）。
    """
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
    """CLI 入口。

    参数:
        argv: 参数序列；None 时由 argparse 取 `sys.argv[1:]`（便于测试注入）。

    返回:
        进程退出码：0 表示成功（列出目录 / 输出 JSON / 完成回放）；1 表示
        `--list` 时目录下没有 trace，或指定的 `run_id` 不存在。

    副作用:
        通过 `print` 向 stdout 输出运行列表、trace JSON 或步骤表与汇总。
    """
    args = build_parser().parse_args(argv)
    base = Path(args.runs_dir)

    # 未提供 run_id 时等同于 --list：先列出可回放的运行，避免用户面对空输出
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
        # trace 文件缺失属于可预期的用户输入错误：打印原因并返回 1，不打印栈
        print(str(exc))
        return 1

    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return 0

    print(f"回放运行 {args.run_id}（{len(records)} 步）｜目录：{base}")
    print(format_trace_table(records))
    summary = trace_summary(records)
    print("-" * 60)
    # total_latency_ms 由 trace_summary 保证为 float，可直接格式化；
    # agents 用 len() 而不是直接打印，避免 Agent 多时刷屏
    print(
        f"汇总：步数 {summary['steps']}｜总耗时 {summary['total_latency_ms']:.2f} ms｜"
        f"参与 Agent {len(summary['agents'])} 个｜状态 {summary['statuses']}"
    )
    print(f"Agent 顺序：{' → '.join(summary['agents'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
