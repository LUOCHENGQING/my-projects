"""建议版本链查看：`python -m src.history <client_id>`。

打印某位客户的建议版本链（v1 → v2 → …，含父版本与变更原因），
并默认对最后两个版本做结构化 diff（也可用 `--from/--to` 指定任意两个版本）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .utils import pct
from .versioning import DEFAULT_CHAIN_PATH, AdviceSnapshot, VersionStore, chain_rows, diff_versions

LINE = "─" * 100


def _print_chain(chain: Sequence[AdviceSnapshot], verbose: bool) -> None:
    """打印版本链表格。"""
    print(LINE)
    print(f"{'版本':<6}{'父版本':<8}{'状态':<26}{'变更原因':<44}{'哈希':<14}时间")
    print(LINE)
    for row in chain_rows(chain):
        parent = "-" if row["parent_version"] is None else f"v{row['parent_version']}"
        print(
            f"v{row['version']:<5}{parent:<8}{row['status']:<26}{row['change_reason'][:40]:<44}"
            f"{row['hash'][:12]:<14}{row['created_at']}"
        )
    if verbose:
        print(LINE)
        for snapshot in chain:
            weights = snapshot.portfolio_weights
            print(
                f"v{snapshot.version} 持仓 {len(weights)} 只｜现金 {pct(snapshot.cash_weight)}｜"
                f"校验哈希一致：{snapshot.verify()}"
            )
            for pid, weight in sorted(weights.items()):
                print(f"    {pid:<12}{pct(weight):>9}")


def _print_diff(before: AdviceSnapshot, after: AdviceSnapshot) -> None:
    """打印两个版本的结构化差异。"""
    diff = diff_versions(before, after)
    print(LINE)
    print(f"版本差异：v{diff['from_version']} → v{diff['to_version']}｜变更原因：{diff['change_reason']}")
    print(LINE)
    print(f"  状态：{diff['status']['from']} → {diff['status']['to']}")
    print(f"  新增产品：{diff['products_added'] or '无'}")
    print(f"  剔除产品：{diff['products_removed'] or '无'}")
    print(f"  现金占比：{pct(diff['cash_weight']['from'])} → {pct(diff['cash_weight']['to'])}")
    if diff["weight_changes"]:
        print("  权重变化：")
        for pid, delta in sorted(diff["weight_changes"].items()):
            print(f"    {pid:<12}{delta * 100:+.2f}pp")
    print("  指标变化：")
    for key, delta in sorted(diff["metric_changes"].items()):
        if abs(delta) > 1e-12:
            print(f"    {key:<24}{delta:+.6f}")
    if diff["rule_hits_added"]:
        print(f"  新增命中规则：{'、'.join(diff['rule_hits_added'])}")
    if diff["rule_hits_removed"]:
        print(f"  解除命中规则：{'、'.join(diff['rule_hits_removed'])}")
    if diff["constraint_changes"]:
        print("  客户约束变化：")
        for key, change in sorted(diff["constraint_changes"].items()):
            print(f"    {key:<28}{change['from']} → {change['to']}")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(
        prog="python -m src.history",
        description="查看某位客户的投顾建议版本链，并 diff 两个版本",
    )
    parser.add_argument("client_id", help="客户号，例如 C004")
    parser.add_argument("--chain-file", default=str(DEFAULT_CHAIN_PATH), help="版本链文件路径")
    parser.add_argument("--from", dest="from_version", type=int, default=None, help="起始版本号")
    parser.add_argument("--to", dest="to_version", type=int, default=None, help="目标版本号")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整载荷")
    parser.add_argument("--quiet", action="store_true", help="只打印版本链，不打印 diff")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(argv)
    store = VersionStore(Path(args.chain_file))
    chain = store.chain(args.client_id)

    if not chain:
        print(f"未找到客户 {args.client_id} 的建议版本链（文件：{store.path}）")
        print("提示：先执行 `python -m src.demo --auto` 生成建议版本。")
        return 1

    print(f"客户 {args.client_id} 建议版本链（共 {len(chain)} 个版本）｜文件：{store.path}")
    if args.json:
        print(json.dumps([item.to_record() for item in chain], ensure_ascii=False, indent=2))
        return 0

    _print_chain(chain, verbose=not args.quiet)

    if args.quiet:
        return 0

    before: AdviceSnapshot | None = None
    after: AdviceSnapshot | None = None
    if args.from_version is not None and args.to_version is not None:
        before = next((item for item in chain if item.version == args.from_version), None)
        after = next((item for item in chain if item.version == args.to_version), None)
        if before is None or after is None:
            print("指定的版本号不存在于该客户的版本链中")
            return 1
    elif len(chain) >= 2:
        before, after = chain[-2], chain[-1]
    else:
        print(LINE)
        print("该客户仅有 1 个版本，无可 diff 的历史版本。")
        return 0

    _print_diff(before, after)
    return 0


if __name__ == "__main__":
    sys.exit(main())
