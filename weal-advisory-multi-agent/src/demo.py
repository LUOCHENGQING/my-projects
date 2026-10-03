"""演示入口：`python -m src.demo --auto`。

依次展示每位样例客户的完整链路：
    客户约束 → 候选池（含剔除原因）→ 组合构建 → 适当性闸门（拦截/打回重配）
    → 反事实解释 → 情景压力测试 → 投顾建议书 → 建议版本链 → 人工确认留痕

默认使用 mock LLM（无 Key 也一定跑得通），`--engine native` 可强切零依赖引擎。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .agents import agent_catalog, tool_names
from .constraints import check_portfolio
from .counterfactual import counterfactual_to_rows
from .dataset import DataBundle, load_data
from .llm import llm_status
from .pipeline import AdvisoryPipeline, PipelineConfig, run_pipeline, state_summary
from .schemas import AdviceRecord, GateDecision, HumanReview, Portfolio
from .stress import stress_to_rows
from .suitability import ALL_RULES, format_rule_table
from .utils import pct
from .versioning import VersionStore, chain_rows

LINE = "─" * 100
DOUBLE = "═" * 100


def _section(title: str) -> None:
    print()
    print(f"【{title}】")


def _weights_table(portfolio: Portfolio, investable: float) -> None:
    """打印持仓表。"""
    if not portfolio.held_ids() and portfolio.cash_weight <= 0:
        print("  （无持仓）")
        return
    amounts = portfolio.holding_amounts(investable)
    print(f"  {'产品代码':<12}{'产品名称':<18}{'类别':<8}{'风险':<6}{'权重':>9}{'参考金额(元)':>16}")
    for pid in portfolio.held_ids():
        product = portfolio.products[pid]
        print(
            f"  {pid:<12}{product.name:<18}{product.asset_class:<8}R{product.risk_level:<5}"
            f"{pct(portfolio.weights[pid]):>9}{amounts[pid]:>16,.0f}"
        )
    if portfolio.cash_weight > 0:
        print(
            f"  {'CASH':<12}{'现金/活期留存':<18}{'现金':<8}{'R1':<6}"
            f"{pct(portfolio.cash_weight):>9}{portfolio.cash_weight * investable:>16,.0f}"
        )


def print_client_report(
    *,
    client_id: str,
    state: dict,
    pipeline: AdvisoryPipeline,
    tracer,
    show_narrative: bool,
) -> None:
    """打印一位客户的完整链路。"""
    client = state["client"]
    summary = state_summary(state)
    portfolio: Portfolio = state["portfolio"]
    gate: GateDecision | None = state.get("gate")
    advice: AdviceRecord = state["advice"]
    review: HumanReview = state["human_review"]

    print(DOUBLE)
    print(f"客户 {client.client_id}｜{client.display_name}｜{client.age} 周岁｜"
          f"风险等级 R{client.risk_capacity}｜投资期限 {client.investment_horizon_years:g} 年｜"
          f"可投金额 {client.investable_amount:,.0f} 元")
    print(DOUBLE)

    _section("1. 客户约束（由 ClientProfilingAgent 提取）")
    for line in state["profile"]["constraint_digest_lines"]:
        print(f"  · {line}")
    if state["profile"].get("summary"):
        print(f"  → {state['profile']['summary']}")
    print(f"  → {state['profile'].get('consistency_note', '')}")

    _section("2. 候选产品池（ProductScreeningAgent，硬约束可行域）")
    screening = state["screening"]
    print(f"  全市场候选 {screening.universe_size} 只 → 可行域 {len(screening.included)} 只")
    print(f"  入选：{'、'.join(screening.included) or '（空）'}")
    for item in screening.excluded[:6]:
        print(f"  剔除 {item.product_id} {item.product_name}：{'/'.join(item.reasons)} — {item.detail}")
    if len(screening.excluded) > 6:
        print(f"  …… 其余 {len(screening.excluded) - 6} 只剔除记录见 trace 与版本快照")

    _section("3. 组合构建（PortfolioOptimizerAgent）")
    _weights_table(portfolio, client.investable_amount)
    metrics = portfolio.metrics
    print(
        f"  组合指标：预期收益 {pct(metrics.get('expected_return', 0.0))}｜"
        f"预期波动 {pct(metrics.get('expected_volatility', 0.0))}｜"
        f"流动性 {pct(metrics.get('liquidity_ratio', 0.0))}｜"
        f"最大单一持仓 {pct(metrics.get('max_single_weight', 0.0))}｜"
        f"综合费率 {pct(metrics.get('expected_fee_rate', 0.0), 3)}｜"
        f"持仓 {int(metrics.get('holding_count', 0))} 只"
    )
    violations = check_portfolio(portfolio, client)
    print(f"  硬约束违反数：{len(violations)}（要求恒为 0）")

    _section("4. 适当性闸门（SuitabilityOfficerAgent，确定性规则）")
    for item in state.get("gate_rounds") or []:
        print(
            f"  第 {item['round'] + 1} 轮：{item['directive']}　"
            f"block={item['blocks'] or '无'}　warn={item['warns'] or '无'}"
        )
    if gate is not None:
        for violation in gate.blocks:
            print(f"    [BLOCK] {violation.rule_id}：{violation.detail}")
        for violation in gate.warns:
            print(f"    [WARN ] {violation.rule_id}：{violation.detail}")
    if state.get("tighten"):
        tighten = {k: v for k, v in state["tighten"].items() if v not in (None, [], ())}
        print(f"  → 打回重配下发的约束收紧指令：{tighten}")

    _section("5. 反事实解释（改约束 → 重新求解 → 结构化 diff）")
    print("  基线：定稿方案实际依据的生效约束；适当性判定始终针对客户真实档案。")
    counterfactual = state.get("counterfactual")
    if counterfactual is not None:
        for row in counterfactual_to_rows(counterfactual):
            print(f"  · {row['question']}")
            print(
                f"      {row['status']}｜新增 {row['products_added'] or '无'}｜"
                f"剔除 {row['products_removed'] or '无'}｜收益变动 {row['return_delta'] * 100:+.2f}pp｜"
                f"新增命中 {row['rules_introduced'] or '无'}"
            )
        print(f"  非空差异覆盖率：{counterfactual.coverage:.0%}")

    _section("6. 情景压力测试（确定性敏感性系数表）")
    stress = state.get("stress")
    if stress is not None:
        print(f"  {stress.formula}")
        for row in stress_to_rows(stress):
            print(
                f"  · {row['name']}：组合估值冲击 {row['portfolio_impact'] * 100:+.2f}%｜"
                f"估计最大回撤 {row['estimated_drawdown'] * 100:.2f}%｜"
                f"{'超出' if row['exceeds_tolerance'] else '未超出'}客户回撤容忍度"
            )

    _section("7. 人工确认（人机协同）")
    if review.required:
        print(f"  需人工确认：是（{review.decision}）｜原因：{'；'.join(review.reasons)}")
        print(f"  操作人：{review.operator}｜降级：{'是' if review.degraded else '否'}｜{review.note}")
    else:
        print("  需人工确认：否")

    _section("8. 投顾建议书（AdvisorNarrativeAgent）")
    elements_ok = sum(1 for ok in advice.elements.values() if ok)
    print(f"  要素齐备：{elements_ok}/{len(advice.elements)}｜正文长度 {len(advice.narrative)} 字")
    if show_narrative:
        print(LINE)
        print(advice.narrative)
        print(LINE)

    _section("9. 建议版本链与运行痕迹")
    chain = pipeline.store.chain(client.client_id)
    for row in chain_rows(chain):
        print(
            f"  v{row['version']}（父版本 {row['parent_version']}）｜{row['status']}｜"
            f"{row['change_reason']}｜持仓 {row['holdings']} 只｜哈希 {row['hash'][:12]}"
        )
    print(f"  最终状态：{summary['status']}｜轮次 {summary['rounds']}｜trace 步数 {tracer.step_count()}")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(
        prog="python -m src.demo",
        description="财富管理投顾多智能体演示（约束驱动：硬约束求解 + 适当性闸门 + 反事实 + 压力测试）",
    )
    parser.add_argument("--client", action="append", default=None, help="指定客户号，可重复；缺省跑全部样例客户")
    parser.add_argument("--engine", choices=("langgraph", "native"), default="langgraph", help="编排引擎")
    parser.add_argument("--auto", action="store_true", help="人工确认环节自动放行（演示用）")
    parser.add_argument("--max-rounds", type=int, default=2, help="适当性打回重配的最大轮次")
    parser.add_argument("--exempt", action="append", default=None, help="人工豁免的 block 规则号，可重复")
    parser.add_argument("--brief", action="store_true", help="不打印建议书全文")
    parser.add_argument("--catalog", action="store_true", help="只打印规则清单与 Agent 能力清单")
    parser.add_argument("--runs-dir", default=None, help="trace 输出目录（缺省 runs/）")
    parser.add_argument("--chain-file", default=None, help="建议版本链文件（缺省 runs/version_chain.jsonl）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(argv)
    data: DataBundle = load_data()

    if args.catalog:
        print("适当性规则清单（%d 条）：" % len(ALL_RULES))
        print(format_rule_table())
        print()
        print("Agent 能力清单：")
        for item in agent_catalog():
            print(f"  · {item['name']}（{item['role']}）")
            print(f"      工具白名单：{'、'.join(item['tools'])}")
            print(f"      权限集合：{'、'.join(item['permissions'])}")
            print(f"      可写状态：{'、'.join(item['can_write_state'])}")
        print()
        print(f"工具总数：{len(tool_names())}")
        return 0

    client_ids = args.client or [c.client_id for c in data.all_clients()]
    runs_dir = Path(args.runs_dir) if args.runs_dir else None
    store = VersionStore(args.chain_file) if args.chain_file else VersionStore()

    print(DOUBLE)
    print("财富管理投顾多智能体（约束驱动）—— 演示运行")
    print(DOUBLE)
    print(f"编排引擎：{args.engine}｜人工确认：{'自动放行(--auto)' if args.auto else '按环境判定'}")
    print(f"样例客户：{'、'.join(client_ids)}")
    print(f"适当性规则：{len(ALL_RULES)} 条（block/warn 分级）｜工具白名单：{len(tool_names())} 个")

    total_latency = 0.0
    total_steps = 0
    for index, client_id in enumerate(client_ids):
        state, tracer, pipeline = run_pipeline(
            client_id,
            engine=args.engine,
            auto=args.auto,
            runs_dir=runs_dir,
            store=store,
            max_repair_rounds=args.max_rounds,
            exempt_rules=tuple(args.exempt or ()),
            interactive=False,
        )
        if index == 0:
            status = llm_status(pipeline.llm)
            print(f"LLM 模式：{status['mode']}｜模型 {status['model']}｜密钥 {status.get('api_key', '-')}")
            if pipeline.engine_note:
                print(f"引擎说明：{pipeline.engine_note}")
        print_client_report(
            client_id=client_id,
            state=state,
            pipeline=pipeline,
            tracer=tracer,
            show_narrative=not args.brief,
        )
        print(f"  （trace 已写入：{tracer.path}）")
        total_latency += tracer.total_latency_ms()
        total_steps += tracer.step_count()

    print(DOUBLE)
    print(f"演示完成：{len(client_ids)} 位客户｜累计 trace 步数 {total_steps}｜累计耗时 {total_latency:.2f} ms")
    print(f"建议版本链文件：{store.path}")
    print(DOUBLE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
