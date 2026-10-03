"""投顾多智能体评估：`python eval/run_eval.py`。

在 5 条样例客户上端到端跑流水线，输出 7 项指标：

1. 约束满足率 constraint_satisfaction_rate —— 被接受组合的硬约束违反数必须为 0
2. 适当性通过率 suitability_pass_rate —— 命中 block 规则是否被正确拦截/打回/拒绝
   （除流水线真实拦截外，另对每位客户做 2 个**对抗探针**：故意构造越界组合，必须被拦下）
3. 候选池过滤正确率 candidate_filter_accuracy —— 与独立实现的"预言机"逐产品交叉校验
4. 反事实解释覆盖率 counterfactual_coverage —— 每个方案都必须有非空反事实
5. 压力测试覆盖率 stress_coverage —— 每个方案都必须有 3 个情景结果
6. 建议书要素完整率 narrative_completeness —— 12 项必备要素
7. 平均耗时 avg_latency_ms

任一指标低于阈值即以退出码 1 结束。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constraints import CONCENTRATION_CODES, check_portfolio, screen_products  # noqa: E402
from src.dataset import DataBundle, load_data  # noqa: E402
from src.narrative import NARRATIVE_ELEMENTS  # noqa: E402
from src.pipeline import run_pipeline, state_summary  # noqa: E402
from src.schemas import ClientProfile, Portfolio, Product  # noqa: E402
from src.suitability import RuleContext, SuitabilityGate  # noqa: E402
from src.versioning import VersionStore  # noqa: E402

#: 指标阈值（不达标即退出码 1）
THRESHOLDS: dict[str, float] = {
    "constraint_satisfaction_rate": 1.0,
    "suitability_pass_rate": 1.0,
    "candidate_filter_accuracy": 1.0,
    "counterfactual_coverage": 1.0,
    "stress_coverage": 1.0,
    "narrative_completeness": 1.0,
    "avg_latency_ms": 5000.0,
}


# ---------------------------------------------------------------------------
# 独立实现的候选池"预言机"：走组合层约束检查这条**不同代码路径**
# ---------------------------------------------------------------------------
def oracle_admissible(product: Product, client: ClientProfile) -> bool:
    """预言机：把产品做成"顶格单产品组合"，用组合级约束检查判定是否可投。

    与 `screen_products` 的产品级判定是两条独立路径，用于交叉校验筛选正确性。
    组合级检查会额外引入集中度类结论，因此这里只关注非集中度类违反项。
    """
    weight = client.max_single_product_ratio
    portfolio = Portfolio(
        weights={product.product_id: weight},
        products={product.product_id: product},
        cash_weight=round(1.0 - weight, 12),
    )
    violations = check_portfolio(portfolio, client)
    blocking = [v for v in violations if v.code not in CONCENTRATION_CODES and v.code != "C-LIQUIDITY"]
    return not blocking


def candidate_filter_accuracy(data: DataBundle) -> tuple[float, list[str]]:
    """逐客户逐产品比较筛选结果与预言机结果。"""
    total = 0
    agree = 0
    mismatches: list[str] = []
    for client in data.sample_clients():
        screening = screen_products(client, data.products, 0)
        included = set(screening.included)
        for pid, product in sorted(data.products.items()):
            total += 1
            expected = oracle_admissible(product, client)
            actual = pid in included
            if expected == actual:
                agree += 1
            else:
                mismatches.append(f"{client.client_id}/{pid}: 预言机={expected} 实际={actual}")
    return (agree / total if total else 0.0), mismatches


# ---------------------------------------------------------------------------
# 对抗探针：故意构造越界组合，闸门必须拦下
# ---------------------------------------------------------------------------
def suitability_probes(client: ClientProfile, data: DataBundle) -> list[tuple[str, bool]]:
    """对单个客户构造 2 个越界组合，返回 (探针名, 是否被正确拦截)。"""
    gate = SuitabilityGate()
    universe = data.products
    results: list[tuple[str, bool]] = []

    # 探针一：把全市场风险最高的产品堆到 95% 权重（必然触发集中度/风险等级规则）
    riskiest = max(universe.values(), key=lambda p: (p.risk_level, p.product_id))
    probe_one = Portfolio(
        weights={riskiest.product_id: 0.95},
        products={riskiest.product_id: riskiest},
        cash_weight=0.05,
    )
    ctx_one = RuleContext.build(client, probe_one, universe, (riskiest.product_id,))
    decision_one = gate.review(ctx_one, 0)
    results.append(("风险/集中度越界探针", not decision_one.passed and bool(decision_one.blocks)))

    # 探针二：把可投池中流动性最低的产品做成满仓（必然触发流动性下限规则）
    candidates = screen_products(client, universe, 0).included
    if candidates:
        least_liquid = min((universe[pid] for pid in candidates), key=lambda p: (p.liquidity_ratio, p.product_id))
        probe_two = Portfolio(
            weights={least_liquid.product_id: 1.0},
            products={least_liquid.product_id: least_liquid},
            cash_weight=0.0,
        )
    else:
        probe_two = probe_one
    ctx_two = RuleContext.build(client, probe_two, universe, candidates)
    decision_two = gate.review(ctx_two, 0)
    results.append(("流动性/集中度越界探针", not decision_two.passed and bool(decision_two.blocks)))
    return results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def evaluate(*, engine: str = "langgraph", runs_dir: Path | None = None) -> dict[str, Any]:
    """执行评估并返回指标与明细。"""
    data = load_data()
    clients = data.sample_clients()

    accepted = 0
    accepted_clean = 0
    block_cases = 0
    block_handled = 0
    probe_total = 0
    probe_passed = 0
    counterfactual_ok = 0
    stress_ok = 0
    completeness_sum = 0.0
    latencies: list[float] = []
    details: list[dict[str, Any]] = []
    mismatch_pool: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        store = VersionStore(Path(tmp) / "chain.jsonl")
        trace_dir = Path(tmp) / "runs"
        for client in clients:
            started = time.perf_counter()
            state, tracer, pipeline = run_pipeline(
                client.client_id,
                engine=engine,
                auto=True,
                runs_dir=trace_dir,
                store=store,
                interactive=False,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(elapsed_ms)

            summary = state_summary(state)
            portfolio: Portfolio = state["portfolio"]
            violations = check_portfolio(portfolio, client)
            gate_rounds = state.get("gate_rounds") or []
            blocked_rounds = [item for item in gate_rounds if item["blocks"]]
            chain = store.chain(client.client_id)
            blocked_snapshots = [item for item in chain if item.status == "blocked"]

            if summary["directive"] == "pass":
                accepted += 1
                if not violations:
                    accepted_clean += 1

            if blocked_rounds:
                block_cases += 1
                # 正确处理的定义：拦截后要么重配通过、要么直接拒绝，且必须留下被拦截版本快照
                if summary["directive"] in {"pass", "reject"} and blocked_snapshots:
                    block_handled += 1

            probes = suitability_probes(client, data)
            probe_total += len(probes)
            probe_passed += sum(1 for _, ok in probes if ok)

            counterfactual = state.get("counterfactual")
            if counterfactual is not None and counterfactual.coverage > 0:
                counterfactual_ok += 1
            stress = state.get("stress")
            if stress is not None and stress.scenario_count >= 3:
                stress_ok += 1

            elements = state.get("elements") or {}
            completeness_sum += sum(1 for ok in elements.values() if ok) / len(NARRATIVE_ELEMENTS)

            details.append(
                {
                    "client_id": client.client_id,
                    "status": summary["status"],
                    "rounds": summary["rounds"],
                    "directive": summary["directive"],
                    "violations": len(violations),
                    "block_rules": summary["block_rules"],
                    "blocked_rounds": [item["round"] for item in blocked_rounds],
                    "warn_rules": summary["warn_rules"],
                    "human_decision": summary["human_decision"],
                    "version": summary["version"],
                    "probes": [{"name": name, "blocked": ok} for name, ok in probes],
                    "counterfactual_coverage": counterfactual.coverage if counterfactual else 0.0,
                    "stress_scenarios": stress.scenario_count if stress else 0,
                    "elements_ok": summary["elements_ok"],
                    "elements_total": len(NARRATIVE_ELEMENTS),
                    "latency_ms": round(elapsed_ms, 3),
                    "trace_steps": tracer.step_count(),
                    "version_chain": [
                        {"version": item.version, "status": item.status, "reason": item.change_reason}
                        for item in chain
                    ],
                }
            )

    filter_accuracy, mismatches = candidate_filter_accuracy(data)
    mismatch_pool.extend(mismatches)

    metrics = {
        "constraint_satisfaction_rate": round(accepted_clean / accepted, 6) if accepted else 1.0,
        "suitability_pass_rate": round((block_handled + probe_passed) / (block_cases + probe_total), 6)
        if (block_cases + probe_total)
        else 1.0,
        "candidate_filter_accuracy": round(filter_accuracy, 6),
        "counterfactual_coverage": round(counterfactual_ok / len(clients), 6) if clients else 0.0,
        "stress_coverage": round(stress_ok / len(clients), 6) if clients else 0.0,
        "narrative_completeness": round(completeness_sum / len(clients), 6) if clients else 0.0,
        "avg_latency_ms": round(statistics.fmean(latencies), 3) if latencies else 0.0,
    }

    failures = [
        name
        for name, threshold in THRESHOLDS.items()
        if (metrics[name] < threshold if name != "avg_latency_ms" else metrics[name] > threshold)
    ]

    return {
        "engine": engine,
        "clients": [client.client_id for client in clients],
        "metrics": metrics,
        "thresholds": THRESHOLDS,
        "failures": failures,
        "counters": {
            "accepted_portfolios": accepted,
            "accepted_with_zero_violation": accepted_clean,
            "block_cases": block_cases,
            "block_handled": block_handled,
            "probe_total": probe_total,
            "probe_blocked": probe_passed,
        },
        "candidate_filter_mismatches": mismatch_pool,
        "details": details,
    }


def render_text(report: dict[str, Any]) -> str:
    """渲染文本报告。"""
    lines = ["=" * 88, "财富管理投顾多智能体 —— 评估报告", "=" * 88]
    lines.append(f"编排引擎：{report['engine']}｜评估样本：{'、'.join(report['clients'])}")
    lines.append("")
    lines.append(f"{'指标':<34}{'取值':>12}{'阈值':>12}    结果")
    lines.append("-" * 88)
    labels = {
        "constraint_satisfaction_rate": "约束满足率",
        "suitability_pass_rate": "适当性通过率",
        "candidate_filter_accuracy": "候选池过滤正确率",
        "counterfactual_coverage": "反事实解释覆盖率",
        "stress_coverage": "压力测试覆盖率",
        "narrative_completeness": "建议书要素完整率",
        "avg_latency_ms": "平均耗时(ms)",
    }
    for name, threshold in report["thresholds"].items():
        value = report["metrics"][name]
        if name == "avg_latency_ms":
            ok = value <= threshold
            shown = f"{value:.2f}"
            limit = f"≤{threshold:.0f}"
        else:
            ok = value >= threshold
            shown = f"{value:.4f}"
            limit = f"≥{threshold:.2f}"
        lines.append(f"{labels[name]:<34}{shown:>12}{limit:>12}    {'通过' if ok else '不通过'}")

    lines.append("")
    lines.append("明细：")
    header = (
        f"  {'客户':<7}{'状态':<22}{'轮次':>4}{'违反':>5}{'block':>7}{'探针':>7}"
        f"{'反事实':>8}{'情景':>5}{'要素':>7}{'耗时(ms)':>10}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for item in report["details"]:
        probes_ok = sum(1 for probe in item["probes"] if probe["blocked"])
        probe_text = f"{probes_ok}/{len(item['probes'])}"
        element_text = f"{item['elements_ok']}/{item['elements_total']}"
        lines.append(
            f"  {item['client_id']:<7}{item['status']:<22}{item['rounds']:>4}{item['violations']:>5}"
            f"{len(item['block_rules']):>7}{probe_text:>7}"
            f"{item['counterfactual_coverage']:>8.0%}{item['stress_scenarios']:>5}"
            f"{element_text:>7}{item['latency_ms']:>10.2f}"
        )

    counters = report["counters"]
    lines.append("")
    lines.append(
        "计数：被接受组合 {accepted_portfolios} 个（其中 0 违反 {accepted_with_zero_violation} 个）｜"
        "闸门拦截场景 {block_cases} 个（正确处理 {block_handled} 个）｜"
        "对抗探针 {probe_total} 个（正确拦截 {probe_blocked} 个）".format(**counters)
    )
    if report["candidate_filter_mismatches"]:
        lines.append("候选池过滤与预言机不一致：")
        lines.extend(f"  - {item}" for item in report["candidate_filter_mismatches"])
    else:
        lines.append("候选池过滤与独立预言机逐产品完全一致。")

    lines.append("")
    if report["failures"]:
        lines.append(f"结论：不达标指标 {report['failures']}，退出码 1")
    else:
        lines.append("结论：全部指标达标，退出码 0")
    lines.append("=" * 88)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(prog="python eval/run_eval.py", description="投顾多智能体评估")
    parser.add_argument("--engine", choices=("langgraph", "native"), default="langgraph")
    parser.add_argument("--json", action="store_true", help="同时打印 JSON 报告")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent), help="报告输出目录")
    args = parser.parse_args(argv)

    report = evaluate(engine=args.engine)
    print(render_text(report))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(
        "# 评估报告\n\n```\n" + render_text(report) + "\n```\n", encoding="utf-8"
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
