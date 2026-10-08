"""投顾多智能体评估：`python eval/run_eval.py`。

层次定位
--------
本模块属于**评估层**（与 `src/` 业务代码分离，只读地调用业务接口，不被业务代码调用）：
向上由命令行 / CI 直接执行；向下调用 `src.pipeline`（端到端跑流水线）、
`src.constraints`（独立复核约束）、`src.suitability`（对抗探针）、`src.versioning`（版本链）。
通过在 `sys.path` 前置 `PROJECT_ROOT`，保证「在仓库根或任意 cwd 下执行」都能 import `src`。

对外暴露的关键对象
------------------
- `THRESHOLDS`：7 项指标的达标阈值（不达标即退出码 1）。
- `oracle_admissible()` / `candidate_filter_accuracy()`：独立实现的筛选「预言机」与交叉校验。
- `suitability_probes()`：对每位客户构造 2 个越界组合的对抗探针。
- `evaluate()`：执行全部评估并返回报告字典（明细 + 指标 + 阈值 + 失败项 + 计数器）。
- `render_text()`：把报告渲染成等宽文本。
- `main()`：CLI 入口（写 `report.json` / `report.md` 并返回退出码）。

主要输入 / 输出
---------------
输入：CLI 参数 `--engine {langgraph,native}`、`--json`、`--out-dir`；数据来自 `data/`（5 位
`eval_sample` 客户）。输出：`report.json`、`report.md` 两个报告文件，stdout 文本报告，
以及退出码（0 = 全部达标，1 = 存在不达标指标）。

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
（注：`avg_latency_ms` 是**越小越好**的唯一例外，其比较方向与其他 6 项相反，见 `evaluate` 与 `render_text`。）
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
#: 6 项比率类阈值均为 1.0（要求满分）；avg_latency_ms 为 5000.0（要求不超过 5 秒，越大越差）。
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

    参数：
        product：待判定的单个产品。
        client：客户档案（提供集中度上限与其余硬约束）。

    返回：True 表示该产品**可投**（无任何非集中度类阻断违反）。

    与 `screen_products` 的产品级判定是两条独立路径，用于交叉校验筛选正确性。
    组合级检查会额外引入集中度类结论，因此这里只关注非集中度类违反项。

    实现要点：构造「单产品顶格权重 = `client.max_single_product_ratio` + 其余补现金」的组合，
    再过滤掉 `CONCENTRATION_CODES`（集中度类代码）与 `C-LIQUIDITY`（流动性下限）两类结论——
    它们是把单品堆到上限的必然后果，与「这个产品本身准不准入」无关。

    副作用：无（纯计算）。
    """
    weight = client.max_single_product_ratio
    portfolio = Portfolio(
        weights={product.product_id: weight},
        products={product.product_id: product},
        cash_weight=round(1.0 - weight, 12),
    )
    violations = check_portfolio(portfolio, client)
    # 只保留「非集中度类、非流动性下限类」违反——它们才能说明产品本身不可投
    blocking = [v for v in violations if v.code not in CONCENTRATION_CODES and v.code != "C-LIQUIDITY"]
    return not blocking


def candidate_filter_accuracy(data: DataBundle) -> tuple[float, list[str]]:
    """逐客户逐产品比较筛选结果与预言机结果。

    参数：
        data：已装载的样例数据（提供 `sample_clients()` 与 `products`）。

    返回：`(正确率, 不一致明细列表)`；正确率 = `一致数 / 总比较数`，保留 6 位小数。
    明细每条形如 `"{client_id}/{pid}: 预言机={expected} 实际={actual}"`，供报告定位差异。

    遍历口径：对每位样例客户调用 `screen_products(client, data.products, 0)`（第 0 轮），
    再对**全市场每个产品**（按 product_id 排序）比对「预言机可投」与「实际入选」是否一致。

    边界：总比较数为 0（无客户或无产品）时正确率返回 **0.0**（而非 1.0）——此处的空集
    按「未验证」处理；注意这与其他指标的空集返回 1.0 的约定不同。

    副作用：无（只读数据）。
    """
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
    """对单个客户构造 2 个越界组合，返回 (探针名, 是否被正确拦截)。

    参数：
        client：客户档案（探针组合按其实约束构造，因此必须越界）。
        data：样例数据（提供产品 universe）。

    返回：固定长度 2 的列表，元素为 `(探针名, 是否被正确拦截)`；
    「正确拦截」的判定是 `not decision.passed and bool(decision.blocks)`
    ——即既不通过、又确实给出了 block 级依据（防止「空拦截」蒙对）。

    两个探针的构造口径：
    - 探针一「风险/集中度越界探针」：取全市场风险等级最高（同级取 product_id 排序）的产品，
      堆到 **0.95** 权重、其余 **0.05** 现金，必然触发集中度与/或风险等级规则；
    - 探针二「流动性/集中度越界探针」：在**通过筛选的候选池**中取流动性最低的产品做成**满仓**
      （权重 1.0、现金 0.0），必然触发流动性下限规则；候选池为空时退化为复用探针一。
    - 两个探针都用 `SuitabilityGate().review(ctx, 0)` 以第 0 轮复核，
      且 `RuleContext.build(...)` 传入对应的候选池上下文。

    副作用：无（不写 state、不落盘）。被 `evaluate()` 累加进 probe_total / probe_passed。
    """
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
    """执行评估并返回指标与明细。

    参数（均为关键字参数）：
        engine：编排引擎名，透传给 `run_pipeline`（默认 "langgraph"）。
        runs_dir：trace 输出目录；为 None 时不额外指定（本函数统一使用临时目录下的 trace_dir，
            因此该参数仅用于覆盖调用方场景）。

    返回：报告字典，键包括 `engine` / `clients` / `metrics`（7 项指标）/ `thresholds`
    （即 `THRESHOLDS`）/ `failures`（不达标指标名列表）/ `counters`（计数器）
    / `candidate_filter_mismatches` / `details`（逐客户明细，含探针、要素、耗时与版本链）。

    执行流程：
    1. `load_data()` 装载数据，取 `sample_clients()` 作为评估样本；
    2. 在 `tempfile.TemporaryDirectory()` 下建临时 `VersionStore` 与 trace 目录，
       对每位客户调用 `run_pipeline(..., auto=True, interactive=False)`——**隔离**：
       评估不会污染仓库真实的版本链与 runs/ 目录；
    3. 每位客户统计：硬约束违反数（独立调用 `check_portfolio` 复核）、
       被拦截轮次与 blocked 快照、探针命中、反事实覆盖率、情景数、要素完整度、耗时；
    4. 循环外单独跑一次 `candidate_filter_accuracy(data)`（与流水线解耦的交叉校验）；
    5. 汇总 7 项指标并计算 `failures`。

    指标计算细节与空集约定：
    - `constraint_satisfaction_rate = accepted_clean / accepted`，`accepted` 为 0 时返回 1.0；
    - `suitability_pass_rate = (block_handled + probe_passed) / (block_cases + probe_total)`，
      分母为 0 时返回 1.0；其中 `block_cases` 只统计「闸门确实命中过 block」的客户，
      `block_handled` 要求 directive ∈ {pass, reject} **且** 版本链里留下被拦截快照；
    - `counterfactual_coverage` / `stress_coverage` / `narrative_completeness` 均除以客户数，
      客户为空时返回 0.0；
    - `narrative_completeness` 是逐客户「达标要素数 / `len(NARRATIVE_ELEMENTS)`」的均值；
    - `avg_latency_ms` 用 `statistics.fmean` 求均值，无样本时为 0.0；
    - `failures` 的判定方向：`avg_latency_ms` 用 `>` 阈值（越小越好），其余用 `<` 阈值。

    副作用：写临时目录下的 trace 与版本链（随 `TemporaryDirectory` 退出被清理）；
    不写 `report.json` / `report.md`（那是 `main()` 的职责）。
    """
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
        # 临时版本链 + 临时 trace 目录：评估运行不污染仓库的 runs/ 与真实版本链
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

            # 覆盖率口径：非空反事实变体占比 > 0 即算达标（阈值 1.0 要求每份方案都有）
            counterfactual = state.get("counterfactual")
            if counterfactual is not None and counterfactual.coverage > 0:
                counterfactual_ok += 1
            # 压力测试口径：情景数 >= 3 才达标（与 THRESHOLDS.stress_coverage = 1.0 配合）
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

    # 达标判定：比率类要求 >= 阈值，唯独 avg_latency_ms 要求 <= 阈值（越小越好）
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
    """渲染文本报告。

    参数：
        report：`evaluate()` 的返回字典。

    返回：多行等宽文本（含指标表、逐客户明细表、计数器汇总、结论行），
    以 `"=" * 88` 分隔线包围，行末用换行符连接（`"\n".join`，**不含**结尾换行）。

    渲染要点：
    - 指标表逐项显示「指标中文名 / 取值 / 阈值 / 结果」，`avg_latency_ms` 显示为 `≤` 阈值
      且保留 2 位小数（判定 `value <= threshold`），其余显示为 `≥` 阈值并保留 4 位小数；
    - 明细表列依次为：客户 / 状态 / 轮次 / 违反 / block / 探针 / 反事实 / 情景 / 要素 / 耗时(ms)；
    - `candidate_filter_mismatches` 非空时逐条列出，为空则打印「与独立预言机逐产品完全一致」；
    - 结论行依 `report["failures"]` 区分「不达标…退出码 1」与「全部指标达标，退出码 0」。

    副作用：无（纯字符串拼装，不打印、不写文件）。
    """
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
    """CLI 入口。

    参数：
        argv：参数列表；None 表示取 `sys.argv[1:]`（由 argparse 决定）。

    返回：退出码——`report["failures"]` 非空时返回 **1**，否则返回 **0**。

    CLI 参数：
        `--engine {langgraph,native}`（默认 langgraph）
        `--json`（额外把 JSON 报告打印到 stdout）
        `--out-dir`（报告输出目录，默认本文件所在目录 `eval/`）

    副作用：
    - 打印文本报告（`report["failures"]` 与阈值逐项可见）；
    - 在 `--out-dir` 下写 `report.json`（`ensure_ascii=False`, indent=2, UTF-8）
      与 `report.md`（把文本报告包进 Markdown 代码块）；目录不存在时自动创建；
    - 被 `if __name__ == "__main__":` 块以 `sys.exit(main())` 调用，退出码即指标达标情况。
    """
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
