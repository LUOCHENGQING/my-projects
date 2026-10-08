"""评测脚本：`python eval/run_eval.py`

在 5 条样例问题上端到端跑完整条多智能体流水线，输出四项指标：

    任务完成率        task_completion_rate      报告完整且结论数达标的问题占比
    检索命中率        retrieval_hit_rate        期望资料来源被召回的平均覆盖率
    引用可追溯率      citation_traceability_rate 报告中每个 [n] 都能回溯到真实来源
    平均耗时          avg_latency_ms            端到端平均耗时

用法：
    python eval/run_eval.py
    python eval/run_eval.py --engine native
    python eval/run_eval.py --json eval/last_report.json

层次与职责
----------
本模块属于「评测层」（离线质量门禁），不在运行时依赖链上：它 import 运行时
（`src.orchestrator.ResearchPipeline` 等），而 `src/` 内没有任何模块 import 它。
它把 `eval/cases.json` 里的用例逐条跑完，做**判定**（不是只看能不能跑通），
再把明细落盘并给出进程退出码供 CI 卡门。

对外关键函数
------------
* `load_cases(path)`                       读取用例文件（默认 `eval/cases.json`）
* `evaluate_case(pipeline, case)`          跑一条用例并按四个维度判定
* `citation_traceability(report, cites)`   单条用例的引用可追溯性判定
* `summarize(rows)`                        把逐条结果聚合成指标字典
* `main(argv)`                             入口：跑全量、打印表格、落盘、返回退出码

四项指标的判定口径（与实现一一对应）
------------------------------------
1. **任务完成率**：单条用例同时满足「报告非空」+「期望章节齐全」+「期望关键词齐全」
   +「结论数 >= min_findings」才算完成；总指标 = 完成条数 / 用例总数。
2. **检索命中率**：期望来源（expect_sources）出现在 evidence 的 source_id 集合里的比例，
   逐条取平均；某条用例没声明期望来源时该条记 1.0。
3. **引用可追溯率**：报告里的每个 `[n]` 都能在 citations 的 citation_no 里找到，
   且报告**至少有一个**引用，才算可追溯（空引用判为不通过）；总指标 = 通过条数 / 用例总数。
4. **平均耗时**：各用例 `summary.total_latency_ms` 的算术平均（`statistics.fmean`），
   同时额外给出 max/min 便于看抖动。
注：实际实现还多输出一项 `verdict_accuracy`（风险裁决符合率，用例写了 expect_verdict 时
    才比较 `state["risk_verdict"]`），它只作为观测项展示，**不参与退出码门禁**。

退出码门禁
----------
`main()` 末尾用三个核心比例卡门：任务完成率、检索命中率、引用可追溯率**都必须 >= 0.99**
才返回 0，否则返回 1（平均耗时只展示、不设阈值）。这样 CI 里直接看退出码即可判断
「质量是否退化」，而不必解析输出文本。指标明细同时写入 `--json` 指定的文件
（默认 `eval/last_report.json`），包含 generated_at / engine / llm_mode / metrics / cases。

离线可跑
--------
无 API Key 时 `use_mock_llm()` 为真，流水线全部走确定性 mock 大脑，
因此**同一份用例在离线机器上重复评测，指标稳定可复现**，可作为回归基线。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EVAL_DIR, use_mock_llm  # noqa: E402
from src.orchestrator import ResearchPipeline  # noqa: E402
from src.utils.console import ensure_utf8_console  # noqa: E402

LINE = "=" * 78


def load_cases(path: Path | None = None) -> List[Dict[str, Any]]:
    """读取用例文件。

    参数：
        path: 用例文件路径；为 None 时取默认的 `EVAL_DIR / "cases.json"`。
    返回：
        List[Dict[str, Any]] —— `cases` 字段的内容；文件里没有该键时返回空列表。
        单条用例常用字段：id / question / expect_sections / expect_keywords /
        min_findings / expect_sources / expect_verdict / expect_period。
    副作用/异常：
        只读文件（UTF-8）；文件不存在或不是合法 JSON 时会抛
        FileNotFoundError / json.JSONDecodeError，**不做兜底**——用例缺失属于配置错误，
        应当立刻失败而不是静默跑出一个漂亮但无意义的分数。
    """
    target = path or (EVAL_DIR / "cases.json")
    payload = json.loads(target.read_text(encoding="utf-8"))
    return list(payload.get("cases") or [])


def citation_traceability(report: str, citations: List[Dict[str, Any]]) -> Tuple[bool, List[int]]:
    """检查报告中出现的引用编号是否全部可回溯。

    参数：
        report:    简报全文，可能为 None/空串。
        citations: 引用清单，每项含 citation_no。
    返回：
        Tuple[bool, List[int]] —— 第一个元素是判定结果，第二个是悬空编号列表（升序）。
        判定为 True 的条件是：`used` 非空（报告里至少出现一个 `[n]`）**且**没有悬空编号。
        注：实际实现为「一条引用都没有」也算**不可追溯**（`bool(used) and not dangling`），
        与 `src/demo.py` 里同名自检函数把空引用按 1.0 处理的口径不同——
        评测更严格，因为「没引用」在投研场景里等于无法审计。
    副作用/异常：
        纯函数，不读文件；`report or ""` 保证 None 不会触发 TypeError。
    """
    used = sorted({int(m) for m in re.findall(r"\[(\d+)\]", report or "")})
    known = {int(c.get("citation_no", 0)) for c in citations}
    dangling = [n for n in used if n not in known]
    return (bool(used) and not dangling, dangling)


def evaluate_case(pipeline: ResearchPipeline, case: Dict[str, Any]) -> Dict[str, Any]:
    """跑一条样例并计算该样例的各项判定。

    参数：
        pipeline: 已装配好的流水线（`--engine` 决定用哪个图引擎），**复用同一实例**跑所有
                  用例以避免重复建索引——代价是用例之间可能经由工具内部缓存互相影响，
                  因此 `registry` 的幂等缓存是评测可复现性的前提。
        case:     单条用例字典，必填 `id` 与 `question`，其余判定字段见 `load_cases`。
    返回：
        Dict[str, Any] —— 单条结果行，除 id/question 外包含：
        判定项 completed / sections_ok / keywords_ok / findings_ok / traceable /
        hit_rate / verdict_ok / period_ok，观测项 run_id / retrieved_sources /
        dangling_citations / latency_ms / verdict / risk_level / revision_round /
        findings / citations / report_chars / errors / error。
    副作用/异常：
        * 会真实执行整条流水线（写轨迹文件、消耗 LLM 调用配额）；
        * **单条失败不中断整轮评测**：`pipeline.run` 抛异常时捕获后返回一条
          `completed=False / hit_rate=0.0 / traceable=False` 的失败行，并把异常类型与
          消息写进 `error`（该失败行的键比成功行少，下游用 `.get()` 读取）；
        * 用例缺 `id`/`question` 时会 KeyError（属于用例文件错误，不兜底）。
    判定细节：
        * 章节判定用 `f"、{s}" in report or s in report`，兼容「标题带顿号连接词」的写法；
        * `min_findings` 缺失时按 0 处理（即不做结论数要求）；
        * `expect_verdict` / `expect_period` 缺失时对应判定直接为 True（不作为约束）。
    """
    started = time.perf_counter()
    try:
        result = pipeline.run(case["question"])
        error = ""
    except Exception as exc:  # noqa: BLE001 - 单条样例失败不应中断整轮评测
        return {
            "id": case["id"],
            "question": case["question"],
            "completed": False,
            "hit_rate": 0.0,
            "traceable": False,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "error": f"{type(exc).__name__}: {exc}",
        }

    state = result["state"]
    report = state.get("report") or ""
    findings = state.get("findings") or []
    evidence = state.get("evidence") or []
    citations = state.get("citations") or []
    plan = state.get("plan") or {}

    # ---- 任务完成率判定 ----
    # 三项都要满足且报告非空：只看「跑通没报错」会把空壳报告也算成功，失去门禁意义
    sections_ok = all(f"、{s}" in report or s in report for s in case.get("expect_sections", []))
    keywords_ok = all(k in report for k in case.get("expect_keywords", []))
    findings_ok = len(findings) >= int(case.get("min_findings", 0))
    completed = bool(report) and sections_ok and keywords_ok and findings_ok

    # ---- 检索命中率判定 ----
    retrieved_sources = {str(e.get("source_id")) for e in evidence}
    expect_sources = [str(s) for s in case.get("expect_sources", [])]
    hits = [s for s in expect_sources if s in retrieved_sources]
    # 没声明期望来源的用例不参与检索判定，记满分而不是记 0（否则会冤枉用例本身）
    hit_rate = len(hits) / len(expect_sources) if expect_sources else 1.0

    # ---- 引用可追溯率判定 ----
    traceable, dangling = citation_traceability(report, citations)

    # ---- 其他观测项 ----
    # 这两项在用例未声明期望值时默认 True：不作为约束，只用于观察行为漂移
    verdict_ok = True
    if case.get("expect_verdict"):
        verdict_ok = state.get("risk_verdict") == case["expect_verdict"]
    period_ok = True
    if case.get("expect_period"):
        period_ok = plan.get("period") == case["expect_period"]

    return {
        "id": case["id"],
        "question": case["question"],
        "run_id": result["run_id"],
        "completed": completed,
        "sections_ok": sections_ok,
        "keywords_ok": keywords_ok,
        "findings_ok": findings_ok,
        "hit_rate": round(hit_rate, 4),
        "retrieved_sources": sorted(retrieved_sources),
        "traceable": traceable,
        "dangling_citations": dangling,
        "latency_ms": round(result["summary"]["total_latency_ms"], 3),
        "verdict": state.get("risk_verdict"),
        "verdict_ok": verdict_ok,
        "period_ok": period_ok,
        "risk_level": state.get("risk_level"),
        "revision_round": state.get("revision_round"),
        "findings": len(findings),
        "citations": len(citations),
        "report_chars": len(report),
        "errors": state.get("errors") or [],
        "error": error,
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把逐条结果聚合成指标字典。

    参数：
        rows: `evaluate_case` 返回的结果行列表（成功行与失败行可混合）。
    返回：
        Dict[str, Any] —— case_count、task_completion_rate、retrieval_hit_rate、
        citation_traceability_rate、verdict_accuracy、avg_latency_ms，
        以及仅作观测的 max_latency_ms / min_latency_ms；各比例保留 4 位小数。
    副作用/异常：
        纯函数，不写文件；`total = len(rows) or 1` 让空用例列表不会触发 ZeroDivisionError
        （此时各比例为 0.0，耗时类指标为 0.0）。失败行缺 `hit_rate` 之类字段时按 0/默认值参与。
    口径：
        三个比例为「满足条件的条数 / 用例总数」，是**逐条判定的占比**，
        不是各条内部比率的平均。
    """
    total = len(rows) or 1
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    return {
        "case_count": len(rows),
        "task_completion_rate": round(sum(1 for r in rows if r["completed"]) / total, 4),
        "retrieval_hit_rate": round(sum(float(r["hit_rate"]) for r in rows) / total, 4),
        "citation_traceability_rate": round(sum(1 for r in rows if r["traceable"]) / total, 4),
        "verdict_accuracy": round(sum(1 for r in rows if r.get("verdict_ok")) / total, 4),
        "avg_latency_ms": round(statistics.fmean(latencies), 3) if latencies else 0.0,
        "max_latency_ms": round(max(latencies), 3) if latencies else 0.0,
        "min_latency_ms": round(min(latencies), 3) if latencies else 0.0,
    }


def main(argv: List[str] | None = None) -> int:
    """评测入口：解析参数 -> 逐条跑用例 -> 打印指标 -> 落盘明细 -> 返回退出码。

    参数：
        argv: 参数列表；None 时由 argparse 读取真实 `sys.argv`。
              支持 `--cases`（用例文件）、`--engine`（auto|langgraph|native）、
              `--json`（指标落盘路径，默认 eval/last_report.json）、`--verbose`（打印失败明细）。
    返回：
        int —— **0 表示三项核心指标（任务完成率 / 检索命中率 / 引用可追溯率）全部 >= 0.99**，
        否则 1；供 CI 直接判断质量门禁（返回 1 时指标仍会正常打印并落盘，便于排查）。
    副作用/异常：
        * 修改控制台编码（`ensure_utf8_console()`）；
        * 构造 ResearchPipeline（加载语料、建索引），对每条用例真实执行整条流水线；
        * 创建 `--json` 的父目录并覆盖写入 UTF-8 JSON 报告；
        * 用例文件缺失或非法 JSON 会直接抛异常终止（不落盘、不返回退出码）。
    """
    ensure_utf8_console()
    parser = argparse.ArgumentParser(prog="python eval/run_eval.py", description="投研多智能体评测")
    parser.add_argument("--cases", default=None, help="用例文件（默认 eval/cases.json）")
    parser.add_argument("--engine", choices=["auto", "langgraph", "native"], default=None)
    parser.add_argument("--json", default=str(EVAL_DIR / "last_report.json"), help="指标落盘路径")
    parser.add_argument("--verbose", action="store_true", help="打印每步失败原因明细")
    args = parser.parse_args(argv)

    cases = load_cases(Path(args.cases) if args.cases else None)

    print("")
    print(LINE)
    print("金融投研多智能体系统  ·  评测")
    print(LINE)
    print(f"用例数量   : {len(cases)}")
    print(f"LLM 模式   : {'mock（确定性规则大脑，离线可复现）' if use_mock_llm() else 'OpenAI 兼容接口'}")
    print(f"用例文件   : {args.cases or (EVAL_DIR / 'cases.json')}")

    started = time.perf_counter()
    # quiet=True：评测不需要逐步刷屏，过程信息由本脚本自己的表格承担
    pipeline = ResearchPipeline(auto=True, engine=args.engine, quiet=True)
    # 运行前引擎名可能还是默认值（auto 尚未解析），此时先展示「待定」而不是误报 native
    print(f"编排引擎   : {pipeline.engine_name if pipeline.engine_name != 'native' else '待定'} "
          f"（优先 LangGraph，失败自动降级自研引擎）")
    print(f"资料库     : {len(pipeline.documents)} 篇文档 / {len(pipeline.children)} 个子块 / "
          f"{len(pipeline.fact_store)} 条结构化事实")

    rows: List[Dict[str, Any]] = []
    print("")
    print(f"{'用例':<10}{'完成':<6}{'命中率':<9}{'可追溯':<8}{'裁决':<10}{'轮次':<6}{'结论':<6}{'耗时(ms)':>12}")
    print("-" * 78)
    for case in cases:
        row = evaluate_case(pipeline, case)
        rows.append(row)
        print(
            f"{row['id']:<10}{'✔' if row['completed'] else '✘':<6}"
            f"{row['hit_rate'] * 100:>6.1f}%  {'✔' if row['traceable'] else '✘':<8}"
            f"{str(row.get('verdict', '-')):<10}{row.get('revision_round', '-'):<6}"
            f"{row.get('findings', '-'):<6}{row['latency_ms']:>12.1f}"
        )
        if args.verbose:
            if not row.get("sections_ok", True):
                print("           └ 缺少期望章节")
            if not row.get("keywords_ok", True):
                print("           └ 报告中缺少期望关键词")
            if not row.get("findings_ok", True):
                print(f"           └ 分析结论数不足（{row.get('findings')}）")
            if row.get("dangling_citations"):
                print(f"           └ 悬空引用：{row['dangling_citations']}")
            if row.get("errors"):
                print(f"           └ 执行告警：{row['errors'][:3]}")
            if row.get("error"):
                print(f"           └ 异常：{row['error']}")

    metrics = summarize(rows)
    wall_ms = (time.perf_counter() - started) * 1000.0

    print("")
    print(LINE)
    print("评测指标")
    print(LINE)
    print(f"任务完成率       task_completion_rate       : {metrics['task_completion_rate'] * 100:.2f}%")
    print(f"检索命中率       retrieval_hit_rate         : {metrics['retrieval_hit_rate'] * 100:.2f}%")
    print(f"引用可追溯率     citation_traceability_rate : {metrics['citation_traceability_rate'] * 100:.2f}%")
    print(f"风险裁决符合率   verdict_accuracy           : {metrics['verdict_accuracy'] * 100:.2f}%")
    print(f"平均耗时         avg_latency_ms             : {metrics['avg_latency_ms']:.1f} ms")
    print(f"               （最小 {metrics['min_latency_ms']:.1f} ms / 最大 {metrics['max_latency_ms']:.1f} ms）")
    print(f"评测总墙钟耗时                              : {wall_ms:.1f} ms")
    print(f"编排引擎                                    : {pipeline.engine_name}")

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": pipeline.engine_name,
        "llm_mode": "mock" if use_mock_llm() else "openai-compatible",
        "metrics": metrics,
        "wall_clock_ms": round(wall_ms, 3),
        "cases": rows,
    }
    out_path = Path(args.json)
    # 允许 --json 指向尚不存在的子目录，避免 CI 首次运行因目录缺失而失败
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已写入                                  : {out_path}")
    print("")

    # 退出码：三项核心指标必须全部达标，便于 CI 直接卡门
    # 阈值取 0.99 而不是 1.0：mock 模式输出确定、通常可拿满分，留 1% 余量容忍边缘用例；
    # avg_latency_ms 不参与门禁（耗时依赖机器负载，用作门槛会导致 CI 假失败）
    ok = (
        metrics["task_completion_rate"] >= 0.99
        and metrics["retrieval_hit_rate"] >= 0.99
        and metrics["citation_traceability_rate"] >= 0.99
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
