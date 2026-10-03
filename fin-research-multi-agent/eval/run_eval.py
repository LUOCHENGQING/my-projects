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
    target = path or (EVAL_DIR / "cases.json")
    payload = json.loads(target.read_text(encoding="utf-8"))
    return list(payload.get("cases") or [])


def citation_traceability(report: str, citations: List[Dict[str, Any]]) -> Tuple[bool, List[int]]:
    """检查报告中出现的引用编号是否全部可回溯。"""
    used = sorted({int(m) for m in re.findall(r"\[(\d+)\]", report or "")})
    known = {int(c.get("citation_no", 0)) for c in citations}
    dangling = [n for n in used if n not in known]
    return (bool(used) and not dangling, dangling)


def evaluate_case(pipeline: ResearchPipeline, case: Dict[str, Any]) -> Dict[str, Any]:
    """跑一条样例并计算该样例的各项判定。"""
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
    sections_ok = all(f"、{s}" in report or s in report for s in case.get("expect_sections", []))
    keywords_ok = all(k in report for k in case.get("expect_keywords", []))
    findings_ok = len(findings) >= int(case.get("min_findings", 0))
    completed = bool(report) and sections_ok and keywords_ok and findings_ok

    # ---- 检索命中率判定 ----
    retrieved_sources = {str(e.get("source_id")) for e in evidence}
    expect_sources = [str(s) for s in case.get("expect_sources", [])]
    hits = [s for s in expect_sources if s in retrieved_sources]
    hit_rate = len(hits) / len(expect_sources) if expect_sources else 1.0

    # ---- 引用可追溯率判定 ----
    traceable, dangling = citation_traceability(report, citations)

    # ---- 其他观测项 ----
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
    pipeline = ResearchPipeline(auto=True, engine=args.engine, quiet=True)
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已写入                                  : {out_path}")
    print("")

    # 退出码：三项核心指标必须全部达标，便于 CI 直接卡门
    ok = (
        metrics["task_completion_rate"] >= 0.99
        and metrics["retrieval_hit_rate"] >= 0.99
        and metrics["citation_traceability_rate"] >= 0.99
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
