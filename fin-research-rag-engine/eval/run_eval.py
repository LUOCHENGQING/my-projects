"""评测脚本：`python eval/run_eval.py`

为什么评测要这么写
------------------
RAG 项目最容易"演示时很好、上线后说不清"：换个问题答错了，没人知道是切分问题、
召回问题还是生成问题。因此评测必须**分段量化**，每一段都有独立指标：

    检索段   检索命中率（Recall@K）      标注的标准资料有没有被召回
             复杂问题召回率提升（A/B）   混合三路 vs 单一向量，同一批问题的召回差
             路由准确率                 问题类型有没有被正确识别
    生成段   引用可追溯率               答案里每个 [n] 都能回到真实证据
             答案相关度（要点覆盖率）   人工标注的必答要点答到了几个
             数字忠实度                 答案里的数字是否都能在证据中定位
    兜底段   拒答正确率                 资料里没有的问题，是否老实说"没有依据"
    服务段   FAQ 命中率                 高频问题直出的准确率
             平均耗时 / 缓存命中耗时    性能
    对照段   单一向量 / 纯关键词        作为 baseline 一起跑，避免"自己跟自己比"

`--baseline` 会额外跑对照策略；退出码非 0 表示核心指标未达标，可直接卡 CI。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.answer.faithfulness import claim_body, evaluate_faithfulness  # noqa: E402
from src.config import EVAL_DIR, use_mock_llm  # noqa: E402
from src.engine import RAGEngine  # noqa: E402
from src.utils.console import ensure_utf8_console  # noqa: E402

LINE = "=" * 86


def load_cases(path: Optional[Path] = None) -> Dict[str, Any]:
    target = path or (EVAL_DIR / "cases.json")
    return json.loads(target.read_text(encoding="utf-8"))


def hit_rate(groups: List[List[str]], got: List[str]) -> float:
    """分组召回率：每个分组内任一资料被召回即算该组命中，返回命中组数 / 组数。

    为什么用「分组」而不是「平铺列表」：同一个答案经常在多份资料里各有一份
    （说明书 / CSV 要素表 / JSON 产品库），要求全部召回是错的；
    而跨文档问题又确实需要每份都召回。分组能把这两种情况分开表达。
    """
    if not groups:
        return 1.0
    got_set = set(got)
    hits = sum(1 for group in groups if any(s in got_set for s in group))
    return hits / len(groups)


def recall_at_1(groups: List[List[str]], got: List[str]) -> float:
    """Top-1 是否落在任一标准答案分组里。"""
    if not groups:
        return 1.0
    if not got:
        return 0.0
    return 1.0 if any(got[0] in group for group in groups) else 0.0


def reciprocal_rank(groups: List[List[str]], got: List[str]) -> float:
    """第一个命中标准答案分组的倒数排名（MRR 的单条贡献）。

    这个指标比"命中率"更能反映**排序质量**：召回了一大堆但正确答案排在第 5 位，
    命中率是 100%，用户拿到的答案却排在很后面（只有 Top-K 的前几条真的进上下文）。
    """
    if not groups:
        return 1.0
    for rank, source in enumerate(got, start=1):
        if any(source in group for group in groups):
            return 1.0 / rank
    return 0.0


def evaluate_case(engine: RAGEngine, case: Dict[str, Any], with_baseline: bool = True) -> Dict[str, Any]:
    """跑一条样例并计算该样例的全部判定。"""
    question = case["question"]
    groups = [list(g) for g in (case.get("expect_source_groups") or [])]
    keyphrases = list(case.get("keyphrases") or [])

    started = time.perf_counter()
    try:
        # 评测检索与生成链路本身，因此关掉 FAQ 直出与缓存
        answer = engine.ask(question, use_cache=False, use_faq=False)
        error = ""
    except Exception as exc:  # noqa: BLE001 - 单条失败不应中断整轮评测
        return {
            "id": case["id"],
            "question": question,
            "error": f"{type(exc).__name__}: {exc}",
            "hit_rate": 0.0,
            "traceable": False,
            "coverage": 0.0,
            "faithful": False,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }

    retrieval = answer.retrieval
    got_sources = retrieval.source_ids if retrieval else []
    top1 = retrieval.evidence[0].source_id if retrieval and retrieval.evidence else ""

    row: Dict[str, Any] = {
        "id": case["id"],
        "type": case.get("type", ""),
        "question": question,
        "route": retrieval.plan.route if retrieval else "",
        "expect_route": case.get("expect_route", ""),
        "expect_source_groups": groups,
        "got_sources": got_sources,
        "hit_rate": round(hit_rate(groups, got_sources), 4),
        "recall_at_1": round(recall_at_1(groups, got_sources if got_sources else [top1]), 4),
        "mrr": round(reciprocal_rank(groups, got_sources), 4),
        "top1_source": top1,
        "traceable": answer.traceable,
        "coverage": round(answer.faithfulness.coverage, 4) if answer.faithfulness else 0.0,
        "numbers_ok": bool(answer.faithfulness and answer.faithfulness.numbers.ok),
        "faithful": bool(answer.faithfulness and answer.faithfulness.faithful),
        "support_rate": round(answer.faithfulness.support_rate, 4) if answer.faithfulness else 0.0,
        "citations": len(answer.citations),
        "dangling": len(answer.citation_check.dangling) if answer.citation_check else 0,
        "answer_chars": len(answer.answer),
        "latency_ms": round(answer.total_latency_ms, 3),
        "error": error,
    }

    if with_baseline:
        dense = engine.search(question, mode="dense")
        bm25 = engine.search(question, mode="bm25")
        row["dense_hit_rate"] = round(hit_rate(groups, dense.source_ids), 4)
        row["bm25_hit_rate"] = round(hit_rate(groups, bm25.source_ids), 4)
        row["dense_mrr"] = round(reciprocal_rank(groups, dense.source_ids), 4)
        row["dense_top1_hit"] = bool(recall_at_1(groups, dense.source_ids))
        row["top1_hit"] = bool(recall_at_1(groups, got_sources))
        row["baseline_gain"] = round(row["hit_rate"] - row["dense_hit_rate"], 4)
    return row


def evaluate_refusal(engine: RAGEngine, case: Dict[str, Any]) -> Dict[str, Any]:
    """拒答用例：资料库里没有依据时，正确行为是拒答，而不是编一段听起来专业的答案。"""
    answer = engine.ask(case["question"], use_cache=False, use_faq=False)
    return {
        "id": case["id"],
        "question": case["question"],
        "refused": answer.mode == "refused" or not answer.citations,
        "mode": answer.mode,
        "answer": answer.answer[:120],
    }


def evaluate_faq(engine: RAGEngine, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        if engine.faq is None:
            rows.append({**case, "hit": False, "score": 0.0, "correct": False})
            continue
        match = engine.faq.best(case["question"])
        hit = match is not None
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expect_hit": bool(case.get("expect_hit")),
                "hit": hit,
                "score": round(match.score, 4) if match else 0.0,
                "matched": match.entry.question if match else "",
                "correct": hit == bool(case.get("expect_hit")),
            }
        )
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows) or 1
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    hybrid = [float(r.get("hit_rate", 0.0)) for r in rows]
    dense = [float(r.get("dense_hit_rate", 0.0)) for r in rows if "dense_hit_rate" in r]
    bm25 = [float(r.get("bm25_hit_rate", 0.0)) for r in rows if "bm25_hit_rate" in r]

    hybrid_avg = sum(hybrid) / total
    dense_avg = (sum(dense) / len(dense)) if dense else 0.0
    lift = ((hybrid_avg - dense_avg) / dense_avg * 100.0) if dense_avg > 0 else 0.0

    return {
        "case_count": len(rows),
        "retrieval_hit_rate": round(hybrid_avg, 4),
        "dense_only_hit_rate": round(dense_avg, 4),
        "bm25_only_hit_rate": round((sum(bm25) / len(bm25)) if bm25 else 0.0, 4),
        "complex_question_retrieval_lift_pct": round(lift, 2),
        "mrr": round(sum(float(r.get("mrr", 0.0)) for r in rows) / total, 4),
        "dense_only_mrr": round(
            sum(float(r.get("dense_mrr", 0.0)) for r in rows if "dense_mrr" in r)
            / max(1, sum(1 for r in rows if "dense_mrr" in r)),
            4,
        ),
        "top1_accuracy": round(sum(1 for r in rows if r.get("top1_hit")) / total, 4),
        "dense_only_top1_accuracy": round(
            sum(1 for r in rows if r.get("dense_top1_hit")) / max(1, sum(1 for r in rows if "dense_top1_hit" in r)),
            4,
        ),
        "recall_at_1": round(sum(float(r.get("recall_at_1", 0.0)) for r in rows) / total, 4),
        "citation_traceability_rate": round(sum(1 for r in rows if r.get("traceable")) / total, 4),
        "answer_relevance": round(sum(float(r.get("coverage", 0.0)) for r in rows) / total, 4),
        "number_faithfulness_rate": round(sum(1 for r in rows if r.get("numbers_ok")) / total, 4),
        "claim_support_rate": round(sum(float(r.get("support_rate", 0.0)) for r in rows) / total, 4),
        "route_accuracy": round(
            sum(1 for r in rows if r.get("expect_route") and r.get("route") == r.get("expect_route"))
            / max(1, sum(1 for r in rows if r.get("expect_route"))),
            4,
        ),
        "avg_latency_ms": round(statistics.fmean(latencies), 3) if latencies else 0.0,
        "max_latency_ms": round(max(latencies), 3) if latencies else 0.0,
        "min_latency_ms": round(min(latencies), 3) if latencies else 0.0,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ensure_utf8_console()
    parser = argparse.ArgumentParser(prog="python eval/run_eval.py", description="金融智研引擎评测")
    parser.add_argument("--cases", default=None, help="用例文件（默认 eval/cases.json）")
    parser.add_argument("--json", default=str(EVAL_DIR / "last_report.json"), help="指标落盘路径")
    parser.add_argument("--no-baseline", action="store_true", help="跳过单一向量 / 纯关键词对照")
    parser.add_argument("--verbose", action="store_true", help="打印每条用例的失败明细")
    args = parser.parse_args(argv)

    payload = load_cases(Path(args.cases) if args.cases else None)
    cases = list(payload.get("cases") or [])
    refusals = list(payload.get("refusal_cases") or [])
    faq_cases = list(payload.get("faq_cases") or [])

    print("")
    print(LINE)
    print("金融智研引擎（RAG）  ·  评测")
    print(LINE)
    print(f"用例数量   : 检索生成 {len(cases)} 条 / 拒答 {len(refusals)} 条 / FAQ {len(faq_cases)} 条")
    print(f"LLM 模式   : {'mock（确定性抽取式作答，离线可复现）' if use_mock_llm() else 'OpenAI 兼容接口'}")

    started = time.perf_counter()
    engine = RAGEngine.build(quiet=True)
    stats = engine.stats()
    print(f"资料库     : {stats['corpus']['documents']} 篇文档 / {stats['corpus']['sections']} 个章节 / "
          f"{stats['corpus']['tables']} 张表格 / {stats['corpus']['faq']} 条 FAQ")
    print(f"切分       : {stats['chunks']['parents']} 个父块 / {stats['chunks']['children']} 个子块 "
          f"（表格子块 {stats['chunks']['table_children']}）")
    print(f"索引       : BM25 词表 {stats['index']['vocabulary']} 词 / "
          f"嵌入后端 {stats['index']['embedding']['backend']}(dim={stats['index']['embedding']['dim']}) / "
          f"重排 {stats['reranker']}")
    print(f"来源格式   : {', '.join(stats['corpus']['formats'])}")
    print(f"解析告警   : {len(engine.corpus.issues)} 条（示例：{engine.corpus.issues[0] if engine.corpus.issues else '无'}）")

    # ---- 检索 + 生成 ----
    rows: List[Dict[str, Any]] = []
    print("")
    print(f"{'用例':<6}{'类型':<8}{'路由':<9}{'命中率':>8}{'单一向量':>10}{'增益':>8}"
          f"{'要点':>7}{'可追溯':>8}{'耗时(ms)':>11}")
    print("-" * 86)
    for case in cases:
        row = evaluate_case(engine, case, with_baseline=not args.no_baseline)
        rows.append(row)
        gain = f"{row.get('baseline_gain', 0.0) * 100:+.1f}%" if "baseline_gain" in row else "-"
        print(
            f"{row['id']:<6}{row.get('type', ''):<8}{row.get('route', ''):<9}"
            f"{row.get('hit_rate', 0.0) * 100:>7.1f}%"
            f"{row.get('dense_hit_rate', 0.0) * 100:>9.1f}%{gain:>8}"
            f"{row.get('coverage', 0.0) * 100:>6.0f}%"
            f"{'✔' if row.get('traceable') else '✘':>8}"
            f"{row.get('latency_ms', 0.0):>11.1f}"
        )
        if args.verbose and row.get("error"):
            print(f"        └ 异常：{row['error']}")

    metrics = summarize(rows)

    # ---- 拒答 ----
    refusal_rows = [evaluate_refusal(engine, c) for c in refusals]
    refusal_ok = sum(1 for r in refusal_rows if r["refused"])
    refusal_rate = (refusal_ok / len(refusal_rows)) if refusal_rows else 1.0

    # ---- FAQ ----
    faq_rows = evaluate_faq(engine, faq_cases)
    faq_correct = sum(1 for r in faq_rows if r["correct"])
    faq_hit_rate = (faq_correct / len(faq_rows)) if faq_rows else 0.0

    # ---- 缓存延迟 ----
    cache_probe = engine.ask(cases[0]["question"]) if cases else None
    warm = engine.ask(cases[0]["question"]) if cases else None
    cache_latency = warm.timings.get("total_ms", 0.0) if warm else 0.0
    cache_hit = bool(warm and warm.cache_hit)

    wall_ms = (time.perf_counter() - started) * 1000.0

    print("")
    print(LINE)
    print("核心指标")
    print(LINE)
    print(f"检索命中率       retrieval_hit_rate        : {metrics['retrieval_hit_rate'] * 100:.2f}%"
          f"　（混合三路）")
    print(f"对照·单一向量    dense_only_hit_rate       : {metrics['dense_only_hit_rate'] * 100:.2f}%")
    print(f"对照·纯关键词    bm25_only_hit_rate        : {metrics['bm25_only_hit_rate'] * 100:.2f}%")
    print(f"复杂问题召回率提升 complex_retrieval_lift    : {metrics['complex_question_retrieval_lift_pct']:+.2f}%"
          f"　（混合三路+重排 vs 单一向量不重排）")
    print(f"首位命中率       top1_accuracy             : {metrics['top1_accuracy'] * 100:.2f}%"
          f"　（对照·单一向量 {metrics['dense_only_top1_accuracy'] * 100:.2f}%）")
    print(f"MRR              mrr                       : {metrics['mrr']:.4f}"
          f"　（对照·单一向量 {metrics['dense_only_mrr']:.4f}）")
    print(f"引用可追溯率     citation_traceability_rate: {metrics['citation_traceability_rate'] * 100:.2f}%")
    print(f"答案相关度       answer_relevance          : {metrics['answer_relevance'] * 100:.2f}%"
          f"　（人工标注要点的覆盖率）")
    print(f"数字忠实度       number_faithfulness_rate  : {metrics['number_faithfulness_rate'] * 100:.2f}%"
          f"　（答案中的数字均可在证据中定位）")
    print(f"句子支撑率       claim_support_rate        : {metrics['claim_support_rate'] * 100:.2f}%")
    print(f"路由命中率       route_accuracy            : {metrics['route_accuracy'] * 100:.2f}%")
    print(f"拒答正确率       refusal_accuracy          : {refusal_rate * 100:.2f}%")
    print(f"FAQ 命中准确率   faq_accuracy              : {faq_hit_rate * 100:.2f}%")
    print(f"平均耗时         avg_latency_ms            : {metrics['avg_latency_ms']:.1f} ms"
          f"　（最小 {metrics['min_latency_ms']:.1f} / 最大 {metrics['max_latency_ms']:.1f}）")
    print(f"缓存命中耗时     cache_hit_latency_ms      : {cache_latency:.1f} ms"
          f"　（{'命中' if cache_hit else '未命中'}）")
    print(f"评测总墙钟耗时                             : {wall_ms:.1f} ms")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "llm_mode": "mock" if use_mock_llm() else "openai-compatible",
        "engine_stats": stats,
        "metrics": {
            **metrics,
            "refusal_accuracy": round(refusal_rate, 4),
            "faq_accuracy": round(faq_hit_rate, 4),
            "cache_hit_latency_ms": round(cache_latency, 3),
            "cache_hit": cache_hit,
            "wall_clock_ms": round(wall_ms, 3),
        },
        "cases": rows,
        "refusal_cases": refusal_rows,
        "faq_cases": faq_rows,
    }
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已写入                                 : {out_path}")
    print("")

    ok = (
        metrics["retrieval_hit_rate"] >= 0.9
        and metrics["citation_traceability_rate"] >= 0.99
        and metrics["number_faithfulness_rate"] >= 0.99
        and metrics["answer_relevance"] >= 0.85
        and refusal_rate >= 0.99
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
