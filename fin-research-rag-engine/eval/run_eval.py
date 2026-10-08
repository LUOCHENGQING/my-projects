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

11 项核心指标：定义与计算口径
=============================
「11 项」指 `main()` 末尾打印的 11 行核心指标（对应下面 1~11）。字段名就是落盘
report 里 `metrics.<字段名>` 的键名；**阈值只写在 `main()` 的门禁里**，本节不重复也不发明。

 1. 检索命中率        retrieval_hit_rate
      定义：标注的标准资料是否被召回，按「分组」计分。
      口径：每组内任一 source_id 命中即该组命中；得分 = 命中组数 / 组数。
            `expect_source_groups` 为空时按 1.0 处理（没有期望就没有失分）。
            单条用例取值见 `evaluate_case` 的 `hit_rate`，汇总为各用例均值。
 2. 对照·单一向量     dense_only_hit_rate
      定义：同一批用例改由单一稠密向量检索（不重排）时的命中率，作为 §对照 的 baseline。
      口径：每条用例额外跑一次 dense 检索并算同样的分组命中率，再取均值。
 3. 对照·纯关键词     bm25_only_hit_rate
      定义：改用纯 BM25 关键词检索（不重排）时的命中率。
      口径：与上一条同法；`--no-baseline` 时不产生该字段，汇总时按缺少对照处理。
 4. 复杂问题召回率提升 complex_question_retrieval_lift_pct
      定义：(混合三路+重排 的命中率 − 单一向量不重排 的命中率) / 单一向量命中率 × 100。
      口径：两个均值都取自同一批用例；分母（单一向量均值）为 0 时该值记 0.0，
            避免除零（也意味着"没有对照就没有提升可说"）。
 5. 首位命中率        top1_accuracy（对照：dense_only_top1_accuracy）
      定义：Top-1 是否落在任一标准答案分组里；汇总为命中用例数 / 用例总数。
      口径：混合侧看 `hit_rate` 用的同一份来源列表，单一向量侧单独跑一次 dense 后判定。
            另有逐条的 `recall_at_1`（0/1）与其均值，但不单独打印。
 6. MRR               mrr（对照：dense_only_mrr）
      定义：第一个命中标准答案分组的**倒数排名**（1/rank）；未命中记 0。
      口径：混合侧对所有用例取均值；单一向量侧只对**确实产生了 dense_mrr 字段**的
            用例取均值（分母是这些用例数，不是总用例数）。
 7. 引用可追溯率      citation_traceability_rate
      定义：答案能否被复核——引用编号没有悬空、且确实用到了引用。
      口径：取自 `AnswerResult.traceable`（FAQ 直出时等价于"答案非空"），
            汇总为可追溯用例数 / 用例总数。
 8. 答案相关度        answer_relevance
      定义：答案与问题的贴合程度。
      口径：实际取 `answer.faithfulness.coverage` 的均值。**注：实际实现为**——
            引擎侧 `evaluate_faithfulness` 未传 `keyphrases`，而 `keyphrase_coverage`
            在无要点时返回 1.0，因此该值当前恒为 1.0，是上限而非实测贴合度；
            要真正度量本指标需把用例的 `keyphrases` 接进生成 / 校验链路。
 9. 数字忠实度        number_faithfulness_rate
      定义：答案里的数字是否都能在证据中定位（本项目最硬的一条反幻觉规则）。
      口径：取 `answer.faithfulness.numbers.ok`，汇总为通过用例数 / 用例总数。
            校验本身由 `NumberCheck` 完成：1~2 位编号与四位年份豁免，其余数字必须命中证据。
10. 句子支撑率        claim_support_rate
      定义：答案的每句话是否都能在证据里找到语义支撑。
      口径：取 `answer.faithfulness.support_rate` 的均值（支撑率 ≥ 0.8 才算 faithful）。
11. 路由命中率        route_accuracy
      定义：问题类型是否被正确识别。
      口径：只对**标注了 `expect_route`** 的用例计算（分母是这些用例数）；
            未标注期望路由的用例整体不参与，避免用"没期望"充数。

另有与上述 11 项同源、同样打印在「核心指标」区块、但门槛各不相同的两项：
    refusal_accuracy   拒答正确率：资料库里没有依据时是否老实拒答（`mode == "refused"`
                       或没有任何引用都算拒答），分母是拒答用例数，无用例时记 1.0。
    faq_accuracy       FAQ 命中准确率：高频问题直出的判定是否与 `expect_hit` 一致，
                       分母是 FAQ 用例数，无用例时记 0.0。
落盘 report 的 `metrics` 里还包含 avg / max / min_latency_ms、cache_hit_latency_ms、
cache_hit、wall_clock_ms 等服务段指标——它们用于观察性能，不参与退出码门禁。

三套检索策略对照：分别对照什么
=============================
本脚本同时跑三套策略，**同一问题、同一资料库，只换检索策略**，因此差异只能归因于策略本身：

    hybrid      混合三路（BM25 倒排 + 稠密向量 + 稀疏词权重）→ RRF 融合 → 去重 → 重排
                → 父块回溯；即 `engine.ask()` / `engine.search(mode="hybrid")` 的主链路。
                **注：实际实现为**——本脚本只经 `engine.ask()` 间接使用混合链路，
                不会单独再跑一次 hybrid（逐条明细里的 `hit_rate` 就是它）。
    dense_only  单一向量（纯稠密），走 `engine.search(mode="dense")`：只这一路，
                **且不做重排**。这是"混合替代单一向量"真正的 baseline。
    bm25_only   纯关键词，走 `engine.search(mode="bm25")`：同样只一路、不重排。

为什么要摘掉重排：如果给 baseline 也加上重排，比的就只是"有没有重排"，
而不是"有没有混合召回"，结论会指向错误的优化方向。

对照产出的字段（逐条用例）：
    dense_hit_rate / bm25_hit_rate   对照策略的分组命中率
    dense_mrr                        对照策略的 MRR
    dense_top1_hit / top1_hit        两侧的 Top-1 是否命中（用于 top1_accuracy）
    baseline_gain = hit_rate − dense_hit_rate   单条用例的召回增益
汇总侧的 `complex_question_retrieval_lift_pct` 就是这些增益在整批用例上的版本。

退出码门禁：不达标以非 0 退出卡 CI
=================================
`main()` 末尾把 5 项核心指标与硬阈值比对，**全部达标返回 0，任一项不达标返回 1**；
`if __name__ == "__main__"` 直接 `sys.exit(main())`，因此 CI 里一条
`python eval/run_eval.py` 就能把回归挡在合并之前。门禁项与阈值（与代码一致）：

    检索命中率       metrics["retrieval_hit_rate"]          >= 0.9
    引用可追溯率     metrics["citation_traceability_rate"]  >= 0.99
    数字忠实度       metrics["number_faithfulness_rate"]    >= 0.99
    答案相关度       metrics["answer_relevance"]            >= 0.85
    拒答正确率       refusal_rate                           >= 0.99

`--verbose` 打印每条用例的异常明细（不参与判定）；`--no-baseline` 跳过对照策略，
此时对照类指标按缺少数据处理（`baseline_gain` 等字段不产生）。

报告落盘格式（--json，默认 eval/last_report.json）
================================================
    JSON，UTF-8，`ensure_ascii=False`，`indent=2`（便于人工 diff 与 code review），
    父目录不存在时自动创建（`mkdir(parents=True, exist_ok=True)`）。顶层结构：

        generated_at    生成时刻，"%Y-%m-%d %H:%M:%S"
        llm_mode        "mock" | "openai-compatible"
        engine_stats    `engine.stats()` 原样落盘（资料库 / 切分 / 索引 / 缓存 / FAQ 规模）
        metrics         `summarize()` 的全部指标 + refusal_accuracy / faq_accuracy /
                        cache_hit_latency_ms / cache_hit / wall_clock_ms
        cases           检索生成用例逐条明细（含 got_sources / 对照字段 / 错误信息）
        refusal_cases   拒答用例逐条明细（refused / mode / 答案前 120 字）
        faq_cases       FAQ 用例逐条明细（expect_hit / hit / score / correct）

逐条明细是刻意保留的：只看一个汇总数字无法定位"是哪条用例退化"，
而门禁失败时最需要的就是直接看到那一条。
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

# 报表分隔线宽度固定为 86 列：与下面各行的中文对齐排版共用同一个宽度常量
LINE = "=" * 86


def load_cases(path: Optional[Path] = None) -> Dict[str, Any]:
    """读评测用例 JSON；未指定路径时用 eval/cases.json。"""
    # 显式指定 path 与留空两种调用方式都保留：前者供 test_eval.py 与临时用例集使用
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
    # 用集合判包含：来源列表可能很长，逐组线性扫描会退化成 O(组数 × 召回数)
    got_set = set(got)
    # any(...) 即"组内任一命中"：同一份答案出现在多份资料里时只算一次命中
    hits = sum(1 for group in groups if any(s in got_set for s in group))
    return hits / len(groups)


def recall_at_1(groups: List[List[str]], got: List[str]) -> float:
    """Top-1 是否落在任一标准答案分组里。

    只认 `got[0]`：因此调用方必须保证传入的列表是**已经排好序**的来源列表，
    否则本指标量的是列表顺序而不是检索排序。
    """
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
    # rank 从 1 开始：1/1 = 1.0 表示排第一，1/2 = 0.5 表示第二，未命中为 0
    for rank, source in enumerate(got, start=1):
        if any(source in group for group in groups):
            return 1.0 / rank
    return 0.0


def evaluate_case(engine: RAGEngine, case: Dict[str, Any], with_baseline: bool = True) -> Dict[str, Any]:
    """跑一条样例并计算该样例的全部判定。

    返回一行字典：命中率 / recall_at_1 / MRR、可追溯性、忠实度三项（coverage /
    numbers_ok / support_rate）、引用与悬空计数、字数、耗时、错误信息；
    `with_baseline` 为真时再补上对照策略字段（dense_hit_rate / bm25_hit_rate /
    dense_mrr / dense_top1_hit / top1_hit / baseline_gain）。

    注：实际实现为——`case["keyphrases"]` 虽被读出，却只赋给局部变量 `keyphrases`
    并在此函数内**未被使用**（引擎侧 `evaluate_faithfulness` 也没有收到要点），
    因此逐条的 `coverage` 恒为 1.0；详见模块 docstring 第 8 项指标的口径说明。
    """
    question = case["question"]
    groups = [list(g) for g in (case.get("expect_source_groups") or [])]
    keyphrases = list(case.get("keyphrases") or [])

    started = time.perf_counter()
    try:
        # 评测检索与生成链路本身，因此关掉 FAQ 直出与缓存
        answer = engine.ask(question, use_cache=False, use_faq=False)
        error = ""
    except Exception as exc:  # noqa: BLE001 - 单条失败不应中断整轮评测
        # 单条用例炸掉只记 0 分并把异常写进这个**精简行**（没有参照行那些对照与忠实度字段）：
        # 整轮评测必须继续跑完，否则一个坏 case 会让后面所有用例的指标凭空消失
        # （注：该分支的行不含 dense_* 字段，因此它按 0 计入检索类分母、但不计入对照类均值）
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
        # source_ids 为空时退化成用 Top-1 单条判定，免得"有证据但 source_ids 未回填"直接记 0
        "recall_at_1": round(recall_at_1(groups, got_sources if got_sources else [top1]), 4),
        "mrr": round(reciprocal_rank(groups, got_sources), 4),
        "top1_source": top1,
        "traceable": answer.traceable,
        # coverage / numbers_ok / support_rate / faithful 都来自忠实度报告；报告缺失时保守记 0
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
        # 对照必须自己单独跑：同一问题、同一资料库，只换策略，差异才归因于策略
        dense = engine.search(question, mode="dense")
        bm25 = engine.search(question, mode="bm25")
        # 对照侧的命中率/MRR 都用同一套分组口径，避免"口径不同造成的假差距"
        row["dense_hit_rate"] = round(hit_rate(groups, dense.source_ids), 4)
        row["bm25_hit_rate"] = round(hit_rate(groups, bm25.source_ids), 4)
        row["dense_mrr"] = round(reciprocal_rank(groups, dense.source_ids), 4)
        row["dense_top1_hit"] = bool(recall_at_1(groups, dense.source_ids))
        row["top1_hit"] = bool(recall_at_1(groups, got_sources))
        # 单条增益可能为负：混合比不上单一向量时也要如实记下来，而不是截断到 0
        row["baseline_gain"] = round(row["hit_rate"] - row["dense_hit_rate"], 4)
    return row


def evaluate_refusal(engine: RAGEngine, case: Dict[str, Any]) -> Dict[str, Any]:
    """拒答用例：资料库里没有依据时，正确行为是拒答，而不是编一段听起来专业的答案。

    判定口径：`mode == "refused"`（主体闸门或生成阶段判定无依据）**或**没有任何引用
    （没有引用等于答案无法被复核，同样不该被当成有效作答）。答案只截前 120 字入库。
    """
    answer = engine.ask(case["question"], use_cache=False, use_faq=False)
    return {
        "id": case["id"],
        "question": case["question"],
        "refused": answer.mode == "refused" or not answer.citations,
        "mode": answer.mode,
        "answer": answer.answer[:120],
    }


def evaluate_faq(engine: RAGEngine, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """FAQ 用例逐条判定：`engine.faq.best()` 命中与否是否与用例标注的 `expect_hit` 一致。

    判定用 `best()`（**已套阈值**）而不是 `search()`：阈值才是"该不该直出"的真实闸门，
    用未套阈值的候选列表会把"分数不够却硬直出"这种错误判成正确。
    引擎没建 FAQ 索引（`engine.faq is None`）时全部记未命中，不抛错。
    """
    rows: List[Dict[str, Any]] = []
    for case in cases:
        if engine.faq is None:
            # 索引缺失时保守记"没命中 + 不正确"，让它在 faq_accuracy 上如实失分
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
                # 命中与否都必须与期望一致：既罚漏答（该直出却没直出），也罚抢答（不该直出却直出）
                "correct": hit == bool(case.get("expect_hit")),
            }
        )
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把逐条明细汇总成整批指标（本函数的字段名就是 report.metrics 的键名）。

    三个容易踩坑的口径（与实现一致）：
        1. 分母：检索 / 生成类指标一律除以 `total = len(rows)`——**固定用用例总数**，
           所以缺失字段的用例按 0 计入，不会被"自动排除"从而虚高；
        2. 对照类（dense / bm25）：只对**确实带了对照字段**的用例取均值，
           `--no-baseline` 时列表为空 → 记 0.0，而不是记成与混合相同（否则提升恒为 0）；
        3. 提升百分比：`lift = (hybrid_avg − dense_avg) / dense_avg × 100`，
           `dense_avg <= 0` 时记 0.0（既避免除零，也避免"0 基线"造出无穷大提升）。

    覆盖率相关的第 8、9、10 项（answer_relevance / number_faithfulness_rate /
    claim_support_rate）分别取 coverage / numbers_ok / support_rate，定义见模块 docstring。
    返回全为纯 Python 数值（已 round），可直接 JSON 序列化。
    """
    total = len(rows) or 1
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    hybrid = [float(r.get("hit_rate", 0.0)) for r in rows]
    dense = [float(r.get("dense_hit_rate", 0.0)) for r in rows if "dense_hit_rate" in r]
    bm25 = [float(r.get("bm25_hit_rate", 0.0)) for r in rows if "bm25_hit_rate" in r]

    hybrid_avg = sum(hybrid) / total
    # 对照侧用 len(dense) 而不是 total：没跑 baseline 的用例不该被当成"对照得 0 分"
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
            # 分母只算"标注了 expect_route"的用例：没标注期望的用例不该被当成路由猜错
            sum(1 for r in rows if r.get("expect_route") and r.get("route") == r.get("expect_route"))
            / max(1, sum(1 for r in rows if r.get("expect_route"))),
            4,
        ),
        "avg_latency_ms": round(statistics.fmean(latencies), 3) if latencies else 0.0,
        "max_latency_ms": round(max(latencies), 3) if latencies else 0.0,
        "min_latency_ms": round(min(latencies), 3) if latencies else 0.0,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """跑完整轮评测：建引擎 → 检索生成 / 拒答 / FAQ / 缓存 → 打印 → 落盘 → 返回退出码。

    参数：`argv` 为 None 时用 `sys.argv`（命令行用法）；显式传列表便于测试直接调用。
    返回：0 表示 5 项门禁指标全部达标，1 表示任一项未达标（详见模块 docstring 的门禁表）。
    副作用：打印两份对齐的报表到 stdout；把 report 写进 `--json` 指定的路径（父目录自动创建）。
    """
    # 控制台先切 UTF-8：否则 Windows 默认编码会把中文指标名打成乱码
    ensure_utf8_console()
    parser = argparse.ArgumentParser(prog="python eval/run_eval.py", description="金融智研引擎评测")
    parser.add_argument("--cases", default=None, help="用例文件（默认 eval/cases.json）")
    parser.add_argument("--json", default=str(EVAL_DIR / "last_report.json"), help="指标落盘路径")
    parser.add_argument("--no-baseline", action="store_true", help="跳过单一向量 / 纯关键词对照")
    parser.add_argument("--verbose", action="store_true", help="打印每条用例的失败明细")
    args = parser.parse_args(argv)

    payload = load_cases(Path(args.cases) if args.cases else None)
    # 三类用例分开取：检索生成 / 拒答 / FAQ 的判定口径完全不同，混在一批里没法算
    cases = list(payload.get("cases") or [])
    refusals = list(payload.get("refusal_cases") or [])
    faq_cases = list(payload.get("faq_cases") or [])

    print("")
    print(LINE)
    print("金融智研引擎（RAG）  ·  评测")
    print(LINE)
    print(f"用例数量   : 检索生成 {len(cases)} 条 / 拒答 {len(refusals)} 条 / FAQ {len(faq_cases)} 条")
    print(f"LLM 模式   : {'mock（确定性抽取式作答，离线可复现）' if use_mock_llm() else 'OpenAI 兼容接口'}")

    # 墙钟从建引擎之前开始计：建索引本身就是评测成本的一部分，不能漏算
    started = time.perf_counter()
    engine = RAGEngine.build(quiet=True)
    # 引擎自述一并打印并落盘：指标不变但语料/后端变了时，必须能从报告里看出环境差异
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
        # 逐条评测：单条异常在 evaluate_case 内部被兜住记成 0 分，整轮不中断
        row = evaluate_case(engine, case, with_baseline=not args.no_baseline)
        rows.append(row)
        # 没跑对照时该列显示 "-"，而不是显示 0.0% 让人误以为"对照全军覆没"
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
    # 分母是拒答用例数而非总用例数；一个拒答用例都没有时记 1.0（没有反向用例就不该失分）
    refusal_rows = [evaluate_refusal(engine, c) for c in refusals]
    refusal_ok = sum(1 for r in refusal_rows if r["refused"])
    refusal_rate = (refusal_ok / len(refusal_rows)) if refusal_rows else 1.0

    # ---- FAQ ----
    # 与拒答相反：一个 FAQ 用例都没有时记 0.0（不能因为"没测"就白拿满分）
    faq_rows = evaluate_faq(engine, faq_cases)
    faq_correct = sum(1 for r in faq_rows if r["correct"])
    faq_hit_rate = (faq_correct / len(faq_rows)) if faq_rows else 0.0

    # ---- 缓存延迟 ----
    # 两次同问：第一次填充缓存（此前的用例都显式 use_cache=False，因此这里是冷启动），
    # 第二次命中的 total_ms 才是「缓存命中耗时」，与门禁无关，只用于观察性能
    cache_probe = engine.ask(cases[0]["question"]) if cases else None
    warm = engine.ask(cases[0]["question"]) if cases else None
    cache_latency = warm.timings.get("total_ms", 0.0) if warm else 0.0
    cache_hit = bool(warm and warm.cache_hit)

    # 总墙钟：与 started 同一基准，因此包含建索引 + 全部用例 + 缓存探测
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
        # 引擎自述整段落盘：将来的指标对比必须能确认"两次跑的是不是同一个资料库/同一套后端"
        "engine_stats": stats,
        "metrics": {
            **metrics,
            "refusal_accuracy": round(refusal_rate, 4),
            "faq_accuracy": round(faq_hit_rate, 4),
            "cache_hit_latency_ms": round(cache_latency, 3),
            "cache_hit": cache_hit,
            "wall_clock_ms": round(wall_ms, 3),
        },
        # 逐条明细一并落盘：门禁失败时最需要的是直接看到"是哪一条退了"
        "cases": rows,
        "refusal_cases": refusal_rows,
        "faq_cases": faq_rows,
    }
    out_path = Path(args.json)
    # 父目录可能不存在（如自定义的深路径），先建目录再写，避免评测跑完却倒在落盘这一步
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False + indent=2：中文可读、可 diff，便于 code review 时看清指标变化
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已写入                                 : {out_path}")
    print("")

    # 退出码门禁：5 项核心指标全部达标才放行；阈值与模块 docstring 的门禁表一致
    ok = (
        metrics["retrieval_hit_rate"] >= 0.9
        and metrics["citation_traceability_rate"] >= 0.99
        and metrics["number_faithfulness_rate"] >= 0.99
        and metrics["answer_relevance"] >= 0.85
        and refusal_rate >= 0.99
    )
    # 非 0 即视为"有回归"：CI 里靠这个退出码把不达标的构建挡下来
    return 0 if ok else 1


if __name__ == "__main__":
    # 用 sys.exit 把 main 的返回值直接变成进程退出码，CI 才能据此判定成败
    sys.exit(main())
