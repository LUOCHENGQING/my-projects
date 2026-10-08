"""命令行演示：`python -m src.demo`

零依赖跑通整条链路 —— 不配 API Key、不装 Milvus / Redis / BGE-M3 / OCR，也能看到
「解析 → 切分 → 三路召回 → 重排 → 引用溯源 → 忠实度校验」的完整效果。

常用用法：

    python -m src.demo                                  # 跑内置示例问题
    python -m src.demo --question "合格投资者的门槛是多少？"   # 指定问题
    python -m src.demo --compare                        # 检索策略 A/B 对比
    python -m src.demo --stats                          # 只看引擎体检
    python -m src.demo --mode dense                     # 用单一向量检索做对照
    python -m src.demo --expr 'year >= 2024'            # 带元数据过滤
    python -m src.demo --json out.json                  # 结果落盘

在 RAG 全链路中的位置
--------------------
本模块是**链路最下游的只读消费者**：不解析、不切分、不检索、不生成，只做三件事——
    1) `RAGEngine.build()` 触发一次完整建索引（解析 → 切分 → 索引 → FAQ）；
    2) 调 `engine.stats()` / `engine.search()` / `engine.ask()` 拿现成结果；
    3) 把结果排版到终端（可选 `--json` 落盘成完整 payload）。
因此它同时是「人工验证入口」和「最小可运行样例」：链路上任何改动都应先在这里肉眼过一遍。

对外关键函数
------------
    print_header(engine)        打印引擎体检表（资料库 / 切分 / 索引 / 重排 / LLM / 缓存）
    print_answer(engine, result) 打印一次问答：路由与权重、召回漏斗、答案、证据、忠实度、耗时拆解
    run_compare(engine, q, k)   同一问题跑 dense / bm25 / hybrid 三种策略并对比
    main(argv)                  命令行入口（解析参数 → 建引擎 → 分发到上面三者）

输入 / 输出 / 调用方
--------------------
    输入：命令行参数（问题、top_k、mode、route、expr、各类开关）+ `data/` 资料目录；
    输出：stdout 体检表 / 问答排版 / 策略对比表；`--json <路径>` 时另写一个 JSON 文件，
          结构为 {stats, results[], compare?}（results 是 `AnswerResult.to_dict()` 列表）；
    被谁调用：人工执行（README 5.2 快速开始）、评审演示；`tests/` 不依赖它。
    退出码：恒为 0（`--stats` 分支也返回 0）；异常不兜底，按 Python 默认栈抛出。

副作用与异常
------------
    建索引（读 `data/`）、打印到 stdout（非 TTY 时由 `ensure_utf8_console()` 切 UTF-8）、
    `--json` 时创建父目录并覆盖写文件；`engine.ask()` 内部失败都会转成 notes / 拒答，
    不会让本模块崩掉。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import FINAL_TOP_K
from .engine import RAGEngine
from .utils.console import ensure_utf8_console

LINE = "=" * 84
THIN = "-" * 84

# 内置示例问题：刻意覆盖多种问题类型路由，用来一眼看出「按类型配权重」确实在起作用——
# 案例类（处罚案例）、指标类（合格投资者标准、应收账款增速）、
# 条款/要素类（C2 客户可买什么、在售产品里哪些是合格投资者专属）。
SAMPLE_QUESTIONS: List[str] = [
    "有没有向低风险承受能力客户销售高风险产品被处罚的案例？",
    "2023 年的合格投资者金融资产标准是多少？",
    "C2 客户可以购买哪些风险等级的产品？",
    "示例集团 2023 年应收账款增速为什么值得关注？",
    "在售产品里哪些是合格投资者专属的？",
]
# FAQ 直出示例（高频问题不走完整链路）：开户需要准备哪些材料？ / 双录资料需要保存多久？


def print_header(engine: RAGEngine) -> None:
    """打印引擎体检横幅：资料库规模与来源格式、切分规模、索引与嵌入后端、重排 / LLM / 缓存后端、
    主体清单规模，以及**解析告警（最多 4 条）**。

    参数：engine 已构造好的引擎（本函数会现场调 `engine.stats()`，只读不建索引）。
    返回：None。副作用：向 stdout 打印；不做任何格式之外的判断（数据缺失会 KeyError）。
    注：实际实现为 `engine.stats()["chunks"]["avg_children_per_parent"]` 等字段直接下标取值，
        因此 stats 结构改动会在这里第一时间暴露——这正是演示脚本的价值。
    """
    stats = engine.stats()
    print("")
    print(LINE)
    print("金融智研引擎（FinRAG-Engine）· 演示")
    print(LINE)
    print(f"资料库     : {stats['corpus']['documents']} 篇文档 / {stats['corpus']['sections']} 个章节 / "
          f"{stats['corpus']['tables']} 张表格 / {stats['corpus']['faq']} 条 FAQ")
    print(f"来源格式   : {', '.join(stats['corpus']['formats'])}")
    print(f"切分       : {stats['chunks']['parents']} 个父块 / {stats['chunks']['children']} 个子块"
          f"（表格子块 {stats['chunks']['table_children']}，平均每父块 "
          f"{stats['chunks']['avg_children_per_parent']} 个子块）")
    print(f"索引       : BM25 词表 {stats['index']['vocabulary']} 词 / "
          f"嵌入 {stats['index']['embedding']['backend']}(dim={stats['index']['embedding']['dim']})")
    print(f"重排       : {stats['reranker']}")
    print(f"LLM        : {stats['llm']['mode']}（{stats['llm']['model']}）")
    print(f"缓存       : {stats['cache']['backend']}")
    print(f"主体清单   : {len(stats['known_entities'])} 个")
    print(f"解析告警   : {len(engine.corpus.issues)} 条")
    for issue in engine.corpus.issues[:4]:
        print(f"             · {issue}")


def print_answer(engine: RAGEngine, result, show_evidence: bool = True) -> None:
    """打印一次问答的全部可解释信息。

    参数：engine 引擎实例（本函数未使用，保留是为了让三个打印函数签名统一、
          后续若要打印引擎侧上下文时无需改调用点）；
          result 任意具备 `AnswerResult` 结构的结果对象（question / retrieval / mode /
          cache_hit / faq_hit / traceable / answer / evidence / faithfulness / timings）；
          show_evidence 是否打印证据明细（`--no-evidence` 时为 False）。
    返回：None。副作用：向 stdout 分块打印——问题 → 检索计划（路由 / 权重 / 查询变体 /
    软硬过滤 / 召回漏斗）→ 模式与命中情况 → 答案正文 → 证据（编号、分数、三路命中、出处、
    前 110 字）→ 忠实度（数字 / 支撑率 / 相关度，含未支撑数字告警）→ 耗时拆解。
    异常：无（各段都以 `None` / 空值判断后跳过，因此缓存命中或 FAQ 直出这类
          没有 retrieval / faithfulness 的结果也能安全打印）。
    """
    print("")
    print(THIN)
    print(f"问题：{result.question}")
    if result.retrieval is not None:
        plan = result.retrieval.plan
        print(f"路由：{plan.route}　权重：{plan.weights}")
        print(f"查询变体：{plan.queries}")
        if plan.soft_expr:
            print(f"软过滤：{plan.soft_expr}（只影响排序）")
        if plan.filter_expr:
            print(f"硬过滤：{plan.filter_expr}")
        stats = result.retrieval.stats()
        print(f"召回：{stats['recalled']} 条候选 → 去重后 {stats['recalled'] - stats['deduplicated']} 条 → "
              f"重排取前 {stats['evidence']} 条　（三路命中分布 {stats['routes']}）")
    print(f"模式：{result.mode}　缓存命中：{result.cache_hit}　FAQ 命中：{result.faq_hit}　"
          f"引用可追溯：{result.traceable}")
    print(THIN)
    print(result.answer.strip())

    if show_evidence and result.evidence:
        print("")
        print("【本次交给模型的证据】（子块精确命中 + 三路命中情况 + 重排分）")
        for ev in result.evidence:
            routes = "+".join(ev.routes_hit) if ev.routes_hit else "-"
            print(f"  {ev.evidence_id} score={ev.score:.3f} cross={ev.cross_score:.3f} "
                  f"routes=[{routes}] kind={ev.kind}")
            print(f"     {ev.citation_label}")
            print(f"     {ev.text[:110].replace(chr(10), ' ')}…")

    if result.faithfulness is not None:
        f = result.faithfulness
        print("")
        print(f"【忠实度校验】数字忠实度={f.numbers.ok}　句子支撑率={f.support_rate:.2%}　"
              f"答案相关度={f.relevance:.4f}")
        if f.numbers.unsupported:
            print(f"  ⚠ 未被证据支撑的数字：{f.numbers.unsupported}")
    if result.timings:
        parts = "　".join(f"{k}={v:.1f}ms" for k, v in result.timings.items())
        print(f"【耗时拆解】{parts}")


def run_compare(engine: RAGEngine, question: str, top_k: int) -> Dict[str, Any]:
    """对同一个问题分别跑「单一向量 / 纯关键词 / 混合三路+重排」。

    参数：engine 引擎实例；question 待对比的问题；top_k 每种策略取回的条数。
    返回：dict——{question, dense_only:{sources, elapsed_ms}, bm25_only:{...}, hybrid:{...}}，
          供 `--json` 落盘；`sources` 是 `RetrievalResult.source_ids`（去重后的资料编号）。
    副作用：向 stdout 打印三种策略的耗时、来源与逐条证据（前 96 字），供现场对比。
    异常：无（`engine.search()` 无命中时只是列表为空）。
    注：dense / bm25 两种模式在 `RetrievalPipeline.run()` 里**刻意不加重排**，
        否则对比的就变成「有没有重排」而不是「有没有混合召回」。
    """
    dense = engine.search(question, top_k=top_k, mode="dense")
    bm25 = engine.search(question, top_k=top_k, mode="bm25")
    hybrid = engine.search(question, top_k=top_k, mode="hybrid")

    print("")
    print(LINE)
    print(f"检索策略对比：{question}")
    print(LINE)
    for name, result in (("单一向量（不重排）", dense), ("纯关键词 BM25（不重排）", bm25), ("混合三路 + 重排", hybrid)):
        print(f"\n【{name}】耗时 {result.elapsed_ms:.1f} ms　来源：{result.source_ids}")
        for rank, ev in enumerate(result.evidence, start=1):
            print(f"  {rank}. [{ev.source_id}] {ev.section_title}")
            print(f"     {ev.text[:96].replace(chr(10), ' ')}…")

    return {
        "question": question,
        "dense_only": {"sources": dense.source_ids, "elapsed_ms": round(dense.elapsed_ms, 3)},
        "bm25_only": {"sources": bm25.source_ids, "elapsed_ms": round(bm25.elapsed_ms, 3)},
        "hybrid": {"sources": hybrid.source_ids, "elapsed_ms": round(hybrid.elapsed_ms, 3)},
    }


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口：解析参数 → 建引擎 → 按参数分发到体检 / 对比 / 逐题问答。

    参数：argv 参数列表（None 时 argparse 取 `sys.argv[1:]`）。
    返回：int 退出码，恒为 0。
    副作用：把 stdout 切到 UTF-8（非 TTY 时）；构造引擎（读 data/ 建索引）；打印；
          `--json <路径>` 时创建父目录并覆盖写文件；`engine.cache` 存在时附带打印缓存命中统计。
    异常：参数非法由 argparse 抛 SystemExit(2)；`--json` 路径不可写抛 OSError（不兜底）。
    注：`RAGEngine.build(..., quiet=True)` 被传了下去，但注：实际实现为 `quiet` 只被
        `RAGEngine.__init__` 存进 `self.quiet`，全项目没有第二个读取点（引擎自身不打印日志），
        因此演示输出实际只来自本文件的 print——不是被 quiet 关掉的。
    """
    ensure_utf8_console()
    parser = argparse.ArgumentParser(prog="python -m src.demo", description="金融智研引擎演示")
    parser.add_argument("--question", "-q", default=None, help="单个问题（默认跑内置示例）")
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K, help="交给模型的证据条数")
    parser.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid", help="检索模式")
    parser.add_argument("--route", choices=["clause", "case", "metric", "general"], default=None, help="强制指定问题类型")
    parser.add_argument("--expr", default=None, help="元数据过滤表达式，如 'year >= 2024'")
    parser.add_argument("--compare", action="store_true", help="检索策略 A/B 对比")
    parser.add_argument("--stats", action="store_true", help="只打印引擎体检")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    parser.add_argument("--no-faq", action="store_true", help="禁用 FAQ 直出")
    parser.add_argument("--no-evidence", action="store_true", help="不打印证据明细")
    parser.add_argument("--json", default=None, help="结果落盘路径")
    args = parser.parse_args(argv)

    engine = RAGEngine.build(
        enable_cache=not args.no_cache,
        enable_faq=not args.no_faq,
        final_top_k=args.top_k,
        quiet=True,
    )
    print_header(engine)

    if args.stats:
        print("")
        print(json.dumps(engine.stats(), ensure_ascii=False, indent=2))
        return 0

    payload: Dict[str, Any] = {"stats": engine.stats(), "results": []}

    if args.compare:
        question = args.question or SAMPLE_QUESTIONS[3]
        payload["compare"] = run_compare(engine, question, args.top_k)
    else:
        questions = [args.question] if args.question else SAMPLE_QUESTIONS
        for question in questions:
            result = engine.ask(
                question,
                top_k=args.top_k,
                expr=args.expr,
                route=args.route,
                use_cache=not args.no_cache,
                use_faq=not args.no_faq,
                mode=args.mode,
            )
            print_answer(engine, result, show_evidence=not args.no_evidence)
            payload["results"].append(result.to_dict())

    if engine.cache is not None:
        print("")
        print(f"缓存统计：{engine.cache.stats.to_dict()}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入：{out}")

    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
