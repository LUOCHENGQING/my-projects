"""检索流水线：三路召回 → 去重 → 重排 → 父块回溯 → 组装证据。

在 RAG 全链路中的位置
---------------------
    切分 / 索引 → 【召回 hybrid.py】 → 【本模块：去重 → 精排 rerank.py → 父块回溯】 → 生成 → 服务
本模块是「检索」这件事对外的**唯一入口**：上层（src/generate、src/service、api）只需要
构造一次 `RetrievalPipeline`，然后调用 `run(question, ...)` 拿 `RetrievalResult`。
把它单独抽出来，是为了让每一段的中间结果都**可见、可测、可回放**：

    三路召回   candidates（含三路分数与名次）
    去重       removed_duplicates（哪些块被判定重复、并到了谁身上）
    重排       reranked（交叉分 / RRF 分 / 元数据分 / 特征明细）
    父块回溯   evidence（子块精确命中 + 父块上下文，一起交给 LLM）

四步的先后顺序及理由（顺序本身就是设计）
----------------------------------------
    1) 召回（hybrid）：目标**不漏**，子块粒度、候选取宽（默认 20×3 路）；
    2) 去重（deduplicate）：放在重排之前——交叉编码器是整条流水线里最贵的一步，
       先并掉重复块能少算几次；也避免同一段话在最终 5 个证据位里反复占位；
    3) 重排（rerank_candidates）：目标**排得准**，用交叉编码器把真正回答问题的那块顶上来；
    4) 父块回溯（_to_evidence）：放在最后——**排序用细粒度子块（命中准），
       交给 LLM 的上下文用完整父块（不丢前提、例外与适用范围）**。
       若把父块也拿去参与排序，父块体积大、噪声多，反而会把排序带偏。

同时提供 `baseline` 接口：只走单一向量检索 / 只走 BM25。
没有对照组，「混合检索让复杂问题召回率提升 40%」就只是一句话；
有了对照组，它是评测脚本里跑出来的一行数字。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..config import DEDUP_JACCARD, FINAL_TOP_K, RERANK_WEIGHTS
from ..ingest.loader import Corpus
from ..utils.jsonable import to_plain
from .hybrid import Candidate, HybridRetriever, deduplicate
from .rerank import Reranker, RerankedItem, get_reranker, rerank_candidates
from .router import QueryPlan, build_query_plan

__all__ = ["Evidence", "RetrievalResult", "RetrievalPipeline", "ROUTE_DOC_TYPES", "route_affinity"]

# 问题类型 → 资料类型偏好。
# 这是「按问题类型配策略」的第二半：路由不只改召回权重，也改**排序先验**。
# 问条款时优先看监管政策与内部制度，问案例时优先看风险案例与尽调档案，
# 问要素时优先看说明书与产品库——这条先验在金融场景里几乎总是成立的。
ROUTE_DOC_TYPES: Dict[str, tuple] = {
    "clause": ("监管政策", "内部制度"),
    "case": ("风险案例", "尽调档案"),
    "metric": ("产品说明书", "产品要素表", "结构化记录"),
}


def route_affinity(doc_type: str, route: str) -> float:
    """资料类型与问题类型的契合度。通用路由不做区分，返回 1.0。

    参数：doc_type 资料的元数据类型（如「监管政策」「风险案例」）；route 问题类型路由。
    返回：命中偏好列表给 1.0，未命中给 0.74（是**软加权**，不是过滤——不匹配的资料仍可能因
          内容相关而排前面，只是先验上吃点亏）。查不到该路由的偏好（如 general）时返回 1.0。
    """
    preferred = ROUTE_DOC_TYPES.get(route)
    if not preferred:
        return 1.0
    return 1.0 if doc_type in preferred else 0.74


@dataclass
class Evidence:
    """一条最终证据：子块（精确命中）+ 父块（上下文）+ 可追溯的出处信息。

    关键属性：
        evidence_id                      证据编号（E1、E2…），供答案正文做引用角标
        child_id / parent_id / source_id / doc_id  子块、父块与来源定位
        text / context                   子块原文（精确命中片段）+ 父块上下文（完整章节）
        title / section_title / institution / doc_type / effective_date / version
                                         出处元数据，业务方据此判断依据是否有效、是否过期
        score                            最终分（重排融合分）
        cross_score / rrf_score / metadata_score  三个分项，可解释「为什么排在这」
        routes_hit / matched_terms / features     三路命中情况、BM25 命中词、重排特征明细
    """

    evidence_id: str
    child_id: str
    parent_id: str
    source_id: str
    doc_id: str
    title: str
    section_title: str
    institution: str
    doc_type: str
    effective_date: str
    version: str
    kind: str
    text: str
    context: str
    score: float
    cross_score: float
    rrf_score: float
    metadata_score: float
    routes_hit: List[str] = field(default_factory=list)
    matched_terms: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)

    @property
    def citation_label(self) -> str:
        """人可读出处，例如「示例监管机构 · 资产管理产品管理办法 · 四、适当性匹配规则」。"""
        head = " · ".join(p for p in (self.institution, self.title) if p)
        return f"{head} · {self.section_title}".strip(" ·")

    @property
    def updated_at(self) -> str:
        """出处的时间信息，回答里必须带（业务方要能判断依据是不是过期）。"""
        return self.effective_date or "未标注"

    def to_dict(self, with_context: bool = False) -> Dict[str, object]:
        """序列化成可 JSON 化的 dict（分数保留 6 位、matched_terms 截断到 12 条）。

        参数：with_context 是否带上父块上下文（默认不带——它体积最大，只有真正要送 LLM 时才需要）。
        返回：含引用标签 citation_label 的 dict。
        """
        payload: Dict[str, object] = {
            "evidence_id": self.evidence_id,
            "child_id": self.child_id,
            "parent_id": self.parent_id,
            "source_id": self.source_id,
            "title": self.title,
            "section_title": self.section_title,
            "institution": self.institution,
            "doc_type": self.doc_type,
            "effective_date": self.effective_date,
            "version": self.version,
            "kind": self.kind,
            "text": self.text,
            "score": round(float(self.score), 6),
            "cross_score": round(float(self.cross_score), 6),
            "rrf_score": round(float(self.rrf_score), 6),
            "metadata_score": round(float(self.metadata_score), 6),
            "routes_hit": list(self.routes_hit),
            "matched_terms": list(self.matched_terms)[:12],
            "features": {k: round(float(v), 4) for k, v in self.features.items()},
            "citation_label": self.citation_label,
        }
        if with_context:
            payload["context"] = self.context
        return to_plain(payload)


@dataclass
class RetrievalResult:
    """一次检索的完整结果与统计（中间产物全部保留，便于回放与评测）。

    关键属性：
        question            原始问题
        plan                本次执行的 QueryPlan（路由、权重、过滤、查询变体）
        candidates          去重后的召回候选（三路分数与名次齐全）
        evidence            最终证据列表（父块回溯后交给 LLM 的部分）
        reranked            重排明细（交叉分 / 特征），用于解释排序
        removed_duplicates  被判重并合并的记录
        mode                检索模式：hybrid / dense / bm25（后两者是对照组）
        elapsed_ms          本次检索耗时（毫秒），写进轨迹做性能回归
    """

    question: str
    plan: QueryPlan
    candidates: List[Candidate] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    reranked: List[RerankedItem] = field(default_factory=list)
    removed_duplicates: List[Dict[str, str]] = field(default_factory=list)
    mode: str = "hybrid"
    elapsed_ms: float = 0.0

    @property
    def source_ids(self) -> List[str]:
        """证据涉及的去重后来源 id 列表（按首次出现顺序）。

        用途：判断"答案是否有多个独立来源互证"，也用于答案里的来源清单。
        """
        seen: List[str] = []
        for item in self.evidence:
            if item.source_id not in seen:
                seen.append(item.source_id)
        return seen

    @property
    def top_source_ids(self) -> List[str]:
        """与 source_ids 同义的别名（保留给早期调用方的命名）。"""
        return self.source_ids

    def stats(self) -> Dict[str, object]:
        """汇总本次检索的统计：模式、召回 / 去重 / 证据条数、三路命中分布、耗时与完整计划。

        返回：可 JSON 化的 dict（写入 runs/ 轨迹，也是评测脚本读取的字段）。
        """
        return {
            "mode": self.mode,
            "recalled": len(self.candidates),
            "deduplicated": len(self.removed_duplicates),
            "evidence": len(self.evidence),
            "routes": self._route_counts(),
            "elapsed_ms": round(self.elapsed_ms, 3),
            "plan": self.plan.to_dict(),
        }

    def _route_counts(self) -> Dict[str, int]:
        """统计候选里各路命中数量，以及三路全中的条数（all_three）。

        用途：判断"多路共识"这个信号在本次检索里到底起了多大作用；
        如果 all_three 长期为 0，说明三路召回其实没形成互补，需要回去看索引与权重。
        """
        counts = {"bm25": 0, "dense": 0, "sparse": 0, "all_three": 0}
        for item in self.candidates:
            hit = item.routes_hit
            for route in hit:
                counts[route] = counts.get(route, 0) + 1
            if len(hit) == 3:
                counts["all_three"] += 1
        return counts

    def to_dict(self, with_context: bool = False) -> Dict[str, object]:
        """序列化成对外 JSON：问题、模式、统计与证据列表。

        参数：with_context 是否在每条证据里带上父块上下文（要送 LLM 时传 True）。
        返回：可 JSON 化的 dict；不含 candidates / reranked 明细以控制体积。
        """
        return {
            "question": self.question,
            "mode": self.mode,
            "stats": self.stats(),
            "evidence": [e.to_dict(with_context=with_context) for e in self.evidence],
        }


class RetrievalPipeline:
    """把「召回 → 去重 → 重排 → 父块回溯」串起来。

    这是检索层对外的唯一入口类：上层只需 `RetrievalPipeline(retriever).run(question)`。
    关键属性：
        retriever        三路召回器（HybridRetriever）
        reranker         重排后端（缺省 get_reranker()：优先真实 bge-reranker，否则本地实现）
        final_top_k      最终交给 LLM 的证据条数（默认 config.FINAL_TOP_K = 5）
        dedup_threshold  近似去重阈值（默认 config.DEDUP_JACCARD = 0.82）
        rerank_weights   重排融合权重（默认 config.RERANK_WEIGHTS：cross 0.55 / rrf 0.30 / metadata 0.15）
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Optional[Reranker] = None,
        final_top_k: int = FINAL_TOP_K,
        dedup_threshold: float = DEDUP_JACCARD,
        rerank_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """装配流水线各段组件。

        参数：retriever 三路召回器；reranker 重排后端（None 时自动选择）；
              final_top_k 最终证据条数；dedup_threshold 去重阈值；rerank_weights 融合权重。
        返回：无。副作用：reranker 为 None 时会触发后端加载（可能加载本地模型），故此处只构造一次。
        """
        self.retriever = retriever
        self.reranker = reranker or get_reranker()
        self.final_top_k = int(final_top_k)
        self.dedup_threshold = float(dedup_threshold)
        self.rerank_weights = dict(rerank_weights or RERANK_WEIGHTS)

    # ------------------------------------------------------------------
    # 年份硬过滤的"有条件提升"
    # ------------------------------------------------------------------
    def _promote_year_filter(self, plan, explicit_expr: Optional[str]) -> None:
        """用户显式指定年份且资料库确实存在该年份版本时，把软加权提升为硬过滤。

        为什么要有条件地提升：问「2023 年的合格投资者标准」时，如果不做硬过滤，
        新版规定（300 万元）因为用词更贴近、篇幅更大，往往排到旧版前面，
        答案就变成了"2023 年的标准是 300 万元"——**时效性错误**，金融场景里这是硬伤。
        但如果资料库里根本没有 2023 年的资料，硬过滤会把唯一的相关证据也筛掉，
        所以只在"该年份确实存在"时才提升。

        参数：plan 待修改的 QueryPlan（就地修改）；explicit_expr 用户显式给的过滤表达式。
        返回：无。
        副作用：满足条件时就地写入 plan.filter_expr（硬过滤表达式）与 plan.promoted_year=True。
        说明：显式表达式存在时直接返回——用户说了算，不叠加推断结果；
              年份解析失败或不可提升（year_promotable 未置位）时也直接返回。
        """
        if explicit_expr:
            return
        year = plan.filters.get("year")
        if year is None or not plan.filters.get("year_promotable"):
            # 年份限定的是"财务数据年度"而不是"文档版本"（如「示例集团 2023 年应收账款」），
            # 不能硬过滤：那样会把当期的尽调档案整体筛掉，正确答案反而没了
            return
        try:
            y = int(year)
        except (TypeError, ValueError):
            return
        if y in self.retriever.known_years:
            # 提升为硬过滤：等价于"召回前就把非该年份版本剔除"，只在资料库确实有该年份时才敢做
            plan.filter_expr = f"year = {y}"
            plan.promoted_year = True

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(
        self,
        question: str,
        top_k: Optional[int] = None,
        expr: Optional[str] = None,
        route: Optional[str] = None,
        corpus: Optional[Corpus] = None,
        mode: str = "hybrid",
    ) -> RetrievalResult:
        """执行一次完整检索，返回含证据与中间产物的 RetrievalResult。

        参数：
            question  用户问题
            top_k     最终证据条数上限（缺省用 self.final_top_k = 5）
            expr      用户**显式**给的元数据过滤表达式（硬过滤；会阻止年份自动提升）
            route     强制指定问题类型路由；None 时自动分类
            corpus    语料对象，用于识别机构名与资料类型（可为 None）
            mode      "hybrid"（默认，三路召回 + 去重 + 重排）或 "dense" / "bm25"（对照组，不重排）
        返回：RetrievalResult；召回为空时返回 candidates/evidence 均为空的空结果（不抛异常）。
        副作用：读取索引与重排后端；无状态写入（耗时与统计只记录在返回值里）。
        """
        started = time.perf_counter()
        limit = max(1, top_k or self.final_top_k)
        plan = build_query_plan(question, corpus=corpus, top_k=limit, route=route, filter_expr=expr)
        self._promote_year_filter(plan, expr)

        if mode in ("dense", "bm25"):
            # 对照组：**单一通路且不重排**。
            # 这才是「混合检索替代单一向量检索」里真正的 baseline ——
            # 如果给 baseline 也加上重排，比的就只是"有没有重排"，而不是"有没有混合召回"。
            candidates = (
                self.retriever.search_dense_only(question, top_k=limit, expr=expr)
                if mode == "dense"
                else self.retriever.search_bm25_only(question, top_k=limit, expr=expr)
            )
            # 单路结果没有交叉分，用 rrf_score 顶替 cross_score / final_score，
            # 只是为了保持 Evidence 字段结构一致，让下游不必为对照组写分支
            evidence = [self._to_evidence(RerankedItem(item=c, cross_score=c.rrf_score, final_score=c.rrf_score), i)
                        for i, c in enumerate(candidates, start=1)]
            return RetrievalResult(
                question=question,
                plan=plan,
                candidates=candidates,
                evidence=evidence,
                reranked=[],
                mode=mode,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        # 召回阶段故意取宽（recall_top_k，默认每路 20 条）：候选多一条只多花一点重排成本，
        # 漏掉一条正确答案却是无法补救的——"召回看漏不漏"就体现在这里
        candidates = self.retriever.retrieve(question, plan=plan, top_k=self.retriever.recall_top_k)

        if not candidates:
            # 空结果也返回完整结构（而不是 None / 抛错），上游的"无证据"话术才有统一入口
            return RetrievalResult(
                question=question,
                plan=plan,
                mode=mode,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        # ---- 去重 ----
        # 放在重排之前：交叉编码器是整条链路最贵的一步，先并掉重复块可少算几次，
        # 也避免同一段话把最终 5 个证据位占掉好几个，导致答案覆盖度上不去
        removed: List[Dict[str, str]] = []
        deduped = deduplicate(candidates, threshold=self.dedup_threshold, merge_log=removed)

        # ---- 重排 ----
        rrf_scores = {c.child_id: c.rrf_score for c in deduped}
        # 元数据先验 = 60% 推断条件命中率 + 40% 资料类型与问题类型的契合度；
        # 它只是参与排序的软信号，不剔除任何候选（"加错过滤比不过滤危险得多"）
        meta_scores = {
            c.child_id: 0.6 * c.metadata_score + 0.4 * route_affinity(str(c.meta.get("doc_type", "")), plan.route)
            for c in deduped
        }
        reranked = rerank_candidates(
            question,
            deduped,
            reranker=self.reranker,
            top_k=limit,
            weights=self.rerank_weights,
            rrf_scores=rrf_scores,
            metadata_scores=meta_scores,
            text_of=lambda item: item.text,
            quality_of=lambda item: item.quality_score,
            id_of=lambda item: item.child_id,
        )

        # ---- 父块回溯 + 组装证据 ----
        # 排序已经在子块粒度上做完（准），这里再把每个命中块换回完整父块作为 context（全），
        # 保证 LLM 看得到前提条件、例外条款与适用范围，而不是一句孤立的结论
        evidence = [self._to_evidence(row, index) for index, row in enumerate(reranked, start=1)]

        return RetrievalResult(
            question=question,
            plan=plan,
            candidates=deduped,
            evidence=evidence,
            reranked=reranked,
            removed_duplicates=removed,
            mode=mode,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ------------------------------------------------------------------
    # 对照组：单一向量检索 / 纯关键词检索
    # ------------------------------------------------------------------
    def baseline_dense(self, question: str, top_k: Optional[int] = None) -> RetrievalResult:
        """对照组：只用稠密向量的单一向量检索（"换模型没用、换策略才有用"的证明）。

        参数：question 问题；top_k 证据条数（缺省 final_top_k）。
        返回：mode="dense" 的 RetrievalResult（无重排、无三路融合）。
        """
        return self.run(question, top_k=top_k, mode="dense")

    def baseline_bm25(self, question: str, top_k: Optional[int] = None) -> RetrievalResult:
        """对照组：纯关键词检索。

        参数：question 问题；top_k 证据条数（缺省 final_top_k）。
        返回：mode="bm25" 的 RetrievalResult（无重排、无向量语义）。
        """
        return self.run(question, top_k=top_k, mode="bm25")

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    @staticmethod
    def _to_evidence(row: RerankedItem, index: int) -> Evidence:
        """把重排结果包装成最终 Evidence（父块回溯在这一步完成）。

        参数：row 重排结果（item 是 Candidate，其 context 已是父块文本）；
              index 1 基序号，用于生成 evidence_id（E1、E2…）。
        返回：Evidence；分数取 row.final_score，并保留 cross_score / rrf_score / metadata_score
              与重排特征明细 features，便于回答"为什么这条排第一"。
        """
        cand: Candidate = row.item
        meta = cand.meta
        return Evidence(
            evidence_id=f"E{index}",
            child_id=cand.child_id,
            parent_id=cand.parent_id,
            source_id=cand.source_id,
            doc_id=cand.doc_id,
            title=str(meta.get("title", "")),
            section_title=cand.section_title,
            institution=str(meta.get("institution", "")),
            doc_type=str(meta.get("doc_type", "")),
            effective_date=str(meta.get("effective_date", "")),
            version=str(meta.get("version", "")),
            kind=cand.kind,
            text=cand.text,
            context=cand.context,
            score=row.final_score,
            cross_score=row.cross_score,
            rrf_score=cand.rrf_score,
            metadata_score=cand.metadata_score,
            routes_hit=cand.routes_hit,
            matched_terms=list(cand.matched_terms),
            features=row.features.to_dict(),
        )
