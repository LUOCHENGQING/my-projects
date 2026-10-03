"""检索流水线：三路召回 → 去重 → 重排 → 父块回溯 → 组装证据。

这是「检索」这件事对外的唯一入口。把它单独抽出来，是为了让每一段的中间结果
都**可见、可测、可回放**：

    三路召回   candidates（含三路分数与名次）
    去重       removed_duplicates（哪些块被判定重复、并到了谁身上）
    重排       reranked（交叉分 / RRF 分 / 元数据分 / 特征明细）
    父块回溯   evidence（子块精确命中 + 父块上下文，一起交给 LLM）

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
    """资料类型与问题类型的契合度。通用路由不做区分，返回 1.0。"""
    preferred = ROUTE_DOC_TYPES.get(route)
    if not preferred:
        return 1.0
    return 1.0 if doc_type in preferred else 0.74


@dataclass
class Evidence:
    """一条最终证据：子块（精确命中）+ 父块（上下文）+ 可追溯的出处信息。"""

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
    """一次检索的完整结果与统计。"""

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
        seen: List[str] = []
        for item in self.evidence:
            if item.source_id not in seen:
                seen.append(item.source_id)
        return seen

    @property
    def top_source_ids(self) -> List[str]:
        return self.source_ids

    def stats(self) -> Dict[str, object]:
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
        counts = {"bm25": 0, "dense": 0, "sparse": 0, "all_three": 0}
        for item in self.candidates:
            hit = item.routes_hit
            for route in hit:
                counts[route] = counts.get(route, 0) + 1
            if len(hit) == 3:
                counts["all_three"] += 1
        return counts

    def to_dict(self, with_context: bool = False) -> Dict[str, object]:
        return {
            "question": self.question,
            "mode": self.mode,
            "stats": self.stats(),
            "evidence": [e.to_dict(with_context=with_context) for e in self.evidence],
        }


class RetrievalPipeline:
    """把「召回 → 去重 → 重排 → 父块回溯」串起来。"""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Optional[Reranker] = None,
        final_top_k: int = FINAL_TOP_K,
        dedup_threshold: float = DEDUP_JACCARD,
        rerank_weights: Optional[Dict[str, float]] = None,
    ) -> None:
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

        candidates = self.retriever.retrieve(question, plan=plan, top_k=self.retriever.recall_top_k)

        if not candidates:
            return RetrievalResult(
                question=question,
                plan=plan,
                mode=mode,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        # ---- 去重 ----
        removed: List[Dict[str, str]] = []
        deduped = deduplicate(candidates, threshold=self.dedup_threshold, merge_log=removed)

        # ---- 重排 ----
        rrf_scores = {c.child_id: c.rrf_score for c in deduped}
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
        """对照组：只用稠密向量的单一向量检索（"换模型没用、换策略才有用"的证明）。"""
        return self.run(question, top_k=top_k, mode="dense")

    def baseline_bm25(self, question: str, top_k: Optional[int] = None) -> RetrievalResult:
        """对照组：纯关键词检索。"""
        return self.run(question, top_k=top_k, mode="bm25")

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    @staticmethod
    def _to_evidence(row: RerankedItem, index: int) -> Evidence:
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
