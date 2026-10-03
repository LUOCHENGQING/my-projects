"""混合检索 + 加权重排。

检索管线：
    查询 -> ┌ BM25（稀疏，词面命中）      ┐
           │ 哈希向量（稠密，语义近似）   ├-> min-max 归一化 -> 加权融合 -> Top-K -> 父块回填
           └ 元数据匹配（公司/年份/文档类型）┘

融合公式（权重来自 config.RERANK_WEIGHTS，可配置）：
    final = w_kw * keyword + w_vec * vector + w_meta * metadata
    keyword = 0.7 * BM25_norm + 0.3 * 命中词占比

命中词占比（hit ratio）是刻意加进来的一路信号：BM25 对长文档有天然偏置，
而「查询里的关键财务词有多少真的出现在这块里」对金融问答是强信号。

召回后做父块回填：返回给 Agent 的是「子块文本（精确命中）+ 父块上下文（完整章节）」，
既准又全。每条结果都带 source_id / 章节标题，供引用可追溯。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import EMBED_DIM, RERANK_WEIGHTS
from ..utils.jsonable import to_plain
from ..utils.text import tokenize
from .bm25 import BM25Index
from .chunking import ChildChunk, ParentChunk
from .embedding import cosine_scores, embed, embed_matrix

__all__ = ["RetrievedSnippet", "HybridRetriever"]


@dataclass
class RetrievedSnippet:
    """一条检索结果。"""

    child_id: str
    parent_id: str
    source_id: str
    doc_id: str
    company: str
    period: str
    year: Optional[int]
    section_title: str
    text: str            # 子块原文（精确命中片段）
    context: str         # 父块原文（回填的上下文）
    score: float
    components: Dict[str, float] = field(default_factory=dict)
    matched_terms: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)

    @property
    def citation_label(self) -> str:
        return f"{self.company} {self.period} · {self.section_title}".strip(" ·")

    def to_dict(self) -> Dict[str, object]:
        return to_plain(
            {
                "child_id": self.child_id,
                "parent_id": self.parent_id,
                "source_id": self.source_id,
                "company": self.company,
                "period": self.period,
                "year": self.year,
                "section_title": self.section_title,
                "text": self.text,
                "score": round(self.score, 4),
                "components": {k: round(float(v), 4) for k, v in self.components.items()},
                "matched_terms": self.matched_terms,
            }
        )


def _minmax(scores: np.ndarray) -> np.ndarray:
    """min-max 归一化到 [0,1]；全相等时返回全 0，避免虚假高分。"""
    if scores.size == 0:
        return scores
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class HybridRetriever:
    """BM25 + 向量 + 元数据的混合检索器。"""

    def __init__(
        self,
        parents: Sequence[ParentChunk],
        children: Sequence[ChildChunk],
        weights: Optional[Dict[str, float]] = None,
        embed_dim: int = EMBED_DIM,
    ) -> None:
        self.parents: List[ParentChunk] = list(parents)
        self.children: List[ChildChunk] = list(children)
        self.weights: Dict[str, float] = dict(weights or RERANK_WEIGHTS)
        self.embed_dim = embed_dim

        self._parent_by_id: Dict[str, ParentChunk] = {p.parent_id: p for p in self.parents}

        self._child_tokens: List[List[str]] = [tokenize(c.text) for c in self.children]
        self._bm25 = BM25Index(self._child_tokens)
        self._matrix: np.ndarray = embed_matrix([c.text for c in self.children], dim=embed_dim)

    # ------------------------------------------------------------------
    # 基础属性
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.children)

    @property
    def companies(self) -> List[str]:
        return sorted({c.meta.get("company", "") for c in self.children if c.meta.get("company")})

    def parent_of(self, child: ChildChunk) -> Optional[ParentChunk]:
        return self._parent_by_id.get(child.parent_id)

    # ------------------------------------------------------------------
    # 元数据打分
    # ------------------------------------------------------------------
    @staticmethod
    def _metadata_score(meta: Dict[str, object], filters: Dict[str, object]) -> float:
        """请求了多少个元数据约束，就按满足比例给分；没给约束时给 1.0。"""
        active = {k: v for k, v in filters.items() if v not in (None, "", [])}
        if not active:
            return 1.0
        hits = 0
        for key, want in active.items():
            got = meta.get(key)
            if isinstance(want, str) and isinstance(got, str):
                if want in got or got in want:
                    hits += 1
            elif got == want:
                hits += 1
        return hits / len(active)

    # ------------------------------------------------------------------
    # 检索主入口
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        company: Optional[str] = None,
        year: Optional[int] = None,
        doc_type: Optional[str] = None,
        strict: bool = False,
        exclude_children: Optional[Sequence[str]] = None,
    ) -> List[RetrievedSnippet]:
        """单查询混合检索。"""
        return self.search_multi(
            [query],
            top_k=top_k,
            company=company,
            year=year,
            doc_type=doc_type,
            strict=strict,
            exclude_children=exclude_children,
        )

    def search_multi(
        self,
        queries: Sequence[str],
        top_k: int = 5,
        company: Optional[str] = None,
        year: Optional[int] = None,
        doc_type: Optional[str] = None,
        strict: bool = False,
        exclude_children: Optional[Sequence[str]] = None,
    ) -> List[RetrievedSnippet]:
        """多查询混合检索：每路查询独立打分，同一子块取各路最高分。"""
        if not self.children or not queries:
            return []

        filters: Dict[str, object] = {"company": company, "year": year, "doc_type": doc_type}
        excluded = set(exclude_children or ())
        n = len(self.children)

        # 元数据硬过滤掩码
        mask = np.ones(n, dtype=bool)
        if strict:
            for idx, child in enumerate(self.children):
                if self._metadata_score(child.meta, filters) < 1.0:
                    mask[idx] = False
        if excluded:
            for idx, child in enumerate(self.children):
                if child.child_id in excluded:
                    mask[idx] = False
        if not mask.any():
            return []

        # 累加各路查询的最佳融合分
        best = np.full(n, -1.0, dtype=np.float64)
        best_components: Dict[int, Dict[str, float]] = {}
        best_terms: Dict[int, List[str]] = {}
        best_queries: Dict[int, List[str]] = {}

        w_kw = float(self.weights.get("keyword", 0.0))
        w_vec = float(self.weights.get("vector", 0.0))
        w_meta = float(self.weights.get("metadata", 0.0))

        for q in queries:
            if not q or not q.strip():
                continue
            q_tokens = tokenize(q)
            if not q_tokens:
                continue

            bm25_raw = self._bm25.score_array(q_tokens)
            vec_raw = cosine_scores(embed(q, dim=self.embed_dim), self._matrix)
            bm25_norm = _minmax(bm25_raw)
            vec_norm = _minmax(vec_raw)

            q_terms = set(q_tokens)
            for idx in range(n):
                if not mask[idx]:
                    continue

                matched = self._bm25.matched_terms(q_tokens, idx)
                hit_ratio = (len(set(matched)) / len(q_terms)) if q_terms else 0.0
                # 统一转成原生 float：numpy 标量会破坏 LangGraph 的 checkpointer 序列化
                keyword = float(0.7 * float(bm25_norm[idx]) + 0.3 * hit_ratio)
                meta_score = float(self._metadata_score(self.children[idx].meta, filters))
                fused = float(w_kw * keyword + w_vec * float(vec_norm[idx]) + w_meta * meta_score)

                # 完全无命中的块（BM25=0 且向量相似度<=0）直接丢弃，避免噪声污染
                if bm25_raw[idx] <= 0.0 and vec_raw[idx] <= 0.0:
                    continue

                if fused > best[idx]:
                    best[idx] = fused
                    best_components[idx] = {
                        "keyword": keyword,
                        "vector": float(vec_norm[idx]),
                        "metadata": meta_score,
                        "bm25_raw": float(bm25_raw[idx]),
                        "cosine_raw": float(vec_raw[idx]),
                    }
                    best_terms[idx] = matched
                    best_queries[idx] = [q]
                elif abs(fused - best[idx]) < 1e-12 and idx in best_queries:
                    best_queries[idx].append(q)
                    for term in matched:
                        if term not in best_terms[idx]:
                            best_terms[idx].append(term)

        ranked = sorted(
            (i for i in range(n) if best[i] >= 0.0),
            key=lambda i: -best[i],
        )[: max(1, top_k)]

        return [self._make_snippet(i, best[i], best_components.get(i, {}), best_terms.get(i, []), best_queries.get(i, [])) for i in ranked]

    def _make_snippet(
        self,
        idx: int,
        score: float,
        components: Dict[str, float],
        matched_terms: List[str],
        queries: List[str],
    ) -> RetrievedSnippet:
        child = self.children[idx]
        parent = self.parent_of(child)
        context = parent.text if parent else child.text
        return RetrievedSnippet(
            child_id=child.child_id,
            parent_id=child.parent_id,
            source_id=child.source_id,
            doc_id=child.doc_id,
            company=str(child.meta.get("company", "")),
            period=str(child.meta.get("period", "")),
            year=child.meta.get("year") if isinstance(child.meta.get("year"), int) else None,
            section_title=str(child.meta.get("section_title", "")),
            text=child.text,
            context=context,
            score=float(score),
            components=components,
            matched_terms=sorted(set(matched_terms))[:12],
            queries=list(queries),
        )
