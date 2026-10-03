"""三路召回 + RRF 融合。

管线：
    查询（含改写变体）
        ├─ BM25 倒排      → 字面精确匹配（条款号、产品代码、专有名词）
        ├─ 稠密向量检索   → 语义相似（"还能不能买" ≈ "投资者适当性要求"）
        └─ 稀疏词权重检索 → 介于两者之间（保留词权重）
                        ↓
                RRF 融合（按排名，不按分数）
                        ↓
                   Top-N 候选 → 交给重排

为什么用 RRF 而不是加权求和
---------------------------
BM25 的分数是无界的（可以到 20+），余弦相似度在 [-1,1]，稀疏点积又是另一个量纲。
加权求和必须先做 min-max 归一化，而归一化**依赖候选集内的极值**——
同一份文档，换一批候选，归一化后的相对高低就可能反转，检索结果因此不稳定。

RRF 只用**名次**：`score = Σ w_r / (k + rank_r)`。它天然免疫量纲不一致，
在不同查询之间也更稳，这就是它在混合检索里几乎成为默认做法的原因。

RRF 之上再做两件事：
    * 三路权重按**问题类型**配置（条款类偏 BM25、案例类偏稠密）；
    * 召回前用元数据过滤表达式把候选集收窄。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..chunking.metadata import MetadataFilter
from ..chunking.parent_child import ChildChunk, ParentChunk
from ..config import FINAL_TOP_K, RECALL_TOP_K, RRF_K
from ..index.bm25 import BM25Index
from ..index.embedding import EmbeddingBackend, get_backend, sparse_from_text
from ..index.vector_store import MilvusLiteClient
from ..utils.jsonable import to_plain
from ..utils.text import jaccard, shingles, tokenize
from .router import QueryPlan, build_query_plan

__all__ = ["Candidate", "HybridRetriever", "COLLECTION_NAME"]

COLLECTION_NAME = "finrag_chunks"


@dataclass
class Candidate:
    """一条召回候选。保留三路各自的分数与名次，因此「为什么它被召回」是可回答的。"""

    child_id: str
    parent_id: str
    source_id: str
    doc_id: str
    section_title: str
    text: str
    kind: str
    meta: Dict[str, object]
    quality_score: float = 1.0
    # 三路信号
    bm25_score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    # 融合与解释
    rrf_score: float = 0.0
    matched_terms: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    route: str = "general"
    # 父块回填
    context: str = ""
    metadata_score: float = 1.0

    @property
    def citation_label(self) -> str:
        institution = str(self.meta.get("institution", ""))
        title = str(self.meta.get("title", ""))
        head = " · ".join(p for p in (institution, title) if p)
        return f"{head} · {self.section_title}".strip(" ·")

    @property
    def routes_hit(self) -> List[str]:
        """命中了哪几路（用于展示"三路共识"这一信号）。"""
        hit: List[str] = []
        if self.bm25_rank is not None:
            hit.append("bm25")
        if self.dense_rank is not None:
            hit.append("dense")
        if self.sparse_rank is not None:
            hit.append("sparse")
        return hit

    def to_dict(self) -> Dict[str, object]:
        return to_plain(
            {
                "child_id": self.child_id,
                "parent_id": self.parent_id,
                "source_id": self.source_id,
                "section_title": self.section_title,
                "kind": self.kind,
                "score": round(float(self.rrf_score), 6),
                "components": {
                    "bm25": round(float(self.bm25_score), 6),
                    "dense": round(float(self.dense_score), 6),
                    "sparse": round(float(self.sparse_score), 6),
                    "rrf": round(float(self.rrf_score), 6),
                },
                "ranks": {"bm25": self.bm25_rank, "dense": self.dense_rank, "sparse": self.sparse_rank},
                "routes_hit": self.routes_hit,
                "matched_terms": self.matched_terms[:12],
                "route": self.route,
                "text": self.text,
            }
        )


class HybridRetriever:
    """BM25 + 稠密向量 + 稀疏权重的三路召回器，带 RRF 融合与元数据过滤。"""

    def __init__(
        self,
        parents: Sequence[ParentChunk],
        children: Sequence[ChildChunk],
        backend: Optional[EmbeddingBackend] = None,
        rrf_k: int = RRF_K,
        recall_top_k: int = RECALL_TOP_K,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.parents: List[ParentChunk] = list(parents)
        self.children: List[ChildChunk] = list(children)
        self.backend = backend or get_backend()
        self.rrf_k = int(rrf_k)
        self.recall_top_k = int(recall_top_k)

        self._parent_by_id: Dict[str, ParentChunk] = {p.parent_id: p for p in self.parents}
        self._child_by_id: Dict[str, ChildChunk] = {c.child_id: c for c in self.children}

        # ---- 一路：BM25 倒排 ----
        self._child_tokens: List[List[str]] = [tokenize(c.text) for c in self.children]
        self._bm25 = BM25Index(self._child_tokens)

        # ---- 二路 / 三路：向量库（稠密 + 稀疏同库，保证同一块的两路表示不会对不齐）----
        self._client = MilvusLiteClient()
        self._collection = self._client.create_collection(collection_name, dim=self.backend.dim, metric="COSINE")
        self._build_vector_index()

    # ------------------------------------------------------------------
    # 建索引
    # ------------------------------------------------------------------
    def _build_vector_index(self) -> None:
        from ..index.vector_store import VectorRecord

        texts = [c.text for c in self.children]
        dense = self.backend.encode(texts) if texts else np.zeros((0, self.backend.dim))
        sparse = self.backend.encode_sparse(texts) if texts else []
        records = [
            VectorRecord(
                record_id=child.child_id,
                dense=dense[i] if i < dense.shape[0] else None,
                sparse=sparse[i] if i < len(sparse) else {},
                meta=dict(child.meta, quality_score=child.quality_score, kind=child.kind),
            )
            for i, child in enumerate(self.children)
        ]
        self._collection.insert(records)
        self._collection.flush()

    # ------------------------------------------------------------------
    # 基础属性
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.children)

    @property
    def vocabulary_size(self) -> int:
        return self._bm25.vocabulary_size

    @property
    def known_years(self) -> List[int]:
        """资料库里实际出现过的年份集合。

        用途：用户明确问「2023 年的标准」时，只有当资料库确实存在 2023 年的版本，
        才敢把这个年份提升为**硬过滤**（召回前剔除）。否则一旦元数据缺失，
        硬过滤会把唯一正确的证据也筛掉——宁可只做软加权。
        """
        years = {c.meta.get("year") for c in self.children if isinstance(c.meta.get("year"), int)}
        return sorted(int(y) for y in years)

    def parent_of(self, child_id: str) -> Optional[ParentChunk]:
        child = self._child_by_id.get(child_id)
        return self._parent_by_id.get(child.parent_id) if child else None

    def child(self, child_id: str) -> Optional[ChildChunk]:
        return self._child_by_id.get(child_id)

    def describe(self) -> Dict[str, object]:
        return {
            "children": len(self.children),
            "parents": len(self.parents),
            "vocabulary": self._bm25.vocabulary_size,
            "rrf_k": self.rrf_k,
            "recall_top_k": self.recall_top_k,
            "embedding": self.backend.describe(),
            "vector_store": self._client.describe(),
        }

    # ------------------------------------------------------------------
    # 单路检索（供 A/B 对比与演示）
    # ------------------------------------------------------------------
    def search_bm25_only(self, query: str, top_k: int = 10, expr: Optional[str] = None) -> List[Candidate]:
        """基线一：只用 BM25（关键词检索）。"""
        return self._search_single_route(query, "bm25", top_k, expr)

    def search_dense_only(self, query: str, top_k: int = 10, expr: Optional[str] = None) -> List[Candidate]:
        """基线二：只用稠密向量（单一向量检索）。这就是本项目的对照组。"""
        return self._search_single_route(query, "dense", top_k, expr)

    def _search_single_route(self, query: str, route: str, top_k: int, expr: Optional[str]) -> List[Candidate]:
        if not query.strip() or not self.children:
            return []
        filter_expr = MetadataFilter.parse(expr)
        if route == "bm25":
            ranked = self._bm25_ranked(query, top_k, filter_expr)
        elif route == "dense":
            ranked = self._dense_ranked(query, top_k, filter_expr)
        elif route == "sparse":
            ranked = self._sparse_ranked(query, top_k, filter_expr)
        else:
            raise ValueError(f"未知通路：{route}")

        out: List[Candidate] = []
        for rank, (idx, score) in enumerate(ranked):
            cand = self._make_candidate(idx, query)
            setattr(cand, f"{route}_score", float(score))
            setattr(cand, f"{route}_rank", rank)
            cand.rrf_score = 1.0 / (self.rrf_k + rank + 1)
            out.append(cand)
        return out[: max(1, top_k)]

    # ------------------------------------------------------------------
    # 三路排名
    # ------------------------------------------------------------------
    def _mask(self, filter_expr: MetadataFilter) -> Optional[np.ndarray]:
        """硬过滤掩码；空表达式返回 None（表示不过滤）。"""
        if filter_expr.is_empty:
            return None
        mask = np.array([filter_expr.matches(c.meta) for c in self.children], dtype=bool)
        return mask

    def _bm25_ranked(self, query: str, top_k: int, filter_expr: MetadataFilter) -> List[Tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.score_array(tokens)
        mask = self._mask(filter_expr)
        if mask is not None:
            scores = np.where(mask, scores, -1.0)
        order = np.argsort(-scores)
        out: List[Tuple[int, float]] = []
        for idx in order:
            if scores[idx] <= 0.0:
                break
            out.append((int(idx), float(scores[idx])))
            if len(out) >= max(1, top_k):
                break
        return out

    def _dense_ranked(self, query: str, top_k: int, filter_expr: MetadataFilter) -> List[Tuple[int, float]]:
        if not self.children or not query.strip():
            return []
        vec = self.backend.encode_one(query)
        # 空查询 / 全是停用词的查询会得到零向量，此时余弦全为 0，
        # 直接返回空而不是把"全库零分"当成有效召回（否则空问题也会给出 5 条证据）
        if not np.any(vec):
            return []
        hits = self._collection.search_dense(vec, top_k=top_k, expr=filter_expr.expr or None)
        index = {c.child_id: i for i, c in enumerate(self.children)}
        return [(index[h.record_id], h.dense_score) for h in hits if h.record_id in index]

    def _sparse_ranked(self, query: str, top_k: int, filter_expr: MetadataFilter) -> List[Tuple[int, float]]:
        if not self.children or not query.strip():
            return []
        hits = self._collection.search_sparse(sparse_from_text(query), top_k=top_k, expr=filter_expr.expr or None)
        index = {c.child_id: i for i, c in enumerate(self.children)}
        return [(index[h.record_id], h.sparse_score) for h in hits if h.record_id in index]

    # ------------------------------------------------------------------
    # 三路召回 + RRF 融合
    # ------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        plan: Optional[QueryPlan] = None,
        top_k: Optional[int] = None,
        expr: Optional[str] = None,
        route: Optional[str] = None,
        corpus=None,
    ) -> List[Candidate]:
        """完整三路召回，返回按 RRF 融合分排序的候选列表。"""
        if not self.children or not (question or "").strip():
            return []
        resolved = plan or build_query_plan(question, corpus=corpus, top_k=top_k or self.recall_top_k, route=route, filter_expr=expr)
        limit = max(1, top_k or resolved.top_k or self.recall_top_k)
        hard = MetadataFilter.parse(resolved.filter_expr or expr)
        weights = resolved.weights or {"bm25": 1 / 3, "dense": 1 / 3, "sparse": 1 / 3}
        queries = resolved.queries or [question]
        share = 1.0 / max(1, len(queries))

        acc: Dict[int, Candidate] = {}

        for query in queries:
            for route_name in ("bm25", "dense", "sparse"):
                if route_name == "bm25":
                    ranked = self._bm25_ranked(query, self.recall_top_k, hard)
                elif route_name == "dense":
                    ranked = self._dense_ranked(query, self.recall_top_k, hard)
                else:
                    ranked = self._sparse_ranked(query, self.recall_top_k, hard)

                weight = float(weights.get(route_name, 0.0))
                for rank, (idx, score) in enumerate(ranked):
                    child = self.children[idx]
                    cand = acc.get(idx)
                    if cand is None:
                        cand = self._make_candidate(idx, query)
                        cand.route = resolved.route
                        acc[idx] = cand
                    # RRF：只看名次，天然免疫三路量纲不一致
                    cand.rrf_score += share * weight * (1.0 / (self.rrf_k + rank + 1))
                    current = getattr(cand, f"{route_name}_score")
                    if score > current:
                        setattr(cand, f"{route_name}_score", float(score))
                    current_rank = getattr(cand, f"{route_name}_rank")
                    if current_rank is None or rank < current_rank:
                        setattr(cand, f"{route_name}_rank", rank)
                    if query not in cand.queries:
                        cand.queries.append(query)
                    for term in self._bm25.matched_terms(tokenize(query), idx):
                        if term not in cand.matched_terms:
                            cand.matched_terms.append(term)

        if not acc:
            return []

        soft_filters = resolved.filters or {}
        results: List[Candidate] = []
        for idx, cand in acc.items():
            cand.metadata_score = self.metadata_score(cand.meta, soft_filters)
            # 多路共识的轻微加成：三路都命中说明这条大概率是真的相关
            consensus = 1.0 + 0.05 * (len(cand.routes_hit) - 1)
            cand.rrf_score *= consensus
            results.append(cand)

        results.sort(key=lambda c: -c.rrf_score)
        return results[:limit]

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _make_candidate(self, idx: int, query: str) -> Candidate:
        child = self.children[idx]
        parent = self._parent_by_id.get(child.parent_id)
        return Candidate(
            child_id=child.child_id,
            parent_id=child.parent_id,
            source_id=child.source_id,
            doc_id=child.doc_id,
            section_title=child.section_title,
            text=child.text,
            kind=child.kind,
            meta=dict(child.meta),
            quality_score=float(child.quality_score),
            matched_terms=self._bm25.matched_terms(tokenize(query), idx),
            queries=[query],
            context=parent.text if parent else child.text,
        )

    @staticmethod
    def metadata_score(meta: Dict[str, object], filters: Dict[str, object]) -> float:
        """软元数据契合度：请求了几个条件，就按满足比例给分；无条件时给 1.0。"""
        active = {k: v for k, v in (filters or {}).items() if v not in (None, "", [])}
        if not active:
            return 1.0
        hits = 0
        for key, want in active.items():
            if key == "year_gte":
                got = meta.get("year")
                hits += 1 if isinstance(got, int) and got >= int(want) else 0
            elif key == "year_lte":
                got = meta.get("year")
                hits += 1 if isinstance(got, int) and got <= int(want) else 0
            else:
                got = meta.get(key)
                if got is None:
                    continue
                if isinstance(want, str) and isinstance(got, str):
                    hits += 1 if (want in got or got in want) else 0
                else:
                    hits += 1 if got == want else 0
        return hits / len(active)


def deduplicate(
    candidates: Sequence[Candidate],
    threshold: float = 0.82,
    keep: Optional[Sequence[str]] = None,
    merge_log: Optional[List[Dict[str, str]]] = None,
) -> List[Candidate]:
    """近似去重：同一段文字常被多路（或多查询变体）重复召回。

    用 token 二元组 Jaccard 判定重复，**保留分数更高的一条**，并把被丢弃者的
    命中通路合并过去（这样"三路共识"的统计不会因为去重而失真）。
    不去重最直接的后果是 5 个上下文位被同一段话占掉 3 个，
    LLM 看到的"证据"其实是同一份，答案覆盖度自然上不去。
    """
    if not candidates:
        return []
    protected = set(keep or ())
    ordered = sorted(candidates, key=lambda c: -c.rrf_score)
    kept: List[Candidate] = []
    signatures: List[set] = []

    for cand in ordered:
        sig = shingles(tokenize(cand.text), 2)
        duplicate_of: Optional[int] = None
        for i, existing in enumerate(signatures):
            if jaccard(sig, existing) >= threshold:
                duplicate_of = i
                break
        if duplicate_of is None or cand.child_id in protected:
            kept.append(cand)
            signatures.append(sig)
            continue

        target = kept[duplicate_of]
        # 合并证据：让"被去重掉的那条命中了哪几路"仍然体现在保留项上
        for route in cand.routes_hit:
            if getattr(target, f"{route}_rank") is None:
                setattr(target, f"{route}_rank", getattr(cand, f"{route}_rank"))
                setattr(target, f"{route}_score", getattr(cand, f"{route}_score"))
        for term in cand.matched_terms:
            if term not in target.matched_terms:
                target.matched_terms.append(term)
        for q in cand.queries:
            if q not in target.queries:
                target.queries.append(q)
        if merge_log is not None:
            merge_log.append({"child_id": cand.child_id, "merged_into": target.child_id})
    return kept
