"""手写 Okapi BM25 稀疏检索。

不引入 rank_bm25：一是保持依赖树精简，二是需要完全掌控打分与归一化方式
（三路召回融合要对 BM25 分数做 min-max 归一化，用别人的实现很难保证口径一致）。

BM25 公式：
    score(q, d) = Σ_t IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    IDF(t)     = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

为什么要它：金融场景里大量提问是**字面精确匹配**——
「资管新规怎么说的」「R2 能不能买」「第四十二条」「WY2024-01 的费率」。
这些内容向量化的表示反而不如字面匹配可靠，BM25 在这里比稠密检索更准，
而且不需要训练、可解释性强（能直接回答"这块为什么被召回"）。
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from ..config import BM25_B, BM25_K1
from ..utils.text import tokenize

__all__ = ["BM25Index"]


class BM25Index:
    """不可变（构建后只读）的 BM25 倒排索引。"""

    def __init__(
        self,
        corpus_tokens: Sequence[Sequence[str]],
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_count = len(corpus_tokens)
        self.doc_lengths: List[int] = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_lengths) / self.doc_count) if self.doc_count else 0.0

        # term -> {doc_index: tf}
        self.postings: Dict[str, Dict[int, int]] = {}
        for idx, tokens in enumerate(corpus_tokens):
            for tok in tokens:
                bucket = self.postings.setdefault(tok, {})
                bucket[idx] = bucket.get(idx, 0) + 1

        # 预算 IDF，避免每次检索重复计算
        self.idf: Dict[str, float] = {}
        n = self.doc_count
        for term, posting in self.postings.items():
            df = len(posting)
            self.idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    # ------------------------------------------------------------------
    # 打分
    # ------------------------------------------------------------------
    def score_array(self, query_tokens: Sequence[str]) -> np.ndarray:
        """返回查询对全部文档的 BM25 分数数组。"""
        scores = np.zeros(self.doc_count, dtype=np.float64)
        if self.doc_count == 0 or not query_tokens:
            return scores

        q_counts: Dict[str, int] = {}
        for tok in query_tokens:
            q_counts[tok] = q_counts.get(tok, 0) + 1

        avgdl = self.avgdl or 1.0
        for term, qtf in q_counts.items():
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            # 查询内重复词做轻度加权（对数），避免「客户客户客户」这类输入直接刷分
            q_weight = 1.0 + math.log(qtf)
            for doc_idx, tf in posting.items():
                dl = self.doc_lengths[doc_idx] or 1
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                scores[doc_idx] += idf * q_weight * (tf * (self.k1 + 1.0)) / denom
        return scores

    def search(self, query: str, top_k: int = 10) -> List[tuple[int, float]]:
        """便捷接口：返回 [(doc_index, score), ...] 降序，score > 0。"""
        scores = self.score_array(tokenize(query))
        order = np.argsort(-scores)[: max(1, top_k)]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0.0]

    def matched_terms(self, query_tokens: Sequence[str], doc_index: int) -> List[str]:
        """返回该文档命中的查询词，用于重排的「关键词命中率」特征与可解释性展示。"""
        hits: List[str] = []
        for term in set(query_tokens):
            posting = self.postings.get(term)
            if posting and doc_index in posting:
                hits.append(term)
        return sorted(hits)

    def term_idf(self, term: str) -> float:
        """取某个词的 IDF（0 表示语料里没这个词）。用于解释「为什么这个词命不中」。"""
        return self.idf.get(term, 0.0)

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)
