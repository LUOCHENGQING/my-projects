"""手写 Okapi BM25 稀疏检索（混合检索里的「词面」一路）。

层次与职责
----------
位于 RAG 证据层的稀疏打分环节：输入是已经分好词的子块（chunking 的产物），
输出是查询对每个子块的词面相关性原始分。本模块只负责打分，不负责归一化、
也不负责融合——min-max 归一化与加权融合统一由 retriever.HybridRetriever 处理，
所以这里刻意保留未经缩放的原始分。

关键类：BM25Index。
主要输入：corpus_tokens = Sequence[Sequence[str]]（utils.text.tokenize 的输出，
          同一 token 重复出现即表示词频 tf）。
主要输出：score_array() 的 (doc_count,) float64 数组；search() 的 [(doc_index, score)]。
被谁调用：src/rag/retriever.py 的 HybridRetriever.__init__（建索引）与 search_multi
          （逐路查询打分，并取 matched_terms 算命中率特征）。

不引入 rank_bm25 等第三方库：一是不给依赖树增加负担，二是需要完全掌控
归一化方式（HybridRetriever 要对 BM25 与向量分数做 min-max 融合）。

BM25 公式：
    score(q, d) = Σ_t IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    IDF(t)     = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

两点口径（读源码前先明确）：
    * IDF 用 ln(1 + ...) 变体，任何 df 下都非负，便于下游做 min-max 融合；
    * 查询词频额外乘 q_weight = 1 + ln(qtf)，这是本项目在标准公式之外补的轻度
      加权——标准 BM25 通常只按查询词是否出现计权。
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from ..config import BM25_B, BM25_K1
from ..utils.text import tokenize

__all__ = ["BM25Index"]


class BM25Index:
    """不可变（构建后只读）的 BM25 倒排索引。

    职责：把一批已分词的文档建成倒排表并预先算好 IDF，之后只提供只读打分查询。

    关键属性：
        k1 / b       超参，默认取自 config.BM25_K1 / BM25_B
        doc_count    文档数（在本项目里 = 子块数）
        doc_lengths  每篇文档的 token 数
        avgdl        平均文档长度；空语料时为 0.0
        postings     term -> {doc_index: tf} 倒排表
        idf          term -> 预先算好的 IDF

    状态流转：__init__ 一次性完成「统计长度 -> 建倒排 -> 预算 IDF」，此后全部属性
    只读，不再变更；多次查询之间没有共享的可变状态。
    """

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = BM25_K1, b: float = BM25_B) -> None:
        """构建索引。

        参数：
            corpus_tokens: 已分词的文档序列（tokenize 的输出）；同一 token 重复出现
                即表示该词的词频 tf。
            k1: 词频饱和参数，越大则高频带来的收益衰减越慢。
            b:  文档长度归一化强度（0 = 不归一化，1 = 完全归一化）。
        返回：None（构造器）。
        副作用：填充 doc_lengths / avgdl / postings / idf。
        异常：不主动抛异常；空语料时 doc_count=0、avgdl=0.0，打分一律返回全 0。
        """
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus_tokens)
        self.doc_lengths: List[int] = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_lengths) / self.doc_count) if self.doc_count else 0.0

        # term -> {doc_index: tf}
        # 为什么先把倒排建完：检索是热路径且会被多路查询反复调用，
        # 这里一次性把「词 -> 文档列表」摊平，避免每次查询都重扫全部文档。
        self.postings: Dict[str, Dict[int, int]] = {}
        for idx, tokens in enumerate(corpus_tokens):
            for tok in tokens:
                bucket = self.postings.setdefault(tok, {})
                bucket[idx] = bucket.get(idx, 0) + 1

        # 预算 IDF，避免检索时重复计算
        # 为什么预算：IDF 只与语料统计有关、与查询无关，属于典型的「构建期算一次」。
        self.idf: Dict[str, float] = {}
        n = self.doc_count
        for term, posting in self.postings.items():
            df = len(posting)
            self.idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score_array(self, query_tokens: Sequence[str]) -> np.ndarray:
        """返回查询对全部文档的 BM25 分数数组。

        参数：query_tokens —— 查询的分词结果（重复 token 表示查询词频 qtf）。
        返回：np.ndarray，形状 (doc_count,)，dtype float64；空语料或查询无量词命中时全 0。
        副作用：无（只读索引）。
        """
        scores = np.zeros(self.doc_count, dtype=np.float64)
        if self.doc_count == 0 or not query_tokens:
            return scores

        # 同一查询内重复词只计一次权重，但保留查询词频的轻度加权
        # 为什么要取 log：直接用 qtf 会让查询里写了三遍的关键词把分数放大三倍，
        # 对数只做轻度倾斜，避免用户措辞习惯主导排序。
        q_counts: Dict[str, int] = {}
        for tok in query_tokens:
            q_counts[tok] = q_counts.get(tok, 0) + 1

        for term, qtf in q_counts.items():
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            q_weight = 1.0 + math.log(qtf)
            for doc_idx, tf in posting.items():
                # 为什么 `or 1`：空文档（分词后长度为 0）会让长度归一化的分母退化，
                # 这里兜底成 1，保证分母恒为正。
                dl = self.doc_lengths[doc_idx] or 1
                # 为什么 `avgdl or 1.0`：空语料时 avgdl=0.0，这里避免除零（b=0 时本项不影响分数）。
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[doc_idx] += idf * q_weight * (tf * (self.k1 + 1.0)) / denom
        return scores

    def search(self, query: str, top_k: int = 10) -> List[tuple[int, float]]:
        """便捷接口：返回 [(doc_index, score), ...] 降序，score > 0。

        参数：query 原始查询串（内部会调用 tokenize）；top_k 截断条数。
        返回：只保留正分文档的降序列表，长度 <= top_k。
        副作用：无。
        """
        scores = self.score_array(tokenize(query))
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0.0]

    def matched_terms(self, query_tokens: Sequence[str], doc_index: int) -> List[str]:
        """返回该文档命中的查询词，用于重排的「关键词命中率」特征与可解释性展示。

        参数：query_tokens 查询分词；doc_index 目标文档下标。
        返回：去重并排序后的命中词列表（对查询去重后逐个查倒排表）。
        副作用：无。doc_index 越界不会抛异常——查的是 posting 字典，只会返回空列表。
        """
        hits: List[str] = []
        for term in set(query_tokens):
            posting = self.postings.get(term)
            if posting and doc_index in posting:
                hits.append(term)
        return sorted(hits)
