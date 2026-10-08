"""手写 Okapi BM25 稀疏检索：三路召回里负责「字面精确匹配」的那一路。

在 RAG 全链路中的位置
---------------------
    chunking 产出子块文本
        -> 上层 `tokenize()` 后传入本模块建倒排（`src/retrieve/hybrid.py`：
           `BM25Index([tokenize(c.text) for c in children])`）
            -> `score_array()` 打分 -> 与稠密/稀疏两路做 RRF 融合
                -> 重排 -> 生成答案

输入：已分词的语料（`Sequence[Sequence[str]]`，由 `src.utils.text.tokenize` 产出，
      不是原始字符串）与已分词的查询。
输出：`score_array()` 返回与语料等长的分数向量（第 i 位对应第 i 篇文档）；
      `search()` 返回 `[(doc_index, score), ...]` 降序且只保留 score > 0 的结果。
调用方：`src/retrieve/hybrid.py`（生产路径）、`tests/test_index.py`、`tests/conftest.py`。
副作用/异常：构造时建立内存倒排，之后**只读**，天然可多线程并发查询；无 IO、无网络。

不引入 rank_bm25：一是保持依赖树精简，二是需要完全掌控打分与归一化方式
（三路召回融合要对 BM25 分数做 min-max 归一化，用别人的实现很难保证口径一致）。

BM25 公式：
    score(q, d) = Σ_t IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    IDF(t)     = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

口径说明（k1 / b，默认取自 `src.config`）：
    k1 = BM25_K1（默认 1.5）——词频饱和系数。tf 越大单次命中收益越小，
         上限约为 (k1 + 1) 倍；k1 越大越"奖励重复出现"。
    b  = BM25_B （默认 0.75）——文档长度归一化强度，取值 [0, 1]。
         b = 1 表示完全按 |d|/avgdl 惩罚长文档，b = 0 表示不惩罚。
         金融条款长短差异大（一行表格 vs 整段制度），b = 0.75 是经验上的折中。
    注：实际实现为 —— 公式里额外乘了一个查询词权重 `q_weight = 1 + ln(qtf)`
    （查询内重复词的轻度加权），见 `score_array()`；这是本实现对标准 Okapi 的唯一扩展。

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
    """不可变（构建后只读）的 BM25 倒排索引。

    关键属性（均在 `__init__` 中一次性算好，查询阶段不再修改）：
        k1              词频饱和系数，默认 `BM25_K1`（1.5）
        b               文档长度归一化强度，默认 `BM25_B`（0.75）
        doc_count       语料文档数 N
        doc_lengths     每篇文档的 token 数 |d|
        avgdl           平均文档长度（空语料时为 0.0）
        postings        term -> {doc_index: tf} 的倒排表，`score_array()` 只遍历它
        idf             term -> IDF 的预计算表（`term_idf()` 也读它）

    只读性带来的好处：并发查询无需加锁；代价是增量更新必须重建索引。
    """

    def __init__(
        self,
        corpus_tokens: Sequence[Sequence[str]],
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        """构建倒排索引与 IDF 表。

        参数：
            corpus_tokens  已分词的语料，外层是文档、内层是该文档的 token 列表
                           （文档下标 i 即后续 `score_array()` 返回数组的下标）
            k1             词频饱和系数，默认 `BM25_K1`（可用环境变量覆盖）
            b              长度归一化强度，默认 `BM25_B`
        返回：无（构造函数）。
        副作用/异常：
            无 IO；只写实例属性。空语料合法（`avgdl` 为 0.0，查询直接返回全零分数）。
            注：实际实现为 —— 未校验 k1/b 取值范围，传 0 或负数不会报错，
            只会得到与标准 BM25 不同的打分口径。
        """
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_count = len(corpus_tokens)
        self.doc_lengths: List[int] = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_lengths) / self.doc_count) if self.doc_count else 0.0

        # term -> {doc_index: tf}
        # 用 dict 而非稠密矩阵：金融语料词表大、单块命中词很少，稀疏存储更省内存
        self.postings: Dict[str, Dict[int, int]] = {}
        for idx, tokens in enumerate(corpus_tokens):
            for tok in tokens:
                bucket = self.postings.setdefault(tok, {})
                bucket[idx] = bucket.get(idx, 0) + 1

        # 预算 IDF，避免每次检索重复计算
        # 用的是 ln(1 + ...) 的平滑形式：即使某词出现在所有文档里，IDF 也 > 0 而不会变负
        self.idf: Dict[str, float] = {}
        n = self.doc_count
        for term, posting in self.postings.items():
            df = len(posting)
            self.idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    # ------------------------------------------------------------------
    # 打分
    # ------------------------------------------------------------------
    def score_array(self, query_tokens: Sequence[str]) -> np.ndarray:
        """返回查询对全部文档的 BM25 分数数组。

        参数：query_tokens 已分词的查询 token 序列（**不要传原始问题字符串**，
              否则会被当成一个超长 token 而永远命不中）。
        返回：
            np.ndarray，float64，长度恒为 `doc_count`，下标 i 对应第 i 篇文档；
            未命中的文档该位为 0.0。空语料或无查询词时直接返回全零数组（不抛异常）。
        副作用/异常：
            无副作用（不修改索引、不改入参）；不抛异常。
        说明：查询词权重记为 1 + ln(qtf)，因此查询内重复出现同一个词只会获得
              对数级加权，无法靠堆词刷分。
        """
        scores = np.zeros(self.doc_count, dtype=np.float64)
        if self.doc_count == 0 or not query_tokens:
            return scores

        q_counts: Dict[str, int] = {}
        for tok in query_tokens:
            q_counts[tok] = q_counts.get(tok, 0) + 1

        # 空语料时 avgdl 为 0；用 or 1.0 兜底避免除零，同时不影响任何文档的长度惩罚（此时也没有文档）
        avgdl = self.avgdl or 1.0
        for term, qtf in q_counts.items():
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            # 查询内重复词做轻度加权（对数），避免「客户客户客户」这类输入直接刷分
            q_weight = 1.0 + math.log(qtf)
            for doc_idx, tf in posting.items():
                # 长度为 0 的文档兜底成 1，避免 dl/avgdl 退化成 0 造成异常口径
                dl = self.doc_lengths[doc_idx] or 1
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                scores[doc_idx] += idf * q_weight * (tf * (self.k1 + 1.0)) / denom
        return scores

    def search(self, query: str, top_k: int = 10) -> List[tuple[int, float]]:
        """便捷接口：返回 [(doc_index, score), ...] 降序，score > 0。

        参数：
            query  原始查询字符串（内部会调用 `tokenize()`，与建索引时同一套分词器）
            top_k  最多返回多少条；实际实现会取 `max(1, top_k)`，因此传 0 也会返回 1 条
        返回：
            List[tuple[int, float]]：`(文档下标, BM25 分数)`，按分数降序；
            **分数为 0 的文档被过滤掉**，所以返回值长度通常小于语料规模。
        副作用/异常：
            无副作用；不抛异常（空查询/空语料返回空列表）。
        """
        scores = self.score_array(tokenize(query))
        # argsort 只保证前 top_k 有序，没用到的那部分顺序无意义，不要依赖
        order = np.argsort(-scores)[: max(1, top_k)]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0.0]

    def matched_terms(self, query_tokens: Sequence[str], doc_index: int) -> List[str]:
        """返回该文档命中的查询词，用于重排的「关键词命中率」特征与可解释性展示。

        参数：
            query_tokens 已分词的查询 token 序列
            doc_index    语料中的文档下标
        返回：
            List[str]：命中的查询词，**去重后按字典序升序**（`set()` 去重 + `sorted()`），
            因此同一组词每次调用结果稳定。未命中任何词时返回 `[]`。
        副作用/异常：
            无副作用；下标越界不报错，只会让所有词都判为未命中而返回 `[]`。
        """
        hits: List[str] = []
        for term in set(query_tokens):
            posting = self.postings.get(term)
            if posting and doc_index in posting:
                hits.append(term)
        return sorted(hits)

    def term_idf(self, term: str) -> float:
        """取某个词的 IDF（0 表示语料里没这个词）。用于解释「为什么这个词命不中」。

        参数：term 单个 token（需与建索引时 `tokenize()` 的切分口径一致）。
        返回：float，该词的 IDF；未登录词返回 0.0。
        副作用/异常：无；纯查表。
        """
        return self.idf.get(term, 0.0)

    @property
    def vocabulary_size(self) -> int:
        """词表大小（倒排表的键数量），用于体检「分词是否过碎」与索引规模展示。

        参数：无。
        返回：int，非负。
        副作用/异常：无；`src/retrieve/hybrid.py` 的 `describe()` 会读它。
        """
        return len(self.postings)
