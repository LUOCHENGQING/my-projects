"""确定性哈希向量（Hashing Trick Embedding）——混合检索里的「语义」一路（稠密）。

层次与职责
----------
RAG 证据层的稠密编码环节：把子块文本与查询串映射到同一个 EMBED_DIM 维空间的
单位向量，让「用词不同但 token 重叠」的文本也能互相命中。本模块只负责编码与
相似度计算，不建索引、不排序（融合与排序由 retriever.HybridRetriever 完成）。

关键函数：
    embed / embed_matrix —— 单条 / 批量编码；要换成真实 embedding 服务，只需替换这两个
    cosine_similarity / cosine_scores —— 单对 / 一对多的余弦相似度
    token_weights / _hash_token —— 子线性词频权重与确定性哈希（内部细节）

主要输入：文本（str）或文本序列。
主要输出：L2 归一化后的 float64 向量（EMBED_DIM 维，默认 256，可用环境变量 EMBED_DIM 覆盖）。
被谁调用：src/rag/retriever.py（HybridRetriever.__init__ 建矩阵、search_multi 编码查询）。
注：cosine_scores 与 token_weights 未列入本模块 __all__，但 retriever 会直接
    `from .embedding import cosine_scores` 使用。

设计动机
--------
真实的金融投研系统会调用外部 embedding 服务，但本项目要求「无 key 也一定能跑通」。
这里用经典的 hashing trick 手写一个**确定性**向量器：

    1. 用 tokenize() 把文本切成 token（中文单字 + 二元组 + 英文数字整词）
    2. 每个 token 经 blake2b 哈希 -> 桶下标 i 与符号位 s（+1 / -1）
    3. v[i] += s * (1 + log(tf))
    4. 对 v 做 L2 归一化

性质：
    * 确定性：同一段文本在任何机器、任何进程都得到完全相同的向量
      （刻意不用 Python 内置 hash()，它带 PYTHONHASHSEED 随机化）
    * 语义近似：共享 token 的文本余弦相似度更高，二元组进一步强化短语匹配
    * 零依赖、零网络、零成本

它的检索效果弱于真实语义 embedding，因此在 HybridRetriever 里与 BM25 加权融合互补。
要换成真实 embedding，只需替换本模块的 embed() / embed_matrix() 两个函数。

确定性的边界：除了 numpy，本模块没有任何外部依赖，也不访问网络；同一个 token 在
任何机器、任何进程都落进同一个桶，所以向量可复现、可缓存、可回归对比。
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

from ..config import EMBED_DIM
from ..utils.text import tokenize

__all__ = ["embed", "embed_matrix", "cosine_similarity", "token_weights"]


def _hash_token(token: str) -> tuple[int, float]:
    """把 token 哈希成 (桶下标, 符号)。使用 blake2b 保证跨进程确定性。

    参数：token —— 单个词条（tokenize 的输出）。
    返回：(bucket, sign)。bucket 是 16 字节摘要前 8 字节的大端无符号整数，**不在此处
        取模**（取模交给调用方，同一个哈希值可复用到任意维度）；sign 取第 9 个字节的
        最低位，为 +1.0 或 -1.0。
    副作用：无。刻意不用内置 hash()——它受 PYTHONHASHSEED 随机化影响，会破坏可复现性。
    """
    raw = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
    bucket = int.from_bytes(raw[:8], "big")
    sign = 1.0 if raw[8] & 1 else -1.0
    return bucket, sign


def token_weights(tokens: Iterable[str]) -> Dict[str, float]:
    """子线性词频权重：w = 1 + log(tf)，抑制高频词支配向量方向。

    参数：tokens —— 可迭代的 token（允许重复；重复即词频 tf）。
    返回：Dict[str, float]，token -> 权重；空输入返回空字典。
    副作用：无。
    """
    counts: Dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    return {tok: 1.0 + math.log(c) for tok, c in counts.items()}


def embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """把一段文本编码成 dim 维的 L2 归一化稠密向量。

    参数：text 原始文本（内部会 tokenize）；dim 目标维度（默认 config.EMBED_DIM）。
    返回：np.ndarray，形状 (dim,)，dtype float64。空文本返回全 0 向量——模长为 0，
        因此在相似度计算里与任何向量都得 0 分。
    副作用：无。
    注：哈希冲突（不同 token 落进同一桶）是设计内行为；符号位让冲突项不总是同向叠加，
        可部分抵消冲突带来的系统性偏差。
    """
    vec = np.zeros(dim, dtype=np.float64)
    if not text:
        return vec
    for tok, weight in token_weights(tokenize(text)).items():
        bucket, sign = _hash_token(tok)
        # 为什么带符号：符号位使撞进同一桶的不同 token 不总是同向累加，
        # 从而降低哈希冲突对向量方向造成的系统性偏置。
        vec[bucket % dim] += sign * weight
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def embed_matrix(texts: Sequence[str], dim: int = EMBED_DIM) -> np.ndarray:
    """批量编码，返回 (n, dim) 矩阵。

    参数：texts 文本序列；dim 与 embed 同义。
    返回：np.ndarray (n, dim)。空序列返回形状 (0, dim) 的全零矩阵（不是空数组）。
    副作用：无。逐条调用 embed，未做并行化——语料规模小，可读性优先。
    """
    if not texts:
        return np.zeros((0, dim), dtype=np.float64)
    return np.vstack([embed(t, dim=dim) for t in texts])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """手写余弦相似度（向量已归一化时等价于点积，这里仍做通用实现）。

    参数：a / b —— 任意同维向量（不要求已归一化）。
    返回：float，落在 [-1, 1]；任一向量模长为 0（例如空文本的向量）时返回 0.0，避免除零。
    副作用：无。
    """
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """一次矩阵运算求出 query 与全部候选的余弦分数，返回 (n,) 数组。

    参数：query_vec 形状 (dim,) 的查询向量；matrix 形状 (n, dim) 的候选矩阵。
    返回：np.ndarray，形状 (n,)，dtype float64。matrix 为空时返回形状 (0,) 的数组；
        query 为零向量时返回全 0。
    副作用：无。就地改写的是临时数组 norms，不会改动入参 matrix。
    """
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float64)
    qn = float(np.linalg.norm(query_vec))
    if qn == 0.0:
        return np.zeros((matrix.shape[0],), dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    # 为什么把 0 模长改成 1：空子块会产生零向量，直接做除法会得到 nan 并污染整列排序；
    # 置 1 后这些位置自然得 0 分（分子本来就是 0）。
    norms[norms == 0.0] = 1.0
    return (matrix @ query_vec) / (norms * qn)


def _demo() -> List[float]:  # pragma: no cover - 手工自检辅助，不参与测试
    """手工自检辅助：确认「近义表述」比「无关表述」更相似。

    参数：无。
    返回：List[float] —— [sim(净利润同比下滑, 净利润同比下降), sim(净利润同比下滑, 研发费用投入)]，
        正常情况下前者应显著大于后者（哈希向量的效果弱于真实语义 embedding，
        但共享 token 的文本得分一定更高）。
    副作用：无（纯计算；带 pragma: no cover，只用于人工验证）。
    """
    a = embed("净利润同比下滑")
    b = embed("净利润同比下降")
    c = embed("研发费用投入")
    return [cosine_similarity(a, b), cosine_similarity(a, c)]
