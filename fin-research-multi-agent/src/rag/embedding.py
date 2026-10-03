"""确定性哈希向量（Hashing Trick Embedding）。

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
    """把 token 哈希成 (桶下标, 符号)。使用 blake2b 保证跨进程确定性。"""
    raw = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
    bucket = int.from_bytes(raw[:8], "big")
    sign = 1.0 if raw[8] & 1 else -1.0
    return bucket, sign


def token_weights(tokens: Iterable[str]) -> Dict[str, float]:
    """子线性词频权重：w = 1 + log(tf)，抑制高频词支配向量方向。"""
    counts: Dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    return {tok: 1.0 + math.log(c) for tok, c in counts.items()}


def embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """把一段文本编码成 dim 维的 L2 归一化稠密向量。"""
    vec = np.zeros(dim, dtype=np.float64)
    if not text:
        return vec
    for tok, weight in token_weights(tokenize(text)).items():
        bucket, sign = _hash_token(tok)
        vec[bucket % dim] += sign * weight
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def embed_matrix(texts: Sequence[str], dim: int = EMBED_DIM) -> np.ndarray:
    """批量编码，返回 (n, dim) 矩阵。"""
    if not texts:
        return np.zeros((0, dim), dtype=np.float64)
    return np.vstack([embed(t, dim=dim) for t in texts])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """手写余弦相似度（向量已归一化时等价于点积，这里仍做通用实现）。"""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """一次矩阵运算求出 query 与全部候选的余弦分数，返回 (n,) 数组。"""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float64)
    qn = float(np.linalg.norm(query_vec))
    if qn == 0.0:
        return np.zeros((matrix.shape[0],), dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0.0] = 1.0
    return (matrix @ query_vec) / (norms * qn)


def _demo() -> List[float]:  # pragma: no cover - 手工自检辅助，不参与测试
    a = embed("净利润同比下滑")
    b = embed("净利润同比下降")
    c = embed("研发费用投入")
    return [cosine_similarity(a, b), cosine_similarity(a, c)]
