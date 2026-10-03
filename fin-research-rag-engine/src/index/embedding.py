"""嵌入与稀疏表示：BGE-M3 风格的「稠密 + 稀疏」双表示。

为什么要同时要稠密和稀疏
------------------------
BGE-M3 的关键设计是**一个模型同时输出三种表示**（稠密向量 / 稀疏词权重 / 多向量）。
本项目取前两种，因为它们正好覆盖两类互补的检索需求：

    稠密向量：把「还能不能买」和「投资者适当性要求」映射到相近位置——管"意思像"
    稀疏权重：把「资管新规」「R2」「第四十二条」「WY2024-01」按词权重保留——管"字面准"

后端可插拔
----------
    local   确定性哈希后端（默认）。零依赖、零网络、零成本，任何机器上结果完全一致，
            用于离线演示 / CI / 单元测试。
    bge-m3  真实后端。装了 FlagEmbedding 或 sentence-transformers 时启用，
            维度对齐后可以直接替换，上层检索逻辑一行都不用改。

接口刻意设计成「批量进、矩阵出」，因为真实部署时 embedding 是网络瓶颈，
必须能一次编码一批；逐条调用在生产里会被放大成几十倍的延迟。
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..config import EMBED_BACKEND, EMBED_DIM, EMBED_MODEL
from ..utils.text import tokenize

__all__ = [
    "EmbeddingBackend",
    "LocalHashingBackend",
    "BGEM3Backend",
    "get_backend",
    "cosine_scores",
    "l2_normalize",
    "sparse_from_text",
    "sparse_dot",
]


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化；零向量保持零向量（不产生 NaN）。"""
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """一次矩阵运算求出 query 与全部候选的余弦分数。"""
    if matrix.size == 0 or query_vec.size == 0:
        return np.zeros((matrix.shape[0] if matrix.ndim == 2 else 0,), dtype=np.float64)
    qn = float(np.linalg.norm(query_vec))
    if qn == 0.0:
        return np.zeros((matrix.shape[0],), dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0.0] = 1.0
    return (matrix @ query_vec) / (norms * qn)


def sparse_from_text(text: str) -> Dict[str, float]:
    """把文本转成稀疏词权重向量：w = 1 + log(tf)。

    子线性词频（而不是原始 tf）是为了抑制高频词支配权重——
    金融条款里「客户」「产品」出现频率极高，用原始词频会让所有条款长得一样。
    """
    counts: Dict[str, int] = {}
    for tok in tokenize(text):
        counts[tok] = counts.get(tok, 0) + 1
    return {tok: 1.0 + math.log(c) for tok, c in counts.items()}


def sparse_dot(query: Dict[str, float], doc: Dict[str, float]) -> float:
    """稀疏点积。遍历较短的一侧，复杂度 O(min(|q|, |d|))。"""
    if not query or not doc:
        return 0.0
    if len(query) > len(doc):
        query, doc = doc, query
    return float(sum(weight * doc.get(term, 0.0) for term, weight in query.items()))


def sparse_norm(vec: Dict[str, float]) -> float:
    return math.sqrt(sum(w * w for w in vec.values())) or 1.0


# ---------------------------------------------------------------------------
# 后端接口
# ---------------------------------------------------------------------------
class EmbeddingBackend:
    """嵌入后端接口。稠密与稀疏必须来自同一后端，否则两路召回的语义会打架。"""

    name = "base"

    def __init__(self, dim: int = EMBED_DIM, model: str = EMBED_MODEL) -> None:
        self.dim = dim
        self.model = model

    @property
    def available(self) -> bool:  # pragma: no cover - 接口默认实现
        return True

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - 接口默认实现
        raise NotImplementedError

    def encode_sparse(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        """默认稀疏实现：基于分词词频。真实 BGE-M3 后端会覆盖它。"""
        return [sparse_from_text(t) for t in texts]

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def describe(self) -> Dict[str, object]:
        return {"backend": self.name, "model": self.model, "dim": self.dim, "available": self.available}


class LocalHashingBackend(EmbeddingBackend):
    """确定性哈希后端（hashing trick）。

        1. tokenize 把文本切成 token（中文单字 + 二元组 + 英文数字整词 + 条款号）
        2. 每个 token 经 blake2b 哈希 -> 桶下标 i 与符号位 s（±1）
        3. v[i] += s * (1 + log(tf))
        4. L2 归一化

    性质：跨机器、跨进程完全确定（刻意不用内置 hash()，它带 PYTHONHASHSEED 随机化）；
    共享 token 的文本余弦相似度更高。效果弱于真实语义 embedding，
    因此在三路召回里与 BM25、稀疏权重互补，而不是单独使用。
    """

    name = "local"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            if not text:
                continue
            counts: Dict[str, int] = {}
            for tok in tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
            for tok, count in counts.items():
                bucket, sign = _hash_token(tok)
                out[row, bucket % self.dim] += sign * (1.0 + math.log(count))
        return l2_normalize(out)


def _hash_token(token: str) -> tuple[int, float]:
    """把 token 哈希成 (桶下标, 符号)。blake2b 保证跨进程确定性。"""
    raw = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
    bucket = int.from_bytes(raw[:8], "big")
    sign = 1.0 if raw[8] & 1 else -1.0
    return bucket, sign


class BGEM3Backend(EmbeddingBackend):
    """真实 BGE-M3 后端（可选依赖）。

    优先用 FlagEmbedding（BGE-M3 官方实现，能同时给出稠密与稀疏权重），
    退而求其次用 sentence-transformers（只有稠密，稀疏回落到词频实现）。
    两者都没装时 `available` 为 False，工厂会自动切回 local 后端，
    因此**引擎代码里不存在「没装模型就跑不了」的分支**。
    """

    name = "bge-m3"

    def __init__(self, dim: int = EMBED_DIM, model: str = EMBED_MODEL) -> None:
        super().__init__(dim=dim, model=model)
        self._mode: Optional[str] = None
        self._impl = None

    def _load(self) -> Optional[str]:
        if self._mode is not None:
            return self._mode
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore

            self._impl = BGEM3FlagModel(self.model, use_fp16=True)
            self._mode = "flagembedding"
            return self._mode
        except Exception:  # noqa: BLE001
            pass
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._impl = SentenceTransformer(self.model)
            self._mode = "sentence-transformers"
            return self._mode
        except Exception:  # noqa: BLE001
            self._mode = ""
            return self._mode

    @property
    def available(self) -> bool:
        return bool(self._load())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        mode = self._load()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        if mode == "flagembedding":
            out = self._impl.encode(list(texts), batch_size=8, max_length=1024)["dense_vecs"]
            return l2_normalize(np.asarray(out, dtype=np.float64))
        if mode == "sentence-transformers":
            out = self._impl.encode(list(texts), normalize_embeddings=True)
            return np.asarray(out, dtype=np.float64)
        # 后端不可用：返回零矩阵会让检索静默失效，所以显式抛错，由工厂层兜住
        raise RuntimeError("BGE-M3 后端不可用：请安装 FlagEmbedding 或 sentence-transformers")

    def encode_sparse(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        mode = self._load()
        if mode == "flagembedding":
            out = self._impl.encode(list(texts), return_sparse=True)["lexical_weights"]
            return [{str(k): float(v) for k, v in item.items()} for item in out]
        return super().encode_sparse(texts)


_BACKEND_CACHE: Dict[str, EmbeddingBackend] = {}


def get_backend(name: str = EMBED_BACKEND, dim: int = EMBED_DIM, model: str = EMBED_MODEL) -> EmbeddingBackend:
    """取嵌入后端。bge-m3 不可用时**自动降级**为本地后端，并在 describe() 里如实标注。"""
    key = f"{name}|{dim}|{model}"
    cached = _BACKEND_CACHE.get(key)
    if cached is not None:
        return cached

    backend: EmbeddingBackend
    if name in ("bge-m3", "bge_m3", "bgem3", "flagembedding"):
        candidate = BGEM3Backend(dim=dim, model=model)
        backend = candidate if candidate.available else LocalHashingBackend(dim=dim, model=f"{model}(fallback:local)")
    else:
        backend = LocalHashingBackend(dim=dim, model="local-hashing")
    _BACKEND_CACHE[key] = backend
    return backend


def encode_sparse_batch(texts: Iterable[str]) -> List[Dict[str, float]]:
    """便捷函数：用默认后端做稀疏编码。"""
    return get_backend().encode_sparse(list(texts))
