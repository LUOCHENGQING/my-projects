"""索引子包。

    bm25            手写 Okapi BM25 倒排索引（字面精确匹配）
    embedding       BGE-M3 风格的稠密 + 稀疏双表示（可插拔后端）
    vector_store    Milvus 语义的内存向量库（稠密 / 稀疏 / 标量过滤）

三路召回的三块地基都在这里：BM25 管字面、稠密管语义、稀疏管词权重。
把它们放在同一个子包里是为了强调一件事——**三路必须共享同一份切分结果与元数据**，
否则「同一块内容被三路各自表述」会在融合阶段变成对不齐的假象。
"""

from __future__ import annotations

from .bm25 import BM25Index
from .embedding import (
    BGEM3Backend,
    EmbeddingBackend,
    LocalHashingBackend,
    cosine_scores,
    get_backend,
    l2_normalize,
    sparse_dot,
    sparse_from_text,
)
from .vector_store import Collection, MilvusLiteClient, SearchHit, VectorRecord

__all__ = [
    "BM25Index",
    "EmbeddingBackend",
    "LocalHashingBackend",
    "BGEM3Backend",
    "get_backend",
    "cosine_scores",
    "l2_normalize",
    "sparse_from_text",
    "sparse_dot",
    "Collection",
    "MilvusLiteClient",
    "SearchHit",
    "VectorRecord",
]
