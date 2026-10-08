"""索引子包：三路召回的三块地基，同一份切分结果 + 同一份元数据的三种索引实现。

在 RAG 全链路中的位置
---------------------
    chunking 产出 (parents, children)
        -> **index（本子包：建 BM25 倒排 / 稠密向量 / 稀疏权重）**
            -> src/retrieve/hybrid.py（HybridRetriever 把三路召回的分数做 RRF 融合）
                -> src/retrieve/rerank.py、src/engine.py（重排、生成、引用）

    bm25            手写 Okapi BM25 倒排索引（字面精确匹配）
    embedding       BGE-M3 风格的稠密 + 稀疏双表示（可插拔后端：local / bge-m3）
    vector_store    Milvus 语义的内存向量库（稠密 / 稀疏 / 标量过滤）

本子包对外的公共名字分为三组：
    BM25Index
    EmbeddingBackend / LocalHashingBackend / BGEM3Backend / get_backend
    cosine_scores / l2_normalize / sparse_from_text / sparse_dot
    Collection / MilvusLiteClient / SearchHit / VectorRecord

输入：子块文本与块级元数据（经 `vector_store.VectorRecord` 落进集合）。
输出：检索命中（`SearchHit` 或 `(doc_index, score)` 对），供上层融合与重排。
调用方：`src/retrieve/hybrid.py`（唯一生产调用点）、`src/faq.py`（复用 embedding）、
`tests/conftest.py` 与 `tests/test_index.py`。

三路召回的三块地基都在这里：BM25 管字面、稠密管语义、稀疏管词权重。
把它们放在同一个子包里是为了强调一件事——**三路必须共享同一份切分结果与元数据**，
否则「同一块内容被三路各自表述」会在融合阶段变成对不齐的假象。

副作用/异常总览：本子包默认零网络、零外部依赖（`bge-m3` 后端需额外安装，
不可用时由 `get_backend()` 自动降级为本地后端，上层无感知）。
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

# 公开 API 清单：`RetrievalBackend` 之类的内部名字刻意不在此列，保持包边界干净
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
