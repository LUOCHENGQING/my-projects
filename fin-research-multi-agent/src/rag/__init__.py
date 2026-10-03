"""RAG（检索增强）子包。

    embedding   确定性哈希向量（手写，无外部 API）
    bm25        Okapi BM25 稀疏检索
    chunking    父子块切分
    corpus      文档加载与解析（front-matter / Markdown 表格）
    facts       结构化财务事实抽取
    retriever   BM25 + 向量 混合检索与加权重排
"""

from __future__ import annotations

from .chunking import ChildChunk, ParentChunk, build_chunks, split_document
from .corpus import Document, DocumentStore, load_documents
from .embedding import cosine_similarity, embed, embed_matrix
from .facts import FactStore, MetricFact, build_fact_store
from .retriever import HybridRetriever, RetrievedSnippet

__all__ = [
    "ChildChunk",
    "ParentChunk",
    "split_document",
    "build_chunks",
    "build_fact_store",
    "Document",
    "DocumentStore",
    "load_documents",
    "embed",
    "embed_matrix",
    "cosine_similarity",
    "FactStore",
    "MetricFact",
    "HybridRetriever",
    "RetrievedSnippet",
]
