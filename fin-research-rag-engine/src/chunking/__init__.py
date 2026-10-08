"""切分子包：RAG 全链路中「文档 -> 可检索块」这一步的唯一入口。

在链路中的位置
--------------
    ingest（解析成 SourceDocument / RawSection / Block）
        -> **chunking（本子包：切分 + 打元数据）**
            -> index（BM25 倒排 / 稠密向量 / 稀疏权重）
                -> retrieve（三路召回、元数据过滤、融合重排）
                    -> engine（组装答案与引用）

本子包是纯计算层：不做 IO、不依赖网络、不依赖向量库，只把内存中的文档对象
变成内存中的块对象（`ParentChunk` / `ChildChunk` 列表），因此可被单测直接覆盖。

子模块职责
----------
    parent_child    按文档结构做父子块切分（段落按句累积带重叠、表格成组并重复表头）
    metadata        块级多维元数据抽取 + Milvus 风格过滤表达式求值器

本包对外的公共名字（`__all__`）分两类：
    切分侧：ParentChunk / ChildChunk / ChunkStats / build_chunks / split_document /
            split_paragraph_into_children / split_table_into_children / chunk_stats /
            BLOCK_PARAGRAPH / BLOCK_TABLE
    元数据侧：extract_metadata / MetadataFilter / build_filter / filter_chunks /
              FilterError / META_FIELDS

输入输出
--------
输入：`src.ingest.loader.SourceDocument` 序列（含章节、段落块与表格块）。
输出：`(parents, children)` 两个列表，通常被 `src.engine.RAGEngine` 建索引时消费，
`children` 的块级元数据同时充当索引层与召回层的**过滤条件载体**。

设计要点：切分依据是文档结构而不是固定字数；表格必须成组切；每个块都带
可用于**召回前过滤**的扁平元数据。
"""

from __future__ import annotations

from .metadata import META_FIELDS, FilterError, MetadataFilter, build_filter, extract_metadata, filter_chunks
from .parent_child import (
    BLOCK_PARAGRAPH,
    BLOCK_TABLE,
    ChildChunk,
    ChunkStats,
    ParentChunk,
    build_chunks,
    chunk_stats,
    split_document,
    split_paragraph_into_children,
    split_table_into_children,
)

# 公开 API 清单：显式声明，避免上层 `from src.chunking import *` 时把内部辅助对象也带出去
__all__ = [
    "ParentChunk",
    "ChildChunk",
    "ChunkStats",
    "build_chunks",
    "split_document",
    "split_paragraph_into_children",
    "split_table_into_children",
    "chunk_stats",
    "BLOCK_PARAGRAPH",
    "BLOCK_TABLE",
    "extract_metadata",
    "MetadataFilter",
    "build_filter",
    "filter_chunks",
    "FilterError",
    "META_FIELDS",
]
