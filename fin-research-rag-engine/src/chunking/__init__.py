"""切分子包。

    parent_child    按文档结构的父子块切分（段落按句累积、表格成组带表头）
    metadata        块级元数据抽取 + Milvus 风格过滤表达式求值器

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
