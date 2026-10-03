"""通用工具子包。

    text        中文分词 / 句子切分 / 归一化 / 脱敏 / 相似度
    console     控制台 UTF-8 编码处理
    jsonable    numpy 标量净化（JSON 序列化前的统一处理）
"""

from __future__ import annotations

from .console import ensure_utf8_console
from .jsonable import to_plain
from .text import (
    amount_tokens,
    digest,
    first_sentence,
    jaccard,
    mask_sensitive,
    normalize_text,
    normalize_whitespace,
    shingles,
    split_paragraphs,
    split_sentences,
    stable_hash,
    token_counts,
    tokenize,
    truncate,
)

__all__ = [
    "ensure_utf8_console",
    "to_plain",
    "normalize_text",
    "normalize_whitespace",
    "mask_sensitive",
    "tokenize",
    "token_counts",
    "split_sentences",
    "split_paragraphs",
    "first_sentence",
    "truncate",
    "shingles",
    "jaccard",
    "amount_tokens",
    "stable_hash",
    "digest",
]
