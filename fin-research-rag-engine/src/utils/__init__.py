"""通用工具子包。

    text        中文分词 / 句子切分 / 归一化 / 脱敏 / 相似度
    console     控制台 UTF-8 编码处理
    jsonable    numpy 标量净化（JSON 序列化前的统一处理）

在 RAG 全链路中的位置
--------------------
这三个模块被**全链路复用**，本身不含业务逻辑，也不依赖任何业务模块：

    `utils.text`     切分层（`chunking.parent_child.split_sentences`）、检索层（`index.bm25` /
                     `index.embedding` / `retrieve.hybrid` 的 `tokenize`）、答案层
                     （`answer.faithfulness` 的 `amount_tokens` / `tokenize`、`answer.generator` 的
                     `split_sentences`）、缓存层（`cache.redis_cache.cache_key` 的 `stable_hash`）、
                     脱敏（`ingest.loader.apply_masking` 的 `mask_sensitive`）、轨迹（`tracing.digest`）
    `utils.console`  CLI 入口：`src/demo.py`、`src/serve.py`、`eval/run_eval.py` 启动时调一次
    `utils.jsonable` 数据出场处：`src.engine`、`src.tracing`、`retrieve.pipeline`、`cache.redis_cache`

为什么单独抽出来：这里的每个函数都要求**确定性、零外部依赖**（不引 jieba、不联网、不读配置），
因为检索与忠实度校验的可回归性完全建立在它们之上——换个分词器，历史指标就全部不可比。
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
    # 下面三个分组：控制台 → 文本归一化/脱敏 → 分词与切分 → 统计与哈希，顺序与 import 一致。
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
