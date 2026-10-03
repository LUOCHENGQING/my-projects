"""文本处理工具：中文分词、句子切分、文本归一化。

分词不依赖 jieba 等外部词典，采用「中文单字 + 相邻中文二元组 + 英文/数字整词」
的确定性方案：
    * 单字保证召回（"净利率" 与 "利润" 都能命中）
    * 二元组提供短语级区分度（"净利" / "利润" / "净利率"）
    * 英文与数字（如 ROE、2024、1,286,400.00）按整词保留
这样离线、可复现，且对 BM25 足够有效。
"""

from __future__ import annotations

import re
from typing import List

_CJK = r"\u4e00-\u9fff"
_ASCII_WORD = r"A-Za-z0-9_%\.\-"

# 中文连续串
_CJK_RUN = re.compile(f"[{_CJK}]+")
# 英文 / 数字 / 百分号 / 小数 / 连字符组成的整词
_ASCII_RUN = re.compile(f"[{_ASCII_WORD}]+")

# 金额类数字：支持 1,286,400.00 / 1286400 / 128.64
_NUMBER_RUN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+")

# 全角标点 -> 半角空格；各类空白统一
_PUNCT_MAP = {
    "，": " ", "。": " ", "、": " ", "；": " ", "：": " ", "？": " ", "！": " ",
    "（": " ", "）": " ", "《": " ", "》": " ", "“": " ", "”": " ", "‘": " ",
    "’": " ", "【": " ", "】": " ", "—": " ", "…": " ", "·": " ",
    "\u3000": " ",
}


def normalize_text(text: str) -> str:
    """归一化：统一标点为空格、压缩空白、去零宽字符。"""
    if not text:
        return ""
    out = text.replace("\ufeff", "").replace("\u200b", "")
    for src, dst in _PUNCT_MAP.items():
        out = out.replace(src, dst)
    out = re.sub(r"[ \t\r\f\v]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _ascii_tokens(text: str) -> List[str]:
    """抽取英文 / 数字整词并统一小写：ROE -> roe，1,286,400.00 -> 1286400.00。"""
    tokens: List[str] = []
    for m in _ASCII_RUN.finditer(text):
        tok = m.group(0).strip(".-_%")
        if not tok:
            continue
        tokens.append(tok.lower())
    return tokens


def tokenize(text: str) -> List[str]:
    """把文本切成检索用的 token 列表（确定性、无外部依赖）。"""
    if not text:
        return []
    norm = normalize_text(text)
    tokens: List[str] = []

    for m in _CJK_RUN.finditer(norm):
        run = m.group(0)
        # 单字
        tokens.extend(list(run))
        # 相邻二元组
        if len(run) >= 2:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))

    tokens.extend(_ascii_tokens(norm))
    # 金额类数字：带千分位与小数点的整串（如 1,286,400.00）单独补一个 token，
    # 否则会被逗号切开，"128.64 万元" 与 "1,286,400.00 万元" 就无法互相命中
    for m in _NUMBER_RUN.finditer(norm):
        joined = m.group(0).replace(",", "")
        if joined not in tokens:
            tokens.append(joined)
            # 整数金额再补一个不带小数的形式，提升 "1286400" 这类查询的召回
            if "." in joined:
                integer_part = joined.split(".", 1)[0]
                if integer_part not in tokens:
                    tokens.append(integer_part)

    return tokens


_SENT_END = re.compile(r"(?<=[。！？；])")


def split_sentences(text: str) -> List[str]:
    """按中文句末标点切句，保留标点。用于子块切分与摘要抽取。"""
    if not text:
        return []
    norm = normalize_text(text)
    parts: List[str] = []
    for line in norm.split("\n"):
        line = line.strip()
        if not line:
            continue
        chunks = [c.strip() for c in _SENT_END.split(line) if c.strip()]
        parts.extend(chunks)
    return parts


def first_sentence(text: str, limit: int = 80) -> str:
    """取首句做标题 / 摘要，过长则截断。"""
    sents = split_sentences(text)
    if not sents:
        return ""
    head = sents[0]
    return head if len(head) <= limit else head[:limit] + "…"
