"""文本处理工具：中文分词、句子切分、文本归一化（utils 层，零业务依赖）。

层次与职责：
    处于最底层，被检索链路的三个模块复用：
        src/rag/bm25.py、src/rag/embedding.py、src/rag/retriever.py —— tokenize
        src/rag/chunking.py —— split_sentences
        tests/test_rag.py —— tokenize
    本模块不读任何外部词典、不依赖网络，所有函数都是纯函数（同输入必得同输出）。

对外关键函数：normalize_text / tokenize / split_sentences / first_sentence
（前三个由 src/utils/__init__.py 再导出；first_sentence 目前仅供模块内/外部直接 import 使用）。

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

# 汉字区间（基本区 U+4E00–U+9FFF），用于切出中文连续串
_CJK = r"\u4e00-\u9fff"
# 英文/数字整词允许的字符集：字母、数字、下划线、百分号、点、连字符
# 注意：这里刻意不含逗号，千分位金额由 _NUMBER_RUN 单独处理
_ASCII_WORD = r"A-Za-z0-9_%\.\-"

# 中文连续串
_CJK_RUN = re.compile(f"[{_CJK}]+")
# 英文 / 数字 / 百分号 / 小数 / 连字符组成的整词
_ASCII_RUN = re.compile(f"[{_ASCII_WORD}]+")

# 金额类数字：支持 1,286,400.00 / 1286400 / 128.64
# 千分位分支必须排在纯数字之前，否则 "1,286,400" 会被拆成 "1"、"286"、"400"
_NUMBER_RUN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+")

# 全角标点 -> 半角空格；各类空白统一
# 全部映射成空格而不是删除：避免相邻词条被"粘"成一个不存在的词
_PUNCT_MAP = {
    "，": " ", "。": " ", "、": " ", "；": " ", "：": " ", "？": " ", "！": " ",
    "（": " ", "）": " ", "《": " ", "》": " ", "“": " ", "”": " ", "‘": " ",
    "’": " ", "【": " ", "】": " ", "—": " ", "…": " ", "·": " ",
    "\u3000": " ",
}


def normalize_text(text: str) -> str:
    """归一化：统一标点为空格、压缩空白、去零宽字符。

    参数：
        text：原始文本（可为 None / 空串，此处按假值处理）。
    返回值：
        str，已 strip 的归一化文本；空输入返回空串。
    副作用 / 异常：
        无副作用、不抛异常；每一步都返回新字符串。
    """
    if not text:
        return ""
    # 先干掉 BOM 与零宽空格：它们不可见但会污染 token，导致检索莫名失配
    out = text.replace("\ufeff", "").replace("\u200b", "")
    for src, dst in _PUNCT_MAP.items():
        out = out.replace(src, dst)
    # 行内连续空白压成单空格（不含 \n，换行结构在下一步单独处理）
    out = re.sub(r"[ \t\r\f\v]+", " ", out)
    # 最多保留一个空行：段落边界对切句有意义，再多就是噪声
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _ascii_tokens(text: str) -> List[str]:
    """抽取英文 / 数字整词并统一小写：ROE -> roe，1,286,400.00 -> 1286400.00。

    参数：
        text：已归一化文本（调用方保证；本函数不再归一化）。
    返回值：
        List[str]，按出现顺序排列的小写 token；空文本返回空列表。
    副作用 / 异常：
        无副作用、不抛异常。注：首尾的 . - _ % 会被 strip 掉
        （"1.5%." 这类边界情形只保留核心词），strip 后为空则丢弃。
    """
    tokens: List[str] = []
    for m in _ASCII_RUN.finditer(text):
        tok = m.group(0).strip(".-_%")
        if not tok:
            continue
        tokens.append(tok.lower())
    return tokens


def tokenize(text: str) -> List[str]:
    """把文本切成检索用的 token 列表（确定性、无外部依赖）。

    参数：
        text：待切分的原始文本（内部会先 normalize_text）。
    返回值：
        List[str]，顺序为「中文单字 + 中文二元组 + 英文/数字整词 + 金额整串」；
        空输入返回空列表。
    副作用 / 异常：
        无副作用、不抛异常。注：不移除重复 token——BM25 依赖词频，
        重复计数正是所需的信号；金额补 token 处才做了去重判断。
    """
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


# 零宽断言放在标点之后：这样切分时句末标点保留在前一句里，不会被丢掉
_SENT_END = re.compile(r"(?<=[。！？；])")


def split_sentences(text: str) -> List[str]:
    """按中文句末标点切句，保留标点。用于子块切分与摘要抽取。

    参数：
        text：原始文本（内部会先 normalize_text）。
    返回值：
        List[str]，已 strip 且非空的句子列表；空输入返回空列表。
    副作用 / 异常：
        无副作用、不抛异常。注：归一化会把换行统一为段落分隔，
        本函数按行分别切句，因此换行天然也是句子边界。
    """
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
    """取首句做标题 / 摘要，过长则截断。

    参数：
        text：原始文本。
        limit：单句最大字符数，超过则截断并补省略号。
    返回值：
        str，首句；无有效句子时返回空串。
    副作用 / 异常：
        无副作用、不抛异常。用于标题生成，不参与检索打分。
    """
    sents = split_sentences(text)
    if not sents:
        return ""
    head = sents[0]
    return head if len(head) <= limit else head[:limit] + "…"
