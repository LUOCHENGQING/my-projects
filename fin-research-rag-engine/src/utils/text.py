"""文本处理工具：中文分词、句子切分、归一化、脱敏、相似度。

分词方案沿用「中文单字 + 相邻二元组 + 英文/数字整词」的确定性方案，不依赖 jieba：
    * 单字保证召回（"净利率" 与 "利润" 都能命中）
    * 二元组提供短语级区分度（"净利" / "利润" / "净利率"）
    * 英文与数字（ROE、R2、2024、1,286,400.00、产品代码）按整词保留

金融资料的特有处理：
    * 条款号（第四十二条 / 12.3.1）必须整串保留，否则会被切成无意义的单字；
    * 产品代码（如 WY2024-01）必须整串保留，否则 BM25 无法精确命中；
    * 身份证 / 手机号 / 银行账号在做检索前要脱敏，但又不能把「长度信息」也抹掉
      （否则「账号长度为 19 位」这类规则叙述会对不上）。
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List, Sequence, Set

__all__ = [
    "normalize_text",
    "normalize_whitespace",
    "tokenize",
    "split_sentences",
    "split_paragraphs",
    "first_sentence",
    "shingles",
    "jaccard",
    "mask_sensitive",
    "amount_tokens",
    "stable_hash",
    "truncate",
    "digest",
    "token_counts",
]

_CJK = r"\u4e00-\u9fff"
_ASCII_WORD = r"A-Za-z0-9_%\.\-"

_CJK_RUN = re.compile(f"[{_CJK}]+")
_ASCII_RUN = re.compile(f"[{_ASCII_WORD}]+")
_NUMBER_RUN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+")
# 条款号：第四十二条 / 第 12 条 / 12.3.1 / （三）
_CLAUSE_RUN = re.compile(r"第\s*[0-9一二三四五六七八九十百零]+\s*条(?:之[一二三四五六七八九十])?")
_DOTTED_CLAUSE = re.compile(r"\b\d+(?:\.\d+){1,3}\b")
# 产品 / 制度代码：字母开头，含数字与连字符，如 WY2024-01、POL-2024-07
_CODE_RUN = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*(?:-\d{2,})+\b")

# 全角标点 -> 空格
_PUNCT_MAP = {
    "，": " ", "。": " ", "、": " ", "；": " ", "：": " ", "？": " ", "！": " ",
    "（": " ", "）": " ", "《": " ", "》": " ", "“": " ", "”": " ", "‘": " ",
    "’": " ", "【": " ", "】": " ", "—": " ", "…": " ", "·": " ", "\u3000": " ",
}

# 脱敏规则：命中即替换成带长度占位的掩码，保留「这是一类什么字段」的信息
_MASK_RULES: Sequence[tuple[str, re.Pattern[str]]] = (
    ("ID", re.compile(r"\b\d{17}[\dXx]\b")),
    ("PHONE", re.compile(r"\b1[3-9]\d{9}\b")),
    ("BANK", re.compile(r"\b\d{16,19}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
)


def normalize_text(text: str) -> str:
    """归一化（**面向分词**）：统一全角标点为空格、压缩空白、去零宽字符与 BOM。

    注意：这个函数会把标点替换成空格，因此**不能用于生成要展示给用户的文本**
    （证据、答案、出处）。需要保留标点时用 `normalize_whitespace()`。
    """
    if not text:
        return ""
    out = text.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    for src, dst in _PUNCT_MAP.items():
        out = out.replace(src, dst)
    out = re.sub(r"[ \t\r\f\v]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def normalize_whitespace(text: str) -> str:
    """归一化（**面向展示**）：只处理空白与不可见字符，**保留全部标点**。

    为什么会需要两个归一化函数：分词时必须把标点当分隔符，而证据与答案必须原样
    保留标点（「C2 客户仅可购买 R1、R2 产品。」和「C2 客户仅可购买 R1 R2 产品」
    在业务人员眼里是两回事）。两者混用会让答案读起来像被洗过一遍。
    """
    if not text:
        return ""
    out = text.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    out = out.replace("\u3000", " ")
    out = re.sub(r"[ \t\r\f\v]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def mask_sensitive(text: str) -> str:
    """对身份证 / 手机号 / 银行卡号 / 邮箱做脱敏。

    脱敏发生在**入库之前**，因此下游索引、日志、轨迹里都不会出现原始敏感串。
    掩码保留字段类型与后四位，便于人工核对时确认「是不是同一个人」。
    """
    if not text:
        return ""
    out = text
    for label, pattern in _MASK_RULES:
        def _repl(m: re.Match[str], _label: str = label) -> str:
            raw = m.group(0)
            tail = raw[-4:] if len(raw) > 4 else raw
            return f"[{_label}:****{tail}]"

        out = pattern.sub(_repl, out)
    return out


def _ascii_tokens(text: str) -> List[str]:
    """抽取英文 / 数字整词并统一小写：ROE -> roe，1,286,400.00 -> 1286400.00。"""
    tokens: List[str] = []
    for m in _ASCII_RUN.finditer(text):
        tok = m.group(0).strip(".-_%")
        if tok:
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
        tokens.extend(list(run))
        if len(run) >= 2:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))

    tokens.extend(_ascii_tokens(norm))

    for m in _NUMBER_RUN.finditer(norm):
        joined = m.group(0).replace(",", "")
        if joined not in tokens:
            tokens.append(joined)
            if "." in joined:
                integer_part = joined.split(".", 1)[0]
                if integer_part not in tokens:
                    tokens.append(integer_part)

    # 条款号与产品代码整串补进 token，保证「第四十二条」「WY2024-01」可被精确命中。
    # 前面按标点归一化后书名号已变成空格，但「第…条」本身不含标点，仍可匹配。
    for m in _CLAUSE_RUN.finditer(norm):
        tok = m.group(0).replace(" ", "")
        if tok not in tokens:
            tokens.append(tok)
    for m in _DOTTED_CLAUSE.finditer(norm):
        tok = m.group(0)
        if tok not in tokens:
            tokens.append(tok)
    for m in _CODE_RUN.finditer(normalize_text(text).upper()):
        tok = m.group(0).lower()
        if tok not in tokens:
            tokens.append(tok)

    return tokens


_SENT_END = re.compile(r"(?<=[。！？；!?;])")


def split_sentences(text: str) -> List[str]:
    """按中英文句末标点切句，**保留原文标点与措辞**。

    刻意不做 normalize：这个函数的产物会直接进入子块文本与最终答案，
    把「。」「，」替换成空格会让证据和答案读起来像被洗过一遍。
    需要归一化的地方（分词、金额抽取）会各自调用 normalize_text()。
    """
    if not text:
        return []
    parts: List[str] = []
    for line in text.split("\n"):
        line = line.replace("\ufeff", "").replace("\u200b", "").strip()
        if not line:
            continue
        parts.extend(c.strip() for c in _SENT_END.split(line) if c.strip())
    return parts


def split_paragraphs(text: str) -> List[str]:
    """按空行 / 换行切段，去掉空段。"""
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]


def first_sentence(text: str, limit: int = 80) -> str:
    """取首句做标题 / 摘要，过长则截断。"""
    sents = split_sentences(text)
    if not sents:
        return ""
    head = sents[0]
    return head if len(head) <= limit else head[:limit] + "…"


def truncate(text: str, limit: int) -> str:
    """按字符数截断并加省略号，用于轨迹摘要而不是正文。"""
    if text is None:
        return ""
    return text if len(text) <= limit else text[:limit] + "…"


def shingles(tokens: Iterable[str], n: int = 2) -> Set[str]:
    """把 token 序列转成 n-gram 集合，用于近似去重与相似度。"""
    seq = list(tokens)
    if len(seq) < n:
        return set(seq)
    return {"".join(seq[i : i + n]) for i in range(len(seq) - n + 1)}


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """集合 Jaccard 相似度；任一为空返回 0.0（而不是 1.0，避免空块被判为重复）。"""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def amount_tokens(text: str) -> List[str]:
    """抽取文本中的金额 / 比例数字（去掉千分位），供答案忠实度校验使用。"""
    out: List[str] = []
    for m in _NUMBER_RUN.finditer(normalize_text(text)):
        tok = m.group(0).replace(",", "")
        if tok not in out:
            out.append(tok)
    return out


def stable_hash(text: str, size: int = 16) -> str:
    """跨进程稳定的短哈希。刻意不用内置 hash()（带 PYTHONHASHSEED 随机化）。"""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=size).hexdigest()


def digest(text: str, limit: int = 120) -> str:
    """轨迹里的 input_digest / output_digest：短哈希 + 截断预览，既能比对又不泄露全文。"""
    h = stable_hash(text or "", size=6)
    return f"{h}:{truncate((text or '').replace(chr(10), ' '), limit)}"


def token_counts(text: str) -> Dict[str, int]:
    """词频统计，供稀疏向量的权重计算使用。"""
    counts: Dict[str, int] = {}
    for tok in tokenize(text):
        counts[tok] = counts.get(tok, 0) + 1
    return counts
