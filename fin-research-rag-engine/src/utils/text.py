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

在 RAG 全链路中的位置
--------------------
    ingest（脱敏 / 归一化） → chunking（切句） → index（分词建索引） → retrieve（分词查询）
    → answer（数字抽取 / 句子支撑度） → cache（稳定哈希） → tracing（摘要哈希）

被谁调用（关键几处）：
    `ingest/loader.py` 的 `apply_masking()` → `mask_sensitive()`、`normalize_text()`；
    `ingest/cleaning.py` → `normalize_whitespace()`、`split_paragraphs()`；
    `chunking/parent_child.py` → `split_sentences()`；
    `index/bm25.py`、`index/embedding.py` → `tokenize()`（倒排与稀疏表示**共用同一分词口径**）；
    `retrieve/hybrid.py`、`retrieve/rerank.py`、`src/faq.py` → `tokenize()`、`shingles()`、`jaccard()`；
    `answer/faithfulness.py` → `amount_tokens()`（数字忠实度）、`tokenize()`（句子支撑率）；
    `answer/generator.py` → `split_sentences()`、`tokenize()`（抽取式作答）；
    `cache/redis_cache.py` → `stable_hash()`（缓存键）；`tracing.py` → `digest()`。
注：实际实现里 `first_sentence()` 与 `token_counts()` 在 `src/` 内**没有调用方**
（只经 `utils/__init__.py` 导出，供外部或测试使用）；`truncate()` 只被本模块的 `digest()` 调用。

两条不许破的规矩
--------------
    1) **不要用 `normalize_text()` 生成给用户看的文本**——它会把标点换成空格，
       展示层要用 `normalize_whitespace()` 或原样 `split_sentences()`；
    2) **不要替换成更"聪明"的分词**（jieba / 模型分词）：分词口径一变，
       BM25 倒排、稀疏向量、缓存键与全部历史评测指标同时失效且不可比。

所有函数都是纯函数：不读配置、不联网、不写盘、**不抛业务异常**（除类型错误外无异常路径）。
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

# 字符类与正则：全部**模块级预编译**，避免在每次分词调用里重复编译（分词是热路径）。
_CJK = r"\u4e00-\u9fff"
_ASCII_WORD = r"A-Za-z0-9_%\.\-"

_CJK_RUN = re.compile(f"[{_CJK}]+")
_ASCII_RUN = re.compile(f"[{_ASCII_WORD}]+")
# 金额三种写法：带千分位、带小数、纯整数——`amount_tokens()` 的忠实度校验就靠它
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
# 顺序有意义：先认身份证（18 位）再认银行卡（16~19 位），否则 18 位身份证会被当成卡号。
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

    参数：text 原始文本（None / 空串直接返回 ""）。
    返回：str —— 依次去掉 BOM 与零宽字符、把不换行空格 `\xa0` 变普通空格、
          按 `_PUNCT_MAP` 把全角标点变空格、把连续横向空白压成一个空格、
          把 3 个以上换行压成两个换行，最后 strip。
    副作用/异常：无（纯字符串处理）。
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

    参数：text 原始文本（None / 空串返回 ""）。
    返回：str —— 与 `normalize_text()` 的差别只有一处：**不做标点替换**，
          额外单独把全角空格 `\u3000` 变成普通空格。
    副作用/异常：无。
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

    参数：text 原始文本（None / 空串返回 ""）。
    返回：str —— 按 `_MASK_RULES` 顺序把身份证 / 手机号 / 银行卡号 / 邮箱替换成
          `[ID:****1234]` 这类掩码（标签 + 四个星号 + 原串后四位；原串长度 ≤ 4 时保留整串）。
    副作用/异常：无；正则按顺序依次替换，前者命中后真正的原串已不存在，不会被后者再匹配。
    注：实际只做"格式识别"，不做校验位验证——假身份证号同样会被掩码，这符合"宁可多掩"的取舍。
    """
    if not text:
        return ""
    out = text
    for label, pattern in _MASK_RULES:
        def _repl(m: re.Match[str], _label: str = label) -> str:
            """替换回调：把命中的敏感片段替换成 [标签:****后四位] 的掩码形式。"""
            raw = m.group(0)
            tail = raw[-4:] if len(raw) > 4 else raw
            return f"[{_label}:****{tail}]"

        out = pattern.sub(_repl, out)
    return out


def _ascii_tokens(text: str) -> List[str]:
    """抽取英文 / 数字整词并统一小写：ROE -> roe，1,286,400.00 -> 1286400.00。

    参数：text 已归一化的文本（调用方保证）。
    返回：List[str] —— 按出现顺序的 token；每个 token 先 `strip(".-_%")` 去掉首尾的连接符与百分号，
          再 `lower()`（**大小写不敏感**，让 ROE 与 roe 命中同一个倒排项）。
    副作用/异常：无；空 token 被跳过。
    """
    tokens: List[str] = []
    for m in _ASCII_RUN.finditer(text):
        tok = m.group(0).strip(".-_%")
        if tok:
            tokens.append(tok.lower())
    return tokens


def tokenize(text: str) -> List[str]:
    """把文本切成检索用的 token 列表（确定性、无外部依赖）。

    输出顺序（也是"召回为什么够"的答案）：
        1. 每个中文连续段：**单字**（保召回）+ 相邻**二元组**（给短语区分度）；
        2. 英文/数字整词小写；
        3. 数字：去千分位后入列；带小数点的再补一份整数部分（`128.60` → `128.60`、`128`）；
        4. 条款号整串（`第四十二条`，去内部空格）与点分条款号（`12.3.1`）；
        5. 产品/制度代码（`WY2024-01`，取自大写化文本后转小写）。
    允许重复：ASCII 整词是无条件追加的，同一个英文词出现几次就入列几次
    （BM25 与稀疏向量按词频统计，重复正是"权重"的来源）；中文单字与二元组也会随出现次数重复。

    参数：text 原始文本（None / 空串返回空列表）。
    返回：List[str] —— token 序列，可能与入参长度无关；同一输入永远得到同一输出（可回归）。
    副作用/异常：无；内部只调用 `normalize_text()`（**不改入参**）。
    """
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


# 句末标点切分：**零宽断言**（lookbehind）保证标点本身留在前一句里，不会被吃掉。
_SENT_END = re.compile(r"(?<=[。！？；!?;])")


def split_sentences(text: str) -> List[str]:
    """按中英文句末标点切句，**保留原文标点与措辞**。

    刻意不做 normalize：这个函数的产物会直接进入子块文本与最终答案，
    把「。」「，」替换成空格会让证据和答案读起来像被洗过一遍。
    需要归一化的地方（分词、金额抽取）会各自调用 normalize_text()。

    参数：text 原始文本（None / 空串返回空列表）。
    返回：List[str] —— 先按换行拆行并去掉 BOM/零宽字符（空行丢弃），再用零宽断言按
          `。！？；!?;` 切句，每句 strip 后只保留非空项；**不切逗号与顿号**（否则条款会被切碎）。
    副作用/异常：无；不修改入参。
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
    """按空行 / 换行切段，去掉空段。

    参数：text 原始文本（None / 空串返回空列表）。
    返回：List[str] —— 每个段落 strip 后的列表；连续空行与单换行都会被切开。
    副作用/异常：无。
    """
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]


def first_sentence(text: str, limit: int = 80) -> str:
    """取首句做标题 / 摘要，过长则截断。

    参数：text 原始文本；limit 字符上限，默认 80。
    返回：str —— 首句；超过 limit 时截断并加「…」；切不出句子时返回 ""。
    副作用/异常：无。
    """
    sents = split_sentences(text)
    if not sents:
        return ""
    head = sents[0]
    return head if len(head) <= limit else head[:limit] + "…"


def truncate(text: str, limit: int) -> str:
    """按字符数截断并加省略号，用于轨迹摘要而不是正文。

    参数：text 原始文本（None 返回 ""）；limit 字符上限。
    返回：str —— 未超限原样返回；超限则取前 limit 个字符并加「…」。
    副作用/异常：无。
    """
    if text is None:
        return ""
    return text if len(text) <= limit else text[:limit] + "…"


def shingles(tokens: Iterable[str], n: int = 2) -> Set[str]:
    """把 token 序列转成 n-gram 集合，用于近似去重与相似度。

    参数：tokens 已分词的 token 序列（内部转 list）；n 窗口大小，默认 2。
    返回：Set[str] —— 相邻 n 个 token 直接拼接（**不加分隔符**，与 `faq.py`、去重逻辑的口径一致）；
          当序列长度 < n 时**退化为原始 token 集合**（而不是空集，避免短句被当成"永不重复"）。
    副作用/异常：无。
    """
    seq = list(tokens)
    if len(seq) < n:
        return set(seq)
    return {"".join(seq[i : i + n]) for i in range(len(seq) - n + 1)}


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """集合 Jaccard 相似度；任一为空返回 0.0（而不是 1.0，避免空块被判为重复）。

    参数：a / b 任意可迭代对象（内部转 set，因此重复项按一次算）。
    返回：float —— `交集大小 / 并集大小`；任一侧为空时返回 0.0。
    用在 `retrieve/hybrid.py` 的去重（`>= DEDUP_JACCARD` 判为重复）、`retrieve/rerank.py` 的短语分
    与 `src/faq.py` 的二元组相似度上："空即 0" 能避免空块被判成"和谁都重复"。
    副作用/异常：无。
    """
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def amount_tokens(text: str) -> List[str]:
    """抽取文本中的金额 / 比例数字（去掉千分位），供答案忠实度校验使用。

    参数：text 待抽取文本（内部先 `normalize_text()`，因此全角标点不影响数字识别）。
    返回：List[str] —— 按出现顺序、**去重**的数字串（`1,286,400.00` → `1286400.00`）。
    副作用/异常：无。
    注：只认数字形态，不区分金额 / 比例 / 期限；`answer/faithfulness.check_numbers()` 会另外
    放行 1~2 位纯数字与 19xx / 20xx 年份——口径在那边，不在本函数。
    """
    out: List[str] = []
    for m in _NUMBER_RUN.finditer(normalize_text(text)):
        tok = m.group(0).replace(",", "")
        if tok not in out:
            out.append(tok)
    return out


def stable_hash(text: str, size: int = 16) -> str:
    """跨进程稳定的短哈希。刻意不用内置 hash()（带 PYTHONHASHSEED 随机化）。

    参数：text 待哈希文本（UTF-8 编码）；size blake2b 摘要字节数，默认 16。
    返回：str —— 十六进制摘要，长度 = `size * 2` 个字符（默认 32 位十六进制）。
    副作用/异常：无；空串也能正常哈希（得到固定的十六进制值）。
    注：这是**缓存键的唯一哈希来源**（`cache.redis_cache.cache_key`），换算法会让旧缓存全部失效。
    """
    return hashlib.blake2b(text.encode("utf-8"), digest_size=size).hexdigest()


def digest(text: str, limit: int = 120) -> str:
    """轨迹里的 input_digest / output_digest：短哈希 + 截断预览，既能比对又不泄露全文。

    参数：text 待摘要文本（None 按空串）；limit 预览部分的字符上限，默认 120。
    返回：str —— `"{6 字节哈希}:{单行化并截断的预览}"`；换行会被替换成空格，保证一行落盘。
    副作用/异常：无。
    注：哈希部分固定用 `stable_hash(text, size=6)`（12 位十六进制），比缓存键短，便于人眼比对。
    """
    h = stable_hash(text or "", size=6)
    return f"{h}:{truncate((text or '').replace(chr(10), ' '), limit)}"


def token_counts(text: str) -> Dict[str, int]:
    """词频统计，供稀疏向量的权重计算使用。

    参数：text 原始文本。
    返回：Dict[str, int] —— token → 出现次数；**保留重复计数**（与 `tokenize()` 的输出一一累加）。
    副作用/异常：无。
    """
    counts: Dict[str, int] = {}
    for tok in tokenize(text):
        counts[tok] = counts.get(tok, 0) + 1
    return counts
