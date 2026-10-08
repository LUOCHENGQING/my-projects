"""清洗与质量评估。

本模块是摄取层（ingest）的第二道工序，位置在**解析之后、切分之前**：

    loader.parse → loader.apply_masking → 【cleaning.clean_*】 → chunking → index → retrieve

输入是 `loader.SourceDocument`（章节 → 块 → 表格 rows），输出是**就地清洗后的同一批
文档** + 逐篇的 `QualityReport`。解析层只保证结构正确，脏字符与模板噪音在这里处理。

金融资料的「脏」有固定套路，这里逐条对着处理：

| 脏法 | 例子 | 处理 |
| --- | --- | --- |
| 页眉页脚重复 | 每页顶部「XX银行 内部资料 第 3 页」 | 按行频统计，跨章节高频重复行判为模板行剔除 |
| OCR 形近字 | 「己经」应为「已经」、「末来」应为「未来」、「O」应为「0」 | 形近字映射表 + 数字上下文里的字母归正 |
| 水印穿插 | 「机 密」「样 本」插在正文中间 | 短行且字符间空格异常的判为水印剔除 |
| 空白 / 零宽字符 | Word 转出来大量 \\u3000 | 统一归一化 |
| 表格串行 | CSV 导出后整行挤成一格 | 单元格 trim + 空行剔除 |

同时给出 **文档质量评分**：OCR 可疑字符比例、模板行占比、空章节比例。
质量分不是装饰——它决定了这块证据在重排阶段要不要被降权
（低质量来源出现在答案里，业务侧会直接质疑可信度）。

清洗口径（与代码一致）
----------------------
- **形近字纠正**：`OCR_CONFUSIONS` 只收 10 组「高置信、低副作用」的错字对，
  命中即全局字符串替换；「未/末」这类必须看上下文的**不在**表里。
- **数字上下文归正**：`_DIGIT_CONTEXT_FIX` 只在数字夹缝里把 `O/o→0`、`l/I→1`、
  `S/s→5`、`B/b→8`（形如 `1O0` 这类金额/比例/条款号）。
- **水印**：`_WATERMARK_RE` 命中「1~5 个汉字，每个字后面跟一个空白，结尾一个汉字」
  的整行（如 `机 密`）即整行丢弃。
- **模板行**：出现次数 ≥ `min_repeat`（默认 3）且长度 ≤ 40 的行，或
  长度 ≤ 24 且含 `_TEMPLATE_HINTS` 任一提示词、同时 `_looks_like_header()` 判为
  页眉特征（页码行，或不含 `。；：:` 的短行）的行。
- **保留标点**：清洗走 `normalize_whitespace`（面向展示），不是会把标点换成空格的
  `normalize_text`（面向分词）——产物会直接成为证据文本与答案原句。

文档质量分的计算依据（score 为 0~1，从 1.0 开始逐项扣分）
---------------------------------------------------------
| 扣分项 | 触发条件 | 扣分（本文件实际实现） |
| --- | --- | --- |
| 正文为空 | `total_chars == 0`（各章节 `char_count` 之和） | +1.0 |
| OCR 可疑 | `raw` 中形近词命中数 + 数字夹缝字母数，逐处 +0.02 | 上限 0.25 |
| 模板行 | 被判为模板行的行数，逐行 +0.01 | 上限 0.15 |
| 空章节 | `section.text.strip()` 为空的章节数，逐个 +0.02 | 上限 0.10 |
| 缺编号 | `meta` 里没有 `source_id` | +0.05 |
| OCR 来源 | `doc.fmt == "text"`（纯文本默认来自 OCR） | +0.05 |

最终 `score = max(0.0, min(1.0, 1.0 - penalty))`。该分数**不只用于报表**：
`src/engine.py` 把它整理成 `{source_id: score}` 传给 `chunking.build_chunks()`，
写进每个父块/子块的 `quality_score`，再随检索项进入 `retrieve/rerank.py` 的
`quality_of` 特征，参与重排降权。

对外关键对象
------------
    clean_text(text)                        单段文本清洗 → 清洗后文本
    strip_boilerplate(lines, min_repeat)    批量剔模板行 → (保留行, 模板行)
    clean_document(doc)                     就地清洗一篇 + 打分 → QualityReport
    clean_corpus(documents)                 批量清洗 → List[QualityReport]
    QualityReport                           质量体检结果（可 `to_dict()` 序列化）

调用方：`src/engine.py` 在建库时调用 `clean_corpus()`；`tests/` 直接在此层做断言。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from ..utils.text import normalize_whitespace, split_paragraphs
from .loader import SourceDocument

__all__ = ["OCR_CONFUSIONS", "clean_text", "strip_boilerplate", "score_quality", "clean_document", "QualityReport"]

# OCR 形近字混淆表（左：识别结果，右：更可能的本字）。
# 只处理**高置信、低副作用**的几组；像「未/末」这种必须看上下文的，交给上下文规则。
# 注：实际实现只有下列 10 组，且是**无条件全局替换**（不做上下文判断、不区分词性）。
OCR_CONFUSIONS: Dict[str, str] = {
    "己经": "已经",
    "末来": "未来",
    "帐号": "账号",
    "登陆": "登录",
    "收溢": "收益",
    "年俩": "年限",
    "风验": "风险",
    "期现": "期限",
    "份客": "份额",
    "担供": "提供",
}
# 数字上下文里被 OCR 认错的字母：金额/比例/条款号里不该出现 O、l、S
_DIGIT_CONTEXT_FIX = (
    (re.compile(r"(?<=\d)[Oo](?=\d)"), "0"),
    (re.compile(r"(?<=\d)[lI](?=\d)"), "1"),
    (re.compile(r"(?<=\d)[Ss](?=\d)"), "5"),
    (re.compile(r"(?<=\d)[Bb](?=\d)"), "8"),
)
# 水印：短行且字符之间被插入空格，如「机 密」「样 本 件」
_WATERMARK_RE = re.compile(r"^(?:[\u4e00-\u9fff]\s){1,5}[\u4e00-\u9fff]$")
# 页码 / 页眉页脚：包含页码或典型模板词的短行
_PAGE_RE = re.compile(r"^(第\s*\d+\s*页(?:\s*[/共]\s*\d+\s*页)?|[-—\s]*\d{1,3}[-—\s]*)$")
# 模板提示词：短行里出现任一个即可能是页眉/水印（还需通过 _looks_like_header 复核）
_TEMPLATE_HINTS = ("内部资料", "机密", "保密", "请勿外传", "样本", "仅供", "版权所有", "翻版必究")


@dataclass
class QualityReport:
    """单篇文档的质量体检结果。

    职责：承载 `score_quality()` 的五项扣分证据，供治理报表与重排降权使用。
    关键属性：
        source_id          资料编号（与 `SourceDocument.source_id` 对齐，作为降权键）
        score              0~1，越大越干净（默认 1.0，被扣分项逐步下调）
        ocr_suspicious     `raw` 中命中的 OCR 可疑点数（形近词 + 数字夹缝字母）
        boilerplate_lines  被判为模板行的行数
        empty_sections     纯文本为空的章节数
        total_chars        各章节 `char_count` 之和（注意：表格文本也计入）
        issues             触发的扣分原因清单（中文短语，逐项追加）
    """

    source_id: str
    score: float = 1.0                     # 0~1，越大越干净
    ocr_suspicious: int = 0                # OCR 可疑字符数
    boilerplate_lines: int = 0             # 被判为模板行的行数
    empty_sections: int = 0                # 空章节数
    total_chars: int = 0
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """拍平成字典供报表/API 输出；`score` 保留 4 位小数，`issues` 复制为 list。"""
        return {
            "source_id": self.source_id,
            "score": round(self.score, 4),
            "ocr_suspicious": self.ocr_suspicious,
            "boilerplate_lines": self.boilerplate_lines,
            "empty_sections": self.empty_sections,
            "total_chars": self.total_chars,
            "issues": list(self.issues),
        }


def clean_text(text: str) -> str:
    """单段文本清洗：空白归一 → OCR 形近字纠正 → 数字上下文归正 → 去水印行。

    参数：text — 单段/单单元格文本（空串这类假值直接返回空串）。

    返回：清洗后的文本；每行已 `strip()`，连续 3 行以上空行压缩为 1 个空行，
          首尾空白去掉。**标点保留**。

    副作用 / 异常：纯函数；水印行（`_WATERMARK_RE` 整行命中）被整行丢弃，
          其余行即使为空也保留在结果里参与空行合并。

    刻意**保留标点与段落空行**：这个函数的产物会成为子块正文、证据文本与答案原句，
    标点被抹掉会让业务人员觉得"这系统把原文改了"。
    """
    if not text:
        return ""
    out = normalize_whitespace(text)

    # 先做字级纠正再去行级水印判定：水印判定依赖「字间空格」形态，必须先归一空白
    for wrong, right in OCR_CONFUSIONS.items():
        if wrong in out:
            out = out.replace(wrong, right)
    for pattern, digit in _DIGIT_CONTEXT_FIX:
        out = pattern.sub(digit, out)

    kept: List[str] = []
    for line in out.split("\n"):
        stripped = line.strip()
        if stripped and _WATERMARK_RE.match(stripped):
            continue                                  # 整行是水印 → 丢弃，不做替换
        kept.append(stripped)
    result = "\n".join(kept)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def strip_boilerplate(lines: Sequence[str], min_repeat: int = 3) -> Tuple[List[str], List[str]]:
    """剔除模板行：出现次数 >= min_repeat 且长度 <= 40 的行判为页眉页脚 / 水印。

    参数：
        lines      待处理的行序列（各行的首尾空白在内部自行 strip）
        min_repeat 判为「高频重复」的最小出现次数，默认 3

    返回：
        `(kept, boiler)`：`kept` 为保留行（**已 strip、空行被丢弃**，顺序不变），
        `boiler` 为被判为模板的行（保留出现顺序，可能含重复行）。

    第二条判定口径（实际实现）：长度 ≤ 24、含 `_TEMPLATE_HINTS` 任一提示词、
    且 `_looks_like_header()` 为真（页码行，或不含 `。；：:` 的短行）也判为模板行。
    调用方注意：返回值是**重新组装**的行列表，不保留原始行对象或行号。

    副作用 / 异常：无。

    短行条件很重要——长段落偶然重复
    （比如反复出现的免责声明条款正文）不应被删掉。
    """
    # 计数基于 strip 后的行；空行不参与计数，避免大量空行把阈值顶满
    counter = Counter(line.strip() for line in lines if line.strip())
    boiler: List[str] = []
    kept: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if counter[stripped] >= min_repeat and len(stripped) <= 40:
            boiler.append(stripped)
            continue
        if len(stripped) <= 24 and any(hint in stripped for hint in _TEMPLATE_HINTS) and _looks_like_header(stripped):
            boiler.append(stripped)
            continue
        kept.append(stripped)
    return kept, boiler


def _looks_like_header(line: str) -> bool:
    """判断一行是否具备页眉/水印的形态特征（供 `strip_boilerplate` 的提示词规则复核）。

    参数：line — 已 strip 的单行文本。
    返回：`True` 表示像页眉页脚——命中 `_PAGE_RE`（「第 3 页」「- 12 -」等页码行），
          或者整行不含 `。；：:` 任何一个（正文说明句通常带标点）。

    副作用 / 异常：无。
    """
    if _PAGE_RE.match(line):
        return True
    return not any(p in line for p in "。；：:")


def _ocr_suspicious_count(text: str) -> int:
    """统计 OCR 可疑点：形近词命中数 + 数字上下文里的字母。

    参数：text — 待体检的文本（`score_quality` 传入的是 `doc.raw`，即**清洗前**原文）。
    返回：形近词在文本中的出现次数（可重叠计数，按 `str.count`）加上
          `_DIGIT_CONTEXT_FIX` 各正则的匹配个数。

    注意（实际实现）：统计的是「错字残留量」，因此必须在清洗前统计；清洗后再统计会
    恒为 0，质量分也就失去意义。
    """
    hits = sum(text.count(wrong) for wrong in OCR_CONFUSIONS)
    for pattern, _ in _DIGIT_CONTEXT_FIX:
        hits += len(pattern.findall(text))
    return hits


def score_quality(doc: SourceDocument) -> QualityReport:
    """给一篇文档打质量分，并记录发现的问题。

    参数：doc — 待体检的文档（只读，**不修改** `doc` 的任何字段）。

    返回：`QualityReport`，`score` 由 1.0 依次扣减以下项后夹逼到 [0, 1]：
          正文为空 +1.0；OCR 可疑 `min(0.25, 命中数*0.02)`；
          模板行 `min(0.15, 行数*0.01)`；空章节 `min(0.10, 个数*0.02)`；
          缺 `meta["source_id"]` +0.05；`fmt == "text"`（OCR 来源）+0.05。

    统计口径（实际实现）：
        - `total_chars` = 各章节 `char_count` 之和（含表格渲染出的文本）；
        - 模板行统计只取**段落块**（`split_paragraphs(block.text)`），表格块不参与；
        - OCR 可疑点在 `doc.raw` 上统计，因此是清洗前口径。

    副作用 / 异常：无（`issues` 只写进新建的 `report`，不写回 `doc.issues`；
    `clean_document()` 才负责把 issues 合并回文档）。
    """
    all_lines: List[str] = []
    for section in doc.sections:
        for block in section.blocks:
            all_lines.extend(split_paragraphs(block.text) if not block.is_table else [])
    total_chars = sum(s.char_count for s in doc.sections)
    empty_sections = sum(1 for s in doc.sections if not s.text.strip())
    _, boiler = strip_boilerplate(all_lines)
    ocr_hits = _ocr_suspicious_count(doc.raw)

    report = QualityReport(
        source_id=doc.source_id,
        total_chars=total_chars,
        ocr_suspicious=ocr_hits,
        boilerplate_lines=len(boiler),
        empty_sections=empty_sections,
    )

    penalty = 0.0
    if total_chars == 0:
        penalty += 1.0                     # 直接归零：没有正文的文档没有降权余地
        report.issues.append("正文为空")
    if ocr_hits:
        penalty += min(0.25, ocr_hits * 0.02)
        report.issues.append(f"OCR 可疑字符 {ocr_hits} 处")
    if boiler:
        penalty += min(0.15, len(boiler) * 0.01)
        report.issues.append(f"模板行 {len(boiler)} 行")
    if empty_sections:
        penalty += min(0.10, empty_sections * 0.02)
        report.issues.append(f"空章节 {empty_sections} 个")
    if not doc.meta.get("source_id"):
        penalty += 0.05                    # 无编号会影响引用与去重，轻罚
        report.issues.append("缺少资料编号")
    if doc.fmt == "text":
        # 纯文本默认来自 OCR，一致性风险更高，轻微降权
        penalty += 0.05
        report.issues.append("来源为 OCR 文本")

    report.score = max(0.0, min(1.0, 1.0 - penalty))
    return report


def clean_document(doc: SourceDocument) -> QualityReport:
    """就地清洗一篇文档（正文与全部块），并返回质量报告。

    参数：doc — 待清洗文档，函数**直接改写**其内容字段（无返回值携带清洗结果）。

    返回：`score_quality(doc)` 算出的 `QualityReport`。

    副作用（就地修改，顺序有意义）：
        - 表格块：逐单元格 `clean_text()`，并把整行全空的行删掉，随后用清理后的
          rows **重建 Markdown 原文**（`| a | b |` 形式），列数不齐时按现有单元格输出；
        - 段落块：先 `split_paragraphs` 切段并 `strip_boilerplate` 剔模板行，
          再把保留行逐行 `clean_text()` 后拼回 `block.text`；
        - `doc.raw` 也整体过一遍 `clean_text()`；
        - 有剔除模板行时追加一条 issue（形如 `剔除模板行 3 行`）；
        - 最后把质量报告里的 issues 去重合并进 `doc.issues`（已有同文案则不重复追加）。
    """
    boiled = 0
    for section in doc.sections:
        for block in section.blocks:
            if block.is_table:
                block.rows = [
                    [clean_text(cell) for cell in row]
                    for row in block.rows
                    if any(str(cell).strip() for cell in row)
                ]
                # rows 变了，Markdown 原文必须同步重建，否则 render_table_rows 与
                # block.text 会各说各话（下游两者都会被用到）
                block.text = "\n".join("| " + " | ".join(r) + " |" for r in block.rows)
                continue
            lines, boiler = strip_boilerplate(split_paragraphs(block.text))
            boiled += len(boiler)
            block.text = "\n".join(clean_text(line) for line in lines)
    doc.raw = clean_text(doc.raw)
    if boiled:
        doc.issues.append(f"剔除模板行 {boiled} 行")

    report = score_quality(doc)
    for msg in report.issues:
        if msg not in doc.issues:          # 去重：同一原因不重复进轨迹
            doc.issues.append(msg)
    return report


def clean_corpus(documents: Iterable[SourceDocument]) -> List[QualityReport]:
    """批量清洗，返回每篇文档的质量报告。

    参数：documents — 可迭代的 `SourceDocument`（通常是 `corpus.documents`）。
    返回：与输入**同序**的 `QualityReport` 列表（逐篇 `clean_document()` 的结果）。
    副作用：原地清洗每一篇文档（见 `clean_document`）。
    """
    return [clean_document(doc) for doc in documents]
