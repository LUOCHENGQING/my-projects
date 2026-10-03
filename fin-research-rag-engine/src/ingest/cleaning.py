"""清洗与质量评估。

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
_TEMPLATE_HINTS = ("内部资料", "机密", "保密", "请勿外传", "样本", "仅供", "版权所有", "翻版必究")


@dataclass
class QualityReport:
    """单篇文档的质量体检结果。"""

    source_id: str
    score: float = 1.0                     # 0~1，越大越干净
    ocr_suspicious: int = 0                # OCR 可疑字符数
    boilerplate_lines: int = 0             # 被判为模板行的行数
    empty_sections: int = 0                # 空章节数
    total_chars: int = 0
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
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

    刻意**保留标点与段落空行**：这个函数的产物会成为子块正文、证据文本与答案原句，
    标点被抹掉会让业务人员觉得"这系统把原文改了"。
    """
    if not text:
        return ""
    out = normalize_whitespace(text)

    for wrong, right in OCR_CONFUSIONS.items():
        if wrong in out:
            out = out.replace(wrong, right)
    for pattern, digit in _DIGIT_CONTEXT_FIX:
        out = pattern.sub(digit, out)

    kept: List[str] = []
    for line in out.split("\n"):
        stripped = line.strip()
        if stripped and _WATERMARK_RE.match(stripped):
            continue
        kept.append(stripped)
    result = "\n".join(kept)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def strip_boilerplate(lines: Sequence[str], min_repeat: int = 3) -> Tuple[List[str], List[str]]:
    """剔除模板行：出现次数 >= min_repeat 且长度 <= 40 的行判为页眉页脚 / 水印。

    返回 (保留行, 被判为模板的行)。短行条件很重要——长段落偶然重复
    （比如反复出现的免责声明条款正文）不应被删掉。
    """
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
    """模板行特征：页码行，或者「无句末标点且不含冒号」的短行（正文说明句一般都带标点）。"""
    if _PAGE_RE.match(line):
        return True
    return not any(p in line for p in "。；：:")


def _ocr_suspicious_count(text: str) -> int:
    """统计 OCR 可疑点：形近词命中数 + 数字上下文里的字母。"""
    hits = sum(text.count(wrong) for wrong in OCR_CONFUSIONS)
    for pattern, _ in _DIGIT_CONTEXT_FIX:
        hits += len(pattern.findall(text))
    return hits


def score_quality(doc: SourceDocument) -> QualityReport:
    """给一篇文档打质量分，并记录发现的问题。"""
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
        penalty += 1.0
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
        penalty += 0.05
        report.issues.append("缺少资料编号")
    if doc.fmt == "text":
        # 纯文本默认来自 OCR，一致性风险更高，轻微降权
        penalty += 0.05
        report.issues.append("来源为 OCR 文本")

    report.score = max(0.0, min(1.0, 1.0 - penalty))
    return report


def clean_document(doc: SourceDocument) -> QualityReport:
    """就地清洗一篇文档（正文与全部块），并返回质量报告。"""
    boiled = 0
    for section in doc.sections:
        for block in section.blocks:
            if block.is_table:
                block.rows = [
                    [clean_text(cell) for cell in row]
                    for row in block.rows
                    if any(str(cell).strip() for cell in row)
                ]
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
        if msg not in doc.issues:
            doc.issues.append(msg)
    return report


def clean_corpus(documents: Iterable[SourceDocument]) -> List[QualityReport]:
    """批量清洗，返回每篇文档的质量报告。"""
    return [clean_document(doc) for doc in documents]
