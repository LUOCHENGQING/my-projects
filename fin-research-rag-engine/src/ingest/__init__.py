"""解析清洗子包。

    loader      多源文档解析（md / txt(OCR) / csv / json）+ 字段映射 + 表格结构化
    cleaning    文本清洗（形近字、模板行、水印）+ 文档质量评分
    ocr         OCR 可插拔后端（paddleocr / sidecar / null）

本层是整个链路的第一道工序，也是「脏数据能不能救回来」的分水岭：
解析错了，后面切得再漂亮、召回再准，答案也一定是错的。
"""

from __future__ import annotations

from .cleaning import OCR_CONFUSIONS, QualityReport, clean_corpus, clean_document, clean_text, score_quality, strip_boilerplate
from .loader import (
    Block,
    Corpus,
    FAQEntry,
    RawSection,
    SourceDocument,
    apply_masking,
    load_corpus,
    load_source_document,
    map_fields,
    parse_front_matter,
    parse_markdown_sections,
    parse_markdown_table,
    render_table_rows,
)
from .ocr import OCREngine, OCRResult, extract_text, get_engine

__all__ = [
    "Block",
    "RawSection",
    "SourceDocument",
    "FAQEntry",
    "Corpus",
    "load_corpus",
    "load_source_document",
    "parse_front_matter",
    "parse_markdown_sections",
    "parse_markdown_table",
    "render_table_rows",
    "map_fields",
    "apply_masking",
    "clean_text",
    "clean_document",
    "clean_corpus",
    "strip_boilerplate",
    "score_quality",
    "QualityReport",
    "OCR_CONFUSIONS",
    "OCREngine",
    "OCRResult",
    "get_engine",
    "extract_text",
]
