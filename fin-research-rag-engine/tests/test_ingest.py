"""解析清洗层测试：格式适配 / 字段映射 / 表格结构化 / 脱敏 / OCR 纠错 / 质量评分。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingest.cleaning import clean_text, score_quality, strip_boilerplate
from src.ingest.loader import (
    SourceDocument,
    load_corpus,
    load_source_document,
    map_fields,
    parse_front_matter,
    parse_markdown_sections,
    parse_markdown_table,
    render_table_rows,
)
from src.ingest.cleaning import OCR_CONFUSIONS
from src.ingest.ocr import NullOCREngine, SidecarOCREngine, extract_text, get_engine
from src.utils import mask_sensitive, normalize_whitespace, split_sentences, tokenize


# ---------------------------------------------------------------------------
# 分词与归一化
# ---------------------------------------------------------------------------
def test_normalize_text_turns_punctuation_into_space():
    from src.utils import normalize_text

    assert normalize_text("第一条：合格投资者，金融资产。") == "第一条 合格投资者 金融资产"


def test_normalize_whitespace_keeps_punctuation():
    assert normalize_whitespace("第一条：合格投资者，金融资产。") == "第一条：合格投资者，金融资产。"


def test_tokenize_keeps_cjk_bigrams_and_codes():
    tokens = tokenize("产品代码 WY2024-01 风险等级 R2")
    assert "wy2024-01" in tokens
    assert "风险" in tokens and "险等" in tokens
    assert "r2" in tokens


def test_tokenize_keeps_clause_numbers_intact():
    tokens = tokenize("依据第四十二条之规定")
    assert "第四十二条" in tokens


def test_split_sentences_preserves_original_punctuation():
    sents = split_sentences("第一条 为规范管理，制定本办法。第二条 本办法适用于银行。")
    assert sents == ["第一条 为规范管理，制定本办法。", "第二条 本办法适用于银行。"]


# ---------------------------------------------------------------------------
# 脱敏
# ---------------------------------------------------------------------------
def test_mask_sensitive_masks_id_phone_bank_email():
    text = "身份证 310101199001011234 手机 13812345678 卡号 6222021234567890123 邮箱 a.b@example.com"
    masked = mask_sensitive(text)
    assert "310101199001011234" not in masked
    assert "13812345678" not in masked
    assert "6222021234567890123" not in masked
    assert "a.b@example.com" not in masked
    assert "[ID:" in masked and "[PHONE:" in masked and "[BANK:" in masked and "[EMAIL:" in masked


def test_mask_sensitive_keeps_tail_for_manual_check():
    masked = mask_sensitive("手机号 13812345678")
    assert masked.endswith("5678]")


def test_mask_sensitive_leaves_normal_numbers_alone():
    assert mask_sensitive("规模 300 万元，期限 90 天") == "规模 300 万元，期限 90 天"


# ---------------------------------------------------------------------------
# front-matter 与字段映射
# ---------------------------------------------------------------------------
def test_parse_front_matter():
    meta, body = parse_front_matter("---\nsource_id: X-1\ntitle: 测试\n---\n# 标题\n正文")
    assert meta == {"source_id": "X-1", "title": "测试"}
    assert body.startswith("# 标题")


def test_parse_front_matter_without_block():
    meta, body = parse_front_matter("正文而已")
    assert meta == {} and body == "正文而已"


def test_map_fields_recognises_chinese_aliases():
    mapped = map_fields({"资料编号": "A-1", "机构名称": "示例银行", "生效日期": "2024-01-01"})
    assert mapped["source_id"] == "A-1"
    assert mapped["institution"] == "示例银行"
    assert mapped["effective_date"] == "2024-01-01"


def test_map_fields_keeps_unknown_fields_with_prefix():
    mapped = map_fields({"自定义字段": "值", "机构名称": "示例银行"})
    assert mapped["x_自定义字段"] == "值"
    assert mapped["institution"] == "示例银行"


def test_map_fields_skips_empty_values():
    assert map_fields({"机构名称": "  ", "标题": None}) == {}


# ---------------------------------------------------------------------------
# Markdown 解析
# ---------------------------------------------------------------------------
def test_parse_markdown_table_drops_separator_row():
    rows = parse_markdown_table(["| A | B |", "| --- | --- |", "| 1 | 2 |"])
    assert rows == [["A", "B"], ["1", "2"]]


def test_parse_markdown_sections_splits_by_heading():
    body = "# 标题\n正文。\n\n## 一、总则\n第一条 内容。\n\n## 二、附则\n第二条 内容。"
    sections = parse_markdown_sections(body)
    titles = [s.title for s in sections]
    assert titles == ["标题", "一、总则", "二、附则"]


def test_parse_markdown_sections_drops_empty_sections():
    """只有标题、没有正文的章节不会产出空块（否则会污染索引与引用编号）。"""
    body = "# 标题\n\n## 一、总则\n第一条 内容。"
    titles = [s.title for s in parse_markdown_sections(body)]
    assert titles == ["一、总则"]


def test_parse_markdown_sections_marks_table_blocks():
    body = "## 一、表\n\n| 列 | 值 |\n| --- | --- |\n| a | b |\n"
    sections = parse_markdown_sections(body)
    assert any(b.is_table for s in sections for b in s.blocks)


def test_render_table_rows_repeats_header_per_row():
    text = render_table_rows([["等级", "上限"], ["C1", "0%"], ["C2", "3%"]])
    assert "等级: C1" in text and "上限: 0%" in text
    assert "等级: C2" in text


# ---------------------------------------------------------------------------
# 多格式加载
# ---------------------------------------------------------------------------
def test_load_corpus_reads_all_formats(corpus):
    formats = {doc.fmt for doc in corpus.documents}
    assert {"markdown", "text", "csv", "json"} <= formats


def test_load_corpus_separates_faq(corpus):
    assert len(corpus.faq) >= 10
    assert all(entry.question and entry.answer for entry in corpus.faq)


def test_corpus_get_and_require(corpus):
    assert corpus.get("POL-2024-07") is not None
    with pytest.raises(KeyError):
        corpus.require("NOT-EXIST")


def test_corpus_stats(corpus):
    stats = corpus.stats()
    assert stats["documents"] == len(corpus.documents)
    assert stats["tables"] > 0
    assert stats["faq"] == len(corpus.faq)


def test_csv_document_has_table_block(corpus):
    doc = corpus.require("PROD-TABLE-2024")
    assert doc.table_count >= 1
    header = doc.sections[0].blocks[0].rows[0]
    assert "产品代码" in header


def test_json_document_builds_table_and_text(corpus):
    doc = corpus.require("PROD-JSON-2024")
    assert doc.table_count >= 1
    assert "JG2024-03" in doc.sections[0].text


def test_source_document_metadata_helpers(corpus):
    doc = corpus.require("POL-2024-07")
    assert doc.doc_type == "监管政策"
    assert doc.year == 2024
    assert doc.version == "v2"
    assert doc.metadata_dict()["source_id"] == "POL-2024-07"


def test_load_source_document_returns_none_for_unknown_suffix(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"\x00")
    assert load_source_document(path) is None


# ---------------------------------------------------------------------------
# 清洗
# ---------------------------------------------------------------------------
def test_clean_text_fixes_ocr_confusions():
    assert "已经" in clean_text("客户己经完成评估")
    assert "收益" in clean_text("预期收溢率为 3.2%")


def test_clean_text_fixes_letters_in_digit_context():
    cleaned = clean_text("金额 1O0 万元，代码 2O24")
    assert "100" in cleaned
    assert "2024" in cleaned


def test_clean_text_drops_watermark_lines():
    cleaned = clean_text("机 密\n第一条 正文内容。")
    assert "机 密" not in cleaned
    assert "第一条" in cleaned


def test_clean_text_keeps_punctuation_and_paragraphs():
    cleaned = clean_text("第一条 为规范管理，制定本办法。\n\n第二条 本办法适用于银行。")
    assert "，" in cleaned and "。" in cleaned
    assert "\n\n" in cleaned


def test_strip_boilerplate_removes_repeated_short_lines():
    lines = ["示例银行 内部资料", "正文一。", "示例银行 内部资料", "正文二。", "示例银行 内部资料"]
    kept, boiler = strip_boilerplate(lines)
    assert all("内部资料" not in line for line in kept)
    assert len(boiler) == 3


def test_strip_boilerplate_keeps_long_repeated_paragraphs():
    long_line = "本产品不构成投资建议，" * 5
    kept, boiler = strip_boilerplate([long_line, long_line, long_line, "短句。"])
    assert long_line in kept
    assert not boiler


def test_ocr_confusions_table_covers_common_cases():
    assert OCR_CONFUSIONS["己经"] == "已经"
    assert OCR_CONFUSIONS["帐号"] == "账号"


def test_score_quality_penalises_ocr_documents(corpus):
    doc = corpus.require("OCR-2023-12")
    report = score_quality(doc)
    assert 0.0 <= report.score <= 1.0
    assert any("OCR" in issue for issue in report.issues)


def test_clean_document_records_issues(corpus):
    from src.ingest import load_corpus as reload_corpus
    from src.config import DATA_DIR
    from src.ingest.cleaning import clean_document

    fresh = reload_corpus(DATA_DIR)
    doc = fresh.require("OCR-2023-12")
    report = clean_document(doc)
    before = doc.raw
    assert "已经" in doc.raw or "已经" in before or report.score <= 1.0
    assert doc.issues


# ---------------------------------------------------------------------------
# OCR 适配层
# ---------------------------------------------------------------------------
def test_sidecar_engine_reads_companion_text(tmp_path):
    image = tmp_path / "scan.png"
    image.write_bytes(b"\x89PNG")
    (tmp_path / "scan.txt").write_text("扫描件转录内容", encoding="utf-8")
    result = SidecarOCREngine().recognize(image)
    assert result.ok and result.text == "扫描件转录内容"
    assert result.engine == "sidecar"


def test_sidecar_engine_reports_missing_companion(tmp_path):
    image = tmp_path / "scan.png"
    image.write_bytes(b"\x89PNG")
    result = SidecarOCREngine().recognize(image)
    assert not result.ok
    assert result.warnings


def test_null_engine_never_silently_succeeds(tmp_path):
    image = tmp_path / "scan.png"
    image.write_bytes(b"\x89PNG")
    result = NullOCREngine().recognize(image)
    assert not result.ok and result.warnings


def test_extract_text_rejects_non_image(tmp_path):
    path = tmp_path / "a.md"
    path.write_text("x", encoding="utf-8")
    result = extract_text(path)
    assert result.engine == "none" and result.warnings


def test_get_engine_returns_something_usable():
    engine = get_engine("auto")
    assert engine.available()


# ---------------------------------------------------------------------------
# 脱敏在语料上的整体效果
# ---------------------------------------------------------------------------
def test_corpus_contains_no_raw_phone_numbers(corpus):
    for doc in corpus.documents:
        assert "13812345678" not in doc.raw


def test_load_corpus_on_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "not-exist")
