"""切分层测试：父子块切分、表格成组切分、重叠、元数据、过滤表达式。"""

from __future__ import annotations

import pytest

from src.chunking import (
    BLOCK_TABLE,
    MetadataFilter,
    build_chunks,
    build_filter,
    chunk_stats,
    extract_metadata,
    filter_chunks,
    split_document,
    split_paragraph_into_children,
    split_table_into_children,
)
from src.chunking.metadata import FilterError


# ---------------------------------------------------------------------------
# 段落切分
# ---------------------------------------------------------------------------
def test_paragraph_chunks_respect_max_chars():
    text = "。".join(["这是第%d句测试内容" % i for i in range(30)]) + "。"
    chunks = split_paragraph_into_children(text, max_chars=80, overlap=10)
    assert chunks
    assert all(len(c) <= 80 + 10 + 5 for c in chunks)


def test_paragraph_chunks_keep_overlap_between_neighbours():
    text = "。".join(["句子内容ABCDEFG%d" % i for i in range(20)]) + "。"
    chunks = split_paragraph_into_children(text, max_chars=60, overlap=20)
    assert len(chunks) >= 2
    # 相邻块应当有交集（跨块语义连续）
    assert any(chunks[i][-8:] in chunks[i + 1] or chunks[i + 1][:8] in chunks[i] for i in range(len(chunks) - 1))


def test_paragraph_chunks_hard_split_single_long_sentence():
    long_sentence = "很长的一句话" * 60
    chunks = split_paragraph_into_children(long_sentence, max_chars=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_paragraph_chunks_empty_input():
    assert split_paragraph_into_children("", 100, 10) == []


# ---------------------------------------------------------------------------
# 表格切分
# ---------------------------------------------------------------------------
def test_table_chunks_repeat_header_in_every_chunk():
    rows = [["产品代码", "风险等级"], ["A1", "R1"], ["A2", "R2"], ["A3", "R3"]]
    chunks = split_table_into_children(rows, rows_per_chunk=1)
    assert len(chunks) == 3
    for text, _ in chunks:
        assert "产品代码" in text and "风险等级" in text


def test_table_chunk_row_span_is_one_based():
    rows = [["h"], ["r1"], ["r2"], ["r3"]]
    chunks = split_table_into_children(rows, rows_per_chunk=2)
    assert chunks[0][1] == (1, 2)
    assert chunks[1][1] == (3, 3)


def test_table_without_body_returns_header_only():
    chunks = split_table_into_children([["列A", "列B"]])
    assert len(chunks) == 1
    assert "列A" in chunks[0][0]


def test_table_chunks_empty_rows():
    assert split_table_into_children([]) == []


# ---------------------------------------------------------------------------
# 文档切分
# ---------------------------------------------------------------------------
def test_split_document_produces_two_level_structure(cleaned_documents):
    doc = next(d for d in cleaned_documents if d.source_id == "POL-2024-07")
    parents, children = split_document(doc)
    assert parents and children
    parent_ids = {p.parent_id for p in parents}
    assert all(c.parent_id in parent_ids for c in children)


def test_child_ids_are_unique_and_traceable(cleaned_documents):
    parents, children = split_document(cleaned_documents[0])
    ids = [c.child_id for c in children]
    assert len(ids) == len(set(ids))
    for child in children:
        assert child.child_id.startswith(child.parent_id)


def test_table_children_marked_as_table(cleaned_documents):
    doc = next(d for d in cleaned_documents if d.source_id == "PROD-TABLE-2024")
    _, children = split_document(doc)
    assert children
    assert all(c.kind == BLOCK_TABLE for c in children)
    assert all(c.row_span for c in children)


def test_children_carry_filterable_metadata(cleaned_documents):
    doc = next(d for d in cleaned_documents if d.source_id == "POL-2024-07")
    _, children = split_document(doc)
    meta = children[0].meta
    for field in ("source_id", "doc_type", "institution", "year", "version", "section_title", "block_kind"):
        assert field in meta


def test_child_citation_label_contains_institution_and_section(cleaned_documents):
    doc = next(d for d in cleaned_documents if d.source_id == "POL-2024-07")
    _, children = split_document(doc)
    label = children[0].citation_label
    assert "示例监管机构" in label
    assert children[0].section_title in label


def test_build_chunks_counts_match_documents(cleaned_documents):
    parents, children = build_chunks(cleaned_documents)
    assert len(parents) > 0 and len(children) > 0
    source_ids = {c.source_id for c in children}
    assert source_ids <= {d.source_id for d in cleaned_documents}


def test_chunk_stats_are_consistent(cleaned_documents):
    parents, children = build_chunks(cleaned_documents)
    stats = chunk_stats(parents, children, document_count=len(cleaned_documents))
    assert stats.parents == len(parents)
    assert stats.children == len(children)
    assert stats.table_children + stats.paragraph_children == stats.children
    assert stats.min_child_chars <= stats.avg_child_chars <= stats.max_child_chars
    assert stats.to_dict()["documents"] == len(cleaned_documents)


def test_quality_score_propagates_into_chunks(cleaned_documents):
    scores = {d.source_id: 0.5 for d in cleaned_documents}
    _, children = build_chunks(cleaned_documents, quality_scores=scores)
    assert all(c.quality_score == 0.5 for c in children)


def test_parent_chunk_to_dict_is_serialisable(cleaned_documents):
    parents, _ = build_chunks(cleaned_documents[:1])
    payload = parents[0].to_dict()
    assert isinstance(payload, dict) and payload["parent_id"]


# ---------------------------------------------------------------------------
# 元数据过滤表达式
# ---------------------------------------------------------------------------
def test_filter_expression_equality():
    cond = MetadataFilter.parse('doc_type = "监管政策"')
    assert cond.matches({"doc_type": "监管政策"})
    assert not cond.matches({"doc_type": "内部制度"})


def test_filter_expression_and_or_precedence():
    cond = MetadataFilter.parse('doc_type = "监管政策" AND year >= 2024 OR institution = "示例银行"')
    assert cond.matches({"doc_type": "监管政策", "year": 2024, "institution": "其它"})
    assert cond.matches({"doc_type": "内部制度", "year": 2020, "institution": "示例银行"})
    assert not cond.matches({"doc_type": "内部制度", "year": 2020, "institution": "其它"})


def test_filter_expression_parentheses():
    cond = MetadataFilter.parse('(doc_type = "监管政策" OR doc_type = "内部制度") AND year >= 2024')
    assert cond.matches({"doc_type": "内部制度", "year": 2025})
    assert not cond.matches({"doc_type": "内部制度", "year": 2023})


def test_filter_expression_in_list():
    cond = MetadataFilter.parse('institution in ["示例银行", "示例证券"]')
    assert cond.matches({"institution": "示例银行"})
    assert not cond.matches({"institution": "示例基金"})


def test_filter_expression_not_in():
    cond = MetadataFilter.parse('version not in ["v1"]')
    assert cond.matches({"version": "v2"})
    assert not cond.matches({"version": "v1"})


def test_filter_expression_contains_and_not():
    cond = MetadataFilter.parse('title contains "适当性" AND not (version = "v1")')
    assert cond.matches({"title": "客户适当性管理办法", "version": "v2"})
    assert not cond.matches({"title": "客户适当性管理办法", "version": "v1"})


def test_filter_expression_quoted_and_unquoted_values():
    assert MetadataFilter.parse("doc_type = 监管政策").matches({"doc_type": "监管政策"})
    cond = MetadataFilter.parse('doc_type = "监管政策"')
    assert cond.matches({"doc_type": "监管政策"})


def test_empty_filter_matches_everything():
    cond = MetadataFilter.parse("")
    assert cond.is_empty and cond.matches({"anything": 1})


def test_filter_error_on_missing_operator():
    with pytest.raises(FilterError):
        MetadataFilter.parse("doc_type 监管政策")


def test_filter_error_on_unclosed_parenthesis():
    with pytest.raises(FilterError):
        MetadataFilter.parse('(doc_type = "监管政策"')


def test_filter_error_on_illegal_character():
    with pytest.raises(FilterError):
        MetadataFilter.parse('doc_type = "监管政策" @ 1')


def test_filter_numeric_comparison_with_string_metadata():
    """元数据里的年份经常是字符串（来自 front-matter），比较时应当能自动识别数字。"""
    cond = MetadataFilter.parse("year >= 2024")
    assert cond.matches({"year": "2024"})
    assert not cond.matches({"year": "2023"})


def test_filter_missing_field_does_not_match():
    cond = MetadataFilter.parse('doc_type = "监管政策"')
    assert not cond.matches({})


def test_build_filter_helper_and_filter_chunks(cleaned_documents):
    _, children = build_chunks(cleaned_documents)
    cond = build_filter('doc_type = "监管政策"')
    kept = filter_chunks(children, cond.expr)
    assert kept
    assert all(c.meta["doc_type"] == "监管政策" for c in kept)


def test_extract_metadata_merges_extra_fields(cleaned_documents):
    doc = cleaned_documents[0]
    meta = extract_metadata(doc, None, block_kind="paragraph", extra={"custom": 1})
    assert meta["custom"] == 1
    assert meta["block_kind"] == "paragraph"
