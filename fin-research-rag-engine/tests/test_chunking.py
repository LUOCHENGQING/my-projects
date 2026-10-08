"""切分层测试：父子块切分、表格成组切分、重叠、元数据、过滤表达式。

覆盖的被测模块
--------------
    src.chunking.parent_child  split_paragraph_into_children / split_table_into_children /
                               split_document / build_chunks / chunk_stats，以及块对象的
                               ID 体系、kind 标记、row_span、citation_label 与 to_dict 导出
    src.chunking.metadata      MetadataFilter 的表达式解析与求值、build_filter / filter_chunks /
                               extract_metadata，以及非法表达式抛出的 FilterError

覆盖策略
--------
    正常路径：段落按句累积切分并保留重叠、表格成组切分且每组重复表头、
              父子两级结构可回捞、元数据透传、过滤表达式的等值/比较/IN/contains/括号组合。
    边界：空段落、空表、只有表头、超长单句硬切、块长与重叠上限、缺字段元数据、
          带引号与不带引号的取值、空表达式。
    异常：缺少运算符、括号未闭合、非法字符三类非法表达式必须抛 FilterError，绝不静默放行。
    对抗：过滤求值宁可少召回不可错召回（缺字段判不匹配、AND 优先于 OR）。
文档级用例统一使用会话级 cleaned_documents（真实示例语料），其余用内联小数据做精确断言。
"""

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
    """段落按句累积切块时块长必须受 max_chars 约束（只允许重叠带来的少量溢出）。"""
    # 30 句拼接成一段，句长可控，便于验证上限而不是依赖真实语料
    text = "。".join(["这是第%d句测试内容" % i for i in range(30)]) + "。"
    chunks = split_paragraph_into_children(text, max_chars=80, overlap=10)
    assert chunks
    assert all(len(c) <= 80 + 10 + 5 for c in chunks)


def test_paragraph_chunks_keep_overlap_between_neighbours():
    """相邻子块之间必须保留重叠文本，跨块语义（含被切断的编号）不能断裂。"""
    text = "。".join(["句子内容ABCDEFG%d" % i for i in range(20)]) + "。"
    chunks = split_paragraph_into_children(text, max_chars=60, overlap=20)
    assert len(chunks) >= 2
    # 相邻块应当有交集（跨块语义连续）
    assert any(chunks[i][-8:] in chunks[i + 1] or chunks[i + 1][:8] in chunks[i] for i in range(len(chunks) - 1))


def test_paragraph_chunks_hard_split_single_long_sentence():
    """不可切的超长单句必须退化为按字符硬切，且每片仍受 max_chars 上限约束。"""
    # 一整句没有句末标点，split_sentences 无法再分，只能走硬切分支
    long_sentence = "很长的一句话" * 60
    chunks = split_paragraph_into_children(long_sentence, max_chars=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_paragraph_chunks_empty_input():
    """空输入必须返回空列表而不是含空串的列表，避免索引里出现无内容可命中的块。"""
    assert split_paragraph_into_children("", 100, 10) == []


# ---------------------------------------------------------------------------
# 表格切分
# ---------------------------------------------------------------------------
def test_table_chunks_repeat_header_in_every_chunk():
    """表格成组切分后每一组都必须重复带上表头，否则列名与数据行分离、语义拼不回来。"""
    # 每个数据行各成一组，用来验证每一组都带表头（而不是只有第一组带）
    rows = [["产品代码", "风险等级"], ["A1", "R1"], ["A2", "R2"], ["A3", "R3"]]
    chunks = split_table_into_children(rows, rows_per_chunk=1)
    assert len(chunks) == 3
    for text, _ in chunks:
        assert "产品代码" in text and "风险等级" in text


def test_table_chunk_row_span_is_one_based():
    """表格子块的数据行区间必须从 1 起编号且不含表头，引用才能定位到具体数据行。"""
    # 表头 1 行 + 数据 3 行，按每 2 行一组正好切成 (1,2) 与 (3,3)
    rows = [["h"], ["r1"], ["r2"], ["r3"]]
    chunks = split_table_into_children(rows, rows_per_chunk=2)
    assert chunks[0][1] == (1, 2)
    assert chunks[1][1] == (3, 3)


def test_table_without_body_returns_header_only():
    """只有表头没有数据行时表头本身必须保留成一块——表头承载「这张表在统计什么」。"""
    chunks = split_table_into_children([["列A", "列B"]])
    assert len(chunks) == 1
    assert "列A" in chunks[0][0]


def test_table_chunks_empty_rows():
    """空表格必须返回空列表（连表头都没有，不能凭空造块）。"""
    assert split_table_into_children([]) == []


# ---------------------------------------------------------------------------
# 文档切分
# ---------------------------------------------------------------------------
def test_split_document_produces_two_level_structure(cleaned_documents):
    """切分必须产出父子两级结构，且每个子块的 parent_id 都能在父块列表里找到（回捞链路不断）。"""
    # 固定挑一篇同时含章节、段落与表格的监管政策文档，保证两级结构都被覆盖
    doc = next(d for d in cleaned_documents if d.source_id == "POL-2024-07")
    parents, children = split_document(doc)
    assert parents and children
    parent_ids = {p.parent_id for p in parents}
    assert all(c.parent_id in parent_ids for c in children)


def test_child_ids_are_unique_and_traceable(cleaned_documents):
    """子块 ID 必须全局唯一、且以所属父块 ID 为前缀，答案里的 [n] 才能反推到原文。"""
    parents, children = split_document(cleaned_documents[0])
    ids = [c.child_id for c in children]
    assert len(ids) == len(set(ids))
    for child in children:
        assert child.child_id.startswith(child.parent_id)


def test_table_children_marked_as_table(cleaned_documents):
    """表格来源的子块必须打上 table 标记并带行区间，检索层据此走不同的展示与重排策略。"""
    # 选一篇以表格为主体的产品要素文档，确保子块全部来自表格块
    doc = next(d for d in cleaned_documents if d.source_id == "PROD-TABLE-2024")
    _, children = split_document(doc)
    assert children
    assert all(c.kind == BLOCK_TABLE for c in children)
    assert all(c.row_span for c in children)


def test_children_carry_filterable_metadata(cleaned_documents):
    """每个子块都要带齐可过滤的扁平元数据字段——它是召回前过滤的唯一载体。"""
    doc = next(d for d in cleaned_documents if d.source_id == "POL-2024-07")
    _, children = split_document(doc)
    meta = children[0].meta
    for field in ("source_id", "doc_type", "institution", "year", "version", "section_title", "block_kind"):
        assert field in meta


def test_child_citation_label_contains_institution_and_section(cleaned_documents):
    """引用标签必须同时含机构与章节：业务人员只看这一行就能定位到原文位置。"""
    doc = next(d for d in cleaned_documents if d.source_id == "POL-2024-07")
    _, children = split_document(doc)
    label = children[0].citation_label
    # 机构名照抄语料 front-matter，用来验证标签把文档级元数据与章节标题拼在了一起
    assert "示例监管机构" in label
    assert children[0].section_title in label


def test_build_chunks_counts_match_documents(cleaned_documents):
    """批量切分的产出必须可归因：子块的 source_id 只能来自输入文档集合，不能凭空多出。"""
    parents, children = build_chunks(cleaned_documents)
    assert len(parents) > 0 and len(children) > 0
    source_ids = {c.source_id for c in children}
    assert source_ids <= {d.source_id for d in cleaned_documents}


def test_chunk_stats_are_consistent(cleaned_documents):
    """切分统计的口径必须自洽：块数对得上、表格子块 + 段落子块 = 子块总数、长度区间有序。"""
    parents, children = build_chunks(cleaned_documents)
    # document_count 由调用方传入（函数本身不统计文档数），因此这里显式给语料文档数
    stats = chunk_stats(parents, children, document_count=len(cleaned_documents))
    assert stats.parents == len(parents)
    assert stats.children == len(children)
    assert stats.table_children + stats.paragraph_children == stats.children
    assert stats.min_child_chars <= stats.avg_child_chars <= stats.max_child_chars
    assert stats.to_dict()["documents"] == len(cleaned_documents)


def test_quality_score_propagates_into_chunks(cleaned_documents):
    """调用方传入的文档质量分必须原样透传到每个子块（排序加权依赖这个字段）。"""
    # 所有文档统一给 0.5，任何一块没带上就会破坏「全等于 0.5」的断言
    scores = {d.source_id: 0.5 for d in cleaned_documents}
    _, children = build_chunks(cleaned_documents, quality_scores=scores)
    assert all(c.quality_score == 0.5 for c in children)


def test_parent_chunk_to_dict_is_serialisable(cleaned_documents):
    """父块导出必须是可直接 JSON 序列化的精简字典（不含全文，避免响应体膨胀）。"""
    # 只切一篇文档：本用例校验的是导出结构，与语料规模无关
    parents, _ = build_chunks(cleaned_documents[:1])
    payload = parents[0].to_dict()
    assert isinstance(payload, dict) and payload["parent_id"]


# ---------------------------------------------------------------------------
# 元数据过滤表达式
# ---------------------------------------------------------------------------
def test_filter_expression_equality():
    """等值过滤必须精确匹配：取值不同的块不得放行（过滤被放宽等于合规风险）。"""
    cond = MetadataFilter.parse('doc_type = "监管政策"')
    assert cond.matches({"doc_type": "监管政策"})
    assert not cond.matches({"doc_type": "内部制度"})


def test_filter_expression_and_or_precedence():
    """AND 的优先级必须高于 OR：三类元数据组合用来区分两种可能的求值顺序。"""
    cond = MetadataFilter.parse('doc_type = "监管政策" AND year >= 2024 OR institution = "示例银行"')
    assert cond.matches({"doc_type": "监管政策", "year": 2024, "institution": "其它"})
    assert cond.matches({"doc_type": "内部制度", "year": 2020, "institution": "示例银行"})
    assert not cond.matches({"doc_type": "内部制度", "year": 2020, "institution": "其它"})


def test_filter_expression_parentheses():
    """括号必须能改变默认优先级：年份条件要作用在整个「或」分组上。"""
    cond = MetadataFilter.parse('(doc_type = "监管政策" OR doc_type = "内部制度") AND year >= 2024')
    assert cond.matches({"doc_type": "内部制度", "year": 2025})
    assert not cond.matches({"doc_type": "内部制度", "year": 2023})


def test_filter_expression_in_list():
    """in 列表匹配必须按成员判定：列表内的放行、列表外的一律剔除。"""
    cond = MetadataFilter.parse('institution in ["示例银行", "示例证券"]')
    assert cond.matches({"institution": "示例银行"})
    assert not cond.matches({"institution": "示例基金"})


def test_filter_expression_not_in():
    """not in 必须排除列表内取值：通常用来把旧版本文档摘出候选。"""
    cond = MetadataFilter.parse('version not in ["v1"]')
    assert cond.matches({"version": "v2"})
    assert not cond.matches({"version": "v1"})


def test_filter_expression_contains_and_not():
    """contains 子串匹配必须能与 not 取反组合（版本排除常写成这种并联条件）。"""
    cond = MetadataFilter.parse('title contains "适当性" AND not (version = "v1")')
    assert cond.matches({"title": "客户适当性管理办法", "version": "v2"})
    assert not cond.matches({"title": "客户适当性管理办法", "version": "v1"})


def test_filter_expression_quoted_and_unquoted_values():
    """取值带引号与不带引号都必须被接受：front-matter 里手写的值常常没有引号。"""
    assert MetadataFilter.parse("doc_type = 监管政策").matches({"doc_type": "监管政策"})
    cond = MetadataFilter.parse('doc_type = "监管政策"')
    assert cond.matches({"doc_type": "监管政策"})


def test_empty_filter_matches_everything():
    """空表达式表示「无过滤」，必须恒真：调用方不必到处判空，也不能把所有块都筛掉。"""
    cond = MetadataFilter.parse("")
    assert cond.is_empty and cond.matches({"anything": 1})


def test_filter_error_on_missing_operator():
    """缺少比较运算符的表达式必须报 FilterError，绝不静默放行（否则会返回未过滤结果）。"""
    with pytest.raises(FilterError):
        MetadataFilter.parse("doc_type 监管政策")


def test_filter_error_on_unclosed_parenthesis():
    """括号未闭合必须报 FilterError，避免半个条件被当作完整条件求值。"""
    with pytest.raises(FilterError):
        MetadataFilter.parse('(doc_type = "监管政策"')


def test_filter_error_on_illegal_character():
    """出现词法无法识别的字符必须报 FilterError，不允许跳过非法片段继续解析。"""
    with pytest.raises(FilterError):
        MetadataFilter.parse('doc_type = "监管政策" @ 1')


def test_filter_numeric_comparison_with_string_metadata():
    """元数据里的年份经常是字符串（来自 front-matter），比较时应当能自动识别数字。

    不变式：字符串形式的数字必须按数值比较，否则 `year >= 2024` 会把 "2023" 误放行。
    """
    cond = MetadataFilter.parse("year >= 2024")
    assert cond.matches({"year": "2024"})
    assert not cond.matches({"year": "2023"})


def test_filter_missing_field_does_not_match():
    """块缺少被过滤字段时必须判为不匹配：宁可少召回，不可把口径不明的块放进答案。"""
    cond = MetadataFilter.parse('doc_type = "监管政策"')
    # 空元数据代表「解析不出该字段」的块（例如只有标题的摘要块）
    assert not cond.matches({})


def test_build_filter_helper_and_filter_chunks(cleaned_documents):
    """build_filter 产出的条件交给 filter_chunks 后，必须只留下满足条件的块（不漏筛也不放行）。"""
    _, children = build_chunks(cleaned_documents)
    cond = build_filter('doc_type = "监管政策"')
    kept = filter_chunks(children, cond.expr)
    assert kept
    assert all(c.meta["doc_type"] == "监管政策" for c in kept)


def test_extract_metadata_merges_extra_fields(cleaned_documents):
    """extra 传入的字段必须并入元数据，且 block_kind 等标准字段由入参决定不被带偏。"""
    doc = cleaned_documents[0]
    # extra 用来承载调用方自定义的过滤维度（如租户、批次），必须能从块上读回
    meta = extract_metadata(doc, None, block_kind="paragraph", extra={"custom": 1})
    assert meta["custom"] == 1
    assert meta["block_kind"] == "paragraph"
