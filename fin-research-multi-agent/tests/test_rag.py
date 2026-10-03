"""RAG 层测试：父子块、确定性哈希向量、BM25、混合重排、结构化事实抽取。"""

from __future__ import annotations

import numpy as np
import pytest

from src.rag import HybridRetriever, embed, embed_matrix
from src.rag.embedding import cosine_similarity
from src.utils.text import tokenize


# ---------------------------------------------------------------------------
# 1. 父子块切分
# ---------------------------------------------------------------------------
def test_parent_child_structure_is_consistent(chunks):
    parents, children = chunks
    assert parents and children

    parent_ids = [p.parent_id for p in parents]
    child_ids = [c.child_id for c in children]
    assert len(parent_ids) == len(set(parent_ids)), "父块 ID 不能重复"
    assert len(child_ids) == len(set(child_ids)), "子块 ID 不能重复"

    parent_set = set(parent_ids)
    for child in children:
        assert child.parent_id in parent_set, "每个子块必须能挂到父块上"
        assert child.text.strip()
        assert child.meta.get("company")
        assert child.meta.get("section_title")

    with_children = {c.parent_id for c in children}
    assert with_children == parent_set, "每个父块都应当至少有一个子块"


def test_child_chunks_are_smaller_than_parents(chunks):
    parents, children = chunks
    parent_len = {p.parent_id: len(p.text) for p in parents}
    for child in children[:20]:
        assert len(child.text) <= parent_len[child.parent_id]


# ---------------------------------------------------------------------------
# 2. 确定性哈希向量
# ---------------------------------------------------------------------------
def test_embedding_is_deterministic_and_normalized():
    a = embed("净利润同比下降")
    b = embed("净利润同比下降")
    assert np.array_equal(a, b)
    assert float(np.linalg.norm(a)) == pytest.approx(1.0)

    empty = embed("")
    assert float(np.linalg.norm(empty)) == 0.0


def test_embedding_similarity_reflects_token_overlap():
    base = embed("经营活动现金流量净额下降")
    near = embed("经营活动现金流下降")
    far = embed("研发费用投入增加")
    assert cosine_similarity(base, near) > cosine_similarity(base, far)


def test_embed_matrix_shape():
    matrix = embed_matrix(["净利润", "营业收入", "风险"])
    assert matrix.shape == (3, 256)


def test_tokenizer_handles_chinese_and_numbers():
    tokens = tokenize("2024年净利润 1,286,400.00 万元 ROE")
    assert "净利" in tokens          # 二元组
    assert "润" in tokens            # 单字
    assert "roe" in tokens           # 英文整词小写
    assert "1286400.00" in tokens    # 去掉千分位后补的整串
    assert tokens == tokenize("2024年净利润 1,286,400.00 万元 ROE")  # 确定性


# ---------------------------------------------------------------------------
# 3. BM25 与混合重排
# ---------------------------------------------------------------------------
def test_bm25_ranks_keyword_matching_chunk_first(retriever):
    results = retriever.search("控股股东股权质押比例", top_k=3)
    assert results
    assert "质押" in results[0].text or "质押" in results[0].context


def test_hybrid_search_returns_provenance(retriever):
    results = retriever.search("未决诉讼 对外担保", top_k=3)
    assert results
    for item in results:
        assert item.source_id and item.section_title
        assert item.text and item.context
        assert set(item.components) >= {"keyword", "vector", "metadata"}
        assert 0.0 <= item.score <= 1.0 + 1e-9


def test_rerank_weights_are_configurable(chunks):
    parents, children = chunks
    keyword_only = HybridRetriever(
        parents, children, weights={"keyword": 1.0, "vector": 0.0, "metadata": 0.0}
    )
    vector_only = HybridRetriever(
        parents, children, weights={"keyword": 0.0, "vector": 1.0, "metadata": 0.0}
    )
    query = "净利润现金含量"
    kw_hits = [(r.child_id, r.components["keyword"]) for r in keyword_only.search(query, top_k=5)]
    vec_hits = [(r.child_id, r.components["vector"]) for r in vector_only.search(query, top_k=5)]

    assert kw_hits and vec_hits
    # 纯关键词模式下，关键词分就是排序依据（单调不增）
    scores = [s for _, s in kw_hits]
    assert scores == sorted(scores, reverse=True)
    # 两种权重下排序结果不必相同，但都必须返回结果
    assert {cid for cid, _ in kw_hits} and {cid for cid, _ in vec_hits}


def test_metadata_filter_strict_restricts_to_company(chunks):
    parents, children = chunks
    retriever = HybridRetriever(parents, children)
    results = retriever.search(
        "不良贷款率 拨备覆盖率", top_k=5, company="示例智造银行股份有限公司", strict=True
    )
    assert results
    assert all(r.source_id == "EX-BANK-2024-AR" for r in results)

    loose = retriever.search("不良贷款率 拨备覆盖率", top_k=5, company="示例智造银行股份有限公司", strict=False)
    assert loose
    # 软过滤时元数据命中作为加权信号，银行资料仍应排在前面
    assert loose[0].source_id == "EX-BANK-2024-AR"


def test_multi_query_merge_keeps_best_score(chunks):
    parents, children = chunks
    retriever = HybridRetriever(parents, children)
    single = retriever.search("股权质押", top_k=5)
    multi = retriever.search_multi(["股权质押", "未决诉讼", "对外担保"], top_k=5)
    assert len(multi) >= len(single)
    assert len({r.child_id for r in multi}) == len(multi), "合并后不应有重复子块"


# ---------------------------------------------------------------------------
# 4. 结构化事实抽取
# ---------------------------------------------------------------------------
def test_fact_store_parses_annual_report(fact_store):
    fact = fact_store.get("示例科技股份有限公司", "营业收入", 2024)
    assert fact is not None
    assert fact.value == pytest.approx(1286400.0)
    assert fact.unit == "万元"
    assert fact.source_id == "EX-TECH-2024-AR"
    assert "关键财务数据" in fact.section_title


def test_fact_store_parses_prior_year(fact_store):
    prior = fact_store.get("示例科技股份有限公司", "营业收入", 2023)
    assert prior is not None and prior.value == pytest.approx(1102300.0)


def test_fact_store_distinguishes_periods(fact_store):
    annual = fact_store.get("示例科技股份有限公司", "营业收入", 2024, period="年度")
    q3 = fact_store.get("示例科技股份有限公司", "营业收入", 2024, period="三季度")
    assert annual.source_id == "EX-TECH-2024-AR"
    assert q3.source_id == "EX-TECH-2024-Q3"
    assert q3.value == pytest.approx(892400.0)


def test_fact_store_resolves_company_short_name(fact_store):
    assert fact_store.resolve_company("示例科技") == "示例科技股份有限公司"
    assert fact_store.resolve_company("示例智造银行") == "示例智造银行股份有限公司"
    assert fact_store.resolve_company("不存在的公司") is None


def test_fact_store_metric_aliases(fact_store):
    alias = fact_store.get("示例科技股份有限公司", "营收", 2024)
    assert alias is not None and alias.metric == "营业收入"
    equity = fact_store.get("示例科技股份有限公司", "净资产", 2024)
    assert equity is not None and equity.metric == "所有者权益"


def test_fact_store_covers_bank_regulatory_metrics(fact_store):
    npl = fact_store.get("示例智造银行股份有限公司", "不良贷款率", 2024)
    coverage = fact_store.get("示例智造银行股份有限公司", "拨备覆盖率", 2024)
    assert npl is not None and npl.value == pytest.approx(1.42)
    assert coverage is not None and coverage.value == pytest.approx(218.50)


def test_documents_are_fictional_and_carry_disclaimer(documents):
    """合规要求：资料库必须是虚构主体，且带免责声明。"""
    assert len(documents) == 3
    forbidden = ("工商银行", "招商银行", "中信证券", "贵州茅台", "宁德时代", "腾讯", "阿里巴巴")
    for doc in documents:
        assert doc.meta.get("disclaimer"), f"{doc.source_id} 缺少免责声明"
        for name in forbidden:
            assert name not in doc.raw, f"{doc.source_id} 出现了真实主体名称：{name}"
        assert "示例" in doc.company
