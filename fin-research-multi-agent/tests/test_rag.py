"""RAG 层测试：父子块、确定性哈希向量、BM25、混合重排、结构化事实抽取。

被测行为（src.rag 的 build_chunks / HybridRetriever / embed / embed_matrix、src.utils.text.tokenize）：
1. 切分：父块与子块 ID 唯一、子必挂父、父必有子，子块不长于父块且带公司 / 章节元数据；
2. 向量：同一文本哈希向量逐位相同、非空文本归一化为单位向量、空串为零向量、相似度随 token 重合度上升、批量矩阵形状固定；
3. 分词：中文二元组 + 单字、英文小写整词、去千分位数字串，且结果确定；
4. 检索：BM25 关键词命中优先、每条结果带完整来源与三路组件分且分数在 [0,1]、权重可配置、strict 元数据过滤、多查询合并去重；
5. 事实库：年报 / 上期 / 不同报告期的指标抽取、公司简称归一、指标别名归一、银行监管指标；
6. 合规：资料必须是虚构主体并携带免责声明。

覆盖策略：正常（检索与抽取主路径）、边界（空串向量、抽样比较、未知公司、模糊查询）、
异常（非法入参在工具层覆盖，本文件聚焦数据层本身）、
对抗（真实主体名黑名单、strict 过滤不得泄露其它公司资料）。
"""

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
    """验证不变式：父子块 ID 全局唯一、每个子块都能挂到父块、每个父块都至少有一个子块，且子块文本与公司 / 章节元数据非空。"""
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
    # 反向检查「光杆父块」：没有子块的父块永远不会被检索到，属于切分缺陷
    assert with_children == parent_set, "每个父块都应当至少有一个子块"


def test_child_chunks_are_smaller_than_parents(chunks):
    """验证规则：子块长度不得超过其所属父块（父子切分的容量前提）。"""
    parents, children = chunks
    # 只抽样前 20 个子块：全量比较不增加信息量，反而拖慢用例
    parent_len = {p.parent_id: len(p.text) for p in parents}
    for child in children[:20]:
        assert len(child.text) <= parent_len[child.parent_id]


# ---------------------------------------------------------------------------
# 2. 确定性哈希向量
# ---------------------------------------------------------------------------
def test_embedding_is_deterministic_and_normalized():
    """验证规则：哈希向量对同一文本逐位可复现、非空文本归一化为单位向量、空串为零向量。"""
    # 同一文本连算两次：若实现里掺入随机种子，这里会立刻暴露
    a = embed("净利润同比下降")
    b = embed("净利润同比下降")
    assert np.array_equal(a, b)
    assert float(np.linalg.norm(a)) == pytest.approx(1.0)

    empty = embed("")
    assert float(np.linalg.norm(empty)) == 0.0


def test_embedding_similarity_reflects_token_overlap():
    """验证规则：余弦相似度必须随 token 重合度单调上升（近义改写 > 无关主题）。"""
    # 构造一组「高度重合」与一组「完全无关」的文本，比较相似度的方向而非绝对值
    base = embed("经营活动现金流量净额下降")
    near = embed("经营活动现金流下降")
    far = embed("研发费用投入增加")
    assert cosine_similarity(base, near) > cosine_similarity(base, far)


def test_embed_matrix_shape():
    """验证契约：批量向量矩阵形状为 (文本数, 256)，维度由哈希函数固定、与文本数量无关。"""
    matrix = embed_matrix(["净利润", "营业收入", "风险"])
    assert matrix.shape == (3, 256)


def test_tokenizer_handles_chinese_and_numbers():
    """验证规则：分词同时产出中文二元组 / 单字、英文小写整词与去千分位数字串，且同一输入结果确定。"""
    # 一句话里塞齐中文、千分位数字、英文缩写三种形态，一次断言多种切法
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
    """验证规则：关键词强命中的块必须排在首位（证明 BM25 关键词通道确实生效）。"""
    # 取一个高度特异的词组，使关键词通道的排序结果可预期
    results = retriever.search("控股股东股权质押比例", top_k=3)
    assert results
    assert "质押" in results[0].text or "质押" in results[0].context


def test_hybrid_search_returns_provenance(retriever):
    """验证契约：每条检索结果都必须带可追溯来源（source_id、章节、text / context）与三路组件分，分数落在 [0,1]。"""
    results = retriever.search("未决诉讼 对外担保", top_k=3)
    assert results
    for item in results:
        assert item.source_id and item.section_title
        assert item.text and item.context
        assert set(item.components) >= {"keyword", "vector", "metadata"}
        assert 0.0 <= item.score <= 1.0 + 1e-9


def test_rerank_weights_are_configurable(chunks):
    """验证规则：重排权重可配置——纯关键词模式下结果按关键词分单调不增排序，纯向量模式也必须能返回结果。"""
    parents, children = chunks
    # 同一份语料建两个只开单通道的检索器，用来隔离关键词与向量各自的贡献
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
    """验证规则：strict=True 时结果必须全部来自目标公司；strict=False 时元数据退化为加权信号，目标公司仍应排首位。"""
    parents, children = chunks
    retriever = HybridRetriever(parents, children)
    # 查询词取自银行年报专属口径，便于断言结果只可能是银行资料
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
    """验证规则：多查询合并后结果数不少于单查询，且子块不重复（每个 child 只保留最优分）。"""
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
    """验证规则：年报抽取的数值、单位、来源文件与章节定位都必须正确（结论可回溯到原文）。"""
    fact = fact_store.get("示例科技股份有限公司", "营业收入", 2024)
    assert fact is not None
    assert fact.value == pytest.approx(1286400.0)
    assert fact.unit == "万元"
    assert fact.source_id == "EX-TECH-2024-AR"
    assert "关键财务数据" in fact.section_title


def test_fact_store_parses_prior_year(fact_store):
    """验证规则：同一指标的上期（2023）对比数也必须被抽取，供同比口径使用。"""
    prior = fact_store.get("示例科技股份有限公司", "营业收入", 2023)
    assert prior is not None and prior.value == pytest.approx(1102300.0)


def test_fact_store_distinguishes_periods(fact_store):
    """验证规则：同一公司同一年的不同报告期（年度 vs 三季度）必须按 period 区分，来源文件与数值各不相同。"""
    # 显式传 period：不传时会命中默认口径，无法验证「按报告期区分」这一能力
    annual = fact_store.get("示例科技股份有限公司", "营业收入", 2024, period="年度")
    q3 = fact_store.get("示例科技股份有限公司", "营业收入", 2024, period="三季度")
    assert annual.source_id == "EX-TECH-2024-AR"
    assert q3.source_id == "EX-TECH-2024-Q3"
    assert q3.value == pytest.approx(892400.0)


def test_fact_store_resolves_company_short_name(fact_store):
    """验证规则：公司简称可唯一归一到全称，未知主体必须返回 None（不得就近误匹配到相似公司）。"""
    assert fact_store.resolve_company("示例科技") == "示例科技股份有限公司"
    assert fact_store.resolve_company("示例智造银行") == "示例智造银行股份有限公司"
    assert fact_store.resolve_company("不存在的公司") is None


def test_fact_store_metric_aliases(fact_store):
    """验证规则：口语化指标别名（营收 / 净资产）必须归一到规范指标名。"""
    alias = fact_store.get("示例科技股份有限公司", "营收", 2024)
    assert alias is not None and alias.metric == "营业收入"
    equity = fact_store.get("示例科技股份有限公司", "净资产", 2024)
    assert equity is not None and equity.metric == "所有者权益"


def test_fact_store_covers_bank_regulatory_metrics(fact_store):
    """验证规则：银行专属监管指标（不良贷款率、拨备覆盖率）也必须进入事实库，而不只有工商企业指标。"""
    npl = fact_store.get("示例智造银行股份有限公司", "不良贷款率", 2024)
    coverage = fact_store.get("示例智造银行股份有限公司", "拨备覆盖率", 2024)
    assert npl is not None and npl.value == pytest.approx(1.42)
    assert coverage is not None and coverage.value == pytest.approx(218.50)


def test_documents_are_fictional_and_carry_disclaimer(documents):
    """验证合规要求：资料库必须由 3 份虚构主体文档组成、每份都带免责声明，且正文不得出现任何真实主体名称。"""
    # 黑名单挑的是最容易被误写进示例数据的真实机构名，命中即视为合规失败
    assert len(documents) == 3
    forbidden = ("工商银行", "招商银行", "中信证券", "贵州茅台", "宁德时代", "腾讯", "阿里巴巴")
    for doc in documents:
        assert doc.meta.get("disclaimer"), f"{doc.source_id} 缺少免责声明"
        for name in forbidden:
            assert name not in doc.raw, f"{doc.source_id} 出现了真实主体名称：{name}"
        assert "示例" in doc.company
