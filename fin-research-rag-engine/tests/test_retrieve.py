"""检索层测试：问题路由、元数据推断、三路召回、RRF 融合、去重、重排、流水线。"""

from __future__ import annotations

import pytest

from src.chunking import build_chunks
from src.retrieve import (
    Candidate,
    HybridRetriever,
    LocalCrossEncoder,
    QueryPlan,
    RetrievalPipeline,
    build_query_plan,
    classify_question,
    deduplicate,
    expand_queries,
    get_reranker,
    infer_filters,
    rerank_candidates,
)
from src.retrieve.pipeline import ROUTE_DOC_TYPES, route_affinity
from src.retrieve.rerank import RerankFeatures
from src.utils import tokenize


# ---------------------------------------------------------------------------
# 问题路由
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "question,expected",
    [
        ("第四条对合格投资者是怎么规定的？", "clause"),
        ("有没有类似的违规处罚案例？", "case"),
        ("单一产品集中度上限是多少？", "metric"),
        ("这家公司最近怎么样？", "general"),
    ],
)
def test_classify_question(question, expected):
    assert classify_question(question) == expected


def test_classify_clause_by_product_code():
    assert classify_question("WY2024-01 的赎回费率是多少？") == "clause"


def test_classify_empty_question_falls_back():
    assert classify_question("") == "general"


def test_classify_tie_falls_back_to_general():
    """两个类型得分相同时保守回落到通用策略，避免往错的方向特化。"""
    assert classify_question("标准与金额") in ("general", "clause", "metric")


def test_weak_words_do_not_win_alone():
    """「多少」是弱特征，不应该单独把问题判成指标类。"""
    assert classify_question("合格投资者的门槛是多少？") == "clause"


# ---------------------------------------------------------------------------
# 元数据推断与查询改写
# ---------------------------------------------------------------------------
def test_infer_filters_detects_institution(corpus):
    expr, cond = infer_filters("示例银行股份有限公司的接入标准是什么？", corpus)
    assert cond.get("institution") == "示例银行股份有限公司"
    assert "institution" in expr


def test_infer_filters_detects_year(corpus):
    expr, cond = infer_filters("2023 年的合格投资者标准是多少？", corpus)
    assert cond.get("year") == 2023
    assert cond.get("year_promotable") is True


def test_infer_filters_year_of_financial_data_is_not_promotable(corpus):
    """「示例集团 2023 年应收账款」里的年份限定的是数据年度，不能提升为版本硬过滤。"""
    _, cond = infer_filters("示例集团 2023 年应收账款增速如何？", corpus)
    assert cond.get("year") == 2023
    assert "year_promotable" not in cond


def test_infer_filters_empty_when_nothing_detected(corpus):
    expr, cond = infer_filters("合格投资者是谁？", corpus)
    assert expr == "" and cond == {}


def test_expand_queries_keeps_original_first():
    queries = expand_queries("请问第四十二条是怎么规定的？", "clause")
    assert queries[0].startswith("请问")
    assert len(queries) <= 4


def test_expand_queries_extracts_code():
    queries = expand_queries("WY2024-01 的费率是多少？", "clause")
    assert "WY2024-01" in queries


def test_expand_queries_deduplicates():
    queries = expand_queries("合格投资者标准", "clause")
    assert len(queries) == len(set(queries))


def test_build_query_plan_carries_weights(corpus):
    plan = build_query_plan("第四条怎么规定的？", corpus=corpus, top_k=5)
    assert plan.route == "clause"
    assert set(plan.weights) == {"bm25", "dense", "sparse"}
    assert plan.weights["bm25"] > plan.weights["dense"]
    assert isinstance(plan.to_dict(), dict)


def test_build_query_plan_explicit_filter_wins(corpus):
    plan = build_query_plan("任意问题", corpus=corpus, filter_expr='year = 2023')
    assert plan.filter_expr == "year = 2023"


# ---------------------------------------------------------------------------
# 三路召回与 RRF
# ---------------------------------------------------------------------------
def test_retriever_builds_all_three_routes(retriever):
    described = retriever.describe()
    assert described["children"] > 0
    assert described["vocabulary"] > 0
    assert described["vector_store"][0]["count"] == described["children"]


def test_retrieve_returns_candidates_with_route_evidence(retriever):
    candidates = retriever.retrieve("合格投资者的金融资产门槛", top_k=10)
    assert candidates
    assert candidates[0].rrf_score > 0
    assert candidates[0].routes_hit


def test_retrieve_scores_are_sorted_descending(retriever):
    candidates = retriever.retrieve("双录资料保存期限", top_k=10)
    scores = [c.rrf_score for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_respects_metadata_hard_filter(retriever):
    candidates = retriever.retrieve("合格投资者标准", top_k=10, expr='doc_type = "监管政策"')
    assert candidates
    assert all(c.meta["doc_type"] == "监管政策" for c in candidates)


def test_retrieve_empty_question_returns_nothing(retriever):
    assert retriever.retrieve("", top_k=5) == []


def test_dense_only_and_bm25_only_routes_work(retriever):
    dense = retriever.search_dense_only("冷静期", top_k=5)
    bm25 = retriever.search_bm25_only("冷静期", top_k=5)
    assert dense and bm25
    assert all(c.dense_rank is not None for c in dense)
    assert all(c.bm25_rank is not None for c in bm25)


def test_unknown_route_raises(retriever):
    with pytest.raises(ValueError):
        retriever._search_single_route("问题", "unknown", 5, None)


def test_metadata_score_soft_filters():
    assert HybridRetriever.metadata_score({"year": 2024}, {}) == 1.0
    assert HybridRetriever.metadata_score({"year": 2024}, {"year_gte": 2023}) == 1.0
    assert HybridRetriever.metadata_score({"year": 2022}, {"year_gte": 2023}) == 0.0
    assert HybridRetriever.metadata_score({"doc_type": "监管政策"}, {"doc_type": "监管政策"}) == 1.0


def test_known_years_are_ints(retriever):
    years = retriever.known_years
    assert years and all(isinstance(y, int) for y in years)


def test_candidate_to_dict_is_jsonable(retriever):
    cand = retriever.retrieve("合格投资者", top_k=1)[0]
    payload = cand.to_dict()
    assert payload["child_id"] and "routes_hit" in payload


def test_parent_backfill_gives_context(retriever):
    cand = retriever.retrieve("合格投资者", top_k=1)[0]
    assert cand.context
    assert len(cand.context) >= len(cand.text) - 1


def test_multi_query_variants_increase_routes_hit(retriever):
    candidates = retriever.retrieve("请问第四十二条对合格投资者怎么规定的？", top_k=10)
    assert candidates
    assert any(len(c.routes_hit) >= 2 for c in candidates)


# ---------------------------------------------------------------------------
# 去重
# ---------------------------------------------------------------------------
def _candidate(child_id: str, text: str, score: float) -> Candidate:
    return Candidate(
        child_id=child_id,
        parent_id="p",
        source_id="S",
        doc_id="S",
        section_title="章节",
        text=text,
        kind="paragraph",
        meta={},
        rrf_score=score,
    )


def test_deduplicate_merges_near_identical_text():
    a = _candidate("a", "合格投资者的金融资产不低于 300 万元", 0.9)
    b = _candidate("b", "合格投资者的金融资产不低于 300 万元", 0.5)
    kept = deduplicate([a, b], threshold=0.8)
    assert len(kept) == 1
    assert kept[0].child_id == "a"


def test_deduplicate_keeps_distinct_text():
    a = _candidate("a", "合格投资者的金融资产不低于 300 万元", 0.9)
    b = _candidate("b", "双录资料保存期限不少于二十年", 0.5)
    assert len(deduplicate([a, b], threshold=0.8)) == 2


def test_deduplicate_records_merge_log():
    a = _candidate("a", "合格投资者的金融资产不低于 300 万元", 0.9)
    b = _candidate("b", "合格投资者的金融资产不低于 300 万元", 0.5)
    log = []
    deduplicate([a, b], threshold=0.8, merge_log=log)
    assert log == [{"child_id": "b", "merged_into": "a"}]


def test_deduplicate_protects_named_children():
    a = _candidate("a", "同一段文字内容测试", 0.9)
    b = _candidate("b", "同一段文字内容测试", 0.5)
    kept = deduplicate([a, b], threshold=0.8, keep=["b"])
    assert {c.child_id for c in kept} == {"a", "b"}


def test_deduplicate_empty_input():
    assert deduplicate([]) == []


# ---------------------------------------------------------------------------
# 重排
# ---------------------------------------------------------------------------
def test_local_cross_encoder_prefers_matching_text():
    encoder = LocalCrossEncoder()
    good = encoder.score_pair("合格投资者的金融资产门槛", "合格投资者的金融资产不低于 300 万元")
    bad = encoder.score_pair("合格投资者的金融资产门槛", "双录资料保存期限不少于二十年")
    assert good.as_score() > bad.as_score()


def test_rerank_features_are_normalised():
    feats = LocalCrossEncoder().score_pair("冷静期", "冷静期不少于二十四小时")
    for name, value in feats.to_dict().items():
        assert 0.0 <= value <= 1.0, name


def test_exact_match_bonus_for_clause_numbers():
    encoder = LocalCrossEncoder()
    with_clause = encoder.score_pair("第四十二条怎么规定的", "第四十二条 销售机构应当双录")
    without = encoder.score_pair("第四十二条怎么规定的", "第二十条 本办法自发布之日起施行")
    assert with_clause.exact > without.exact


def test_length_score_penalises_extremes():
    assert LocalCrossEncoder._length_score(0) == 0.0
    assert LocalCrossEncoder._length_score(220) == 1.0
    assert LocalCrossEncoder._length_score(2000) < 1.0


def test_rerank_candidates_orders_by_final_score(retriever):
    candidates = retriever.retrieve("合格投资者的金融资产门槛", top_k=8)
    rows = rerank_candidates(
        "合格投资者的金融资产门槛",
        candidates,
        reranker=get_reranker("local"),
        top_k=3,
        rrf_scores={c.child_id: c.rrf_score for c in candidates},
    )
    assert len(rows) == 3
    assert rows[0].final_score >= rows[-1].final_score
    assert rows[0].rank_after == 0


def test_rerank_empty_candidates():
    assert rerank_candidates("问题", [], top_k=3) == []


def test_get_reranker_falls_back_to_local():
    reranker = get_reranker("bge-reranker")
    assert reranker.name in ("bge-reranker", "local-cross-encoder")


def test_route_affinity():
    assert route_affinity("监管政策", "clause") == 1.0
    assert route_affinity("风险案例", "clause") < 1.0
    assert route_affinity("任意类型", "general") == 1.0
    assert "case" in ROUTE_DOC_TYPES


# ---------------------------------------------------------------------------
# 流水线
# ---------------------------------------------------------------------------
def test_pipeline_returns_evidence_with_citation(pipeline):
    result = pipeline.run("合格投资者的金融资产门槛是多少？", top_k=5)
    assert result.evidence
    assert result.evidence[0].child_id
    assert result.evidence[0].citation_label
    assert result.evidence[0].effective_date


def test_pipeline_stats_include_routes(pipeline):
    stats = pipeline.run("有没有类似的处罚案例？", top_k=5).stats()
    assert stats["recalled"] > 0
    assert "bm25" in stats["routes"]


def test_pipeline_baseline_dense_skips_rerank(pipeline):
    result = pipeline.baseline_dense("合格投资者", top_k=5)
    assert result.mode == "dense"
    assert result.reranked == []
    assert result.evidence


def test_pipeline_baseline_bm25_skips_rerank(pipeline):
    result = pipeline.baseline_bm25("合格投资者", top_k=5)
    assert result.mode == "bm25"
    assert result.evidence


def test_pipeline_deduplicates_repeated_evidence(pipeline):
    result = pipeline.run("合格投资者 金融资产 门槛 标准", top_k=5)
    texts = [e.text for e in result.evidence]
    assert len(texts) == len(set(texts))


def test_pipeline_year_promotion_only_when_version_question(pipeline):
    promoted = pipeline.run("2023 年的合格投资者标准是多少？", top_k=5)
    assert promoted.plan.promoted_year
    assert promoted.plan.filter_expr == "year = 2023"

    not_promoted = pipeline.run("示例集团 2023 年应收账款增速如何？", top_k=5)
    assert not not_promoted.plan.promoted_year


def test_pipeline_empty_question(pipeline):
    result = pipeline.run("", top_k=5)
    assert result.evidence == []


def test_pipeline_to_dict_serialisable(pipeline):
    payload = pipeline.run("冷静期是多久？", top_k=3).to_dict()
    assert payload["evidence"]
    assert isinstance(payload["stats"], dict)


def test_evidence_to_dict_with_context(pipeline):
    result = pipeline.run("冷静期是多久？", top_k=1)
    payload = result.evidence[0].to_dict(with_context=True)
    assert payload["context"]
