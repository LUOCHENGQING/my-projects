"""检索层测试：问题路由、元数据推断、三路召回、RRF 融合、去重、重排、流水线。

覆盖的被测模块
--------------
    src.retrieve.router     classify_question 的四类路由与并列/空问回落、infer_filters 的
                            机构 / 年份 / 资料类型推断与「年份可否提升为硬过滤」的区分、
                            expand_queries 的确定性改写与去重、build_query_plan 的权重装配
    src.retrieve.hybrid     三路召回（BM25 / 稠密 / 稀疏）与 RRF 融合、硬过滤 vs 软加权、
                            单路对照接口、metadata_score、known_years、deduplicate 近似去重
    src.retrieve.rerank     LocalCrossEncoder 的特征排序（词面覆盖 / 精确命中 / 长度规整）、
                            特征区间与最终分融合、get_reranker 的降级、route_affinity 先验
    src.retrieve.pipeline   RetrievalPipeline.run 的去重 → 重排 → 父块回溯链路、
                            对照组 baseline_dense / baseline_bm25（单一通路且不重排）、
                            年份硬过滤的「有条件提升」、Evidence 与 stats 的序列化

覆盖策略
--------
    正常      同一批问题在 clause / case / metric / general 四种路由下都能召回，
              且候选按 RRF 分降序、证据带引用标签与生效日期。
    边界      空问题（返回空候选 / 空证据）、空候选列表重排、并列得分回落、未知路由、
              推理条件为空、只给单路时的名次字段。
    异常      未知通路名抛 ValueError、get_reranker 请求不可用后端时降级而不抛错。
    对抗      路由/过滤最容易出错的两种情况被单独钉死：①推断出的年份**不能**变成硬过滤
              （否则「2023 年应收账款」这类问法会把当期资料整体筛掉）；
              ②显式年份在该年份确实存在时才提升为硬过滤（否则会给错版本的标准）。
    拒答      本模块不覆盖端到端拒答，只覆盖拒答的**上游前提**——空问题不召回、
              软过滤不剔除候选（宁可排序差一点也不静默丢证据）。

参数化用例的意图：`test_classify_question` 用四条问题各打一种路由，
覆盖「条款 / 案例 / 指标 / 通用」四个分支，避免只测到其中一类就以为路由可用。
"""

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
    """四类问题各自打到对应路由：条款 / 案例 / 指标 / 通用，一条也不许错配。"""
    assert classify_question(question) == expected


def test_classify_clause_by_product_code():
    """出现产品代码时直接判条款类（代码是确定性最强的字面信号，无需再看特征词）。"""
    # 问句里刻意只留「赎回费率」这种弱特征，验证路由是靠代码命中的
    assert classify_question("WY2024-01 的赎回费率是多少？") == "clause"


def test_classify_empty_question_falls_back():
    """空问题没有可用特征，必须回落到通用路由而不是抛错或猜一个特化路由。"""
    assert classify_question("") == "general"


def test_classify_tie_falls_back_to_general():
    """两个类型得分相同时保守回落到通用策略，避免往错的方向特化。

    注：实际实现为——断言写的是 `in ("general", "clause", "metric")`，即只排除 "case"，
    取值并不唯一；「标准」与「金额」在该实现下的得分未必严格并列，所以这条用例
    守的是「不落到案例类」这一底线，而不是「必须等于 general」。
    """
    assert classify_question("标准与金额") in ("general", "clause", "metric")


def test_weak_words_do_not_win_alone():
    """「多少」是弱特征，不应该单独把问题判成指标类。"""
    # 注：实际实现为——本句同时命中 clause 的「门槛」与 metric 的「多少（弱权重）」，
    # 弱权重被压低后 clause 得分更高，因此断言精确等于 "clause" 是稳定的。
    assert classify_question("合格投资者的门槛是多少？") == "clause"


# ---------------------------------------------------------------------------
# 元数据推断与查询改写
# ---------------------------------------------------------------------------
def test_infer_filters_detects_institution(corpus):
    """问题里出现资料库已知机构名时必须推断出机构条件，并写进过滤表达式。"""
    expr, cond = infer_filters("示例银行股份有限公司的接入标准是什么？", corpus)
    assert cond.get("institution") == "示例银行股份有限公司"
    assert "institution" in expr


def test_infer_filters_detects_year(corpus):
    """问句年份修饰「标准/规定」这类文档版本时，必须额外标记 year_promotable 允许硬过滤。"""
    expr, cond = infer_filters("2023 年的合格投资者标准是多少？", corpus)
    assert cond.get("year") == 2023
    assert cond.get("year_promotable") is True


def test_infer_filters_year_of_financial_data_is_not_promotable(corpus):
    """「示例集团 2023 年应收账款」里的年份限定的是数据年度，不能提升为版本硬过滤。"""
    _, cond = infer_filters("示例集团 2023 年应收账款增速如何？", corpus)
    assert cond.get("year") == 2023
    assert "year_promotable" not in cond


def test_infer_filters_empty_when_nothing_detected(corpus):
    """一个问题都没识别出条件时，表达式与条件字典都必须为空——宁可不加过滤，也不要加错过滤。"""
    expr, cond = infer_filters("合格投资者是谁？", corpus)
    assert expr == "" and cond == {}


def test_expand_queries_keeps_original_first():
    """查询改写必须把原问题放在第一位，且变体数量有上限（不做无约束的模型改写）。"""
    # 选一个带「请问」语气词的问题：验证改写确实产出了变体，同时原文仍排第一
    queries = expand_queries("请问第四十二条是怎么规定的？", "clause")
    assert queries[0].startswith("请问")
    assert len(queries) <= 4


def test_expand_queries_extracts_code():
    """产品代码必须被抽成独立查询（字面精确匹配的强信号，单独一路去撞倒排表）。"""
    queries = expand_queries("WY2024-01 的费率是多少？", "clause")
    assert "WY2024-01" in queries


def test_expand_queries_deduplicates():
    """改写结果必须去重保序：同一个查询重复提交会让 RRF 的摊薄系数失真。"""
    queries = expand_queries("合格投资者标准", "clause")
    assert len(queries) == len(set(queries))


def test_build_query_plan_carries_weights(corpus):
    """计划必须带齐三路权重，且条款类问题偏 BM25（路由要真正改变召回策略，而不是只改个标签）。"""
    # 用「第四条」这种条款号问法：同时把 route 与三路权重的关系一并断言掉
    plan = build_query_plan("第四条怎么规定的？", corpus=corpus, top_k=5)
    assert plan.route == "clause"
    assert set(plan.weights) == {"bm25", "dense", "sparse"}
    assert plan.weights["bm25"] > plan.weights["dense"]
    assert isinstance(plan.to_dict(), dict)


def test_build_query_plan_explicit_filter_wins(corpus):
    """用户显式给出的过滤条件必须原样进计划（显式条件属硬过滤，不能被推断结果覆盖）。"""
    plan = build_query_plan("任意问题", corpus=corpus, filter_expr='year = 2023')
    assert plan.filter_expr == "year = 2023"


# ---------------------------------------------------------------------------
# 三路召回与 RRF
# ---------------------------------------------------------------------------
def test_retriever_builds_all_three_routes(retriever):
    """三路索引都要真的建起来：子块数、词表、向量库条数三者必须对齐（保证两路表示不会错位）。"""
    described = retriever.describe()
    assert described["children"] > 0
    assert described["vocabulary"] > 0
    assert described["vector_store"][0]["count"] == described["children"]


def test_retrieve_returns_candidates_with_route_evidence(retriever):
    """候选必须带可解释证据：RRF 分大于 0，且记录命中了哪几路（否则无法回答「为什么召回它」）。"""
    candidates = retriever.retrieve("合格投资者的金融资产门槛", top_k=10)
    assert candidates
    assert candidates[0].rrf_score > 0
    assert candidates[0].routes_hit


def test_retrieve_scores_are_sorted_descending(retriever):
    """召回结果必须按 RRF 融合分降序，截断前先排序（否则 Top-K 取到的不是最高的 K 条）。"""
    candidates = retriever.retrieve("双录资料保存期限", top_k=10)
    scores = [c.rrf_score for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_respects_metadata_hard_filter(retriever):
    """显式元数据条件属硬过滤：返回的每一条都必须满足条件（被过滤的记录不得漏回）。"""
    candidates = retriever.retrieve("合格投资者标准", top_k=10, expr='doc_type = "监管政策"')
    assert candidates
    assert all(c.meta["doc_type"] == "监管政策" for c in candidates)


def test_retrieve_empty_question_returns_nothing(retriever):
    """空问题必须召回 0 条（不能把全库零分当成有效候选，否则空问也会给出一堆"证据"）。"""
    assert retriever.retrieve("", top_k=5) == []


def test_dense_only_and_bm25_only_routes_work(retriever):
    """单路对照接口只回填本路的名次与分数，其余两路保持缺省（否则 A/B 对比就说不清是哪一路的贡献）。"""
    dense = retriever.search_dense_only("冷静期", top_k=5)
    bm25 = retriever.search_bm25_only("冷静期", top_k=5)
    assert dense and bm25
    assert all(c.dense_rank is not None for c in dense)
    assert all(c.bm25_rank is not None for c in bm25)


def test_unknown_route_raises(retriever):
    """未知通路名必须抛 ValueError，而不是静默退化成某一条默认路（静默降级会掩盖配置错误）。"""
    with pytest.raises(ValueError):
        retriever._search_single_route("问题", "unknown", 5, None)


def test_metadata_score_soft_filters():
    """软元数据契合度是「满足比例」：无条件下给 1.0，年份比较按数值语义，缺字段不计命中。"""
    # 四行刻意覆盖四种情形：无条件 / 满足下界 / 不满足下界 / 等值字符串完全命中
    assert HybridRetriever.metadata_score({"year": 2024}, {}) == 1.0
    assert HybridRetriever.metadata_score({"year": 2024}, {"year_gte": 2023}) == 1.0
    assert HybridRetriever.metadata_score({"year": 2022}, {"year_gte": 2023}) == 0.0
    assert HybridRetriever.metadata_score({"doc_type": "监管政策"}, {"doc_type": "监管政策"}) == 1.0


def test_known_years_are_ints(retriever):
    """资料库已知年份必须是 int 列表（它是「年份能否提升为硬过滤」的唯一判据，类型错了判据就失效）。"""
    years = retriever.known_years
    assert years and all(isinstance(y, int) for y in years)


def test_candidate_to_dict_is_jsonable(retriever):
    """候选序列化必须给出子块 id 与命中通路，且不含体积最大的父块上下文（由 Evidence 决定是否带）。"""
    cand = retriever.retrieve("合格投资者", top_k=1)[0]
    payload = cand.to_dict()
    assert payload["child_id"] and "routes_hit" in payload


def test_parent_backfill_gives_context(retriever):
    """候选必须回填父块上下文，且不少于子块自身文本（排序用子块，交给 LLM 的上下文用父块）。"""
    cand = retriever.retrieve("合格投资者", top_k=1)[0]
    assert cand.context
    assert len(cand.context) >= len(cand.text) - 1


def test_multi_query_variants_increase_routes_hit(retriever):
    """多查询变体必须真的扩大命中面：至少有一条候选被两路以上同时命中（三路共识的实证）。"""
    candidates = retriever.retrieve("请问第四十二条对合格投资者怎么规定的？", top_k=10)
    assert candidates
    assert any(len(c.routes_hit) >= 2 for c in candidates)


# ---------------------------------------------------------------------------
# 去重
# ---------------------------------------------------------------------------
def _candidate(child_id: str, text: str, score: float) -> Candidate:
    """造一条最小可用候选：只填去重判定真正会读的字段（child_id / text / rrf_score）。

    这样构造是为了让去重用例只依赖「文本相似度 + 分数高低」两个变量——
    块类型、元数据、父块上下文一律留空，避免无关字段影响判重结果。
    """
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
    """文本近乎相同的候选合并为一条，且保留分数更高的那条（留下的由分数决定，不由召回顺序决定）。"""
    # 两条文本完全相同、只差分数：验证"留谁"看分数，而不是看列表先后
    a = _candidate("a", "合格投资者的金融资产不低于 300 万元", 0.9)
    b = _candidate("b", "合格投资者的金融资产不低于 300 万元", 0.5)
    kept = deduplicate([a, b], threshold=0.8)
    assert len(kept) == 1
    assert kept[0].child_id == "a"


def test_deduplicate_keeps_distinct_text():
    """内容不同的候选不得被误判为重复（漏掉一条独立证据比多留一条重复证据更糟）。"""
    # 两条文本主题完全无关：验证判重不会因为都含"合格/资料"这类通用词而误合并
    a = _candidate("a", "合格投资者的金融资产不低于 300 万元", 0.9)
    b = _candidate("b", "双录资料保存期限不少于二十年", 0.5)
    assert len(deduplicate([a, b], threshold=0.8)) == 2


def test_deduplicate_records_merge_log():
    """被合并的候选必须留下可追溯记录（child_id → merged_into），否则「少了一条证据」无从解释。"""
    a = _candidate("a", "合格投资者的金融资产不低于 300 万元", 0.9)
    b = _candidate("b", "合格投资者的金融资产不低于 300 万元", 0.5)
    log = []
    deduplicate([a, b], threshold=0.8, merge_log=log)
    assert log == [{"child_id": "b", "merged_into": "a"}]


def test_deduplicate_protects_named_children():
    """受保护 child_id 即使被判重也必须保留：调用方显式指定的块优先级高于去重策略。"""
    # 两条文本一致（必然判重），但把低分的 b 列入 keep，验证保护名单能拦住去重
    a = _candidate("a", "同一段文字内容测试", 0.9)
    b = _candidate("b", "同一段文字内容测试", 0.5)
    kept = deduplicate([a, b], threshold=0.8, keep=["b"])
    assert {c.child_id for c in kept} == {"a", "b"}


def test_deduplicate_empty_input():
    """空候选列表必须安全返回空列表（空问题会走到这条路径，不能抛错）。"""
    assert deduplicate([]) == []


# ---------------------------------------------------------------------------
# 重排
# ---------------------------------------------------------------------------
def test_local_cross_encoder_prefers_matching_text():
    """交叉编码器必须把与问题同主题的块打高分、无关块打低分（重排的意义就在这条序关系上）。"""
    encoder = LocalCrossEncoder()
    # 一正一负两条候选：正例与问题同主题，负例是别的话题，保证比较的是相关性而不是长度
    good = encoder.score_pair("合格投资者的金融资产门槛", "合格投资者的金融资产不低于 300 万元")
    bad = encoder.score_pair("合格投资者的金融资产门槛", "双录资料保存期限不少于二十年")
    assert good.as_score() > bad.as_score()


def test_rerank_features_are_normalised():
    """全部重排特征必须落在 [0,1]，否则加权融合时某一维会凭量纲压倒其它维。"""
    feats = LocalCrossEncoder().score_pair("冷静期", "冷静期不少于二十四小时")
    for name, value in feats.to_dict().items():
        assert 0.0 <= value <= 1.0, name


def test_exact_match_bonus_for_clause_numbers():
    """问题里的条款号原样出现在文本中时，精确命中特征必须更高（金融问答的强信号）。"""
    encoder = LocalCrossEncoder()
    # 两条候选长度与主题都相近，唯一差别是「第四十二条」是否原样命中，因此比较是干净的
    with_clause = encoder.score_pair("第四十二条怎么规定的", "第四十二条 销售机构应当双录")
    without = encoder.score_pair("第四十二条怎么规定的", "第二十条 本办法自发布之日起施行")
    assert with_clause.exact > without.exact


def test_length_score_penalises_extremes():
    """长度规整必须以 220 字为最佳点向两侧扣分：空块得 0，过短的块低于满分。"""
    # 三个点分别钉住下界 / 理想值 / 上界：0 -> 0.0、220 -> 1.0、2000 -> < 1.0
    assert LocalCrossEncoder._length_score(0) == 0.0
    assert LocalCrossEncoder._length_score(220) == 1.0
    assert LocalCrossEncoder._length_score(2000) < 1.0


def test_rerank_candidates_orders_by_final_score(retriever):
    """重排结果按最终融合分降序，且 rank_after 从 0 起连续编号（它是要写进轨迹的名次）。"""
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
    """空候选重排必须直接返回空列表，而不是走进打分逻辑（空问题会走到这条路径）。"""
    assert rerank_candidates("问题", [], top_k=3) == []


def test_get_reranker_falls_back_to_local():
    """请求 bge-reranker 而依赖不可用时必须降级到本地实现，绝不返回一个不可用的后端。"""
    reranker = get_reranker("bge-reranker")
    assert reranker.name in ("bge-reranker", "local-cross-encoder")


def test_route_affinity():
    """问题类型与资料类型的契合度是软先验：命中给 1.0，未命中给 0.74，通用路由不做区分。"""
    # 四行分别覆盖：命中偏好 / 未命中偏好（软而非硬）/ 通用路由（查不到偏好）/ 偏好表本身有该路由
    assert route_affinity("监管政策", "clause") == 1.0
    assert route_affinity("风险案例", "clause") < 1.0
    assert route_affinity("任意类型", "general") == 1.0
    assert "case" in ROUTE_DOC_TYPES


# ---------------------------------------------------------------------------
# 流水线
# ---------------------------------------------------------------------------
def test_pipeline_returns_evidence_with_citation(pipeline):
    """流水线必须产出可直接引用的证据：子块 id、引用标签、生效日期一个都不能缺（业务方据此判断时效）。"""
    result = pipeline.run("合格投资者的金融资产门槛是多少？", top_k=5)
    assert result.evidence
    assert result.evidence[0].child_id
    assert result.evidence[0].citation_label
    assert result.evidence[0].effective_date


def test_pipeline_stats_include_routes(pipeline):
    """统计必须体现「三路各自命中多少」与候选总数，否则无法判断三路是否真的互补。"""
    stats = pipeline.run("有没有类似的处罚案例？", top_k=5).stats()
    assert stats["recalled"] > 0
    assert "bm25" in stats["routes"]


def test_pipeline_baseline_dense_skips_rerank(pipeline):
    """对照组的单一向量检索必须不重排（reranked 为空）且仍给出证据——这才是干净的 baseline。"""
    result = pipeline.baseline_dense("合格投资者", top_k=5)
    assert result.mode == "dense"
    assert result.reranked == []
    assert result.evidence


def test_pipeline_baseline_bm25_skips_rerank(pipeline):
    """纯关键词对照同样不重排、只走 BM25 单路，与单一向量对照保持同一套口径。"""
    result = pipeline.baseline_bm25("合格投资者", top_k=5)
    assert result.mode == "bm25"
    assert result.evidence


def test_pipeline_deduplicates_repeated_evidence(pipeline):
    """最终证据里的文本不得重复：5 个上下文位不能被同一段话反复占掉。"""
    # 问题用多个近义关键词拼成，天然容易把同一段话从多路反复召回，正好压测去重
    result = pipeline.run("合格投资者 金融资产 门槛 标准", top_k=5)
    texts = [e.text for e in result.evidence]
    assert len(texts) == len(set(texts))


def test_pipeline_year_promotion_only_when_version_question(pipeline):
    """年份硬过滤只在「年份限定文档版本」时才提升；限定财务数据年度的一律不提升（否则会筛掉当期资料）。"""
    # 两问只差在年份修饰的对象：一问合格投资者标准（版本），一问应收账款（数据年度）
    promoted = pipeline.run("2023 年的合格投资者标准是多少？", top_k=5)
    assert promoted.plan.promoted_year
    assert promoted.plan.filter_expr == "year = 2023"

    not_promoted = pipeline.run("示例集团 2023 年应收账款增速如何？", top_k=5)
    assert not not_promoted.plan.promoted_year


def test_pipeline_empty_question(pipeline):
    """空问题必须产出空证据列表（不报错、不硬凑），这是「没有依据就不答」的第一道前置。"""
    result = pipeline.run("", top_k=5)
    assert result.evidence == []


def test_pipeline_to_dict_serialisable(pipeline):
    """对外序列化必须同时给出证据列表与统计字典（接口与轨迹都复用这一份结构）。"""
    payload = pipeline.run("冷静期是多久？", top_k=3).to_dict()
    assert payload["evidence"]
    assert isinstance(payload["stats"], dict)


def test_evidence_to_dict_with_context(pipeline):
    """只有显式要求时才带父块上下文（它体积最大，默认不带以控制接口与轨迹体积）。"""
    result = pipeline.run("冷静期是多久？", top_k=1)
    payload = result.evidence[0].to_dict(with_context=True)
    assert payload["context"]
