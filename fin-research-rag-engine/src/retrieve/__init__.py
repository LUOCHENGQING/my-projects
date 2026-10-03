"""检索子包。

    router      问题类型路由（条款 / 案例 / 指标 / 通用）+ 元数据推断 + 查询改写
    hybrid      三路召回（BM25 / 稠密 / 稀疏）+ RRF 融合 + 元数据过滤 + 近似去重
    rerank      交叉编码器重排（bge-reranker 语义）+ 与 RRF 分加权融合
    pipeline    召回 → 去重 → 重排 → 父块回溯 → 组装证据，并提供对照组基线

一句话总结这个子包的设计取舍：**召回看「不漏」，重排看「排得准」，
两者之间的融合只用名次不用分数**（RRF），这样三路召回的量纲差异不会污染结果。
"""

from __future__ import annotations

from .hybrid import Candidate, HybridRetriever, deduplicate
from .pipeline import Evidence, RetrievalPipeline, RetrievalResult
from .rerank import (
    BGEReranker,
    LocalCrossEncoder,
    RerankFeatures,
    RerankedItem,
    Reranker,
    get_reranker,
    rerank_candidates,
)
from .router import (
    QUERY_TYPES,
    QueryPlan,
    build_query_plan,
    classify_question,
    expand_queries,
    infer_filters,
)

__all__ = [
    "Candidate",
    "HybridRetriever",
    "deduplicate",
    "Evidence",
    "RetrievalResult",
    "RetrievalPipeline",
    "Reranker",
    "LocalCrossEncoder",
    "BGEReranker",
    "RerankFeatures",
    "RerankedItem",
    "get_reranker",
    "rerank_candidates",
    "QueryPlan",
    "build_query_plan",
    "classify_question",
    "infer_filters",
    "expand_queries",
    "QUERY_TYPES",
]
