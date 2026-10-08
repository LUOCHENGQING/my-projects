"""检索子包（RAG 全链路的「召回 + 精排」层）。

    在 RAG 全链路中的位置
    ---------------------
        切分 / 索引（src/chunking、src/index）→ **本子包：检索** → 生成（src/generate）→ 服务（src/service）
    上游：切好的父块 / 子块（chunking）与建好的 BM25 倒排、向量库（index、config）；
    下游：拿到 Evidence 去拼 prompt 的生成模块，以及 eval/、scripts/ 里的评测与 A/B 脚本。

    职责
    ----
    把「用户问题」变成「少量、可追溯、按相关性排序的证据」，
    并且让中间每一步（召回 / 去重 / 重排 / 回溯）都**可见、可测、可回放**。

    子模块分工
    ----------
    router      问题类型路由（条款 / 案例 / 指标 / 通用）+ 元数据推断 + 查询改写
    hybrid      三路召回（BM25 / 稠密 / 稀疏）+ RRF 融合 + 元数据过滤 + 近似去重
    rerank      交叉编码器重排（bge-reranker 语义）+ 与 RRF 分加权融合
    pipeline    召回 → 去重 → 重排 → 父块回溯 → 组装证据，并提供对照组基线

    对外关键对象与数据流
    --------------------
    输入：`HybridRetriever(parents, children, backend=...)` —— 父块、子块与嵌入后端
    过程：`build_query_plan` 产出 `QueryPlan`（路由 / 权重 / 过滤 / 查询变体）
          → `HybridRetriever.retrieve` 产出 `List[Candidate]`（含三路分数与名次）
          → `deduplicate` 合并重复块 → `rerank_candidates` 产出 `List[RerankedItem]`
    输出：`RetrievalPipeline.run` 产出 `RetrievalResult`（内含 `List[Evidence]`，即最终给 LLM 的证据）

一句话总结这个子包的设计取舍：**召回看「不漏」，重排看「排得准」，
两者之间的融合只用名次不用分数**（RRF），这样三路召回的量纲差异不会污染结果。
"""

from __future__ import annotations

# 再导出：调用方只依赖检索层的这组公开名字，不必知道子模块的文件布局，
# 因此后续在子模块内部重构（拆文件、改私有实现）不会波及下游代码。
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
    # 召回层：候选对象、三路召回器、近似去重
    "Candidate",
    "HybridRetriever",
    "deduplicate",
    # 证据与流水线（对外的唯一入口是 RetrievalPipeline）
    "Evidence",
    "RetrievalResult",
    "RetrievalPipeline",
    # 重排后端与融合
    "Reranker",
    "LocalCrossEncoder",
    "BGEReranker",
    "RerankFeatures",
    "RerankedItem",
    "get_reranker",
    "rerank_candidates",
    # 路由与检索计划
    "QueryPlan",
    "build_query_plan",
    "classify_question",
    "infer_filters",
    "expand_queries",
    "QUERY_TYPES",
]
