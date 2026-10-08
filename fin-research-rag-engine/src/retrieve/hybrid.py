"""三路召回 + RRF 融合。

在 RAG 全链路中的位置
---------------------
    切分 / 索引（src/chunking、src/index）→ **本模块：召回** → 重排（retrieve/rerank.py）
        → 父块回溯与证据组装（retrieve/pipeline.py）→ 生成（src/generate）
本模块只负责「尽可能不漏」地把相关子块捞出来并给出初步次序；「排得准」交给 rerank。
上游给它切好的 ParentChunk / ChildChunk 与嵌入后端，
下游（RetrievalPipeline.run）消费它返回的 `List[Candidate]`。

对外关键对象
------------
    Candidate           单条候选：三路分数与名次、RRF 分、父块上下文，全部可追溯
    HybridRetriever     三路召回器（BM25 倒排 + 稠密向量 + 稀疏词权重），含元数据硬过滤
    deduplicate         近似去重：按 token 二元组 Jaccard 合并重复块
被谁调用：retrieve/pipeline.py（主链路与对照组），以及 eval/、scripts/ 里的直接演示。

管线：
    查询（含改写变体）
        ├─ BM25 倒排      → 字面精确匹配（条款号、产品代码、专有名词）
        ├─ 稠密向量检索   → 语义相似（"还能不能买" ≈ "投资者适当性要求"）
        └─ 稀疏词权重检索 → 介于两者之间（保留词权重）
                        ↓
                RRF 融合（按排名，不按分数）
                        ↓
                   Top-N 候选 → 交给重排

三路召回各自的分工与短板
------------------------
    BM25（字面）：分词 + 倒排 + IDF / 长度归一。条款号「第四十二条」、产品代码、
        机构名这类**低频字面信号**只有它能稳稳抓住；短板是**词汇鸿沟**——
        用户说「还能不能买」，原文写「投资者适当性要求」，一个词都对不上，直接 0 分。
    稠密向量（语义）：整句编码成一个向量比余弦，能跨越措辞差异，是「案例类」问题的主力；
        短板是**细粒度失真**——数字、条款号、代码在向量里几乎被抹平，
        「300 万」与「100 万」的余弦可能非常接近，而金融问答里差一个数字就是硬伤。
    稀疏词权重：词级权重（保留词的显式身份，不像稠密那样压成一个点），
        是前两者的中间态，对专有名词、专业术语比较友好；短板是仍依赖分词边界，
        表达不了长距离语义。
三路是**互补**而非三选一：BM25 保字面、稠密保语义、稀疏居中兜底。
也正因为三者量纲完全不同（见下），融合只能用名次。

为什么用 RRF 而不是加权求和
---------------------------
BM25 的分数是无界的（可以到 20+），余弦相似度在 [-1,1]，稀疏点积又是另一个量纲。
加权求和必须先做 min-max 归一化，而归一化**依赖候选集内的极值**——
同一份文档，换一批候选，归一化后的相对高低就可能反转，检索结果因此不稳定。

RRF 只用**名次**：`score = Σ w_r / (k + rank_r)`。它天然免疫量纲不一致，
在不同查询之间也更稳，这就是它在混合检索里几乎成为默认做法的原因。
    注：实际实现为 `rrf_score += share * w_r / (rrf_k + rank + 1)`，
    其中 `rank` 从 0 开始计数；这样 `rrf_k + rank + 1` 等价于「k + 1 基名次」。
    `share = 1 / len(queries)` 是查询变体数的摊薄系数，保证改写多出几个变体
    不会让总分整体被抬高、破坏与其它检索结果的可比性。

RRF 之上再做两件事：
    * 三路权重按**问题类型**配置（条款类偏 BM25、案例类偏稠密）；
    * 召回前用元数据过滤表达式把候选集收窄。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..chunking.metadata import MetadataFilter
from ..chunking.parent_child import ChildChunk, ParentChunk
from ..config import FINAL_TOP_K, RECALL_TOP_K, RRF_K
from ..index.bm25 import BM25Index
from ..index.embedding import EmbeddingBackend, get_backend, sparse_from_text
from ..index.vector_store import MilvusLiteClient
from ..utils.jsonable import to_plain
from ..utils.text import jaccard, shingles, tokenize
from .router import QueryPlan, build_query_plan

__all__ = ["Candidate", "HybridRetriever", "COLLECTION_NAME"]

# 向量集合名：稠密与稀疏两路共用同一个集合，靠 record_id（= child_id）对齐
COLLECTION_NAME = "finrag_chunks"


@dataclass
class Candidate:
    """一条召回候选。保留三路各自的分数与名次，因此「为什么它被召回」是可回答的。

    关键属性：
        child_id / parent_id / source_id / doc_id   子块与父块定位信息（引用可追溯）
        text / context                              子块原文（精确命中）+ 父块上下文（父块回溯后填入）
        meta / kind / quality_score                 元数据、块类型、来源质量（OCR 脏文档会降权）
        bm25_score / dense_score / sparse_score     三路各自的原始分数（量纲不同，仅供展示与调试）
        bm25_rank / dense_rank / sparse_rank        三路各自的名次；None 表示未命中该路
        rrf_score                                   RRF 融合分；多路命中时还乘过共识加成
        matched_terms / queries / route             BM25 命中的词、召回它的查询变体、问题类型路由
        metadata_score                             软元数据契合度（推断条件只影响它，不剔除候选）
    """

    child_id: str
    parent_id: str
    source_id: str
    doc_id: str
    section_title: str
    text: str
    kind: str
    meta: Dict[str, object]
    quality_score: float = 1.0
    # 三路信号
    bm25_score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    # 融合与解释
    rrf_score: float = 0.0
    matched_terms: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    route: str = "general"
    # 父块回填
    context: str = ""
    metadata_score: float = 1.0

    @property
    def citation_label(self) -> str:
        """人可读的出处标签「机构 · 标题 · 章节」；缺项自动跳过（供答案里做引用脚注）。"""
        institution = str(self.meta.get("institution", ""))
        title = str(self.meta.get("title", ""))
        head = " · ".join(p for p in (institution, title) if p)
        return f"{head} · {self.section_title}".strip(" ·")

    @property
    def routes_hit(self) -> List[str]:
        """命中了哪几路（用于展示"三路共识"这一信号）。"""
        hit: List[str] = []
        if self.bm25_rank is not None:
            hit.append("bm25")
        if self.dense_rank is not None:
            hit.append("dense")
        if self.sparse_rank is not None:
            hit.append("sparse")
        return hit

    def to_dict(self) -> Dict[str, object]:
        """序列化成可 JSON 化的 dict：分数保留 6 位小数、matched_terms 截断到 12 条。

        只导出展示与解释需要的字段，不含 context（父块上下文体积大，由 Evidence 决定是否带上）。
        """
        return to_plain(
            {
                "child_id": self.child_id,
                "parent_id": self.parent_id,
                "source_id": self.source_id,
                "section_title": self.section_title,
                "kind": self.kind,
                "score": round(float(self.rrf_score), 6),
                "components": {
                    "bm25": round(float(self.bm25_score), 6),
                    "dense": round(float(self.dense_score), 6),
                    "sparse": round(float(self.sparse_score), 6),
                    "rrf": round(float(self.rrf_score), 6),
                },
                "ranks": {"bm25": self.bm25_rank, "dense": self.dense_rank, "sparse": self.sparse_rank},
                "routes_hit": self.routes_hit,
                "matched_terms": self.matched_terms[:12],
                "route": self.route,
                "text": self.text,
            }
        )


class HybridRetriever:
    """BM25 + 稠密向量 + 稀疏权重的三路召回器，带 RRF 融合与元数据过滤。

    关键属性（构造时一次性建索引，之后只读）：
        parents / children   父块与子块全集（子块是检索单元，父块是回溯出的上下文单元）
        backend              嵌入后端，同时提供稠密 `encode` 与稀疏 `encode_sparse`
        rrf_k                RRF 平滑常数 k（默认 config.RRF_K = 60；越大越弱化名次差异）
        recall_top_k         每一路各自取多少条候选（默认 config.RECALL_TOP_K = 20，"召回看漏不漏"）
        _child_tokens        子块分词结果（与 children 同序，供 BM25 复用）
        _bm25                BM25 倒排索引
        _collection          向量集合，稠密与稀疏**同库**存放（保证同一块的两路表示不会对不齐）
    """

    def __init__(
        self,
        parents: Sequence[ParentChunk],
        children: Sequence[ChildChunk],
        backend: Optional[EmbeddingBackend] = None,
        rrf_k: int = RRF_K,
        recall_top_k: int = RECALL_TOP_K,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        """构造索引：分词建 BM25，编码建稠密 + 稀疏向量库。

        参数：
            parents          父块序列（只用于回溯上下文，不进索引）
            children         子块序列（检索的最小单元，全部进 BM25 与向量库）
            backend          嵌入后端；缺省用 get_backend() 按配置选择
            rrf_k            RRF 平滑常数 k
            recall_top_k     单路召回条数上限
            collection_name  向量集合名（默认 "finrag_chunks"）
        返回：无。
        副作用：在本进程内创建集合、写入全部子块向量并 flush（真后端换 BGE-M3 时维度需对齐）。
        异常：底层向量库 / 嵌入后端初始化或写入失败时向上抛出。
        """
        self.parents: List[ParentChunk] = list(parents)
        self.children: List[ChildChunk] = list(children)
        self.backend = backend or get_backend()
        self.rrf_k = int(rrf_k)
        self.recall_top_k = int(recall_top_k)

        self._parent_by_id: Dict[str, ParentChunk] = {p.parent_id: p for p in self.parents}
        self._child_by_id: Dict[str, ChildChunk] = {c.child_id: c for c in self.children}

        # ---- 一路：BM25 倒排 ----
        self._child_tokens: List[List[str]] = [tokenize(c.text) for c in self.children]
        self._bm25 = BM25Index(self._child_tokens)

        # ---- 二路 / 三路：向量库（稠密 + 稀疏同库，保证同一块的两路表示不会对不齐）----
        self._client = MilvusLiteClient()
        self._collection = self._client.create_collection(collection_name, dim=self.backend.dim, metric="COSINE")
        self._build_vector_index()

    # ------------------------------------------------------------------
    # 建索引
    # ------------------------------------------------------------------
    def _build_vector_index(self) -> None:
        """把全部子块编码成稠密 + 稀疏向量并写入集合（一次性建库）。

        参数：无（数据来自 self.children 与 self.backend）。
        返回：无。
        副作用：调用 collection.insert / flush 写向量库；空语料时写入 0 条记录而不报错。
        """
        # 局部导入：只有真正建索引时才需要 VectorRecord，
        # 放在函数内可避免模块级导入把向量库依赖强加给所有使用方
        from ..index.vector_store import VectorRecord

        texts = [c.text for c in self.children]
        dense = self.backend.encode(texts) if texts else np.zeros((0, self.backend.dim))
        sparse = self.backend.encode_sparse(texts) if texts else []
        records = [
            VectorRecord(
                record_id=child.child_id,
                dense=dense[i] if i < dense.shape[0] else None,
                sparse=sparse[i] if i < len(sparse) else {},
                # 元数据随向量一起存：硬过滤（expr）由向量库侧执行，不需要回表
                meta=dict(child.meta, quality_score=child.quality_score, kind=child.kind),
            )
            for i, child in enumerate(self.children)
        ]
        self._collection.insert(records)
        self._collection.flush()

    # ------------------------------------------------------------------
    # 基础属性
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """子块总数，也就是可被检索的最小单元数。"""
        return len(self.children)

    @property
    def vocabulary_size(self) -> int:
        """BM25 词表大小（子块分词后的去重词数），用于自检分词是否正常。"""
        return self._bm25.vocabulary_size

    @property
    def known_years(self) -> List[int]:
        """资料库里实际出现过的年份集合。

        用途：用户明确问「2023 年的标准」时，只有当资料库确实存在 2023 年的版本，
        才敢把这个年份提升为**硬过滤**（召回前剔除）。否则一旦元数据缺失，
        硬过滤会把唯一正确的证据也筛掉——宁可只做软加权。
        """
        years = {c.meta.get("year") for c in self.children if isinstance(c.meta.get("year"), int)}
        return sorted(int(y) for y in years)

    def parent_of(self, child_id: str) -> Optional[ParentChunk]:
        """按子块 id 反查父块（父块回溯的入口）；子块不存在或父块缺失时返回 None。"""
        child = self._child_by_id.get(child_id)
        return self._parent_by_id.get(child.parent_id) if child else None

    def child(self, child_id: str) -> Optional[ChildChunk]:
        """按 id 取子块对象；不存在返回 None。"""
        return self._child_by_id.get(child_id)

    def describe(self) -> Dict[str, object]:
        """汇总索引自检信息（父/子块数、词表大小、RRF 参数、嵌入后端与向量库描述）。

        返回：可 JSON 化的 dict；写入运行轨迹，便于复现一次检索当时用的是哪套索引。
        """
        return {
            "children": len(self.children),
            "parents": len(self.parents),
            "vocabulary": self._bm25.vocabulary_size,
            "rrf_k": self.rrf_k,
            "recall_top_k": self.recall_top_k,
            "embedding": self.backend.describe(),
            "vector_store": self._client.describe(),
        }

    # ------------------------------------------------------------------
    # 单路检索（供 A/B 对比与演示）
    # ------------------------------------------------------------------
    def search_bm25_only(self, query: str, top_k: int = 10, expr: Optional[str] = None) -> List[Candidate]:
        """基线一：只用 BM25（关键词检索）。

        参数：query 查询串；top_k 取前多少条（默认 10）；expr 可选元数据过滤表达式（硬过滤）。
        返回：Candidate 列表；只填 bm25_score / bm25_rank。
        """
        return self._search_single_route(query, "bm25", top_k, expr)

    def search_dense_only(self, query: str, top_k: int = 10, expr: Optional[str] = None) -> List[Candidate]:
        """基线二：只用稠密向量（单一向量检索）。这就是本项目的对照组。

        参数：query 查询串；top_k 取前多少条（默认 10）；expr 可选元数据过滤表达式（硬过滤）。
        返回：Candidate 列表；只填 dense_score / dense_rank。
        """
        return self._search_single_route(query, "dense", top_k, expr)

    def _search_single_route(self, query: str, route: str, top_k: int, expr: Optional[str]) -> List[Candidate]:
        """单路检索的公共实现：按 route 分派到对应排名函数，再把结果包成 Candidate。

        参数：query 查询串；route 取 "bm25" / "dense" / "sparse"；
              top_k 截断条数；expr 元数据过滤表达式（None / 空串表示不过滤）。
        返回：已回填 `{route}_score` 与 `{route}_rank` 的 Candidate 列表，
              此时 rrf_score 只是「单路名次分」1/(rrf_k + rank + 1)，不是三路融合分。
        异常：route 不在三路之内时抛 ValueError；空查询或空索引直接返回 []。
        """
        if not query.strip() or not self.children:
            return []
        filter_expr = MetadataFilter.parse(expr)
        if route == "bm25":
            ranked = self._bm25_ranked(query, top_k, filter_expr)
        elif route == "dense":
            ranked = self._dense_ranked(query, top_k, filter_expr)
        elif route == "sparse":
            ranked = self._sparse_ranked(query, top_k, filter_expr)
        else:
            raise ValueError(f"未知通路：{route}")

        out: List[Candidate] = []
        for rank, (idx, score) in enumerate(ranked):
            cand = self._make_candidate(idx, query)
            # 用 setattr 按通路名回填，避免为三路各写一份几乎相同的赋值代码
            setattr(cand, f"{route}_score", float(score))
            setattr(cand, f"{route}_rank", rank)
            cand.rrf_score = 1.0 / (self.rrf_k + rank + 1)
            out.append(cand)
        return out[: max(1, top_k)]

    # ------------------------------------------------------------------
    # 三路排名
    # ------------------------------------------------------------------
    def _mask(self, filter_expr: MetadataFilter) -> Optional[np.ndarray]:
        """硬过滤掩码；空表达式返回 None（表示不过滤）。

        参数：filter_expr 元数据过滤条件。
        返回：长度等于子块数的 bool 数组（True = 通过条件、保留），或 None（无过滤）。
        说明：只有用户**显式**给出的条件才会走到这里；从问题里推断出的条件不做硬过滤，
              因为推断错了会静默丢证据（见 retrieve 里的软加权分支）。
        """
        if filter_expr.is_empty:
            return None
        mask = np.array([filter_expr.matches(c.meta) for c in self.children], dtype=bool)
        return mask

    def _bm25_ranked(self, query: str, top_k: int, filter_expr: MetadataFilter) -> List[Tuple[int, float]]:
        """BM25 排名：按分数降序，只保留正分。

        参数：query 查询串；top_k 取前多少条；filter_expr 硬过滤条件。
        返回：[(子块下标, BM25 分数)]，长度不超过 top_k。
        说明：分词后一个词都没有时直接返回 []（无从打分，返回全库等于噪声）。
        """
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.score_array(tokens)
        mask = self._mask(filter_expr)
        if mask is not None:
            # 不满足硬过滤的块压成负分，靠下面 `<= 0.0` 的截断线自然排除，无需额外分支
            scores = np.where(mask, scores, -1.0)
        order = np.argsort(-scores)
        out: List[Tuple[int, float]] = []
        for idx in order:
            # BM25 分数为 0 说明查询词一个都没命中，后面的只会更低（已降序），直接收尾
            if scores[idx] <= 0.0:
                break
            out.append((int(idx), float(scores[idx])))
            if len(out) >= max(1, top_k):
                break
        return out

    def _dense_ranked(self, query: str, top_k: int, filter_expr: MetadataFilter) -> List[Tuple[int, float]]:
        """稠密向量排名：余弦相似度降序（走向量库做 ANN 检索）。

        参数：query 查询串；top_k 取前多少条；filter_expr 硬过滤条件。
        返回：[(子块下标, 余弦分)]；查询为空或编码结果为零向量时返回 []。
        异常：向量库检索失败时由底层抛出；仅保留能在 children 里定位到的命中。
        """
        if not self.children or not query.strip():
            return []
        vec = self.backend.encode_one(query)
        # 空查询 / 全是停用词的查询会得到零向量，此时余弦全为 0，
        # 直接返回空而不是把"全库零分"当成有效召回（否则空问题也会给出 5 条证据）
        if not np.any(vec):
            return []
        hits = self._collection.search_dense(vec, top_k=top_k, expr=filter_expr.expr or None)
        # record_id 是 child_id，这里换回 children 的下标，统一三路的返回格式
        index = {c.child_id: i for i, c in enumerate(self.children)}
        return [(index[h.record_id], h.dense_score) for h in hits if h.record_id in index]

    def _sparse_ranked(self, query: str, top_k: int, filter_expr: MetadataFilter) -> List[Tuple[int, float]]:
        """稀疏词权重排名：稀疏点积降序（与稠密同一个库、同一份过滤条件）。

        参数：query 查询串；top_k 取前多少条；filter_expr 硬过滤条件。
        返回：[(子块下标, 稀疏分)]；查询为空时返回 []。
        """
        if not self.children or not query.strip():
            return []
        hits = self._collection.search_sparse(sparse_from_text(query), top_k=top_k, expr=filter_expr.expr or None)
        index = {c.child_id: i for i, c in enumerate(self.children)}
        return [(index[h.record_id], h.sparse_score) for h in hits if h.record_id in index]

    # ------------------------------------------------------------------
    # 三路召回 + RRF 融合
    # ------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        plan: Optional[QueryPlan] = None,
        top_k: Optional[int] = None,
        expr: Optional[str] = None,
        route: Optional[str] = None,
        corpus=None,
    ) -> List[Candidate]:
        """完整三路召回，返回按 RRF 融合分排序的候选列表。

        参数：
            question  用户原始问题（空串直接返回 []）
            plan      预构造的 QueryPlan；缺省时用 build_query_plan 现算路由 / 权重 / 过滤 / 变体
            top_k     返回条数上限；缺省依次回落 plan.top_k、self.recall_top_k
            expr      显式元数据过滤表达式（属硬过滤，透传给 build_query_plan）
            route     强制指定问题类型路由（如 "clause" / "case" / "metric" / "general"）
            corpus    语料对象，用于从问题里识别机构名与资料类型；可为 None
        返回：`List[Candidate]`，按 rrf_score 降序截断到 limit 条；无候选时返回 []。
        副作用：只读索引，不修改传入的 plan。
        """
        if not self.children or not (question or "").strip():
            return []
        # 计划里已经定好路由、三路权重、查询变体与过滤表达式，召回阶段只照着执行
        resolved = plan or build_query_plan(question, corpus=corpus, top_k=top_k or self.recall_top_k, route=route, filter_expr=expr)
        limit = max(1, top_k or resolved.top_k or self.recall_top_k)
        # 硬过滤只用显式条件（resolved.filter_expr）；resolved.filters 里的推断条件在末尾做软加权
        hard = MetadataFilter.parse(resolved.filter_expr or expr)
        # 兜底：plan 没给权重时三路等分（注：config 里 general 的实际值是 0.34/0.33/0.33，并非严格等分）
        weights = resolved.weights or {"bm25": 1 / 3, "dense": 1 / 3, "sparse": 1 / 3}
        queries = resolved.queries or [question]
        # 查询变体数摊薄：变体多了命中机会自然变多，不摊薄就会让"改写多"的查询凭空占优
        share = 1.0 / max(1, len(queries))

        acc: Dict[int, Candidate] = {}

        for query in queries:
            for route_name in ("bm25", "dense", "sparse"):
                if route_name == "bm25":
                    ranked = self._bm25_ranked(query, self.recall_top_k, hard)
                elif route_name == "dense":
                    ranked = self._dense_ranked(query, self.recall_top_k, hard)
                else:
                    ranked = self._sparse_ranked(query, self.recall_top_k, hard)

                weight = float(weights.get(route_name, 0.0))
                for rank, (idx, score) in enumerate(ranked):
                    child = self.children[idx]
                    cand = acc.get(idx)
                    if cand is None:
                        cand = self._make_candidate(idx, query)
                        cand.route = resolved.route
                        acc[idx] = cand
                    # RRF：只看名次（rank），完全不用三路各自的分数，
                    # 因此 BM25 的无界分、余弦的 [-1,1]、稀疏点积不需要任何归一化就能放一起
                    cand.rrf_score += share * weight * (1.0 / (self.rrf_k + rank + 1))
                    current = getattr(cand, f"{route_name}_score")
                    # 同一块被多个查询变体命中时，分数与名次都取最好的一次（对候选更有利）
                    if score > current:
                        setattr(cand, f"{route_name}_score", float(score))
                    current_rank = getattr(cand, f"{route_name}_rank")
                    if current_rank is None or rank < current_rank:
                        setattr(cand, f"{route_name}_rank", rank)
                    if query not in cand.queries:
                        cand.queries.append(query)
                    for term in self._bm25.matched_terms(tokenize(query), idx):
                        if term not in cand.matched_terms:
                            cand.matched_terms.append(term)

        if not acc:
            return []

        soft_filters = resolved.filters or {}
        results: List[Candidate] = []
        for idx, cand in acc.items():
            # 软加权：推断出来的条件只折算成 metadata_score，交给重排阶段参与加权，
            # 绝不在这里剔除候选——推断错了顶多排序差一点，不会把正确答案整批筛掉
            cand.metadata_score = self.metadata_score(cand.meta, soft_filters)
            # 多路共识的轻微加成：三路都命中说明这条大概率是真的相关
            # 注：实际实现为 1.0 + 0.05 × (命中路数 - 1)，即每多一路 +5%，最多 +10%，
            # 只做微调，不足以盖过重排阶段的判断
            consensus = 1.0 + 0.05 * (len(cand.routes_hit) - 1)
            cand.rrf_score *= consensus
            results.append(cand)

        results.sort(key=lambda c: -c.rrf_score)
        return results[:limit]

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _make_candidate(self, idx: int, query: str) -> Candidate:
        """按下标构造 Candidate（含父块上下文与 BM25 命中词）。

        参数：idx 子块在 self.children 里的下标；query 用于抽取 matched_terms 的查询串。
        返回：新建的 Candidate；父块缺失时 context 退化为子块自身文本（保证字段不为空）。
        """
        child = self.children[idx]
        parent = self._parent_by_id.get(child.parent_id)
        return Candidate(
            child_id=child.child_id,
            parent_id=child.parent_id,
            source_id=child.source_id,
            doc_id=child.doc_id,
            section_title=child.section_title,
            text=child.text,
            kind=child.kind,
            meta=dict(child.meta),
            quality_score=float(child.quality_score),
            matched_terms=self._bm25.matched_terms(tokenize(query), idx),
            queries=[query],
            context=parent.text if parent else child.text,
        )

    @staticmethod
    def metadata_score(meta: Dict[str, object], filters: Dict[str, object]) -> float:
        """软元数据契合度：请求了几个条件，就按满足比例给分；无条件时给 1.0。

        参数：meta 候选的元数据；filters 推断出的条件，支持 year_gte / year_lte 与其它等值条件。
        返回：[0, 1] 的命中比例（hits / 有效条件数）。
        说明：字符串按互相包含判定（宽松），避免「示例银行」与「示例银行股份有限公司」
              这种措辞差异被误判成不匹配；元数据缺该项时该条件不计入命中（但计入分母）。
        """
        active = {k: v for k, v in (filters or {}).items() if v not in (None, "", [])}
        if not active:
            return 1.0
        hits = 0
        for key, want in active.items():
            if key == "year_gte":
                got = meta.get("year")
                hits += 1 if isinstance(got, int) and got >= int(want) else 0
            elif key == "year_lte":
                got = meta.get("year")
                hits += 1 if isinstance(got, int) and got <= int(want) else 0
            else:
                got = meta.get(key)
                if got is None:
                    continue
                if isinstance(want, str) and isinstance(got, str):
                    hits += 1 if (want in got or got in want) else 0
                else:
                    hits += 1 if got == want else 0
        return hits / len(active)


def deduplicate(
    candidates: Sequence[Candidate],
    threshold: float = 0.82,
    keep: Optional[Sequence[str]] = None,
    merge_log: Optional[List[Dict[str, str]]] = None,
) -> List[Candidate]:
    """近似去重：同一段文字常被多路（或多查询变体）重复召回。

    用 token 二元组 Jaccard 判定重复，**保留分数更高的一条**，并把被丢弃者的
    命中通路合并过去（这样"三路共识"的统计不会因为去重而失真）。
    不去重最直接的后果是 5 个上下文位被同一段话占掉 3 个，
    LLM 看到的"证据"其实是同一份，答案覆盖度自然上不去。

    参数：
        candidates  召回候选；内部会先按 rrf_score 降序，保证被保留的永远是更高分那条
        threshold   判重阈值（默认 0.82，与 config.DEDUP_JACCARD 一致）；越大越宽松（越少判重）
        keep        受保护的 child_id 集合，即使被判重也强制保留（如人工指定的关键块）
        merge_log   可选输出列表，追加 {"child_id", "merged_into"} 记录，供 UI 展示与调试
    返回：去重后的 `List[Candidate]`（保持分数降序）。
    副作用：就地修改被保留的 Candidate（补 rank / score / matched_terms / queries），
            并在 merge_log 非 None 时往里追加记录。
    """
    if not candidates:
        return []
    protected = set(keep or ())
    # 先排序：这样"重复组里留下谁"由分数决定，而不是由召回顺序决定，结果才可复现
    ordered = sorted(candidates, key=lambda c: -c.rrf_score)
    kept: List[Candidate] = []
    signatures: List[set] = []

    for cand in ordered:
        # 二元组（bigram）而不是单词集合：单字重叠太多，会误判"客户风险 / 风险客户"这类反序短语
        sig = shingles(tokenize(cand.text), 2)
        duplicate_of: Optional[int] = None
        for i, existing in enumerate(signatures):
            if jaccard(sig, existing) >= threshold:
                duplicate_of = i
                break
        # 受保护项即使判重也不丢：调用方显式要求保留的块优先级高于去重策略
        if duplicate_of is None or cand.child_id in protected:
            kept.append(cand)
            signatures.append(sig)
            continue

        target = kept[duplicate_of]
        # 合并证据：让"被去重掉的那条命中了哪几路"仍然体现在保留项上
        for route in cand.routes_hit:
            if getattr(target, f"{route}_rank") is None:
                setattr(target, f"{route}_rank", getattr(cand, f"{route}_rank"))
                setattr(target, f"{route}_score", getattr(cand, f"{route}_score"))
        for term in cand.matched_terms:
            if term not in target.matched_terms:
                target.matched_terms.append(term)
        for q in cand.queries:
            if q not in target.queries:
                target.queries.append(q)
        if merge_log is not None:
            merge_log.append({"child_id": cand.child_id, "merged_into": target.child_id})
    return kept
