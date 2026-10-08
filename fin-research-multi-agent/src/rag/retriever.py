"""混合检索 + 加权重排（RAG 证据层的核心出口）。

层次与职责
----------
把「词面（BM25）+ 语义（哈希向量）+ 元数据（公司 / 年份 / 文档类型）」三路信号放在
**同一个口径**下融合排序，命中子块后回填父块上下文，产出可引用的 RetrievedSnippet。
它是 rag 子包对外唯一的检索入口，自身不调用 LLM、不下结论。

关键类：
    HybridRetriever  —— 建索引 + 多路检索 + 融合排序 + 父块回填
    RetrievedSnippet —— 一条带出处的检索结果（子块 text + 父块 context）

主要输入：chunking 产出的 parents / children，查询串（可多路）与元数据约束。
主要输出：List[RetrievedSnippet]，已按融合分降序并截断到 top_k。
被谁调用：src/orchestrator.py 构建实例（ResearchPipeline.__init__）；
          src/tools/builtin.py 的 search_filings 工具，即 RetrieverAgent 的取数口。

检索管线：
    查询 -> ┌ BM25（稀疏，词面命中）      ┐
           │ 哈希向量（稠密，语义近似）   ├-> min-max 归一化 -> 加权融合 -> Top-K -> 父块回填
           └ 元数据匹配（公司/年份/文档类型）┘

融合公式（权重来自 config.RERANK_WEIGHTS，可配置）：
    final = w_kw * keyword + w_vec * vector + w_meta * metadata
    keyword = 0.7 * BM25_norm + 0.3 * 命中词占比

命中词占比（hit ratio）是刻意加进来的一路信号：BM25 对长文档有天然偏置，
而「查询里的关键财务词有多少真的出现在这块里」对金融问答是强信号。

召回后做父块回填：返回给 Agent 的是「子块文本（精确命中）+ 父块上下文（完整章节）」，
既准又全。每条结果都带 source_id / 章节标题，供引用可追溯。

融合口径（与代码逐条对应）
--------------------------
    * 三路原始分先各自 min-max 归一化到 [0,1] 再加权，权重因此可以横向比较；
      分数全相等时 _minmax 返回全 0，不会凭空造出高分。
    * keyword = 0.7 * BM25_norm + 0.3 * 命中词占比，其中 0.7 / 0.3 是写死的内层配比；
      外层再乘 w_kw（默认 0.45）。vector、metadata 各乘自己的权重（默认 0.40 / 0.15），
      三者求和即 final。权重来自 config.RERANK_WEIGHTS，可用 RERANK_W_KEYWORD /
      RERANK_W_VECTOR / RERANK_W_METADATA 环境变量覆盖。
    * 元数据分 = 满足的约束数 / 给出的约束数；一个约束都没给时记 1.0（不奖不罚）。
    * 多路查询取「各路最高分」而非求和：同一子块被多路命中不会重复加分，只有分数
      并列时才合并 matched_terms 与 queries。
    * 硬过滤（strict=True 或 exclude_children）在打分之前就把 mask 置 False，
      被排除的子块全程不参与计算。
    * 完全无命中的块（BM25 与余弦都 <= 0）直接丢弃，不让噪声块靠元数据分挤进结果。
    * 确定性：同一语料 + 同一查询必得同一排序（BM25 与哈希向量都是确定性的），
      所以检索结果可写进 trace 复现、可在 eval 里回归对比。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import EMBED_DIM, RERANK_WEIGHTS
from ..utils.jsonable import to_plain
from ..utils.text import tokenize
from .bm25 import BM25Index
from .chunking import ChildChunk, ParentChunk
from .embedding import cosine_scores, embed, embed_matrix

__all__ = ["RetrievedSnippet", "HybridRetriever"]


@dataclass
class RetrievedSnippet:
    """一条检索结果（子块精确命中 + 父块上下文回填）。

    关键属性：
        child_id / parent_id —— 命中子块与回填父块的 ID（引用与去重都靠前者）
        source_id / doc_id   —— 出处资料编号与文档编号
        company / period / year / section_title —— 展示与元数据过滤用的扁平字段
        text     —— 子块原文（精确命中片段，喂给模型的"证据原文"）
        context  —— 父块原文（完整章节上下文；回填失败时降级为子块原文）
        score    —— 融合总分（0~1 量级，权重和为 1 时上界为 1）
        components —— 分项明细 keyword / vector / metadata / bm25_raw / cosine_raw，
                      用于 trace 解释"为什么排在这"，不参与排序
        matched_terms —— 命中的查询词（最多 12 个，用于可解释性展示）
        queries  —— 命中的查询路（只有并列最高分时才可能多于一条）
    状态流转：由 HybridRetriever._make_snippet 构造，构造后只读。
    """

    child_id: str
    parent_id: str
    source_id: str
    doc_id: str
    company: str
    period: str
    year: Optional[int]
    section_title: str
    text: str            # 子块原文（精确命中片段）
    context: str         # 父块原文（回填的上下文）
    score: float
    components: Dict[str, float] = field(default_factory=dict)
    matched_terms: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)

    @property
    def citation_label(self) -> str:
        """人可读的出处标签（公司 + 期间 + 章节标题）。

        参数：无（属性访问）。返回：str。副作用：无。
        """
        return f"{self.company} {self.period} · {self.section_title}".strip(" ·")

    def to_dict(self) -> Dict[str, object]:
        """转成可 JSON 序列化的扁平字典（search_filings 工具的返回体）。

        参数：无。
        返回：dict —— 含 child_id / parent_id / source_id / company / period / year /
            section_title / text / score / components / matched_terms。
            注：实际实现里 doc_id / context / queries 都**不在**返回体里——工具层只回传
            子块 text；RetrievedSnippet 上的父块 context 不会随工具结果进入 Agent 上下文。
        score 与 components 保留 4 位小数，纯为 trace 可读性（不影响排序）。
        副作用：无（经 utils.jsonable.to_plain 转换，numpy 标量会转成原生类型）。
        """
        return to_plain(
            {
                "child_id": self.child_id,
                "parent_id": self.parent_id,
                "source_id": self.source_id,
                "company": self.company,
                "period": self.period,
                "year": self.year,
                "section_title": self.section_title,
                "text": self.text,
                "score": round(self.score, 4),
                "components": {k: round(float(v), 4) for k, v in self.components.items()},
                "matched_terms": self.matched_terms,
            }
        )


def _minmax(scores: np.ndarray) -> np.ndarray:
    """min-max 归一化到 [0,1]；全相等时返回全 0，避免虚假高分。

    参数：scores —— 一路原始分数组。
    返回：同形状的 float64 数组；空数组原样返回。
    副作用：无（返回新数组，不改入参）。
    注：极差小于 1e-12（含全相等、只有一个候选）时返回全 0——这是刻意的：此时这一路
        无法区分任何候选，给满分会让它凭空主导融合结果。
    """
    if scores.size == 0:
        return scores
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class HybridRetriever:
    """BM25 + 向量 + 元数据的混合检索器（本项目唯一的检索入口）。

    关键属性：
        parents / children —— 语料切块结果（父块用于回填，子块进索引）
        weights   —— 三路融合权重 {"keyword", "vector", "metadata"}，构造时拷贝自
                     config.RERANK_WEIGHTS，之后不再变化
        embed_dim —— 哈希向量维度
        _parent_by_id —— parent_id -> ParentChunk，父块回填用
        _child_tokens —— 子块分词结果（只算一次，BM25 与命中率特征共用）
        _bm25         —— BM25Index 实例
        _matrix       —— (子块数, embed_dim) 的子块向量矩阵

    状态流转：__init__ 一次性完成「分词 -> 建倒排 -> 批量编码」，此后只读；
    查询期不产生持久状态，search / search_multi 可被反复调用且结果一致。
    """

    def __init__(
        self,
        parents: Sequence[ParentChunk],
        children: Sequence[ChildChunk],
        weights: Optional[Dict[str, float]] = None,
        embed_dim: int = EMBED_DIM,
    ) -> None:
        """构建索引。

        参数：
            parents / children: chunking 的输出；子块通过 parent_id 与父块对应。
            weights: 覆盖默认融合权重，None 时用 config.RERANK_WEIGHTS。
            embed_dim: 哈希向量维度，默认 config.EMBED_DIM。
        返回：None。
        副作用：无外部副作用；只做内存索引构建，开销（对全部子块分词 + 编码）只在
            启动时付一次。
        """
        self.parents: List[ParentChunk] = list(parents)
        self.children: List[ChildChunk] = list(children)
        # 为什么拷贝一份：防止调用方之后修改自己那份 dict 时，悄悄改变已建好检索器的口径。
        self.weights: Dict[str, float] = dict(weights or RERANK_WEIGHTS)
        self.embed_dim = embed_dim

        self._parent_by_id: Dict[str, ParentChunk] = {p.parent_id: p for p in self.parents}

        # 为什么在构造期就全部算完：检索是每次提问都要走的热路径，分词、倒排、向量矩阵
        # 都只与语料有关，提前固化后多路查询可以反复复用。
        self._child_tokens: List[List[str]] = [tokenize(c.text) for c in self.children]
        self._bm25 = BM25Index(self._child_tokens)
        self._matrix: np.ndarray = embed_matrix([c.text for c in self.children], dim=embed_dim)

    # ------------------------------------------------------------------
    # 基础属性
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """索引里的子块数量（不是父块数）。

        参数：无。返回：int。副作用：无。
        """
        return len(self.children)

    @property
    def companies(self) -> List[str]:
        """语料覆盖的公司名，去重后排序（从子块元数据取）。

        参数：无（属性访问）。返回：List[str]。副作用：无。
        """
        return sorted({c.meta.get("company", "") for c in self.children if c.meta.get("company")})

    def parent_of(self, child: ChildChunk) -> Optional[ParentChunk]:
        """按 parent_id 取子块所属的父块。

        参数：child 子块。
        返回：ParentChunk；parent_id 对不上时返回 None（调用方需兜底）。
        副作用：无。
        """
        return self._parent_by_id.get(child.parent_id)

    # ------------------------------------------------------------------
    # 元数据打分
    # ------------------------------------------------------------------
    @staticmethod
    def _metadata_score(meta: Dict[str, object], filters: Dict[str, object]) -> float:
        """请求了多少个元数据约束，就按满足比例给分；没给约束时给 1.0。

        参数：meta 子块的扁平元数据；filters 约束字典（company / year / doc_type）。
        返回：float，落在 [0,1]。
        副作用：无。
        注：值为 None / "" / [] 的约束视为"没提要求"，直接忽略；字符串用双向子串匹配
            （所以简称「示例科技」能匹配全称），其余类型要求严格相等（year 因此能区分
            2024 与 2023，而不会把 int 当 str 做子串比较）。
        """
        active = {k: v for k, v in filters.items() if v not in (None, "", [])}
        if not active:
            return 1.0
        hits = 0
        for key, want in active.items():
            got = meta.get(key)
            if isinstance(want, str) and isinstance(got, str):
                if want in got or got in want:
                    hits += 1
            elif got == want:
                hits += 1
        return hits / len(active)

    # ------------------------------------------------------------------
    # 检索主入口
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        company: Optional[str] = None,
        year: Optional[int] = None,
        doc_type: Optional[str] = None,
        strict: bool = False,
        exclude_children: Optional[Sequence[str]] = None,
    ) -> List[RetrievedSnippet]:
        """单查询混合检索（search_multi 的语法糖）。

        参数：query 查询串；top_k 返回条数；company / year / doc_type 元数据约束；
            strict=True 时把不满足全部约束的子块硬过滤掉；
            exclude_children 需要排除的 child_id（例如换一批证据时排除已用过的）。
        返回：List[RetrievedSnippet]，按融合分降序。
        副作用：无（直接委托 search_multi，语义完全一致）。
        """
        return self.search_multi(
            [query],
            top_k=top_k,
            company=company,
            year=year,
            doc_type=doc_type,
            strict=strict,
            exclude_children=exclude_children,
        )

    def search_multi(
        self,
        queries: Sequence[str],
        top_k: int = 5,
        company: Optional[str] = None,
        year: Optional[int] = None,
        doc_type: Optional[str] = None,
        strict: bool = False,
        exclude_children: Optional[Sequence[str]] = None,
    ) -> List[RetrievedSnippet]:
        """多查询混合检索：每路查询独立打分，同一子块取各路最高分。

        参数：
            queries: 多路查询串（空串，或只有标点导致分不出 token 的，会被跳过）；
            top_k: 返回条数，实现里用 max(1, top_k) 兜底，至少返回一条；
            company / year / doc_type: 元数据约束；
            strict: True 时用 _metadata_score 硬过滤，不满足全部约束的子块直接出局；
            exclude_children: child_id 黑名单，用于"换一批证据"。
        返回：List[RetrievedSnippet]，按融合分降序；无子块、无有效查询或 mask 全空时返回 []。
        副作用：无（只读索引；中间量都是局部变量）。注意所有分数在封装前都显式转成
            原生 float——numpy 标量会破坏 LangGraph checkpointer 的序列化。
        """
        if not self.children or not queries:
            return []

        filters: Dict[str, object] = {"company": company, "year": year, "doc_type": doc_type}
        excluded = set(exclude_children or ())
        n = len(self.children)

        # 元数据硬过滤掩码
        # 为什么 strict 用硬过滤而不是降权：调用方明确要"只看这家公司 / 这一年"时，
        # 混进一条别家的证据比什么都不返回更糟——降权挡不住它排进 top_k。
        mask = np.ones(n, dtype=bool)
        if strict:
            for idx, child in enumerate(self.children):
                if self._metadata_score(child.meta, filters) < 1.0:
                    mask[idx] = False
        if excluded:
            for idx, child in enumerate(self.children):
                if child.child_id in excluded:
                    mask[idx] = False
        if not mask.any():
            return []

        # 累加各路查询的最佳融合分
        # 为什么初值取 -1.0：融合分非负，用 -1 当"该块从未被任何一路查询评估过"的哨兵，
        # 排序时才能把"没被选中"和"得 0 分"区分开。
        best = np.full(n, -1.0, dtype=np.float64)
        best_components: Dict[int, Dict[str, float]] = {}
        best_terms: Dict[int, List[str]] = {}
        best_queries: Dict[int, List[str]] = {}

        w_kw = float(self.weights.get("keyword", 0.0))
        w_vec = float(self.weights.get("vector", 0.0))
        w_meta = float(self.weights.get("metadata", 0.0))

        for q in queries:
            if not q or not q.strip():
                continue
            q_tokens = tokenize(q)
            if not q_tokens:
                continue

            bm25_raw = self._bm25.score_array(q_tokens)
            vec_raw = cosine_scores(embed(q, dim=self.embed_dim), self._matrix)
            bm25_norm = _minmax(bm25_raw)
            vec_norm = _minmax(vec_raw)

            q_terms = set(q_tokens)
            for idx in range(n):
                if not mask[idx]:
                    continue

                matched = self._bm25.matched_terms(q_tokens, idx)
                hit_ratio = (len(set(matched)) / len(q_terms)) if q_terms else 0.0
                # 统一转成原生 float：numpy 标量会破坏 LangGraph 的 checkpointer 序列化
                # 为什么 keyword 里还要掺命中率：BM25 对长块有长度偏置，而"查询里的关键
                # 财务词有多少真的出现在这块里"对金融问答是强信号；0.7 / 0.3 是内层固定配比，
                # 不随 RERANK_WEIGHTS 变化（外层权重才可配置）。
                keyword = float(0.7 * float(bm25_norm[idx]) + 0.3 * hit_ratio)
                meta_score = float(self._metadata_score(self.children[idx].meta, filters))
                fused = float(w_kw * keyword + w_vec * float(vec_norm[idx]) + w_meta * meta_score)

                # 完全无命中的块（BM25=0 且向量相似度<=0）直接丢弃，避免噪声污染
                # 为什么是 and 而不是 or：只要有一路给正分就保留，允许"纯语义命中"或
                # "纯词面命中"单独成立。
                if bm25_raw[idx] <= 0.0 and vec_raw[idx] <= 0.0:
                    continue

                if fused > best[idx]:
                    best[idx] = fused
                    best_components[idx] = {
                        "keyword": keyword,
                        "vector": float(vec_norm[idx]),
                        "metadata": meta_score,
                        "bm25_raw": float(bm25_raw[idx]),
                        "cosine_raw": float(vec_raw[idx]),
                    }
                    best_terms[idx] = matched
                    best_queries[idx] = [q]
                elif abs(fused - best[idx]) < 1e-12 and idx in best_queries:
                    # 为什么并列时也要记一笔：多路查询命中同一子块时把命中词与查询路取并集，
                    # 下游才能解释"这条证据是被哪几种问法共同命中的"。
                    best_queries[idx].append(q)
                    for term in matched:
                        if term not in best_terms[idx]:
                            best_terms[idx].append(term)

        # 为什么用 max(1, top_k)：调用契约是"给我证据"，top_k 为 0 或负数时仍返回至少一条，
        # 避免上层把空列表误判成"检索不到相关内容"。
        ranked = sorted(
            (i for i in range(n) if best[i] >= 0.0),
            key=lambda i: -best[i],
        )[: max(1, top_k)]

        return [self._make_snippet(i, best[i], best_components.get(i, {}), best_terms.get(i, []), best_queries.get(i, [])) for i in ranked]

    def _make_snippet(
        self,
        idx: int,
        score: float,
        components: Dict[str, float],
        matched_terms: List[str],
        queries: List[str],
    ) -> RetrievedSnippet:
        """把内部下标与分数封装成对外的 RetrievedSnippet（含父块回填）。

        参数：idx 子块下标；score 融合总分；components 分项明细（keyword / vector /
            metadata / bm25_raw / cosine_raw）；matched_terms 命中词；queries 命中的查询路。
        返回：RetrievedSnippet。
        副作用：无。字段全部做类型收敛（元数据一律 str，year 只保留 int 或 None），
            保证结果能被 JSON 序列化写进 trace。
        """
        child = self.children[idx]
        parent = self.parent_of(child)
        # 为什么要有兜底：正常情况下子块必然有父块，这里仍用 child.text 降级，
        # 避免父块缺失时整次检索直接崩掉。
        context = parent.text if parent else child.text
        return RetrievedSnippet(
            child_id=child.child_id,
            parent_id=child.parent_id,
            source_id=child.source_id,
            doc_id=child.doc_id,
            company=str(child.meta.get("company", "")),
            period=str(child.meta.get("period", "")),
            # 为什么只认 int：meta 里的 year 可能是 None 或字符串，这里统一收敛成
            # Optional[int]，避免下游做数值比较时踩类型不一致的坑（True 也算 int，但语料不会出现）。
            year=child.meta.get("year") if isinstance(child.meta.get("year"), int) else None,
            section_title=str(child.meta.get("section_title", "")),
            text=child.text,
            context=context,
            score=float(score),
            components=components,
            # 为什么截断到 12 个：命中词只用于展示与可解释性，留太多会让 trace 与 prompt 膨胀。
            matched_terms=sorted(set(matched_terms))[:12],
            queries=list(queries),
        )
