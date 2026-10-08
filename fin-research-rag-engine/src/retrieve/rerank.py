"""重排（Rerank）：召回看「不漏」，重排看「排得准」。

在 RAG 全链路中的位置
---------------------
    三路召回（retrieve/hybrid.py）→ 近似去重（同一模块）→ **本模块：精排**
        → 父块回溯与证据组装（retrieve/pipeline.py）→ 生成（src/generate）
本模块只做一件事：给**已经召回的少量候选**打一个可信的相关性分并重排序。
被谁调用：retrieve/pipeline.py 的 `rerank_candidates`（主链路），
以及 eval/、scripts/ 里做重排消融实验的脚本。

对外关键对象
------------
    Reranker / BGEReranker / LocalCrossEncoder   重排后端：接口 + 真实实现 + 本地实现
    RerankFeatures                              一条候选的全部可解释特征
    RerankedItem                                重排结果（交叉分、最终分、特征、前后名次）
    rerank_candidates(query, candidates, ...)   对候选重排并与 RRF 分、元数据分加权融合
    get_reranker(name="auto")                   取后端：优先真实 bge-reranker，不可用回落本地
输入：query + 候选列表（约 recall_top_k × 路数 条）；输出：按 final_score 降序的 RerankedItem。

为什么必须有重排
----------------
召回阶段的本质是**用便宜的方法从全库里筛出几十条候选**，所以它宁可多捞（高召回、
低精度）。但交给 LLM 的上下文窗口只有 5 条左右的位置，如果直接把召回结果塞进去，
真正有用的那条可能排在第 12 位，等于没召回。

重排用**交叉编码器**（把 query 和 document 拼在一起编码）做精排：
精度显著高于召回阶段的双塔模型，但计算贵一个量级，所以只对少量候选做。
这是流水线上的两个工种，不能互相替代。

它在流水线里的位置为什么是这里
------------------------------
    先召回（K 大）→ 再近似去重（便宜，且防止重复块占位）→ 再精排（贵，所以候选要少）
        → 最后父块回溯（只影响送给 LLM 的上下文，不影响排序）
顺序不是随意的：**重排必须放在去重之后**，否则同一段文字会被交叉编码器重复计算、白花钱；
**父块回溯必须放在重排之后**，因为父块又长又含无关段落，拿它参与打分只会把排序带偏
——排序看子块（准），喂给 LLM 看父块（全）。

本模块的实现
------------
    BGEReranker        真实后端（bge-reranker），装了 FlagEmbedding 才启用
    LocalCrossEncoder  确定性本地实现，离线 / CI / 演示用

本地实现不是"随便打个分"，它把交叉编码器能学到的信号显式拆了出来：
    词面覆盖（含 IDF 直觉：数字 / 条款号 / 代码 权重更高）
    短语命中（bigram 级，避免"客户风险"与"风险客户"被当成同一件事）
    语义相似（稠密余弦，来自同一嵌入后端）
    精确命中（问题里的条款号 / 产品代码是否原样出现——这是金融问答的强信号）
    长度规整（过短的块信息不足，过长的块是噪声）
    来源质量（OCR 脏文档要降权）
每个特征都可单独查看，因此「为什么这条排第一」能直接回答出来。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import RERANK_WEIGHTS
from ..index.embedding import EmbeddingBackend, cosine_scores, get_backend, sparse_from_text
from ..utils.text import jaccard, shingles, tokenize

__all__ = [
    "RerankFeatures",
    "RerankedItem",
    "Reranker",
    "LocalCrossEncoder",
    "BGEReranker",
    "get_reranker",
    "rerank_candidates",
    "IDEAL_CHUNK_CHARS",
]

# 重排时认为"信息密度最佳"的子块长度；偏离越远扣分越多
IDEAL_CHUNK_CHARS = 220

# 金融问答里的三类强字面信号：条款号（"第四十二条"）、产品代码（如 XX-2023-01）、关键数字。
# 单独抽成模块级正则是为了让"精确命中"这一特征可复现、可单独测试
_CLAUSE_RE = re.compile(r"第\s*[0-9一二三四五六七八九十百零]+\s*条")
_CODE_RE = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*(?:-\d{2,})+\b")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# 本地交叉编码器的特征权重（和为 1.0，便于解释）
FEATURE_WEIGHTS: Dict[str, float] = {
    "coverage": 0.34,
    "phrase": 0.16,
    "dense": 0.26,
    "exact": 0.14,
    "length": 0.05,
    "quality": 0.05,
}


@dataclass
class RerankFeatures:
    """一条候选的全部重排特征，全部落在 [0, 1]。

    字段含义（对应 LocalCrossEncoder.score_pair 里的六个信号）：
        coverage  词面覆盖：查询词有多少原样出现在块里（0.34，主导项）
        phrase    短语命中：bigram 级 Jaccard，区分「客户风险」与「风险客户」（0.16）
        dense     语义相似：稠密余弦从 [-1,1] 线性映射到 [0,1]（0.26）
        exact     精确命中：条款号 / 产品代码 / 关键数字的原样出现比例（0.14）
        length    长度规整：越接近 IDEAL_CHUNK_CHARS 越高（0.05）
        quality   来源质量：OCR 脏文档降权（0.05）
        metadata  元数据契合度：**不在 FEATURE_WEIGHTS 里**，只在 rerank_candidates
                  的 final_score 中单独加权（见 as_score 的说明）
    """

    coverage: float = 0.0
    phrase: float = 0.0
    dense: float = 0.0
    exact: float = 0.0
    length: float = 0.0
    quality: float = 1.0
    metadata: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        """把全部特征导出成 dict（保留 4 位小数），用于轨迹记录与"为什么排第一"的解释。"""
        return {k: round(float(v), 4) for k, v in self.__dict__.items()}

    def as_score(self) -> float:
        """把特征加权成一个 [0, 1] 的交叉编码器分数。

        返回：Σ FEATURE_WEIGHTS[k] × 特征值。
        注：实际实现里 `metadata` 不在 FEATURE_WEIGHTS 中，`get(k, 0.0)` 取到 0.0，
            因此它**不参与** as_score；元数据分由 rerank_candidates 在 final_score 里单独加权。
            另外若外部直接构造 RerankFeatures，未赋值的字段为默认值，同样按权重计入。
        """
        return float(sum(FEATURE_WEIGHTS.get(k, 0.0) * float(v) for k, v in self.__dict__.items()))


@dataclass
class RerankedItem:
    """重排后的一条结果。

    关键属性：
        item         原始候选对象（Candidate）
        cross_score  交叉编码器分（本地实现即 feats.as_score()）
        final_score  最终分 = w_cross·cross + w_rrf·rrf_norm + w_metadata·metadata
        features     特征明细（仅本地实现会填满；真实后端只有 length / quality / metadata）
        rrf_score    召回阶段的 RRF 融合分（未归一化）
        rank_before  重排前名次（在传入候选序列中的下标，0 基，RerankedItem 构造时写入）
        rank_after   重排后名次（0 基，按 final_score 排序后回填），用于观察重排把谁顶上来了
    """

    item: Any
    cross_score: float
    final_score: float
    features: RerankFeatures = field(default_factory=RerankFeatures)
    rrf_score: float = 0.0
    rank_before: int = 0
    rank_after: int = 0

    @property
    def record_id(self) -> str:
        """候选标识：优先取 child_id（Candidate），退化为 record_id，都没有则空串。"""
        return str(getattr(self.item, "child_id", getattr(self.item, "record_id", "")))


class Reranker:
    """重排后端接口。`compute_score` 的入参形式与 bge-reranker 保持一致。

    这样切换真实后端与本地实现时，调用方（rerank_candidates）不需要任何改动。
    """

    name = "base"

    @property
    def available(self) -> bool:  # pragma: no cover - 接口默认实现
        """后端是否可用；接口默认视为可用，真实后端会真的去尝试加载模型。"""
        return True

    def compute_score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:  # pragma: no cover
        """对 [(query, document)] 逐对打分，返回与输入等长的分数列表。

        参数：pairs 查询—文档对（bge-reranker 的入参形式）。
        返回：相关性分数列表；抽象接口直接抛 NotImplementedError。
        """
        raise NotImplementedError


class BGEReranker(Reranker):
    """真实 bge-reranker 后端（可选依赖 FlagEmbedding）。

    关键属性：model 模型名（默认 "BAAI/bge-reranker-v2-m3"）；_impl 已加载的实现；
              _tried 是否尝试过加载（避免依赖缺失时反复触发昂贵的导入失败）。
    """

    name = "bge-reranker"

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3") -> None:
        """记录模型名，暂不加载（真正加载推迟到第一次打分）。

        参数：model HuggingFace 模型名或本地路径。
        返回：无。副作用：仅设置 _impl=None、_tried=False，不触发任何 IO。
        """
        self.model = model
        self._impl = None
        self._tried = False

    def _load(self):
        """惰性加载 FlagReranker；失败时把 _impl 置 None 并只尝试一次。

        返回：FlagReranker 实例，或 None（未安装 FlagEmbedding / 模型加载失败）。
        副作用：首次调用会导入 FlagEmbedding 并加载模型权重（可能很慢、可能联网）；
                _tried 一经置位就不会再重试，避免每次检索都卡在失败的导入上。
        异常：本方法内部吞掉所有加载异常，不向上抛。
        """
        if not self._tried:
            self._tried = True
            try:
                from FlagEmbedding import FlagReranker  # type: ignore

                self._impl = FlagReranker(self.model, use_fp16=True)
            except Exception:  # noqa: BLE001
                # 静默降级：装了就用真实模型，没装就交给上层回落到本地实现
                self._impl = None
        return self._impl

    @property
    def available(self) -> bool:
        """bge-reranker 是否真的可用（会触发一次加载尝试并缓存结果）。"""
        return self._load() is not None

    def compute_score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """用真实交叉编码器给候选对打分（normalize=True，分数已归一到 [0,1] 区间）。

        参数：pairs [(query, document)] 列表。
        返回：与 pairs 等长的 float 列表（后端只返回单值时也包成列表）。
        异常：后端不可用时抛 RuntimeError（提示安装 FlagEmbedding）。
        """
        impl = self._load()
        if impl is None:
            raise RuntimeError("bge-reranker 后端不可用：请安装 FlagEmbedding")
        raw = impl.compute_score([list(p) for p in pairs], normalize=True)
        if isinstance(raw, float):
            return [float(raw)]
        return [float(x) for x in raw]


class LocalCrossEncoder(Reranker):
    """确定性本地交叉编码器。

    真实交叉编码器把 [query, doc] 一起送进 Transformer，让 query 的每个词与 doc 的
    每个词做注意力交互。本地实现把这个交互拆成若干**可解释的显式特征**，
    在中文金融语料上足以把「真正回答问题的那一块」顶到前面。

    与稠密双塔的关键差别：双塔是 query 与 doc **各自**编码后比一个余弦，
    交互只发生在最后一刻，细粒度信息（数字、条款号）早被压掉了；
    本实现里的 coverage / phrase / exact 是**逐词、逐短语**的比较，
    因此能把「300 万」与「100 万」这种双塔分不出来的差异区分开。

    关键属性：backend 嵌入后端（只为 dense 特征提供余弦分，缺省 get_backend()）。
    """

    name = "local-cross-encoder"

    def __init__(self, backend: Optional[EmbeddingBackend] = None) -> None:
        """记录嵌入后端。

        参数：backend 嵌入后端；None 时用 get_backend() 按配置选择。
        返回：无。副作用：无（后端自身可能是惰性加载的）。
        """
        self.backend = backend or get_backend()

    def score_pair(self, query: str, text: str, quality: float = 1.0, dense: Optional[float] = None) -> RerankFeatures:
        """给一条 (query, text) 算出全部可解释特征。

        参数：query 用户问题；text 候选子块正文；
              quality 来源质量（OCR 脏文档 < 1.0，会传导到 quality 特征）；
              dense 可选的稠密余弦分，**传了就不再重复编码**（批量重排时用来省算力）。
        返回：RerankFeatures，各特征已落在 [0, 1]。
        副作用：dense 为 None 时会调用后端 encode（一次编码 query 与 text 两条文本）。
        """
        q_tokens = tokenize(query)
        d_tokens = tokenize(text)
        q_set, d_set = set(q_tokens), set(d_tokens)

        # 1) 词面覆盖：查询里的词有多少真的出现在这块里
        # 用"比例"而不是"命中个数"：长问题词多，用个数会让长问题天然占优
        coverage = (len(q_set & d_set) / len(q_set)) if q_set else 0.0

        # 2) 短语命中：bigram 级重叠，能区分「客户风险」与「风险客户」
        # 单词集合会认为这两者完全一样，而金融语料里它们的含义恰好相反
        phrase = jaccard(shingles(q_tokens, 2), shingles(d_tokens, 2))

        # 3) 语义相似：稠密余弦，从 [-1,1] 映射到 [0,1]
        if dense is None:
            vec = self.backend.encode([query, text])
            dense = float(cosine_scores(vec[0], vec[1:2])[0]) if vec.shape[0] == 2 else 0.0
        # 特征必须非负才能加权，所以把余弦线性拉到 [0,1]（-1 → 0，0 → 0.5，1 → 1）
        dense01 = max(0.0, min(1.0, (float(dense) + 1.0) / 2.0))

        # 4) 精确命中：问题里的条款号 / 产品代码 / 关键数字原样出现
        # 这是本地实现对双塔最关键的补偿：数字与编号在向量里几乎不可分
        exact = self._exact_score(query, text)

        # 5) 长度规整
        length = self._length_score(len(text))

        feats = RerankFeatures(
            coverage=coverage,
            phrase=phrase,
            dense=dense01,
            exact=exact,
            length=length,
            quality=max(0.0, min(1.0, float(quality))),
            metadata=1.0,
        )
        return feats

    @staticmethod
    def _exact_score(query: str, text: str) -> float:
        """条款号 / 产品代码 / 数字的精确命中率。命中越多分越高，没这类元素时给中性分。

        参数：query 问题；text 候选正文。
        返回：[0, 1] 的命中比例；问题里没有可精确匹配的元素时返回 0.75 中性分
              （没有可匹配项不是候选的错，不该因此被扣分）。
        """
        signals: List[str] = []
        signals.extend(m.group(0).replace(" ", "") for m in _CLAUSE_RE.finditer(query))
        signals.extend(m.group(0).upper() for m in _CODE_RE.finditer(query.upper()))
        # 只取长度 >= 3 的数字，避免 "2" 这种噪声
        signals.extend(n for n in _NUM_RE.findall(query) if len(n) >= 3)

        if not signals:
            return 0.75  # 中性：问题里没有可精确匹配的元素，不该因此被扣分
        # 直接做子串判断（而不是匹配分词结果）：编号、金额常被分词切碎，子串更稳
        hit = sum(1 for s in signals if s and s in text)
        return hit / len(signals)

    @staticmethod
    def _length_score(length: int) -> float:
        """离 IDEAL_CHUNK_CHARS 越远分越低；过短的块信息量不足，过长的块是噪声。

        参数：length 候选正文的字符数。
        返回：min/max 比值（越接近 IDEAL_CHUNK_CHARS = 220 越接近 1.0）；长度 <= 0 时返回 0.0。
        """
        if length <= 0:
            return 0.0
        ratio = min(length, IDEAL_CHUNK_CHARS) / max(length, IDEAL_CHUNK_CHARS)
        return ratio

    def compute_score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """批量打分：逐对调用 score_pair 并取加权总分。

        参数：pairs [(query, text)] 列表。
        返回：与 pairs 等长的 [0, 1] 分数列表。
        说明：这里每对各自编码，不复用稠密分；若要复用请走 rerank_candidates
              （它用 score_pair 并统一管理后端实例）。
        """
        return [self.score_pair(q, d).as_score() for q, d in pairs]


# 后端缓存：get_reranker 按名字缓存实例，避免每问一次就重新尝试加载模型
_RERANKER_CACHE: Dict[str, Reranker] = {}


def get_reranker(name: str = "auto") -> Reranker:
    """取重排后端；auto 时优先真实 bge-reranker，不可用则回落本地实现。

    参数：name 后端名，可取 "bge-reranker" / "bge" / "bgereranker"、
          "local" / "local-cross-encoder"，其它（含 "auto"）走自动选择。
    返回：Reranker 实例；同一 name 复用缓存实例，保证模型只加载一次。
    副作用：名字含 bge 时会触发一次模型加载尝试（可能很慢 / 需要联网），失败则静默降级。
    """
    key = (name or "auto").lower()
    if key in _RERANKER_CACHE:
        return _RERANKER_CACHE[key]
    reranker: Reranker
    if key in ("bge-reranker", "bge", "bgereranker"):
        # 明确点名要真实后端：不可用也回落本地，而不是直接报错（服务不因缺依赖而挂掉）
        candidate = BGEReranker()
        reranker = candidate if candidate.available else LocalCrossEncoder()
    elif key in ("local", "local-cross-encoder"):
        reranker = LocalCrossEncoder()
    else:
        candidate = BGEReranker()
        reranker = candidate if candidate.available else LocalCrossEncoder()
    _RERANKER_CACHE[key] = reranker
    return reranker


def rerank_candidates(
    query: str,
    candidates: Sequence[Any],
    reranker: Optional[Reranker] = None,
    top_k: int = 5,
    weights: Optional[Dict[str, float]] = None,
    rrf_scores: Optional[Dict[str, float]] = None,
    metadata_scores: Optional[Dict[str, float]] = None,
    text_of=lambda item: item.text,
    quality_of=lambda item: getattr(item, "quality_score", 1.0),
    id_of=lambda item: str(getattr(item, "child_id", getattr(item, "record_id", ""))),
) -> List[RerankedItem]:
    """对候选做交叉编码器重排，再与 RRF 分、元数据契合度加权融合。

    融合而不是直接采用重排分，是因为三者回答的是不同问题：
        交叉分   —— 这条**内容**和问题有多相关（最贵也最准）
        RRF 分   —— 这条在**三路召回**里的共识程度（多路都命中说明可信）
        元数据分 —— 这条**属不属于**用户限定的范围（机构 / 年份 / 版本）
    只看其中任何一个都会有明显短板。

    参数：
        query           用户问题
        candidates      待重排候选（Candidate 或任何具备 text/quality_score 的对象）
        reranker        重排后端；None 时取 get_reranker()
        top_k           返回条数（默认 5，与 FINAL_TOP_K 一致）
        weights         融合权重；None 时取 config.RERANK_WEIGHTS（cross 0.55 / rrf 0.30 / metadata 0.15）
        rrf_scores      {候选 id: RRF 融合分}，缺失时按 0 处理
        metadata_scores {候选 id: 元数据分}，缺失时按 1.0 处理（不因缺分而受罚）
        text_of / quality_of / id_of  取字段的适配函数，默认按 Candidate 的属性取
    返回：按 final_score 降序的 `List[RerankedItem]`，截断到 top_k；候选为空时返回 []。
    副作用：会真正调用重排后端（可能是模型推理）；不修改传入候选本身。
    """
    if not candidates:
        return []

    engine = reranker or get_reranker()
    w = dict(weights or RERANK_WEIGHTS)
    rows = list(candidates)

    pairs = [(query, text_of(item)) for item in rows]
    cross_scores = engine.compute_score(pairs)

    rrf = rrf_scores or {}
    meta_map = metadata_scores or {}
    # RRF 分与交叉分不同量纲，融合前必须归一到 [0,1]。
    # 注：min-max 的极值来自"当前这批候选"，所以 rrf_norm 只在本批内可比；
    # 好在 RRF 分本身表达的是名次共识，跨批次绝对高低本来也不该直接比较。
    rrf_norm = _minmax_dict({id_of(item): float(rrf.get(id_of(item), 0.0)) for item in rows})

    results: List[RerankedItem] = []
    for idx, item in enumerate(rows):
        key = id_of(item)
        feats = RerankFeatures(
            coverage=0.0,
            phrase=0.0,
            dense=0.0,
            exact=0.0,
            # 真实后端不暴露中间特征，只能把"能自己算的"补上（长度 / 质量 / 元数据），
            # 让前端展示与日志至少在两种后端下字段结构一致
            length=LocalCrossEncoder._length_score(len(text_of(item))),
            quality=max(0.0, min(1.0, float(quality_of(item)))),
            metadata=float(meta_map.get(key, 1.0)),
        )
        # 若后端是本地实现，把可解释特征一并取回
        if isinstance(engine, LocalCrossEncoder):
            feats = engine.score_pair(query, text_of(item), quality=quality_of(item))
            feats.metadata = float(meta_map.get(key, 1.0))
            cross = feats.as_score()
        else:
            cross = float(cross_scores[idx])

        # 三项各回答一个不同的问题：cross=内容相关性、rrf=三路共识、metadata=是否落在用户限定范围；
        # 权重见 config.RERANK_WEIGHTS（0.55 / 0.30 / 0.15），任何一项为 0 都意味着放弃对应信号
        final = (
            float(w.get("cross", 0.0)) * cross
            + float(w.get("rrf", 0.0)) * float(rrf_norm.get(key, 0.0))
            + float(w.get("metadata", 0.0)) * feats.metadata
        )
        results.append(
            RerankedItem(
                item=item,
                cross_score=float(cross),
                final_score=float(final),
                features=feats,
                rrf_score=float(rrf.get(key, 0.0)),
                rank_before=idx,
            )
        )

    # 同分时用 rank_before 兜底：保证同样输入永远得到同样顺序（评测结果可复现）
    results.sort(key=lambda r: (-r.final_score, r.rank_before))
    for pos, row in enumerate(results):
        row.rank_after = pos
    return results[: max(1, top_k)]


def _minmax_dict(scores: Dict[str, float]) -> Dict[str, float]:
    """把字典形式的分数 min-max 归一化到 [0,1]；全相等时返回全 0，避免虚假高分。

    参数：scores {id: 原始分}。
    返回：{id: 归一化分}；空输入返回 {}。
    说明：全相等时若仍归一化成 1.0，会让"这批候选都不相关"看起来像满分，
          所以这里返回全 0——宁可让该项不贡献分数，也不要给出误导性的高分。
    """
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
