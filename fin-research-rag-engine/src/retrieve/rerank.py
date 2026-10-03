"""重排（Rerank）：召回看「不漏」，重排看「排得准」。

为什么必须有重排
----------------
召回阶段的本质是**用便宜的方法从全库里筛出几十条候选**，所以它宁可多捞（高召回、
低精度）。但交给 LLM 的上下文窗口只有 5 条左右的位置，如果直接把召回结果塞进去，
真正有用的那条可能排在第 12 位，等于没召回。

重排用**交叉编码器**（把 query 和 document 拼在一起编码）做精排：
精度显著高于召回阶段的双塔模型，但计算贵一个量级，所以只对少量候选做。
这是流水线上的两个工种，不能互相替代。

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
    """一条候选的全部重排特征，全部落在 [0, 1]。"""

    coverage: float = 0.0
    phrase: float = 0.0
    dense: float = 0.0
    exact: float = 0.0
    length: float = 0.0
    quality: float = 1.0
    metadata: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {k: round(float(v), 4) for k, v in self.__dict__.items()}

    def as_score(self) -> float:
        """把特征加权成一个 [0, 1] 的交叉编码器分数。"""
        return float(sum(FEATURE_WEIGHTS.get(k, 0.0) * float(v) for k, v in self.__dict__.items()))


@dataclass
class RerankedItem:
    """重排后的一条结果。"""

    item: Any
    cross_score: float
    final_score: float
    features: RerankFeatures = field(default_factory=RerankFeatures)
    rrf_score: float = 0.0
    rank_before: int = 0
    rank_after: int = 0

    @property
    def record_id(self) -> str:
        return str(getattr(self.item, "child_id", getattr(self.item, "record_id", "")))


class Reranker:
    """重排后端接口。`compute_score` 的入参形式与 bge-reranker 保持一致。"""

    name = "base"

    @property
    def available(self) -> bool:  # pragma: no cover - 接口默认实现
        return True

    def compute_score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:  # pragma: no cover
        raise NotImplementedError


class BGEReranker(Reranker):
    """真实 bge-reranker 后端（可选依赖 FlagEmbedding）。"""

    name = "bge-reranker"

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model = model
        self._impl = None
        self._tried = False

    def _load(self):
        if not self._tried:
            self._tried = True
            try:
                from FlagEmbedding import FlagReranker  # type: ignore

                self._impl = FlagReranker(self.model, use_fp16=True)
            except Exception:  # noqa: BLE001
                self._impl = None
        return self._impl

    @property
    def available(self) -> bool:
        return self._load() is not None

    def compute_score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
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
    """

    name = "local-cross-encoder"

    def __init__(self, backend: Optional[EmbeddingBackend] = None) -> None:
        self.backend = backend or get_backend()

    def score_pair(self, query: str, text: str, quality: float = 1.0, dense: Optional[float] = None) -> RerankFeatures:
        q_tokens = tokenize(query)
        d_tokens = tokenize(text)
        q_set, d_set = set(q_tokens), set(d_tokens)

        # 1) 词面覆盖：查询里的词有多少真的出现在这块里
        coverage = (len(q_set & d_set) / len(q_set)) if q_set else 0.0

        # 2) 短语命中：bigram 级重叠，能区分「客户风险」与「风险客户」
        phrase = jaccard(shingles(q_tokens, 2), shingles(d_tokens, 2))

        # 3) 语义相似：稠密余弦，从 [-1,1] 映射到 [0,1]
        if dense is None:
            vec = self.backend.encode([query, text])
            dense = float(cosine_scores(vec[0], vec[1:2])[0]) if vec.shape[0] == 2 else 0.0
        dense01 = max(0.0, min(1.0, (float(dense) + 1.0) / 2.0))

        # 4) 精确命中：问题里的条款号 / 产品代码 / 关键数字原样出现
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
        """条款号 / 产品代码 / 数字的精确命中率。命中越多分越高，没这类元素时给中性分。"""
        signals: List[str] = []
        signals.extend(m.group(0).replace(" ", "") for m in _CLAUSE_RE.finditer(query))
        signals.extend(m.group(0).upper() for m in _CODE_RE.finditer(query.upper()))
        # 只取长度 >= 3 的数字，避免 "2" 这种噪声
        signals.extend(n for n in _NUM_RE.findall(query) if len(n) >= 3)

        if not signals:
            return 0.75  # 中性：问题里没有可精确匹配的元素，不该因此被扣分
        hit = sum(1 for s in signals if s and s in text)
        return hit / len(signals)

    @staticmethod
    def _length_score(length: int) -> float:
        """离 IDEAL_CHUNK_CHARS 越远分越低；过短的块信息量不足，过长的块是噪声。"""
        if length <= 0:
            return 0.0
        ratio = min(length, IDEAL_CHUNK_CHARS) / max(length, IDEAL_CHUNK_CHARS)
        return ratio

    def compute_score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        return [self.score_pair(q, d).as_score() for q, d in pairs]


_RERANKER_CACHE: Dict[str, Reranker] = {}


def get_reranker(name: str = "auto") -> Reranker:
    """取重排后端；auto 时优先真实 bge-reranker，不可用则回落本地实现。"""
    key = (name or "auto").lower()
    if key in _RERANKER_CACHE:
        return _RERANKER_CACHE[key]
    reranker: Reranker
    if key in ("bge-reranker", "bge", "bgereranker"):
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
    rrf_norm = _minmax_dict({id_of(item): float(rrf.get(id_of(item), 0.0)) for item in rows})

    results: List[RerankedItem] = []
    for idx, item in enumerate(rows):
        key = id_of(item)
        feats = RerankFeatures(
            coverage=0.0,
            phrase=0.0,
            dense=0.0,
            exact=0.0,
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

    results.sort(key=lambda r: (-r.final_score, r.rank_before))
    for pos, row in enumerate(results):
        row.rank_after = pos
    return results[: max(1, top_k)]


def _minmax_dict(scores: Dict[str, float]) -> Dict[str, float]:
    """把字典形式的分数 min-max 归一化到 [0,1]；全相等时返回全 0，避免虚假高分。"""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
