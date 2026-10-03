"""向量库（Milvus 语义的内存实现）。

为什么要自己写一层
------------------
生产环境用 Milvus，但**项目的正确性不该依赖一个需要 docker-compose 才能起来的组件**。
本模块按 Milvus 的语义实现了一个内存版本，接口刻意对齐：

    create_collection / has_collection / drop_collection / list_collections
    collection.insert(records) / search(...) / query(expr) / count() / flush()

因此切换到真实 Milvus 时，只需要替换本模块的三个方法（insert / search / query），
上层检索、重排、缓存、评测**一行都不用改**。同时它让「过滤表达式」这件事
可以被单元测试覆盖——过滤写错会静默丢证据，是最难查的一类线上问题。

支持的能力：
    * 稠密向量检索（COSINE / IP / L2）
    * 稀疏词权重检索（点积）
    * 标量过滤表达式（复用 chunking.metadata.MetadataFilter，语法同 Milvus）
    * 批量插入与按表达式删除
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..chunking.metadata import MetadataFilter
from ..index.embedding import cosine_scores, sparse_dot, sparse_norm

__all__ = ["VectorRecord", "SearchHit", "Collection", "MilvusLiteClient"]


@dataclass
class VectorRecord:
    """一条向量记录：稠密向量 + 稀疏权重 + 标量元数据。"""

    record_id: str
    dense: Optional[np.ndarray] = None
    sparse: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class SearchHit:
    """一条检索命中。同时保留稠密分与稀疏分，便于解释「为什么它被召回」。"""

    record_id: str
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "record_id": self.record_id,
            "score": round(float(self.score), 6),
            "dense_score": round(float(self.dense_score), 6),
            "sparse_score": round(float(self.sparse_score), 6),
            "source_id": self.meta.get("source_id", ""),
            "section_title": self.meta.get("section_title", ""),
        }


class Collection:
    """一个向量集合。构建后支持追加与按表达式删除。"""

    def __init__(self, name: str, dim: int, metric: str = "COSINE") -> None:
        self.name = name
        self.dim = int(dim)
        self.metric = metric.upper()
        if self.metric not in ("COSINE", "IP", "L2"):
            raise ValueError(f"不支持的度量方式：{metric}")
        self._records: List[VectorRecord] = []
        self._index: Dict[str, int] = {}
        self._matrix: Optional[np.ndarray] = None   # 稠密矩阵缓存，插入后失效
        self._dirty = True

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def insert(self, records: Sequence[VectorRecord]) -> int:
        inserted = 0
        for rec in records:
            if rec.dense is not None and rec.dense.shape != (self.dim,):
                raise ValueError(f"向量维度不符：期望 {self.dim}，得到 {rec.dense.shape}")
            if rec.record_id in self._index:
                # 幂等：同 ID 覆盖，避免重复灌数据导致同一块命中两次
                self._records[self._index[rec.record_id]] = rec
            else:
                self._index[rec.record_id] = len(self._records)
                self._records.append(rec)
            inserted += 1
        self._dirty = True
        return inserted

    def delete(self, expr: Optional[str] = None) -> int:
        """按过滤表达式删除；expr 为空则清空集合。"""
        if not expr:
            removed = len(self._records)
            self._records, self._index = [], {}
            self._dirty = True
            return removed
        cond = MetadataFilter.parse(expr)
        kept = [rec for rec in self._records if not cond.matches(rec.meta)]
        removed = len(self._records) - len(kept)
        self._records = kept
        self._index = {rec.record_id: i for i, rec in enumerate(kept)}
        self._dirty = True
        return removed

    def flush(self) -> None:
        """构建向量矩阵缓存（等价于 Milvus 的 flush + load）。"""
        self._matrix = self._build_matrix()
        self._dirty = False

    def _build_matrix(self) -> np.ndarray:
        if not self._records:
            return np.zeros((0, self.dim), dtype=np.float64)
        rows = [
            rec.dense if rec.dense is not None else np.zeros(self.dim, dtype=np.float64)
            for rec in self._records
        ]
        return np.vstack(rows).astype(np.float64)

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None or self._dirty:
            self.flush()
        assert self._matrix is not None
        return self._matrix

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def count(self) -> int:
        return len(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> List[VectorRecord]:
        return list(self._records)

    def get(self, record_id: str) -> Optional[VectorRecord]:
        idx = self._index.get(record_id)
        return self._records[idx] if idx is not None else None

    def query(self, expr: Optional[str] = None, limit: int = 100) -> List[Dict[str, object]]:
        """标量查询：只按元数据取记录（不走向量检索）。"""
        cond = MetadataFilter.parse(expr)
        out: List[Dict[str, object]] = []
        for rec in self._records:
            if cond.matches(rec.meta):
                out.append({"record_id": rec.record_id, **rec.meta})
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def _candidate_indexes(self, expr: Optional[str]) -> List[int]:
        cond = MetadataFilter.parse(expr)
        if cond.is_empty:
            return list(range(len(self._records)))
        return [i for i, rec in enumerate(self._records) if cond.matches(rec.meta)]

    def search_dense(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        expr: Optional[str] = None,
    ) -> List[SearchHit]:
        """稠密向量检索（返回 (record_id, 原始度量分)）。"""
        if not self._records or vector.size == 0:
            return []
        candidates = self._candidate_indexes(expr)
        if not candidates:
            return []

        if self.metric == "COSINE":
            raw = cosine_scores(vector, self.matrix)
            scores = {i: float(raw[i]) for i in candidates}
        elif self.metric == "IP":
            raw = self.matrix @ vector
            scores = {i: float(raw[i]) for i in candidates}
        else:  # L2：转成负距离，保证"越大越好"的统一口径
            diff = self.matrix[candidates] - vector
            dist = np.linalg.norm(diff, axis=1)
            scores = {idx: -float(d) for idx, d in zip(candidates, dist)}

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[: max(1, top_k)]
        return [
            SearchHit(
                record_id=self._records[i].record_id,
                score=score,
                dense_score=score,
                meta=dict(self._records[i].meta),
            )
            for i, score in ranked
        ]

    def search_sparse(
        self,
        sparse: Dict[str, float],
        top_k: int = 10,
        expr: Optional[str] = None,
    ) -> List[SearchHit]:
        """稀疏词权重检索（归一化后的点积）。"""
        if not self._records or not sparse:
            return []
        candidates = self._candidate_indexes(expr)
        if not candidates:
            return []
        q_norm = sparse_norm(sparse)
        scores: List[Tuple[int, float]] = []
        for i in candidates:
            raw = sparse_dot(sparse, self._records[i].sparse)
            if raw > 0.0:
                scores.append((i, raw / q_norm))
        scores.sort(key=lambda kv: -kv[1])
        return [
            SearchHit(
                record_id=self._records[i].record_id,
                score=score,
                sparse_score=score,
                meta=dict(self._records[i].meta),
            )
            for i, score in scores[: max(1, top_k)]
        ]


class MilvusLiteClient:
    """极简 Milvus 客户端：管理多个 Collection。"""

    def __init__(self) -> None:
        self._collections: Dict[str, Collection] = {}

    def has_collection(self, name: str) -> bool:
        return name in self._collections

    def create_collection(self, name: str, dim: int, metric: str = "COSINE") -> Collection:
        if name in self._collections:
            existing = self._collections[name]
            if existing.dim != int(dim):
                raise ValueError(f"集合 {name} 已存在且维度不同（{existing.dim} != {dim}）")
            return existing
        collection = Collection(name=name, dim=dim, metric=metric)
        self._collections[name] = collection
        return collection

    def get_collection(self, name: str) -> Collection:
        if name not in self._collections:
            raise KeyError(f"集合不存在：{name}（请先 create_collection）")
        return self._collections[name]

    def list_collections(self) -> List[str]:
        return sorted(self._collections)

    def drop_collection(self, name: str) -> bool:
        return self._collections.pop(name, None) is not None

    def describe(self) -> List[Dict[str, object]]:
        return [
            {"name": name, "dim": col.dim, "metric": col.metric, "count": col.count()}
            for name, col in sorted(self._collections.items())
        ]
