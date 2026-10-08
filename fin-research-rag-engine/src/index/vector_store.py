"""向量库（Milvus 语义的内存实现）：三路召回里「稠密 + 稀疏」两路的存储与检索层。

在 RAG 全链路中的位置
---------------------
    chunking 产出子块 -> embedding 得到稠密向量与稀疏权重
        -> **本模块：把 (record_id, dense, sparse, meta) 存进集合，并按向量或标量条件取回**
            -> src/retrieve/hybrid.py（`HybridRetriever` 建集合、查稠密/稀疏两路，
               标量过滤直接以表达式字符串传入 `search_dense(..., expr=...)`）
                -> 融合、重排、生成

输入：
    `VectorRecord`（稠密向量 / 稀疏权重 / 扁平元数据三件套）
    查询向量、查询稀疏权重、Milvus 风格过滤表达式字符串
输出：
    `SearchHit`（含 record_id、总分、稠密分、稀疏分、元数据副本）或
    `Collection.query()` 的元数据字典列表
主要调用方：`src/retrieve/hybrid.py`（生产唯一调用点）、`tests/conftest.py`、`tests/test_index.py`。
副作用/异常：
    纯内存、无网络、无磁盘；`create_collection`/`insert`/`get_collection` 在参数非法时抛
    ValueError / KeyError；过滤表达式非法时由 `MetadataFilter` 抛 `FilterError`。

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
    """一条向量记录：稠密向量 + 稀疏权重 + 标量元数据。

    三件套并存是刻意的——同一个 record_id 的稠密与稀疏表示必须来自同一段文本、
    同一后端，否则两路召回会各说各话，融合阶段出现对不齐的假象。

    关键属性：
        record_id  记录唯一 ID。本项目里就是 `ChildChunk.child_id`，
                   命中后靠它回链到子块 -> 父块 -> 原文（引用可追溯的起点）
        dense      稠密向量（np.ndarray，形状应为 (dim,)）；可为 None（只做稀疏/标量检索时）
        sparse     token -> 权重的稀疏表示（默认空字典）
        meta       扁平元数据字典；标量过滤、排序特征、引用展示都读它
    """

    record_id: str
    dense: Optional[np.ndarray] = None
    sparse: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class SearchHit:
    """一条检索命中。同时保留稠密分与稀疏分，便于解释「为什么它被召回」。

    关键属性：
        record_id     命中的记录 ID（= 子块 ID）
        score         本次检索使用的总分。**注：实际实现为**稠密检索时等于 dense_score、
                      稀疏检索时等于 sparse_score；本类不做两路融合（融合在 RRF 层完成）
        dense_score   稠密度量分，只在 `search_dense()` 里被填充
        sparse_score  稀疏度量分，只在 `search_sparse()` 里被填充
        meta          元数据**副本**（`dict(...)`），改动它不会影响集合里的记录
    """

    record_id: str
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """导出命中对象的字典形式（供 API / 日志 / 评测使用）。

        参数：无。
        返回：
            dict，含 record_id、score、dense_score、sparse_score 以及从 meta 里
            抽出的 source_id、section_title；三个分数都 round 到 6 位小数，
            meta 缺失的字段回退为空串（`""`）而不是 None。
        副作用/异常：无；不抛异常（meta 非 dict 的极端情况会 AttributeError，
                    正常构造路径不会出现）。
        """
        return {
            "record_id": self.record_id,
            "score": round(float(self.score), 6),
            "dense_score": round(float(self.dense_score), 6),
            "sparse_score": round(float(self.sparse_score), 6),
            "source_id": self.meta.get("source_id", ""),
            "section_title": self.meta.get("section_title", ""),
        }


class Collection:
    """一个向量集合。构建后支持追加与按表达式删除。

    语义上对齐 Milvus 的 collection：名字 + 维度 + 度量方式固定，记录列表可增长。
    检索前会先按标量表达式算候选下标（召回前过滤），再只在候选上算距离，
    因此过滤条件越精确，计算量越小、结果越干净。

    关键属性：
        name       集合名（`src.config.COLLECTION_NAME` 传入）
        dim        向量维度，插入时会校验每条记录的 dense 形状
        metric     度量方式，已归一化为大写：COSINE / IP / L2
        _records   记录列表；**下标即检索返回的 i，顺序稳定不变**
        _index     record_id -> _records 下标，用于幂等覆盖与 O(1) 取回
        _matrix    稠密矩阵缓存（(N, dim)）；插入/删除后失效，见 `_dirty`
        _dirty     缓存是否失效；`matrix` 属性会在需要时自动重建
    """

    def __init__(self, name: str, dim: int, metric: str = "COSINE") -> None:
        """创建一个空集合。

        参数：
            name    集合名
            dim     向量维度（会被 `int()` 转换；后续 insert 按它校验）
            metric  度量方式，默认 "COSINE"；内部统一 `upper()` 后只接受
                    COSINE / IP / L2
        返回：无（构造函数）。
        副作用/异常：
            初始化空记录列表与失效的矩阵缓存；
            metric 不受支持时抛 `ValueError`（消息里带上原始入参）。
        """
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
        """批量插入/覆盖记录（幂等：同 record_id 覆盖而不追加）。

        参数：records 待写入的 `VectorRecord` 序列（可为空）。
        返回：int，实际处理的记录条数（**含覆盖**，不是净增条数）。
        副作用/异常：
            修改 `_records` / `_index` 并将矩阵缓存标记为失效（下次访问 `matrix` 时重建）。
            dense 形状不等于 (dim,) 时抛 `ValueError`——这条校验必须在写之前失败，
            否则矩阵堆叠阶段才报错，堆栈会指向无关的 `_build_matrix`，极难排查。
        """
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
        """按过滤表达式删除；expr 为空则清空集合。

        参数：
            expr  Milvus 风格过滤表达式（如 `version = "v1"`）；
                  None 或空串表示**清空全部记录**
        返回：int，被删除的记录条数。
        副作用/异常：
            重建 `_records` 与 `_index`（下标会整体重排），并让矩阵缓存失效。
            表达式非法时抛 `FilterError`；**异常发生在删除之前**，集合保持原样不会被删一半。
        """
        if not expr:
            removed = len(self._records)
            self._records, self._index = [], {}
            self._dirty = True
            return removed
        cond = MetadataFilter.parse(expr)
        # 保留「不匹配」的记录：过滤语义是"删掉命中的"，与 query 的保留语义正好相反
        kept = [rec for rec in self._records if not cond.matches(rec.meta)]
        removed = len(self._records) - len(kept)
        self._records = kept
        self._index = {rec.record_id: i for i, rec in enumerate(kept)}
        self._dirty = True
        return removed

    def flush(self) -> None:
        """构建向量矩阵缓存（等价于 Milvus 的 flush + load）。

        参数：无。
        返回：None。
        副作用/异常：
            写入 `_matrix` 并把 `_dirty` 置 False；无异常（无 dense 的记录按零向量补齐）。
        """
        self._matrix = self._build_matrix()
        self._dirty = False

    def _build_matrix(self) -> np.ndarray:
        """把所有记录的 dense 堆叠成 (N, dim) 矩阵（内部方法）。

        参数：无。
        返回：
            np.ndarray，float64，形状 (N, dim)；无记录时返回 (0, dim) 的空矩阵
            （**保持二维**，否则上层 `cosine_scores()` 的 axis 语义会错位）。
        副作用/异常：
            无副作用（返回新数组）；不抛异常。缺 dense 的记录用零向量占位，
            这样矩阵行号与 `_records` 下标严格一致——行号错位会让检索结果张冠李戴。
        """
        if not self._records:
            return np.zeros((0, self.dim), dtype=np.float64)
        rows = [
            rec.dense if rec.dense is not None else np.zeros(self.dim, dtype=np.float64)
            for rec in self._records
        ]
        return np.vstack(rows).astype(np.float64)

    @property
    def matrix(self) -> np.ndarray:
        """稠密矩阵（惰性重建：缓存失效时自动 `flush()`）。

        参数：无。
        返回：np.ndarray，形状 (N, dim)。
        副作用/异常：
            缓存失效时会**触发重建并改动** `_matrix` / `_dirty`（读操作带写副作用，
            是为了让上层不必手动 flush）；不抛异常。
        """
        if self._matrix is None or self._dirty:
            self.flush()
        assert self._matrix is not None
        return self._matrix

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def count(self) -> int:
        """记录条数。

        参数：无。
        返回：int，当前集合内的记录数。
        副作用/异常：无。
        """
        return len(self._records)

    def __len__(self) -> int:
        """让集合支持 `len(collection)`。

        参数：无。
        返回：int，同 `count()`。
        副作用/异常：无。
        """
        return len(self._records)

    def records(self) -> List[VectorRecord]:
        """取出全部记录。

        参数：无。
        返回：`List[VectorRecord]`——**浅拷贝的新 list**，增删元素不影响集合，
              但元素对象本身是同一引用（改元素属性会影响到集合内的记录）。
        副作用/异常：无。
        """
        return list(self._records)

    def get(self, record_id: str) -> Optional[VectorRecord]:
        """按 ID 取单条记录。

        参数：record_id 记录 ID。
        返回：`VectorRecord`；不存在时返回 None（不抛异常）。
        副作用/异常：无。
        """
        idx = self._index.get(record_id)
        return self._records[idx] if idx is not None else None

    def query(self, expr: Optional[str] = None, limit: int = 100) -> List[Dict[str, object]]:
        """标量查询：只按元数据取记录（不走向量检索）。

        参数：
            expr  过滤表达式；None/空串由 `MetadataFilter` 视为无过滤，即返回全部
            limit 最多返回多少条，默认 100；达到上限立即停止扫描（**先到先得，非相似度排序**）
        返回：
            List[Dict[str, object]]：每条为 `{"record_id": ..., **meta}` 的字典。
            无过滤且记录数超过 limit 时只返回前 limit 条——需要全量请自行调大 limit。
        副作用/异常：
            无副作用；表达式非法时抛 `FilterError`。
        """
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
        """算出通过标量过滤的候选记录下标（内部方法，**召回前过滤**的落点）。

        参数：expr 过滤表达式；None/空串表示无过滤。
        返回：List[int]，候选在 `_records` 中的下标，保持原始顺序。
        副作用/异常：
            无副作用；表达式非法时抛 `FilterError`（不会退化成「全量放行」）。
        设计要点：无过滤时返回全部下标列表，让调用方走同一条代码路径，
                 避免出现「有过滤/无过滤」两套分支逻辑而不一致。
        """
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
        """稠密向量检索（返回 (record_id, 原始度量分)）。

        参数：
            vector  查询向量（一维）；空数组直接返回空结果
            top_k   最多返回条数；实际实现取 `max(1, top_k)`，传 0 也会返回 1 条
            expr    标量过滤表达式（**召回前过滤**：先算候选下标，再只在候选上打分）
        返回：
            List[SearchHit]：按 score 降序；`score` 与 `dense_score` 同为原始度量分，
            `sparse_score` 保持默认 0.0。空集合/无候选/空查询向量时返回 `[]`。
        副作用/异常：
            只读（可能间接触发矩阵缓存重建）；表达式非法时抛 `FilterError`。
            度量口径见 `metric`：COSINE 走 `cosine_scores()`、IP 用矩阵点积、
            L2 转成负距离，三者统一成「越大越相似」。
        """
        if not self._records or vector.size == 0:
            return []
        candidates = self._candidate_indexes(expr)
        if not candidates:
            return []

        if self.metric == "COSINE":
            raw = cosine_scores(vector, self.matrix)
            # 只在候选下标上取分：被过滤掉的记录即使分数很高也不会进入排序
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
                # 稠密路只填 dense_score；sparse_score 留给稀疏路，融合层据此区分两路贡献
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
        """稀疏词权重检索（归一化后的点积）。

        参数：
            sparse  查询侧稀疏权重（通常是 `sparse_from_text(query)` 的结果）
            top_k   最多返回条数；实际实现取 `max(1, top_k)`
            expr    标量过滤表达式（同样在打分前收窄候选）
        返回：
            List[SearchHit]：按 score 降序，`score` 与 `sparse_score` 相同、
            `dense_score` 保持默认 0.0；**只保留点积 > 0 的记录**
            （权重恒正，故等价于"至少共享一个 token"）。
        副作用/异常：
            只读；表达式非法时抛 `FilterError`。
            归一化说明：只除以查询侧范数 `sparse_norm(sparse)`，未除以文档侧范数，
            因此分数并非严格余弦——口径由上层融合统一处理。
        """
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
    """极简 Milvus 客户端：管理多个 Collection。

    只做集合的注册表（名字 -> Collection），不持有向量数据本身；
    如此上层代码的写法（`client.create_collection(...).insert(...)`）与真实 Milvus 基本一致，
    换实现时改动面被限制在本模块内。

    关键属性：
        _collections  name -> Collection 的字典，按插入顺序保存（`list_collections()` 会排序输出）
    """

    def __init__(self) -> None:
        """初始化空客户端（不含任何集合）。

        参数：无。
        返回：无（构造函数）。
        副作用/异常：无。
        """
        self._collections: Dict[str, Collection] = {}

    def has_collection(self, name: str) -> bool:
        """集合是否存在。

        参数：name 集合名。
        返回：bool。
        副作用/异常：无。
        """
        return name in self._collections

    def create_collection(self, name: str, dim: int, metric: str = "COSINE") -> Collection:
        """创建集合（已存在且维度相同则直接复用，幂等）。

        参数：
            name    集合名
            dim     向量维度
            metric  度量方式，默认 "COSINE"（仅在**新建**时生效）
        返回：
            Collection：新建的集合；若同名集合已存在则返回**已存在的那个实例**
            （此时忽略本次传入的 `metric`，避免语义漂移）。
        副作用/异常：
            新建时写 `_collections`。同名但维度不同时抛 `ValueError`——
            静默复用会把后续插入变成维度不符的错误，越早暴露越好。
        """
        if name in self._collections:
            existing = self._collections[name]
            if existing.dim != int(dim):
                raise ValueError(f"集合 {name} 已存在且维度不同（{existing.dim} != {dim}）")
            return existing
        collection = Collection(name=name, dim=dim, metric=metric)
        self._collections[name] = collection
        return collection

    def get_collection(self, name: str) -> Collection:
        """按名取集合。

        参数：name 集合名。
        返回：Collection。
        副作用/异常：无副作用；集合不存在时抛 `KeyError`（消息提示先 create_collection）。
        """
        if name not in self._collections:
            raise KeyError(f"集合不存在：{name}（请先 create_collection）")
        return self._collections[name]

    def list_collections(self) -> List[str]:
        """列出全部集合名。

        参数：无。
        返回：List[str]，**按字典序排序**（而非创建顺序），保证输出稳定可断言。
        副作用/异常：无。
        """
        return sorted(self._collections)

    def drop_collection(self, name: str) -> bool:
        """删除集合（连带其中的记录）。

        参数：name 集合名。
        返回：bool，True 表示确实删除了一个集合，False 表示本来就不存在（不抛错）。
        副作用/异常：从 `_collections` 移除条目；不抛异常。
        """
        return self._collections.pop(name, None) is not None

    def describe(self) -> List[Dict[str, object]]:
        """导出所有集合的自述信息，供体检报告 / 接口响应使用。

        参数：无。
        返回：
            List[Dict[str, object]]：每项含 name、dim、metric、count 四个字段，
            并**按集合名排序**，因此同一状态下输出可复现。
        副作用/异常：无（`count()` 为只读）。
        """
        return [
            {"name": name, "dim": col.dim, "metric": col.metric, "count": col.count()}
            for name, col in sorted(self._collections.items())
        ]
