"""嵌入与稀疏表示：BGE-M3 风格的「稠密 + 稀疏」双表示。

在 RAG 全链路中的位置
---------------------
    chunking 产出子块文本
        -> **本模块：编码成稠密向量与稀疏词权重**（同一份文本、同一个后端）
            -> src/index/vector_store.py（存进内存向量库，同时供稠密与稀疏检索）
                -> src/retrieve/hybrid.py（召回的「二路」「三路」）
                    -> src/retrieve/rerank.py、src/faq.py（也复用同一后端做语义相似度）

输入：文本（单条或批量 `Sequence[str]`；批量是刻意设计，见下）。
输出：`encode()` -> `np.ndarray`（形状 (N, dim)，已 L2 归一化）；
      `encode_sparse()` -> `List[Dict[str, float]]`（token -> 权重）。
调用方：`src/retrieve/hybrid.py`、`src/retrieve/rerank.py`、`src/faq.py`、
`src/engine.py`（经 `get_backend()` 取后端）、`tests/conftest.py`。
副作用/异常：`local` 后端纯计算、无 IO；`bge-m3` 后端首次调用会加载模型（慢、可能联网
下载权重），不可用时由 `get_backend()` 降级为 local，见下。

关键常量来源（`src/config.py`）：`EMBED_BACKEND`（默认 local）、`EMBED_DIM`（默认 512）、
`EMBED_MODEL`（默认 BAAI/bge-m3）。

为什么要同时要稠密和稀疏
------------------------
BGE-M3 的关键设计是**一个模型同时输出三种表示**（稠密向量 / 稀疏词权重 / 多向量）。
本项目取前两种，因为它们正好覆盖两类互补的检索需求：

    稠密向量：把「还能不能买」和「投资者适当性要求」映射到相近位置——管"意思像"
    稀疏权重：把「资管新规」「R2」「第四十二条」「WY2024-01」按词权重保留——管"字面准"

后端可插拔（同接口换实现，上层检索逻辑不用改）
----------------------------------------------
    local   确定性哈希后端（默认 `LocalHashingBackend`）。零依赖、零网络、零成本，
            任何机器上结果完全一致，用于离线演示 / CI / 单元测试。
    bge-m3  真实后端 `BGEM3Backend`。装了 FlagEmbedding 或 sentence-transformers 时启用，
            维度对齐后可以直接替换，上层检索逻辑一行都不用改；
            **两者都没装时不会报错**——`get_backend()` 自动降级回 local，
            并在 `describe()["model"]` 里标注 `(fallback:local)`，保证「没装模型也能跑通全链路」。

接口刻意设计成「批量进、矩阵出」，因为真实部署时 embedding 是网络瓶颈，
必须能一次编码一批；逐条调用在生产里会被放大成几十倍的延迟。
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..config import EMBED_BACKEND, EMBED_DIM, EMBED_MODEL
from ..utils.text import tokenize

__all__ = [
    "EmbeddingBackend",
    "LocalHashingBackend",
    "BGEM3Backend",
    "get_backend",
    "cosine_scores",
    "l2_normalize",
    "sparse_from_text",
    "sparse_dot",
]


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化；零向量保持零向量（不产生 NaN）。

    参数：matrix 二维数组（形状 (N, dim)），每行是一个待归一化的向量。
    返回：
        np.ndarray：与入参同形状的归一化结果（**返回新数组，入参 `matrix` 不被改动**）；
        空数组原样返回。
    副作用/异常：
        无副作用；不抛异常（零向量通过把范数置 1.0 规避除零）。
    """
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # 零范数置 1，使 0/1 = 0：宁可留下零向量，也不要 NaN 污染整批分数
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """一次矩阵运算求出 query 与全部候选的余弦分数。

    参数：
        query_vec 查询向量（一维，长度应为 dim）
        matrix    候选矩阵（二维 (N, dim)，通常直接来自 `Collection.matrix`）
    返回：
        np.ndarray，float64，长度 N，第 i 位是 query 与第 i 行的余弦相似度（[-1, 1]）。
        矩阵为空或查询为空时返回全零数组（长度按 `matrix.shape[0]` 推断，无法推断时为 0）。
    副作用/异常：
        无副作用（`norms` 是副本）；不抛异常。
    """
    if matrix.size == 0 or query_vec.size == 0:
        return np.zeros((matrix.shape[0] if matrix.ndim == 2 else 0,), dtype=np.float64)
    qn = float(np.linalg.norm(query_vec))
    if qn == 0.0:
        # 零查询向量没有方向，余弦无定义，统一返回全零而不是 NaN
        return np.zeros((matrix.shape[0],), dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0.0] = 1.0
    return (matrix @ query_vec) / (norms * qn)


def sparse_from_text(text: str) -> Dict[str, float]:
    """把文本转成稀疏词权重向量：w = 1 + log(tf)。

    子线性词频（而不是原始 tf）是为了抑制高频词支配权重——
    金融条款里「客户」「产品」出现频率极高，用原始词频会让所有条款长得一样。

    参数：text 原始文本（内部按与 BM25 相同的 `tokenize()` 切词，保证两路口径一致）。
    返回：
        Dict[str, float]：token -> 权重，权重恒 > 0（tf >= 1 时 1 + ln(tf) >= 1）。
        空文本或切不出 token 时返回 `{}`。
    副作用/异常：无；不抛异常。
    """
    counts: Dict[str, int] = {}
    for tok in tokenize(text):
        counts[tok] = counts.get(tok, 0) + 1
    # 1 + ln(tf)：tf = 1 时权重恰为 1，长文里重复词增长缓慢，天然抑制高频词
    return {tok: 1.0 + math.log(c) for tok, c in counts.items()}


def sparse_dot(query: Dict[str, float], doc: Dict[str, float]) -> float:
    """稀疏点积。遍历较短的一侧，复杂度 O(min(|q|, |d|))。

    参数：
        query  查询侧稀疏权重（如 `sparse_from_text(question)` 的结果）
        doc    文档侧稀疏权重（如入库记录 `VectorRecord.sparse`）
    返回：
        float：仅对**共同出现的 token** 累加权重乘积；任一侧为空时返回 0.0。
        本函数**不做归一化**，需要余弦口径时由调用方除以两侧范数
        （`vector_store.search_sparse()` 只除以了 `sparse_norm(query)`）。
    副作用/异常：无副作用；不抛异常。
    """
    if not query or not doc:
        return 0.0
    # 交换成「短的在 query 位」：遍历量取两者较小值，长文档不会拖慢查询
    if len(query) > len(doc):
        query, doc = doc, query
    return float(sum(weight * doc.get(term, 0.0) for term, weight in query.items()))


def sparse_norm(vec: Dict[str, float]) -> float:
    """稀疏向量的 L2 范数（内部辅助函数，未列入 `__all__`）。

    参数：vec 稀疏权重字典。
    返回：
        float：sqrt(Σ w²)；空字典时 `or 1.0` 兜底返回 1.0，避免调用方除零。
    副作用/异常：无；不抛异常。
    """
    return math.sqrt(sum(w * w for w in vec.values())) or 1.0


# ---------------------------------------------------------------------------
# 后端接口
# ---------------------------------------------------------------------------
class EmbeddingBackend:
    """嵌入后端接口。稠密与稀疏必须来自同一后端，否则两路召回的语义会打架。"""

    name = "base"

    def __init__(self, dim: int = EMBED_DIM, model: str = EMBED_MODEL) -> None:
        """初始化后端的通用属性。

        参数：
            dim    稠密向量维度，默认 `EMBED_DIM`（默认 512）；
                   所有子类必须保持同一维度口径，否则向量库插入会报维度不符
            model  模型标识（写进 `describe()` 便于排障），默认 `EMBED_MODEL`
        返回：无（构造函数）。
        副作用/异常：无；不加载模型（真正加载发生在子类首次 `encode` / 读取 `available` 时）。
        """
        self.dim = dim
        self.model = model

    @property
    def available(self) -> bool:  # pragma: no cover - 接口默认实现
        """后端是否可用（基类默认 True，可选依赖型后端需覆盖）。

        参数：无。
        返回：bool。`get_backend()` 靠它决定是否降级到 `LocalHashingBackend`。
        副作用/异常：无（子类实现可能触发模型加载，见 `BGEM3Backend`）。
        """
        return True

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - 接口默认实现
        """把一批文本编码成稠密向量矩阵（**抽象方法，子类必须实现**）。

        参数：texts 文本序列，允许为空（子类应返回形状 (0, dim) 的数组）。
        返回：np.ndarray，形状 (len(texts), dim)，float64，通常已 L2 归一化。
        副作用/异常：基类直接 `raise NotImplementedError`；子类可能触发模型加载。
        """
        raise NotImplementedError

    def encode_sparse(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        """默认稀疏实现：基于分词词频。真实 BGE-M3 后端会覆盖它。

        参数：texts 文本序列。
        返回：`List[Dict[str, float]]`，与入参等长且**顺序一一对应**，
              每项是该文本的 token -> 权重。
        副作用/异常：无；不抛异常。
        """
        return [sparse_from_text(t) for t in texts]

    def encode_one(self, text: str) -> np.ndarray:
        """编码单条文本，取批量结果的第 0 行（查询侧便捷方法）。

        参数：text 查询文本。
        返回：np.ndarray，形状 (dim,)。
        副作用/异常：
            无副作用。**注：实际实现为**走批量 `encode([text])`——空输入未做特殊处理，
            由子类决定（local 后端对空串返回全零行，不抛错）。
        """
        return self.encode([text])[0]

    def describe(self) -> Dict[str, object]:
        """导出后端自述信息，用于体检报告与接口响应，让「用的哪个后端」永远可见。

        参数：无。
        返回：`{"backend": name, "model": model, "dim": dim, "available": available}`。
        副作用/异常：无（但读 `available` 在 bge-m3 后端上会触发一次模型加载探测）。
        """
        return {"backend": self.name, "model": self.model, "dim": self.dim, "available": self.available}


class LocalHashingBackend(EmbeddingBackend):
    """确定性哈希后端（hashing trick）。

        1. tokenize 把文本切成 token（中文单字 + 二元组 + 英文数字整词 + 条款号）
        2. 每个 token 经 blake2b 哈希 -> 桶下标 i 与符号位 s（±1）
        3. v[i] += s * (1 + log(tf))
        4. L2 归一化

    性质：跨机器、跨进程完全确定（刻意不用内置 hash()，它带 PYTHONHASHSEED 随机化）；
    共享 token 的文本余弦相似度更高。效果弱于真实语义 embedding，
    因此在三路召回里与 BM25、稀疏权重互补，而不是单独使用。
    """

    name = "local"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """把一批文本编码成确定性哈希向量（hashing trick）。

        参数：texts 文本序列。
        返回：
            np.ndarray，形状 (len(texts), self.dim)，float64，已逐行 L2 归一化。
            空序列返回 (0, dim) 的全零矩阵；序列中的空串/纯空白文本该行为全零。
        副作用/异常：
            无副作用、无网络、无随机性；不抛异常。
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            # 空文本直接留零行：既不报错也不产生 NaN，调用方按分数 0 处理即可
            if not text:
                continue
            counts: Dict[str, int] = {}
            for tok in tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
            for tok, count in counts.items():
                bucket, sign = _hash_token(tok)
                # 取模把 64 位哈希压进 dim 个桶（必然有冲突，但符号位让冲突正负相抵而非一味叠加）
                out[row, bucket % self.dim] += sign * (1.0 + math.log(count))
        return l2_normalize(out)


def _hash_token(token: str) -> tuple[int, float]:
    """把 token 哈希成 (桶下标, 符号)。blake2b 保证跨进程确定性。

    参数：token 单个分词结果。
    返回：
        tuple[int, float]：(64 位无符号整型桶号, ±1.0 符号)。
        桶号由前 8 字节解释，符号由第 9 字节最低位决定。
    副作用/异常：
        无副作用；不抛异常（唯一风险是 token 含无法 UTF-8 编码的代理字符，
        由 Python 编码层抛 UnicodeEncodeError，正常文本不会命中）。
    """
    raw = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
    bucket = int.from_bytes(raw[:8], "big")
    sign = 1.0 if raw[8] & 1 else -1.0
    return bucket, sign


class BGEM3Backend(EmbeddingBackend):
    """真实 BGE-M3 后端（可选依赖）。

    优先用 FlagEmbedding（BGE-M3 官方实现，能同时给出稠密与稀疏权重），
    退而求其次用 sentence-transformers（只有稠密，稀疏回落到词频实现）。
    两者都没装时 `available` 为 False，工厂会自动切回 local 后端，
    因此**引擎代码里不存在「没装模型就跑不了」的分支**。
    """

    name = "bge-m3"

    def __init__(self, dim: int = EMBED_DIM, model: str = EMBED_MODEL) -> None:
        """初始化真实后端（**此时不加载模型**，延迟到首次使用）。

        参数：
            dim    稠密向量维度，默认 `EMBED_DIM`；
                   注：真实模型的实际输出维度由权重决定，本参数主要用于口径声明与零矩阵兜底
            model  模型名，默认 `EMBED_MODEL`（默认 BAAI/bge-m3），传给 FlagEmbedding /
                   sentence-transformers 去加载
        返回：无（构造函数）。
        副作用/异常：无；只设置 `_mode`（探测缓存）与 `_impl`（真正的模型对象，初始为 None）。
        """
        super().__init__(dim=dim, model=model)
        self._mode: Optional[str] = None
        self._impl = None

    def _load(self) -> Optional[str]:
        """探测并加载可用实现，结果**缓存**在 `self._mode` 里（只探测一次）。

        参数：无。
        返回：
            Optional[str]：`"flagembedding"` / `"sentence-transformers"` 表示可用；
            `""`（空串，假值）表示两个依赖都没装；已有缓存时原样返回缓存值。
        副作用/异常：
            会真实 import 并在成功时加载模型（耗时、可能联网下载权重），结果写入
            `self._mode` 与 `self._impl`。**内部吞掉所有异常**（`except Exception`）
            并继续尝试下一个实现，因此本方法自身不抛异常——失败只表现为返回空串。
        """
        if self._mode is not None:
            return self._mode
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore

            self._impl = BGEM3FlagModel(self.model, use_fp16=True)
            self._mode = "flagembedding"
            return self._mode
        except Exception:  # noqa: BLE001
            pass
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._impl = SentenceTransformer(self.model)
            self._mode = "sentence-transformers"
            return self._mode
        except Exception:  # noqa: BLE001
            # 缓存空串而不是 None：None 会被当成「还没探测过」而每次重试导入
            self._mode = ""
            return self._mode

    @property
    def available(self) -> bool:
        """后端是否真的可用（触发一次延迟加载探测）。

        参数：无。
        返回：bool，`_load()` 返回非空字符串时为 True。
        副作用/异常：首次访问会加载模型（可能较慢）；不抛异常（失败被 `_load()` 吞掉）。
        """
        return bool(self._load())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """用真实模型编码稠密向量。

        参数：texts 文本序列（空序列返回 (0, dim) 零矩阵，且**不会**触发模型加载）。
        返回：
            np.ndarray，float64，形状 (len(texts), 实际模型维度)。
            flagembedding 分支已做 L2 归一化；sentence-transformers 分支依赖
            `normalize_embeddings=True`，同样归一化。
        副作用/异常：
            首次调用会加载模型；`batch_size=8` / `max_length=1024` 为固定推理参数。
            后端不可用时抛 `RuntimeError`（刻意不返回零矩阵，由 `get_backend()` 在构造阶段兜住）。
        """
        mode = self._load()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        if mode == "flagembedding":
            # 字典取 dense_vecs，是因为该接口在同一批里还能给出稀疏权重（见 encode_sparse）
            out = self._impl.encode(list(texts), batch_size=8, max_length=1024)["dense_vecs"]
            return l2_normalize(np.asarray(out, dtype=np.float64))
        if mode == "sentence-transformers":
            out = self._impl.encode(list(texts), normalize_embeddings=True)
            return np.asarray(out, dtype=np.float64)
        # 后端不可用：返回零矩阵会让检索静默失效，所以显式抛错，由工厂层兜住
        raise RuntimeError("BGE-M3 后端不可用：请安装 FlagEmbedding 或 sentence-transformers")

    def encode_sparse(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        """编码稀疏词权重；只有 FlagEmbedding 后端能给出真实 lexical_weights。

        参数：texts 文本序列。
        返回：
            `List[Dict[str, float]]`，与入参等长；flagembedding 分支返回模型的
            lexical_weights（键统一 `str()`、值统一 `float()`，便于入库）；
            其余情况回落到基类的词频实现（`super().encode_sparse()`）。
        副作用/异常：
            首次调用会加载模型；不抛异常（后方不可用时自动回落词频实现）。
        """
        mode = self._load()
        if mode == "flagembedding":
            out = self._impl.encode(list(texts), return_sparse=True)["lexical_weights"]
            return [{str(k): float(v) for k, v in item.items()} for item in out]
        return super().encode_sparse(texts)


# 进程级后端缓存：key 为 "name|dim|model"，避免同一进程里反复构造后端（bge-m3 每次构造都要加载模型）
_BACKEND_CACHE: Dict[str, EmbeddingBackend] = {}


def get_backend(name: str = EMBED_BACKEND, dim: int = EMBED_DIM, model: str = EMBED_MODEL) -> EmbeddingBackend:
    """取嵌入后端。bge-m3 不可用时**自动降级**为本地后端，并在 describe() 里如实标注。

    参数：
        name   后端名，默认 `EMBED_BACKEND`（默认 local）。
               取 `bge-m3` / `bge_m3` / `bgem3` / `flagembedding`（不分大小写以外的变体，见下）
               时尝试真实模型；**其余任何取值**都走本地哈希后端。
               注：实际实现为按字符串精确匹配，不做 `lower()` 归一，因此 `"BGE-M3"` 会被当作 local
        dim    向量维度，默认 `EMBED_DIM`
        model  模型名，默认 `EMBED_MODEL`；降级时会被改写成 `"{model}(fallback:local)"` 以便排障
    返回：
        EmbeddingBackend：`LocalHashingBackend` 或可用的 `BGEM3Backend`。
        **同一 (name, dim, model) 组合会命中进程级缓存，返回同一个实例**
        （降级结果同样进缓存，因此「探测一次、全程复用」）。
    副作用/异常：
        可能加载模型（bge-m3 路径）；写 `_BACKEND_CACHE`。不抛异常。
    """
    key = f"{name}|{dim}|{model}"
    cached = _BACKEND_CACHE.get(key)
    if cached is not None:
        return cached

    backend: EmbeddingBackend
    if name in ("bge-m3", "bge_m3", "bgem3", "flagembedding"):
        candidate = BGEM3Backend(dim=dim, model=model)
        backend = candidate if candidate.available else LocalHashingBackend(dim=dim, model=f"{model}(fallback:local)")
    else:
        backend = LocalHashingBackend(dim=dim, model="local-hashing")
    _BACKEND_CACHE[key] = backend
    return backend


def encode_sparse_batch(texts: Iterable[str]) -> List[Dict[str, float]]:
    """便捷函数：用默认后端做稀疏编码。

    参数：texts 文本可迭代对象（内部会转成 list 一次性编码）。
    返回：`List[Dict[str, float]]`，与输入等长、顺序一致。
    副作用/异常：
        无副作用；不抛异常。**注：实际实现为**该函数未列入本模块 `__all__`，
        属于给脚本/实验用的便捷出口，生产链路走 `HybridRetriever` 持有的后端实例。
    """
    return get_backend().encode_sparse(list(texts))
