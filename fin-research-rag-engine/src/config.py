"""全局配置。

设计原则
--------
1. 所有可调参数集中在此，避免魔法数字散落各处；
2. 环境变量优先于默认值，便于在 CI / 不同机器上做无侵入调整；
3. 检索策略（三路召回权重、TopK、融合参数）必须是**显式配置项**而不是写死的常数
   ——「按问题类型配权重」是本项目区别于「一套参数打天下」的关键，
   它必须能被调、能被评测、能被回滚。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
# src/config.py -> src/ -> 项目根目录
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RUNS_DIR: Path = PROJECT_ROOT / "runs"
EVAL_DIR: Path = PROJECT_ROOT / "eval"

# ---------------------------------------------------------------------------
# 切分（分块）参数
# ---------------------------------------------------------------------------
# 父块 = 一个语义完整的章节（进入 LLM 上下文窗口的单元）
# 子块 = 进入索引的最小检索单元
CHILD_CHUNK_CHARS: int = int(os.getenv("CHILD_CHUNK_CHARS", "180"))
CHILD_CHUNK_OVERLAP: int = int(os.getenv("CHILD_CHUNK_OVERLAP", "48"))
PARENT_MAX_CHARS: int = int(os.getenv("PARENT_MAX_CHARS", "1400"))
# 表格行块：一张宽表怎么切才不会把「表头」和「数据行」拆散
TABLE_ROWS_PER_CHUNK: int = int(os.getenv("TABLE_ROWS_PER_CHUNK", "2"))

# ---------------------------------------------------------------------------
# 索引参数
# ---------------------------------------------------------------------------
# 稠密向量维度。本地后端用确定性哈希向量（无外部 embedding 服务也能跑通），
# 真实部署时把 backend 换成 BGE-M3 即可，维度对齐即可无缝切换。
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "512"))
EMBED_BACKEND: str = os.getenv("EMBED_BACKEND", "local").strip().lower() or "local"
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3").strip()

# BM25 超参
BM25_K1: float = float(os.getenv("BM25_K1", "1.5"))
BM25_B: float = float(os.getenv("BM25_B", "0.75"))

# ---------------------------------------------------------------------------
# 检索参数：三路召回 + 融合 + 重排
# ---------------------------------------------------------------------------
# 每一路的候选数（召回看「不漏」，所以候选放宽；精度交给重排）
RECALL_TOP_K: int = int(os.getenv("RECALL_TOP_K", "20"))
# 重排后交给 LLM 的证据条数
FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "5"))
# RRF（Reciprocal Rank Fusion）的平滑常数 k。
# RRF 只依赖「排名」不依赖「分数」，因此天然规避了 BM25 与余弦相似度量纲不一致的问题。
RRF_K: int = int(os.getenv("RRF_K", "60"))
# 去重阈值：两个子块的 token Jaccard 超过它即视为重复，只保留分数更高的那条
DEDUP_JACCARD: float = float(os.getenv("DEDUP_JACCARD", "0.82"))
# 重排融合权重：交叉编码器得分 / RRF 得分 / 元数据契合度
RERANK_WEIGHTS: Dict[str, float] = {
    "cross": float(os.getenv("RERANK_W_CROSS", "0.55")),
    "rrf": float(os.getenv("RERANK_W_RRF", "0.30")),
    "metadata": float(os.getenv("RERANK_W_METADATA", "0.15")),
}

# 问题类型路由：不同问题类型走不同的召回权重，这是本项目的核心策略之一。
# 键为问题类型，值为 {bm25, dense, sparse} 三路权重。
#   - 条款类（"第几条""是否符合""准入要求"）：条款号/产品代码/proper noun 多，BM25 权重最高
#   - 案例类（"有没有类似案例""处罚"）：语义相似更重要，稠密权重最高
#   - 指标类（"集中度怎么算""比例是多少"）：需要字面 + 语义兼顾
#   - 通用：三路均衡
QUERY_ROUTE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "clause": {"bm25": 0.55, "dense": 0.20, "sparse": 0.25},
    "case": {"bm25": 0.20, "dense": 0.55, "sparse": 0.25},
    "metric": {"bm25": 0.35, "dense": 0.30, "sparse": 0.35},
    "general": {"bm25": 0.34, "dense": 0.33, "sparse": 0.33},
}
DEFAULT_ROUTE: str = os.getenv("DEFAULT_ROUTE", "general").strip() or "general"

# ---------------------------------------------------------------------------
# FAQ
# ---------------------------------------------------------------------------
# FAQ 近似匹配阈值：命中直接返回，不再走完整链路
FAQ_MATCH_THRESHOLD: float = float(os.getenv("FAQ_MATCH_THRESHOLD", "0.62"))
FAQ_TOP_K: int = int(os.getenv("FAQ_TOP_K", "3"))

# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------
CACHE_TTL_S: int = int(os.getenv("CACHE_TTL_S", "600"))
CACHE_MAX_ENTRIES: int = int(os.getenv("CACHE_MAX_ENTRIES", "512"))
REDIS_URL: str = os.getenv("REDIS_URL", "").strip()

# ---------------------------------------------------------------------------
# 服务
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "127.0.0.1").strip()
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ---------------------------------------------------------------------------
# LLM（OpenAI 兼容；无 key 自动降级为确定性抽取式作答）
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini").strip()
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_TIMEOUT_S: float = float(os.getenv("LLM_TIMEOUT_S", "30.0"))

_TRUTHY = {"1", "true", "yes", "on", "y"}


def _env_flag(name: str) -> bool:
    """把环境变量解析成布尔值（1/true/yes/on 视为真）。"""
    return os.getenv(name, "").strip().lower() in _TRUTHY


def current_api_key() -> str:
    """实时读取 API Key（而不是导入时的快照），便于测试运行期改环境变量。"""
    return os.getenv("OPENAI_API_KEY", "").strip()


def use_mock_llm() -> bool:
    """是否需要进入 mock 模式：显式打开 MOCK_LLM，或者根本没有 key。"""
    if _env_flag("MOCK_LLM"):
        return True
    return not current_api_key()


def current_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL).strip() or OPENAI_BASE_URL


@dataclass
class RuntimeConfig:
    """一次运行期内的有效配置快照，写进轨迹便于复现一次线上问题。"""

    embed_backend: str = EMBED_BACKEND
    embed_model: str = EMBED_MODEL
    embed_dim: int = EMBED_DIM
    child_chunk_chars: int = CHILD_CHUNK_CHARS
    child_chunk_overlap: int = CHILD_CHUNK_OVERLAP
    recall_top_k: int = RECALL_TOP_K
    final_top_k: int = FINAL_TOP_K
    rrf_k: int = RRF_K
    dedup_jaccard: float = DEDUP_JACCARD
    rerank_weights: Dict[str, float] = field(default_factory=lambda: dict(RERANK_WEIGHTS))
    route_weights: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in QUERY_ROUTE_WEIGHTS.items()}
    )
    cache_backend: str = "memory"
    llm_mode: str = "mock"

    def to_dict(self) -> Dict[str, object]:
        return {
            "embed_backend": self.embed_backend,
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
            "child_chunk_chars": self.child_chunk_chars,
            "child_chunk_overlap": self.child_chunk_overlap,
            "recall_top_k": self.recall_top_k,
            "final_top_k": self.final_top_k,
            "rrf_k": self.rrf_k,
            "dedup_jaccard": self.dedup_jaccard,
            "rerank_weights": self.rerank_weights,
            "cache_backend": self.cache_backend,
            "llm_mode": self.llm_mode,
        }


def runtime_config(cache_backend: str = "memory") -> RuntimeConfig:
    """构造当前进程的有效配置。"""
    return RuntimeConfig(cache_backend=cache_backend, llm_mode="mock" if use_mock_llm() else "openai")
