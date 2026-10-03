"""全局配置。

所有可调参数集中在这里，避免魔法数字散落各处。
环境变量优先于默认值，便于在不同机器 / CI 上做无侵入调整。
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
# RAG
# ---------------------------------------------------------------------------
# 确定性哈希向量的维度。手写哈希 embedding，不依赖任何外部 embedding API。
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "256"))

# 子块（进入索引的最小检索单元）与父块（回填给 LLM 的上下文单元）
CHILD_CHUNK_CHARS: int = int(os.getenv("CHILD_CHUNK_CHARS", "160"))
CHILD_CHUNK_OVERLAP: int = int(os.getenv("CHILD_CHUNK_OVERLAP", "40"))
PARENT_MAX_CHARS: int = int(os.getenv("PARENT_MAX_CHARS", "1600"))

# BM25 超参
BM25_K1: float = 1.5
BM25_B: float = 0.75

# 混合重排权重（关键词 / 向量 / 元数据），三者加权求和后排序，权重可配置
RERANK_WEIGHTS: Dict[str, float] = {
    "keyword": float(os.getenv("RERANK_W_KEYWORD", "0.45")),
    "vector": float(os.getenv("RERANK_W_VECTOR", "0.40")),
    "metadata": float(os.getenv("RERANK_W_METADATA", "0.15")),
}

# ---------------------------------------------------------------------------
# 工具层
# ---------------------------------------------------------------------------
DEFAULT_TOOL_TIMEOUT_S: float = float(os.getenv("TOOL_TIMEOUT_S", "5.0"))
DEFAULT_TOOL_RETRIES: int = int(os.getenv("TOOL_RETRIES", "2"))

# ---------------------------------------------------------------------------
# Agent / 编排
# ---------------------------------------------------------------------------
# RiskChecker 反思循环上限：发现「数据不足 / 结论不支撑」时把任务打回 Analyst，
# 最多重算 MAX_REVISION_ROUNDS 轮，超出则升级为「需人工确认」。
MAX_REVISION_ROUNDS: int = int(os.getenv("MAX_REVISION_ROUNDS", "2"))

# 图最大执行步数，防止条件边出现意外自环导致死循环
GRAPH_RECURSION_LIMIT: int = int(os.getenv("GRAPH_RECURSION_LIMIT", "40"))

# 高风险自动升级为人工确认的等级
HUMAN_REVIEW_LEVELS = ("high",)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini").strip()

_TRUTHY = {"1", "true", "yes", "on", "y"}


def _env_flag(name: str) -> bool:
    """把环境变量解析成布尔值（1/true/yes/on 视为真）。"""
    return os.getenv(name, "").strip().lower() in _TRUTHY


# 强制 mock：即使配置了 key 也走确定性规则大脑，用于离线回归与评测
FORCE_MOCK: bool = _env_flag("MOCK_LLM")

# 编排引擎选择：auto / langgraph / native
GRAPH_ENGINE_PREF: str = os.getenv("GRAPH_ENGINE", "auto").strip().lower() or "auto"


def current_api_key() -> str:
    """实时读取 API Key（而不是导入时的快照）。

    这样测试、Notebook、CI 在运行期改环境变量都能立刻生效。
    """
    return os.getenv("OPENAI_API_KEY", "").strip()


def current_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL).strip() or OPENAI_BASE_URL


def use_mock_llm() -> bool:
    """是否需要进入 mock 模式（显式打开 MOCK_LLM，或没有 key）。"""
    if _env_flag("MOCK_LLM"):
        return True
    return not current_api_key()


@dataclass
class RuntimeConfig:
    """一次运行期内的有效配置快照，写进 trace 便于复现。"""

    model_name: str = MODEL_NAME
    base_url: str = OPENAI_BASE_URL
    mock_llm: bool = True
    max_revision_rounds: int = MAX_REVISION_ROUNDS
    engine: str = "auto"
    embed_dim: int = EMBED_DIM
    rerank_weights: Dict[str, float] = field(default_factory=lambda: dict(RERANK_WEIGHTS))
    child_chunk_chars: int = CHILD_CHUNK_CHARS
    child_chunk_overlap: int = CHILD_CHUNK_OVERLAP

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_name": self.model_name,
            "base_url": self.base_url if not self.mock_llm else "(mock)",
            "mock_llm": self.mock_llm,
            "max_revision_rounds": self.max_revision_rounds,
            "engine": self.engine,
            "embed_dim": self.embed_dim,
            "rerank_weights": self.rerank_weights,
            "child_chunk_chars": self.child_chunk_chars,
            "child_chunk_overlap": self.child_chunk_overlap,
        }


def runtime_config() -> RuntimeConfig:
    """构造当前进程的有效配置。"""
    return RuntimeConfig(mock_llm=use_mock_llm(), engine=GRAPH_ENGINE_PREF)
