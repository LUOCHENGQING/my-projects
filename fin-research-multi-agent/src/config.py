"""全局配置（架构中的「配置层」，被所有其它层单向依赖）。

所有可调参数集中在这里，避免魔法数字散落各处。
环境变量优先于默认值，便于在不同机器 / CI 上做无侵入调整。

层次与职责
----------
本模块不 import 项目内任何其它模块，处于依赖图的最底层：`llm`、`rag`、`agents`、
`orchestrator`、`demo`、`eval` 都从这里取值，因此它必须保持「零副作用、可安全导入」。

对外关键对象
------------
* 常量：`PROJECT_ROOT` / `DATA_DIR` / `RUNS_DIR` / `EVAL_DIR`（路径）、
  `EMBED_DIM` / `CHILD_CHUNK_*` / `PARENT_MAX_CHARS` / `RERANK_WEIGHTS`（RAG）、
  `MAX_REVISION_ROUNDS` / `GRAPH_RECURSION_LIMIT` / `HUMAN_REVIEW_LEVELS`（编排兜底）、
  `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL_NAME` / `FORCE_MOCK` / `GRAPH_ENGINE_PREF`（LLM 与引擎）。
* 函数：`current_api_key()` / `current_base_url()`（运行期实时读环境变量）、
  `use_mock_llm()`（是否进入 mock 模式）、`runtime_config()`（生成一次运行的有效配置快照）。
* 数据类 `RuntimeConfig`：写进 trace，用于事后复现「这次跑的是什么配置」。

主要输入输出
------------
输入：进程环境变量（`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`MODEL_NAME`、`MOCK_LLM`、
`GRAPH_ENGINE`、`EMBED_DIM`、`CHILD_CHUNK_CHARS`、`CHILD_CHUNK_OVERLAP`、
`PARENT_MAX_CHARS`、`RERANK_W_*`、`TOOL_TIMEOUT_S`、`TOOL_RETRIES`、
`MAX_REVISION_ROUNDS`、`GRAPH_RECURSION_LIMIT`）。
输出：标量常量与 `RuntimeConfig` 实例。

被谁调用
--------
`src/llm/client.py`、`src/orchestrator.py`、`src/demo.py`、`eval/run_eval.py`、
`tests/conftest.py` 等。

副作用与注意事项
----------------
模块级常量在**首次导入时**读取环境变量并定格；若要在运行期改变生效值，必须调用
`current_api_key()` / `current_base_url()` / `use_mock_llm()`（它们每次都重新读），
而不是依赖导入时的快照。本模块不联网、不写文件。
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
# 用 resolve() 拿到绝对路径，避免以不同 cwd 启动时把 runs/ 写得到处都是
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RUNS_DIR: Path = PROJECT_ROOT / "runs"
EVAL_DIR: Path = PROJECT_ROOT / "eval"

# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
# 确定性哈希向量的维度。手写哈希 embedding，不依赖任何外部 embedding API。
# 维度越大区分度越高但内存/耗时线性上升；离线复现要求它固定，故允许用环境变量覆盖默认值。
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "256"))

# 子块（进入索引的最小检索单元）与父块（回填给 LLM 的上下文单元）
# 三者共同决定「检索精度 vs 上下文完整度」的权衡：小块好命中，父块给模型完整语境
CHILD_CHUNK_CHARS: int = int(os.getenv("CHILD_CHUNK_CHARS", "160"))
CHILD_CHUNK_OVERLAP: int = int(os.getenv("CHILD_CHUNK_OVERLAP", "40"))
PARENT_MAX_CHARS: int = int(os.getenv("PARENT_MAX_CHARS", "1600"))

# BM25 超参
# 这两个是 BM25 的标准经验值，硬编码为常量（不做环境变量覆盖）以保证检索结果可复现
BM25_K1: float = 1.5
BM25_B: float = 0.75

# 混合重排权重（关键词 / 向量 / 元数据），三者加权求和后排序，权重可配置
# 注：此处只声明权重，实际归一化与打分在 src/rag/retriever.py 中完成
RERANK_WEIGHTS: Dict[str, float] = {
    "keyword": float(os.getenv("RERANK_W_KEYWORD", "0.45")),
    "vector": float(os.getenv("RERANK_W_VECTOR", "0.40")),
    "metadata": float(os.getenv("RERANK_W_METADATA", "0.15")),
}

# ---------------------------------------------------------------------------
# 工具层
# ---------------------------------------------------------------------------
# 工具调用的默认超时与重试：工具出问题要快速失败并被上层记入 errors，而不是拖死整张图
DEFAULT_TOOL_TIMEOUT_S: float = float(os.getenv("TOOL_TIMEOUT_S", "5.0"))
DEFAULT_TOOL_RETRIES: int = int(os.getenv("TOOL_RETRIES", "2"))

# ---------------------------------------------------------------------------
# Agent / 编排
# ---------------------------------------------------------------------------
# RiskChecker 反思循环上限：发现「数据不足 / 结论不支撑」时把任务打回 Analyst，
# 最多重算 MAX_REVISION_ROUNDS 轮，超出则升级为「需人工确认」。
# 这个上限同时被 mock 大脑（_task_risk_review 的 gate.max_rounds）与真机裁决规则复用
MAX_REVISION_ROUNDS: int = int(os.getenv("MAX_REVISION_ROUNDS", "2"))

# 图最大执行步数，防止条件边出现意外自环导致死循环
GRAPH_RECURSION_LIMIT: int = int(os.getenv("GRAPH_RECURSION_LIMIT", "40"))

# 高风险自动升级为人工确认的等级
# 只有 "high" 会强制转人工；"info"/"low"/"medium" 直接放行（闸门不委托给模型）
HUMAN_REVIEW_LEVELS = ("high",)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# 这三个是「导入时快照」：用于打印 banner、构造 RuntimeConfig 默认值。
# 真正发起请求前的实时值请走 current_api_key() / current_base_url()。
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini").strip()

# 被视为「真」的取值集合；统一转小写后比较，容忍 "TRUE" / "Yes " 这类写法
_TRUTHY = {"1", "true", "yes", "on", "y"}


def _env_flag(name: str) -> bool:
    """把环境变量解析成布尔值（1/true/yes/on 视为真）。

    参数：
        name: 环境变量名，例如 "MOCK_LLM"。
    返回：
        bool —— 变量存在且去掉空白、转小写后落在 _TRUTHY 里则为 True，
        未设置、空串或其它取值一律为 False（不存在即视为假，不做报错）。
    副作用/异常：
        只读环境变量；不抛异常（包含非法布尔值也只是当成 False）。
    """
    return os.getenv(name, "").strip().lower() in _TRUTHY


# 强制 mock：即使配置了 key 也走确定性规则大脑，用于离线回归与评测
FORCE_MOCK: bool = _env_flag("MOCK_LLM")

# 编排引擎选择：auto / langgraph / native
# auto = 优先 LangGraph，导入失败或运行出错时自动降级到自研引擎（保证离线可跑）
GRAPH_ENGINE_PREF: str = os.getenv("GRAPH_ENGINE", "auto").strip().lower() or "auto"


def current_api_key() -> str:
    """实时读取 API Key（而不是导入时的快照）。

    这样测试、Notebook、CI 在运行期改环境变量都能立刻生效。

    参数：无。
    返回：
        str —— 当前 `OPENAI_API_KEY` 去空白后的值；未设置时返回空串 ""。
    副作用/异常：
        只读环境变量，不联网、不校验密钥格式，也不会抛异常。
        注意它**不读**模块级常量 OPENAI_API_KEY，二者可能不一致。
    """
    return os.getenv("OPENAI_API_KEY", "").strip()


def current_base_url() -> str:
    """实时读取 OpenAI 兼容服务的 base URL。

    参数：无。
    返回：
        str —— 当前 `OPENAI_BASE_URL` 去空白后的值；若该变量未设置或只有空白，
        则回退到导入时快照 OPENAI_BASE_URL（其默认值为 "https://api.openai.com/v1"）。
    副作用/异常：
        只读环境变量；不联网、不抛异常。
    """
    return os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL).strip() or OPENAI_BASE_URL


def use_mock_llm() -> bool:
    """是否需要进入 mock 模式（显式打开 MOCK_LLM，或没有 key）。

    参数：无。
    返回：
        bool —— 满足下列任一条件即为 True：
        1) 环境变量 `MOCK_LLM` 属于真值（1/true/yes/on/y，大小写不敏感）；
        2) 实时读到的 `OPENAI_API_KEY` 为空。
        否则为 False（即走真实 OpenAI 兼容接口）。
    副作用/异常：
        只读环境变量，无网络访问；这是全系统「无 key 也能离线跑」的判定入口。
    """
    if _env_flag("MOCK_LLM"):
        return True
    return not current_api_key()


@dataclass
class RuntimeConfig:
    """一次运行期内的有效配置快照，写进 trace 便于复现。

    职责：
        把散落的环境变量/常量收敛成一个可序列化对象，随 trace 落盘，
        让「同一条问题、同一份配置」能被回放比对。

    关键属性（字段名与 trace 中的键一致）：
        model_name:           真实模式请求的模型名（mock 模式下由 client 前缀 `mock::` 标注）
        base_url:             真实模式的 OpenAI 兼容网关地址
        mock_llm:             是否走确定性 mock 大脑；**默认 True** 是保守选择——
                              未显式指定时绝不误连外部服务
        max_revision_rounds:  RiskChecker 把结论打回 Analyst 重算的轮次上限
        engine:               期望使用的编排引擎（"auto"/"langgraph"/"native"）
        embed_dim:            哈希向量维度
        rerank_weights:       关键词/向量/元数据三路混合重排权重
        child_chunk_chars:    子块目标字符数
        child_chunk_overlap:  相邻子块重叠字符数

    状态流转：
        构造后即视为只读快照；`LLMClient.__init__` 会按自身实际模式回写
        `mock_llm` 字段（见 src/llm/client.py），除此之外不应被修改。
    """

    model_name: str = MODEL_NAME
    base_url: str = OPENAI_BASE_URL
    mock_llm: bool = True
    max_revision_rounds: int = MAX_REVISION_ROUNDS
    engine: str = "auto"
    embed_dim: int = EMBED_DIM
    # 用 lambda 深拷一份，避免调用方原地修改 RERANK_WEIGHTS 影响其它实例
    rerank_weights: Dict[str, float] = field(default_factory=lambda: dict(RERANK_WEIGHTS))
    child_chunk_chars: int = CHILD_CHUNK_CHARS
    child_chunk_overlap: int = CHILD_CHUNK_OVERLAP

    def to_dict(self) -> Dict[str, object]:
        """转成可 JSON 序列化的 dict（供 trace / 评测报告落盘）。

        参数：无。
        返回：
            Dict[str, object] —— 与字段同名的一组键值；其中 base_url 在 mock 模式下
            被替换为字符串 "(mock)"，避免把真实网关地址写进可分享的 trace。
        副作用/异常：
            纯函数，不改动实例自身；返回的 rerank_weights 是原字典的引用
            （调用方如需修改请自行拷贝）。
        """
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
    """构造当前进程的有效配置。

    参数：无。
    返回：
        RuntimeConfig —— `mock_llm` 取 `use_mock_llm()` 的实时判定结果（因此无 key
        时自动为 True），`engine` 取环境变量 GRAPH_ENGINE_PREF；其余字段沿用模块默认值。
    副作用/异常：
        只读环境变量；每次调用都新建实例，不做缓存（便于测试在运行期切换环境变量
        后立刻生效）。
    """
    return RuntimeConfig(mock_llm=use_mock_llm(), engine=GRAPH_ENGINE_PREF)
