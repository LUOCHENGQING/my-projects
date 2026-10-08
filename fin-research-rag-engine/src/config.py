"""全局配置。

设计原则
--------
1. 所有可调参数集中在此，避免魔法数字散落各处；
2. 环境变量优先于默认值，便于在 CI / 不同机器上做无侵入调整；
3. 检索策略（三路召回权重、TopK、融合参数）必须是**显式配置项**而不是写死的常数
   ——「按问题类型配权重」是本项目区别于「一套参数打天下」的关键，
   它必须能被调、能被评测、能被回滚。

在本项目中的位置
----------------
本模块是**唯一的配置真相源**，位于链路最底层：所有子模块（ingest / chunking / index /
retrieve / answer / cache / faq）与编排层 `src/engine.py`、服务层 `src/api.py`、
`src/serve.py`、可观测层 `src/tracing.py` 都从这里取常量或 `RuntimeConfig`。
被谁调用：`src/engine.py`（DATA_DIR / FINAL_TOP_K / RUNS_DIR / RuntimeConfig /
runtime_config）、`src/serve.py`（API_HOST / API_PORT）、`src/tracing.py`（RUNS_DIR）、
`src/faq.py`（FAQ_MATCH_THRESHOLD / FAQ_TOP_K）、`src/retrieve/pipeline.py`
（DEDUP_JACCARD / FINAL_TOP_K / RERANK_WEIGHTS）、`eval/run_eval.py`（EVAL_DIR /
use_mock_llm）。
输入：进程环境变量；输出：模块级常量 + `RuntimeConfig` 配置快照；副作用：无（不建目录、不读资料）。
异常：仅 `int()/float()` 解析失败会抛 ValueError——环境变量写错时**启动即失败**，
      而不是带着半截配置跑出一个看似正常的错答案。

配置项与默认值口径（`os.getenv(名字, 默认值)`，故都可以用环境变量覆盖）
--------------------------------------------------------------------
路径（不做环境变量覆盖，一律由 src/config.py 上溯两级算出，保证跨机器可移植）：
    PROJECT_ROOT            项目根目录；DATA_DIR=data/（资料）、RUNS_DIR=runs/（轨迹）、
                            EVAL_DIR=eval/（用例与报告）
切分（父块进 LLM 上下文，子块进索引）：
    CHILD_CHUNK_CHARS=180   子块目标字数，太小丢上下文、太大降精度
    CHILD_CHUNK_OVERLAP=48  相邻子块重叠字数，防止关键句被切在边界
    PARENT_MAX_CHARS=1400   父块字数上限（LLM 上下文预算）
    TABLE_ROWS_PER_CHUNK=2  表格每块行数；与表头绑定，避免「有行无头」的表块
索引：
    EMBED_DIM=512           稠密向量维度，换后端必须对齐
    EMBED_BACKEND=local     local（确定性哈希，零下载）| bge-m3（不可用自动回落 local）
    EMBED_MODEL=BAAI/bge-m3 真实模型名，仅 bge-m3 后端使用（降级时 describe() 会加 fallback 标注）
    BM25_K1=1.5             Okapi BM25 词频饱和参数
    BM25_B=0.75             Okapi BM25 文档长度归一化参数（0=不归一，1=完全归一）
检索（三路召回 + 融合 + 重排）：
    RECALL_TOP_K=20         单路召回候选数；召回放宽、精度交给重排
    FINAL_TOP_K=5           重排后交给 LLM 的证据条数
    RRF_K=60                RRF 平滑常数 k，只吃名次不吃分数，规避量纲不一致
    DEDUP_JACCARD=0.82      子块 token Jaccard 去重阈值
    RERANK_W_CROSS=0.55     重排三项权重：交叉编码器 / RRF / 元数据契合度
    RERANK_W_RRF=0.30
    RERANK_W_METADATA=0.15
    QUERY_ROUTE_WEIGHTS     问题类型 → {bm25, dense, sparse} 权重（clause/case/metric/general）
    DEFAULT_ROUTE=general   未识别出问题类型时的兜底路由
FAQ：  FAQ_MATCH_THRESHOLD=0.62（低于此值不直出，老实走 RAG）/ FAQ_TOP_K=3
缓存：  CACHE_TTL_S=600 / CACHE_MAX_ENTRIES=512 / REDIS_URL=（空串即内存 LRU）
服务：  API_HOST=127.0.0.1 / API_PORT=8000
LLM（OpenAI 兼容；无 key 自动降级为确定性抽取式作答）：
    OPENAI_API_KEY=（空 → mock）/ OPENAI_BASE_URL=https://api.openai.com/v1
    MODEL_NAME=gpt-4o-mini / LLM_TEMPERATURE=0.0（金融问答要可复现）/ LLM_TIMEOUT_S=30.0
    另有一个**没有模块级常量**的开关：环境变量 MOCK_LLM（1/true/yes/on/y 为真），
    由 `use_mock_llm()` 解析，用于「有 key 也强制走 mock」的离线评测。

六处「同接口换实现」的降级开关里，本模块掌握其中三处的**选择权**（另外三处在
`src/index/vector_store.py`、`src/ingest/ocr.py` 与 `src/answer/llm.py` 内部判定）：
    EMBED_BACKEND → 向量模型（`src/index/embedding.py:get_backend()`）
    REDIS_URL     → 缓存后端（`src/cache/redis_cache.py:build_cache()`）
    OPENAI_API_KEY / MOCK_LLM → 生成后端（`src/answer/llm.py:LLMClient`）
本模块只描述"希望用哪个实现"，**实际能不能用由各后端自己探测并如实降级**。
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
    """把环境变量解析成布尔值（1/true/yes/on 视为真）。

    参数：name 环境变量名。
    返回：bool——取值为 1/true/yes/on/y（忽略大小写与首尾空白）时为 True，其余（含未设置）为 False。
    副作用/异常：无；空值、拼写错误一律当 False，因此布尔开关必须是「显式打开」语义。
    """
    return os.getenv(name, "").strip().lower() in _TRUTHY


def current_api_key() -> str:
    """实时读取 API Key（而不是导入时的快照），便于测试运行期改环境变量。

    返回：OPENAI_API_KEY 的当前值（去首尾空白），未设置返回空串。
    副作用/异常：无。
    """
    return os.getenv("OPENAI_API_KEY", "").strip()


def use_mock_llm() -> bool:
    """是否需要进入 mock 模式：显式打开 MOCK_LLM，或者根本没有 key。

    返回：bool——MOCK_LLM 为真，或当前取不到 API Key 时为 True。
    副作用/异常：无。
    注：这是「生成」这一处降级开关的判定入口，`LLMClient` 与 `runtime_config()` 都调它。
    """
    if _env_flag("MOCK_LLM"):
        return True
    return not current_api_key()


def current_base_url() -> str:
    """实时读取 OpenAI 兼容接口的 base_url（同样不取导入期快照）。

    返回：OPENAI_BASE_URL 的当前值；为空时回落到模块常量 OPENAI_BASE_URL。
    副作用/异常：无。之所以实时读，是为了让测试能在运行期把请求指向本地假服务。
    """
    return os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL).strip() or OPENAI_BASE_URL


@dataclass
class RuntimeConfig:
    """一次运行期内的有效配置快照，写进轨迹便于复现一次线上问题。

    关键属性（默认值即 config 模块常量，`runtime_config()` 只会改两个字段）：
        embed_backend / embed_model / embed_dim      嵌入后端与维度
        child_chunk_chars / child_chunk_overlap      切分口径（便于对齐轨迹里看到的块长）
        recall_top_k / final_top_k                   召回与证据条数
        rrf_k / dedup_jaccard                        融合与去重参数
        rerank_weights                               重排三项权重（拷贝，不会串改全局常量）
        route_weights                                问题类型 → 三路权重（逐层拷贝）
        cache_backend                                本次实际使用的缓存后端名
        llm_mode                                     "mock" | "openai"

    状态流转：只在构造时确定一次；`to_dict()` 是只读导出，不产生新状态。
    注：`route_weights` 可以自定义，但**没有被 `to_dict()` 导出**（to_dict 只写
        rerank_weights），排查路由问题时需要另看 `config.QUERY_ROUTE_WEIGHTS`。
    """

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
        """导出可 JSON 序列化的配置快照（`RAGEngine.stats()["config"]` 就是它）。

        返回：扁平 dict，键为上面各字段名；含 dict 值（rerank_weights）。
        副作用/异常：无。注意输出**不含** route_weights，字段名与属性名严格一致。
        """
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
    """构造当前进程的有效配置。

    参数：cache_backend 本次实际生效的缓存后端名（由调用方传入，如 `RAGEngine.config()`
          传 `getattr(self.cache, "name", "none")`，缓存关闭时为 "none"）；
          其余字段取模块常量默认值。
    返回：RuntimeConfig 快照，`llm_mode` 由 `use_mock_llm()` 决定（"mock" 或 "openai"）。
    副作用/异常：无（只是读环境变量 + 拷贝常量）。
    """
    return RuntimeConfig(cache_backend=cache_backend, llm_mode="mock" if use_mock_llm() else "openai")
