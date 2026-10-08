"""金融智研引擎（FinRAG-Engine）源码包。

本包在全链路中的位置
--------------------
    磁盘资料（data/）
      → ingest（解析 / 脱敏 / 清洗 / 质量评分）
      → chunking（父块 / 子块切分 + 多维元数据）
      → index（BM25 倒排 / 稠密 + 稀疏向量 / Milvus 语义的内存向量库）
      → retrieve（三路召回 → 去重 → 重排 → 父块回溯）
      → answer（引用分配 + 作答 + 忠实度校验）
      → cache / tracing（结果复用与逐步可观测）

本文件只声明包版本与包级公开面，**不再导出任何子模块名字**：
    输入：无（import 本包不触发任何 IO，也不拉起 numpy / FastAPI / Redis）；
    输出：`__version__` 与 `__all__`；
    被谁调用：`python -m src.demo` / `python -m src.serve` / `eval/run_eval.py` /
              `tests/`，它们各自 `from src.xxx import yyy` 显式取用具体模块。
    注：包级 __init__ 刻意不做隐式大导入，避免「只想 import src」顺带付出建索引、
        连 Redis、导入第三方 SDK 的代价与失败面。

对外关键类 / 函数（定义在各自子模块，此处仅索引，便于按图索骥）
------------------------------------------------------------
    src.engine.RAGEngine / AnswerResult        问答主入口与一次问答的完整结果
    src.engine.unknown_entities / ENTITY_RE    主体闸门（防「张冠李戴」）
    src.api.create_app / handle_*              两种服务形态共用的纯函数 handler
    src.api.FEEDBACK_LOG                       进程内人工反馈缓冲
    src.serve.run / make_handler / main        stdlib 兜底 HTTP 服务
    src.config.runtime_config / RuntimeConfig  运行期有效配置快照
    src.tracing.TraceRecorder / read_trace     逐步轨迹的写入与回放

模块划分：
    engine      主链路编排（解析→切分→索引→召回→重排→作答→缓存）
    config      全局配置（路径、切分参数、三路召回权重、缓存、LLM 环境变量）
    ingest      多源文档解析、清洗、脱敏、字段映射、OCR 适配
    chunking    按文档结构的父子块切分与多维元数据
    index       BM25 倒排索引 / 稠密+稀疏向量 / Milvus 风格向量库（内存实现）
    retrieve    三路召回 + RRF 融合 + 交叉编码器重排 + 去重 + 父块回溯
    answer      引用分配与溯源校验 + LLM 客户端 + 问答编排主入口
    cache       Redis 兼容的缓存层（无 Redis 自动降级为内存 LRU）
    faq         高频问题直出模块
    api         FastAPI 接口定义（纯函数式 handler，便于单测）
    serve       stdlib http.server 兜底服务（没装 FastAPI 也能起服务演示）
    tracing     每一步一行 JSONL 的可观测记录
    demo        命令行演示入口

六处「同接口换实现」的降级开关（全项目只有这六处会换实现，其余逻辑零分支）
------------------------------------------------------------------
    向量库    Milvus ↔ 内存实现          src/index/vector_store.py
    向量模型  BGE-M3 ↔ 确定性哈希        src/index/embedding.py `get_backend()`
    重排      bge-reranker ↔ 本地交叉编码器 src/retrieve/rerank.py `get_reranker()`
    缓存      Redis ↔ 内存 LRU           src/cache/redis_cache.py `build_cache()`
    OCR       PaddleOCR ↔ 旁挂校对文本   src/ingest/ocr.py `get_engine()`
    生成      OpenAI 兼容 ↔ 确定性抽取式  src/answer/llm.py `LLMClient`（配合
                                        `config.use_mock_llm()` 判定是否 mock）

六处的共同约定：换实现只换后端对象，上层（engine / retrieve / answer）拿到的
数据结构与 `describe()` 口径不变，因此「没装某个组件」在本项目里表现为一条
降级说明，而不是一个跑不起来的分支。
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
