"""金融智研引擎（FinRAG-Engine）源码包。

模块划分：
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
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
