"""金融投研多智能体系统（FinResearch-MAS）源码包。

模块划分：
    config      全局配置（路径、检索权重、循环上限、LLM 环境变量）
    state       Blackboard 共享状态定义
    tools       MCP 风格工具注册中心与内置工具
    rag         父子块切分 / BM25 / 哈希向量 / 混合重排检索
    llm         OpenAI 兼容客户端 + 确定性 mock 大脑
    agents      Planner / Retriever / Analyst / RiskChecker / Writer
    engine_*    图执行引擎（LangGraph 主引擎 + 自研兼容降级引擎）
    orchestrator 编排装配、条件边与反思循环
    tracing     每一步一行 JSONL 的可观测记录
    hitl        人机协同（高风险结论暂停等待人工确认）
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
