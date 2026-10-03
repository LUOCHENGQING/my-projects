"""五个投顾 Agent 与工具权限体系。

| Agent | 职责 | 是否调用模型 | 是否可否决 |
| --- | --- | --- | --- |
| `ClientProfilingAgent` | 客户画像与硬约束提取 | 是（措辞） | 否 |
| `ProductScreeningAgent` | 硬约束可行域内筛选候选池 | 是（措辞） | 否 |
| `PortfolioOptimizerAgent` | 可行域内多目标权重求解 | 是（措辞） | 否 |
| `SuitabilityOfficerAgent` | 适当性规则复核（硬闸门） | 否（纯规则） | **是** |
| `AdvisorNarrativeAgent` | 建议书 / 反事实 / 压力测试 / 留痕 | 是（措辞） | 否 |
"""

from __future__ import annotations

from .advisor_narrative import AdvisorNarrativeAgent
from .base import BaseAgent, Stopwatch
from .client_profiling import ClientProfilingAgent
from .portfolio_optimizer import PortfolioOptimizerAgent
from .product_screening import ProductScreeningAgent
from .suitability_officer import SuitabilityOfficerAgent
from .tools import (
    ALL_TOOLS,
    TOOL_REGISTRY,
    AgentSpec,
    Tool,
    ToolRegistry,
    registry_tools,
    tool_names,
)

#: 五个 Agent 的构造顺序即流水线顺序
AGENT_ORDER: tuple[str, ...] = (
    "ClientProfilingAgent",
    "ProductScreeningAgent",
    "PortfolioOptimizerAgent",
    "SuitabilityOfficerAgent",
    "AdvisorNarrativeAgent",
)


def build_agents(llm: object, tracer: object | None = None) -> dict[str, BaseAgent]:
    """构建五个 Agent（共享同一 LLM 与 tracer，但各自持有独立能力契约）。"""
    classes = (
        ClientProfilingAgent,
        ProductScreeningAgent,
        PortfolioOptimizerAgent,
        SuitabilityOfficerAgent,
        AdvisorNarrativeAgent,
    )
    agents: dict[str, BaseAgent] = {}
    for cls in classes:
        agent = cls(llm=llm, tracer=tracer)
        agents[agent.name] = agent
    return agents


def agent_catalog() -> list[dict[str, object]]:
    """Agent 能力清单（demo / README / 评估报告使用）。"""
    classes = (
        ClientProfilingAgent,
        ProductScreeningAgent,
        PortfolioOptimizerAgent,
        SuitabilityOfficerAgent,
        AdvisorNarrativeAgent,
    )
    return [cls().describe() for cls in classes]


__all__ = [
    "AGENT_ORDER",
    "ALL_TOOLS",
    "AdvisorNarrativeAgent",
    "AgentSpec",
    "BaseAgent",
    "ClientProfilingAgent",
    "PortfolioOptimizerAgent",
    "ProductScreeningAgent",
    "Stopwatch",
    "SuitabilityOfficerAgent",
    "TOOL_REGISTRY",
    "Tool",
    "ToolRegistry",
    "agent_catalog",
    "build_agents",
    "registry_tools",
    "tool_names",
]
