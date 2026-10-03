"""五个 Agent 的实现。"""

from __future__ import annotations

from .analyst import AnalystAgent
from .base import AgentContext, BaseAgent
from .planner import PlannerAgent
from .retriever import RetrieverAgent
from .risk_checker import RiskCheckerAgent
from .writer import WriterAgent

__all__ = [
    "AgentContext",
    "BaseAgent",
    "PlannerAgent",
    "RetrieverAgent",
    "AnalystAgent",
    "RiskCheckerAgent",
    "WriterAgent",
]


def build_agents(ctx: AgentContext) -> dict:
    """按固定顺序实例化五个 Agent（供编排层使用）。"""
    return {
        "planner": PlannerAgent(ctx),
        "retriever": RetrieverAgent(ctx),
        "analyst": AnalystAgent(ctx),
        "risk_checker": RiskCheckerAgent(ctx),
        "writer": WriterAgent(ctx),
    }
