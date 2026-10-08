"""五个 Agent 的实现（agents 层）——本包的唯一出口与装配点。

层次与职责：
    本包位于「编排层（src/orchestrator.py）之下、工具层（src/tools/registry.py）与
    LLM 层（src/llm/client.py）之上」，是投研流程中真正干活的执行单元集合。
    五个 Agent 彼此**从不直接调用**，一律通过 blackboard 共享状态（src/state.py 的
    ``ResearchState``）读写协作；谁先谁后、是否回退，完全由编排层的边与路由函数决定。
    因此本模块只做「导出 + 装配」，不放任何业务逻辑，业务逻辑一律沉在各 Agent 的 ``_execute``。

对外关键类 / 函数：
    * ``AgentContext``      —— 一次运行内共享的依赖容器（工具注册中心 / LLM / trace 记录器 / 配置）。
    * ``BaseAgent``         —— 模板方法基类，统一「计时 -> _execute -> 落 trace -> 回填 steps」。
    * ``PlannerAgent``      —— 任务分解与路由（不持工具）。
    * ``RetrieverAgent``    —— 多路检索与证据筛选（只召回，不判断）。
    * ``AnalystAgent``      —— 指标计算与结论生成（数字出自工具，语言出自模型）。
    * ``RiskCheckerAgent``  —— 合规核查门与反思循环驱动者（唯一能打回上游）。
    * ``WriterAgent``       —— 结构化简报撰写与引用编号绑定（独占 cite_source 的 write 权限）。
    * ``build_agents()``    —— 装配工厂，见下。

主要输入输出：
    输入是编排层构造好的 ``AgentContext``；输出是 ``{"planner": ..., "writer": ...}``
    这个「角色名 -> Agent 实例」字典，编排层再逐个注册成图的节点并调用其 ``run()``。

被谁调用：
    * ``src/orchestrator.py`` 的 ``ResearchPipeline._build_spec()`` 调用 ``build_agents(ctx)``；
    * ``tests/test_citation_traceability.py`` 与 ``tests/test_risk_loop.py`` 直接导入具体
      Agent 类与 ``AgentContext``，用于单步构造状态、绕过整张图做单元验证。
"""

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
    """按固定顺序实例化五个 Agent（供编排层使用）。

    参数：
        ctx: 本次运行共享的 ``AgentContext``；五个实例共用同一个 ctx，
            因此它们看到的是同一份工具注册中心、同一个 LLM 客户端、同一个 trace 记录器。
            权限隔离不靠 ctx，而靠各子类自己声明的 ``allowed_tools`` / ``permissions``。

    返回：
        ``dict``，键为角色名（"planner" / "retriever" / "analyst" / "risk_checker" / "writer"），
        值为对应的 Agent 实例；键名与 trace 里的 ``agent`` 字段、以及 state 的字段分组命名一致。

    副作用 / 异常：
        会触发五个 Agent 的 ``__init__``（仅保存 ctx 并初始化内部工具调用日志），
        **不发起任何工具调用或 LLM 请求**，也不读写 ``ResearchState``——
        真正的执行发生在编排层调用各自的 ``run(state)`` 时。
        由于各 Agent 构造器不做 I/O，此处不预期抛异常。

    注：字典字面量的书写顺序即为实例化顺序，但该顺序不表达执行顺序；
        执行顺序由 ``src/orchestrator.py`` 的节点与条件边决定。
    """
    return {
        "planner": PlannerAgent(ctx),
        "retriever": RetrieverAgent(ctx),
        "analyst": AnalystAgent(ctx),
        "risk_checker": RiskCheckerAgent(ctx),
        "writer": WriterAgent(ctx),
    }
