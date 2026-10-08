"""五个投顾 Agent 与工具权限体系。

层级
----
Agent 层（`src/agents/`），位于 `src/pipeline.py` 的编排层之下、`src/constraints.py`
等确定性业务模块之上：每个 Agent 通过 `tools.TOOL_REGISTRY` 里的白名单工具调用
底层确定性能力，自身只负责组织输入、校验输出边界与记录 trace。

解决的问题
----------
把"谁能做什么"变成**可断言的数据**：每个 Agent 的能力由一份 `AgentSpec`
（工具白名单 + 权限标签集合 + 可写 state 键）声明，越权调用或越权写共享状态
由 `BaseAgent` 直接抛 `PermissionError`，而不是靠代码约定。

| Agent | 职责 | 是否调用模型 | 是否可否决 |
| --- | --- | --- | --- |
| `ClientProfilingAgent` | 客户画像与硬约束提取 | 是（措辞） | 否 |
| `ProductScreeningAgent` | 硬约束可行域内筛选候选池 | 是（措辞） | 否 |
| `PortfolioOptimizerAgent` | 可行域内多目标权重求解 | 是（措辞） | 否 |
| `SuitabilityOfficerAgent` | 适当性规则复核（硬闸门） | 否（纯规则） | **是** |
| `AdvisorNarrativeAgent` | 建议书 / 反事实 / 压力测试 / 留痕 | 是（措辞） | 否 |

对外暴露
--------
- `AGENT_ORDER`：五个 Agent 名称及其流水线顺序（构造顺序即执行顺序）。
- `build_agents(llm, tracer)`：按 `AGENT_ORDER` 构造 `{agent_name: BaseAgent}`；
  五个 Agent 共享同一个 LLM 与 tracer 实例。入参 `llm` 通常来自
  `src.llm.build_llm()`，`tracer` 为 `src.observability.Tracer`。
- `agent_catalog()`：无参构造五个 Agent，返回各自 `describe()` 的能力清单
  （`name` / `role` / `tools` / `permissions` / `can_write_state`）。
- 转出（re-export）`Tool` / `ToolRegistry` / `TOOL_REGISTRY` / `ALL_TOOLS` /
  `AgentSpec` / `registry_tools` / `tool_names`，使调用方只需 `from .agents import ...`。

被谁使用
--------
`src/pipeline.py`（构造并驱动 Agent）与 `src/demo.py`（`--catalog` 打印能力清单）。

说明：本模块只做装配与转出，不承载业务逻辑；具体角色提示词与业务实现分别在
`client_profiling.py` / `product_screening.py` / `portfolio_optimizer.py` /
`suitability_officer.py` / `advisor_narrative.py` 中。
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
#: （画像 → 筛选 → 组合 → 适当性闸门 → 建议书；适当性闸门可打回上游重配）
AGENT_ORDER: tuple[str, ...] = (
    "ClientProfilingAgent",
    "ProductScreeningAgent",
    "PortfolioOptimizerAgent",
    "SuitabilityOfficerAgent",
    "AdvisorNarrativeAgent",
)


def build_agents(llm: object, tracer: object | None = None) -> dict[str, BaseAgent]:
    """构建五个 Agent（共享同一 LLM 与 tracer，但各自持有独立能力契约）。

    参数：
        llm：LLM 客户端（`src.llm.BaseLLM` 的任一实现），注入给每个 Agent 的 `llm` 属性；
             Agent 只把它透传给白名单工具，不直接调用。
        tracer：可选 trace 记录器（`src.observability.Tracer`）；为 None 时 Agent 的
                `trace()` 静默跳过，不产生 runs/*.jsonl。

    返回：
        `{agent.name: BaseAgent}`，键顺序与 `AGENT_ORDER` 一致；值为各类 Agent 实例
        （构造顺序即流水线顺序）。

    副作用：仅实例化对象，不读写共享状态、不发起网络调用。
    """
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
    """Agent 能力清单（demo / README / 评估报告使用）。

    无入参；按 `AGENT_ORDER` 的顺序无参构造五个 Agent（不注入 llm / tracer），
    再调用各自的 `describe()`。

    返回：
        长度 5 的列表，每项含 `name` / `role` / `tools` / `permissions` /
        `can_write_state`（`permissions` 已排序，`tools` 保留白名单声明顺序）。

    副作用：无（不修改共享状态、不写 trace）。
    """
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
    # 工具与权限体系（转出自 .tools）
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
