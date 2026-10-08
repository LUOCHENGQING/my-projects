"""Agent 基类：工具白名单、权限集合、越权拦截与逐步 trace。

所属层次
--------
Agent 层（`src/agents/`）的最底层基类，被 5 个投顾 Agent 继承；
编排层（`src/pipeline.py`）通过各 Agent 的 `run()` 调用本层能力。

解决什么问题
------------
把「谁能做什么」从散落的约定变成**可断言的数据**：每个 Agent 只拿到自己的
`AgentSpec`（工具白名单 + 权限集合 + 可写状态键），越权在基类统一拦截，
业务子类只写业务逻辑，不需要重复写权限判断。

三个约束在基类里统一落地，子类只写业务：
1. `call_tool` 双重校验（工具白名单 + 权限标签），越权直接 `PermissionError`；
2. `guard_output` 限制每个 Agent 只能写自己负责的共享状态键；
3. `trace` 统一记录 `step / agent / input_digest / output_digest / latency_ms / status / tool_calls`。

对外暴露
--------
- `BaseAgent`：所有投顾 Agent 的基类（`call_tool` / `guard_output` / `trace` / `run`）
- `Stopwatch`：毫秒级轻量计时器（各 Agent 统计 `latency_ms` 用）

被谁调用
--------
5 个 Agent 模块（`client_profiling` / `product_screening` / `portfolio_optimizer` /
`suitability_officer` / `advisor_narrative`）；`src/agents/__init__.py` 再导出；
`src/pipeline.py` 与 `eval/run_eval.py` 通过 `build_agents` 间接使用。
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Mapping

from .tools import TOOL_REGISTRY, AgentSpec, ToolRegistry


class BaseAgent:
    """所有投顾 Agent 的基类。

    本基类只提供**能力边界与留痕**，不提供任何投顾业务逻辑，也**不调用模型**
    （模型只出现在子类里，且仅用于组织措辞）。

    五个子类共享同一套契约，但各自的能力集合彼此**互斥**，原因是流水线要求
    「一个状态键只有一个责任方」，否则出现问题时无法定位是谁写坏的：

    ========================== ==========================================================
    Agent                      `can_write_state`（可写状态键，五者两两不相交）
    ========================== ==========================================================
    `ClientProfilingAgent`     `client` / `effective_client` / `profile`
    `ProductScreeningAgent`    `screening` / `candidates` / `screening_note`
    `PortfolioOptimizerAgent`  `portfolio` / `portfolio_note` / `binding` / `product_scores`
    `SuitabilityOfficerAgent`  `gate` / `suitability_comment` / `tighten`
    `AdvisorNarrativeAgent`    `narrative` / `elements` / `advice` / `stress` / `counterfactual`
    ========================== ==========================================================

    互斥的具体含义与动机：
    - **写入互斥**：`guard_output` 按 `can_write_state` 白名单校验，任一 Agent
      尝试写别人的键都会立刻 `PermissionError`（例如筛选 Agent 无法直接改
      `portfolio`，构建 Agent 无法直接改 `gate`），从而保证「谁产出谁负责」；
    - **职责互斥**：画像 Agent 不产出投资观点、筛选 Agent 不做收益预测、
      构建 Agent 不做合规判断、复核 Agent 不改权重、撰写 Agent 不自行计算数字；
    - **否决权互斥**：`suitability:veto` 权限只有 `SuitabilityOfficerAgent` 持有，
      因此只有它能给出 `reject`，其余 Agent 无法越权拦截流程。
    """

    def __init__(
        self,
        spec: AgentSpec,
        registry: ToolRegistry | None = None,
        llm: Any = None,
        tracer: Any = None,
    ) -> None:
        """绑定能力契约与外部依赖（纯赋值，不做任何校验或 IO）。

        参数：
            spec: 该 Agent 的能力契约（名称、角色、system prompt、工具白名单、
                权限集合、可写状态键）。通常由各子类模块级常量 `SPEC` 提供。
            registry: 工具注册表；为 None 时使用全局默认 `TOOL_REGISTRY`。
            llm: LLM 客户端（需实现 `compose(task, context)`）；为 None 时
                相关工具调用会失败，通常由 `src/llm.py` 的 `build_llm` 注入。
            tracer: 逐步追踪器（需实现 `step(...)`）；为 None 时 `trace()` 静默跳过。

        返回：None。

        副作用/异常：无。
        """
        self.spec = spec
        self.registry = registry or TOOL_REGISTRY
        self.llm = llm
        self.tracer = tracer

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Agent 名称。

        参数：无（属性）。
        返回：`spec.name`，例如 `"ClientProfilingAgent"`。
        副作用/异常：无。
        """
        return self.spec.name

    def call_tool(self, name: str, **kwargs: Any) -> Any:
        """调用白名单内的工具（含权限标签校验）。

        这是 Agent 触碰外部确定性能力的**唯一入口**，双重校验缺一不可：
        先查「工具是否在该 Agent 的白名单内」，再查「该 Agent 是否持有工具
        要求的权限标签」——因此白名单写漏但权限足够、或权限不足但白名单写了，
        两种错配都会在运行时立刻暴露。

        参数：
            name: 工具名，须已登记在 `registry` 且在 `self.spec.tools` 中。
            **kwargs: 直接透传给工具 handler 的关键字参数（工具均为
                keyword-only 实现，调用方须按签名传参）。

        返回：
            工具 handler 的返回值（各工具类型不同：模型对象、dict、list 等）。

        副作用：
            取决于具体工具；生成文案类工具会触发 LLM（或 mock）调用。

        异常：
            PermissionError: 工具不在 `spec.tools` 白名单内，或 `spec.permissions`
                缺少该工具声明的权限标签。
            KeyError: 工具名未在注册表中登记（由 `registry.get` 抛出）。
        """
        if name not in self.spec.tools:
            raise PermissionError(
                f"{self.name} 无权调用工具 {name}；其工具白名单为 {list(self.spec.tools)}"
            )
        tool = self.registry.get(name)
        if tool.permission not in self.spec.permissions:
            raise PermissionError(
                f"{self.name} 缺少权限 {tool.permission}，无法调用 {name}"
            )
        return tool.handler(**kwargs)

    # ------------------------------------------------------------------
    def guard_output(self, updates: Mapping[str, Any]) -> dict[str, Any]:
        """校验该 Agent 只写自己负责的状态键。

        每个 `run()` 的最后一步都应把输出交给本方法，让「越权写共享状态」
        在开发期就报错，而不是等到三个 Agent 互相覆盖状态后才发现。

        参数：
            updates: 该 Agent 想写回共享状态的键值对（通常是 `dict[状态键, 值]`）。

        返回：
            校验通过后的**浅拷贝** `dict(updates)`（不是原对象，
            避免调用方后续修改影响已校验的内容）。

        副作用/异常：
            无副作用；`updates` 中存在不属于 `spec.can_write_state` 的键时
            抛 PermissionError，并在消息里列出非法键与允许键。
        """
        allowed = set(self.spec.can_write_state)
        illegal = sorted(set(updates) - allowed)
        if illegal:
            raise PermissionError(
                f"{self.name} 越权写入共享状态：{illegal}；允许写入 {sorted(allowed)}"
            )
        return dict(updates)

    # ------------------------------------------------------------------
    def trace(
        self,
        node: str,
        *,
        payload_in: Any = None,
        payload_out: Any = None,
        status: str = "ok",
        tool_calls: Iterable[str] = (),
        latency_ms: float = 0.0,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """记录一步 trace（未配置 tracer 时静默跳过）。

        参数（除 node 外均为 keyword-only）：
            node: 步骤名，如 `"profile"` / `"screen"` / `"optimize"` /
                `"suitability"` / `"narrative"`。
            payload_in: 输入载荷（可 JSON 序列化的摘要，不要求全量）。
            payload_out: 输出载荷摘要。
            status: 步骤状态，默认 `"ok"`（失败路径由调用方显式传其他取值）。
            tool_calls: 本次调用过的工具名（用于审计"用了哪些能力"）。
            latency_ms: 本步耗时毫秒（通常来自 `Stopwatch.ms()`）。
            extra: 额外结构化字段（如生效风险等级、命中规则号等）。

        返回：
            tracer 记录后的条目 dict；`self.tracer is None` 时返回 None。

        副作用/异常：
            写追踪记录（可能落盘，由 tracer 实现决定）；本方法不捕获异常，
            tracer 自身抛错会向上传播。
        """
        if self.tracer is None:
            return None
        return self.tracer.step(
            self.name,
            node=node,
            input_payload=payload_in,
            output_payload=payload_out,
            status=status,
            tool_calls=list(tool_calls),
            latency_ms=latency_ms,
            extra=extra,
        )

    def describe(self) -> dict[str, Any]:
        """能力清单。

        参数：无。
        返回：`spec.describe()` 的字典（name / role / tools / permissions /
            can_write_state），供 demo、README 与评估报告展示权限边界。
        副作用/异常：无。
        """
        return self.spec.describe()

    # ------------------------------------------------------------------
    def run(self, **kwargs: Any) -> dict[str, Any]:
        """执行一次 Agent 任务，返回该 Agent 负责的状态更新。

        参数：
            **kwargs: 各子类自定义的关键字参数（客户档案、产品池、轮次等），
                完整签名见各子类 `run()` 的 docstring。

        返回：
            可写状态键 -> 值 的字典，且**必须**经 `guard_output` 校验后返回。

        副作用/异常：
            NotImplementedError: 基类占位实现，必须由子类覆写。
        """
        raise NotImplementedError


class Stopwatch:
    """轻量计时器（毫秒）。

    用 `time.perf_counter()` 取单调时钟，仅用于给 trace 填 `latency_ms`，
    不参与任何业务判定（因此不影响结果的可复现性）。

    用法：`sw = Stopwatch()` → 干完活 → `sw.ms()`。
    """

    def __init__(self) -> None:
        """记录起始时刻（构造即开始计时）。

        参数：无。
        返回：None。
        副作用：读取一次单调时钟。
        """
        self._start = time.perf_counter()

    def ms(self) -> float:
        """已耗时（毫秒）。

        参数：无。
        返回：自构造（或上次无）以来的浮点毫秒数；可重复调用，每次都重新取值。
        副作用/异常：无。
        """
        return (time.perf_counter() - self._start) * 1000.0
