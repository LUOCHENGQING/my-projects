"""Agent 基类：工具白名单、权限集合、越权拦截与逐步 trace。

三个约束在基类里统一落地，子类只写业务：
1. `call_tool` 双重校验（工具白名单 + 权限标签），越权直接 `PermissionError`；
2. `guard_output` 限制每个 Agent 只能写自己负责的共享状态键；
3. `trace` 统一记录 `step / agent / input_digest / output_digest / latency_ms / status / tool_calls`。
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Mapping

from .tools import TOOL_REGISTRY, AgentSpec, ToolRegistry


class BaseAgent:
    """所有投顾 Agent 的基类。"""

    def __init__(
        self,
        spec: AgentSpec,
        registry: ToolRegistry | None = None,
        llm: Any = None,
        tracer: Any = None,
    ) -> None:
        self.spec = spec
        self.registry = registry or TOOL_REGISTRY
        self.llm = llm
        self.tracer = tracer

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Agent 名称。"""
        return self.spec.name

    def call_tool(self, name: str, **kwargs: Any) -> Any:
        """调用白名单内的工具（含权限标签校验）。"""
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
        """校验该 Agent 只写自己负责的状态键。"""
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
        """记录一步 trace（未配置 tracer 时静默跳过）。"""
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
        """能力清单。"""
        return self.spec.describe()

    # ------------------------------------------------------------------
    def run(self, **kwargs: Any) -> dict[str, Any]:
        """执行一次 Agent 任务，返回该 Agent 负责的状态更新。"""
        raise NotImplementedError


class Stopwatch:
    """轻量计时器（毫秒）。"""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def ms(self) -> float:
        """已耗时（毫秒）。"""
        return (time.perf_counter() - self._start) * 1000.0
