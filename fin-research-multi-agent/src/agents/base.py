"""Agent 基类与运行上下文。

每个 Agent 都是一个「有限职责 + 最小权限」的执行单元：
    * `allowed_tools`   —— 它能调用的工具白名单（越权直接拒绝，不是靠提示词约束）
    * `permissions`     —— 它在工具层持有的权限等级集合（工具调用时强校验）
    * `system_prompt`   —— 独立人设与输出契约（见 llm/prompts.py）
    * `run()`           —— 统一模板方法：计时 -> 执行 -> 落 trace -> 累积工具调用记录

模板方法保证了「每个 Agent 的输入输出都落到 trace」这条硬性要求不是靠自觉，
而是靠基类强制执行。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from ..config import RuntimeConfig
from ..llm.client import LLMClient
from ..state import ResearchState
from ..tracing import TraceRecorder
from ..tools.registry import PermissionLevel, ToolRegistry, ToolResult

__all__ = ["AgentContext", "BaseAgent"]


@dataclass
class AgentContext:
    """一次运行内所有 Agent 共享的依赖容器。"""

    registry: ToolRegistry
    llm: LLMClient
    recorder: TraceRecorder
    config: RuntimeConfig
    extras: Dict[str, Any] = field(default_factory=dict)

    # ---- 便捷访问 ----
    @property
    def fact_store(self) -> Any:
        return self.registry.context.get("fact_store")

    @property
    def document_store(self) -> Any:
        return self.registry.context.get("document_store")

    @property
    def retriever(self) -> Any:
        return self.registry.context.get("retriever")


class BaseAgent(ABC):
    """所有 Agent 的基类。"""

    #: Agent 名称（写进 trace 的 agent 字段）
    name: str = "agent"
    #: 职能说明（用于 README / 状态展示）
    role: str = ""
    #: 允许调用的工具白名单
    allowed_tools: Tuple[str, ...] = ()
    #: 持有的工具权限等级
    permissions: FrozenSet[PermissionLevel] = frozenset()

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx
        object.__setattr__(self, "_tool_calls", [])

    # ------------------------------------------------------------------
    # 工具调用（带白名单与权限强校验）
    # ------------------------------------------------------------------
    def call_tool(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> ToolResult:
        """调用工具。不在白名单内直接拒绝，不消耗工具层调用。"""
        if tool_name not in self.allowed_tools:
            return ToolResult(
                tool=tool_name,
                ok=False,
                error={
                    "code": "TOOL_NOT_ALLOWED",
                    "tool": tool_name,
                    "message": (
                        f"{self.name} 未被授权使用 {tool_name}；"
                        f"它只持有：{list(self.allowed_tools)}"
                    ),
                    "detail": {"allowed": list(self.allowed_tools)},
                },
                latency_ms=0.0,
            )
        return self.ctx.registry.call(tool_name, args, granted=set(self.permissions))

    def tool_descriptions(self) -> List[Dict[str, Any]]:
        """导出自己可用工具的 JSON Schema 描述（真机模式下可喂给模型）。"""
        return self.ctx.registry.describe(self.allowed_tools)

    def _drain_tool_log(self) -> List[Dict[str, Any]]:
        """取出本轮工具调用记录（用于写入 trace 的 extra 字段）。"""
        return self.ctx.registry.drain_call_log()

    # ------------------------------------------------------------------
    # 模板方法
    # ------------------------------------------------------------------
    def run(self, state: ResearchState) -> ResearchState:
        """执行本 Agent 的一步，统一记录 trace。"""
        started = time.perf_counter()
        status = "ok"
        try:
            new_state = self._execute(state)
        except Exception as exc:  # noqa: BLE001 - 单步失败不应炸掉整张图
            status = "error"
            new_state = dict(state)  # type: ignore[assignment]
            errors = list(new_state.get("errors") or [])
            errors.append(f"{self.name}: {type(exc).__name__}: {exc}")
            new_state["errors"] = errors
        latency_ms = (time.perf_counter() - started) * 1000.0
        tool_calls = self._drain_tool_log()
        entry = self.ctx.recorder.record(
            agent=self.name,
            input_obj=self._trace_input(state),
            output_obj=self._trace_output(new_state),
            latency_ms=latency_ms,
            status=status,
            extra={"tool_calls": tool_calls, **self._trace_extra(new_state)},
        )
        steps = list(new_state.get("steps") or [])
        steps.append({"step": entry["step"], "agent": self.name, "status": status,
                      "latency_ms": entry["latency_ms"], "tool_calls": tool_calls})
        new_state["steps"] = steps  # type: ignore[typeddict-item]
        return new_state

    # ------------------------------------------------------------------
    # 子类实现
    # ------------------------------------------------------------------
    @abstractmethod
    def _execute(self, state: ResearchState) -> ResearchState:
        """真正的业务逻辑。"""

    def _trace_input(self, state: ResearchState) -> Dict[str, Any]:
        return {"question": state.get("question", ""), "route": state.get("route", [])}

    def _trace_output(self, state: ResearchState) -> Dict[str, Any]:
        return {"agent": self.name}

    def _trace_extra(self, state: ResearchState) -> Dict[str, Any]:
        return {}

    # ------------------------------------------------------------------
    # 通用小工具
    # ------------------------------------------------------------------
    @staticmethod
    def _copy(state: ResearchState) -> Dict[str, Any]:
        """浅拷贝状态（列表/字典字段在需要时由子类显式替换）。"""
        return dict(state)  # type: ignore[return-value]

    @staticmethod
    def _merge_errors(state: Dict[str, Any], prefix: str, messages: Iterable[str]) -> None:
        if not messages:
            return
        errors = list(state.get("errors") or [])
        errors.extend(f"{prefix}: {m}" for m in messages)
        state["errors"] = errors

    @staticmethod
    def _llm_stats_fragment(response: Any) -> Dict[str, Any]:
        """把一次 LLM 响应压成 trace 用的片段。"""
        return response.to_trace() if hasattr(response, "to_trace") else {"raw": str(response)}
