"""工具层（tools 包）：MCP 风格的工具注册中心与内置工具。

层次与职责：
    位于「Agent 层」之下、「数据/RAG 层」之上：Agent 不直接调用检索器或事实库，
    只能经由本包暴露的工具接口行动，从而把权限、校验、超时、审计收敛到一处。

包内模块：
    schema_validator  轻量 JSON Schema 校验器（工具入参校验）
    registry          工具注册中心（权限分级 / 超时 / 重试 / 幂等缓存 / 统一异常）
    builtin           业务工具：search_filings / get_financial_metric / calc_ratio /
                      check_risk_rules / cite_source

对外主要导出（见 __all__）：PermissionLevel、ToolSpec（工具声明）、
ToolResult（调用结果）、ToolRegistry（注册中心）、ToolCallRecord（审计记录）、
各类 ToolError 子类、以及 build_default_registry（一键装配）。

被谁调用：
    src/orchestrator.py:148 与 tests/conftest.py:60 调用 build_default_registry 装配；
    src/agents/base.py:89 通过 registry.call(..., granted=set(self.permissions)) 发起调用，
    第 93/97 行分别用 registry.describe() 与 registry.drain_call_log() 取工具描述与审计日志。
    注：builtin 模块不在此导出，它由 build_default_registry 内部延迟导入并注册。
"""

from __future__ import annotations

# 只做「注册中心侧」符号的再导出；builtin.register_all 由 build_default_registry 内部调用
from .registry import (
    PermissionLevel,
    ToolCallRecord,
    ToolError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolRegistry,
    ToolResult,
    ToolSchemaError,
    ToolSpec,
    ToolTimeoutError,
    build_default_registry,
)

# 包级对外契约：Agent 层只需 from ..tools import 这些符号即可，无需深入子模块
__all__ = [
    "PermissionLevel",
    "ToolSpec",
    "ToolResult",
    "ToolRegistry",
    "ToolCallRecord",
    "ToolError",
    "ToolNotFoundError",
    "ToolPermissionError",
    "ToolSchemaError",
    "ToolTimeoutError",
    "build_default_registry",
]
