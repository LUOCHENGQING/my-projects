"""工具层：MCP 风格的工具注册中心与内置工具。

    schema_validator  轻量 JSON Schema 校验器（工具入参校验）
    registry          工具注册中心（权限分级 / 超时 / 重试 / 幂等缓存 / 统一异常）
    builtin           业务工具：search_filings / get_financial_metric / calc_ratio /
                      check_risk_rules / cite_source
"""

from __future__ import annotations

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
