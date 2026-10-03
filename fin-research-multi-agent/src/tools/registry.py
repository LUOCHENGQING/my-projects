"""MCP 风格的工具注册中心。

设计目标（对应 README「工具层为什么要权限分级与幂等」）：
    * 每个工具用 JSON Schema 声明入参，**调用前**强校验，杜绝 LLM 幻觉参数直接落到业务逻辑；
    * 工具属性齐备：name / description / schema / permission_level / timeout_s / idempotent；
    * 统一异常体系 + 超时 + 重试 + 幂等键缓存；
    * 每次调用留痕（ToolCallRecord），供 Agent 写进 trace，做到「工具调用可审计」。

权限分级（PermissionLevel）
    public_read     读公开资料（检索年报、取指标）——风险最低
    restricted_read 读受控资料（合规规则库）——需要显式授权
    compute         纯计算（比率计算）——无副作用
    write           有副作用 / 产生对外产物（生成引用编号、落盘）——最严格

Agent 只持有自己需要的权限集合（见 agents/base.py），例如 AnalystAgent 拿不到 write
权限，就不可能误触发写操作。这是「最小权限」在 Agent 层面的落地。
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from ..utils.jsonable import to_plain
from .schema_validator import SchemaValidationError, apply_defaults, validate

__all__ = [
    "PermissionLevel",
    "ToolSpec",
    "ToolResult",
    "ToolCallRecord",
    "ToolRegistry",
    "ToolError",
    "ToolNotFoundError",
    "ToolPermissionError",
    "ToolSchemaError",
    "ToolTimeoutError",
    "ToolExecutionError",
]


class PermissionLevel(str, Enum):
    """工具权限等级。"""

    PUBLIC_READ = "public_read"
    RESTRICTED_READ = "restricted_read"
    COMPUTE = "compute"
    WRITE = "write"


# ---------------------------------------------------------------------------
# 统一异常体系
# ---------------------------------------------------------------------------
class ToolError(Exception):
    """工具层异常基类。"""

    code = "TOOL_ERROR"
    retryable = False

    def __init__(self, message: str, *, tool: str = "", detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.tool = tool
        self.detail = detail or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "tool": self.tool, "message": self.message, "detail": self.detail}


class ToolNotFoundError(ToolError):
    code = "TOOL_NOT_FOUND"


class ToolPermissionError(ToolError):
    """调用方未被授予该工具所需的权限等级。"""

    code = "PERMISSION_DENIED"


class ToolSchemaError(ToolError):
    """入参未通过 JSON Schema 校验。"""

    code = "SCHEMA_INVALID"

    def __init__(self, message: str, *, tool: str = "", errors: Optional[Sequence[str]] = None) -> None:
        super().__init__(message, tool=tool, detail={"errors": list(errors or [])})
        self.errors: List[str] = list(errors or [])


class ToolTimeoutError(ToolError):
    code = "TIMEOUT"
    retryable = True


class ToolExecutionError(ToolError):
    code = "EXECUTION_ERROR"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class ToolSpec:
    """一个工具的完整声明。"""

    name: str
    description: str
    schema: Dict[str, Any]
    handler: Callable[..., Any]
    permission_level: PermissionLevel = PermissionLevel.PUBLIC_READ
    timeout_s: float = 5.0
    idempotent: bool = True
    max_retries: int = 2
    tags: List[str] = field(default_factory=list)

    def describe(self) -> Dict[str, Any]:
        """导出成 OpenAI function-calling 风格的描述，可直接喂给 LLM。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.schema,
            "permission_level": self.permission_level.value,
            "timeout_s": self.timeout_s,
            "idempotent": self.idempotent,
        }

    def to_dict(self) -> Dict[str, Any]:
        data = self.describe()
        data["max_retries"] = self.max_retries
        data["tags"] = list(self.tags)
        return data


@dataclass
class ToolResult:
    """工具调用结果（永不抛异常给 Agent，统一用 ok 标记成败）。"""

    tool: str
    ok: bool
    data: Any = None
    error: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    attempts: int = 1
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 3),
            "attempts": self.attempts,
            "cached": self.cached,
        }


@dataclass
class ToolCallRecord:
    """一次工具调用的审计记录。"""

    tool: str
    ok: bool
    permission_level: str
    latency_ms: float
    attempts: int
    cached: bool
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "permission_level": self.permission_level,
            "latency_ms": round(self.latency_ms, 3),
            "attempts": self.attempts,
            "cached": self.cached,
            "error_code": self.error_code,
        }


# ---------------------------------------------------------------------------
# 注册中心
# ---------------------------------------------------------------------------
class ToolRegistry:
    """工具注册中心：注册、描述、校验、限权、超时、重试、幂等缓存。"""

    def __init__(self, context: Optional[Dict[str, Any]] = None) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self._order: List[str] = []
        self._cache: Dict[str, ToolResult] = {}
        self._cache_lock = threading.Lock()
        self.call_log: List[ToolCallRecord] = []
        # 工具实现所需的共享依赖（检索器、事实库等），由 build_default_registry 注入
        self.context: Dict[str, Any] = dict(context or {})
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")

    # ---------------- 注册 ----------------
    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"工具重复注册：{spec.name}")
        if not isinstance(spec.schema, dict) or spec.schema.get("type") != "object":
            raise ValueError(f"工具 {spec.name} 的 schema 必须是 type=object 的 JSON Schema")
        if spec.timeout_s <= 0:
            raise ValueError(f"工具 {spec.name} 的 timeout_s 必须为正数")
        spec.permission_level = (
            spec.permission_level
            if isinstance(spec.permission_level, PermissionLevel)
            else PermissionLevel(spec.permission_level)
        )
        self._tools[spec.name] = spec
        self._order.append(spec.name)

    def tool(
        self,
        name: str,
        description: str,
        schema: Dict[str, Any],
        permission_level: PermissionLevel = PermissionLevel.PUBLIC_READ,
        timeout_s: float = 5.0,
        idempotent: bool = True,
        max_retries: int = 2,
        tags: Optional[Sequence[str]] = None,
    ):
        """装饰器写法注册工具。"""

        def wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                ToolSpec(
                    name=name,
                    description=description,
                    schema=schema,
                    handler=fn,
                    permission_level=permission_level,
                    timeout_s=timeout_s,
                    idempotent=idempotent,
                    max_retries=max_retries,
                    tags=list(tags or []),
                )
            )
            return fn

        return wrapper

    def replace(self, spec: ToolSpec) -> None:
        """替换已有工具（测试里注入 stub 用）。"""
        self._tools[spec.name] = spec
        if spec.name not in self._order:
            self._order.append(spec.name)

    # ---------------- 查询 ----------------
    def names(self) -> List[str]:
        return list(self._order)

    def specs(self) -> List[ToolSpec]:
        return [self._tools[n] for n in self._order]

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def describe(self, names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        """导出工具描述（可只导出某个 Agent 被授权的那几个）。"""
        target = list(names) if names is not None else self._order
        out: List[Dict[str, Any]] = []
        for n in target:
            spec = self._tools.get(n)
            if spec is not None:
                out.append(spec.describe())
        return out

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    @staticmethod
    def cache_key(name: str, args: Dict[str, Any]) -> str:
        """幂等键：工具名 + 规范化 JSON 参数（sort_keys 保证参数顺序无关）。"""
        payload = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
        return f"{name}::{payload}"

    # ---------------- 调用 ----------------
    def call(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        *,
        granted: Optional[Set[PermissionLevel]] = None,
        idempotency_key: Optional[str] = None,
    ) -> ToolResult:
        """统一调用入口。任何异常都被收敛成 ok=False 的 ToolResult。"""
        raw_args = dict(args or {})
        started = time.perf_counter()

        spec = self._tools.get(name)
        if spec is None:
            return self._fail(
                name, ToolNotFoundError(f"工具不存在：{name}", tool=name),
                started, PermissionLevel.PUBLIC_READ.value,
            )

        # 1) 权限校验
        if granted is not None and spec.permission_level not in granted:
            return self._fail(
                name,
                ToolPermissionError(
                    f"调用方未被授予 {spec.permission_level.value} 权限，无法调用 {name}",
                    tool=name,
                    detail={"required": spec.permission_level.value},
                ),
                started,
                spec.permission_level.value,
            )

        # 2) JSON Schema 校验（含 default 填充）
        try:
            candidate = apply_defaults(raw_args, spec.schema)
            validate(candidate, spec.schema)
        except SchemaValidationError as exc:
            return self._fail(
                name,
                ToolSchemaError(f"参数校验失败：{exc}", tool=name, errors=exc.errors),
                started,
                spec.permission_level.value,
            )
        args = candidate

        # 3) 幂等缓存命中
        key = idempotency_key or self.cache_key(name, args)
        if spec.idempotent:
            with self._cache_lock:
                hit = self._cache.get(key)
            if hit is not None:
                cached = ToolResult(
                    tool=hit.tool, ok=hit.ok, data=hit.data, error=hit.error,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    attempts=1, cached=True,
                )
                self._log(cached, spec)
                return cached

        # 4) 带重试与超时的执行
        attempts = 0
        last_error: Optional[ToolError] = None
        max_attempts = max(1, spec.max_retries + 1)
        while attempts < max_attempts:
            attempts += 1
            try:
                data = self._run_with_timeout(spec, args)
            except ToolError as exc:
                last_error = exc
                if not exc.retryable or attempts >= max_attempts:
                    break
                time.sleep(min(0.05 * attempts, 0.2))  # 轻量退避
                continue
            except Exception as exc:  # noqa: BLE001 - 统一收敛为工具异常
                error = ToolExecutionError(
                    f"工具执行失败：{type(exc).__name__}: {exc}", tool=name
                )
                # 把「瞬时故障」显式标成可重试：网络抖动、连接断开、下游超时
                # 这类错误重试往往就能成功；而业务异常（如分母为 0）重试没有意义。
                if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
                    error.retryable = True
                last_error = error
                if error.retryable and attempts < max_attempts:
                    time.sleep(min(0.05 * attempts, 0.2))
                    continue
                break
            else:
                # 净化返回值：确保不把 numpy 标量等非原生类型带到上层状态里
                result = ToolResult(
                    tool=name, ok=True, data=to_plain(data),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    attempts=attempts, cached=False,
                )
                if spec.idempotent:
                    with self._cache_lock:
                        self._cache[key] = result
                self._log(result, spec)
                return result

        assert last_error is not None
        return self._fail(name, last_error, started, spec.permission_level.value, attempts=attempts)

    def _run_with_timeout(self, spec: ToolSpec, args: Dict[str, Any]) -> Any:
        """在线程池里执行，超时抛 ToolTimeoutError。

        注意：Python 无法强行终止线程，超时后该线程仍会跑完（只是结果被丢弃）。
        工具本身都是幂等的纯读/纯算，因此这种"泄漏"是可接受的。
        """
        future = self._pool.submit(spec.handler, **args)
        try:
            return future.result(timeout=spec.timeout_s)
        except FutureTimeoutError:
            future.cancel()
            raise ToolTimeoutError(
                f"工具 {spec.name} 执行超过 {spec.timeout_s}s 超时", tool=spec.name
            ) from None

    def _fail(
        self,
        name: str,
        error: ToolError,
        started: float,
        permission_level: str,
        attempts: int = 1,
    ) -> ToolResult:
        result = ToolResult(
            tool=name, ok=False, data=None, error=error.to_dict(),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            attempts=attempts, cached=False,
        )
        record = ToolCallRecord(
            tool=name, ok=False, permission_level=permission_level,
            latency_ms=result.latency_ms, attempts=attempts, cached=False,
            error_code=error.code,
        )
        self.call_log.append(record)
        return result

    def _log(self, result: ToolResult, spec: ToolSpec) -> None:
        self.call_log.append(
            ToolCallRecord(
                tool=result.tool, ok=result.ok,
                permission_level=spec.permission_level.value,
                latency_ms=result.latency_ms, attempts=result.attempts,
                cached=result.cached, error_code=None if result.ok else (result.error or {}).get("code"),
            )
        )

    def drain_call_log(self) -> List[Dict[str, Any]]:
        """取出并清空调用日志（Agent 在每步结束后收集，写进 trace）。"""
        records = [r.to_dict() for r in self.call_log]
        self.call_log.clear()
        return records


# ---------------------------------------------------------------------------
# 默认注册表装配
# ---------------------------------------------------------------------------
def build_default_registry(
    retriever: Any = None,
    fact_store: Any = None,
    document_store: Any = None,
) -> ToolRegistry:
    """构建带全部内置工具、并注入数据依赖的注册中心。"""
    from . import builtin  # 延迟导入，避免循环依赖

    registry = ToolRegistry(
        context={"retriever": retriever, "fact_store": fact_store, "document_store": document_store}
    )
    builtin.register_all(registry)
    return registry
