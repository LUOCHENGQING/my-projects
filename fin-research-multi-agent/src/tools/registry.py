"""MCP 风格的工具注册中心（最小权限工具层的核心）。

层次与职责：
    tools 包的中心模块，向上只暴露给 Agent 层（src/agents/base.py 的
    AgentContext.call_tool / tool_schemas / drain_tool_log），
    向下依赖 src/utils/jsonable.to_plain 与同包的 schema_validator。
    它把「Agent 能用什么工具、参数长什么样、有没有权限、失败怎么办」全部收敛到这里，
    Agent 自身不再直接触碰检索器 / 事实库。

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

调用链（call 的固定顺序，越权与非法参数都在这里被挡下）：
    工具存在性 -> 权限校验(granted) -> default 填充 -> JSON Schema 强校验
    -> 幂等键缓存命中 -> 带超时与重试的执行 -> to_plain 净化 -> 写缓存 -> 记审计日志。
    任何失败都不抛给 Agent，而是收敛成 ok=False 的 ToolResult（error.code 见异常类）。

对外关键符号：ToolRegistry（含 call/describe/register）、ToolSpec、ToolResult、
ToolCallRecord、PermissionLevel、ToolError 及其子类、build_default_registry。
注：__all__ 未收录 build_default_registry，但它仍可被 tools/__init__.py 按名导入。
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

# 注册中心对外契约（工具声明 + 结果 + 异常），供 tools/__init__.py 再导出
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
    """工具权限等级（继承 str 便于直接 JSON 序列化进 trace）。

    四个等级的语义见模块 docstring。等级之间没有包含关系：
    调用方必须**显式**拥有该工具声明的等级，高等级不会自动覆盖低等级
    （例如只授 WRITE 的 WriterAgent 仍拿不到 public_read 工具）。
    """

    PUBLIC_READ = "public_read"
    RESTRICTED_READ = "restricted_read"
    COMPUTE = "compute"
    WRITE = "write"


# ---------------------------------------------------------------------------
# 统一异常体系
# ---------------------------------------------------------------------------
class ToolError(Exception):
    """工具层异常基类：让上层只认一种异常，并统一转成 error 字典。

    类属性：
        code：稳定的机器可读错误码（子类覆写），用于 trace 与重试决策。
        retryable：是否可重试；默认 False（业务性失败重试没有意义），
                   注：实际实现为——它是类属性，但 ToolTimeoutError 与
                   执行期的瞬时故障会在实例上覆写该属性。
    实例属性：
        message：人类可读描述；tool：工具名；detail：附加上下文（如校验错误列表）。
    对外方法：to_dict() 导出为可 JSON 化的 dict，直接进 ToolResult.error。
    """

    code = "TOOL_ERROR"
    retryable = False

    def __init__(self, message: str, *, tool: str = "", detail: Optional[Dict[str, Any]] = None) -> None:
        """构造工具层异常：message 为错误消息，tool 为相关工具名，detail 为结构化细节（None 视为空）。"""
        super().__init__(message)
        self.message = message
        self.tool = tool
        self.detail = detail or {}

    def to_dict(self) -> Dict[str, Any]:
        """导出为可序列化字典：{code, tool, message, detail}。"""
        return {"code": self.code, "tool": self.tool, "message": self.message, "detail": self.detail}


class ToolNotFoundError(ToolError):
    """调用了未注册的工具名（通常是 LLM 编造了工具名）。code=TOOL_NOT_FOUND。"""

    code = "TOOL_NOT_FOUND"


class ToolPermissionError(ToolError):
    """调用方未被授予该工具所需的权限等级。

    code=PERMISSION_DENIED；detail 里带 required 字段说明缺少哪一级权限。
    """

    code = "PERMISSION_DENIED"


class ToolSchemaError(ToolError):
    """入参未通过 JSON Schema 校验。

    code=SCHEMA_INVALID；errors 为逐条错误描述（见 schema_validator），
    detail 里同样带一份 errors，方便直接回灌给 LLM 自纠。
    """

    code = "SCHEMA_INVALID"

    def __init__(self, message: str, *, tool: str = "", errors: Optional[Sequence[str]] = None) -> None:
        """构造参数校验异常：errors 为逐条校验错误，同时写入 detail["errors"] 供上层审计。"""
        super().__init__(message, tool=tool, detail={"errors": list(errors or [])})
        self.errors: List[str] = list(errors or [])


class ToolTimeoutError(ToolError):
    """工具执行超过 timeout_s。code=TIMEOUT，retryable=True（超时多为瞬时抖动）。"""

    code = "TIMEOUT"
    retryable = True


class ToolExecutionError(ToolError):
    """工具内部抛出的非 ToolError 异常被包装成此错误。code=EXECUTION_ERROR。

    注：实际实现为——retryable 初始为 False，仅当原始异常属于
    TimeoutError / ConnectionError / OSError 时才在实例上被置为 True。
    """

    code = "EXECUTION_ERROR"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class ToolSpec:
    """一个工具的完整声明（注册的最小单位）。

    关键属性：
        name / description：工具名与给 LLM 看的自然语言说明。
        schema：type=object 的 JSON Schema，既是校验依据也是 function-calling 描述。
        handler：实际实现，按 **args 关键字调用（参数名必须与 schema.properties 对齐）。
        permission_level：PermissionLevel，调用方需显式持有。
        timeout_s：单次执行的墙钟超时（秒），必须为正数。
        idempotent：是否幂等；为 True 才启用幂等键缓存。
        max_retries：额外重试次数，总尝试次数 = max_retries + 1。
        tags：便于筛选/展示的标签，不参与执行逻辑。
    """

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
        """导出成 OpenAI function-calling 风格的描述，可直接喂给 LLM。

        返回值：dict，含 name / description / parameters（即 schema）/
        permission_level / timeout_s / idempotent。副作用：无。
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.schema,
            "permission_level": self.permission_level.value,
            "timeout_s": self.timeout_s,
            "idempotent": self.idempotent,
        }

    def to_dict(self) -> Dict[str, Any]:
        """describe() 的完整版：额外带上 max_retries 与 tags，用于落盘/审计。"""
        data = self.describe()
        data["max_retries"] = self.max_retries
        data["tags"] = list(self.tags)
        return data


@dataclass
class ToolResult:
    """工具调用结果（永不抛异常给 Agent，统一用 ok 标记成败）。

    关键属性：
        tool：工具名；ok：是否成功。
        data：成功时的返回值（已经过 to_plain 净化）。
        error：失败时的错误字典（ToolError.to_dict()），含 code/tool/message/detail。
        latency_ms：本次调用的墙钟耗时（含重试与等待）。
        attempts：实际尝试次数，首次成功为 1。
        cached：是否命中幂等缓存（命中时 attempts 固定为 1）。
    """

    tool: str
    ok: bool
    data: Any = None
    error: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    attempts: int = 1
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """导出为 dict（latency_ms 保留 3 位小数，便于比对性能）。副作用：无。"""
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
    """一次工具调用的审计记录（只留元信息，不留参数与返回值）。

    关键属性：
        tool / ok / permission_level：调用对象与结果、当时声明的权限等级。
        latency_ms / attempts / cached：性能与缓存命中情况。
        error_code：失败时的错误码（成功为 None）。
    说明：不落 args/output 是有意为之——trace 里只保留 digest 级别的信息，
    避免把业务数据与潜在敏感内容写进日志。由 drain_call_log() 取出并清空。
    """

    tool: str
    ok: bool
    permission_level: str
    latency_ms: float
    attempts: int
    cached: bool
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """导出为 dict（latency_ms 保留 3 位小数）。副作用：无。"""
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
    """工具注册中心：注册、描述、校验、限权、超时、重试、幂等缓存。

    关键属性：
        _tools：name -> ToolSpec 的注册表（唯一事实来源）。
        _order：注册顺序（保序输出工具列表，让 LLM 侧看到稳定顺序）。
        _cache：幂等键 -> 成功结果的缓存；失败结果不入缓存。
        _cache_lock：保护 _cache 的互斥锁（工具在线程池里执行）。
        call_log：ToolCallRecord 审计日志，由 drain_call_log() 取出并清空。
        context：注入给工具实现的共享依赖（retriever / fact_store / document_store）。
        _pool：4 线程的执行池，仅用于实现「超时后放弃等待」。

    状态流转：
        创建（可空 context）-> register/replace 装载工具 -> call 反复调用并累积 call_log
        -> drain_call_log 清空日志 -> clear_cache 可随时重置幂等缓存。
        注：实际实现为——call_log 的 append 未加锁，依赖 Agent 侧单线程调用；
        并发调用时日志顺序不作保证，但不会丢记录。
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None) -> None:
        """初始化注册表：工具表、注册顺序、幂等缓存、调用日志、共享上下文与工具线程池（context 为注入给工具的共享上下文）。"""
        self._tools: Dict[str, ToolSpec] = {}
        # 单独维护顺序表：dict 虽保序，但显式列表更直观，也便于 replace 时判断存在性
        self._order: List[str] = []
        # 值保存的是完整 ToolResult（含 data），命中后按其重建一个 cached=True 的新结果
        self._cache: Dict[str, ToolResult] = {}
        self._cache_lock = threading.Lock()
        self.call_log: List[ToolCallRecord] = []
        # 工具实现所需的共享依赖（检索器、事实库等），由 build_default_registry 注入
        self.context: Dict[str, Any] = dict(context or {})
        # max_workers=4：工具都是短耗时的读/算，留出并发余量即可，避免线程数失控
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")

    # ---------------- 注册 ----------------
    def register(self, spec: ToolSpec) -> None:
        """注册一个新工具（重复注册直接报错，避免静默覆盖）。

        参数：spec —— 完整工具声明。
        返回值：None。
        副作用：写入 _tools / _order。
        异常：ValueError —— 名称重复、schema 不是 type=object、timeout_s <= 0；
              permission_level 传字符串时会被尝试转成 PermissionLevel，非法值同样抛错。
        """
        if spec.name in self._tools:
            raise ValueError(f"工具重复注册：{spec.name}")
        # 强制 type=object：Agent 侧固定以关键字参数调用，顶层必须是对象
        if not isinstance(spec.schema, dict) or spec.schema.get("type") != "object":
            raise ValueError(f"工具 {spec.name} 的 schema 必须是 type=object 的 JSON Schema")
        # 非正超时会让 future.result 立即超时，属于配置错误，注册期就拦下
        if spec.timeout_s <= 0:
            raise ValueError(f"工具 {spec.name} 的 timeout_s 必须为正数")
        # 允许写字符串（如 "write"），在此归一成枚举，后续权限比较才是同类型比较
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
        """装饰器写法注册工具。

        参数：与 ToolSpec 字段一一对应（tags 为可选序列）。
        返回值：装饰器函数；被装饰的函数**原样返回**，因此 handler 仍可被直接调用。
        副作用：装饰时立即调用 self.register()，注册失败会在导入期抛 ValueError。
        异常：见 register()。
        """

        def wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
            """装饰器内层：把被装饰函数连同工具元信息注册进注册表，并原样返回该函数。"""
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
        """替换已有工具（测试里注入 stub 用）。

        参数：spec —— 新声明，按 spec.name 覆盖。
        返回值：None。
        副作用：写 _tools；若该名字此前未注册，则追加进 _order（保持可枚举）。
        异常：不校验 schema/timeout（与 register 有意不同，便于测试注入最简 stub）。
        """
        self._tools[spec.name] = spec
        if spec.name not in self._order:
            self._order.append(spec.name)

    # ---------------- 查询 ----------------
    def names(self) -> List[str]:
        """返回按注册顺序排列的工具名列表（副本，改动不影响内部状态）。"""
        return list(self._order)

    def specs(self) -> List[ToolSpec]:
        """返回按注册顺序排列的 ToolSpec 列表（元素为内部对象，非副本）。"""
        return [self._tools[n] for n in self._order]

    def get(self, name: str) -> Optional[ToolSpec]:
        """按名取工具声明；不存在返回 None（不抛异常，便于调用方自行决定策略）。"""
        return self._tools.get(name)

    def describe(self, names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        """导出工具描述（可只导出某个 Agent 被授权的那几个）。

        参数：names —— 指定要导出的工具名；None 表示全部。
        返回值：List[dict]（ToolSpec.describe() 的结果），跳过不存在的名字。
        副作用 / 异常：无；名字不存在时静默忽略，避免权限集合与工具表不一致时直接崩。
        """
        target = list(names) if names is not None else self._order
        out: List[Dict[str, Any]] = []
        for n in target:
            spec = self._tools.get(n)
            if spec is not None:
                out.append(spec.describe())
        return out

    def clear_cache(self) -> None:
        """清空幂等缓存（例如底层语料被重新装载后，旧结果已过期）。副作用：清空 _cache。"""
        with self._cache_lock:
            self._cache.clear()

    @staticmethod
    def cache_key(name: str, args: Dict[str, Any]) -> str:
        """幂等键：工具名 + 规范化 JSON 参数（sort_keys 保证参数顺序无关）。

        参数：name —— 工具名；args —— 已完成默认值填充的参数字典。
        返回值：str，形如 "calc_ratio::{\"denominator\": 2, ...}"。
        副作用 / 异常：无；default=str 兜住不可序列化的参数值。
        """
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
        """统一调用入口。任何异常都被收敛成 ok=False 的 ToolResult。

        参数：
            name：工具名。
            args：参数字典（可为 None，按空参处理）；会用 schema 补默认值后再校验。
            granted：调用方被授予的权限集合。**安全要点**——传 None 表示跳过权限校验
                     （仅供内部/测试直连使用）；Agent 必须传入自己的 permissions
                     （见 src/agents/base.py:89 的 granted=set(self.permissions)），
                     否则最小权限形同虚设。
            idempotency_key：显式指定幂等键，覆盖默认的「工具名 + 参数」计算值；
                     可用于让不同参数共享缓存，或在同一逻辑步骤内复用结果。
        返回值：
            ToolResult —— 失败时 ok=False 且 error 为 ToolError.to_dict()，
            error.code 可能是 TOOL_NOT_FOUND / PERMISSION_DENIED / SCHEMA_INVALID /
            TIMEOUT / EXECUTION_ERROR。本方法自身不抛异常。
        副作用：
            命中/写入幂等缓存、追加一条 ToolCallRecord 到 call_log、可能 sleep 退避。
        注：失败结果**不**写入缓存，避免一次瞬时故障被永久缓存。
        """
        raw_args = dict(args or {})
        started = time.perf_counter()

        spec = self._tools.get(name)
        if spec is None:
            # 工具不存在时按 PUBLIC_READ 记账：此时还不知道该工具声明什么权限
            return self._fail(
                name, ToolNotFoundError(f"工具不存在：{name}", tool=name),
                started, PermissionLevel.PUBLIC_READ.value,
            )

        # 1) 权限校验
        # 越权在代码层直接拒绝（不是靠 prompt 约束），权限不足即返回 PERMISSION_DENIED
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
        # 先填默认值再校验：schema 里 type 为 ["integer","null"] 的字段靠 default 落地
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
        # 只有声明 idempotent=True 的工具才走缓存；非幂等工具（如 cite_source 分配引用序号）
        # 即使传了 idempotency_key 也不会命中，避免把有副作用的调用折叠掉
        key = idempotency_key or self.cache_key(name, args)
        if spec.idempotent:
            with self._cache_lock:
                hit = self._cache.get(key)
            if hit is not None:
                # 重建结果而非直接返回缓存对象：保证 latency_ms/cached 反映本次调用
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
        # 总尝试次数 = 重试次数 + 1；max(1, ...) 兜住 max_retries 配置成负数的异常情形
        max_attempts = max(1, spec.max_retries + 1)
        while attempts < max_attempts:
            attempts += 1
            try:
                data = self._run_with_timeout(spec, args)
            except ToolError as exc:
                last_error = exc
                # 不可重试（如 schema/权限类）或已达上限就退出；否则轻量退避后重试
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
                # 只缓存成功结果；写缓存与读缓存共用同一把锁，避免并发下的撕裂读
                if spec.idempotent:
                    with self._cache_lock:
                        self._cache[key] = result
                self._log(result, spec)
                return result

        # 循环正常结束的唯一出口：所有尝试都失败，last_error 必然已被赋值
        assert last_error is not None
        return self._fail(name, last_error, started, spec.permission_level.value, attempts=attempts)

    def _run_with_timeout(self, spec: ToolSpec, args: Dict[str, Any]) -> Any:
        """在线程池里执行，超时抛 ToolTimeoutError。

        参数：spec —— 工具声明（提供 handler 与 timeout_s）；args —— 已校验的参数。
        返回值：handler 的原始返回值（尚未 to_plain 净化）。
        副作用：占用执行池线程；超时后仅 cancel 未开始的任务。
        异常：ToolTimeoutError（超时）；handler 自身抛出的异常原样向上传递，
              由 call() 包装成 ToolExecutionError。

        注意：Python 无法强行终止线程，超时后该线程仍会跑完（只是结果被丢弃）。
        工具本身都是幂等的纯读/纯算，因此这种"泄漏"是可接受的。
        """
        # 以关键字参数提交：参数名与 schema.properties 对齐是工具实现的硬约束
        future = self._pool.submit(spec.handler, **args)
        try:
            return future.result(timeout=spec.timeout_s)
        except FutureTimeoutError:
            # cancel 只能拦住尚未开始执行的任务；已开始的会继续跑完
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
        """构造失败结果并记一条审计日志（供 call 的所有失败分支复用）。

        参数：
            name：工具名；error：具体的 ToolError；started：perf_counter 起点；
            permission_level：暴露在审计记录里的权限等级字符串；
            attempts：实际尝试次数（首次失败默认 1）。
        返回值：ToolResult(ok=False, data=None, error=error.to_dict())。
        副作用：向 call_log 追加一条 ok=False 的记录（error_code 取 error.code）。
        异常：无。
        """
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
        """把一次成功/缓存命中的调用追加进审计日志。

        参数：result —— 已构造好的结果；spec —— 提供 permission_level。
        返回值：None。
        副作用：call_log.append（error_code 成功时为 None）。
        异常：无。这里不调用 _fail，是为了让失败路径只在 _fail 一处记账。
        """
        self.call_log.append(
            ToolCallRecord(
                tool=result.tool, ok=result.ok,
                permission_level=spec.permission_level.value,
                latency_ms=result.latency_ms, attempts=result.attempts,
                cached=result.cached, error_code=None if result.ok else (result.error or {}).get("code"),
            )
        )

    def drain_call_log(self) -> List[Dict[str, Any]]:
        """取出并清空调用日志（Agent 在每步结束后收集，写进 trace）。

        参数：无。
        返回值：List[dict]，本次取出的审计记录（已转 dict）。
        副作用：清空 call_log —— 非幂等，重复调用第二次返回空列表；
                因此同一批日志只会被写入 trace 一次。
        异常：无。
        """
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
    """构建带全部内置工具、并注入数据依赖的注册中心。

    参数：
        retriever：检索器（search_filings 使用，缺省时该工具调用会报"检索器未初始化"）。
        fact_store：结构化事实库（get_financial_metric / check_risk_rules 使用）。
        document_store：文档库（cite_source 使用）。
    返回值：ToolRegistry，已注册 5 个内置工具。
    副作用：创建线程池；把三个依赖放进 registry.context（工具通过 context 取用）。
    异常：注册期配置错误会抛 ValueError（见 register）。
    被谁调用：src/orchestrator.py:148、tests/conftest.py:60。
    """
    from . import builtin  # 延迟导入，避免循环依赖

    registry = ToolRegistry(
        context={"retriever": retriever, "fact_store": fact_store, "document_store": document_store}
    )
    builtin.register_all(registry)
    return registry
