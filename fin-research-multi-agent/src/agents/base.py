"""Agent 基类与运行上下文（agents 层的地基）。

层次与职责：
    本模块在依赖链上处于「上层编排（orchestrator）与具体 Agent 实现」之间：
    向下依赖工具层（tools/registry 的白名单与权限校验）、LLM 层、tracing 与 config，
    向上被 ``planner.py`` / ``retriever.py`` / ``analyst.py`` / ``risk_checker.py`` /
    ``writer.py`` 五个子类继承，以及被 ``__init__.py`` 的 ``build_agents()`` 装配。

每个 Agent 都是一个「有限职责 + 最小权限」的执行单元：
    * `allowed_tools`   —— 它能调用的工具白名单（越权直接拒绝，不是靠提示词约束）
    * `permissions`     —— 它在工具层持有的权限等级集合（工具调用时强校验）
    * `system_prompt`   —— 独立人设与输出契约（见 llm/prompts.py）
      注：实际实现为 —— Agent 身上**并没有** `system_prompt` 这个属性。各 Agent 只声明
      `name` / `role` / `allowed_tools` / `permissions` 四个类属性；人设与输出契约是按
      **任务名**（"plan" / "retrieve" / "analyze" / "risk_review" / "write"）在 LLM 层查表得到的：
      ``LLMClient.chat(task, payload)`` 内部执行 ``AGENT_PROMPTS.get(_PROMPT_KEYS.get(task, task), "")``。
    * `run()`           —— 统一模板方法：计时 -> 执行 -> 落 trace -> 累积工具调用记录

模板方法保证了「每个 Agent 的输入输出都落到 trace」这条硬性要求不是靠自觉，
而是靠基类强制执行。

对外关键对象：
    * ``AgentContext`` —— 一次运行内的依赖容器（dataclass），由 orchestrator 构造后
      被五个 Agent 共享；它同时是访问 fact_store / document_store / retriever 的便捷入口。
    * ``BaseAgent``    —— 抽象基类，定义了工具调用入口 ``call_tool()`` 与模板方法 ``run()``。

主要输入输出：
    * 输入：``ResearchState``（TypedDict，即 blackboard 共享状态）；
    * 输出：**新的** state 字典（子类通过 ``_copy`` 浅拷贝后再改，避免原地污染上游快照）；
    * 侧信道：trace 记录（``TraceRecorder.record``）与工具调用日志（``registry.drain_call_log``）。

被谁调用：
    * ``src/orchestrator.py`` 在 ``_build_spec()`` 中构造 ``AgentContext`` 并把各 Agent 的
      ``run`` 注册为图节点；引擎（engine_native / engine_langgraph）逐节点调用它。
    * ``tests/conftest.py`` 与 ``tests/test_citation_traceability.py`` 直接构造
      ``AgentContext``，用于脱离图做单 Agent 测试。
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
    """一次运行内所有 Agent 共享的依赖容器。

    它是「依赖注入」的落点：五个 Agent 都只拿到这一个对象，而不是各自去 import 全局单例，
    因此测试时替换工具/LLM/trace 只需要换一个 ctx。

    关键属性（均为构造时注入，运行期不再变更）：
        registry: 工具注册中心；Agent 的一切外部能力（检索、取指标、算比率、生成引用）
            都必须经过它，白名单与权限等级也在这一层强校验。
        llm: 统一 LLM 入口；按 ``task`` 名选择 system prompt 与 mock 应答。
        recorder: trace 记录器；``BaseAgent.run()`` 靠它满足「每步可审计」的硬性要求。
        config: 运行期配置（模型名、阈值、轮次上限等）。
        extras: 自由扩展位；orchestrator 目前放入 ``{"pipeline": self}``，
            便于将来做进程内回溯调试，Agent 逻辑本身不依赖它。
    """

    registry: ToolRegistry
    llm: LLMClient
    recorder: TraceRecorder
    config: RuntimeConfig
    extras: Dict[str, Any] = field(default_factory=dict)

    # ---- 便捷访问 ----
    # 说明：这三个属性只是对 registry.context 的转发读取（读不到时返回 None），
    # 用属性而非字段是为了避免在 dataclass 构造参数里再塞一遍同样的对象。
    @property
    def fact_store(self) -> Any:
        """结构化事实库（Planner 用它枚举公司/年份，Analyst 由工具间接访问）。

        返回：
            ``registry.context["fact_store"]``；未注册该键时为 ``None``（调用方需自行判空，
            例如 PlannerAgent 就写了 ``if store is not None``）。
        """
        return self.registry.context.get("fact_store")

    @property
    def document_store(self) -> Any:
        """文档库（Planner 用它列文档目录，RiskChecker 用它校验 source_id 可回溯）。

        返回：
            ``registry.context["document_store"]``；未注册时为 ``None``。
        """
        return self.registry.context.get("document_store")

    @property
    def retriever(self) -> Any:
        """混合检索器本体（供需要直连检索的场景使用）。

        返回：
            ``registry.context["retriever"]``；未注册时为 ``None``。
            注：实际实现为 —— 本包内五个 Agent 均未直接读取该属性，
            RetrieverAgent 是经由 ``search_filings`` 工具间接使用检索器的。
        """
        return self.registry.context.get("retriever")


class BaseAgent(ABC):
    """所有 Agent 的基类：定义权限契约 + 统一执行模板。

    职责：
        1. 用类属性声明「我是谁、我能用什么工具、我持有什么权限」；
        2. 提供 ``call_tool()`` 作为唯一的工具调用入口（白名单 + 权限双重把关）；
        3. 提供 ``run()`` 模板方法，把「执行 + 异常兜底 + 计时 + 落 trace + 回填 steps」
          固化下来，子类只需实现 ``_execute()``。

    关键属性（子类覆写，基类给的是最小权限默认值）：
        name: Agent 名称，写进 trace 的 agent 字段（也是 build_agents 的字典键）。
        role: 中文职能说明，用于 README / 状态展示。
        allowed_tools: 工具白名单元组；空元组 = 该 Agent 一个工具都调不了。
        permissions: 持有的 ``PermissionLevel`` 集合，调用工具时透传给注册中心强校验。

    状态流转：
        基类自身**无状态**（不持有 ResearchState，也不缓存跨步结果）。
        实例上仅有一个占位属性 ``self._tool_calls``（构造函数里置为空列表）。
        注：实际实现为 —— ``_tool_calls`` 目前**只写不读**：没有代码向它追加内容，
        ``_drain_tool_log()`` 走的是 ``registry.drain_call_log()``，trace 里的
        ``tool_calls`` 也来自注册中心的调用日志。因此它只是一个预留字段，
        真正的「本轮工具调用记录」由工具注册中心持有、每步取空。
        跨步状态一律通过 ``run()`` 返回的新 state 传递（blackboard 模式）。

    生命周期示例（以 analyst 为例）::

        编排层构造 ctx -> build_agents() 实例化 -> 引擎调用 run(state)
        -> _execute(state) 调工具/LLM -> recorder.record(...) -> 返回新 state
    """

    #: Agent 名称（写进 trace 的 agent 字段）
    name: str = "agent"
    #: 职能说明（用于 README / 状态展示）
    role: str = ""
    #: 允许调用的工具白名单（空元组表示该 Agent 不能调用任何工具，如 PlannerAgent）
    allowed_tools: Tuple[str, ...] = ()
    #: 持有的工具权限等级；与 allowed_tools 是「两道锁」：
    #: 白名单管「能不能碰这个工具」，权限集合管「即便工具在册，我有没有这个权限等级」。
    permissions: FrozenSet[PermissionLevel] = frozenset()

    def __init__(self, ctx: AgentContext) -> None:
        """保存共享依赖容器，并初始化占位字段。

        参数：
            ctx: 本次运行的 ``AgentContext``（工具注册中心 / LLM / trace / 配置）。

        返回：
            None。

        副作用：
            写入实例属性 ``ctx`` 与 ``_tool_calls``（空列表，见类 docstring 的说明：
            该字段目前只写不读）。构造函数**不做 I/O**，不校验白名单是否在注册中心真实存在，
            因此构造阶段不会因为工具名拼错而报错——拼错要到 ``call_tool`` 时才暴露。
        """
        self.ctx = ctx
        object.__setattr__(self, "_tool_calls", [])

    # ------------------------------------------------------------------
    # 工具调用（带白名单与权限强校验）
    # ------------------------------------------------------------------
    def call_tool(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> ToolResult:
        """调用工具。不在白名单内直接拒绝，不消耗工具层调用。

        参数：
            tool_name: 工具名，必须出现在本类的 ``allowed_tools`` 中。
            args: 工具入参字典；``None`` 时注册中心会按 schema 默认值处理。

        返回：
            ``ToolResult``。被白名单拦下时返回 ``ok=False``、
            ``error["code"] == "TOOL_NOT_ALLOWED"`` 的失败结果（其中带上本 Agent 实际持有的
            白名单，便于排障），而不是抛异常——这样越权只是一条可观测的失败记录，
            不会把整张图炸掉；正常放行时返回 ``registry.call()`` 的原始结果
            （其内已包含超时、重试、schema 校验与幂等缓存）。

        副作用 / 异常：
            成功的调用会产生工具层副作用（如检索、计算、cite_source 分配引用编号），
            并写入注册中心的调用日志。权限不足时由注册中心按统一异常体系处理。
            本方法自身不抛异常。
        """
        # 第一道锁：白名单。放在最前面是为了「越权不产生任何工具层开销」——
        # 被拒的调用不会进注册中心、不占并发、不写调用日志（因此 trace 里看不到这条越权记录）。
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
        # 第二道锁：权限等级。白名单只说「这个工具归我管」，具体权限由注册中心按 permissions 复核。
        return self.ctx.registry.call(tool_name, args, granted=set(self.permissions))

    def tool_descriptions(self) -> List[Dict[str, Any]]:
        """导出自己可用工具的 JSON Schema 描述（真机模式下可喂给模型）。

        返回：
            ``list[dict]``，每项是该工具的 name / description / JSON Schema 等元信息，
            范围严格等于 ``allowed_tools``（不会泄露其它 Agent 的工具）。

        副作用 / 异常：
            无副作用（纯读注册中心元数据）。若白名单里写了注册中心不存在的工具名，
            由 ``registry.describe`` 决定是忽略还是报错。
        """
        return self.ctx.registry.describe(self.allowed_tools)

    def _drain_tool_log(self) -> List[Dict[str, Any]]:
        """取出本轮工具调用记录（用于写入 trace 的 extra 字段）。

        返回：
            ``list[dict]``，本次 ``run()`` 期间发生的工具调用记录（工具名、是否成功、
            权限等级、耗时、重试次数、是否命中缓存、错误码等）。

        副作用：
            **有状态副作用**——注册中心的调用日志会被取空。因此一次 ``run()`` 只能排空一次，
            否则第二轮会拿到空列表；这也是它只在 ``run()`` 末尾被调用一次的原因。
        """
        return self.ctx.registry.drain_call_log()

    # ------------------------------------------------------------------
    # 模板方法
    # ------------------------------------------------------------------
    def run(self, state: ResearchState) -> ResearchState:
        """执行本 Agent 的一步，统一记录 trace。

        这是模板方法，也是编排层唯一会调用的入口（子类不要覆写它，只实现 ``_execute``）。

        参数：
            state: 上游传来的 blackboard 共享状态；本方法**不修改**它，
                异常分支里的 ``dict(state)`` 也只是浅拷贝后写副本。

        返回：
            新的 ``ResearchState``。正常路径是 ``_execute`` 的返回值；
            异常路径是「原状态浅拷贝 + errors 追加一条 ``{name}: {异常类型}: {异常信息}``」。
            两条路径最后都会被追加一条 ``steps`` 记录（step 序号 / agent / status /
            latency_ms / tool_calls），从而保证「每一步都留痕、可回放」。

        副作用 / 异常：
            * 副作用一：向 ``self.ctx.recorder`` 写入一条 trace（含输入摘要、输出摘要、
              耗时、状态、工具调用明细），**异常路径也照样记录**（status="error"），
              这正是「失败可审计」的含义；
            * 副作用二：排空工具注册中心的调用日志（见 ``_drain_tool_log``）；
            * 副作用三：返回的新 state 被追加一项 ``steps``。
            本方法**不向外抛异常**：``_execute`` 里的任何异常都被就地捕获并降级为
            errors 条目，避免单个 Agent 失败炸掉整张图（对应 except 上的 noqa: BLE001）。
        """
        started = time.perf_counter()
        status = "ok"
        try:
            new_state = self._execute(state)
        except Exception as exc:  # noqa: BLE001 - 单步失败不应炸掉整张图
            # 降级策略：保留上游状态原样，只追加错误信息。下游据此仍能继续（例如
            # risk_checker 会因为 findings 为空而报 GAP-NO-FINDING，把问题显式化）。
            status = "error"
            new_state = dict(state)  # type: ignore[assignment]
            errors = list(new_state.get("errors") or [])
            errors.append(f"{self.name}: {type(exc).__name__}: {exc}")
            new_state["errors"] = errors
        latency_ms = (time.perf_counter() - started) * 1000.0
        # 无论成功失败都排空一次：本轮调过的工具必须全部计入这条 trace，不能漏记。
        tool_calls = self._drain_tool_log()
        entry = self.ctx.recorder.record(
            agent=self.name,
            input_obj=self._trace_input(state),
            output_obj=self._trace_output(new_state),
            latency_ms=latency_ms,
            status=status,
            extra={"tool_calls": tool_calls, **self._trace_extra(new_state)},
        )
        # steps 是「给人看的执行序列」，也是 orchestrator._summary 统计 agents_visited 的来源，
        # 因此必须在基类里统一追加，不能靠子类自觉。
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
        """真正的业务逻辑。

        参数：
            state: 上游共享状态（只读约定；实现里应先用 ``_copy`` 再改）。

        返回：
            新的 state 字典；至少应写回本 Agent 「负责」的那组状态字段
            （见 src/state.py 的字段分组注释），不要越权改写别的 Agent 的字段。

        异常：
            允许直接抛出；``run()`` 会捕获并转记为 errors + status="error"。
            但更推荐像 Analyst / RiskChecker 那样把「可预期的缺失」写成 errors 条目，
            因为异常会丢失该步的业务产出，而 errors 不会。
        """

    def _trace_input(self, state: ResearchState) -> Dict[str, Any]:
        """trace 的输入摘要（默认只记问题与路由）。

        参数：
            state: 进入本步时的状态。

        返回：
            可 JSON 序列化的字典；默认 ``{"question": ..., "route": [...]}``。
            子类覆写以记录各自关心的输入（查询、findings id 列表等）。

        副作用 / 异常：无。
        """
        return {"question": state.get("question", ""), "route": state.get("route", [])}

    def _trace_output(self, state: ResearchState) -> Dict[str, Any]:
        """trace 的输出摘要（默认只记 agent 名）。

        参数：
            state: 本步执行后的状态（异常时为降级后的副本）。

        返回：
            可 JSON 序列化的字典；默认 ``{"agent": self.name}``。
            子类覆写以暴露关键产出；注意 trace 日志不宜塞全量长文本。

        副作用 / 异常：无。
        """
        return {"agent": self.name}

    def _trace_extra(self, state: ResearchState) -> Dict[str, Any]:
        """trace 的补充字段（默认无）。

        参数：
            state: 本步执行后的状态。

        返回：
            可 JSON 序列化的字典；默认空字典。子类可借此记录裁决结果、
            缺口数量等「不适合放进 output」的观测指标。

        副作用 / 异常：无。
        """
        return {}

    # ------------------------------------------------------------------
    # 通用小工具
    # ------------------------------------------------------------------
    @staticmethod
    def _copy(state: ResearchState) -> Dict[str, Any]:
        """浅拷贝状态（列表/字典字段在需要时由子类显式替换）。

        参数：
            state: 上游状态。

        返回：
            新的 dict，顶层键值与原状态相同，但与原状态**不共享顶层容器身份**；
            嵌套的 list / dict 仍是同一对象引用，因此子类改写嵌套结构时必须整体替换
            （例如 ``new_state["steps"] = list(...)``），否则会污染上游快照。

        副作用 / 异常：无（不修改入参）。
        """
        return dict(state)  # type: ignore[return-value]

    @staticmethod
    def _merge_errors(state: Dict[str, Any], prefix: str, messages: Iterable[str]) -> None:
        """把一组错误信息按 ``prefix: message`` 追加进状态的 errors 列表。

        参数：
            state: 待原地修改的状态字典（用 ``dict`` 而非 TypedDict，
                因此可以接受 ``_copy`` 出来的可变副本）。
            prefix: 前缀，惯例是 Agent 名（如 "analyst" / "risk_checker" / "writer"）。
            messages: 错误信息序列；**空序列直接返回**，不会写入空列表。

        返回：
            None（原地修改 ``state["errors"]``）。

        副作用：
            修改入参 state；若 ``messages`` 为空则完全不产生副作用（保持 errors 原样，
            所以调用方可以无脑拼接而不必先判空）。
        """
        if not messages:
            return
        errors = list(state.get("errors") or [])
        errors.extend(f"{prefix}: {m}" for m in messages)
        state["errors"] = errors

    @staticmethod
    def _llm_stats_fragment(response: Any) -> Dict[str, Any]:
        """把一次 LLM 响应压成 trace 用的片段。

        参数：
            response: LLM 调用返回对象；预期是 ``LLMResponse``，
                但这里做鸭子类型判断，任何对象都能安全传入。

        返回：
            有 ``to_trace()`` 时返回其精简视图（task / model / mocked / degraded /
            latency_ms / prompt_digest / usage / error 等，不含全量 prompt）；
            否则退化为 ``{"raw": str(response)}``，保证 trace 永远可序列化。

        副作用 / 异常：无；对未知对象也不抛异常。
        """
        return response.to_trace() if hasattr(response, "to_trace") else {"raw": str(response)}
