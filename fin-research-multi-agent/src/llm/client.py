"""OpenAI 兼容 LLM 客户端（`src.llm` 包的实际实现主体）。

行为契约
--------
* 配置了 `OPENAI_API_KEY` -> 调用 `{OPENAI_BASE_URL}/chat/completions`（任何 OpenAI 兼容
  服务都行：官方、vLLM、Ollama、各类中转网关）。
* 没有 key（或 `MOCK_LLM=1`）-> 直接走 `src/llm/mock.py` 的确定性规则大脑。
* 真机调用出错（网络/额度/鉴权）-> **自动降级**回 mock，并在响应里标记 `degraded=True`，
  保证 demo 与评测永远不因为外部服务而中断。

无论走哪条路，返回的都是 `LLMResponse`，`data` 字段都是同一套结构化 JSON。

层次与职责
----------
本模块属于「模型访问层」，是 `src.llm` 对上层唯一的出入口：
* 上游调用方：`src/agents/*.py`（经 `AgentContext.llm`）、`src/orchestrator.py`（装配并读 `stats()`）；
* 下游依赖：`src/config.py`（模式与凭据判定）、`src/llm/prompts.py`（提示词）、
  `src/llm/mock.py`（降级大脑）、`src/utils/digest.py`（提示词指纹）；
* 本模块不做业务计算，只负责「组装 prompt -> 拿回 JSON -> 包装成 LLMResponse」。

离线可复现的三道保险
--------------------
1. **入口判定**：`use_mock_llm()` 只要发现没有 key 就直接置为 mock，不尝试联网；
2. **延迟依赖**：`openai` 包与凭据都在 `_ensure_client()` 里才导入/创建，导入本模块无副作用；
3. **兜底降级**：真机路径上的任何异常都被 `chat()` 捕获，转交给 mock 大脑并打上 `degraded=True`，
   因此 `python -m src.demo` 与 `pytest` 在完全离线的机器上也能跑完全链路。

被谁调用
--------
`src/orchestrator.py` 的 `ResearchPipeline` 构造 `LLMClient`；
五个 Agent 通过 `self.ctx.llm.chat(<任务名>, payload)` 使用；`tests/test_mock_llm.py`
直接对 `LLMClient(force_mock=True)` 做断言。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..config import RuntimeConfig, runtime_config, use_mock_llm
from ..utils.digest import digest
from . import mock as mock_brain
from .prompts import AGENT_PROMPTS, build_user_prompt

__all__ = ["LLMClient", "LLMResponse"]

# 任务名 -> 提示词键名（保持任务名与 mock 派发键一致）
# 键必须与 mock._DISPATCH 的任务名一致（plan/retrieve/analyze/risk_review/write），
# 值必须存在于 prompts.AGENT_PROMPTS；新增任务时这三处要同步，否则会静默拿到空 system prompt
_PROMPT_KEYS = {
    "plan": "planner",
    "retrieve": "retriever",
    "analyze": "analyst",
    "risk_review": "risk_checker",
    "write": "writer",
}

# 匹配 markdown 代码围栏（```json ... ``` 或裸 ``` ... ```），用于剥离模型爱加的包装
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class LLMResponse:
    """一次 LLM 调用的完整结果。

    职责：
        把「真机返回」与「mock 返回」抹平成同一个对象，使上层 Agent 完全不需要
        `if mock:` 分支；同时携带足够的元信息用于 trace、评测与问题定位。

    关键属性：
        task:           任务名（plan/retrieve/analyze/risk_review/write），与 _PROMPT_KEYS 键一致
        data:           结构化 JSON 结果（dict），真机与 mock 同构
        raw_text:       模型原始文本；mock 模式下是 `json.dumps(data, sort_keys=True)` 的确定性文本
        model:          实际使用的模型标识；mock 模式下形如 "mock::<model_name>"
        mocked:         本次是否由确定性规则大脑产出
        degraded:       是否因真机调用失败而**被动降级**（mocked=True 且 degraded=True 表示降级路径）
        latency_ms:     本次调用耗时（毫秒，真机含网络时间）
        prompt_digest:  user prompt 的短指纹（12 位），用于比对提示词是否变化
        system_digest:  system prompt 的短指纹
        usage:          token 用量（真机才有；mock 与保守路径下为空 dict）
        error:          降级原因，形如 "TimeoutError: ..."；正常路径为 None

    状态流转：
        由 `chat()` / `_mock_response()` 构造后即基本只读，唯一例外是降级路径：
        `chat()` 在 mock 结果上再回填 `degraded=True` 与 `error`。`to_trace()` 只读取它。
    """

    task: str
    data: Dict[str, Any]
    raw_text: str = ""
    model: str = ""
    mocked: bool = True
    degraded: bool = False
    latency_ms: float = 0.0
    prompt_digest: str = ""
    system_digest: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_trace(self) -> Dict[str, Any]:
        """trace 里记录的精简视图（不落全量 prompt，避免日志膨胀）。

        参数：无（读取自身字段）。
        返回：
            Dict[str, Any] —— 只含元信息的扁平字典，latency_ms 四舍五入到 3 位小数；
            刻意**不含** data 与 raw_text，防止把大段模型输出写进每一步的 trace。
        副作用/异常：
            纯函数，不修改自身；无异常。
        """
        return {
            "task": self.task,
            "model": self.model,
            "mocked": self.mocked,
            "degraded": self.degraded,
            "latency_ms": round(self.latency_ms, 3),
            "prompt_digest": self.prompt_digest,
            "system_digest": self.system_digest,
            "usage": self.usage,
            "error": self.error,
        }


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出里稳妥地抽出 JSON 对象。

    参数：
        text: 模型返回的原始文本，可能带 ```json 围栏、前后寒暄或多余解释。
    返回：
        Optional[Dict[str, Any]] —— 解析成功且顶层是对象时返回该 dict；
        空输入、非对象（如数组/字符串）、完全无法解析时返回 None。
    副作用/异常：
        纯函数，不联网；内部两次 `json.loads` 的 `JSONDecodeError` 都被吞掉，
        因此**不会抛异常**，把「格式不合法」的判断权交给调用方（调用方会据此降级）。
    抽取顺序：
        1) 优先取 ``` 围栏内的内容（若存在围栏）；
        2) 退一步截取第一个 `{` 到最后一个 `}` 之间（容忍围栏写法异常或前后废话）。
    """
    if not text:
        return None
    candidate = text.strip()
    block = _JSON_BLOCK.search(candidate)
    if block:
        candidate = block.group(1).strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    # 退一步：截取第一个 { 到最后一个 }（模型偶发把 JSON 混在解释文字里时仍能救回）
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


class LLMClient:
    """统一 LLM 入口。

    职责：
        为五个 Agent 提供同一套 `chat(task, payload) -> LLMResponse` 接口，
        内部决定走真机还是 mock，并累计调用计数供 `stats()` / trace 使用。

    关键属性：
        config:          RuntimeConfig 快照（构造时会把 `mock_llm` 回写成实际模式）
        _mock:           本实例是否走确定性规则大脑（一旦确定，生命周期内不变）
        _client:         惰性创建的 `openai.OpenAI` 句柄；mock 模式下永远为 None
        call_count:      `chat()` 被调用的总次数（真机 + mock + 降级都计入）
        mock_count:      实际由 mock 大脑产出结果的次数（含被动降级）
        degraded_count:  真机调用抛异常而被迫降级的次数

    状态流转：
        构造 -> （可选）首次真机调用时创建底层客户端 -> 逐次调用累加计数；
        任一真机调用失败都会把该次结果切换为 mock 并在响应上打 `degraded=True`，
        但**不会**改变 `_mock`（即下一次仍会尝试真机，避免一次网络抖动就把整轮降级到底）。
    """

    def __init__(self, config: Optional[RuntimeConfig] = None, force_mock: Optional[bool] = None) -> None:
        """构造客户端（此时不联网、不导入 openai 包）。

        参数：
            config:     运行配置快照；为 None 时调用 `runtime_config()` 取当前环境配置。
            force_mock: 强制指定模式；None 表示按 `use_mock_llm()` 自动判定
                        （无 key 或 MOCK_LLM=1 即 mock），True/False 则以其布尔值为准。
                        测试用 `LLMClient(force_mock=True)` 锁定离线行为。
        返回：
            None。
        副作用/异常：
            会**回写** `self.config.mock_llm`（使 trace 里的配置与实际行为一致）；
            只做赋值与计时器初始化，不联网、不抛异常（openai 未安装也不影响构造）。
        """
        self.config = config or runtime_config()
        self._mock = use_mock_llm() if force_mock is None else bool(force_mock)
        # 回写配置，保证 trace 里记录的 mock_llm 与实际执行路径一致
        self.config.mock_llm = self._mock
        self._client: Any = None  # 延迟创建，避免无 key 环境导入即报错
        self.call_count = 0
        self.mock_count = 0
        self.degraded_count = 0

    # ------------------------------------------------------------------
    @property
    def is_mock(self) -> bool:
        """本实例是否处于确定性 mock 模式。

        参数：无。
        返回：
            bool —— True 表示全部走 mock 大脑；注意降级（degraded）不改变本属性。
        副作用/异常：无。
        """
        return self._mock

    @property
    def model_name(self) -> str:
        """用于展示与 trace 的模型名。

        参数：无。
        返回：
            str —— mock 模式下形如 "mock::<config.model_name>"（一眼可辨），
            真机模式下就是 `config.model_name`。
        副作用/异常：无。
        """
        return self.config.model_name if not self._mock else f"mock::{self.config.model_name}"

    def _ensure_client(self) -> Any:
        """按需创建 OpenAI 客户端。

        参数：无。
        返回：
            Any —— 底层 `openai.OpenAI` 实例（同一实例会被复用缓存到 `self._client`）。
        副作用/异常：
            首次调用会导入 `openai` 包并实时读取凭据（`current_api_key()` /
            `current_base_url()`），因此**改环境变量后新建的客户端才生效**；
            若 `openai` 未安装或凭据非法会抛异常，由调用方 `chat()` 捕获并降级。
        """
        if self._client is not None:
            return self._client
        from openai import OpenAI  # 延迟导入

        from ..config import current_api_key, current_base_url

        # 60s 超时：给慢网关留余量，同时避免整条流水线被单次调用拖死
        self._client = OpenAI(api_key=current_api_key(), base_url=current_base_url(), timeout=60.0)
        return self._client

    # ------------------------------------------------------------------
    def chat(
        self,
        task: str,
        payload: Dict[str, Any],
        *,
        json_mode: bool = True,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """执行一次结构化 LLM 调用。

        参数：
            task:        任务名，取值应为 plan/retrieve/analyze/risk_review/write
                         （用于选 system prompt 与 mock 派发；未知名会拿到空 system prompt，
                         mock 侧则返回 error 结构）。
            payload:     结构化输入 dict，会被渲染成 user prompt 并原样交给 mock 大脑。
            json_mode:   真机模式下是否请求 `response_format={"type":"json_object"}`；
                         对 mock 无影响（mock 永远返回 dict）。
            temperature: 真机模式的采样温度；默认 0.0 以求可复现。
        返回：
            LLMResponse —— 三种来源之一：
            1) mock 模式：`mocked=True`、`degraded=False`；
            2) 真机成功：`mocked=False`、`degraded=False`、带 usage；
            3) 真机失败降级：`mocked=True`、`degraded=True`、`error` 记录异常摘要。
        副作用/异常：
            累加 `call_count`（以及 `mock_count` / `degraded_count`）；
            真机失败时**不向外抛异常**，一律降级为 mock 结果，保证离线/弱网可跑完；
            绝不修改传入的 payload。
        """
        started = time.perf_counter()
        self.call_count += 1

        # 提示词按任务名查表；即使走 mock 也照样构造并算指纹，保证 trace 里的提示词可评审
        system = AGENT_PROMPTS.get(_PROMPT_KEYS.get(task, task), "")
        user = build_user_prompt(task, payload)
        system_digest = digest(system, 12)
        prompt_digest = digest(user, 12)

        if self._mock:
            return self._mock_response(task, payload, started, system_digest, prompt_digest)

        try:
            client = self._ensure_client()
            kwargs: Dict[str, Any] = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
            }
            # 仅在需要时携带 JSON 模式参数：部分兼容网关不认这个字段，会直接报错
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            completion = client.chat.completions.create(**kwargs)
            text = completion.choices[0].message.content or ""
            data = _extract_json(text)
            # 模型答非 JSON 视为调用失败，走统一降级路径而不是把脏数据漏给上层
            if data is None:
                raise ValueError("模型未返回合法 JSON")
            usage = {}
            if getattr(completion, "usage", None) is not None:
                usage = {
                    "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(completion.usage, "completion_tokens", None),
                    "total_tokens": getattr(completion.usage, "total_tokens", None),
                }
            return LLMResponse(
                task=task,
                data=data,
                raw_text=text,
                model=self.config.model_name,
                mocked=False,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                prompt_digest=prompt_digest,
                system_digest=system_digest,
                usage=usage,
            )
        except Exception as exc:  # noqa: BLE001 - 任何外部异常都必须降级而不是中断
            self.degraded_count += 1
            # data 复用 mock 结果：结构与成功路径完全一致，上层无需感知降级
            response = self._mock_response(task, payload, started, system_digest, prompt_digest)
            response.degraded = True
            response.error = f"{type(exc).__name__}: {exc}"
            return response

    # ------------------------------------------------------------------
    def _mock_response(
        self,
        task: str,
        payload: Dict[str, Any],
        started: float,
        system_digest: str,
        prompt_digest: str,
    ) -> LLMResponse:
        """走确定性规则大脑。

        参数：
            task:          任务名（转发给 `mock.run_mock` 派发）。
            payload:       结构化输入，mock 大脑只依赖它做判断。
            started:       本次调用开始时的 `time.perf_counter()` 读数，用于算耗时。
            system_digest: system prompt 指纹（原样带进响应，保证 trace 同构）。
            prompt_digest: user prompt 指纹（同上）。
        返回：
            LLMResponse —— `mocked=True`、`model="mock::<model_name>"`、
            `latency_ms` 为真实计时（并非固定 0，因此无 API Key 也能观测到耗时）；
            `usage` 保持空 dict（无 token 概念）。
        副作用/异常：
            累加 `mock_count`；副作用仅此一处——**不联网、不写文件、不用随机数/时间戳**，
            同一份 payload 的 `data` 与 `raw_text` 逐字节可复现。
        """
        self.mock_count += 1
        data = mock_brain.run_mock(task, payload)
        # sort_keys 让同一 dict 的文本表示唯一，便于断言与 trace 比对
        text = json.dumps(data, ensure_ascii=False, sort_keys=True)
        return LLMResponse(
            task=task,
            data=data,
            raw_text=text,
            model=self.model_name,
            mocked=True,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            prompt_digest=prompt_digest,
            system_digest=system_digest,
        )

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """汇总本实例的调用统计（供 trace / demo 汇总区展示）。

        参数：无。
        返回：
            Dict[str, Any] —— 键为：
            "calls"（chat 总次数）、"mock_calls"（mock 大脑产出次数，含降级）、
            "degraded_calls"（被动降级次数）、"mode"（"mock" 或 "openai-compatible"）、
            "model"（`model_name`）。注意 "mode" 反映的是配置模式，
            降级次数需看 "degraded_calls" 单独判断。
        副作用/异常：
            纯读取，不修改计数器；无异常。
        """
        return {
            "calls": self.call_count,
            "mock_calls": self.mock_count,
            "degraded_calls": self.degraded_count,
            "mode": "mock" if self._mock else "openai-compatible",
            "model": self.model_name,
        }
