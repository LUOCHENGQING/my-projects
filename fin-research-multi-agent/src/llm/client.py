"""OpenAI 兼容 LLM 客户端。

行为契约
--------
* 配置了 `OPENAI_API_KEY` -> 调用 `{OPENAI_BASE_URL}/chat/completions`（任何 OpenAI 兼容
  服务都行：官方、vLLM、Ollama、各类中转网关）。
* 没有 key（或 `MOCK_LLM=1`）-> 直接走 `src/llm/mock.py` 的确定性规则大脑。
* 真机调用出错（网络/额度/鉴权）-> **自动降级**回 mock，并在响应里标记 `degraded=True`，
  保证 demo 与评测永远不因为外部服务而中断。

无论走哪条路，返回的都是 `LLMResponse`，`data` 字段都是同一套结构化 JSON。
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
_PROMPT_KEYS = {
    "plan": "planner",
    "retrieve": "retriever",
    "analyze": "analyst",
    "risk_review": "risk_checker",
    "write": "writer",
}

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class LLMResponse:
    """一次 LLM 调用的完整结果。"""

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
        """trace 里记录的精简视图（不落全量 prompt，避免日志膨胀）。"""
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
    """从模型输出里稳妥地抽出 JSON 对象。"""
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
    # 退一步：截取第一个 { 到最后一个 }
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


class LLMClient:
    """统一 LLM 入口。"""

    def __init__(self, config: Optional[RuntimeConfig] = None, force_mock: Optional[bool] = None) -> None:
        self.config = config or runtime_config()
        self._mock = use_mock_llm() if force_mock is None else bool(force_mock)
        self.config.mock_llm = self._mock
        self._client: Any = None  # 延迟创建，避免无 key 环境导入即报错
        self.call_count = 0
        self.mock_count = 0
        self.degraded_count = 0

    # ------------------------------------------------------------------
    @property
    def is_mock(self) -> bool:
        return self._mock

    @property
    def model_name(self) -> str:
        return self.config.model_name if not self._mock else f"mock::{self.config.model_name}"

    def _ensure_client(self) -> Any:
        """按需创建 OpenAI 客户端。"""
        if self._client is not None:
            return self._client
        from openai import OpenAI  # 延迟导入

        from ..config import current_api_key, current_base_url

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
        """执行一次结构化 LLM 调用。"""
        started = time.perf_counter()
        self.call_count += 1

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
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            completion = client.chat.completions.create(**kwargs)
            text = completion.choices[0].message.content or ""
            data = _extract_json(text)
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
        """走确定性规则大脑。"""
        self.mock_count += 1
        data = mock_brain.run_mock(task, payload)
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
        return {
            "calls": self.call_count,
            "mock_calls": self.mock_count,
            "degraded_calls": self.degraded_count,
            "mode": "mock" if self._mock else "openai-compatible",
            "model": self.model_name,
        }
