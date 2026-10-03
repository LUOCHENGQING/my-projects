"""LLM 接入层（OpenAI 兼容 + 无 Key 自动 mock）。

设计
----
1. **协议兼容**：使用标准库 `urllib` 直接调用 `/chat/completions`，
   不引入 `openai` SDK，减少依赖面（任何 OpenAI 兼容网关均可直接使用）。
2. **零配置可跑**：未配置 `OPENAI_API_KEY`（或 `FORCE_MOCK_LLM=1`）时，
   自动切到 `MockLLM`（见 `src/mock_brain.py`），demo 与测试一定跑得通。
3. **绝不把决策交给模型**：LLM 只负责把已经算好的结构化事实组织成中文措辞；
   真实调用失败（超时、限流、返回非 JSON）也会自动回退到 mock，并记录 `mode=fallback`。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .mock_brain import TASK_SCHEMAS, mock_compose

#: 默认 base_url / 模型名
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

#: 系统提示：统一约束 LLM 的角色（只做措辞，不做决策）
SYSTEM_PROMPT = (
    "你是一名财富管理投资顾问助手，服务于持牌机构的理财经理。"
    "你只负责把给定的结构化事实组织成严谨、克制、合规的中文措辞。"
    "禁止编造任何未在输入中出现的数字、产品名称或机构名称；"
    "禁止给出任何保证收益的表述；禁止推荐具体真实产品。"
    "必须严格输出 JSON 对象，不得输出任何额外解释文字。"
)


@dataclass(frozen=True)
class LLMConfig:
    """LLM 配置。"""

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.2
    timeout: float = 30.0
    force_mock: bool = False

    @property
    def use_mock(self) -> bool:
        """是否强制使用 mock 模式。"""
        return self.force_mock or not self.api_key.strip()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "LLMConfig":
        """从环境变量读取配置（缺失即视为 mock 模式）。"""
        source = env if env is not None else os.environ
        return cls(
            api_key=str(source.get("OPENAI_API_KEY", "") or "").strip(),
            base_url=str(source.get("OPENAI_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip("/"),
            model=str(source.get("MODEL_NAME", DEFAULT_MODEL) or DEFAULT_MODEL),
            temperature=float(source.get("LLM_TEMPERATURE", 0.2) or 0.2),
            timeout=float(source.get("LLM_TIMEOUT", 30) or 30),
            force_mock=str(source.get("FORCE_MOCK_LLM", "0")).strip() in {"1", "true", "True"},
        )


@dataclass
class LLMResult:
    """一次 `compose` 调用的结果。"""

    task: str
    mode: str  # mock | openai | fallback
    data: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    error: str = ""

    @property
    def used_network(self) -> bool:
        """是否真的发起了网络调用。"""
        return self.mode in {"openai", "fallback"}


class BaseLLM:
    """LLM 抽象基类。"""

    mode = "base"

    def complete(self, system: str, user: str) -> str:
        """返回一段自由文本。"""
        raise NotImplementedError

    def compose(self, task: str, context: Mapping[str, Any]) -> LLMResult:
        """按任务生成结构化文案（子类实现）。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _prompt_for(self, task: str, context: Mapping[str, Any]) -> str:
        """把任务与结构化上下文拼成提示词。"""
        keys = TASK_SCHEMAS.get(task, ("note",))
        return (
            f"任务：{task}\n"
            f"请基于以下结构化事实，生成 JSON 对象，字段必须且只能是：{list(keys)}。\n"
            f"所有数值必须原样引用，不得自行计算或改写。\n"
            f"结构化事实：\n{json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)}"
        )


class MockLLM(BaseLLM):
    """确定性规则大脑（无 Key 时的默认实现）。"""

    mode = "mock"

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig(force_mock=True)

    def complete(self, system: str, user: str) -> str:
        """mock 模式下不生成自由文本，返回固定说明。"""
        return "（mock 模式：未配置 OPENAI_API_KEY，自由文本生成已跳过）"

    def compose(self, task: str, context: Mapping[str, Any]) -> LLMResult:
        """调用确定性生成器。"""
        data = mock_compose(task, context)
        return LLMResult(task=task, mode=self.mode, data=data, raw=json.dumps(data, ensure_ascii=False))


class OpenAICompatibleLLM(BaseLLM):
    """OpenAI 兼容端点客户端（标准库实现）。"""

    mode = "openai"

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def complete(self, system: str, user: str) -> str:
        """调用 /chat/completions 并返回首条回复文本。"""
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            url=f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            raise ValueError("LLM 返回体缺少 choices 字段")
        return str(choices[0].get("message", {}).get("content", ""))

    def compose(self, task: str, context: Mapping[str, Any]) -> LLMResult:
        """调用真实模型并解析 JSON；失败时由上层回退到 mock。"""
        text = self.complete(SYSTEM_PROMPT, self._prompt_for(task, context))
        data = _extract_json(text)
        return LLMResult(task=task, mode=self.mode, data=data, raw=text)


class FallbackLLM(BaseLLM):
    """带降级的 LLM：真实调用失败自动回退到 mock，并显式标记 mode。"""

    def __init__(self, primary: BaseLLM, fallback: MockLLM) -> None:
        self.primary = primary
        self.fallback = fallback
        self.mode = "fallback-ready"

    def complete(self, system: str, user: str) -> str:
        """优先真实模型，异常时回退。"""
        try:
            return self.primary.complete(system, user)
        except Exception:
            return self.fallback.complete(system, user)

    def compose(self, task: str, context: Mapping[str, Any]) -> LLMResult:
        """优先真实模型，任何异常（网络/解析）都回退到确定性 mock。"""
        try:
            result = self.primary.compose(task, context)
        except Exception as exc:  # noqa: BLE001 - 需要兜住所有网络与解析异常
            fallback = self.fallback.compose(task, context)
            fallback.mode = "fallback"
            fallback.error = f"{type(exc).__name__}: {exc}"
            return fallback

        expected = set(TASK_SCHEMAS.get(task, ("note",)))
        if not expected <= set(result.data):
            fallback = self.fallback.compose(task, context)
            fallback.mode = "fallback"
            fallback.error = f"模型返回字段缺失，期望 {sorted(expected)}，实际 {sorted(result.data)}"
            return fallback
        return result


def _extract_json(text: str) -> dict[str, Any]:
    """从模型回复中提取 JSON 对象（容忍 ```json 代码块包裹）。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"模型回复中未找到 JSON 对象：{text[:120]}")
    parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("模型回复的 JSON 不是对象")
    return parsed


def build_llm(config: LLMConfig | None = None) -> BaseLLM:
    """按配置构建 LLM：无 Key 用 mock，有 Key 用真实端点 + mock 兜底。"""
    effective = config or LLMConfig.from_env()
    fallback = MockLLM(effective)
    if effective.use_mock:
        return fallback
    return FallbackLLM(OpenAICompatibleLLM(effective), fallback)


def llm_status(llm: BaseLLM) -> dict[str, str]:
    """LLM 状态摘要（demo / trace 使用）。"""
    config = getattr(llm, "config", None)
    if config is None and isinstance(llm, FallbackLLM):
        config = getattr(llm.primary, "config", None)
    if config is None:
        return {"mode": "mock", "model": "-", "base_url": "-"}
    mode = "mock" if config.use_mock else "openai-compatible"
    return {
        "mode": mode,
        "model": config.model,
        "base_url": config.base_url,
        "api_key": "已配置" if config.api_key else "未配置（自动 mock）",
    }


def load_dotenv(path: Path | str | None = None) -> dict[str, str]:
    """轻量读取 .env（不覆盖已有环境变量），返回本次注入的键值。"""
    env_path = Path(path) if path is not None else Path(__file__).resolve().parent.parent / ".env"
    injected: dict[str, str] = {}
    if not env_path.exists():
        return injected
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            injected[key] = value
    return injected
