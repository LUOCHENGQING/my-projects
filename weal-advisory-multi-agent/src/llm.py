"""LLM 接入层（OpenAI 兼容 + 无 Key 自动 mock）。

层级
----
模型接入层（`src/llm.py`）：位于 `src/pipeline.py` 编排层之下，被 Agent 通过
白名单工具间接使用（`narrative.compose_text` 等最终会调用 `BaseLLM.compose`），
向下只依赖标准库与 `src/mock_brain.py`。它**不参与任何业务判断**。

解决的问题
----------
把"有没有模型、模型好不好用"从业务链路里隔离出去：业务数字全部由确定性代码算出，
本模块只负责把结构化事实转成中文措辞，并在模型不可用时仍然产出可用文案。

对外暴露
--------
- `LLMConfig`：不可变配置（含 `api_key` / `base_url` / `model` / `temperature` /
  `timeout` / `force_mock`）与 `LLMConfig.from_env()` 环境变量装载。
- `LLMResult`：一次 `compose` 的结果（`task` / `mode` / `data` / `raw` / `error`）。
- `BaseLLM` / `MockLLM` / `OpenAICompatibleLLM` / `FallbackLLM` 四个实现类。
- `build_llm()`：工厂，按配置决定用哪种实现。
- `llm_status()`：`mode` / `model` / `base_url` / `api_key` 状态摘要，供 demo 与 trace 展示。
- `load_dotenv()`：轻量读取项目根目录 `.env`（不覆盖已有环境变量）。

三种模式如何切换
----------------
模式由 `build_llm()` 一处决定，判据是 `LLMConfig.use_mock`：

1. **mock**：`use_mock` 为真 → 直接返回 `MockLLM`（`mode = "mock"`）。
   `use_mock` = `force_mock` 为真 **或** `api_key` 去空白后为空。
   即：未配置 `OPENAI_API_KEY`，或显式设置 `FORCE_MOCK_LLM=1` 时进入本模式。
   `MockLLM` 不做网络调用、不生成自由文本，`compose` 直接转发给
   `src/mock_brain.py` 的确定性生成器。
2. **openai**：`use_mock` 为假 → 返回 `FallbackLLM(OpenAICompatibleLLM(...), MockLLM(...))`。
   真实调用成功且返回字段齐全时，`LLMResult.mode == "openai"`。
3. **fallback**：真实调用出了任何问题 → 用 mock 兜底并把 `LLMResult.mode` 改写为
   `"fallback"`，同时把异常摘要写入 `LLMResult.error`。

为什么"无 Key 也能跑通"
-----------------------
1. `LLMConfig.use_mock` 把"没 Key"直接判为 mock 模式，`build_llm()` 根本不会构造
   网络客户端，demo / 单测无需网络与密钥即可产出完整建议书；
2. 即使配了 Key，`FallbackLLM` 仍把 `MockLLM` 作为兜底：网络异常、超时、非 JSON、
   JSON 缺字段都会落到确定性 mock；
3. mock 本身是纯函数（见 `src/mock_brain.py`），任何输入都不抛异常。

失败降级路径
------------
`OpenAICompatibleLLM.complete` → `urllib` 请求（`timeout=LLMConfig.timeout`，默认 30s）
→ 解析响应体 `choices`（为空则抛 `ValueError`）→ `_extract_json()` 提取 JSON
（找不到对象或不是对象则抛 `ValueError`，非法 JSON 抛 `json.JSONDecodeError`）
→ 任一环节抛异常都会被 `FallbackLLM.compose` 捕获 → 改用 mock → `mode="fallback"`、
`error` 记录 `"{异常类名}: {异常信息}"`。
除异常外还有一条**字段缺失**降级：模型返回的 JSON 缺少 `TASK_SCHEMAS[task]` 里声明的
任一字段时同样回退 mock，`error` 记录期望字段与实际字段。

被谁调用
--------
`src/pipeline.py`：`build_pipeline()` 调 `load_dotenv()`，`AdvisoryPipeline.__init__`
调 `build_llm()`；`src/demo.py` 调 `llm_status()` 打印模式；`tests/test_llm.py` 直接断言。
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
#: 可用环境变量覆盖：OPENAI_BASE_URL / MODEL_NAME
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

#: 系统提示：统一约束 LLM 的角色（只做措辞，不做决策）
#: 由 OpenAICompatibleLLM.complete 作为 messages[0]（role=system）发送。
SYSTEM_PROMPT = (
    "你是一名财富管理投资顾问助手，服务于持牌机构的理财经理。"
    "你只负责把给定的结构化事实组织成严谨、克制、合规的中文措辞。"
    "禁止编造任何未在输入中出现的数字、产品名称或机构名称；"
    "禁止给出任何保证收益的表述；禁止推荐具体真实产品。"
    "必须严格输出 JSON 对象，不得输出任何额外解释文字。"
)


@dataclass(frozen=True)
class LLMConfig:
    """LLM 配置。

    不可变（`frozen=True`），构造后不可修改，便于在 Agent 之间安全共享。

    关键属性：
        api_key：OpenAI 兼容密钥；为空即视为 mock 模式（见 `use_mock`）。
        base_url：端点根地址，构造时由 `from_env` 去掉尾部 `/`；
            实际请求地址为 `{base_url}/chat/completions`。
        model：模型名。
        temperature：采样温度（默认 0.2，偏确定性措辞）。
        timeout：单次网络请求超时秒数，直接传给 `urllib.request.urlopen`。
        force_mock：强制 mock 开关，优先于 `api_key`。

    被谁使用：`build_llm()`（决定实现类）、`llm_status()`（回显）、
    `MockLLM.__init__`（保存引用）。
    """

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.2
    timeout: float = 30.0
    force_mock: bool = False

    @property
    def use_mock(self) -> bool:
        """是否强制使用 mock 模式。

        返回：
            `force_mock` 为真，或 `api_key` 去空白后为空 → True。
        这是三种模式切换的**唯一判据**（由 `build_llm` 读取）。
        """
        return self.force_mock or not self.api_key.strip()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "LLMConfig":
        """从环境变量读取配置（缺失即视为 mock 模式）。

        参数：
            env：键值映射；为 None 时读取 `os.environ`。
        返回：
            新的 `LLMConfig`。对应关系：`OPENAI_API_KEY` → api_key、
            `OPENAI_BASE_URL` → base_url、`MODEL_NAME` → model、
            `LLM_TEMPERATURE` → temperature（默认 0.2）、`LLM_TIMEOUT` → timeout（默认 30）、
            `FORCE_MOCK_LLM` → force_mock（取值 `"1"` / `"true"` / `"True"` 时为真）。
        异常：环境变量值无法转成 float 时抛 `ValueError`。
        """
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
    """一次 `compose` 调用的结果。

    关键属性：
        task：任务名（`src/mock_brain.py` 的 `TASK_SCHEMAS` key）。
        mode：三种模式之一，取值只能是 `mock` / `openai` / `fallback`。
            `mock` = 未走网络；`openai` = 真实调用成功且字段齐全；
            `fallback` = 真实调用失败或字段缺失，已改用 mock 兜底。
        data：结构化文案字典，键与 `TASK_SCHEMAS[task]` 一致。
        raw：模型原始回复文本（mock 模式下为 `data` 的 JSON 序列化结果）。
        error：降级原因摘要，仅 `mode == "fallback"` 时非空。

    说明：本类的 `mode` 是**结果级**标记，与实现类的类属性 `mode` 不是同一概念
    （`FallbackLLM.mode` 为 `"fallback-ready"`，`BaseLLM.mode` 为 `"base"`）。
    """

    task: str
    mode: str  # mock | openai | fallback
    data: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    error: str = ""

    @property
    def used_network(self) -> bool:
        """是否真的发起了网络调用。

        返回：
            `mode` 为 `openai` 或 `fallback` 时为 True；`mock` 为 False。
            注意：`fallback` 表示"曾尝试并失败于网络路径"，因此计为发起过。
        """
        return self.mode in {"openai", "fallback"}


class BaseLLM:
    """LLM 抽象基类。

    定义统一契约：`complete()` 返回自由文本，`compose()` 返回结构化 `LLMResult`。
    子类必须实现这两个方法，否则抛 `NotImplementedError`。

    被谁使用：`src/pipeline.py` 与各 Agent 只依赖本抽象类型；`build_llm()` 返回其子类。
    """

    mode = "base"

    def complete(self, system: str, user: str) -> str:
        """返回一段自由文本。

        参数：
            system：system 角色提示词。
            user：user 角色提示词。
        返回：
            模型回复文本。
        异常：基类未实现，抛 `NotImplementedError`。
        """
        raise NotImplementedError

    def compose(self, task: str, context: Mapping[str, Any]) -> LLMResult:
        """按任务生成结构化文案（子类实现）。

        参数：
            task：任务名。
            context：结构化事实字典。
        返回：
            `LLMResult`。
        异常：基类未实现，抛 `NotImplementedError`。
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _prompt_for(self, task: str, context: Mapping[str, Any]) -> str:
        """把任务与结构化上下文拼成提示词。

        参数：
            task：任务名；未登记时按 `("note",)` 兜底字段。
            context：结构化事实字典，以 `sort_keys=True` 的 JSON 内联进提示词。
        返回：
            提示词字符串，显式列出"字段必须且只能是 `TASK_SCHEMAS[task]`"并禁止模型自行计算。
        说明：`default=str` 保证不可直接序列化的对象不会导致 `json.dumps` 抛错。
        """
        keys = TASK_SCHEMAS.get(task, ("note",))
        return (
            f"任务：{task}\n"
            f"请基于以下结构化事实，生成 JSON 对象，字段必须且只能是：{list(keys)}。\n"
            f"所有数值必须原样引用，不得自行计算或改写。\n"
            f"结构化事实：\n{json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)}"
        )


class MockLLM(BaseLLM):
    """确定性规则大脑（无 Key 时的默认实现）。

    职责：把 `compose` 直接转发给 `src/mock_brain.py` 的 `mock_compose`，
    不发起网络请求、不做任何随机生成，因此输出逐字节可复现。

    关键属性：
        config：`LLMConfig`；`__init__` 未传入时用 `LLMConfig(force_mock=True)` 兜底。
        mode：类属性固定为 `"mock"`，会原样写进 `LLMResult.mode`。

    被谁使用：`build_llm()` 在 `use_mock` 为真时返回本类；否则作为 `FallbackLLM`
    的兜底实现；`tests/test_llm.py` 直接实例化。
    """

    mode = "mock"

    def __init__(self, config: LLMConfig | None = None) -> None:
        """初始化。

        参数：
            config：LLM 配置；为 None 时使用 `LLMConfig(force_mock=True)`（保证走 mock）。
        """
        self.config = config or LLMConfig(force_mock=True)

    def complete(self, system: str, user: str) -> str:
        """mock 模式下不生成自由文本，返回固定说明。

        参数：
            system / user：仅为保持接口一致而接收，**未被使用**。
        返回：
            固定中文字符串"（mock 模式：未配置 OPENAI_API_KEY，自由文本生成已跳过）"。
        副作用：无（不访问网络）。
        """
        return "（mock 模式：未配置 OPENAI_API_KEY，自由文本生成已跳过）"

    def compose(self, task: str, context: Mapping[str, Any]) -> LLMResult:
        """调用确定性生成器。

        参数：
            task：任务名。
            context：结构化事实字典（只读）。
        返回：
            `LLMResult(mode="mock")`，`data` 为 `mock_compose` 的结果，
            `raw` 为其 JSON 序列化文本，`error` 为空。
        异常：无（`mock_compose` 对未知任务也返回兜底文案）。
        """
        data = mock_compose(task, context)
        return LLMResult(task=task, mode=self.mode, data=data, raw=json.dumps(data, ensure_ascii=False))


class OpenAICompatibleLLM(BaseLLM):
    """OpenAI 兼容端点客户端（标准库实现）。

    用标准库 `urllib` 直接 POST `{base_url}/chat/completions`，不依赖 `openai` SDK，
    因此任何 OpenAI 兼容网关都可直接使用。

    关键属性：
        config：`LLMConfig`，提供 model / temperature / timeout / api_key / base_url。

    被谁使用：仅由 `build_llm()` 包装进 `FallbackLLM` 使用；单测直接实例化以校验提示词。
    """

    mode = "openai"

    def __init__(self, config: LLMConfig) -> None:
        """初始化。

        参数：
            config：LLM 配置（必填，不做校验，空 api_key 会由服务端返回 401）。
        """
        self.config = config

    def complete(self, system: str, user: str) -> str:
        """调用 /chat/completions 并返回首条回复文本。

        参数：
            system：system 角色提示词。
            user：user 角色提示词。
        返回：
            `choices[0].message.content` 的字符串形式。
        副作用：发起一次 HTTPS 请求（`Authorization: Bearer <api_key>`），
            超时由 `self.config.timeout` 控制。
        异常：网络/HTTP 错误由 `urllib` 抛出（`urllib.error.URLError` /
            `HTTPError`、`TimeoutError` 等）；反序列化失败抛 `json.JSONDecodeError`；
            `choices` 为空抛 `ValueError("LLM 返回体缺少 choices 字段")`。
            这些异常都由上层 `FallbackLLM` 统一兜住。
        """
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
        """调用真实模型并解析 JSON；失败时由上层回退到 mock。

        参数：
            task：任务名。
            context：结构化事实字典。
        返回：
            `LLMResult(mode="openai")`，`data` 为解析出的 JSON 对象，`raw` 为原始回复文本。
        副作用：发起一次网络请求；使用固定的 `SYSTEM_PROMPT` 作为 system 提示词。
        异常：本方法**不自行降级**——网络异常与 `_extract_json` 的解析异常都会向上抛出，
            由 `FallbackLLM.compose` 捕获后改用 mock。
        """
        text = self.complete(SYSTEM_PROMPT, self._prompt_for(task, context))
        data = _extract_json(text)
        return LLMResult(task=task, mode=self.mode, data=data, raw=text)


class FallbackLLM(BaseLLM):
    """带降级的 LLM：真实调用失败自动回退到 mock，并显式标记 mode。

    关键属性：
        primary：主实现（通常是 `OpenAICompatibleLLM`）。
        fallback：兜底实现（`MockLLM`）。
        mode：类实例属性，构造后为 `"fallback-ready"`；注意它**不是**结果模式，
            真正的结果模式写在每个 `LLMResult.mode` 上（`openai` / `fallback`）。

    被谁使用：`build_llm()` 在 `use_mock` 为假时构造并返回本类。
    """

    def __init__(self, primary: BaseLLM, fallback: MockLLM) -> None:
        """初始化。

        参数：
            primary：主 LLM。
            fallback：兜底 LLM（mock），必须不是 None，否则降级会二次抛错。
        """
        self.primary = primary
        self.fallback = fallback
        self.mode = "fallback-ready"

    def complete(self, system: str, user: str) -> str:
        """优先真实模型，异常时回退。

        参数：
            system / user：提示词。
        返回：
            主实现正常时返回其文本；抛任何 `Exception` 时返回 `fallback.complete()`
            的固定 mock 说明文本。
        副作用：可能发起网络请求。
        说明：注：本方法**不**记录模式或错误（`complete` 返回的是裸字符串，无载体），
            降级痕迹只在 `compose` 的 `LLMResult` 上可见。
        """
        try:
            return self.primary.complete(system, user)
        except Exception:
            return self.fallback.complete(system, user)

    def compose(self, task: str, context: Mapping[str, Any]) -> LLMResult:
        """优先真实模型，任何异常（网络/解析）都回退到确定性 mock。

        参数：
            task：任务名。
            context：结构化事实字典。
        返回：
            - 主实现成功且字段齐全：`mode="openai"` 的原结果；
            - 主实现抛异常：mock 结果，`mode` 被改写为 `"fallback"`，
              `error` 为 `"{异常类名}: {异常信息}"`；
            - 主实现成功但缺字段（`TASK_SCHEMAS[task]` 的字段未全部出现在
              `result.data` 中）：同样是 `mode="fallback"` 的 mock 结果，
              `error` 为期望字段与实际字段的对比说明。
        副作用：可能发起网络请求；失败路径会消耗一次确定性 mock 生成（无外部副作用）。
        """
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
    """从模型回复中提取 JSON 对象（容忍 ```json 代码块包裹）。

    参数：
        text：模型原始回复。
    返回：
        解析出的 `dict`。
    处理步骤：去掉首尾空白 → 若以 ``` 开头则去掉围栏与可选的 `json` 语言标记
    → 取**第一个 `{` 到最后一个 `}`** 之间的子串 → `json.loads`。
    异常：
        未找到 `{` 或 `}`（或 `}` 在 `{` 之前）抛 `ValueError`；
        子串不是合法 JSON 抛 `json.JSONDecodeError`；
        解析结果不是对象（如数组、字符串）抛 `ValueError("模型回复的 JSON 不是对象")`。
    说明：这些异常都由 `FallbackLLM.compose` 兜住，触发 mock 降级。
    """
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
    """按配置构建 LLM：无 Key 用 mock，有 Key 用真实端点 + mock 兜底。

    参数：
        config：LLM 配置；为 None 时调用 `LLMConfig.from_env()` 从环境变量装载。
    返回：
        - `config.use_mock` 为真 → `MockLLM`（`mode = "mock"`，不走网络）；
        - 否则 → `FallbackLLM(OpenAICompatibleLLM(config), MockLLM(config))`，
          真实调用失败时结果 `mode` 为 `"fallback"`。
    副作用：可能读取环境变量；构造本身不发起网络请求（请求发生在 `complete`/`compose` 时）。
    """
    effective = config or LLMConfig.from_env()
    fallback = MockLLM(effective)
    if effective.use_mock:
        return fallback
    return FallbackLLM(OpenAICompatibleLLM(effective), fallback)


def llm_status(llm: BaseLLM) -> dict[str, str]:
    """LLM 状态摘要（demo / trace 使用）。

    参数：
        llm：任意 `BaseLLM` 实现。
    返回：
        状态字典。取不到配置时返回 `{"mode": "mock", "model": "-", "base_url": "-"}`；
        否则含 `mode`（取值为 `"mock"` 或 `"openai-compatible"`）、`model`、
        `base_url`、`api_key`（`"已配置"` 或 `"未配置（自动 mock）"`，
        **仅回显是否配置，不泄露密钥内容**）。
    探测顺序：先取 `llm.config`；没有则（当 `llm` 是 `FallbackLLM` 时）取
    `llm.primary.config`。
    说明：注：此处的 `mode` 词表（`mock` / `openai-compatible`）与
    `LLMResult.mode` 的词表（`mock` / `openai` / `fallback`）不同，前者用于展示、
    后者用于标记单次调用是否走了网络与降级。
    """
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
    """轻量读取 .env（不覆盖已有环境变量），返回本次注入的键值。

    参数：
        path：`.env` 文件路径；为 None 时默认取项目根目录（`src/` 的上一级）下的 `.env`。
    返回：
        `{键: 值}`，只包含**本次真正注入**的键；文件不存在、或所有键都已存在于
        `os.environ` 时返回空字典（不抛异常）。
    副作用：**直接写入 `os.environ`**（故 `build_pipeline()` 只需在启动时调用一次），
        已存在的环境变量一律跳过，因此不会覆盖外部显式配置。
    解析规则：跳过空行、`#` 开头的注释行、以及不含 `=` 的行；
        值会去掉首尾空白并剥掉一层成对的单/双引号。
    """
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
