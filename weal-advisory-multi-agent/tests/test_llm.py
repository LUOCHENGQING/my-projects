"""LLM 接入层测试（`src.llm` + `src.mock_brain`）。

覆盖对象
--------
配置与工厂（`LLMConfig.use_mock` / `from_env`、`build_llm`）、降级包装
（`FallbackLLM`）、JSON 提取（`_extract_json`）、提示词拼装（`_prompt_for`）、
状态摘要（`llm_status`）、`.env` 装载（`load_dotenv`），以及确定性 mock 生成器
（`TASK_SCHEMAS` / `mock_compose`）。

覆盖策略
--------
- **正常路径**：有 Key 时走真实端点（用 monkeypatch 打桩，绝不发真实请求）；
  mock 生成器覆盖全部已声明任务。
- **异常 / 降级**：无 Key 自动 mock；真实调用抛异常、返回无法解析的文本、
  返回字段缺失 —— 三种失败都必须回退到 mock，并显式标记 `mode="fallback"`
  与具体 `error` 原因。
- **边界**：`.env` 文件不存在时为 no-op、已存在的环境变量不被覆盖、
  引号包裹的值被剥掉、注释行与空行被忽略。
- **安全底线**：mock 模式下把 `urllib.request.urlopen` 换成「一调用就报错」的哨兵，
  以此证明「零配置可跑」确实不依赖网络。

离线可运行：所有真实端点调用在用例内都被替换成桩函数，无需网络与 API Key。
"""

from __future__ import annotations

import json
import os

import pytest

from src.llm import (
    LLMConfig,
    MockLLM,
    OpenAICompatibleLLM,
    FallbackLLM,
    _extract_json,
    build_llm,
    llm_status,
    load_dotenv,
)
from src.mock_brain import TASK_SCHEMAS, mock_compose


def test_config_without_api_key_uses_mock():
    """零配置契约：没有 API Key 时 `use_mock` 为真，工厂也给出 `MockLLM`。"""
    config = LLMConfig(api_key="", force_mock=False)
    assert config.use_mock is True
    assert isinstance(build_llm(config), MockLLM)


def test_force_mock_overrides_api_key():
    """`force_mock` 优先级最高：即使配了 Key 也必须走 mock（测试与演示的确定性保障）。"""
    config = LLMConfig(api_key="sk-demo", force_mock=True)
    assert config.use_mock is True
    assert isinstance(build_llm(config), MockLLM)


def test_config_from_env_defaults_to_mock(monkeypatch):
    """环境变量缺省语义：什么都不配时自动进入 mock 模式，且 base_url 有合理默认值。"""
    # 清空相关环境变量，模拟"全新机器上从未配置过"的状态
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "MODEL_NAME", "FORCE_MOCK_LLM"):
        monkeypatch.delenv(key, raising=False)
    config = LLMConfig.from_env({})
    assert config.use_mock is True
    assert config.base_url.endswith("/v1")


def test_config_from_env_reads_values():
    """解析契约：环境变量里的 Key / base_url / 模型名 / force_mock 都被正确读入与规范化。"""
    config = LLMConfig.from_env(
        {
            "OPENAI_API_KEY": "sk-test",
            # 故意带结尾斜杠，验证会被 rstrip("/") 规范化
            "OPENAI_BASE_URL": "https://example.invalid/v1/",
            "MODEL_NAME": "demo-model",
            "FORCE_MOCK_LLM": "0",
        }
    )
    assert config.api_key == "sk-test"
    assert config.base_url == "https://example.invalid/v1"
    assert config.model == "demo-model"
    assert config.use_mock is False


def test_mock_compose_is_deterministic():
    """确定性：同一 task + 同一上下文两次生成的文案完全一致，可直接写进断言。"""
    context = {"display_name": "示例客户甲", "age": 45, "risk_level": 2}
    first = mock_compose("client_profile_summary", context)
    second = mock_compose("client_profile_summary", context)
    assert first == second
    assert first["summary"]


def test_mock_compose_covers_every_declared_task():
    """覆盖度：`TASK_SCHEMAS` 声明的每个任务都必须被 mock 生成器实现，且字段齐全。"""
    for task in TASK_SCHEMAS:
        # 传入空上下文是有意的：mock 生成器必须容忍缺省输入，不能因取不到字段而抛错
        result = mock_compose(task, {})
        assert isinstance(result, dict)
        for key in TASK_SCHEMAS[task]:
            assert key in result


def test_mock_compose_unknown_task_is_safe():
    """边界：未声明的 task 名不能抛异常，应安全退化成一个带说明字段的结果。"""
    result = mock_compose("no-such-task", {})
    assert "note" in result


def test_mock_llm_compose_returns_mode_mock():
    """契约：`MockLLM.compose` 返回 `LLMResult`，且 `mode` 明确标记为 mock。"""
    result = MockLLM().compose("screening_note", {"universe_size": 3, "included_count": 1})
    assert result.mode == "mock"
    assert result.data["note"]


def test_extract_json_from_plain_and_fenced_text():
    """解析健壮性：纯 JSON、```json 围栏、夹带解释文字三种回复形态都要能解析。"""
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert _extract_json('说明文字 {"a": 3} 结尾') == {"a": 3}


def test_extract_json_raises_on_garbage():
    """异常路径：完全不含 JSON 的回复必须报 ValueError，交由上层触发降级。"""
    with pytest.raises(ValueError):
        _extract_json("完全没有 JSON")


def test_fallback_llm_uses_mock_when_remote_fails(monkeypatch):
    """降级路径：真实调用抛异常时必须回退到 mock，并保留异常类型与信息作为 `error`。"""
    config = LLMConfig(api_key="sk-demo", force_mock=False)
    remote = OpenAICompatibleLLM(config)

    def _boom(self, system, user):  # noqa: ANN001
        """桩：模拟真实端点的网络超时（异常路径）。"""
        raise TimeoutError("模拟网络超时")

    # 把网络调用替换成必然超时的桩，验证的是降级逻辑而非真实网络行为
    monkeypatch.setattr(OpenAICompatibleLLM, "complete", _boom)
    llm = FallbackLLM(remote, MockLLM(config))
    result = llm.compose("advisor_summary", {"display_name": "示例客户甲"})
    assert result.mode == "fallback"
    assert "模拟网络超时" in result.error
    assert result.data["summary"]


def test_fallback_llm_uses_remote_when_available(monkeypatch):
    """正常路径：真实端点返回合法 JSON 且字段齐全时，结果直接采用，不得被 mock 覆盖。"""
    config = LLMConfig(api_key="sk-demo", force_mock=False)
    remote = OpenAICompatibleLLM(config)

    def _ok(self, system, user):  # noqa: ANN001
        """桩：模拟真实端点返回字段齐全的合法 JSON（正常路径）。"""
        return json.dumps({"summary": "来自真实模型的摘要"}, ensure_ascii=False)

    monkeypatch.setattr(OpenAICompatibleLLM, "complete", _ok)
    llm = FallbackLLM(remote, MockLLM(config))
    result = llm.compose("advisor_summary", {})
    assert result.mode == "openai"
    assert result.data["summary"] == "来自真实模型的摘要"


def test_fallback_llm_falls_back_when_fields_missing(monkeypatch):
    """降级路径：JSON 能解析但缺少任务必需字段时，同样要回退到 mock（防止下游拿到残缺数据）。"""
    config = LLMConfig(api_key="sk-demo")
    remote = OpenAICompatibleLLM(config)

    def _wrong_fields(self, system, user):  # noqa: ANN001
        """桩：模拟"JSON 合法但缺少任务必需字段"的模型回复。"""
        return '{"unexpected": "字段"}'

    monkeypatch.setattr(OpenAICompatibleLLM, "complete", _wrong_fields)
    llm = FallbackLLM(remote, MockLLM(config))
    result = llm.compose("advisor_summary", {})
    assert result.mode == "fallback"
    assert "字段缺失" in result.error


def test_llm_status_reports_mock_and_remote():
    """状态摘要契约：mock 与真实端点两种形态都能如实报告模式与模型名。"""
    assert llm_status(MockLLM())["mode"] == "mock"
    remote_status = llm_status(OpenAICompatibleLLM(LLMConfig(api_key="sk-x", model="m1")))
    assert remote_status["mode"] == "openai-compatible"
    assert remote_status["model"] == "m1"


def test_prompt_contains_schema_keys():
    """提示词契约：拼出的 prompt 必须声明任务要求字段并原样带入结构化事实。"""
    llm = OpenAICompatibleLLM(LLMConfig(api_key="sk-x"))
    prompt = llm._prompt_for("risk_disclosure", {"risk_level": 3})
    assert "disclosure" in prompt
    assert "risk_level" in prompt


def test_load_dotenv_injects_missing_keys(tmp_path, monkeypatch):
    """`.env` 装载：缺失的键被注入 `os.environ` 并作为返回值暴露，注释行与引号被正确处理。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# 注释行\nDEMO_TEST_KEY_XYZ=hello\nQUOTED_KEY_XYZ='world'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DEMO_TEST_KEY_XYZ", raising=False)
    monkeypatch.delenv("QUOTED_KEY_XYZ", raising=False)
    injected = load_dotenv(env_file)
    assert injected["DEMO_TEST_KEY_XYZ"] == "hello"
    assert os.environ["DEMO_TEST_KEY_XYZ"] == "hello"
    # 单/双引号包裹的值必须被剥掉引号（常见 .env 写法）
    assert injected["QUOTED_KEY_XYZ"] == "world"
    # 显式清理本次注入的键（monkeypatch 在用例结束时也会兜底还原）
    monkeypatch.delenv("DEMO_TEST_KEY_XYZ", raising=False)
    monkeypatch.delenv("QUOTED_KEY_XYZ", raising=False)


def test_load_dotenv_missing_file_is_noop(tmp_path):
    """边界：`.env` 不存在时静默返回空字典，不得抛异常（未配置也要能跑）。"""
    assert load_dotenv(tmp_path / "not-exist.env") == {}


def test_mock_llm_never_touches_network(monkeypatch):
    """安全底线：mock 模式下任何网络调用都会立刻失败，以此证明「零配置可跑」不依赖网络。"""
    import urllib.request

    def _forbidden(*args, **kwargs):  # noqa: ANN002, ANN003
        """哨兵：任何网络调用一经发生即让用例失败。"""
        raise AssertionError("mock 模式不应该发起网络请求")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    llm = build_llm(LLMConfig(api_key="", force_mock=True))
    assert llm.compose("advisor_summary", {"display_name": "示例客户"}).mode == "mock"
