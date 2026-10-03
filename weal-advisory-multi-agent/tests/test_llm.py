"""LLM 接入层测试：无 Key 自动 mock、降级回退、JSON 解析、任务覆盖。"""

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
    config = LLMConfig(api_key="", force_mock=False)
    assert config.use_mock is True
    assert isinstance(build_llm(config), MockLLM)


def test_force_mock_overrides_api_key():
    config = LLMConfig(api_key="sk-demo", force_mock=True)
    assert config.use_mock is True
    assert isinstance(build_llm(config), MockLLM)


def test_config_from_env_defaults_to_mock(monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "MODEL_NAME", "FORCE_MOCK_LLM"):
        monkeypatch.delenv(key, raising=False)
    config = LLMConfig.from_env({})
    assert config.use_mock is True
    assert config.base_url.endswith("/v1")


def test_config_from_env_reads_values():
    config = LLMConfig.from_env(
        {
            "OPENAI_API_KEY": "sk-test",
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
    context = {"display_name": "示例客户甲", "age": 45, "risk_level": 2}
    first = mock_compose("client_profile_summary", context)
    second = mock_compose("client_profile_summary", context)
    assert first == second
    assert first["summary"]


def test_mock_compose_covers_every_declared_task():
    for task in TASK_SCHEMAS:
        result = mock_compose(task, {})
        assert isinstance(result, dict)
        for key in TASK_SCHEMAS[task]:
            assert key in result


def test_mock_compose_unknown_task_is_safe():
    result = mock_compose("no-such-task", {})
    assert "note" in result


def test_mock_llm_compose_returns_mode_mock():
    result = MockLLM().compose("screening_note", {"universe_size": 3, "included_count": 1})
    assert result.mode == "mock"
    assert result.data["note"]


def test_extract_json_from_plain_and_fenced_text():
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert _extract_json('说明文字 {"a": 3} 结尾') == {"a": 3}


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        _extract_json("完全没有 JSON")


def test_fallback_llm_uses_mock_when_remote_fails(monkeypatch):
    config = LLMConfig(api_key="sk-demo", force_mock=False)
    remote = OpenAICompatibleLLM(config)

    def _boom(self, system, user):  # noqa: ANN001
        raise TimeoutError("模拟网络超时")

    monkeypatch.setattr(OpenAICompatibleLLM, "complete", _boom)
    llm = FallbackLLM(remote, MockLLM(config))
    result = llm.compose("advisor_summary", {"display_name": "示例客户甲"})
    assert result.mode == "fallback"
    assert "模拟网络超时" in result.error
    assert result.data["summary"]


def test_fallback_llm_uses_remote_when_available(monkeypatch):
    config = LLMConfig(api_key="sk-demo", force_mock=False)
    remote = OpenAICompatibleLLM(config)

    def _ok(self, system, user):  # noqa: ANN001
        return json.dumps({"summary": "来自真实模型的摘要"}, ensure_ascii=False)

    monkeypatch.setattr(OpenAICompatibleLLM, "complete", _ok)
    llm = FallbackLLM(remote, MockLLM(config))
    result = llm.compose("advisor_summary", {})
    assert result.mode == "openai"
    assert result.data["summary"] == "来自真实模型的摘要"


def test_fallback_llm_falls_back_when_fields_missing(monkeypatch):
    config = LLMConfig(api_key="sk-demo")
    remote = OpenAICompatibleLLM(config)

    def _wrong_fields(self, system, user):  # noqa: ANN001
        return '{"unexpected": "字段"}'

    monkeypatch.setattr(OpenAICompatibleLLM, "complete", _wrong_fields)
    llm = FallbackLLM(remote, MockLLM(config))
    result = llm.compose("advisor_summary", {})
    assert result.mode == "fallback"
    assert "字段缺失" in result.error


def test_llm_status_reports_mock_and_remote():
    assert llm_status(MockLLM())["mode"] == "mock"
    remote_status = llm_status(OpenAICompatibleLLM(LLMConfig(api_key="sk-x", model="m1")))
    assert remote_status["mode"] == "openai-compatible"
    assert remote_status["model"] == "m1"


def test_prompt_contains_schema_keys():
    llm = OpenAICompatibleLLM(LLMConfig(api_key="sk-x"))
    prompt = llm._prompt_for("risk_disclosure", {"risk_level": 3})
    assert "disclosure" in prompt
    assert "risk_level" in prompt


def test_load_dotenv_injects_missing_keys(tmp_path, monkeypatch):
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
    assert injected["QUOTED_KEY_XYZ"] == "world"
    monkeypatch.delenv("DEMO_TEST_KEY_XYZ", raising=False)
    monkeypatch.delenv("QUOTED_KEY_XYZ", raising=False)


def test_load_dotenv_missing_file_is_noop(tmp_path):
    assert load_dotenv(tmp_path / "not-exist.env") == {}


def test_mock_llm_never_touches_network(monkeypatch):
    import urllib.request

    def _forbidden(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("mock 模式不应该发起网络请求")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    llm = build_llm(LLMConfig(api_key="", force_mock=True))
    assert llm.compose("advisor_summary", {"display_name": "示例客户"}).mode == "mock"
