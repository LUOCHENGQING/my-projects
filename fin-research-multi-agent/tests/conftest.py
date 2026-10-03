"""pytest 公共夹具。

会话级夹具缓存语料 / 检索器 / 事实库，避免每个用例重复建索引；
运行级夹具用 tmp_path 隔离轨迹文件，保证测试不污染项目里的 runs/ 目录。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def _force_mock_env():
    """测试全程强制 mock：不依赖网络、不依赖 API Key，结果可复现。"""
    os.environ["MOCK_LLM"] = "1"
    os.environ.pop("OPENAI_API_KEY", None)
    yield

from src.agents.base import AgentContext  # noqa: E402
from src.config import runtime_config  # noqa: E402
from src.llm.client import LLMClient  # noqa: E402
from src.orchestrator import ResearchPipeline  # noqa: E402
from src.rag import HybridRetriever, build_chunks, build_fact_store, load_documents  # noqa: E402
from src.tools import build_default_registry  # noqa: E402
from src.tracing import TraceRecorder  # noqa: E402


@pytest.fixture(scope="session")
def documents():
    return load_documents()


@pytest.fixture(scope="session")
def chunks(documents):
    return build_chunks(list(documents))


@pytest.fixture(scope="session")
def retriever(chunks):
    parents, children = chunks
    return HybridRetriever(parents, children)


@pytest.fixture(scope="session")
def fact_store(documents):
    return build_fact_store(list(documents))


@pytest.fixture
def registry(retriever, fact_store, documents):
    """函数级：每次拿到干净的幂等缓存与引用计数器。"""
    return build_default_registry(retriever, fact_store, documents)


@pytest.fixture
def pipeline(tmp_path_factory):
    """一次完整的管线（轨迹写入临时目录，auto 模式不阻塞交互）。"""
    runs_dir = tmp_path_factory.mktemp("runs")
    return ResearchPipeline(auto=True, runs_dir=runs_dir, quiet=True)


@pytest.fixture
def agent_ctx(registry, tmp_path):
    """给单个 Agent 做单元测试用的运行上下文。"""
    recorder = TraceRecorder("unit-test", runs_dir=tmp_path)
    return AgentContext(
        registry=registry,
        llm=LLMClient(force_mock=True),
        recorder=recorder,
        config=runtime_config(),
    )
