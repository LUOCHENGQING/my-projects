"""pytest 公共夹具。

会话级夹具缓存语料 / 检索器 / 事实库，避免每个用例重复建索引；
运行级夹具用 tmp_path 隔离轨迹文件，保证测试不污染项目里的 runs/ 目录。

本文件不含测试用例，只提供 tests/ 下各测试模块共用的导入环境与夹具：
- 环境：_force_mock_env（session、autouse）全程强制离线 mock；
- 数据：documents / chunks / retriever / fact_store（session）只建一次索引；
- 运行时：registry / pipeline / agent_ctx（function）每个用例一份干净状态。

夹具与用例的对应关系：
- documents -> test_rag.py::test_documents_are_fictional_and_carry_disclaimer、
  test_citation_traceability.py::test_citations_point_to_real_documents，并被 chunks / fact_store 间接依赖；
- chunks -> test_rag.py 的父子块与重排权重用例（retriever 亦间接依赖）；
- retriever -> test_rag.py 的 BM25 / 混合检索用例，并注入 registry；
- fact_store -> test_rag.py 的 fact_store_* 全部用例，并注入 registry；
- registry -> test_tools_schema.py 全部用例、test_citation_traceability.py 的 cite_source 用例；
- pipeline -> test_citation_traceability.py、test_mock_llm.py、test_risk_loop.py、
  test_state_flow.py 的端到端用例；
- agent_ctx -> test_risk_loop.py 的 gate 判定与循环上限用例。
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
    """提供：全程强制 mock 的运行环境开关。

    作用域：session + autouse —— 自动作用于 tests/ 下所有用例，无需显式声明。
    使用者：全部测试用例（保证无网络、无 API Key 也能复现结果）。
    """
    # 先立起 MOCK_LLM 再删 Key（双保险）：防止本地 .env 里的真实 Key 让用例偷偷走网络
    os.environ["MOCK_LLM"] = "1"
    os.environ.pop("OPENAI_API_KEY", None)
    yield

# 注：项目模块在本文件中部（夹具定义之后）导入，故逐行带 noqa: E402 抑制 import 位置告警。
from src.agents.base import AgentContext  # noqa: E402
from src.config import runtime_config  # noqa: E402
from src.llm.client import LLMClient  # noqa: E402
from src.orchestrator import ResearchPipeline  # noqa: E402
from src.rag import HybridRetriever, build_chunks, build_fact_store, load_documents  # noqa: E402
from src.tools import build_default_registry  # noqa: E402
from src.tracing import TraceRecorder  # noqa: E402


@pytest.fixture(scope="session")
def documents():
    """提供：资料库的全部文档对象（3 份虚构主体的年报 / 季报）。

    作用域：session —— 文档解析较慢，全套用例共用同一份只读数据。
    使用者：test_rag.py 的资料合规用例、test_citation_traceability.py 的引用落盘用例，
    并被 chunks / fact_store / registry 间接依赖。
    """
    return load_documents()


@pytest.fixture(scope="session")
def chunks(documents):
    """提供：(parents, children) 两级切分结果，即父子块结构的最小语料视图。

    作用域：session —— 切块只做一次，子块带公司 / 章节元数据。
    使用者：test_rag.py 的父子块结构与重排权重用例（retriever 亦间接依赖）。
    """
    return build_chunks(list(documents))


@pytest.fixture(scope="session")
def retriever(chunks):
    """提供：默认权重的 HybridRetriever，已建好 BM25 与向量索引。

    作用域：session —— 索引构建成本最高，且检索过程只读、可安全共享。
    使用者：test_rag.py 的 BM25 / 混合检索用例，并由 registry 间接复用。
    """
    parents, children = chunks
    return HybridRetriever(parents, children)


@pytest.fixture(scope="session")
def fact_store(documents):
    """提供：从资料中抽取的结构化事实库（按公司 / 指标 / 年份 / 报告期查询）。

    作用域：session —— 抽取一次即可，查询只读。
    使用者：test_rag.py 的 fact_store_* 用例，并由 registry 间接复用。
    """
    return build_fact_store(list(documents))


@pytest.fixture
def registry(retriever, fact_store, documents):
    """提供：注册了全部业务工具的 ToolRegistry。

    作用域：函数级 —— 每个用例拿到干净的幂等缓存与引用计数器（引用序号从 1 重新开始）。
    使用者：test_tools_schema.py 全部用例、test_citation_traceability.py 的 cite_source 用例。
    """
    return build_default_registry(retriever, fact_store, documents)


@pytest.fixture
def pipeline(tmp_path_factory):
    """提供：一整套 ResearchPipeline 实例（含 registry、mock LLM 与轨迹记录器）。

    作用域：函数级 —— 轨迹写入临时 runs 目录，auto 模式不阻塞交互。
    使用者：test_citation_traceability.py、test_mock_llm.py、test_risk_loop.py、
    test_state_flow.py 的端到端用例。
    """
    # 隔离轨迹文件：用例之间不共享 runs/，也不会污染项目目录
    runs_dir = tmp_path_factory.mktemp("runs")
    return ResearchPipeline(auto=True, runs_dir=runs_dir, quiet=True)


@pytest.fixture
def agent_ctx(registry, tmp_path):
    """提供：给单个 Agent 做单元测试用的运行上下文（registry + mock LLM + 轨迹记录器）。

    作用域：函数级 —— tmp_path 保证各用例的轨迹文件互不干扰。
    使用者：test_risk_loop.py 的 gate 判定与循环上限用例。
    """
    # 单元用例只需要 registry 与轨迹落点，不必起整条 pipeline
    recorder = TraceRecorder("unit-test", runs_dir=tmp_path)
    return AgentContext(
        registry=registry,
        llm=LLMClient(force_mock=True),
        recorder=recorder,
        config=runtime_config(),
    )
