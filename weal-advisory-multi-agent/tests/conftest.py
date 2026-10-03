"""pytest 公共夹具。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DataBundle, load_data  # noqa: E402
from src.llm import LLMConfig, MockLLM  # noqa: E402
from src.observability import Tracer  # noqa: E402
from src.pipeline import AdvisoryPipeline, PipelineConfig  # noqa: E402
from src.versioning import VersionStore  # noqa: E402


@pytest.fixture(scope="session")
def data() -> DataBundle:
    """全量样例数据（会话级复用）。"""
    return load_data()


@pytest.fixture(scope="session")
def mock_llm() -> MockLLM:
    """确定性 mock 大脑。"""
    return MockLLM(LLMConfig(force_mock=True))


@pytest.fixture
def temp_store(tmp_path: Path) -> VersionStore:
    """临时版本链存储。"""
    return VersionStore(tmp_path / "chain.jsonl")


@pytest.fixture
def temp_tracer(tmp_path: Path) -> Tracer:
    """临时 trace。"""
    return Tracer(run_id="test-run", runs_dir=tmp_path / "runs")


@pytest.fixture
def make_pipeline(data: DataBundle, mock_llm: MockLLM, temp_store: VersionStore, tmp_path: Path):
    """流水线工厂夹具。"""

    def _factory(**overrides):
        config = PipelineConfig(
            engine=overrides.pop("engine", "native"),
            auto=overrides.pop("auto", True),
            max_repair_rounds=overrides.pop("max_repair_rounds", 2),
            exempt_rules=overrides.pop("exempt_rules", ()),
            interactive=overrides.pop("interactive", False),
        )
        tracer = Tracer(run_id=overrides.pop("run_id", "test-run"), runs_dir=tmp_path / "runs")
        pipeline = AdvisoryPipeline(
            data, llm=mock_llm, tracer=tracer, store=temp_store, config=config
        )
        return pipeline, tracer

    return _factory
