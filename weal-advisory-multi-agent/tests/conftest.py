"""pytest 公共夹具（conftest）：为 tests/ 下全部测试模块提供共享依赖。

本文件做两件事
--------------
1. **准备导入路径**：把项目根目录插入 `sys.path`，保证无论从哪个工作目录
   启动 pytest、或直接执行单个测试文件，`from src.xxx import ...` 都能解析
   （与 pytest.ini 的 `pythonpath = .` 形成双保险）。
2. **供给夹具**：全量样例数据、确定性 mock 大脑、临时版本链 / 追踪器，
   以及流水线工厂 `make_pipeline`。

作用域取舍
----------
数据装载与 mock 大脑构建成本较高、且对外只读，因此用 `session` 作用域复用
同一份实例；凡涉及写盘的夹具（版本链、trace、流水线）一律使用默认的
`function` 作用域并落在 `tmp_path` 下，避免测试之间通过文件系统相互污染。

注：`src.*` 的导入必须发生在 `sys.path` 注入之后，故这批导入刻意放在文件中部
并标注 `# noqa: E402`（module level import not at top of file）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 项目根目录 = tests/ 的上一级；后续所有 `src.*` / `eval.*` 导入都依赖它进入 sys.path
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
    """提供全量样例数据 `DataBundle`（客户 / 产品 / 问卷 / 压力情景）。

    作用域：`session` —— 数据来自 `data/*.json`，且 `DataBundle` 是冻结的，
    多个测试模块复用同一实例既省开销又不会互相污染。
    使用者：本仓库中绝大多数数据驱动的测试模块（`test_agents.py`、
    `test_constraints.py`、`test_counterfactual.py`、`test_data_hygiene.py`、
    `test_engine_parity.py`、`test_eval.py` 等）。
    注意：`DataBundle.client()` 返回的是深拷贝，调用方需要收紧约束时
    不必担心污染内部缓存。
    """
    return load_data()


@pytest.fixture(scope="session")
def mock_llm() -> MockLLM:
    """提供强制 mock 的确定性"大脑"（不联网、输出可复现）。

    作用域：`session` —— `MockLLM` 无状态、构建一次即可复用。
    使用者：`test_agents.py`（构造五个 Agent）与 `test_engine_parity.py`
    （双引擎对照），以及本文件的 `make_pipeline` 工厂。
    `force_mock=True` 保证即使开发机配置了真实 `OPENAI_API_KEY`，
    测试也不会发起网络调用。
    """
    return MockLLM(LLMConfig(force_mock=True))


@pytest.fixture
def temp_store(tmp_path: Path) -> VersionStore:
    """提供落在 `tmp_path` 下的临时版本链存储。

    作用域：默认 `function` —— 版本链会写盘，必须每个测试一份，避免跨用例串链。
    使用者：本文件的 `make_pipeline` 工厂（流水线构造需要 `store` 参数）。
    """
    return VersionStore(tmp_path / "chain.jsonl")


@pytest.fixture
def temp_tracer(tmp_path: Path) -> Tracer:
    """提供落在 `tmp_path` 下的临时 `Tracer`（`run_id` 固定为 `test-run`，便于断言）。

    作用域：默认 `function` —— trace 按 run 落盘，需逐用例隔离。
    说明：`make_pipeline` 会自行构造 Tracer，因此当前仓库里没有测试直接引用
    本夹具；它保留给"只想要一个干净 tracer"的场景。
    """
    return Tracer(run_id="test-run", runs_dir=tmp_path / "runs")


@pytest.fixture
def make_pipeline(data: DataBundle, mock_llm: MockLLM, temp_store: VersionStore, tmp_path: Path):
    """提供流水线工厂：按关键字覆盖配置，返回 `(pipeline, tracer)` 二元组。

    作用域：默认 `function`。工厂把常用参数（引擎 / 自动模式 / 最大修复轮数 /
    豁免规则 / 交互模式）以 `overrides.pop(..., 默认值)` 的形式抽出，因此调用方
    只需传想改的那一项，例如 `make_pipeline(max_repair_rounds=0)`；
    未被识别的关键字会继续留在 `overrides` 中（当前实现未再使用）。
    `tracer` 与 `pipeline` 一起返回，便于测试直接断言 trace 步骤与 Agent 序列。

    使用者：`test_pipeline.py`（端到端流水线测试）。
    """

    def _factory(**overrides):
        """按 overrides 构造一条全新的流水线（复用同一 mock 大脑与临时 store）。"""
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
