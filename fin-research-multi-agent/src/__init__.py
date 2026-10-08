"""金融投研多智能体系统（FinResearch-MAS）源码包。

层次与职责
----------
这是整个运行时的顶层包（架构中的「应用层入口」）。它本身不实现业务逻辑，只声明
包身份与版本号；实际的模块划分如下（由 orchestrator 组装成一张有向图后执行）：

    config      全局配置（路径、检索权重、循环上限、LLM 环境变量）
    state       Blackboard 共享状态定义
    tools       MCP 风格工具注册中心与内置工具
    rag         父子块切分 / BM25 / 哈希向量 / 混合重排检索
    llm         OpenAI 兼容客户端 + 确定性 mock 大脑
    agents      Planner / Retriever / Analyst / RiskChecker / Writer
    engine_*    图执行引擎（LangGraph 主引擎 + 自研兼容降级引擎）
    orchestrator 编排装配、条件边与反思循环
    tracing     每一步一行 JSONL 的可观测记录
    hitl        人机协同（高风险结论暂停等待人工确认）

对外关键对象
------------
* `__version__`  版本号字符串，供 CLI / trace / 打包流程读取；
* `__all__`      仅导出 `__version__`，业务符号一律按子模块路径显式导入
  （例如 `from src.llm.client import LLMClient`），避免顶层 `src` 变成隐式大杂烩。

主要输入输出
------------
导入本包**没有输入、没有输出、没有副作用**：不读环境变量、不建索引、不联网。
所有重量级初始化都发生在 `src.orchestrator.ResearchPipeline(...)` 构造时。

被谁调用
--------
* `src/demo.py`（`python -m src.demo`）、`eval/run_eval.py`、`src/replay.py` 以
  `src.*` 形式导入本包下的各模块；
* `conftest.py`（项目根与 `tests/` 各一份）把项目根目录压进 `sys.path`，
  使 pytest 也能用同一套 `src.*` 导入路径。
"""

from __future__ import annotations

# 语义化版本号：写进 trace 与评测报告，便于对照不同版本的结果漂移
__version__ = "1.0.0"
# 刻意收窄导出面：只暴露版本号，业务类/函数必须从具体子模块导入
__all__ = ["__version__"]
