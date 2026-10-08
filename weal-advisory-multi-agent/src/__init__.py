"""财富管理投顾多智能体（Wealth Advisory Multi-Agent）。

层次定位
--------
本文件是 `src` 包的**包入口**，位于整个项目的最上层命名空间：它只声明包级元信息，
不做任何导入副作用（注意：包内并不在此处 re-export 子模块，避免 import 顺序与循环依赖问题，
各使用方一律按 `from src.pipeline import ...` 这样的完整路径导入）。

对外暴露
--------
- `__version__`：包版本字符串（当前 "1.0.0"）。
- `__all__`：公开导出清单，仅含 `"__version__"`。
而不暴露任何业务类/函数——业务入口是 `src.pipeline.build_pipeline` / `run_pipeline`
（见 `src/pipeline.py`）与 `src/demo.py`（CLI 演示）。

主要输入 / 输出
---------------
无输入输出：导入本包不产生任何副作用（不读 .env、不读数据文件、不建目录）。

被谁调用
--------
被 `tests/`（例如通过 `import src` 定位包）、`src/demo.py`、`eval/run_eval.py`
以及任何以 `from src.xxx import ...` 方式使用的调用方间接导入。

第三条技术路线：**约束驱动**。
先把客户硬约束求解成可行域，再让 Agent 在可行域内做多目标权衡与解释，
核心机制是「硬约束求解 + 适当性闸门 + 反事实解释 + 情景压力测试 + 建议版本链」。
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
