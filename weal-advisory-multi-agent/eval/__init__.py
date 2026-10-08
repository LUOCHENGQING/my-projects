"""评估脚本包（使 `from eval.run_eval import evaluate` 可用）。

层次定位
--------
本文件是 `eval` 包的**包入口**，与 `src/` 业务代码完全解耦：它不导入任何业务模块，
只把 `eval` 标记为常规包（存在 `__init__.py` 才能用 `from eval.run_eval import ...`
这种带包名的绝对导入，且便于 `pytest tests/test_eval.py` 直接引用评估接口）。

对外暴露
--------
本模块**不 re-export 任何对象**——真正的评估接口在 `eval/run_eval.py` 中
（`evaluate()` / `render_text()` / `main()` 与 `THRESHOLDS`），使用方需显式写
`from eval.run_eval import evaluate`。

主要输入 / 输出
---------------
无：导入本包不读数据、不跑评估、不产生副作用。

被谁调用
--------
被 `tests/test_eval.py` 与命令行 `python eval/run_eval.py` 中的绝对导入间接使用。
"""
