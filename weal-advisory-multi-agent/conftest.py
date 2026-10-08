"""仓库根 conftest：保证 `python -m pytest` 与 `pytest` 两种调用方式都能 import src。

层次定位与作用
--------------
本文件是 pytest **在仓库根自动加载的插件/配置钩子**（pytest 会把 conftest.py 所在目录
加入 rootdir 搜索路径并在收集测试前导入它），因此它承担两件事：

1. **导入路径修正**：把 `PROJECT_ROOT`（本文件所在目录）前置进 `sys.path`。
   pytest 在 rootdir 下运行时并不保证 cwd 一定在 `sys.path` 中，
   这一步保证 `tests/` 里 `from src.xxx import ...` 在两种调用方式下都成立。
   注意 `tests/conftest.py` 会做同样的修正，二者互相独立、互为兜底。
2. **暴露共享常量**：`PROJECT_ROOT` / `DATA_DIR`，供各测试模块直接 `from conftest import DATA_DIR`。

对外暴露
--------
- `PROJECT_ROOT`：仓库根目录（`Path`）。
- `DATA_DIR`：样例数据目录（`PROJECT_ROOT / "data"`）。

主要输入 / 输出
---------------
输入：无（只读取自身位置 `__file__` 推导路径）。
输出：无返回值；副作用是**修改进程级 `sys.path`**（幂等：已在其中时不重复插入）。

注意：本文件只放路径与常量，**不定义任何 fixture**——所有 fixture（`data` / `mock_llm` /
`temp_store` / `temp_tracer` / `make_pipeline`）都定义在 `tests/conftest.py` 中。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
