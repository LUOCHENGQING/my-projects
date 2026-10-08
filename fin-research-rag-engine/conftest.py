"""pytest 根 conftest：确保项目根目录在 sys.path 上，`import src.*` 可用。

职责边界（与 `tests/conftest.py` 分工）
--------------------------------------
    conftest.py（本文件）   只做「环境前置」：让 `import src.*` 在任意工作目录下都能成立
    tests/conftest.py       只做「测试夹具」：语料 / 切分 / 检索器 / 引擎等共享对象

为什么要单独放一份在根目录：`python -m pytest` 与 `pytest tests/` 两种调用方式下，
rootdir 与 sys.path 的初始内容不同；把根目录插进 sys.path 后，`tests/` 里的
`from src.xxx import ...` 不依赖调用时的工作目录，本地与 CI 表现一致。

为什么插在 `sys.path[0]`（insert 而非 append）：同名模块若已存在于环境中，
项目内的 `src` 必须优先命中，否则会串到别的项目或已安装的同名包上。

被谁依赖：`tests/` 下全部测试模块、`eval/run_eval.py`（后者自带一份同样的 sys.path 处理）。
覆盖策略：本文件不含测试用例，因此没有正常 / 边界 / 异常路径之分；它只保证
「导入可用」这一前置条件成立，属于测试基础设施而非被测行为。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 用 __file__ 反推项目根，而不是依赖 cwd：pytest 可能从任意目录启动
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
