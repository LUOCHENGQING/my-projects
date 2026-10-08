"""pytest 根 conftest：确保项目根目录在 sys.path 上，`import src.*` 可用。

层次与职责
----------
这是**测试基础设施**，不属于运行时架构（`src/` 里的任何模块都不会 import 它）。
pytest 会在收集用例时自动加载 rootdir 下的 conftest.py，所以放在项目根目录可以让
`tests/` 下的用例文件即使以「根目录为 cwd」之外的方式启动，也能直接 `import src.*`，
不需要额外安装包或设置 `PYTHONPATH`。

关键行为
--------
* 输入：无（模块级代码，pytest 加载时即执行）。
* 输出/副作用：修改进程级 `sys.path`（把项目根插入到最前面），使 `src`、
  `eval` 等顶层目录可被 import；不产生文件、不联网。
* 幂等：已存在相同路径时不再重复插入，避免 `sys.path` 被反复污染。

被谁使用
--------
`pytest` 本身，覆盖 `tests/` 下所有用例；`tests/conftest.py` 也会做同样的根目录
插入（它的父目录才是根，所以需要各自算一次），两者互不冲突。
"""

from __future__ import annotations

import sys
from pathlib import Path

# __file__ 指向本文件（项目根/conftest.py），.parent 即项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
# 幂等插入：重复运行时不再往 sys.path 里塞重复项
if str(PROJECT_ROOT) not in sys.path:
    # 插到最前面（index 0），保证测试用的是仓库内的 src，而不是环境里同名包
    sys.path.insert(0, str(PROJECT_ROOT))
