"""仓库根 conftest：保证 `python -m pytest` 与 `pytest` 两种调用方式都能 import src。

同时暴露 PROJECT_ROOT / DATA_DIR 常量，供各测试模块共用。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
