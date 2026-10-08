"""控制台编码处理（utils 层，零业务依赖）。

层次与职责：
    最底层工具模块，仅被「进程入口」调用一次：src/demo.py、src/replay.py、
    eval/run_eval.py 在 main 起始处调用 ensure_utf8_console()，避免中文输出乱码。

Windows 下 Python 默认按系统 locale（cp936/GBK）编码标准输出，被管道 / 重定向
捕获时容易出现中文乱码。这里在「非交互式输出」场景下强制切到 UTF-8，
保证 demo / replay / eval 的输出在任何终端里都是可读的中文。
"""

from __future__ import annotations

import sys


def ensure_utf8_console() -> None:
    """在非 TTY（被管道 / 文件捕获）时把 stdout/stderr 切到 UTF-8。

    参数：无。
    返回值：None。
    副作用：就地修改 sys.stdout / sys.stderr 的编码（调用其 reconfigure）。
    异常：不向外抛出——任何流的异常都被吞掉，编码问题不应影响主流程。

    TTY 场景保持原样：Windows 控制台本身通过宽字符 API 输出，不会乱码，
    强行改编码反而可能让部分旧终端显示异常。
    """
    # 只处理这两个标准流：库内部的其它日志流不在本函数的职责范围内
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is None:
                continue
            # 交互式终端（isatty 为 True）跳过；用 getattr 兜底，兼容 isatty 缺失的替代流对象
            if getattr(stream, "isatty", lambda: False)():
                continue
            # errors="replace"：即便有无法编码的字符，也宁可显示成替代符，也不要让程序崩掉
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            # 某些被替换过的流对象不支持 reconfigure，静默跳过即可，不影响主流程
            pass
