"""控制台编码处理。

Windows 下 Python 默认按系统 locale（cp936/GBK）编码标准输出，被管道 / 重定向
捕获时容易出现中文乱码。这里在「非交互式输出」场景下强制切到 UTF-8，
保证 demo / replay / eval 的输出在任何终端里都是可读的中文。
"""

from __future__ import annotations

import sys


def ensure_utf8_console() -> None:
    """在非 TTY（被管道 / 文件捕获）时把 stdout/stderr 切到 UTF-8。

    TTY 场景保持原样：Windows 控制台本身通过宽字符 API 输出，不会乱码，
    强行改编码反而可能让部分旧终端显示异常。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is None:
                continue
            if getattr(stream, "isatty", lambda: False)():
                continue
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            # 某些被替换过的流对象不支持 reconfigure，静默跳过即可，不影响主流程
            pass
