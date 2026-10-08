"""控制台编码处理。

Windows 下 Python 默认按系统 locale（cp936/GBK）编码标准输出，被管道 / 重定向捕获时
中文会乱码。这里在「非交互式输出」场景下强制切到 UTF-8，保证 demo / eval / serve
的输出在任何终端里都可读。

在 RAG 全链路中的位置
--------------------
它不是 RAG 链路的一环，而是**所有 CLI 入口的第一行副作用**：
`src/demo.py`、`src/serve.py`、`eval/run_eval.py` 都在 `main()` 开头调用一次，
之后再打印任何中文都不会乱码。库代码（answer / retrieve / index 等）**不应调用它**，
否则会污染宿主的 stdout 配置。

对外只有一个函数：`ensure_utf8_console()`，无参、无返回、失败静默。
"""

from __future__ import annotations

import sys

__all__ = ["ensure_utf8_console"]


def ensure_utf8_console() -> None:
    """在非 TTY（被管道 / 文件捕获）时把 stdout/stderr 切到 UTF-8。

    TTY 场景保持原样：Windows 控制台本身通过宽字符 API 输出，不会乱码，
    强行改编码反而可能让部分旧终端显示异常。

    参数：无。
    返回：None。
    副作用：把**非 TTY** 的 `sys.stdout` / `sys.stderr` 重配为 UTF-8 且 `errors="replace"`
          （遇到不可编码字符替换而不是抛 `UnicodeEncodeError`）；TTY 流与 None 流跳过。
    异常：无——不支持 `reconfigure` 的流（被替换过的包装对象、StringIO）会抛错并被静默忽略。
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
