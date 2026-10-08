"""stdlib 兜底服务：没装 FastAPI 也能把接口跑起来。

为什么需要它：演示 / 评审环境经常是「一台干净的机器 + 一个 Python」，
如果起个服务要先 `pip install fastapi uvicorn`，那这套东西就永远停在"看 README"。

`python -m src.serve` 只依赖标准库，暴露与 `src.api` 完全相同的路由与返回结构，
因此前端 / 调用方在两种模式下**不需要改任何代码**。

两种服务形态的关系
------------------
    路由与请求/响应契约的唯一真相源是 `src/api.py`（该模块 docstring 里有完整表格：
    GET /health、GET /stats、POST /ask、POST /search、POST /feedback 的请求体字段、
    响应字段与 400/404 的触发条件）。本模块**不复制业务逻辑**，只做 HTTP 装卸：
    解析请求行与 JSON body → 调 `src.api` 的纯函数 handler → 序列化回 JSON。
    注：实际实现为——HTTP 400 由本模块判定（看 handler 返回的 `status == "error"`），
    而 FastAPI 侧同一份错误体仍是 HTTP 200；这是两种形态唯一的行为差异，见 api.py 注释。

进程模型与并发
--------------
    传输层用 `http.server.ThreadingHTTPServer`，每个请求一个线程；
    引擎是**进程内惰性单例**（`_ENGINE` + `_ENGINE_LOCK` 双检锁），首次请求才建索引，
    因此多个线程可能同时等在建索引那一步上，但只会建一次。
    `make_handler(engine=...)` 允许测试注入现成引擎，跳过单例与建索引。

输入 / 输出
-----------
    输入：命令行 `--host/--port`（默认取 `config.API_HOST` / `config.API_PORT`）；
    输出：进程内 HTTP 服务（默认 http://127.0.0.1:8000）+ 终端打印的规模信息；
    被谁调用：`python -m src.serve`（`__main__` → `main()` → `run()`）、
              `tests/test_engine_api.py`（`make_handler(engine)` 直接驱动 handler）。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from .api import handle_ask, handle_feedback, handle_health, handle_search, handle_stats
from .config import API_HOST, API_PORT
from .engine import RAGEngine
from .utils.console import ensure_utf8_console

__all__ = ["run", "make_handler"]

_ENGINE: Optional[RAGEngine] = None
_ENGINE_LOCK = threading.Lock()


def _engine() -> RAGEngine:
    """惰性构造引擎：首次请求时才建索引，避免服务启动时长时间无响应。

    参数：无（配置来自 `src.config`，资料目录取默认 DATA_DIR）。
    返回：进程内唯一的 `RAGEngine`（模块级 `_ENGINE`）。
    副作用：首次调用会读资料并建三路索引 + FAQ 索引，耗时集中在这一次。
    异常：建索引失败会把异常抛给第一个请求的调用方，并**不缓存失败结果**——
          下一次请求会重新尝试（`_ENGINE` 仍为 None）。
    注：双重检查 + `_ENGINE_LOCK` 是为了并发首请求下只建一次索引。
    """
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = RAGEngine.build()
    return _ENGINE


def make_handler(engine: Optional[RAGEngine] = None):
    """构造 HTTP handler。传入 engine 便于测试时复用同一个索引。

    参数：engine 现成引擎；None 时请求期走模块级单例 `_engine()`。
    返回：`BaseHTTPRequestHandler` 子类（类对象，交给 `ThreadingHTTPServer` 实例化）。
    副作用：无（此时不建索引、不监听端口）；`engine` 通过闭包被 Handler 捕获。
    异常：无。
    """

    class Handler(BaseHTTPRequestHandler):
        """每个 HTTP 请求一个实例：`_read_json()` → 路由 → `_send()` JSON。

        状态流转：`do_GET` / `do_POST` 解析路径 → 命中则调 `src.api` 的 handler 并 `_send`，
        未命中则 `_send` 404；处理完即请求结束（HTTP/1.0 风格短连接，无会话状态）。
        关键属性：`server_version = "FinRAG/1.0"`（响应头里的服务标识）；`engine` 为闭包变量。
        """

        server_version = "FinRAG/1.0"

        # ---- 工具 ----
        def _send(self, payload: Dict[str, Any], code: int = 200) -> None:
            """把 dict 作为 JSON 响应写回（UTF-8 中文不转义）。

            参数：payload 响应体（须可 `json.dumps`）；code HTTP 状态码，默认 200。
            返回：None。副作用：写响应头 Content-Type/Content-Length 并输出 body。
            异常：payload 不可序列化会抛 TypeError（本项目的 handler 都经 `to_plain()` 净化，正常不会发生）。
            """
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, Any]:
            """读取并解析请求体 JSON。

            参数：无（从 `Content-Length` 与 `rfile` 读）。
            返回：dict；长度为 0、非 UTF-8、非法 JSON 或顶层不是对象时一律返回 {}，
                  由下游 handler 按"缺参数"处理成 400，而不是在这里抛 500。
            副作用：消费请求体（不可重复读）。异常：无。
            """
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {}
            return data if isinstance(data, dict) else {}

        def _core(self) -> RAGEngine:
            """取本次请求要用的引擎：优先用 `make_handler(engine=...)` 注入的实例。

            返回：RAGEngine。副作用：注入为空时会触发 `_engine()` 的惰性建索引。
            """
            return engine if engine is not None else _engine()

        # ---- 路由 ----
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 的命名约定
            """GET 路由：`/health` 与 `/` 走存活探针，`/stats` 走体检，其余 404。

            参数：无（路径取 `self.path`，查询串被 `urlparse` 丢弃）。
            返回：None。副作用：发响应。异常：无（未知路径也是正常 404 响应）。
            """
            path = urlparse(self.path).path
            core = self._core()
            if path in ("/health", "/"):
                self._send(handle_health(core))
                return
            if path == "/stats":
                self._send(handle_stats(core))
                return
            self._send({"status": "error", "error": f"未知路径：{path}", "code": 404}, 404)

        def do_POST(self) -> None:  # noqa: N802
            """POST 路由：`/ask`、`/search`、`/feedback`，其余 404。

            参数：无。返回：None。副作用：发响应；`/ask` 会写轨迹与缓存。
            异常：无。注：`/ask` 与 `/search` 会把 handler 返回的 `status == "error"`
                  翻译成 HTTP 400（FastAPI 形态下同一错误是 200 + body.code，见 api.py）。
            """
            path = urlparse(self.path).path
            body = self._read_json()
            core = self._core()
            if path == "/ask":
                payload = handle_ask(core, body)
                self._send(payload, 400 if payload.get("status") == "error" else 200)
                return
            if path == "/search":
                payload = handle_search(core, body)
                self._send(payload, 400 if payload.get("status") == "error" else 200)
                return
            if path == "/feedback":
                self._send(handle_feedback(core, body))
                return
            self._send({"status": "error", "error": f"未知路径：{path}", "code": 404}, 404)

        def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover - 静音默认日志
            """覆盖基类的访问日志：演示场景不刷屏（真实部署应改为接入日志系统）。

            参数：fmt 日志格式串；*args 格式参数。返回：None。副作用：无（刻意不输出）。
            """
            return

    return Handler


def run(host: str = API_HOST, port: int = API_PORT, engine: Optional[RAGEngine] = None) -> None:
    """启动兜底 HTTP 服务。

    参数：host 监听地址（默认 config.API_HOST=127.0.0.1）；port 监听端口（默认 config.API_PORT=8000）；
          engine 可选现成引擎，None 时用单例（**注意这里会先在主线程建索引再监听**）。
    返回：None——`serve_forever()` 阻塞，Ctrl-C 后打印关闭信息并 `server_close()` 释放端口。
    副作用：建索引、占端口、打印启动横幅（含资料库规模与路由清单）。
    异常：端口被占用抛 OSError；KeyboardInterrupt 被捕获（正常退出路径）。
    """
    ensure_utf8_console()
    core = engine or _engine()
    print(f"金融智研引擎已启动：http://{host}:{port}")
    print(f"  资料库：{len(core.documents)} 篇文档 / {len(core.children)} 个子块 / {len(core.parents)} 个父块")
    print("  路由：GET /health, GET /stats, POST /ask, POST /search, POST /feedback")
    server = ThreadingHTTPServer((host, port), make_handler(core))
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - 手工中断
        print("\n正在关闭……")
    finally:
        server.server_close()


def main(argv: Optional[list] = None) -> int:  # pragma: no cover - 手工启动
    """`python -m src.serve` 的命令行入口。

    参数：argv 参数列表（None 时用 `sys.argv[1:]`，测试可显式传入）。
    返回：int 退出码，正常关闭恒为 0（`__main__` 里作为 SystemExit 抛出）。
    副作用：解析 `--host` / `--port`（默认取 config.API_HOST / API_PORT），随后进入 `run()` 阻塞。
    异常：未知参数由 argparse 抛 SystemExit(2)。
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m src.serve", description="金融智研引擎 HTTP 服务（stdlib 兜底）")
    parser.add_argument("--host", default=API_HOST)
    parser.add_argument("--port", type=int, default=API_PORT)
    args = parser.parse_args(argv)
    run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
