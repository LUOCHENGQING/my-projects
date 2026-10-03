"""stdlib 兜底服务：没装 FastAPI 也能把接口跑起来。

为什么需要它：演示 / 评审环境经常是「一台干净的机器 + 一个 Python」，
如果起个服务要先 `pip install fastapi uvicorn`，那这套东西就永远停在"看 README"。

`python -m src.serve` 只依赖标准库，暴露与 `src.api` 完全相同的路由与返回结构，
因此前端 / 调用方在两种模式下**不需要改任何代码**。
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
    """惰性构造引擎：首次请求时才建索引，避免服务启动时长时间无响应。"""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = RAGEngine.build()
    return _ENGINE


def make_handler(engine: Optional[RAGEngine] = None):
    """构造 HTTP handler。传入 engine 便于测试时复用同一个索引。"""

    class Handler(BaseHTTPRequestHandler):
        server_version = "FinRAG/1.0"

        # ---- 工具 ----
        def _send(self, payload: Dict[str, Any], code: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, Any]:
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
            return engine if engine is not None else _engine()

        # ---- 路由 ----
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 的命名约定
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
            return

    return Handler


def run(host: str = API_HOST, port: int = API_PORT, engine: Optional[RAGEngine] = None) -> None:
    """启动兜底 HTTP 服务。"""
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
    import argparse

    parser = argparse.ArgumentParser(prog="python -m src.serve", description="金融智研引擎 HTTP 服务（stdlib 兜底）")
    parser.add_argument("--host", default=API_HOST)
    parser.add_argument("--port", type=int, default=API_PORT)
    args = parser.parse_args(argv)
    run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
