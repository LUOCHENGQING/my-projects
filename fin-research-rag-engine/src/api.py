"""HTTP 接口层。

设计要点：**业务逻辑全部放在纯函数 handler 里，框架只做路由**。

这样做的直接好处是：
    * 没有安装 FastAPI 也能单测全部接口行为（直接调 handler）；
    * 换框架（FastAPI → Flask → 内部 RPC 框架）时不需要重写任何业务逻辑；
    * 线上排查时可以脱离 HTTP 直接复现同一个请求。

接口：
    GET  /health          存活探测 + 引擎规模
    GET  /stats           引擎体检（资料库 / 切分 / 索引 / 缓存 / FAQ）
    POST /ask             问答（含缓存与 FAQ 直出）
    POST /search          只检索不生成（"找依据"场景）
    POST /feedback        记录一次人工反馈（用于后续评测集补充）

所有响应都包含 `traceable` 与引用明细，因为「答案能不能被复核」是金融场景的准入条件。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .engine import RAGEngine

__all__ = [
    "handle_health",
    "handle_stats",
    "handle_ask",
    "handle_search",
    "handle_feedback",
    "create_app",
    "FEEDBACK_LOG",
]

# 进程内的人工反馈缓冲区。生产环境应替换为落库 / 消息队列，
# 这里保留是为了让「反馈 → 补评测集 → 回归」的闭环在演示里可跑通。
FEEDBACK_LOG: List[Dict[str, Any]] = []


def handle_health(engine: RAGEngine) -> Dict[str, Any]:
    """存活探针：返回引擎是否可用以及最基本的规模信息。"""
    return {
        "status": "ok",
        "documents": len(engine.documents),
        "children": len(engine.children),
        "parents": len(engine.parents),
        "faq_entries": len(engine.faq) if engine.faq else 0,
        "llm_mode": engine.llm.mode,
    }


def handle_stats(engine: RAGEngine) -> Dict[str, Any]:
    return {"status": "ok", "stats": engine.stats()}


def handle_ask(engine: RAGEngine, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """问答接口。body: {question, top_k?, expr?, route?, use_cache?, use_faq?, mode?}"""
    body = payload or {}
    question = str(body.get("question", "")).strip()
    if not question:
        return {"status": "error", "error": "缺少 question 参数", "code": 400}

    top_k = body.get("top_k")
    result = engine.ask(
        question,
        top_k=int(top_k) if top_k else None,
        expr=body.get("expr") or None,
        route=body.get("route") or None,
        use_cache=bool(body.get("use_cache", True)),
        use_faq=bool(body.get("use_faq", True)),
        mode=str(body.get("mode", "hybrid")),
    )
    payload_out = result.to_dict()
    payload_out["status"] = "ok"
    return payload_out


def handle_search(engine: RAGEngine, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """检索接口。body: {question, top_k?, expr?, route?, mode?}"""
    body = payload or {}
    question = str(body.get("question", "")).strip()
    if not question:
        return {"status": "error", "error": "缺少 question 参数", "code": 400}

    top_k = body.get("top_k")
    retrieval = engine.search(
        question,
        top_k=int(top_k) if top_k else None,
        expr=body.get("expr") or None,
        route=body.get("route") or None,
        mode=str(body.get("mode", "hybrid")),
    )
    return {
        "status": "ok",
        "question": question,
        "stats": retrieval.stats(),
        "evidence": [e.to_dict(with_context=True) for e in retrieval.evidence],
    }


def handle_feedback(engine: RAGEngine, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """记录人工反馈：哪条答案有问题、问题出在哪一环。"""
    body = payload or {}
    record = {
        "run_id": str(body.get("run_id", "")),
        "question": str(body.get("question", "")),
        "verdict": str(body.get("verdict", "unknown")),      # good / bad / partial
        "stage": str(body.get("stage", "unknown")),          # 解析 / 切分 / 召回 / 重排 / 生成
        "comment": str(body.get("comment", "")),
    }
    FEEDBACK_LOG.append(record)
    return {"status": "ok", "recorded": record, "total": len(FEEDBACK_LOG)}


def create_app(engine: Optional[RAGEngine] = None):
    """构造 FastAPI 应用。未安装 FastAPI 时给出明确的安装指引而不是 ImportError 堆栈。"""
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "未安装 FastAPI：请执行 `pip install fastapi uvicorn`，"
            "或改用 `python -m src.serve`（stdlib 兜底服务，零额外依赖）"
        ) from exc

    app = FastAPI(
        title="金融智研引擎（FinRAG-Engine）",
        description="多源金融资料的解析 / 父子块切分 / 三路混合召回 / 重排 / 可溯源问答",
        version="1.0.0",
    )
    core = engine or RAGEngine.build()

    class AskRequest(BaseModel):
        question: str
        top_k: Optional[int] = None
        expr: Optional[str] = None
        route: Optional[str] = None
        use_cache: bool = True
        use_faq: bool = True
        mode: str = "hybrid"

    class SearchRequest(BaseModel):
        question: str
        top_k: Optional[int] = None
        expr: Optional[str] = None
        route: Optional[str] = None
        mode: str = "hybrid"

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return handle_health(core)

    @app.get("/stats")
    def stats() -> Dict[str, Any]:
        return handle_stats(core)

    @app.post("/ask")
    def ask(req: AskRequest) -> Dict[str, Any]:
        return handle_ask(core, req.model_dump())

    @app.post("/search")
    def search(req: SearchRequest) -> Dict[str, Any]:
        return handle_search(core, req.model_dump())

    @app.post("/feedback")
    def feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
        return handle_feedback(core, payload)

    app.state.engine = core
    return app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover - 手动启动
    """用 uvicorn 启动 FastAPI 服务。"""
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
