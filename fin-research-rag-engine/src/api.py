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
注：实际实现为只有 `POST /ask` 的响应体带 `traceable` 与引用明细；`/health` 与 `/stats`
返回的是引擎规模信息，`/search` 返回的是证据列表（`Evidence.to_dict`），不含 `traceable`。

接口契约（FastAPI 与 stdlib 兜底两种服务形态**完全一致**，调用方不需要区分）
--------------------------------------------------------------------------
    GET  /health    无 body
                    → {status, documents, children, parents, faq_entries, llm_mode}
    GET  /stats     无 body
                    → {status, stats: <engine.stats() 全量体检>}
    POST /ask       body {question, top_k?, expr?, route?, use_cache?, use_faq?, mode?}
                    → AnswerResult.to_dict() 的字段 + {status:"ok"}
    POST /search    body {question, top_k?, expr?, route?, mode?}
                    → {status, question, stats: <RetrievalResult.stats()>,
                       evidence:[Evidence.to_dict(with_context=True)]}
    POST /feedback  body {run_id, question, verdict, stage, comment}
                    → {status, recorded: <规范化后的记录>, total: <缓冲区内条数>}
    未知路径        → {status:"error", error:"未知路径：<path>", code:404}（仅 serve.py 判路由）

    参数缺省与 `RAGEngine.ask()` 保持一致：question 必填；top_k 省略时用引擎的 final_top_k；
    expr / route 省略即 None（由检索层自行路由与推断）；use_cache / use_faq 默认 True；
    mode 默认 "hybrid"（可选 dense / bm25 走对照组）。
    question 缺失或全空白 → {status:"error", error:"缺少 question 参数", code:400}，
    **五个 handler 都不会抛异常**（错误都走返回值），因此两套服务都不需要 try/except。

    注：实际实现为 FastAPI 路由直接把 handler 的返回值当 JSON 响应体，
        HTTP 状态码始终是 200（错误信息在 body 的 code 字段里）；只有 stdlib 兜底服务
        会把 `status == "error"` 翻译成 HTTP 400。调用方若要判断成败，应看 body 的
        status 字段，而不是 HTTP 状态码——这样两种形态的行为才真正等价。

两种服务形态与调用方
--------------------
    有 FastAPI：`create_app()` → `uvicorn.run(...)` / `python -c "from src.api import serve; serve()"`；
    无 FastAPI：`python -m src.serve`（`src/serve.py` 复用本文件的全部 handler）。
    测试直接调 handler（不需要起服务）：`tests/test_engine_api.py`；
    注：实际实现为 `eval/run_eval.py` 并不经过接口层，它直接用 `RAGEngine.ask()/search()`
        跑用例并汇总报告。
    输入：`RAGEngine` 实例 + 请求 dict；输出：可 JSON 序列化的 dict。
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
    """存活探针：返回引擎是否可用以及最基本的规模信息。

    参数：engine 已构造好的 `RAGEngine`（索引已建好，因此这里只是数长度，不做 IO）。
    返回：dict——status 固定 "ok"；documents / children / parents 为资料与块计数；
          faq_entries（无 FAQ 时为 0）；llm_mode（"mock" 或 "openai"，便于一眼看出当前是否离线）。
    副作用/异常：无。
    """
    return {
        "status": "ok",
        "documents": len(engine.documents),
        "children": len(engine.children),
        "parents": len(engine.parents),
        "faq_entries": len(engine.faq) if engine.faq else 0,
        "llm_mode": engine.llm.mode,
    }


def handle_stats(engine: RAGEngine) -> Dict[str, Any]:
    """引擎体检接口：把 `RAGEngine.stats()` 的完整报告包一层状态字段。

    参数：engine 引擎实例。
    返回：{status:"ok", stats:<引擎体检>}——stats 内含 corpus / chunks / index / reranker /
          llm / faq / cache / masked_documents / known_entities / quality / config。
    副作用/异常：无（只读聚合，但 `stats()` 会现场调用各后端的 `describe()`）。
    """
    return {"status": "ok", "stats": engine.stats()}


def handle_ask(engine: RAGEngine, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """问答接口。body: {question, top_k?, expr?, route?, use_cache?, use_faq?, mode?}

    参数：engine 引擎实例；payload 请求体，None 视为空 dict。
    返回：`AnswerResult.to_dict()` 的全部字段（question / answer / mode / run_id / traceable /
          cache_hit / faq_hit / citation_numbers / source_ids / citations / citation_check /
          faithfulness / evidence / retrieval_stats / timings / total_latency_ms / notes）
          再追加 status:"ok"；question 缺失时返回
          {status:"error", error:"缺少 question 参数", code:400}。
    副作用：会写轨迹（runs/<run_id>.jsonl）与缓存（use_cache=True 时），并推进 engine.last_run_id。
    异常：不向调用方抛——底层各组件的失败都在 engine 内被转成 notes / 拒答 / 降级作答。
    """
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
    """检索接口。body: {question, top_k?, expr?, route?, mode?}

    参数：engine 引擎实例；payload 请求体，None 视为空 dict。
    返回：{status:"ok", question, stats:<RetrievalResult.stats()，
          含 mode / recalled / deduplicated / evidence / routes / elapsed_ms / plan>,
          evidence:[`Evidence.to_dict(with_context=True)`]}——**带父块上下文**，
          因为这个接口就是给"我要看依据原文"的场景用的；
          question 缺失时返回 {status:"error", error:"缺少 question 参数", code:400}。
    副作用：无缓存、无轨迹、不改 engine 状态（纯检索，可安全用于 A/B 与压测）。
    异常：不向外抛；无命中时 evidence 为空列表而不是错误。
    """
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
    """记录人工反馈：哪条答案有问题、问题出在哪一环。

    参数：engine 引擎实例（本 handler 不使用它，保留是为了与其余 handler 签名一致、
          让两套服务能共用同一张路由表）；
          payload 请求体 {run_id, question, verdict, stage, comment}，字段缺失一律补 ""。
    返回：{status:"ok", recorded:<补齐后的记录>, total:<FEEDBACK_LOG 当前条数>}；
          verdict 约定 good / bad / partial，stage 约定 解析 / 切分 / 召回 / 重排 / 生成，
          但实现**不做校验**，缺省值分别是 "unknown"，以便先把反馈收上来再谈规范。
    副作用：向进程内全局 `FEEDBACK_LOG` 追加一条（进程重启即丢，生产需换落库/消息队列）。
    异常：无。
    """
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
    """构造 FastAPI 应用。未安装 FastAPI 时给出明确的安装指引而不是 ImportError 堆栈。

    参数：engine 可选的现成引擎（测试里用来复用同一份索引）；None 时本函数内部
          `RAGEngine.build()` 建一份，因此**构造 app 的耗时≈建索引耗时**。
    返回：FastAPI 应用对象，路由即模块 docstring 里的五条；引擎同时挂在 `app.state.engine`
          上，方便外部脚本取用同一个实例。
    副作用：可能加载资料并建索引；注册路由。/docs 有 OpenAPI 文档（title/description/version 已填中文）。
    异常：缺依赖时抛 `RuntimeError`（消息包含 `pip install fastapi uvicorn` 与
          `python -m src.serve` 两条出路），并把原始 ImportError 作为 cause 链上去。
    """
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
        """POST /ask 的请求体契约（字段名即 JSON 键名，默认值与 `engine.ask()` 对齐）。"""

        question: str
        top_k: Optional[int] = None
        expr: Optional[str] = None
        route: Optional[str] = None
        use_cache: bool = True
        use_faq: bool = True
        mode: str = "hybrid"

    class SearchRequest(BaseModel):
        """POST /search 的请求体契约（只检索不生成，故没有 use_cache / use_faq）。"""

        question: str
        top_k: Optional[int] = None
        expr: Optional[str] = None
        route: Optional[str] = None
        mode: str = "hybrid"

    @app.get("/health")
    def health() -> Dict[str, Any]:
        """GET /health 路由：转调 `handle_health(core)`。"""
        return handle_health(core)

    @app.get("/stats")
    def stats() -> Dict[str, Any]:
        """GET /stats 路由：转调 `handle_stats(core)`。"""
        return handle_stats(core)

    @app.post("/ask")
    def ask(req: AskRequest) -> Dict[str, Any]:
        """POST /ask 路由：pydantic 校验后 `model_dump()` 交给 `handle_ask()`。"""
        return handle_ask(core, req.model_dump())

    @app.post("/search")
    def search(req: SearchRequest) -> Dict[str, Any]:
        """POST /search 路由：pydantic 校验后 `model_dump()` 交给 `handle_search()`。"""
        return handle_search(core, req.model_dump())

    @app.post("/feedback")
    def feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST /feedback 路由：反馈字段自由，故直接用原始 dict 而非 pydantic 模型。"""
        return handle_feedback(core, payload)

    app.state.engine = core
    return app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover - 手动启动
    """用 uvicorn 启动 FastAPI 服务。

    参数：host 监听地址（默认 127.0.0.1，只对本机开放）；port 监听端口（默认 8000）。
    返回：None——函数在 `uvicorn.run()` 里阻塞，直到进程被中断。
    副作用：`create_app()` 会先建索引；随后占端口并提供 HTTP 服务。
    异常：未装 uvicorn 时 ImportError；端口被占用时由 uvicorn 抛 OSError。
    """
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
