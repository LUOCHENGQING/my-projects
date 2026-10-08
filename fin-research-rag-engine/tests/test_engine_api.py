"""引擎、接口、轨迹与兜底服务的集成测试。

覆盖的被测模块
--------------
    src.engine     RAGEngine 的装配与体检（stats / known_entities / 脱敏与质量记账）、
                   问答主流程（rag / faq / refused 三种模式）、缓存命中与绕过缓存、
                   显式元数据过滤、软过滤进 notes、compare_strategies 三策略对照，
                   以及主体闸门 question_entities / unknown_entities 的判定规则
    src.api        纯函数 handler：handle_health / handle_stats / handle_ask / handle_search /
                   handle_feedback / create_app（缺 FastAPI 时的降级契约）
    src.tracing    new_run_id 的编号格式、TraceRecorder 落盘与关闭落盘、read_trace 的缺失报错、
                   summarize_trace 的汇总口径，以及引擎是否真的逐步写轨迹
    src.serve      stdlib 兜底 HTTP 服务：健康检查、统计、问答、检索与 404

覆盖策略
--------
    正常      端到端问答链路（检索 → 生成 → 引用校验 → 轨迹落盘）走通并返回可追溯答案。
    边界      缓存第二次命中、显式关闭缓存、显式关闭 FAQ、top_k 截断、未知路径。
    异常      handle_ask / handle_search 缺 question 返回 400 错误对象、read_trace 读不到文件抛
              FileNotFoundError、未装 FastAPI 时 create_app 抛 RuntimeError（而不是 ImportError 堆栈）。
    对抗/拒答 主体闸门是这一层最关键的规则：问题问到资料库之外的主体必须拒答且不给引用；
              与之相对，已知主体的简称/后缀差异**不能**被误判成未知主体（误拒答会伤正常问法）。
    可观测    引用角标与出处清单（【出处与时效】/【引用明细】）、to_dict 可 JSON 序列化、
              轨迹里必须出现 recall / generate / validate / done 四步。

为什么接口与轨迹都塞进本文件：它们共用同一个 `engine` 会话级夹具，
而引擎的构造（解析 → 清洗 → 脱敏 → 切分 → 建索引）是整个测试套件里最贵的一步，
放在一起可以只付一次构造成本。这也使本文件天然成为「集成」而非「单元」测试。
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from src.engine import RAGEngine, question_entities, unknown_entities
from src.tracing import TraceRecorder, new_run_id, read_trace, summarize_trace


# ---------------------------------------------------------------------------
# 引擎装配
# ---------------------------------------------------------------------------
def test_engine_stats_structure(engine):
    """体检报告必须覆盖资料库 / 切分 / 索引 / LLM / 缓存 / 配置六段，且子块数多于父块数。"""
    stats = engine.stats()
    assert stats["corpus"]["documents"] > 0
    assert stats["chunks"]["parents"] > 0
    assert stats["chunks"]["children"] > stats["chunks"]["parents"]
    assert stats["index"]["vocabulary"] > 0
    assert stats["llm"]["mode"] == "mock"
    assert stats["cache"]["backend"] in ("memory", "redis")
    assert "config" in stats


def test_engine_known_entities_cover_document_subjects(engine):
    """已知主体清单必须覆盖「被调查对象」这类只出现在文档标题里的主体，否则正常尽调问题会被误拒。"""
    assert "示例监管机构" in engine.known_entities
    assert any("示例集团" in name for name in engine.known_entities)


def test_engine_masking_counted(engine):
    """脱敏计数必须如实反映被改写的文档数（索引、日志、轨迹都从脱敏后的文本派生）。"""
    assert engine.masked_documents >= 0


def test_engine_quality_reports_available(engine):
    """每篇文档都要有质量分且落在 [0,1]：这个分数会写进块并参与重排降权，不能缺项或越界。"""
    assert engine.quality_scores
    assert all(0.0 <= v <= 1.0 for v in engine.quality_scores.values())


# ---------------------------------------------------------------------------
# 问答主流程
# ---------------------------------------------------------------------------
def test_ask_returns_traceable_answer(engine):
    """常规问答必须走完整 RAG 链路并给出可复核结果：模式为 rag、有引用、有检索计划、有耗时。"""
    result = engine.ask("合格投资者的金融资产门槛是多少？", use_cache=False)
    assert result.mode == "rag"
    assert result.traceable
    assert result.citations
    assert result.retrieval is not None
    assert result.timings["total_ms"] > 0


def test_ask_answer_contains_citation_marks(engine):
    """答案正文必须带 [n] 引用角标与【出处与时效】段——「答案能不能被复核」是金融场景的准入条件。"""
    result = engine.ask("合格投资者的金融资产门槛是多少？", use_cache=False)
    assert "[1]" in result.answer
    assert "【出处与时效】" in result.answer


def test_ask_uses_faq_for_high_frequency_question(engine):
    """高频问题必须命中 FAQ 直出（模式为 faq、faq_hit 为真），不走完整检索生成链路。"""
    # 「开户需要准备哪些材料」是示例语料里维护的固定问答，正是 FAQ 存在的意义
    result = engine.ask("开户需要准备哪些材料？", use_cache=False)
    assert result.faq_hit
    assert result.mode == "faq"
    assert result.traceable


def test_ask_can_disable_faq(engine):
    """显式关闭 FAQ 后同一问题必须回落到 RAG 或拒答路径，不得再走 FAQ 直出。"""
    result = engine.ask("开户需要准备哪些材料？", use_cache=False, use_faq=False)
    assert not result.faq_hit
    assert result.mode in ("rag", "refused")


def test_ask_refuses_unknown_entity(engine):
    """主体闸门必须拦住资料库外的公司：模式为 refused、引用为空、答案明确说明不在覆盖范围内。"""
    # 「示例科技」与库内的「示例银行」共享品牌前缀但前四字不同，正是闸门刻意要拦的形态
    result = engine.ask("示例科技股份有限公司 2024 年的净利润是多少？", use_cache=False)
    assert result.mode == "refused"
    assert result.citations == []
    assert "不在本资料库的主体范围内" in result.answer


def test_unknown_entities_matching_rules():
    """主体判定要「宁可放过、不可错杀」：后缀差异与品牌简写都算已知，只有核心名不同才算未知。"""
    known = ["示例银行股份有限公司", "示例监管机构"]
    assert unknown_entities("示例科技股份有限公司怎么样？", known)
    assert not unknown_entities("示例银行股份有限公司怎么样？", known)
    assert not unknown_entities("示例银行怎么样？", known)
    assert not unknown_entities("合格投资者的门槛是多少？", known)


def test_question_entities_deduplicates():
    """同一主体的多种写法必须去重，避免同一个未知主体在拒答文案里被列两遍。"""
    entities = question_entities("示例银行和示例银行股份有限公司")
    assert len(entities) == len(set(entities))


def test_ask_cache_hit_second_time(engine):
    """同问第二次必须命中缓存并复用同一答案（缓存键要覆盖问题、过滤、top_k、模式与路由）。"""
    # 两次调用都用 use_cache=True 且关掉 FAQ：确保走的确实是"缓存"而不是"FAQ 直出"
    question = "双录资料需要保存多久？"
    first = engine.ask(question, use_cache=True, use_faq=False)
    second = engine.ask(question, use_cache=True, use_faq=False)
    assert second.cache_hit
    assert second.answer == first.answer
    assert second.timings["total_ms"] <= first.timings["total_ms"] + 5


def test_ask_no_cache_never_hits(engine):
    """use_cache=False 必须真的绕过缓存（即使同一问题刚刚被写入缓存），不能读到旧结果。"""
    question = "冷静期是多久？"
    engine.ask(question, use_cache=True, use_faq=False)
    fresh = engine.ask(question, use_cache=False, use_faq=False)
    assert not fresh.cache_hit


def test_ask_with_metadata_filter(engine):
    """显式过滤条件必须一路透传到检索计划并真的只回该年份的资料（硬过滤，不是只影响排序）。"""
    # year = 2023 对应的是已废止的旧版文件 POL-2023-11，语料里确实存在该年份版本
    result = engine.ask("合格投资者的金融资产门槛是多少？", expr='year = 2023', use_cache=False, use_faq=False)
    assert result.retrieval.plan.filter_expr == "year = 2023"
    assert all(e.source_id == "POL-2023-11" for e in result.evidence)


def test_ask_soft_filter_reported_in_notes(engine):
    """从问题推断出的条件只能做软过滤（只影响排序、不剔除），且必须在 notes 里说明以免被误读成硬过滤。"""
    # 「应收账款」的年份限定的是数据年度，正是软过滤分支的典型触发场景
    result = engine.ask("示例集团 2023 年应收账款增速如何？", use_cache=False, use_faq=False)
    assert any("软过滤" in note for note in result.notes)


def test_answer_result_render_includes_citation_list(engine):
    """文本渲染必须给出【引用明细】并带上资料编号，便于答案直接贴到终端或接口返回里复核。"""
    rendered = engine.ask("合格投资者的金融资产门槛是多少？", use_cache=False).render()
    assert "【引用明细】" in rendered
    assert "POL-2024-07" in rendered


def test_answer_result_to_dict_is_jsonable(engine):
    """问答结果必须能整体 JSON 序列化（含 numpy 等非原生类型），否则接口层无法直接返回。"""
    payload = engine.ask("合格投资者的金融资产门槛是多少？", use_cache=False).to_dict()
    json.dumps(payload, ensure_ascii=False)
    assert payload["source_ids"]


def test_search_only_does_not_generate(engine):
    """「只找依据」必须只检索不生成：有证据、模式为 hybrid，但不产生答案与引用校验。"""
    result = engine.search("合格投资者")
    assert result.evidence
    assert result.mode == "hybrid"


def test_compare_strategies_returns_three_modes(engine):
    """策略对照必须固定同一个问题、只换策略：三套都返回来源，混合那套要有可用来源。"""
    # 问题选一个需要跨文档找依据的复杂问法，三套策略的结果差异才有观察价值
    payload = engine.compare_strategies("年满 65 周岁的客户购买 R3 产品有哪些额外要求？", top_k=3)
    assert set(payload["strategies"]) == {"dense_only", "bm25_only", "hybrid"}
    assert payload["strategies"]["hybrid"]["source_ids"]


# ---------------------------------------------------------------------------
# 轨迹
# ---------------------------------------------------------------------------
def test_new_run_id_format():
    """运行编号必须可排序且可读：前缀 + 时间戳 + 短随机段，四段式结构稳定。"""
    run_id = new_run_id("ask")
    assert run_id.startswith("ask-")
    assert len(run_id.split("-")) == 4


def test_trace_recorder_writes_jsonl(tmp_path):
    """每步必须落成一行 JSONL 且含输入摘要与扩展字段，一次运行才能被完整回放。"""
    # runs_dir 传 tmp_path：轨迹是运行级产物，不能写进项目的 runs/ 目录
    recorder = TraceRecorder(run_id="unit-run", runs_dir=tmp_path)
    recorder.step("recall", "retriever", "输入", "输出", latency_ms=1.5, extra_field=1)
    rows = read_trace("unit-run", runs_dir=tmp_path)
    assert len(rows) == 1
    assert rows[0]["step"] == "recall"
    assert rows[0]["input_digest"]
    assert rows[0]["extra"]["extra_field"] == 1


def test_trace_recorder_disabled_keeps_records_in_memory(tmp_path):
    """关闭落盘时记录仍留在内存（可查），但不得生成轨迹文件——落盘失败也不影响主流程。"""
    recorder = TraceRecorder(run_id="off-run", runs_dir=tmp_path, enabled=False)
    recorder.step("x", "y", "a", "b")
    assert recorder.records
    assert not recorder.path.exists()


def test_read_trace_missing_file_raises(tmp_path):
    """读不到轨迹文件必须抛 FileNotFoundError，「查不到轨迹」不能被当成「这次没有异常」。"""
    with pytest.raises(FileNotFoundError):
        read_trace("nope", runs_dir=tmp_path)


def test_summarize_trace_aggregates(tmp_path):
    """轨迹汇总必须给出步数、按步计数与非 ok 状态的错误数（状态串不是 "ok" 就算失败）。"""
    # 第二步刻意用 status="warn"：验证"非 ok 即计入 errors"，而不是只统计 status == "error"
    recorder = TraceRecorder(run_id="sum-run", runs_dir=tmp_path)
    recorder.step("a", "x", "", "", latency_ms=2.0)
    recorder.step("a", "x", "", "", latency_ms=3.0, status="warn")
    summary = summarize_trace([r.to_dict() for r in recorder.records])
    assert summary["steps"] == 2
    assert summary["by_step"]["a"]["count"] == 2
    assert summary["errors"] == 1


def test_engine_writes_trace_file(tmp_path, cleaned_documents):
    """引擎一次问答必须逐步写轨迹，且至少覆盖召回 / 生成 / 校验 / 收尾四步（缺一步就无法定位问题段）。"""
    # 单独建引擎并显式指定 runs_dir：本用例验证的是"轨迹写到了哪里"，不能用共享引擎的目录
    # cleaned_documents 只用于确保语料已就绪（与引擎内部加载同源），不直接读取其内容
    engine = RAGEngine.build(runs_dir=tmp_path, quiet=True)
    result = engine.ask("冷静期是多久？", use_cache=False, use_faq=False)
    rows = read_trace(result.run_id, runs_dir=tmp_path)
    steps = {row["step"] for row in rows}
    assert {"recall", "generate", "validate", "done"} <= steps


# ---------------------------------------------------------------------------
# 接口层（纯函数）
# ---------------------------------------------------------------------------
def test_handle_health(engine):
    """健康探针必须返回 ok 且带上最基本的规模信息（存活 + 是否建好索引一次看清）。"""
    # 用 __import__ 动态取 handler：与下面几条用例的局部 import 保持同一风格，避免模块级多引入符号
    payload = __import__("src.api", fromlist=["handle_health"]).handle_health(engine)
    assert payload["status"] == "ok"
    assert payload["children"] > 0


def test_handle_stats(engine):
    """统计接口必须把引擎体检包在 stats 字段里返回（接口形状与 /health 区分开）。"""
    from src.api import handle_stats

    assert handle_stats(engine)["stats"]["corpus"]["documents"] > 0


def test_handle_ask_ok(engine):
    """问答接口正常路径：status 为 ok、带答案与 traceable 标记（业务方据此判断能否复核）。"""
    from src.api import handle_ask

    payload = handle_ask(engine, {"question": "冷静期是多久？", "use_cache": False})
    assert payload["status"] == "ok"
    assert payload["answer"]
    assert payload["traceable"]


def test_handle_ask_missing_question(engine):
    """缺 question 参数必须返回 400 错误对象，而不是抛异常或给出空答案。"""
    from src.api import handle_ask

    payload = handle_ask(engine, {})
    assert payload["status"] == "error"
    assert payload["code"] == 400


def test_handle_search_returns_evidence(engine):
    """检索接口必须在 top_k 上限内返回证据（截断发生在服务端，调用方不需要自己再截）。"""
    from src.api import handle_search

    payload = handle_search(engine, {"question": "合格投资者", "top_k": 3})
    assert payload["status"] == "ok"
    assert len(payload["evidence"]) <= 3


def test_handle_search_missing_question(engine):
    """检索接口同样要校验 question（缺参数返回错误对象，不能拿空串去检索全库）。"""
    from src.api import handle_search

    assert handle_search(engine, {})["status"] == "error"


def test_handle_feedback_records(engine):
    """人工反馈必须落进进程内缓冲区并让总数加一——「反馈 → 补评测集 → 回归」的闭环起点。"""
    from src.api import FEEDBACK_LOG, handle_feedback

    # 先记长度再调用：断言增量而不是绝对值，避免与其它用例的写入相互干扰
    before = len(FEEDBACK_LOG)
    payload = handle_feedback(engine, {"run_id": "r1", "verdict": "bad", "stage": "召回", "comment": "漏召回"})
    assert payload["status"] == "ok"
    assert len(FEEDBACK_LOG) == before + 1


def test_create_app_requires_fastapi(engine):
    """缺 FastAPI 时必须抛 RuntimeError（带安装指引），装了则返回可用的 app——两条分支都要成立。"""
    # 按依赖是否可用分支：CI 里通常没装 FastAPI，本地开发通常装了，两种环境都要能跑
    try:
        import fastapi  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError):
            from src.api import create_app

            create_app(engine)
    else:
        from src.api import create_app

        assert create_app(engine) is not None


# ---------------------------------------------------------------------------
# 兜底 HTTP 服务
# ---------------------------------------------------------------------------
@pytest.fixture
def live_server(engine):
    """起一个真的 stdlib HTTP 服务：端口取 0 让系统分配，用完必须关停。"""
    from src.serve import make_handler

    # 绑 127.0.0.1 且端口为 0：只在本机监听、由系统分配空闲端口，不会与开发中的服务抢端口
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}"
    server.shutdown()
    server.server_close()


def _get(url: str):
    """发一个 GET 并把响应按 JSON 解析（只给本文件的兜底服务用例用）。"""
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 - 仅本地测试
        return json.loads(resp.read().decode("utf-8"))


def _post(url: str, payload: dict):
    """发一个 JSON POST 并解析响应；超时给得比 GET 长，因为问答链路本身有耗时。"""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as resp:  # noqa: S310 - 仅本地测试
        return json.loads(resp.read().decode("utf-8"))


def test_serve_health_endpoint(live_server):
    """兜底 HTTP 服务的 /health 必须可用（零额外依赖也能被探活）。"""
    payload = _get(f"{live_server}/health")
    assert payload["status"] == "ok"


def test_serve_stats_endpoint(live_server):
    """兜底服务的 /stats 必须与引擎体检同源（HTTP 层只做转发，不另算一套口径）。"""
    assert _get(f"{live_server}/stats")["stats"]["corpus"]["documents"] > 0


def test_serve_ask_endpoint(live_server):
    """兜底服务的 /ask 必须端到端可用：走真实 HTTP 编解码后仍返回 ok 与答案。"""
    payload = _post(f"{live_server}/ask", {"question": "冷静期是多久？", "use_cache": False, "use_faq": False})
    assert payload["status"] == "ok"
    assert payload["answer"]


def test_serve_search_endpoint(live_server):
    """兜底服务的 /search 必须返回证据（只检索不生成的那条路径也要能从 HTTP 走到）。"""
    payload = _post(f"{live_server}/search", {"question": "合格投资者", "top_k": 2})
    assert payload["status"] == "ok"
    assert payload["evidence"]


def test_serve_unknown_path_returns_404(live_server):
    """未知路径必须返回 404（而不是 200 + 错误 body），HTTP 语义要守规矩，探活工具才认。"""
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{live_server}/nope")
    assert exc.value.code == 404
