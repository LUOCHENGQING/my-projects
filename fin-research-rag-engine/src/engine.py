"""问答引擎：把解析、切分、索引、检索、生成、缓存、可观测串成一条链路。

这是整个项目对外的门面。它把每一步的**中间产物都保留下来**（召回候选、被去重的块、
重排特征、引用账本、忠实度报告、逐步轨迹），因此：

* 业务侧能拿到「答案 + 出处 + 更新时间 + 检索依据」；
* 工程侧能回答「这次为什么召回这几条、为什么这条排第一、耗时花在哪一段」；
* 评测侧能对同一套用例跑不同策略（单一向量 vs 混合）做 A/B。

初始化成本集中在建索引：一次 `RAGEngine()` 会完成
解析 → 清洗 → 脱敏 → 切分 → 建 BM25 / 向量索引 → 建 FAQ 索引。
之后每次提问只做检索与生成，缓存命中时整个链路可以压到毫秒级。
"""

from __future__ import annotations

import dataclasses
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .answer.citations import Citation, CitationCheck
from .answer.faithfulness import FaithfulnessReport, evaluate_faithfulness
from .answer.generator import AnswerGenerator, GeneratedAnswer
from .answer.llm import LLMClient
from .cache.redis_cache import build_cache, cache_key
from .chunking.parent_child import ChildChunk, ParentChunk, build_chunks, chunk_stats
from .config import DATA_DIR, FINAL_TOP_K, RUNS_DIR, RuntimeConfig, runtime_config
from .faq import FAQIndex
from .index.embedding import EmbeddingBackend, get_backend
from .ingest.cleaning import QualityReport, clean_corpus
from .ingest.loader import Corpus, SourceDocument, apply_masking, load_corpus
from .retrieve.pipeline import Evidence, RetrievalPipeline, RetrievalResult
from .retrieve.rerank import Reranker, get_reranker
from .retrieve.hybrid import HybridRetriever
from .tracing import TraceRecorder, new_run_id
from .utils.console import ensure_utf8_console
from .utils.jsonable import to_plain

__all__ = ["AnswerResult", "RAGEngine", "ENTITY_RE", "unknown_entities"]

# 主体识别：金融问答里最危险的一类错误是「张冠李戴」——
# 用户问 A 公司，资料库里只有 B 公司，模型却拿 B 公司的数字回答了 A 公司。
# 因此这里做一道**主体闸门**：问题里提到的公司/机构如果不在资料库主体清单内，
# 直接拒答，而不是拿相近的资料硬凑。
ENTITY_RE = re.compile(
    r"[\u4e00-\u9fff]{2,12}"
    r"(?:股份有限公司|有限责任公司|基金管理有限公司|有限公司|集团公司|集团|银行|证券公司|证券|基金|研究所|研究院)"
)
_GENERIC_SUFFIXES = (
    "股份有限公司",
    "有限责任公司",
    "基金管理有限公司",
    "有限公司",
    "集团公司",
    "集团",
    "证券公司",
    "银行",
    "证券",
    "基金",
    "研究所",
    "研究院",
)


def _core_name(name: str) -> str:
    """反复剥掉「股份有限公司」这类通用后缀，得到可比较的主体核心名。"""
    core = (name or "").strip()
    changed = True
    while changed and core:
        changed = False
        for suffix in _GENERIC_SUFFIXES:
            if core.endswith(suffix) and len(core) > len(suffix):
                core = core[: -len(suffix)]
                changed = True
                break
    return core.strip()


def question_entities(question: str) -> List[str]:
    """抽出问题里提到的公司 / 机构名（按出现顺序去重）。"""
    seen: List[str] = []
    for match in ENTITY_RE.finditer(question or ""):
        name = match.group(0).strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def _is_known_entity(mention: str, known: Sequence[str]) -> bool:
    """判断一个问题里提到的主体能否被资料库里的已知主体"解释"。

    判据有两级，刻意都写得保守（宁可放过，不可错杀——错杀会让正常问题被拒答）：
        1. 与某个已知主体互为子串（用户少写/多写后缀的常见情况）；
        2. 与某个已知主体的**前四个字**一致（用户只写了品牌前缀，如「示例银行」）。
    注意 `示例科技` 与 `示例银行` 共享品牌前缀「示例」，但前四字不同，
    因此仍会被判为未知主体——这正是我们要的行为。
    """
    text = (mention or "").strip()
    if not text:
        return True
    for name in known:
        if not name:
            continue
        if text in name or name in text:
            return True
        if name.startswith(text[:4]) or text.startswith(name[:4]):
            return True
    return False


def unknown_entities(question: str, known: Sequence[str]) -> List[str]:
    """返回问题中提到、但不在资料库主体清单里的主体。"""
    out: List[str] = []
    for entity in question_entities(question):
        core = _core_name(entity)
        # 纯通用后缀（如「股份有限公司」）不构成主体，跳过
        if len(core) < 2:
            continue
        if _is_known_entity(entity, known):
            continue
        out.append(entity)
    return out


@dataclass
class AnswerResult:
    """一次问答的完整结果。"""

    question: str
    answer: str
    mode: str = "rag"                     # rag | faq | refused
    run_id: str = ""
    citations: List[Citation] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    retrieval: Optional[RetrievalResult] = None
    faithfulness: Optional[FaithfulnessReport] = None
    citation_check: Optional[CitationCheck] = None
    cache_hit: bool = False
    faq_hit: bool = False
    total_latency_ms: float = 0.0
    timings: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def traceable(self) -> bool:
        if self.faq_hit:
            return bool(self.answer.strip())
        if self.citation_check is None:
            return False
        return not self.citation_check.dangling and bool(self.citation_check.used)

    @property
    def source_ids(self) -> List[str]:
        seen: List[str] = []
        for c in self.citations:
            if c.source_id not in seen:
                seen.append(c.source_id)
        return seen

    @property
    def citation_numbers(self) -> List[int]:
        return [c.citation_no for c in self.citations]

    def render(self) -> str:
        """带出处清单的纯文本渲染，可直接贴到终端 / 接口返回里。"""
        lines = [self.answer.strip()]
        if self.citations:
            lines.append("")
            lines.append("【引用明细】")
            for c in self.citations:
                lines.append(
                    f"[{c.citation_no}] {c.label}｜资料编号 {c.source_id}｜"
                    f"更新/生效日期 {c.updated_at}｜版本 {c.version or '未标注'}"
                )
        return "\n".join(lines)

    def to_dict(self, with_context: bool = False) -> Dict[str, Any]:
        return to_plain(
            {
                "question": self.question,
                "answer": self.answer,
                "mode": self.mode,
                "run_id": self.run_id,
                "traceable": self.traceable,
                "cache_hit": self.cache_hit,
                "faq_hit": self.faq_hit,
                "citation_numbers": self.citation_numbers,
                "source_ids": self.source_ids,
                "citations": [c.to_dict() for c in self.citations],
                "citation_check": self.citation_check.to_dict() if self.citation_check else None,
                "faithfulness": self.faithfulness.to_dict() if self.faithfulness else None,
                "evidence": [e.to_dict(with_context=with_context) for e in self.evidence],
                "retrieval_stats": self.retrieval.stats() if self.retrieval else None,
                "timings": {k: round(float(v), 3) for k, v in self.timings.items()},
                "total_latency_ms": round(float(self.total_latency_ms), 3),
                "notes": list(self.notes),
            }
        )


class RAGEngine:
    """金融智研引擎主入口。"""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        runs_dir: Optional[Path] = None,
        embed_backend: Optional[EmbeddingBackend] = None,
        reranker: Optional[Reranker] = None,
        cache: Any = None,
        enable_cache: bool = True,
        enable_faq: bool = True,
        enable_trace: bool = True,
        final_top_k: int = FINAL_TOP_K,
        llm: Optional[LLMClient] = None,
        quiet: bool = False,
    ) -> None:
        self.quiet = quiet
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.runs_dir = Path(runs_dir) if runs_dir else RUNS_DIR
        self.enable_cache = enable_cache
        self.enable_faq = enable_faq
        self.enable_trace = enable_trace
        self.final_top_k = int(final_top_k)

        self.cache = cache if cache is not None else (build_cache("memory") if enable_cache else None)

        # ---- 解析 / 清洗 / 脱敏 ----
        self.corpus: Corpus = load_corpus(self.data_dir)
        self.masked_documents = apply_masking(self.corpus.documents)
        self.quality_reports: List[QualityReport] = clean_corpus(self.corpus.documents)
        self.quality_scores: Dict[str, float] = {r.source_id: r.score for r in self.quality_reports}

        # ---- 切分 ----
        self.parents, self.children = build_chunks(
            list(self.corpus.documents),
            quality_scores=self.quality_scores,
        )
        self.chunk_report = chunk_stats(self.parents, self.children, document_count=len(self.corpus.documents))

        # ---- 索引与检索 ----
        self.backend = embed_backend or get_backend()
        self.reranker = reranker or get_reranker()
        self.retriever = HybridRetriever(self.parents, self.children, backend=self.backend)
        self.pipeline = RetrievalPipeline(
            self.retriever,
            reranker=self.reranker,
            final_top_k=self.final_top_k,
        )

        # ---- 生成 ----
        self.llm = llm or LLMClient()
        self.generator = AnswerGenerator(self.llm)

        # ---- FAQ ----
        self.faq = FAQIndex(self.corpus.faq, backend=self.backend) if self.corpus.faq else None

        # ---- 主体清单：机构 + 文档标题里出现的主体 + 产品名 ----
        # 只把 institutions 当主体是不够的：尽调档案的主体（如"示例集团股份有限公司"）
        # 是**被调查对象**，它出现在文档标题里而不是元数据的 institution 字段里。
        # 漏掉它会让正常的尽调问题被主体闸门误拒。
        known = set(self.corpus.institutions)
        for doc in self.corpus.documents:
            known.update(question_entities(doc.title))
            product = str(doc.meta.get("product", "")).strip()
            if product:
                known.add(product)
        self.known_entities: List[str] = sorted(e for e in known if e)

        self.last_run_id: str = ""
        self.stats_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 类方法：一行构造
    # ------------------------------------------------------------------
    @classmethod
    def build(cls, data_dir: Optional[Path] = None, **kwargs: Any) -> "RAGEngine":
        ensure_utf8_console()
        return cls(data_dir=data_dir, **kwargs)

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    @property
    def documents(self) -> List[SourceDocument]:
        return list(self.corpus.documents)

    def config(self) -> RuntimeConfig:
        return runtime_config(cache_backend=getattr(self.cache, "name", "none"))

    def stats(self) -> Dict[str, Any]:
        """引擎体检：资料库、切分、索引、缓存、FAQ 的全部规模指标。"""
        info: Dict[str, Any] = {
            "corpus": self.corpus.stats(),
            "chunks": self.chunk_report.to_dict(),
            "index": self.retriever.describe(),
            "reranker": getattr(self.reranker, "name", "unknown"),
            "llm": self.llm.describe(),
            "faq": self.faq.describe() if self.faq else {"entries": 0},
            "cache": self.cache.describe() if self.cache is not None else {"backend": "off"},
            "masked_documents": self.masked_documents,
            "known_entities": self.known_entities,
            "quality": {
                "avg_score": round(
                    sum(self.quality_scores.values()) / len(self.quality_scores), 4
                ) if self.quality_scores else 0.0,
                "low_quality": sorted(
                    [r.to_dict() for r in self.quality_reports if r.score < 0.9],
                    key=lambda r: r["score"],
                ),
            },
            "config": self.config().to_dict(),
        }
        return info

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(
        self,
        question: str,
        top_k: Optional[int] = None,
        expr: Optional[str] = None,
        route: Optional[str] = None,
        mode: str = "hybrid",
    ) -> RetrievalResult:
        """只做检索，不生成答案（给"找依据"这类需求用）。"""
        return self.pipeline.run(
            question,
            top_k=top_k or self.final_top_k,
            expr=expr,
            route=route,
            corpus=self.corpus,
            mode=mode,
        )

    # ------------------------------------------------------------------
    # 问答主流程
    # ------------------------------------------------------------------
    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        expr: Optional[str] = None,
        route: Optional[str] = None,
        use_cache: bool = True,
        use_faq: bool = True,
        mode: str = "hybrid",
    ) -> AnswerResult:
        started = time.perf_counter()
        run_id = new_run_id("ask")
        self.last_run_id = run_id
        recorder = TraceRecorder(run_id=run_id, runs_dir=self.runs_dir, enabled=self.enable_trace)
        timings: Dict[str, float] = {}
        notes: List[str] = []

        # ---- 1. 缓存 ----
        key = cache_key(question, expr or "", top_k or self.final_top_k, mode, route or "")
        if self.cache is not None and use_cache:
            phase = time.perf_counter()
            cached = self.cache.get(key)
            timings["cache_ms"] = (time.perf_counter() - phase) * 1000.0
            recorder.step("cache_lookup", "cache", key, "hit" if cached else "miss",
                          latency_ms=timings["cache_ms"])
            if cached:
                result = _answer_from_cache(cached, run_id)
                result.cache_hit = True
                result.total_latency_ms = (time.perf_counter() - started) * 1000.0
                # 时间口径与未命中路径保持一致，否则调用方按 total_ms 取耗时会 KeyError
                timings["total_ms"] = result.total_latency_ms
                result.timings = dict(timings)
                recorder.step("done", "engine", question, result.answer,
                              latency_ms=result.total_latency_ms, cache_hit=True)
                note = "命中缓存，未重新检索与生成"
                if note not in result.notes:
                    result.notes.append(note)
                return result

        # ---- 2. FAQ 直出 ----
        faq_hit_result: Optional[AnswerResult] = None
        if self.enable_faq and use_faq and self.faq is not None:
            phase = time.perf_counter()
            faq_payload = self.faq.answer(question)
            timings["faq_ms"] = (time.perf_counter() - phase) * 1000.0
            recorder.step("faq_lookup", "faq", question,
                          faq_payload["faq"]["question"] if faq_payload else "miss",
                          latency_ms=timings["faq_ms"])
            if faq_payload:
                faq_hit_result = AnswerResult(
                    question=question,
                    answer=str(faq_payload["answer"]),
                    mode="faq",
                    run_id=run_id,
                    faq_hit=True,
                    notes=[f"FAQ 命中（相似度 {faq_payload['score']}），未走完整检索链路"],
                )
                faq_hit_result.timings = dict(timings)

        if faq_hit_result is not None:
            faq_hit_result.total_latency_ms = (time.perf_counter() - started) * 1000.0
            faq_hit_result.timings["total_ms"] = faq_hit_result.total_latency_ms
            if self.cache is not None and use_cache:
                self.cache.set(key, faq_hit_result.to_dict())
            recorder.step("done", "engine", question, faq_hit_result.answer,
                          latency_ms=faq_hit_result.total_latency_ms, mode="faq")
            return faq_hit_result

        # ---- 3. 主体闸门：问题问的主体不在资料库范围内 → 拒答 ----
        phase = time.perf_counter()
        unknown = unknown_entities(question, self.known_entities)
        timings["entity_gate_ms"] = (time.perf_counter() - phase) * 1000.0
        if unknown:
            recorder.step("entity_gate", "guard", question, unknown,
                          latency_ms=timings["entity_gate_ms"], status="refused")
            text = (
                f"现有资料不足以回答该问题：「{unknown[0]}」不在本资料库的主体范围内。"
                f"资料库当前覆盖的主体为：{'、'.join(self.known_entities)}。"
                "请补充该主体的制度文件、产品说明书或尽调档案后再试。"
            )
            refused = AnswerResult(
                question=question,
                answer=text,
                mode="refused",
                run_id=run_id,
                notes=["主体闸门拦截：问题中的主体不在资料库范围内，按拒答策略返回"],
            )
            refused.citation_check = CitationCheck(ok=True, used=[], dangling=[], unused=[], cleaned_text=text)
            refused.total_latency_ms = (time.perf_counter() - started) * 1000.0
            timings["total_ms"] = refused.total_latency_ms
            refused.timings = dict(timings)
            if self.cache is not None and use_cache:
                self.cache.set(key, refused.to_dict())
            return refused

        # ---- 4. 检索 ----
        phase = time.perf_counter()
        retrieval = self.search(question, top_k=top_k, expr=expr, route=route, mode=mode)
        timings["retrieval_ms"] = (time.perf_counter() - phase) * 1000.0
        recorder.step(
            "recall", "retriever", question,
            f"{len(retrieval.candidates)} candidates -> {len(retrieval.evidence)} evidence",
            latency_ms=timings["retrieval_ms"],
            route=retrieval.plan.route,
            queries=len(retrieval.plan.queries),
            filter_expr=retrieval.plan.filter_expr or "(none)",
            removed_duplicates=len(retrieval.removed_duplicates),
        )
        if retrieval.evidence:
            recorder.step(
                "rerank", "reranker",
                question,
                [f"{e.child_id}:{round(e.score, 4)}" for e in retrieval.evidence],
                latency_ms=0.0,
                top=[f"{e.child_id}#{e.routes_hit}" for e in retrieval.evidence],
            )

        # ---- 5. 生成 + 校验 ----
        phase = time.perf_counter()
        plan_note = f"问题类型={retrieval.plan.route}；召回候选={len(retrieval.candidates)} 条；过滤={retrieval.plan.filter_expr or '无'}"
        generated: GeneratedAnswer = self.generator.generate(question, retrieval.evidence, plan_note)
        timings["generate_ms"] = (time.perf_counter() - phase) * 1000.0

        evidence_texts = [e.text for e in retrieval.evidence]
        phase = time.perf_counter()
        faithfulness = evaluate_faithfulness(generated.text, question, evidence_texts)
        timings["verify_ms"] = (time.perf_counter() - phase) * 1000.0

        recorder.step("generate", "generator", question, generated.text,
                      latency_ms=timings["generate_ms"], mode=generated.mode,
                      refused=generated.refused)
        recorder.step("validate", "validator", generated.text,
                      {
                          "dangling": generated.check.dangling if generated.check else [],
                          "faithful": faithfulness.faithful,
                          "support_rate": round(faithfulness.support_rate, 4),
                          "unsupported_numbers": faithfulness.numbers.unsupported,
                      },
                      latency_ms=timings["verify_ms"],
                      status="ok" if faithfulness.faithful else "warn")

        notes.extend(generated.notes)
        if faithfulness.numbers.unsupported:
            notes.append(f"发现 {len(faithfulness.numbers.unsupported)} 个无法在证据中定位的数字")
        if retrieval.plan.soft_expr:
            notes.append(f"按问题推断的软过滤条件：{retrieval.plan.soft_expr}（只影响排序，不剔除）")

        result = AnswerResult(
            question=question,
            answer=generated.text,
            mode="refused" if generated.refused else "rag",
            run_id=run_id,
            citations=generated.citations,
            evidence=retrieval.evidence,
            retrieval=retrieval,
            faithfulness=faithfulness,
            citation_check=generated.check,
            notes=notes,
        )
        result.total_latency_ms = (time.perf_counter() - started) * 1000.0
        timings["total_ms"] = result.total_latency_ms
        result.timings = dict(timings)

        # ---- 6. 回填缓存 ----
        if self.cache is not None and use_cache:
            self.cache.set(key, result.to_dict())

        recorder.step("done", "engine", question, result.answer,
                      latency_ms=result.total_latency_ms, traceable=result.traceable)
        return result

    # ------------------------------------------------------------------
    def compare_strategies(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """对同一个问题跑「单一向量 / 纯关键词 / 混合三路」三种策略，输出对比。

        这就是"混合检索让复杂问题召回率提升 40%"的现场证明：
        同一个问题、同一份资料库，只换检索策略，看 Top-K 里命中情况的变化。
        """
        limit = max(1, top_k or self.final_top_k)
        runs = {
            "dense_only": self.search(question, top_k=limit, mode="dense"),
            "bm25_only": self.search(question, top_k=limit, mode="bm25"),
            "hybrid": self.search(question, top_k=limit, mode="hybrid"),
        }
        return {
            "question": question,
            "top_k": limit,
            "strategies": {
                name: {
                    "source_ids": r.source_ids,
                    "evidence": [e.text[:80] for e in r.evidence],
                    "elapsed_ms": round(r.elapsed_ms, 3),
                }
                for name, r in runs.items()
            },
        }


# ---------------------------------------------------------------------------
# 缓存反序列化
# ---------------------------------------------------------------------------
def _answer_from_cache(payload: Dict[str, Any], run_id: str) -> AnswerResult:
    """把缓存里的 dict 还原成 AnswerResult（引用与证据都要能继续被使用）。"""
    citations = [_rebuild(Citation, c) for c in payload.get("citations") or []]
    check_payload = payload.get("citation_check")
    check = _rebuild(CitationCheck, check_payload) if isinstance(check_payload, dict) else None
    return AnswerResult(
        question=str(payload.get("question", "")),
        answer=str(payload.get("answer", "")),
        mode=str(payload.get("mode", "rag")),
        run_id=run_id,
        citations=citations,
        evidence=[],
        retrieval=None,
        faithfulness=None,
        citation_check=check,
        faq_hit=bool(payload.get("faq_hit")),
        notes=["（来自缓存的结果，未重新检索）"],
    )


def _rebuild(cls: type, payload: Dict[str, Any]):
    """按 dataclass 字段过滤后重建对象，容忍缓存里有新增/缺失字段。"""
    if not dataclasses.is_dataclass(cls):
        return payload
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in payload.items() if k in names})
