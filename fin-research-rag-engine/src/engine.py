"""问答引擎：把解析、切分、索引、检索、生成、缓存、可观测串成一条链路。

这是整个项目对外的门面。它把每一步的**中间产物都保留下来**（召回候选、被去重的块、
重排特征、引用账本、忠实度报告、逐步轨迹），因此：

* 业务侧能拿到「答案 + 出处 + 更新时间 + 检索依据」；
* 工程侧能回答「这次为什么召回这几条、为什么这条排第一、耗时花在哪一段」；
* 评测侧能对同一套用例跑不同策略（单一向量 vs 混合）做 A/B。

初始化成本集中在建索引：一次 `RAGEngine()` 会完成
解析 → 清洗 → 脱敏 → 切分 → 建 BM25 / 向量索引 → 建 FAQ 索引。
注：实际实现为**脱敏在清洗之前**——`__init__` 的调用顺序是 `load_corpus()` →
`apply_masking()` → `clean_corpus()`（脱敏先于清洗，这样清洗阶段处理的就已经是掩码文本，
掩码不会因为清洗规则而被还原或改写）。
之后每次提问只做检索与生成，缓存命中时整个链路可以压到毫秒级。

主链路编排（`RAGEngine.ask()` 的固定顺序，即「解析 → 切分 → 索引 → 召回 → 重排 → 作答 → 缓存」）
----------------------------------------------------------------------------------------
    A) 构造期（`RAGEngine.__init__`，每个进程只做一次）
       load_corpus(data_dir)     → Corpus（文档 + FAQ）
       apply_masking(documents)  → 就地脱敏（返回被改动的文档数）
       clean_corpus(documents)   → List[QualityReport]，质量分随子块进入检索参与降权
       build_chunks(...)         → (parents, children)，再建 HybridRetriever / RetrievalPipeline /
                                   AnswerGenerator / FAQIndex
       失败处理：目录缺失或单篇解析失败**不会中断构造**，而是降级为 `corpus.issues` 里的告警；
                 资料为空时索引为空，后续提问会因证据不足走拒答，而不是抛异常。
    B) 提问期（`ask()`，每次提问走一遍；可预期的失败一律转成 notes / 拒答 / 降级作答，
       缓存、轨迹、LLM 的异常在各自模块内被吞掉或降级，本链路不主动抛业务异常）
       1. cache_lookup   输入 cache_key(问题, expr, top_k, mode, route)；
                         命中 → 直接用缓存 dict 还原 AnswerResult（cache_hit=True），链路到此结束；
                         失败 → 缓存层内部吞异常视作 miss，主链路继续。
       2. faq_lookup     输入原问题；命中（相似度 ≥ FAQ_MATCH_THRESHOLD 且标识符一致）→
                         mode="faq" 的答案，回填缓存后返回；不命中 → 继续。
       3. entity_gate    输入原问题 + known_entities；出现资料库不认识的主体 →
                         mode="refused" 拒答（**不检索、不生成**），同样回填缓存后返回。
       4. recall/rerank  输入问题 + top_k/expr/route/mode → RetrievalResult
                         （candidates 候选 / removed_duplicates 去重记录 / evidence 最终证据）；
                         无候选 → evidence 为空，交由生成层的 min_evidence 触发拒答。
       5. generate/validate  输入问题 + evidence + 检索计划摘要 → GeneratedAnswer 与
                         FaithfulnessReport；真实模型不可用（无 key / 报错 / 空文本）→
                         自动降级为抽取式作答并在 notes 写明原因；悬空引用当场剔除。
       6. done           把 `AnswerResult.to_dict()` 回填缓存，并记最后一行轨迹。
       每一步耗时都写进 `AnswerResult.timings`，键固定为 cache_ms / faq_ms / entity_gate_ms /
       retrieval_ms / generate_ms / verify_ms / total_ms——**没走到的步骤就没有对应的键**
       （例如缓存命中只有 cache_ms 与 total_ms），调用方取耗时需用 `.get()` 或先判断。

对外关键类 / 函数
----------------
    RAGEngine            主入口：`build()` 一行构造、`ask()` 问答、`search()` 只检索、
                         `stats()` 引擎体检、`compare_strategies()` 三策略 A/B
    AnswerResult         一次问答的完整结果（答案 + 引用 + 证据 + 忠实度 + 耗时 + notes）
    ENTITY_RE / question_entities / unknown_entities    主体闸门（防「张冠李戴」）
    _answer_from_cache / _rebuild                       缓存 dict 反序列化（模块私有）

输入 / 输出 / 调用方
--------------------
    输入：`data/` 资料目录（默认 config.DATA_DIR）与问题字符串；
    输出：`AnswerResult`；副作用：写 `runs/<run_id>.jsonl` 轨迹、写缓存、推进 `last_run_id`；
    被谁调用：`src/api.py`（handle_ask / handle_search / handle_stats / handle_health）、
              `src/serve.py`、`src/demo.py`、`eval/run_eval.py`、`tests/test_engine_api.py`。

六处「同接口换实现」的降级开关在本模块的体现
------------------------------------------
    `__init__` 的四个可选入参就是注入点：embed_backend（BGE-M3 ↔ 本地确定性哈希）、
    reranker（bge-reranker ↔ 本地交叉编码器）、cache（Redis ↔ 内存 LRU，None 时由
    `build_cache("memory")` 决定）、llm（OpenAI 兼容 ↔ 确定性抽取式）；
    向量库与 OCR 的切换发生在 index / ingest 内部，本模块无感知。
    因此这里**没有任何 "if 组件不可用" 的分支**，只有统一的降级语义（notes + describe()）。
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
# 通用后缀表：用于把「示例银行股份有限公司」这类全称反复剥成核心名做比较。
# 注：实际实现为这张表并非严格的"由长到短"排序（前五项是从长到短，后面的
# 证券公司 / 研究所 / 研究院 排在「集团 / 银行」之后）；因为 _core_name() 是**循环**剥离，
# 顺序不影响最终结果，只影响需要转几轮，所以这里没有刻意重排。
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
    """反复剥掉「股份有限公司」这类通用后缀，得到可比较的主体核心名。

    参数：name 主体全称或用户写法（None/空串按空处理）。
    返回：str 核心名；全称只剩后缀时返回剩余部分（可能为空串）。
    副作用/异常：无；不做模糊匹配，只按 `_GENERIC_SUFFIXES` 做后缀剥离，
                且**只在剩余长度大于后缀本身时**才剥，避免把「银行」这类短名剥成空串。
    """
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
    """抽出问题里提到的公司 / 机构名（按出现顺序去重）。

    参数：question 用户问题（None/空串返回空列表）。
    返回：List[str]——`ENTITY_RE` 命中的原文片段，保持出现顺序，完全相同的只留第一次。
    副作用/异常：无。注：这是**纯字面**识别（机构后缀驱动），不做实体链接；
                取不到主体时返回空列表，主体闸门据此判定"没有问题主体"而放行。
    """
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

    参数：mention 问题里提到的主体；known 资料库已知主体清单（来自 `RAGEngine.known_entities`）。
    返回：bool——能被"解释"（互为子串，或前四字有一致）为 True；mention 为空白时**返回 True**
          （没有主体就不拦，避免把不含主体的正常问题拒掉）。
    副作用/异常：无；`known` 里的空串会被跳过。
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
    """返回问题中提到、但不在资料库主体清单里的主体。

    参数：question 用户问题；known 资料库已知主体清单。
    返回：List[str]——未知主体原文（按出现顺序）；空列表表示"主体都在范围内"，可正常检索。
    副作用/异常：无。注：核心名长度 < 2 的（如孤零零的「集团」）不算主体，跳过；
              这是主体闸门唯一的判定输入，`RAGEngine.ask()` 拿到非空即直接拒答。
    """
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
    """一次问答的完整结果。

    关键属性（按字段声明顺序）：
        question / answer       原问题与最终答案正文（FAQ 直出或生成结果）
        mode                    **结果来源**：rag（走完整检索 + 生成）| faq（FAQ 直出）| refused（拒答）
        run_id                  本次运行编号，对应 `runs/<run_id>.jsonl` 轨迹文件
        citations               List[Citation]：答案用到的引用（编号 / 出处 / 更新日期 / 版本）
        evidence                List[Evidence]：交给模型的证据（子块命中片段 + 父块上下文）
        retrieval               完整 RetrievalResult（候选 / 去重记录 / 检索计划）；仅 rag 路径非空
        faithfulness            FaithfulnessReport（数字忠实度 / 句子支撑率 / 相关度）；仅 rag 路径非空
        citation_check          CitationCheck（引用校验结果：used / dangling / unused / 清理后正文）
        cache_hit / faq_hit     本次是否命中缓存 / 是否 FAQ 直出
        total_latency_ms        端到端毫秒数（含缓存查询与 FAQ 查询本身）
        timings                 分步耗时（键见模块 docstring，**没走到的步骤没有键**）
        notes                   人类可读的「这次发生了什么」：降级说明、命中说明、软过滤提示、
                                无法在证据中定位的数字条数等

    状态流转：`RAGEngine.ask()` 按 cache → faq → 主体闸门 → 检索 → 生成 的顺序逐个字段填充，
              最后统一补 total_latency_ms 与 timings；缓存命中路径由 `_answer_from_cache()`
              重建，此时 retrieval / faithfulness / evidence 均为 None 或空。
    """

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
        """答案是否「可复核」——金融场景的准入条件，也是接口响应里的固定字段。

        返回：bool。判定口径（与 `src/answer/citations.py` 的校验结果绑定）：
              FAQ 直出 → 只要答案正文非空就算可追溯（FAQ 自带人工维护的出处）；
              其余情况 → 必须有 citation_check，且**无悬空引用**（dangling 为空）
              且**至少用到一个引用**（used 非空）。
              注：主体闸门拒答时 citation_check 是空账本（used=[]），因此 traceable 为 False——
                  拒答本就无可追溯的引用。
        """
        if self.faq_hit:
            return bool(self.answer.strip())
        if self.citation_check is None:
            return False
        return not self.citation_check.dangling and bool(self.citation_check.used)

    @property
    def source_ids(self) -> List[str]:
        """答案引用的**资料编号**去重列表（如 ["POL-2024-07", "PROD-WY2024-01"]）。

        返回：List[str]，按首次出现顺序；缓存命中的结果也能给出（引用是从缓存重建的）。
        副作用/异常：无。
        """
        seen: List[str] = []
        for c in self.citations:
            if c.source_id not in seen:
                seen.append(c.source_id)
        return seen

    @property
    def citation_numbers(self) -> List[int]:
        """答案里出现的引用角标编号列表（如 [1, 2]），顺序与 citations 一致。"""
        return [c.citation_no for c in self.citations]

    def render(self) -> str:
        """带出处清单的纯文本渲染，可直接贴到终端 / 接口返回里。

        参数：无。返回：str——答案正文 + 空行 + 「【引用明细】」逐条
              `[编号] 出处标签｜资料编号 …｜更新/生效日期 …｜版本 …`（版本为空时写「未标注」）。
        副作用/异常：无；没有引用时只返回答案正文，不输出空的明细段。
        """
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
        """导出接口 / 缓存 / 落盘用的扁平 dict。

        参数：with_context 是否带上每条证据的**父块上下文**（`Evidence.to_dict()` 的开关）；
              接口 `/ask` 用默认 False（响应更瘦），`/search` 走的是检索层自己的
              `RetrievalResult.to_dict(with_context=True)`，与本方法无关。
        返回：dict——question / answer / mode / run_id / traceable / cache_hit / faq_hit /
              citation_numbers / source_ids / citations / citation_check / faithfulness /
              evidence / retrieval_stats（来自 `retrieval.stats()`，无检索则 None）/
              timings（各段耗时四舍五入到 3 位小数）/ total_latency_ms / notes，
              整体经 `to_plain()` 净化（numpy 标量会变成原生类型，可直接 JSON 化）。
        副作用/异常：无。注：**不含** `evidence` 之外的原始候选与检索计划细节，
              需要完整中间产物请直接读 `result.retrieval`。
        """
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
    """金融智研引擎主入口。

    职责：一次构造把资料变成三类索引（BM25 字面 / 稠密向量 / 稀疏权重）+ FAQ 索引；
          之后对外只用四个方法：`ask()` 问答、`search()` 只检索、`stats()` 体检、
          `compare_strategies()` 三策略对比。

    关键属性（构造后即就绪，多为"活对象"而非快照）：
        corpus / masked_documents / quality_reports / quality_scores   资料层产物
        parents / children / chunk_report                              切分层产物
        backend（嵌入后端）/ reranker / retriever / pipeline            检索层
        llm / generator                                                生成层
        faq（资料里没有 FAQ 时为 None）/ cache（enable_cache=False 时为 None）
        known_entities                                                 主体闸门清单（已排序）
        data_dir / runs_dir / final_top_k / enable_cache / enable_faq / enable_trace / quiet
        last_run_id（最近一次 `ask()` 的 run_id，未问过为空串）

    状态流转：构造期一次性建索引；运行期可变状态只有 `last_run_id` 与缓存/统计的计数，
              检索与生成这两条主路径是只读的。
    注：实际实现为 `stats_cache` 只在 `__init__` 里初始化为空 dict，全项目没有第二个
        读写点（看起来是预留的体检结果缓存），因此不要指望它缓存 `stats()` 的结果。
    """

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
        """构造引擎：解析 / 脱敏 / 清洗 / 切分 / 建索引 / 装载 FAQ 与主体清单，全部一次做完。

        参数：
            data_dir      资料目录，None → `config.DATA_DIR`（只扫顶层，不递归）
            runs_dir      轨迹目录，None → `config.RUNS_DIR`
            embed_backend 嵌入后端，None → `get_backend()`（按 EMBED_BACKEND 探测并自动降级）
            reranker      重排后端，None → `get_reranker()`（优先 bge-reranker，不可用回落本地）
            cache         缓存对象；None 时：enable_cache=True → `build_cache("memory")`，否则 None
            enable_cache  是否启用缓存（False 时 cache 为 None，`ask()` 跳过缓存两段）
            enable_faq    是否启用 FAQ 直出（同时决定是否建 FAQ 索引）
            enable_trace  是否**落盘**轨迹（False 时仍累积在内存，只是不写 JSONL）
            final_top_k   重排后交给 LLM 的证据条数（`search()` / `ask()` 未传 top_k 时用它）
            llm           LLM 客户端，None → `LLMClient()`（无 OPENAI_API_KEY 时自动 mock）
            quiet         注：实际实现为只被赋值到 `self.quiet`，全项目没有第二个读取点，
                          引擎本身不打印日志，因此它当前不产生任何效果
        返回：None。
        副作用：读 `data/`；`apply_masking()` **就地**改写文档文本；建三路索引与 FAQ 索引；
                `known_entities` 由 institutions + 文档标题里的主体 + 每篇文档的 product 元数据合成。
        异常：基本不抛——解析失败降级为 `corpus.issues`；仅 `int(final_top_k)` 在传入不可转换值时
              抛 TypeError/ValueError。
        """
        self.quiet = quiet
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.runs_dir = Path(runs_dir) if runs_dir else RUNS_DIR
        self.enable_cache = enable_cache
        self.enable_faq = enable_faq
        self.enable_trace = enable_trace
        self.final_top_k = int(final_top_k)

        self.cache = cache if cache is not None else (build_cache("memory") if enable_cache else None)

        # ---- 解析 / 清洗 / 脱敏 ----
        # 注：实际执行顺序是**解析 → 脱敏 → 清洗**（见下面三行），脱敏在清洗之前，
        # 这样清洗规则面对的一直是掩码文本，敏感串不会因为清洗步骤被"还原"回去。
        self.corpus: Corpus = load_corpus(self.data_dir)
        # 注：masked_documents 是**被改动的文档数（int）**，不是文档列表
        self.masked_documents = apply_masking(self.corpus.documents)
        # 质量分按 source_id 汇总，随后传给 build_chunks()，让低质量文档的块在检索里自然降权
        self.quality_reports: List[QualityReport] = clean_corpus(self.corpus.documents)
        self.quality_scores: Dict[str, float] = {r.source_id: r.score for r in self.quality_reports}

        # ---- 切分 ----
        # 父块进 LLM 上下文（保证不丢前提与例外），子块进索引（保证命中精度）
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
        """一行构造引擎（demo / serve / 测试的默认入口）。

        参数：data_dir 可选资料目录；**kwargs 原样透传给 `__init__`
              （enable_cache / enable_faq / enable_trace / final_top_k / quiet / cache / llm …）。
        返回：构造好的 `RAGEngine`。
        副作用：先 `ensure_utf8_console()`（Windows 下把重定向的 stdout/stderr 切 UTF-8，防中文乱码），
              再走 `__init__` 建索引。
        异常：与 `__init__` 相同（解析失败不抛，只降级为 corpus.issues）。
        """
        ensure_utf8_console()
        return cls(data_dir=data_dir, **kwargs)

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    @property
    def documents(self) -> List[SourceDocument]:
        """资料库文档列表的**浅拷贝**：调用方增删这个列表不会影响引擎内部状态。"""
        return list(self.corpus.documents)

    def config(self) -> RuntimeConfig:
        """导出本次运行的有效配置快照。

        返回：`RuntimeConfig`——缓存后端名取自 `getattr(self.cache, "name", "none")`
              （缓存关闭时为 "none"），`llm_mode` 由是否有 API Key 决定。
        副作用/异常：无。
        """
        return runtime_config(cache_backend=getattr(self.cache, "name", "none"))

    def stats(self) -> Dict[str, Any]:
        """引擎体检：资料库、切分、索引、缓存、FAQ 的全部规模指标。

        返回：dict——
            corpus / chunks / index（`retriever.describe()`）/ reranker（后端名）/
            llm（`describe()`）/ faq（`describe()`，无 FAQ 时 {"entries": 0}）/
            cache（`describe()`，无缓存时 {"backend": "off"}）/ masked_documents /
            known_entities / quality（平均分 + 低于 0.9 分的文档明细，按分升序）/
            config（`self.config().to_dict()`）。
        副作用：会现场调用各后端的 `describe()`（Redis 缓存的 describe 会如实报告是否连上）。
        异常：无（quality 为空时平均分记 0.0）。
        """
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
        """只做检索，不生成答案（给"找依据"这类需求用）。

        参数：question 问题；top_k 取回的最终证据条数，None → `self.final_top_k`
              （注：实际实现用 `top_k or self.final_top_k`，所以传 0 也会被当作未传）；
              expr 元数据硬过滤表达式（如 `year = 2023`、`doc_type = "监管政策"`），None 走推断；
              route 强制指定问题类型（clause / case / metric / general），None 自动分类；
              mode 检索模式："hybrid"（默认，三路召回 + 重排）| "dense" | "bm25"
                   （后两者是**单一通路且不重排**的对照组基线）。
        返回：`RetrievalResult`——candidates（三路候选含各路分数与名次）、
              removed_duplicates（去重记录）、evidence（最终证据，含父块上下文）、
              plan（路由 / 权重 / 查询变体 / 软硬过滤）、mode、elapsed_ms；
              封装在小对象里便于整体做 A/B（`compare_strategies()` 就是连调三次）。
        副作用：无缓存、无轨迹、不改引擎状态——可安全用于压测与对照实验。
        异常：不向外抛；无命中时返回 evidence 为空的 RetrievalResult。
        """
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
        """问答主链路（解析 / 切分 / 索引已在构造期完成，这里只做缓存、FAQ、闸门、检索与生成）。

        参数：
            question  用户问题（原样使用，不做改写；"查询变体"发生在检索层内部）
            top_k     取回的最终证据条数，None → `self.final_top_k`
            expr      元数据硬过滤表达式，None 由检索层按问题推断
            route     强制问题类型（clause / case / metric / general），None 自动分类
            use_cache 本次是否读写缓存（还需 `self.cache is not None` 才真正生效）
            use_faq   本次是否允许 FAQ 直出（还需 `self.enable_faq` 且 FAQ 索引存在）
            mode      检索模式 "hybrid"（默认）| "dense" | "bm25"
        返回：`AnswerResult`，按出口分三种 mode：
              "rag"      走完整检索 + 生成；
              "faq"      FAQ 直出（相似度过阈值且标识符一致）；
              "refused"  主体闸门拦截，或证据不足由生成层拒答。
              三条出口都带 run_id / total_latency_ms / timings / notes，字段口径一致。
        副作用：写 `runs/<run_id>.jsonl`（enable_trace 时）；把结果 `to_dict()` 回填缓存
              （use_cache 且缓存可用时，FAQ 与拒答结果同样回填）；更新 `self.last_run_id`。
        异常：不主动抛业务异常——检索失败表现为空证据（进而拒答），生成失败降级为抽取式作答
              并写进 notes；缓存与轨迹的异常在各自模块内部被吞掉。传入非法参数（如 mode 拼错）
              不会被校验，而是按检索层自己的分支处理。
        """
        started = time.perf_counter()
        run_id = new_run_id("ask")
        self.last_run_id = run_id
        recorder = TraceRecorder(run_id=run_id, runs_dir=self.runs_dir, enabled=self.enable_trace)
        timings: Dict[str, float] = {}
        notes: List[str] = []

        # ---- 1. 缓存 ----
        # 缓存键必须覆盖所有会影响结果的参数（问题/过滤/TopK/模式/路由），
        # 否则"换了过滤条件却命中旧答案"会返回错误结果且极难发现。
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
        # 命中 FAQ 也先不 return：还要补 total_ms、回填缓存、补写 done 轨迹，
        # 让"FAQ 出口"与"完整链路出口"在字段与轨迹上完全同形，调用方无需分情况处理。
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
        # 放在检索之前而不是生成之后：等生成完再发现主体不对，代价是白跑一遍检索与 LLM，
        # 而且模型很可能已经拿相近主体的资料"编"出了一个像模像样的答案。
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
            # 拒答也构造一个空引用账本：让 citation_check 在三种出口上口径一致
            # （used 为空 → traceable 为 False，即"拒答没有可追溯的引用"）
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
            # 注：这一行的 latency_ms 固定传 0.0——重排耗时已经算在 recall 那一步的
            # retrieval_ms 里（两者都在 `pipeline.run()` 内部），这里不重复计时。
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
        # 存 to_dict() 而不是对象本身：缓存后端可能是 Redis，必须只放可序列化内容；
        # 读回时由 `_answer_from_cache()` 负责重建（见其 docstring 的取舍说明）。
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

        参数：question 问题；top_k 每种策略取回的条数，None → `self.final_top_k`
              （内部用 `max(1, ...)` 兜底，避免 0 或负数导致空结果）。
        返回：dict——{question, top_k, strategies:{dense_only / bm25_only / hybrid:
              {source_ids, evidence（每条只截前 80 字）, elapsed_ms}}}。
              **三种策略都是单次 `search()`**，没有重排差异之外的额外变量。
        副作用：无（三次检索都不写缓存、不写轨迹）；异常：不抛，无命中即空列表。
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
    """把缓存里的 dict 还原成 AnswerResult（引用与证据都要能继续被使用）。

    参数：payload 缓存里存的 `AnswerResult.to_dict()`（缺字段/多字段都要能容忍）；
          run_id **本次**运行编号（不复用缓存里的旧编号，否则一次问两会共用同一个轨迹号）。
    返回：AnswerResult——citations / citation_check 被重建为对象，notes 固定为
          「（来自缓存的结果，未重新检索）」；`cache_hit` 由调用方置 True。
    副作用/异常：无（构造过程不碰缓存与文件）。
    注：实际实现为 `evidence` 固定为空列表、`retrieval` / `faithfulness` 固定为 None，
        也**没有**恢复 `timings` / `total_latency_ms` / `notes`——这些由 `ask()` 现场补写，
        因此缓存命中的结果拿不到"当时"的忠实度报告与分步耗时（这也是它快的原因）。
    """
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
    """按 dataclass 字段过滤后重建对象，容忍缓存里有新增/缺失字段。

    参数：cls 目标 dataclass 类型（本项目里实际只传 Citation / CitationCheck）；
          payload 缓存里反序列化出来的 dict。
    返回：`cls(**过滤后的字段)`；若 `cls` 不是 dataclass，则原样返回 payload（不抛错）。
          注：过滤只按**字段名**白名单，不做类型转换，因此缓存与代码版本不一致时
          可能得到字段类型不对的对象——这是"先能读出来，再谈兼容"的取舍；
          payload 里缺失的字段交给 dataclass 默认值兜底。
    副作用/异常：无；payload 里混入未知键会被静默丢弃（避免 `TypeError: unexpected keyword`）。
    """
    if not dataclasses.is_dataclass(cls):
        return payload
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in payload.items() if k in names})
