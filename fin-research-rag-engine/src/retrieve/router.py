"""问题类型路由与元数据推断。

在 RAG 全链路中的位置
---------------------
    用户问题 → **本模块：分类 + 推断过滤 + 查询改写，产出 QueryPlan**
        → 三路召回（retrieve/hybrid.py，按 plan.weights 配权、按 plan.filter_expr 硬过滤）
        → 重排（retrieve/rerank.py，按 plan.route 应用资料类型先验）
        → 生成（src/generate）
本模块是**检索策略的决策层**：它不碰索引、不算分数，只回答"这个问题该用哪套权重、
该不该收窄候选集、要用哪几个查询去检索"。被谁调用：retrieve/hybrid.py 的
`retrieve()`（未显式传 plan 时）与 retrieve/pipeline.py 的 `run()`（每个问题都必建）。

对外关键对象
------------
    classify_question(question) -> str               问题类型：clause / case / metric / general
    infer_filters(question, corpus) -> (expr, dict)  推断过滤表达式与结构化条件
    expand_queries(question, route) -> List[str]     查询改写（确定性变体）
    build_query_plan(...) -> QueryPlan               汇总上述三者的执行计划
输入：问题字符串（infer_filters 还可选传语料对象以识别机构名 / 资料类型）；输出：QueryPlan。

为什么必须路由
--------------
「制度条款类」问题和「风险案例类」问题，对召回排序的要求是**相反**的：

    问「第四十二条对合格投资者怎么规定」→ 要的是字面精确 → BM25 权重必须最高
    问「有没有类似的处罚案例」        → 要的是语义相似 → 稠密权重必须最高
    问「单一产品集中度上限是多少」    → 要的是条款里的数字 → 字面 + 词权重并重

具体的三路权重（见 config.QUERY_ROUTE_WEIGHTS，改这里的说明必须同步改配置）：
    clause   bm25 0.55 / dense 0.20 / sparse 0.25   条款类：条款号、产品代码是强字面信号
    case     bm25 0.20 / dense 0.55 / sparse 0.25   案例类：换个说法还是同一类风险，靠语义
    metric   bm25 0.35 / dense 0.30 / sparse 0.35   指标类：既要命中条款又要保住数字与术语
    general  bm25 0.34 / dense 0.33 / sparse 0.33   通用：均衡（DEFAULT_ROUTE 默认取它）

一套权重打不通所有场景，这是本项目「混合检索替代单一向量检索后复杂问题召回率
提升 40% 以上」的直接来源——提升不是换模型换来的，是**按问题类型配召回策略**换来的。

同时这里做**元数据推断**：用户问「2024 年之后的产品说明」时，
如果能识别出机构、年份、资料类型，就在召回**之前**把候选集收窄，
而不是召回一大堆再让模型自己挑。

硬过滤 vs 软加权（本模块最重要的分级原则）
------------------------------------------
    用户**显式**给出的 filter_expr  → hard filter（召回前剔除）——用户说了算；
    从问题里**推断**出来的条件      → soft weight（只影响排序）——模型猜的有可能错。
原因很直接：**加错过滤比不过滤危险得多**。硬过滤是"静默丢证据"，
答案会凭空缺失且没有任何报错；软加权即使猜错了，也只是排序略有偏差。
唯一的例外是年份版本过滤的"有条件提升"（见 pipeline._promote_year_filter）：
只有当资料库确实存在该年份版本时，才敢把它提升为硬过滤。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import DEFAULT_ROUTE, QUERY_ROUTE_WEIGHTS
from ..ingest.loader import Corpus

__all__ = ["QUERY_TYPES", "classify_question", "infer_filters", "expand_queries", "QueryPlan", "build_query_plan"]

# 四类问题类型；不在这个集合里的路由名一律回落 DEFAULT_ROUTE
QUERY_TYPES = ("clause", "case", "metric", "general")

# 分类用的强特征词。刻意用「短语 + 正则」而不是训练一个分类器：
# 规则可读、可回归、可被业务方直接改，出错时能一眼看出是哪条规则误判。
_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "clause": (
        "第", "条", "款规定", "条款", "规定", "是否符合", "是否满足", "准入", "条件",
        "标准", "上限", "下限", "禁止", "不得", "应当", "可以吗", "合规", "管理办法",
        "适当性", "要求", "期限", "门槛", "可以购买", "可购买", "能不能买", "能买",
        "购买哪些", "保存多久", "多久",
    ),
    "case": (
        "案例", "处罚", "罚单", "违规", "警示函", "处分", "判例", "类似", "有没有",
        "发生过", "事件", "通报", "前车之鉴", "被罚",
    ),
    "metric": (
        "多少", "几成", "比例", "占比", "集中度", "费率", "收益", "收益率", "数值",
        "计算", "公式", "怎么算", "几个百分点", "上限是多少", "金额", "规模",
    ),
}
# 字面信号：条款号与产品代码。命中即可直接判条款类（确定性最强，无须打分）
_CLAUSE_RE = re.compile(r"第\s*[0-9一二三四五六七八九十百零]+\s*条")
_CODE_RE = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*(?:-\d{2,})+\b")
# 年份信号：先看区间（之后 / 之前），再退回"明确提到某一年"
_YEAR_RE = re.compile(r"(19|20)\d{2}\s*年?")
_AFTER_RE = re.compile(r"(20\d{2})\s*年\s*(?:之后|以后|以来|起)")
_BEFORE_RE = re.compile(r"(20\d{2})\s*年\s*(?:之前|以前|以前)")
# 「2023 年的……标准/规定/办法」这种句式里，年份限定的是**文档版本**；
# 而「示例集团 2023 年应收账款」里的年份限定的是**财务数据年度**。
# 两者必须区分：前者可以提升为版本硬过滤，后者一旦硬过滤就会把当期尽调档案全部筛掉。
_YEAR_VERSION_RE = re.compile(
    r"(20\d{2})\s*年[^，。？；]{0,12}?(?:标准|规定|办法|政策|要求|制度|条款|版本|细则|通知|指引|管理办法)"
)
_TOP_K_RE = re.compile(r"(?:前|top)\s*(\d{1,2})", re.IGNORECASE)

# 弱特征词：这些词在各类问题里都会出现，区分度低，因此权重压低。
# 不这么做的话，「…门槛是多少」会被 metric 的「多少」和 clause 的「门槛」拉成平局，
# 直接掉回通用策略，白白丢掉路由带来的召回增益。
_WEAK_WORDS = {"多少", "金额", "规模", "几个百分点", "什么", "怎么", "如何", "哪些"}
_WEAK_WEIGHT = 0.4


def _word_weight(word: str) -> float:
    """特征词的打分权重。

    参数：word 命中 _PATTERNS 的特征词。
    返回：弱特征词固定 0.4，其它为 1.0 + 0.1 × (词长 - 2)（长词更有区分度）。
    副作用：无。用途：避免「多少」这类高频词把分类拉成平局。
    """
    if word in _WEAK_WORDS:
        return _WEAK_WEIGHT
    # 长词更有区分度
    return 1.0 + 0.1 * max(0, len(word) - 2)


def classify_question(question: str) -> str:
    """把问题分成 clause / case / metric / general 四类。

    参数：question 用户问题。
    返回：类型名（一定是 QUERY_TYPES 中的一个；空问题、无特征词、最高分并列都返回 DEFAULT_ROUTE）。
    副作用：无（纯规则匹配，不加载模型，因此分类结果完全可复现、可回归测试）。
    """
    if not question or not question.strip():
        return DEFAULT_ROUTE
    text = question.strip()

    # 出现条款号或产品代码：直接判条款类，这类问题的确定性最强
    # （正则比特征词更可靠，命中就无须再比分数，也避免被其它类的词干扰）
    if _CLAUSE_RE.search(text) or _CODE_RE.search(text):
        return "clause"

    scores: Dict[str, float] = {k: 0.0 for k in _PATTERNS}
    for label, words in _PATTERNS.items():
        for word in words:
            # 用子串包含而不是分词匹配：保证「能不能买」这类口语短语也能命中
            if word in text:
                scores[label] += _word_weight(word)

    best = max(scores.items(), key=lambda kv: kv[1])
    if best[1] <= 0.0:
        # 一个特征词都没命中：不特化，走通用权重（宁可不特化，也不乱特化）
        return DEFAULT_ROUTE
    # 并列时保守回落到通用策略：宁可不特化，也不要往错的方向特化
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) > 1 and abs(ordered[0] - ordered[1]) < 1e-9:
        return DEFAULT_ROUTE
    return best[0]


def infer_filters(question: str, corpus: Optional[Corpus] = None) -> Tuple[str, Dict[str, object]]:
    """从问题里推断元数据过滤条件，返回 (表达式, 结构化条件)。

    识别三类：机构名、资料类型、年份（"2024年之后" / "2024年之前" / 明确提到 2024 年）。
    识别不到就返回空条件——**宁可不加过滤，也不要加错过滤**，
    因为加错过滤会静默丢证据，比不过滤危险得多。

    参数：question 用户问题；corpus 语料对象（提供 institutions / doc_types 词表；
          None 时跳过机构名与资料类型的识别）。
    返回：(过滤表达式, 结构化条件 dict)。表达式形如 `institution = "X" AND year >= 2024`；
          条件 dict 另含 `year` / `year_gte` / `year_lte` / `year_promotable` 等键。
    副作用：无。注意：返回的表达式本例中只用于**软加权**；只有用户显式给的
          filter_expr（或 pipeline 里"有条件提升"的年份）才会变成硬过滤。
    """
    conditions: Dict[str, object] = {}
    parts: List[str] = []
    text = question or ""

    if corpus is not None:
        # 必须"整词出现在问题里"才认：词表来自语料本身，拼不出语料里不存在的机构，
        # 所以这一步的误判率很低——但它仍然只做软加权，不硬过滤
        for institution in corpus.institutions:
            if institution and institution in text:
                conditions["institution"] = institution
                parts.append(f'institution = "{institution}"')
                break
        for doc_type in corpus.doc_types:
            if doc_type and doc_type in text:
                conditions["doc_type"] = doc_type
                parts.append(f'doc_type = "{doc_type}"')
                break

    after = _AFTER_RE.search(text)
    before = _BEFORE_RE.search(text)
    # 区间优先于单点：用户说"2024 年之后"时，不该退化成"就是 2024 年"
    if after:
        year = int(after.group(1))
        conditions["year_gte"] = year
        parts.append(f"year >= {year}")
    elif before:
        year = int(before.group(1))
        conditions["year_lte"] = year
        parts.append(f"year <= {year}")
    else:
        years = [int(m.group(0)[:4]) for m in _YEAR_RE.finditer(text)]
        if years:
            # 只取第一个年份：多个年份时无法判断用户要哪个，取最先提到的那个
            conditions["year"] = years[0]
            parts.append(f"year = {years[0]}")
            # 只有「年份限定的是文档版本」时，才允许后续提升为硬过滤
            version_hit = _YEAR_VERSION_RE.search(text)
            if version_hit and int(version_hit.group(1)) == years[0]:
                conditions["year_promotable"] = True

    return " AND ".join(parts), conditions


def expand_queries(question: str, route: str) -> List[str]:
    """查询改写：生成少量确定性变体，提升召回面。

    刻意不做「让模型改写查询」——那样每次改写的措辞不可控，检索结果无法复现，
    线上出了问题也没法定位。这里只用可解释的规则：
        * 原文（永远第一位）
        * 剥离疑问语气词后的核心短语
        * 抽出条款号 / 产品代码作为独立查询（字面精确匹配的强信号）

    参数：question 用户问题；route 问题类型（当前实现不做按类型分支，仅为接口稳定保留）。
    返回：去重保序后的查询列表，最多 4 条（配额限制，避免候选集被变体撑爆）。
    副作用：无。注意：变体越多该问题的 RRF 总分会越大，
         所以 hybrid.retrieve 里用 1/len(queries) 做了摊薄。
    """
    text = (question or "").strip()
    if not text:
        return []

    queries: List[str] = [text]
    core = re.sub(r"(请问|麻烦|帮我|想知道|是什么|有哪些|怎么样|如何|吗|呢|？|\?)", "", text).strip()
    # 长度阈值 4：太短的"核心短语"会退化成通用词，召回的噪声大于收益
    if core and core != text and len(core) >= 4:
        queries.append(core)

    # 条款号单独成一条查询：BM25 对它的命中是决定性信号，混在长句里会被稀释
    for m in _CLAUSE_RE.finditer(text):
        token = m.group(0).replace(" ", "")
        if token not in queries:
            queries.append(token)
    for m in _CODE_RE.finditer(text):
        if m.group(0) not in queries:
            queries.append(m.group(0))

    # 去重保序
    seen: set[str] = set()
    out: List[str] = []
    for q in queries:
        key = q.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out[:4]


@dataclass
class QueryPlan:
    """一次检索的执行计划。写进轨迹，便于复现「当时为什么这么检索」。

    关键属性：
        question       原始问题
        route          问题类型（clause / case / metric / general）
        weights        三路召回权重 {bm25, dense, sparse}，来自 QUERY_ROUTE_WEIGHTS[route]
        filter_expr    **硬过滤**表达式（召回前剔除）；只装用户显式给的，或被提升的年份
        soft_expr      从问题里推断出的**软过滤**表达式（只做排序加权，不剔除）
        promoted_year  年份是否被"有条件提升"为硬过滤
        filters        推断出的结构化条件（供 HybridRetriever.metadata_score 做软加权）
        queries        查询改写后的变体列表（原文永远在第一位）
        top_k          本次要取的条数
    """

    question: str
    route: str
    weights: Dict[str, float] = field(default_factory=dict)
    filter_expr: str = ""        # 用户显式给定的过滤条件 → 硬过滤（召回前剔除）
    soft_expr: str = ""          # 从问题里推断出的过滤条件 → 软加权（只影响排序）
    promoted_year: bool = False  # 显式年份被"提升"为硬过滤（资料库确实存在该年份版本时才提升）
    filters: Dict[str, object] = field(default_factory=dict)
    queries: List[str] = field(default_factory=list)
    top_k: int = 0

    def to_dict(self) -> Dict[str, object]:
        """把执行计划导出成 dict（权重保留 4 位小数），写入 runs/ 轨迹供复现与回归对比。"""
        return {
            "question": self.question,
            "route": self.route,
            "weights": {k: round(float(v), 4) for k, v in self.weights.items()},
            "filter_expr": self.filter_expr,
            "soft_expr": self.soft_expr,
            "promoted_year": self.promoted_year,
            "filters": self.filters,
            "queries": self.queries,
            "top_k": self.top_k,
        }


def build_query_plan(
    question: str,
    corpus: Optional[Corpus] = None,
    top_k: int = 0,
    route: Optional[str] = None,
    filter_expr: Optional[str] = None,
) -> QueryPlan:
    """构造检索计划：路由 + 权重 + 过滤条件 + 查询变体。

    过滤条件刻意分成两级：
        * 用户**显式**给的 `filter_expr` → 硬过滤，召回前直接剔除不符合的记录；
        * 从问题里**推断**出来的条件 → 软加权，只影响排序。
    这样即使推断错了（比如把「2024 年之前」听成了「2024 年」），也只是排序略有偏差，
    不会把正确答案整批剔掉——**加错过滤会静默丢证据，比不过滤危险得多**。

    参数：
        question     用户问题
        corpus       语料对象，用于识别机构名与资料类型；可为 None
        top_k        本次要取的条数（写进 plan，0 表示未指定）
        route        强制指定问题类型；None 时用 classify_question 自动分类
        filter_expr  用户显式给的过滤表达式（硬过滤；None / 空串表示无）
    返回：QueryPlan。分类结果不在 QUERY_ROUTE_WEIGHTS 里时回落 DEFAULT_ROUTE
          （防止上游传了拼错的路由名导致 KeyError）。
    副作用：无（不查询索引、不改全局状态）。
    """
    resolved_route = (route or classify_question(question)).lower()
    if resolved_route not in QUERY_ROUTE_WEIGHTS:
        resolved_route = DEFAULT_ROUTE
    weights = dict(QUERY_ROUTE_WEIGHTS[resolved_route])

    inferred_expr, inferred = infer_filters(question, corpus)

    return QueryPlan(
        question=question,
        route=resolved_route,
        weights=weights,
        # 硬过滤只认显式条件；推断出来的表达式进 soft_expr，不参与剔除
        filter_expr=(filter_expr or "").strip(),
        soft_expr=inferred_expr,
        filters=inferred,
        queries=expand_queries(question, resolved_route),
        top_k=top_k,
    )
