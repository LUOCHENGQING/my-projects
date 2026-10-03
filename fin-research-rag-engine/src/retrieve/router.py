"""问题类型路由与元数据推断。

为什么必须路由
--------------
「制度条款类」问题和「风险案例类」问题，对召回排序的要求是**相反**的：

    问「第四十二条对合格投资者怎么规定」→ 要的是字面精确 → BM25 权重必须最高
    问「有没有类似的处罚案例」        → 要的是语义相似 → 稠密权重必须最高
    问「单一产品集中度上限是多少」    → 要的是条款里的数字 → 字面 + 词权重并重

一套权重打不通所有场景，这是本项目「混合检索替代单一向量检索后复杂问题召回率
提升 40% 以上」的直接来源——提升不是换模型换来的，是**按问题类型配召回策略**换来的。

同时这里做**元数据推断**：用户问「2024 年之后的产品说明」时，
如果能识别出机构、年份、资料类型，就在召回**之前**把候选集收窄，
而不是召回一大堆再让模型自己挑。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import DEFAULT_ROUTE, QUERY_ROUTE_WEIGHTS
from ..ingest.loader import Corpus

__all__ = ["QUERY_TYPES", "classify_question", "infer_filters", "expand_queries", "QueryPlan", "build_query_plan"]

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
_CLAUSE_RE = re.compile(r"第\s*[0-9一二三四五六七八九十百零]+\s*条")
_CODE_RE = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*(?:-\d{2,})+\b")
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
    if word in _WEAK_WORDS:
        return _WEAK_WEIGHT
    # 长词更有区分度
    return 1.0 + 0.1 * max(0, len(word) - 2)


def classify_question(question: str) -> str:
    """把问题分成 clause / case / metric / general 四类。"""
    if not question or not question.strip():
        return DEFAULT_ROUTE
    text = question.strip()

    # 出现条款号或产品代码：直接判条款类，这类问题的确定性最强
    if _CLAUSE_RE.search(text) or _CODE_RE.search(text):
        return "clause"

    scores: Dict[str, float] = {k: 0.0 for k in _PATTERNS}
    for label, words in _PATTERNS.items():
        for word in words:
            if word in text:
                scores[label] += _word_weight(word)

    best = max(scores.items(), key=lambda kv: kv[1])
    if best[1] <= 0.0:
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
    """
    conditions: Dict[str, object] = {}
    parts: List[str] = []
    text = question or ""

    if corpus is not None:
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
    """
    text = (question or "").strip()
    if not text:
        return []

    queries: List[str] = [text]
    core = re.sub(r"(请问|麻烦|帮我|想知道|是什么|有哪些|怎么样|如何|吗|呢|？|\?)", "", text).strip()
    if core and core != text and len(core) >= 4:
        queries.append(core)

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
    """一次检索的执行计划。写进轨迹，便于复现「当时为什么这么检索」。"""

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
        filter_expr=(filter_expr or "").strip(),
        soft_expr=inferred_expr,
        filters=inferred,
        queries=expand_queries(question, resolved_route),
        top_k=top_k,
    )
