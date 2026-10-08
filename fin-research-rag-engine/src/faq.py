"""FAQ 模块：高频问题直出。

为什么 RAG 之外还要一个 FAQ
---------------------------
RAG 再快也快不过查表。真实业务里 80% 的提问会反复集中在少数几十个问题上
（「开户要什么材料」「产品起投金额是多少」「双录怎么走」）。
这些都走一遍「三路召回 + 重排 + 生成」是纯浪费——既慢又不稳定，
而它们的答案是**固定的、有人维护的、带更新日期的**。

所以分工是：
    FAQ     管高频、答案固定、更新由人负责的问题  → 直出，毫秒级
    RAG     管长尾、需要跨文档找依据的问题        → 走完整链路

命中的前提是**阈值**：相似度不够就老实走 RAG。
把长尾问题错当成 FAQ 直出，是最典型的"答得又快又错"。

在 RAG 全链路中的位置
--------------------
    缓存查询 → 【本模块：FAQ 直出（命中即返回）】 → 主体闸门 → 三路召回 → 重排 → 生成 → 校验

被谁调用：`src/engine.py` 的 `RAGEngine.ask()` 第 2 步（在缓存之后、主体闸门之前）调用
`FAQIndex.answer(question)`；命中则组装 `mode="faq"` 的 `AnswerResult` 并直接回填缓存，
**完全跳过检索与生成**。`FAQIndex.describe()` 进 `/health`；`tests/test_cache_faq.py` 断言打分与闸门口径。

输入：用户原问题（字符串）；`FAQEntry` 列表来自 `ingest.loader`（`corpus.faq`，JSON 里的人工维护条目）。
输出：`FAQMatch` / `answer()` 的 dict（答案 + 出处 + 更新时间 + 分数 + faq 明细），
      或 `None`（未达阈值 → 交给 RAG 链路，**这是默认路径而不是异常**）。

两道打分与闸门口径（阈值/常数与代码一致）
--------------------------------------
    lexical         `max(token 覆盖率, 问题二元组 Jaccard)` —— 两把尺子取大者，
                    「开户需要什么材料」token 覆盖高、bigram 低，反之亦然，只取一把会漏
    semantic        `cosine(问题向量, 该条问题向量)`，负值截到 0.0
    total           `lexical_weight * lexical + (1 - lexical_weight) * semantic`（默认 0.55 / 0.45）
    标识符闸门      问题里的等级/代码/数字（`identifiers()`）与该条 FAQ 的**无交集**时，
                    `total *= _ID_MISMATCH_PENALTY`（0.35）——总分上限是 1.0，乘完必然低于
                    默认阈值 `FAQ_MATCH_THRESHOLD`（0.62），于是这题老实走 RAG
    直出阈值        `best()` 要求 `score >= threshold`；`search()` **不套阈值**，只为观察边界案例
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .config import FAQ_MATCH_THRESHOLD, FAQ_TOP_K
from .index.embedding import EmbeddingBackend, cosine_scores, get_backend
from .ingest.loader import FAQEntry
from .utils.text import jaccard, shingles, tokenize

__all__ = ["FAQMatch", "FAQIndex", "identifiers"]

# 标识符：等级（C2/R4）、产品代码（WY2024-01）、金额（300）、期限（90）。
# 金融问答里这些是**不可替换**的：C1 和 C2 是两个完全不同的答案。
# 注：正则 `_ID_RE` 的取值口径是「1~4 个字母 + 1~4 位数字（可再接 - 两位以上数字）」或「2 位以上数字」，
# 命中后统一转大写去重；它只是一种**弱集合**判据（不区分"金额 300"与"期限 300"），
# 因此只用来做否决（不一致就压分），不用来做确认。
_ID_RE = re.compile(r"[A-Za-z]{1,4}\d{1,4}(?:-\d{2,})?|\d{2,}")
# 标识符不一致时的惩罚系数：把分数压到阈值以下，让它老实走 RAG
_ID_MISMATCH_PENALTY = 0.35


def identifiers(text: str) -> set:
    """抽取文本里的强标识符（等级 / 代码 / 数字）。

    参数：text 任意文本（None 按空串处理）。
    返回：set[str] —— `_ID_RE` 命中片段统一转大写后的集合（"c2" 与 "C2" 视为同一个）。
    副作用/异常：无；纯正则，不抛异常。
    """
    return {m.group(0).upper() for m in _ID_RE.finditer(text or "")}


@dataclass
class FAQMatch:
    """一条 FAQ 命中。得分要保留下来，方便解释"为什么这题被判成 FAQ"。

    字段：
        entry     命中的 `FAQEntry`（含 faq_id / question / answer / category / updated_at / source_id）
        score     最终分（已含标识符闸门的惩罚系数）
        lexical   词面分（token 覆盖率与二元组 Jaccard 取大者）
        semantic  语义分（余弦相似度，负值已截到 0）
    关键派生属性：`answer` —— 直接透出 `entry.answer`，调用方不必层层取属性。
    """

    entry: FAQEntry
    score: float
    lexical: float = 0.0
    semantic: float = 0.0

    @property
    def answer(self) -> str:
        """该条 FAQ 的答案原文。参数：无；返回：str。副作用/异常：无（直出人工维护的文案，不做改写）。"""
        return self.entry.answer

    def to_dict(self) -> Dict[str, object]:
        """导出为 dict（faq_id / question / answer / category / updated_at / source_id / 三分），供接口与轨迹。

        参数：无。
        返回：Dict[str, object]；三个分数都保留 4 位小数。
        副作用/异常：无。
        """
        return {
            "faq_id": self.entry.faq_id,
            "question": self.entry.question,
            "answer": self.entry.answer,
            "category": self.entry.category,
            "updated_at": self.entry.updated_at,
            "source_id": self.entry.source_id,
            "score": round(float(self.score), 4),
            "lexical": round(float(self.lexical), 4),
            "semantic": round(float(self.semantic), 4),
        }


class FAQIndex:
    """FAQ 索引：词面 + 语义双路打分，超过阈值才直出。

    关键属性（构造时一次性建好，之后只读，可安全并发查询）：
        entries          FAQ 条目列表（保持外部传入顺序）
        backend          嵌入后端（默认 `get_backend()`；本地哈希后端也能跑，零依赖）
        threshold        直出阈值，默认 `FAQ_MATCH_THRESHOLD`（配置默认 0.62）
        lexical_weight   词面权重，默认 0.55（语义权重 = 1 - 该值）
        _tokens          每条问题的 token 集合（词面打分之用）
        _bigrams         每条问题的二元组集合（短语级区分度）
        _matrix          每条问题的稠密向量矩阵；**没有条目时为 None**（`score()` 里有 None 判断）

    设计取舍：语义分**不预计算查询向量以外的任何东西**，每次 `score()` 只编码一次问题，
    因此单条 FAQ 的打分开销与条目数无关；问题在于 `search()` 会逐条打分，
    条目量大时应在外面先做粗筛——目前规模（几十条）下不值得引入这层复杂度。
    """

    def __init__(
        self,
        entries: Sequence[FAQEntry],
        backend: Optional[EmbeddingBackend] = None,
        threshold: float = FAQ_MATCH_THRESHOLD,
        lexical_weight: float = 0.55,
    ) -> None:
        """建索引：预切词、预编码全部 FAQ 问题。

        参数：entries FAQ 条目序列（可为空，空索引永不命中）；
              backend 嵌入后端，None 时用 `get_backend()`；
              threshold 直出阈值，默认 `FAQ_MATCH_THRESHOLD`；lexical_weight 词面权重，默认 0.55。
        返回：无（构造函数）。
        副作用：条目非空时会调用后端 `encode()`（本地哈希后端纯计算；bge-m3 后端首次会加载模型）。
        异常：不主动抛出；后端加载失败由 `get_backend()` 的降级逻辑处理。
        """
        self.entries: List[FAQEntry] = list(entries)
        self.backend = backend or get_backend()
        self.threshold = float(threshold)
        self.lexical_weight = float(lexical_weight)
        self._tokens: List[set] = [set(tokenize(e.question)) for e in self.entries]
        self._bigrams: List[set] = [shingles(tokenize(e.question), 2) for e in self.entries]
        self._matrix = (
            self.backend.encode([e.question for e in self.entries])
            if self.entries
            else None
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """FAQ 条目数。参数：无；返回：int；副作用/异常：无。"""
        return len(self.entries)

    def score(self, question: str, index: int) -> FAQMatch:
        """给「问题 × 第 index 条 FAQ」打分（本模块的核心口径，闸门也在这里）。

        打分步骤：
            1. 词面 `lexical = max(token 覆盖率, 问题二元组 Jaccard)`；
            2. 语义 `semantic = max(0.0, cosine(问题向量, 该条向量))`（无向量矩阵时保持 0.0）；
            3. `total = lexical_weight * lexical + (1 - lexical_weight) * semantic`；
            4. **标识符一致性闸门**：问题里的标识符集合非空、且与该条问题的标识符集合**无交集**时，
               `total *= _ID_MISMATCH_PENALTY`（0.35）。这条闸门防的是「C2 客户可以买什么」被
               「R2 可以卖给 C1 吗」这条 FAQ 抢答——看起来相关但答非所问，比慢一点危险得多。

        参数：question 用户原问题；index 该条 FAQ 在 `entries` 里的下标（不校验越界）。
        返回：`FAQMatch`（score / lexical / semantic 都留档，便于解释判定）。
        副作用/异常：会调用后端 `encode_one(question)`（本地后端纯计算）；下标越界时抛 `IndexError`。
        """
        entry = self.entries[index]
        q_tokens = tokenize(question)
        q_set, q_bigrams = set(q_tokens), shingles(q_tokens, 2)

        # 词面：token 覆盖率与二元组 Jaccard 取较大者——
        # 「开户需要什么材料」与「开户材料」token 覆盖高但 bigram 低，反之亦然
        coverage = (len(q_set & self._tokens[index]) / len(q_set)) if q_set else 0.0
        bigram = jaccard(q_bigrams, self._bigrams[index])
        lexical = max(coverage, bigram)

        semantic = 0.0
        # 空索引（_matrix 为 None）时不编码、语义分保持 0：此时词面分要独占阈值口径，
        # 不能白送 0.45 的语义权重，否则空索引下任何问题都会"看起来很像"。
        if self._matrix is not None and self._matrix.size:
            vec = self.backend.encode_one(question)
            semantic = max(0.0, float(cosine_scores(vec, self._matrix[index : index + 1])[0]))

        w = self.lexical_weight
        # 加权融合：词面权重默认 0.55 —— 金融问答的同义改写少、术语复述多，
        # 词面分数比语义分数更稳，语义分只用来兜"换了说法"的提问。
        total = w * lexical + (1.0 - w) * semantic

        # 标识符一致性闸门：问题里的等级 / 代码 / 数字必须在该 FAQ 条目里出现，
        # 否则「C2 客户可以买什么」会被「R2 可以卖给 C1 吗」这条 FAQ 抢答，
        # 直出的是一个**看起来相关但答非所问**的答案——比慢一点危险得多。
        q_ids = identifiers(question)
        if q_ids:
            entry_ids = identifiers(entry.question)
            if not (q_ids & entry_ids):
                total *= _ID_MISMATCH_PENALTY

        return FAQMatch(entry=entry, score=total, lexical=lexical, semantic=semantic)

    def search(self, question: str, top_k: int = FAQ_TOP_K) -> List[FAQMatch]:
        """返回按分数降序的候选（**不套阈值**，便于观察边界案例）。

        参数：question 用户原问题（空白问题直接返回空列表）；top_k 取前几条，默认 `FAQ_TOP_K`（配置默认 3）。
        返回：List[FAQMatch] —— 降序排列，长度 = `max(1, top_k)` 与条目数的较小值；
              **末尾可能低于 threshold**，判断是否直出必须用 `best()`。
        副作用/异常：逐条调用 `score()`（条目多时是全量线性扫描）。
        """
        if not self.entries or not question.strip():
            return []
        matches = [self.score(question, i) for i in range(len(self.entries))]
        matches.sort(key=lambda m: -m.score)
        return matches[: max(1, top_k)]

    def best(self, question: str) -> Optional[FAQMatch]:
        """取超过阈值的最佳命中；没有则返回 None（交给 RAG 链路）。

        参数：question 用户原问题。
        返回：`FAQMatch`（要求 `score >= self.threshold`，默认阈值 0.62）；否则 None。
        副作用/异常：无；内部只调 `search(top_k=1)`，不做额外计算。
        """
        top = self.search(question, top_k=1)
        if not top:
            return None
        return top[0] if top[0].score >= self.threshold else None

    def answer(self, question: str) -> Optional[Dict[str, object]]:
        """直出答案。返回结构与 RAG 答案对齐（带出处与更新时间）。

        参数：question 用户原问题。
        返回：Dict[str, object] —— `answer`（FAQ 原文）、`source="faq"`、`faq`（`FAQMatch.to_dict()`）、
              `updated_at`（空则「未标注」）、`source_id`、`score`（4 位小数）；
              未达阈值时返回 None。**注意 `source` 固定为 "faq"**，上层据此把 mode 标成 faq。
        副作用/异常：无。
        """
        match = self.best(question)
        if match is None:
            return None
        return {
            "answer": match.entry.answer,
            "source": "faq",
            "faq": match.to_dict(),
            "updated_at": match.entry.updated_at or "未标注",
            "source_id": match.entry.source_id,
            "score": round(match.score, 4),
        }

    def describe(self) -> Dict[str, object]:
        """自述信息，进 `/health` 的 `faq` 段（条目数 / 阈值 / 分类）。

        参数：无。
        返回：Dict[str, object] —— entries（条目数）、threshold、categories（分类去重排序，空值不计入）。
        副作用/异常：无。
        """
        categories = sorted({e.category for e in self.entries if e.category})
        return {
            "entries": len(self.entries),
            "threshold": self.threshold,
            "categories": categories,
        }
