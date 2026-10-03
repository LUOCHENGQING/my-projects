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
_ID_RE = re.compile(r"[A-Za-z]{1,4}\d{1,4}(?:-\d{2,})?|\d{2,}")
# 标识符不一致时的惩罚系数：把分数压到阈值以下，让它老实走 RAG
_ID_MISMATCH_PENALTY = 0.35


def identifiers(text: str) -> set:
    """抽取文本里的强标识符（等级 / 代码 / 数字）。"""
    return {m.group(0).upper() for m in _ID_RE.finditer(text or "")}


@dataclass
class FAQMatch:
    """一条 FAQ 命中。得分要保留下来，方便解释"为什么这题被判成 FAQ"。"""

    entry: FAQEntry
    score: float
    lexical: float = 0.0
    semantic: float = 0.0

    @property
    def answer(self) -> str:
        return self.entry.answer

    def to_dict(self) -> Dict[str, object]:
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
    """FAQ 索引：词面 + 语义双路打分，超过阈值才直出。"""

    def __init__(
        self,
        entries: Sequence[FAQEntry],
        backend: Optional[EmbeddingBackend] = None,
        threshold: float = FAQ_MATCH_THRESHOLD,
        lexical_weight: float = 0.55,
    ) -> None:
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
        return len(self.entries)

    def score(self, question: str, index: int) -> FAQMatch:
        entry = self.entries[index]
        q_tokens = tokenize(question)
        q_set, q_bigrams = set(q_tokens), shingles(q_tokens, 2)

        # 词面：token 覆盖率与二元组 Jaccard 取较大者——
        # 「开户需要什么材料」与「开户材料」token 覆盖高但 bigram 低，反之亦然
        coverage = (len(q_set & self._tokens[index]) / len(q_set)) if q_set else 0.0
        bigram = jaccard(q_bigrams, self._bigrams[index])
        lexical = max(coverage, bigram)

        semantic = 0.0
        if self._matrix is not None and self._matrix.size:
            vec = self.backend.encode_one(question)
            semantic = max(0.0, float(cosine_scores(vec, self._matrix[index : index + 1])[0]))

        w = self.lexical_weight
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
        """返回按分数降序的候选（**不套阈值**，便于观察边界案例）。"""
        if not self.entries or not question.strip():
            return []
        matches = [self.score(question, i) for i in range(len(self.entries))]
        matches.sort(key=lambda m: -m.score)
        return matches[: max(1, top_k)]

    def best(self, question: str) -> Optional[FAQMatch]:
        """取超过阈值的最佳命中；没有则返回 None（交给 RAG 链路）。"""
        top = self.search(question, top_k=1)
        if not top:
            return None
        return top[0] if top[0].score >= self.threshold else None

    def answer(self, question: str) -> Optional[Dict[str, object]]:
        """直出答案。返回结构与 RAG 答案对齐（带出处与更新时间）。"""
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
        categories = sorted({e.category for e in self.entries if e.category})
        return {
            "entries": len(self.entries),
            "threshold": self.threshold,
            "categories": categories,
        }
