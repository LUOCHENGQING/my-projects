"""答案忠实度与相关性校验（RAGAS 风格的可量化指标）。

RAG 最容易翻车的地方不是"答不上来"，而是**答得很像但数字是编的**。
因此除了引用校验，还需要一层针对内容的确定性校验：

    数字忠实度  number_faithfulness   答案里的每个金额/比例/期限，是否都能在证据里找到
    句子支撑度  sentence_support      答案的每句话，是否都能在证据里找到语义支撑
    答案相关性  answer_relevance      答案与问题的贴合程度
    关键点覆盖  keyphrase_coverage    应回答的要点是否都答到了（评测集提供）

为什么不做成"再调一个模型来打分"：模型打分本身不可复现、不可回归，
CI 里没法用。这里的指标全部是确定性计算，因此可以卡门禁、可以逐次对比。
模型打分留作人工抽检，而不是自动化依赖。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..utils.text import amount_tokens, jaccard, split_sentences, tokenize

__all__ = [
    "NumberCheck",
    "FaithfulnessReport",
    "check_numbers",
    "sentence_support",
    "answer_relevance",
    "keyphrase_coverage",
    "evaluate_faithfulness",
    "claim_body",
    "strip_meta",
]

# 答案里「出处清单」段的起始标记。这一段是元数据（引用了哪些资料），
# 不是对事实的陈述，因此忠实度校验必须把它排除，否则会误报"未被证据支撑"。
SOURCE_SECTION_MARKERS = ("【出处与时效】", "【引用明细】", "【出处】")


def claim_body(answer: str) -> str:
    """截取答案中**属于事实陈述**的部分（去掉出处清单段）。"""
    text = answer or ""
    cut = len(text)
    for marker in SOURCE_SECTION_MARKERS:
        pos = text.find(marker)
        if pos >= 0:
            cut = min(cut, pos)
    return text[:cut].strip()

# 答案里允许出现的"非事实数字"：编号、条款序号、年份等
_IGNORABLE = re.compile(r"^\d{1,2}$")
_YEAR = re.compile(r"^(19|20)\d{2}$")


def check_numbers(answer: str, evidence_texts: Sequence[str]) -> "NumberCheck":
    """检查答案里的数字是否都能在证据中找到。

    这是本项目最硬的一条反幻觉规则：**数字只能来自证据**。
    模型自己算出的比例、自己补的金额，会在这一步被直接标出来。
    """
    evidence_blob = "\n".join(evidence_texts)
    evidence_tokens = set(amount_tokens(evidence_blob))
    supported: List[str] = []
    unsupported: List[str] = []

    for token in amount_tokens(answer):
        if _IGNORABLE.match(token) or _YEAR.match(token):
            continue
        if token in evidence_tokens:
            supported.append(token)
            continue
        # 允许千分位 / 小数位差异：1,286,400 与 1286400、128.6 与 128.60
        variants = {token, token.rstrip("0").rstrip(".") if "." in token else token}
        if any(v and v in evidence_blob.replace(",", "") for v in variants):
            supported.append(token)
            continue
        unsupported.append(token)

    return NumberCheck(supported=supported, unsupported=unsupported)

@dataclass
class NumberCheck:
    """数字校验结果。"""

    supported: List[str] = field(default_factory=list)
    unsupported: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unsupported

    @property
    def rate(self) -> float:
        total = len(self.supported) + len(self.unsupported)
        return len(self.supported) / total if total else 1.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "rate": round(self.rate, 4),
            "supported": self.supported,
            "unsupported": self.unsupported,
        }


# 答案句子里属于「元信息」的片段：引用编号与出处前缀。
# 校验一句话是否被证据支撑时，必须先把这些剥掉——它们本来就不在证据原文里。
_META_PATTERNS = (
    re.compile(r"\[\d{1,2}\]"),
    re.compile(r"（[^（）]{0,80}）"),
    re.compile(r"\([^()]{0,80}\)"),
    re.compile(r"【[^【】]{0,20}】"),
    re.compile(r"^[-•\s]+"),
)


def strip_meta(sentence: str) -> str:
    """剥掉句子里的引用编号与出处前缀，只留下事实陈述。"""
    out = sentence
    for pattern in _META_PATTERNS:
        out = pattern.sub(" ", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def sentence_support(answer: str, evidence_texts: Sequence[str], threshold: float = 0.34) -> Tuple[float, List[str]]:
    """计算答案中有多少句子能被证据支撑，返回 (支撑率, 未被支撑的句子)。"""
    sentences = [s for s in split_sentences(answer) if len(s) >= 8]
    if not sentences:
        return 1.0, []

    evidence_tokens = [set(tokenize(t)) for t in evidence_texts]
    unsupported: List[str] = []
    for sent in sentences:
        claim = strip_meta(sent)
        toks = set(tokenize(claim))
        if not toks:
            continue
        best = 0.0
        for ev in evidence_tokens:
            if not ev:
                continue
            overlap = len(toks & ev) / len(toks)
            best = max(best, overlap)
            if best >= threshold:
                break
        if best < threshold:
            unsupported.append(sent)

    supported = len(sentences) - len(unsupported)
    return supported / len(sentences), unsupported


def answer_relevance(answer: str, question: str) -> float:
    """答案与问题的 token 级 F1（不依赖任何模型，可重复、可回归）。"""
    q = set(tokenize(question))
    a = set(tokenize(answer))
    if not q or not a:
        return 0.0
    common = len(q & a)
    if common == 0:
        return 0.0
    precision = common / len(a)
    recall = common / len(q)
    return 2 * precision * recall / (precision + recall)


def keyphrase_coverage(answer: str, keyphrases: Sequence[str]) -> float:
    """评测集给的要点覆盖情况（例如必须提到"合格投资者""50 万元"）。"""
    if not keyphrases:
        return 1.0
    hits = sum(1 for phrase in keyphrases if phrase and phrase in answer)
    return hits / len(keyphrases)


@dataclass
class FaithfulnessReport:
    """一次答案的忠实度体检报告。"""

    numbers: NumberCheck = field(default_factory=NumberCheck)
    support_rate: float = 1.0
    unsupported_sentences: List[str] = field(default_factory=list)
    relevance: float = 0.0
    coverage: float = 1.0

    @property
    def faithful(self) -> bool:
        return self.numbers.ok and self.support_rate >= 0.8

    def to_dict(self) -> Dict[str, object]:
        return {
            "faithful": self.faithful,
            "numbers": self.numbers.to_dict(),
            "support_rate": round(self.support_rate, 4),
            "unsupported_sentences": self.unsupported_sentences[:5],
            "relevance": round(self.relevance, 4),
            "coverage": round(self.coverage, 4),
        }


def evaluate_faithfulness(
    answer: str,
    question: str,
    evidence_texts: Sequence[str],
    keyphrases: Sequence[str] = (),
) -> FaithfulnessReport:
    """一次算完全部确定性指标。

    只对**事实陈述部分**（`claim_body`）做校验：出处清单里的资料编号、日期、版本号
    不是对事实的断言，把它们算进忠实度会制造大量假阳性。
    """
    body = claim_body(answer) or answer
    support, unsupported = sentence_support(body, evidence_texts)
    return FaithfulnessReport(
        numbers=check_numbers(body, evidence_texts),
        support_rate=support,
        unsupported_sentences=unsupported,
        relevance=answer_relevance(body, question),
        coverage=keyphrase_coverage(body, keyphrases),
    )
