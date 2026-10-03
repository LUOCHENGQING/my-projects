"""答案生成：确定性抽取式作答 + 真实 LLM 作答。

为什么要有「抽取式」这条路径
----------------------------
很多 RAG 项目演示时必须配 API Key，一旦额度用完或断网，整套东西就"讲不清了"。
本项目的做法是：**生成层可替换，链路层不可省**。

    mock 路径：从证据里抽取最相关的原句，按固定结构组织成答案。
               因为它只搬运原文、不做任何计算，答案的每个数字都能在证据里找到——
               忠实度是**结构性保证**，不是靠提示词祈祷。
    真实路径：走 OpenAI 兼容接口，提示词里写死"数字只能来自证据、不足就拒答"，
               返回后再过一遍 `citations.validate()` 与忠实度校验。

两条路径共用同一套引用编号、同一套出处渲染、同一套校验，
因此"换个模型"不会改变答案的可追溯性——这正是把检索与生成解耦的意义。

拒绝作答也是能力：资料库里没有依据时，正确答案是「现有资料不足以回答」，
而不是一段听起来很专业的编造。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..retrieve.pipeline import Evidence
from ..utils.text import split_sentences, tokenize
from .citations import Citation, CitationCheck, CitationLedger
from .llm import SYSTEM_PROMPT, LLMClient, build_user_prompt
__all__ = ["GeneratedAnswer", "AnswerGenerator", "best_sentence", "REFUSAL_TEXT"]

REFUSAL_TEXT = "现有资料不足以回答该问题。请补充相关制度文件、产品说明书或风险案例后再试。"


def best_sentence(text: str, query_tokens: Sequence[str], fallback_limit: int = 160) -> str:
    """从一段文本里挑出与问题最相关的一句（抽取式作答的基本单元）。

    打分只看「查询词在句子里出现了多少」，不做任何生成，因此
    **选出来的句子一定是原文**，不会引入幻觉。

    入参的 query_tokens 允许是「词」也允许是「已经切好的 token」：内部统一再切一次，
    否则传入「合格投资者」这种整词时会与句子切出来的二元组对不上，永远判为不相关。
    """
    sentences = split_sentences(text)
    if not sentences:
        stripped = text.strip()
        return stripped[:fallback_limit] + ("…" if len(stripped) > fallback_limit else "")

    wanted: set = set()
    for item in query_tokens or ():
        wanted.update(tokenize(item))
    if not wanted:
        return sentences[0]

    best, best_score = sentences[0], -1.0
    for pos, sent in enumerate(sentences):
        tokens = set(tokenize(sent))
        overlap = len(wanted & tokens)
        # 位置轻微加权：同类句子靠前的通常是定义 / 结论
        score = overlap * 1.0 + (0.05 if pos == 0 else 0.0)
        if score > best_score:
            best, best_score = sent, score
    return best


@dataclass
class GeneratedAnswer:
    """一次生成的完整结果：答案正文 + 引用账本 + 校验结论。"""

    question: str
    text: str
    mode: str = "mock"
    citations: List[Citation] = field(default_factory=list)
    check: Optional[CitationCheck] = None
    evidence: List[Evidence] = field(default_factory=list)
    refused: bool = False
    notes: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    llm: Dict[str, object] = field(default_factory=dict)

    @property
    def citation_numbers(self) -> List[int]:
        return [c.citation_no for c in self.citations]

    @property
    def source_ids(self) -> List[str]:
        seen: List[str] = []
        for c in self.citations:
            if c.source_id not in seen:
                seen.append(c.source_id)
        return seen

    @property
    def traceable(self) -> bool:
        """引用是否全部可追溯：每个 [n] 都能回到真实证据，且至少用到一条。"""
        if self.check is None:
            return False
        return not self.check.dangling and bool(self.check.used)

    def to_dict(self) -> Dict[str, object]:
        return {
            "question": self.question,
            "answer": self.text,
            "mode": self.mode,
            "refused": self.refused,
            "traceable": self.traceable,
            "citation_numbers": self.citation_numbers,
            "source_ids": self.source_ids,
            "citations": [c.to_dict() for c in self.citations],
            "check": self.check.to_dict() if self.check else None,
            "notes": list(self.notes),
            "latency_ms": round(self.latency_ms, 3),
            "llm": self.llm,
        }


class AnswerGenerator:
    """按可用性选择作答路径，并对结果做引用与忠实度校验。"""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        max_bullets: int = 3,
        min_evidence: int = 1,
    ) -> None:
        self.llm = llm or LLMClient()
        self.max_bullets = int(max_bullets)
        self.min_evidence = int(min_evidence)

    # ------------------------------------------------------------------
    def generate(
        self,
        question: str,
        evidence: Sequence[Evidence],
        plan_note: str = "",
    ) -> GeneratedAnswer:
        started = time.perf_counter()
        items = list(evidence)
        ledger = CitationLedger()

        if len(items) < self.min_evidence:
            check = ledger.validate("")
            return GeneratedAnswer(
                question=question,
                text=REFUSAL_TEXT,
                mode=self.llm.mode,
                citations=[],
                check=check,
                evidence=[],
                refused=True,
                notes=["证据不足，按拒答策略返回"],
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        if self.llm.available:
            # 真实模型可能引用任意一条证据，因此先把编号全部备好
            ledger.allocate_all(items)
            result = self.llm.complete(SYSTEM_PROMPT, build_user_prompt(question, items, plan_note))
            if result.error or not result.text.strip():
                notes = [f"真实模型不可用，已降级为抽取式作答：{result.error}"]
                text, mode = self._extractive(question, items, ledger), "fallback-extractive"
            else:
                text, mode = result.text.strip(), "openai"
                notes = []
        else:
            text, mode = self._extractive(question, items, ledger), "mock-extractive"
            notes = ["mock 模式：抽取式作答，答案中的每句话都来自证据原文"]

        # ---- 结构性校验：悬空引用当场剔除 ----
        check = ledger.validate(text)
        text = check.cleaned_text
        if check.dangling:
            notes.append(f"已剔除 {len(check.dangling)} 处悬空引用（模型编造了不存在的出处编号）")
        if mode != "mock-extractive":
            # 真实模型路径：补上出处清单，保证格式统一
            rendered = ledger.render(check.used)
            if rendered and "出处与时效" in text and rendered not in text:
                text = text.rstrip() + "\n\n" + rendered

        used = [c for c in ledger.citations if c.citation_no in set(check.used)]
        return GeneratedAnswer(
            question=question,
            text=text,
            mode=mode,
            citations=used,
            check=check,
            evidence=items,
            refused=False,
            notes=notes,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            llm={"mode": self.llm.mode, "model": self.llm.model},
        )

    # ------------------------------------------------------------------
    def _extractive(self, question: str, evidence: Sequence[Evidence], ledger: CitationLedger) -> str:
        """确定性抽取式作答：只搬运原文句子，因此不会产生数字幻觉。

        选句策略是**全局择优**而不是"每条证据各挑一句"：把所有证据里的句子放在一起
        按与问题的相关性排序，取前 N 句。差别很大——逐条挑会硬凑出"每条都要说一句"，
        结果把相关性很低的句子也塞进依据里；全局择优只会保留真正回答问题的句子。

        引用编号**按需分配**：只有真正被写进答案的证据才占编号，
        这样「出处与时效」清单不会出现一堆没被用到的来源。
        """
        q_tokens = set(tokenize(question))
        scored: List[tuple] = []
        seen_sentences: set = set()

        for item in evidence:
            # 表格子块按**整块**作为候选：一行「条件项: 金融资产；标准: 不低于 300 万元。」
            # 如果按「；」切开，「条件项: 金融资产」与「标准: 不低于 300 万元」会被拆成两句，
            # 引用时只显示半句，业务人员看到的就是「条件项: 金融资产；」这种没头没尾的话。
            if item.kind == "table" and len(item.text) <= 260:
                sentences = [item.text.strip()]
            else:
                sentences = split_sentences(item.text)

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10 or sentence in seen_sentences:
                    continue
                overlap = len(q_tokens & set(tokenize(sentence)))
                if overlap <= 0:
                    continue
                seen_sentences.add(sentence)
                scored.append((overlap, -len(sentence), item, sentence))

        # 相关性优先，同分时短句优先（更聚焦），再按证据顺序稳定排序
        scored.sort(key=lambda row: (-row[0], -row[1], row[2].evidence_id))

        if not scored:
            # 没有任何句子与问题有词面交集：不硬凑，改用检索得分最高的那条原文，
            # 并明确标注"措辞差异较大，建议人工确认"——拒答与否由检索层决定，
            # 抽取层只负责"照抄最相关的一句"，不越权做相关性判断。
            top_item = evidence[0]
            top_cite = self._cite_for(ledger, top_item)
            sentence = best_sentence(top_item.text, [question])
            lines = [
                "【结论】",
                f"{sentence} [{top_cite.citation_no}]",
                "",
                "【依据】",
                f"- （检索得分最高｜{top_item.citation_label}）{sentence} [{top_cite.citation_no}]",
                "- 注：问题措辞与资料原文差异较大，以上为检索得分最高的原文条款，建议人工确认。",
                "",
                "【出处与时效】",
                ledger.render() or "（无）",
            ]
            return "\n".join(lines)

        top_item, top_sentence = scored[0][2], scored[0][3]
        top_cite = self._cite_for(ledger, top_item)
        lines: List[str] = ["【结论】", f"{top_sentence} [{top_cite.citation_no}]", "", "【依据】"]

        emitted = {top_sentence}
        for _, _, item, sentence in scored[1:]:
            if len(emitted) > self.max_bullets:
                break
            cite = self._cite_for(ledger, item)
            tag = "条款/表格" if item.kind == "table" else "原文"
            lines.append(f"- （{tag}｜{item.citation_label}）{sentence} [{cite.citation_no}]")
            emitted.add(sentence)

        if len(lines) == 4:
            lines.append("- （本次命中的证据内容与结论一致，未发现其他相互印证的条款）")

        lines.extend(["", "【出处与时效】"])
        rendered = ledger.render()
        lines.append(rendered if rendered else "（无）")
        return "\n".join(lines)

    @staticmethod
    def _cite_for(ledger: CitationLedger, item: Evidence) -> Citation:
        """按 child_id 取回引用；不存在则新分配（保证 `[n]` 与账本一一对应）。"""
        for citation in ledger.citations:
            if citation.child_id == item.child_id:
                return citation
        return ledger.allocate(item)
