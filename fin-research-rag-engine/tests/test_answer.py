"""答案层测试：引用账本、引用校验、抽取式作答、拒答、忠实度指标。"""

from __future__ import annotations

import pytest

from src.answer import (
    REFUSAL_TEXT,
    AnswerGenerator,
    CitationLedger,
    LLMClient,
    answer_relevance,
    best_sentence,
    check_numbers,
    claim_body,
    evaluate_faithfulness,
    keyphrase_coverage,
    sentence_support,
    strip_meta,
)
from src.retrieve import Evidence


def make_evidence(
    child_id: str = "S#s0-c0",
    text: str = "合格投资者的金融资产不低于 300 万元。",
    score: float = 0.9,
    kind: str = "paragraph",
    **kwargs,
) -> Evidence:
    base = dict(
        evidence_id="E1",
        child_id=child_id,
        parent_id="S#s0",
        source_id="POL-2024-07",
        doc_id="POL-2024-07",
        title="资产管理产品适当性管理办法",
        section_title="第五章 合格投资者准入",
        institution="示例监管机构",
        doc_type="监管政策",
        effective_date="2024-07-01",
        version="v2",
        kind=kind,
        text=text,
        context=text,
        score=score,
        cross_score=0.6,
        rrf_score=0.02,
        metadata_score=1.0,
        routes_hit=["bm25", "dense"],
        matched_terms=["合格", "投资"],
    )
    base.update(kwargs)
    return Evidence(**base)


# ---------------------------------------------------------------------------
# 引用账本
# ---------------------------------------------------------------------------
def test_ledger_allocates_sequential_numbers(ledger):
    first = ledger.allocate(make_evidence("a"))
    second = ledger.allocate(make_evidence("b"))
    assert first.citation_no == 1
    assert second.citation_no == 2
    assert len(ledger) == 2


def test_ledger_reuses_number_for_same_chunk(ledger):
    a1 = ledger.allocate(make_evidence("a"))
    a2 = ledger.allocate(make_evidence("a"))
    assert a1 is a2
    assert len(ledger) == 1


def test_ledger_citation_carries_traceable_chain(ledger):
    citation = ledger.allocate(make_evidence("a"))
    assert citation.source_id == "POL-2024-07"
    assert citation.parent_id == "S#s0"
    assert citation.label.endswith("第五章 合格投资者准入")
    assert citation.updated_at == "2024-07-01"


def test_ledger_allocate_all(ledger):
    citations = ledger.allocate_all([make_evidence("a"), make_evidence("b")])
    assert [c.citation_no for c in citations] == [1, 2]


def test_ledger_validate_accepts_known_numbers(ledger):
    ledger.allocate_all([make_evidence("a"), make_evidence("b")])
    check = ledger.validate("结论一 [1]，结论二 [2]。")
    assert check.ok
    assert check.used == [1, 2]
    assert check.dangling == []


def test_ledger_validate_strips_dangling_citations(ledger):
    ledger.allocate(make_evidence("a"))
    check = ledger.validate("结论一 [1]，编造的出处 [7]。")
    assert check.dangling == [7]
    assert "[7]" not in check.cleaned_text
    assert "[1]" in check.cleaned_text
    assert not check.ok
    assert ledger.dropped and ledger.dropped[0]["citation_no"] == "7"


def test_ledger_validate_reports_unused(ledger):
    ledger.allocate_all([make_evidence("a"), make_evidence("b")])
    check = ledger.validate("只用了 [1]。")
    assert check.unused == [2]


def test_ledger_get_and_by_source(ledger):
    ledger.allocate_all([make_evidence("a"), make_evidence("b", source_id="INS-2024-09")])
    assert ledger.get(1).source_id == "POL-2024-07"
    assert ledger.get(99) is None
    assert len(ledger.by_source("INS-2024-09")) == 1


def test_ledger_render_contains_source_and_date(ledger):
    ledger.allocate(make_evidence("a"))
    rendered = ledger.render()
    assert "[1]" in rendered
    assert "POL-2024-07" in rendered
    assert "2024-07-01" in rendered


def test_ledger_render_subset(ledger):
    ledger.allocate_all([make_evidence("a"), make_evidence("b")])
    rendered = ledger.render([2])
    assert "[2]" in rendered and "[1]" not in rendered


def test_ledger_known_numbers_property(ledger):
    ledger.allocate_all([make_evidence("a"), make_evidence("b")])
    assert ledger.known_numbers == {1, 2}


def test_ledger_to_dict(ledger):
    ledger.allocate(make_evidence("a"))
    payload = ledger.to_dict()
    assert payload["count"] == 1
    assert payload["citations"][0]["citation_no"] == 1


# ---------------------------------------------------------------------------
# 抽取式作答
# ---------------------------------------------------------------------------
def test_best_sentence_prefers_query_overlap():
    text = "本条与问题无关。合格投资者的金融资产不低于 300 万元。另一句也无关。"
    assert "300 万元" in best_sentence(text, ["合格投资者", "金融资产"])


def test_best_sentence_falls_back_to_first_when_no_overlap():
    result = best_sentence("第一句话内容。第二句话内容。", ["完全不相关"])
    assert result.startswith("第一句")


def test_generator_produces_structured_extractive_answer(generator):
    answer = generator.generate("合格投资者的门槛是多少？", [make_evidence("a")])
    assert "【结论】" in answer.text
    assert "【依据】" in answer.text
    assert "【出处与时效】" in answer.text
    assert answer.mode == "mock-extractive"
    assert answer.traceable
    assert answer.citations


def test_generator_refuses_without_evidence(generator):
    answer = generator.generate("完全无关的问题", [])
    assert answer.refused
    assert answer.mode == "mock"
    assert answer.text == REFUSAL_TEXT
    assert answer.citations == []


def test_generator_never_invents_numbers(generator):
    answer = generator.generate("合格投资者的门槛是多少？", [make_evidence("a")])
    check = check_numbers(answer.text, [make_evidence("a").text])
    assert check.ok


def test_generator_deduplicates_citation_numbers(generator):
    evidence = [make_evidence("a"), make_evidence("a")]
    answer = generator.generate("合格投资者的门槛是多少？", evidence)
    assert answer.citation_numbers == sorted(set(answer.citation_numbers))


def test_generator_uses_real_llm_when_available():
    class FakeLLM(LLMClient):
        def __init__(self):
            super().__init__(force_mock=False)
            self.model = "fake"

        @property
        def available(self):
            return True

        def complete(self, system, user, temperature=0.0):
            from src.answer.llm import LLMResult

            return LLMResult(text="【结论】根据资料 [1] 门槛为 300 万元。", mode="openai", model="fake")

    answer = AnswerGenerator(FakeLLM()).generate("门槛是多少？", [make_evidence("a")])
    assert answer.mode == "openai"
    assert answer.traceable


def test_generator_falls_back_when_llm_errors():
    class BrokenLLM(LLMClient):
        def __init__(self):
            super().__init__(force_mock=False)

        @property
        def available(self):
            return True

        def complete(self, system, user, temperature=0.0):
            from src.answer.llm import LLMResult

            return LLMResult(text="", mode="openai", error="ConnectionError: 网络不可达")

    answer = AnswerGenerator(BrokenLLM()).generate("门槛是多少？", [make_evidence("a")])
    assert answer.mode == "fallback-extractive"
    assert answer.traceable
    assert any("降级" in note for note in answer.notes)


def test_generator_marks_low_overlap_answers_for_review():
    """问题措辞与原文差异大时不硬编，但仍给出检索得分最高的原文并提示人工确认。"""
    unrelated = make_evidence("z", text="这是一段与问题完全没有词面交集的制度描述文字。")
    answer = AnswerGenerator(LLMClient(force_mock=True)).generate("量子计算机如何实现纠错？", [unrelated])
    assert not answer.refused
    assert answer.traceable
    assert "人工确认" in answer.text
    assert "制度描述文字" in answer.text


def test_generated_answer_to_dict(generator):
    payload = generator.generate("门槛是多少？", [make_evidence("a")]).to_dict()
    assert payload["traceable"] is True
    assert payload["citations"]


# ---------------------------------------------------------------------------
# 忠实度
# ---------------------------------------------------------------------------
def test_check_numbers_flags_unsupported_values():
    evidence = ["合格投资者的金融资产不低于 300 万元。"]
    assert check_numbers("门槛是 300 万元。", evidence).ok
    bad = check_numbers("门槛是 500 万元。", evidence)
    assert not bad.ok
    assert "500" in bad.unsupported


def test_check_numbers_ignores_years_and_small_ordinals():
    assert check_numbers("2024 年第一条", ["完全不同的证据文本"]).ok


def test_check_numbers_rate():
    check = check_numbers("金额 300 万元，比例 500%。", ["金额 300 万元"])
    assert 0.0 < check.rate < 1.0


def test_sentence_support_detects_unsupported_claims():
    rate, unsupported = sentence_support(
        "合格投资者的金融资产不低于 300 万元。明天股市一定会上涨并创新高。",
        ["合格投资者的金融资产不低于 300 万元。"],
    )
    assert rate < 1.0
    assert unsupported


def test_sentence_support_strips_citation_metadata():
    rate, _ = sentence_support(
        "[1] 合格投资者的金融资产不低于 300 万元。",
        ["合格投资者的金融资产不低于 300 万元。"],
    )
    assert rate == 1.0


def test_strip_meta_removes_brackets_and_parentheses():
    cleaned = strip_meta("- （原文｜示例银行 · 办法）第一条 内容。 [2]")
    assert "原文" not in cleaned
    assert "第一条 内容。" in cleaned


def test_answer_relevance_bounds():
    assert answer_relevance("合格投资者金融资产门槛", "合格投资者的金融资产门槛是多少？") > 0.5
    assert answer_relevance("完全无关内容", "合格投资者门槛") == 0.0


def test_keyphrase_coverage():
    assert keyphrase_coverage("门槛是 300 万元", ["300 万元", "50 万元"]) == 0.5
    assert keyphrase_coverage("任意", []) == 1.0


def test_claim_body_drops_source_section():
    text = "【结论】内容 [1]。\n\n【出处与时效】[1] 某资料 2024-01-01"
    body = claim_body(text)
    assert "出处与时效" not in body
    assert "内容" in body


def test_evaluate_faithfulness_full_report():
    report = evaluate_faithfulness(
        answer="【结论】合格投资者的金融资产不低于 300 万元 [1]。\n\n【出处与时效】[1] 某资料",
        question="合格投资者的金融资产门槛是多少？",
        evidence_texts=["合格投资者的金融资产不低于 300 万元。"],
        keyphrases=["300 万元"],
    )
    payload = report.to_dict()
    assert report.faithful
    assert payload["numbers"]["ok"] is True
    assert payload["coverage"] == 1.0
    assert 0.0 <= payload["relevance"] <= 1.0
