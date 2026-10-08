"""答案层测试：引用账本、引用校验、抽取式作答、拒答、忠实度指标。

覆盖的被测模块
--------------
    src.answer.citations    CitationLedger 的编号签发 / 查询 / 悬空引用校验 / 出处渲染
    src.answer.generator    AnswerGenerator 的抽取式作答、拒答、真实 LLM 路径与降级、best_sentence
    src.answer.faithfulness check_numbers / sentence_support / answer_relevance /
                            keyphrase_coverage / claim_body / strip_meta / evaluate_faithfulness

覆盖策略
--------
    正常路径：编号顺序签发、结构化三段答案、真实 LLM 作答（用子类假模型顶掉网络调用）。
    边界：同一子块重复分配、只渲染编号子集、空证据列表、问题与原文无词面交集、空要点列表。
    异常 / 降级：LLM 返回 error 或空文本时降级为抽取式作答并留下说明。
    对抗 / 拒答：模型编造账本外编号（幻觉引用）必须被剔除；无证据必须返回固定拒答文案；
                答案中出现证据里没有的数字必须被判为不忠实。
所有用例都跑在 mock 环境（见 conftest._force_mock_env），不依赖网络与 API Key，指标可精确断言。
"""

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
    """构造一条可追溯的 Evidence：默认值照抄制度原文里的一句「合格投资者」条款。

    参数 kwargs 用于按需覆盖任意字段（child_id / source_id / text / kind 等），
    这样各用例只需声明自己关心的差异，其余字段保持同一份可信基线。
    """
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
    """引用编号必须由账本按分配顺序连续签发，从 1 起且不跳号。"""
    first = ledger.allocate(make_evidence("a"))
    second = ledger.allocate(make_evidence("b"))
    assert first.citation_no == 1
    assert second.citation_no == 2
    assert len(ledger) == 2


def test_ledger_reuses_number_for_same_chunk(ledger):
    """同一子块只允许占一个编号：重复分配必须返回同一条引用记录，不允许新增。"""
    a1 = ledger.allocate(make_evidence("a"))
    a2 = ledger.allocate(make_evidence("a"))
    assert a1 is a2
    assert len(ledger) == 1


def test_ledger_citation_carries_traceable_chain(ledger):
    """引用记录必须带齐回溯链路：source_id → parent_id → 出处标签 → 更新/生效日期。"""
    citation = ledger.allocate(make_evidence("a"))
    assert citation.source_id == "POL-2024-07"
    assert citation.parent_id == "S#s0"
    assert citation.label.endswith("第五章 合格投资者准入")
    assert citation.updated_at == "2024-07-01"


def test_ledger_allocate_all(ledger):
    """批量分配与逐条分配必须给出相同的编号序列（编号只取决于分配顺序）。"""
    citations = ledger.allocate_all([make_evidence("a"), make_evidence("b")])
    assert [c.citation_no for c in citations] == [1, 2]


def test_ledger_validate_accepts_known_numbers(ledger):
    """账本内编号必须全部通过校验，并被登记为「已使用」，不得误报为悬空或未用。"""
    ledger.allocate_all([make_evidence("a"), make_evidence("b")])
    check = ledger.validate("结论一 [1]，结论二 [2]。")
    assert check.ok
    assert check.used == [1, 2]
    assert check.dangling == []


def test_ledger_validate_strips_dangling_citations(ledger):
    """账本外编号属于幻觉引用，必须从答案正文剔除、并把剔除原因留痕到 dropped。"""
    ledger.allocate(make_evidence("a"))
    # 只签发过 [1]，正文里的 [7] 是凭空编造的出处
    check = ledger.validate("结论一 [1]，编造的出处 [7]。")
    assert check.dangling == [7]
    assert "[7]" not in check.cleaned_text
    assert "[1]" in check.cleaned_text
    assert not check.ok
    assert ledger.dropped and ledger.dropped[0]["citation_no"] == "7"


def test_ledger_validate_reports_unused(ledger):
    """签发了却没被答案引用的编号要单独标为 unused（供出处清单裁剪），而不是报错。"""
    ledger.allocate_all([make_evidence("a"), make_evidence("b")])
    check = ledger.validate("只用了 [1]。")
    assert check.unused == [2]


def test_ledger_get_and_by_source(ledger):
    """两条查询路径都要可用：按编号取单条（未知编号返回 None）、按 source_id 聚合。"""
    ledger.allocate_all([make_evidence("a"), make_evidence("b", source_id="INS-2024-09")])
    assert ledger.get(1).source_id == "POL-2024-07"
    assert ledger.get(99) is None
    assert len(ledger.by_source("INS-2024-09")) == 1


def test_ledger_render_contains_source_and_date(ledger):
    """出处清单必须自解释：编号 + 资料编号 + 更新/生效日期，业务据此才能复核原文。"""
    ledger.allocate(make_evidence("a"))
    rendered = ledger.render()
    assert "[1]" in rendered
    assert "POL-2024-07" in rendered
    assert "2024-07-01" in rendered


def test_ledger_render_subset(ledger):
    """render 必须支持只渲染指定编号子集：未被点名的条目不得出现（供 unused 裁剪）。"""
    ledger.allocate_all([make_evidence("a"), make_evidence("b")])
    rendered = ledger.render([2])
    assert "[2]" in rendered and "[1]" not in rendered


def test_ledger_known_numbers_property(ledger):
    """known_numbers 必须恰好等于账本已签发编号的集合——它是悬空判定的唯一依据。"""
    ledger.allocate_all([make_evidence("a"), make_evidence("b")])
    assert ledger.known_numbers == {1, 2}


def test_ledger_to_dict(ledger):
    """账本导出的摘要必须带总数与逐条编号，供接口响应与轨迹落盘消费。"""
    ledger.allocate(make_evidence("a"))
    payload = ledger.to_dict()
    assert payload["count"] == 1
    assert payload["citations"][0]["citation_no"] == 1


# ---------------------------------------------------------------------------
# 抽取式作答
# ---------------------------------------------------------------------------
def test_best_sentence_prefers_query_overlap():
    """抽取式选句的排序规则：与问题词面交集多的句子优先于无关句子。"""
    text = "本条与问题无关。合格投资者的金融资产不低于 300 万元。另一句也无关。"
    assert "300 万元" in best_sentence(text, ["合格投资者", "金融资产"])


def test_best_sentence_falls_back_to_first_when_no_overlap():
    """全部句子都没有词面交集时，必须退回首句而不是返回空串（保证总有候选可引用）。

    注：实际实现为——并没有专门的「无交集」分支，而是首句带 0.05 位置加权，
    在全员 0 分时靠这一点加权胜出；用例名里的 fallback 指的是结果表现，不是代码路径。
    """
    # 查询词与两句话均无交集，此时只有首句的位置加权能起作用
    result = best_sentence("第一句话内容。第二句话内容。", ["完全不相关"])
    assert result.startswith("第一句")


def test_generator_produces_structured_extractive_answer(generator):
    """mock 路径的答案必须是固定三段结构（结论 / 依据 / 出处与时效），且整体可追溯。"""
    answer = generator.generate("合格投资者的门槛是多少？", [make_evidence("a")])
    assert "【结论】" in answer.text
    assert "【依据】" in answer.text
    assert "【出处与时效】" in answer.text
    assert answer.mode == "mock-extractive"
    assert answer.traceable
    assert answer.citations


def test_generator_refuses_without_evidence(generator):
    """证据不足时必须拒答：返回固定拒答文案，且不得携带任何引用编号。"""
    answer = generator.generate("完全无关的问题", [])
    assert answer.refused
    assert answer.mode == "mock"
    assert answer.text == REFUSAL_TEXT
    assert answer.citations == []


def test_generator_never_invents_numbers(generator):
    """抽取式作答的数字忠实度必须是结构性保证：答案里每个数字都能在证据中原样找到。"""
    answer = generator.generate("合格投资者的门槛是多少？", [make_evidence("a")])
    check = check_numbers(answer.text, [make_evidence("a").text])
    assert check.ok


def test_generator_deduplicates_citation_numbers(generator):
    """同一子块被多路召回重复送进来时，答案里的引用编号必须去重（一个子块一个编号）。"""
    # 两条证据共用 child_id "a"，模拟三路召回命中同一段落
    evidence = [make_evidence("a"), make_evidence("a")]
    answer = generator.generate("合格投资者的门槛是多少？", evidence)
    assert answer.citation_numbers == sorted(set(answer.citation_numbers))


def test_generator_uses_real_llm_when_available():
    """LLM 可用时走真实生成路径，但引用编号仍由账本签发、答案仍须可追溯。"""
    # 继承真实 LLMClient 只替换 available/complete：除网络调用外，编号与校验逻辑都走真实实现
    class FakeLLM(LLMClient):
        """测试替身：始终可用、返回固定答案的假 LLM，用于验证真实模型调用路径。"""
        def __init__(self):
            """初始化假 LLM：强制非 mock 模式，并把模型名标记为 fake。"""
            super().__init__(force_mock=False)
            self.model = "fake"

        @property
        def available(self):
            """属性：恒为 True，表示该假 LLM 始终可用。"""
            return True

        def complete(self, system, user, temperature=0.0):
            """返回固定答案文本（含引用编号 [1]），用于验证生成与引用链路。"""
            from src.answer.llm import LLMResult

            return LLMResult(text="【结论】根据资料 [1] 门槛为 300 万元。", mode="openai", model="fake")

    answer = AnswerGenerator(FakeLLM()).generate("门槛是多少？", [make_evidence("a")])
    assert answer.mode == "openai"
    assert answer.traceable


def test_generator_falls_back_when_llm_errors():
    """真实模型报错或返回空文本时不得整体失败，必须降级为抽取式作答并留下降级说明。"""
    # 构造「可用但调用失败」的模型：available 为真、complete 带 error，正好命中降级分支
    class BrokenLLM(LLMClient):
        """测试替身：自称可用但调用即返回错误，用于验证降级路径。"""
        def __init__(self):
            """初始化故障替身：强制非 mock 模式。"""
            super().__init__(force_mock=False)

        @property
        def available(self):
            """属性：恒为 True，让引擎先走到真实调用分支再失败。"""
            return True

        def complete(self, system, user, temperature=0.0):
            """返回带 error 字段的空结果，模拟网络不可达。"""
            from src.answer.llm import LLMResult

            return LLMResult(text="", mode="openai", error="ConnectionError: 网络不可达")

    answer = AnswerGenerator(BrokenLLM()).generate("门槛是多少？", [make_evidence("a")])
    assert answer.mode == "fallback-extractive"
    assert answer.traceable
    assert any("降级" in note for note in answer.notes)


def test_generator_marks_low_overlap_answers_for_review():
    """问题措辞与原文差异大时不硬编，但仍给出检索得分最高的原文并提示人工确认。

    不变式：词面零交集时「拒答与否」由检索层决定，抽取层只照抄最相关的一句、
    标注「建议人工确认」并照常给出可追溯引用，不越权做相关性判断。
    """
    unrelated = make_evidence("z", text="这是一段与问题完全没有词面交集的制度描述文字。")
    answer = AnswerGenerator(LLMClient(force_mock=True)).generate("量子计算机如何实现纠错？", [unrelated])
    assert not answer.refused
    assert answer.traceable
    assert "人工确认" in answer.text
    assert "制度描述文字" in answer.text


def test_generated_answer_to_dict(generator):
    """GeneratedAnswer 的序列化必须暴露 traceable 与引用明细，供接口/轨迹直接消费。"""
    payload = generator.generate("门槛是多少？", [make_evidence("a")]).to_dict()
    assert payload["traceable"] is True
    assert payload["citations"]


# ---------------------------------------------------------------------------
# 忠实度
# ---------------------------------------------------------------------------
def test_check_numbers_flags_unsupported_values():
    """数字只能来自证据：证据里查不到的数值必须被标为 unsupported，而不是放过。"""
    evidence = ["合格投资者的金融资产不低于 300 万元。"]
    assert check_numbers("门槛是 300 万元。", evidence).ok
    bad = check_numbers("门槛是 500 万元。", evidence)
    assert not bad.ok
    assert "500" in bad.unsupported


def test_check_numbers_ignores_years_and_small_ordinals():
    """年份与一两位短序号属于非事实数字（条款序号、时间），不参与数字忠实度判定。

    注：实际实现为——本用例文本里只有四位年份命中数字正则（由 `_YEAR` 跳过），
    一两位序号的分支由 `_IGNORABLE` 覆盖；证据文本刻意与答案无关，通过即说明确有数字被跳过。
    """
    assert check_numbers("2024 年第一条", ["完全不同的证据文本"]).ok


def test_check_numbers_rate():
    """rate 是支持数字的占比：部分数字查不到时须落在 0 与 1 的开区间内。"""
    # 两个数字一个在证据里、一个不在，正好把比例卡在中间
    check = check_numbers("金额 300 万元，比例 500%。", ["金额 300 万元"])
    assert 0.0 < check.rate < 1.0


def test_sentence_support_detects_unsupported_claims():
    """逐句校验：证据里找不到支撑的句子要被计入 unsupported 并拉低支撑率。"""
    # 第一句照抄证据、第二句是凭空的股市预测，用来区分「有支撑」与「无支撑」
    rate, unsupported = sentence_support(
        "合格投资者的金融资产不低于 300 万元。明天股市一定会上涨并创新高。",
        ["合格投资者的金融资产不低于 300 万元。"],
    )
    assert rate < 1.0
    assert unsupported


def test_sentence_support_strips_citation_metadata():
    """校验支撑度前必须先剥掉引用编号等元信息，否则 [1] 会被误判成无支撑内容。"""
    rate, _ = sentence_support(
        "[1] 合格投资者的金融资产不低于 300 万元。",
        ["合格投资者的金融资产不低于 300 万元。"],
    )
    assert rate == 1.0


def test_strip_meta_removes_brackets_and_parentheses():
    """strip_meta 只剥元信息（引用编号、括号出处、行首项目符号），事实正文必须原样留下。"""
    cleaned = strip_meta("- （原文｜示例银行 · 办法）第一条 内容。 [2]")
    assert "原文" not in cleaned
    assert "第一条 内容。" in cleaned


def test_answer_relevance_bounds():
    """相关性是 token 级 F1：有交集则为正，完全无交集必须为 0（不得出现负值或虚高）。"""
    assert answer_relevance("合格投资者金融资产门槛", "合格投资者的金融资产门槛是多少？") > 0.5
    assert answer_relevance("完全无关内容", "合格投资者门槛") == 0.0


def test_keyphrase_coverage():
    """关键点覆盖率按命中要点比例计算；没有给要点时视为满分，不因缺失标注而扣分。"""
    # 两个要点只答到一个，覆盖率落在 0 与 1 之间
    assert keyphrase_coverage("门槛是 300 万元", ["300 万元", "50 万元"]) == 0.5
    assert keyphrase_coverage("任意", []) == 1.0


def test_claim_body_drops_source_section():
    """出处清单段属于元数据而非事实陈述，必须从 claim_body 里切掉再做忠实度判定。"""
    text = "【结论】内容 [1]。\n\n【出处与时效】[1] 某资料 2024-01-01"
    body = claim_body(text)
    assert "出处与时效" not in body
    assert "内容" in body


def test_evaluate_faithfulness_full_report():
    """一次调用汇总全部确定性指标，并产出可序列化、可卡门禁的忠实度体检报告。"""
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
