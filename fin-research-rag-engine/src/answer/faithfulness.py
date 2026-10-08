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

在 RAG 全链路中的位置
--------------------
    切分 / 索引 → 三路召回 → 去重重排 → 生成 → 【本模块：数字 / 句子 / 相关性体检】 → 缓存 / 接口

被谁调用：
    `src/engine.py` 第 5 步生成之后调用 `evaluate_faithfulness()`，把结论写进 `AnswerResult.faithfulness`
    与轨迹（`validate` 那一步的 status 取 `faithful`）；
    `eval/run_eval.py` 用 `claim_body()` + `evaluate_faithfulness()` 跑离线评测并汇总指标；
    `tests/test_answer.py` 直接断言这些口径。

输入：答案正文（由上层传 `GeneratedAnswer.text`）、原问题、证据原文列表（`Evidence.text`）、
      可选的评测要点 keyphrases。
输出：`FaithfulnessReport`（numbers / support_rate / unsupported_sentences / relevance / coverage）。

阈值与判定口径（**改阈值前请先看这里，代码里就是这些数**）
------------------------------------------------------
    sentence_support(threshold=0.34)  单句 token 与**任一条**证据的覆盖率
                                      `len(句子token ∩ 证据token) / len(句子token) ≥ 0.34` 即视为被支撑；
                                      只统计长度 ≥ 8 的句子；一句话都统计不到时返回 1.0
    FaithfulnessReport.faithful       `numbers.ok and support_rate >= 0.8`（数字全对 且 支撑率 ≥ 0.8）
    check_numbers                     `ok` 等价于「unsupported 为空」，即**一个编造的数字都不容忍**
    answer_relevance                  token 集合的 F1；任一侧为空或交集为 0 时返回 0.0
    keyphrase_coverage                keyphrase 以子串形式出现在答案里的比例；keyphrases 为空时返回 1.0
    claim_body                        先切掉「【出处与时效】/【引用明细】/【出处】」之后的内容再校验

为什么必须先 `claim_body`：出处清单里的资料编号、日期、版本号不是对事实的断言，
把它们算进忠实度会制造大量假阳性（把正确的答案判成"未被支撑"）。
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
# 注：实际实现里 `SOURCE_SECTION_MARKERS` 定义在 `claim_body()` **之前**，
# 且三处 import 的 `jaccard` / `Iterable` / `Optional` 在本模块内未被使用（保留为公共依赖口径）。


def claim_body(answer: str) -> str:
    """截取答案中**属于事实陈述**的部分（去掉出处清单段）。

    参数：answer 完整答案正文（允许为 None / 空串）。
    返回：str —— 三个 `SOURCE_SECTION_MARKERS` 中最早出现的位置之前的文本并 strip；
          一个标记都没出现时返回**整段**答案（都当事实陈述处理）。
    副作用/异常：无；不修改入参。
    """
    text = answer or ""
    cut = len(text)
    for marker in SOURCE_SECTION_MARKERS:
        pos = text.find(marker)
        if pos >= 0:
            cut = min(cut, pos)
    return text[:cut].strip()

# 答案里允许出现的"非事实数字"：编号、条款序号、年份等
# 注：实际实现里 `_IGNORABLE` 只放行 1~2 位纯数字（引用编号 / 条款序号 / 等级里的小数字），
# 4 位年份由紧邻的 `_YEAR`（19xx / 20xx）单独放行。
_IGNORABLE = re.compile(r"^\d{1,2}$")
_YEAR = re.compile(r"^(19|20)\d{2}$")


def check_numbers(answer: str, evidence_texts: Sequence[str]) -> "NumberCheck":
    """检查答案里的数字是否都能在证据中找到。

    这是本项目最硬的一条反幻觉规则：**数字只能来自证据**。
    模型自己算出的比例、自己补的金额，会在这一步被直接标出来。

    判定口径（逐个数字走三步，命中即算 supported）：
        1. 放行：1~2 位纯数字（`_IGNORABLE`）与 19xx / 20xx 年份（`_YEAR`）跳过不查；
        2. 精确：数字 token 出现在证据的数字 token 集合里；
        3. 容差：把证据整体去掉千分位后做子串比对，容忍 `1,286,400` 与 `1286400`、
           `128.6` 与 `128.60` 这类写法差异（token 含小数点时另取去掉尾零的变体）。
        三步都不中 → 进 `unsupported`。

    参数：answer 待校验文本（上层传的是 `claim_body()` 的结果，不做出处清单的校验）；
          evidence_texts 证据原文序列（`Evidence.text`）。
    返回：`NumberCheck`（supported / unsupported 两个列表，保持出现顺序且已去重）。
    副作用/异常：无；`amount_tokens` 内部只读，不会修改入参。
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
    """数字校验结果。

    字段：supported 能在证据里定位到的数字（含容差命中）；unsupported 定位不到的（**疑似编造**）。
    关键派生属性：`ok`（等价于 unsupported 为空）、`rate`（supported / 总数，无数字时按 1.0）。
    """

    supported: List[str] = field(default_factory=list)
    unsupported: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """数字是否全部有据可查。参数：无；返回：bool（`unsupported` 为空即 True）。副作用/异常：无。"""
        return not self.unsupported

    @property
    def rate(self) -> float:
        """数字忠实度。参数：无。

        返回：float —— `len(supported) / (len(supported) + len(unsupported))`；
              答案为**一个数字都没有**时返回 1.0（没有可编造的数字，视为通过）。
        副作用/异常：无。
        """
        total = len(self.supported) + len(self.unsupported)
        return len(self.supported) / total if total else 1.0

    def to_dict(self) -> Dict[str, object]:
        """导出为 dict（ok / rate / supported / unsupported），供轨迹与评测汇总。

        参数：无。
        返回：Dict[str, object]，其中 rate 保留 4 位小数，两个明细列表**原样引用**（未做拷贝）。
        副作用/异常：无。
        """
        return {
            "ok": self.ok,
            "rate": round(self.rate, 4),
            "supported": self.supported,
            "unsupported": self.unsupported,
        }


# 答案句子里属于「元信息」的片段：引用编号与出处前缀。
# 校验一句话是否被证据支撑时，必须先把这些剥掉——它们本来就不在证据原文里。
# 注：两个括号正则都限长（中/英文括号 0~80 字符、方括号 0~20 字符），
# 这样「（2024 年 1 月生效）」这类短注会被剥掉，而长句里的括号不会被整段吞掉。
_META_PATTERNS = (
    re.compile(r"\[\d{1,2}\]"),
    re.compile(r"（[^（）]{0,80}）"),
    re.compile(r"\([^()]{0,80}\)"),
    re.compile(r"【[^【】]{0,20}】"),
    re.compile(r"^[-•\s]+"),
)


def strip_meta(sentence: str) -> str:
    """剥掉句子里的引用编号与出处前缀，只留下事实陈述。

    参数：sentence 一句答案文本。
    返回：str —— 按 `_META_PATTERNS` 依次把命中片段替换成空格（引用编号 `[n]`、中英文括号注、
          `【…】`、行首的 `-` / `•` 列表符号），最后把连续空格压成一个并 strip。
    副作用/异常：无；纯字符串处理，不修改入参、不抛异常。
    """
    out = sentence
    for pattern in _META_PATTERNS:
        out = pattern.sub(" ", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def sentence_support(answer: str, evidence_texts: Sequence[str], threshold: float = 0.34) -> Tuple[float, List[str]]:
    """计算答案中有多少句子能被证据支撑，返回 (支撑率, 未被支撑的句子)。

    判定口径（**阈值就是入参默认值 0.34，不是省略号里的"差不多"**）：
        1. 先 `split_sentences()` 切句，再丢掉长度 < 8 的短句（「【结论】」这类小标题不算句子）；
        2. 每句先 `strip_meta()` 去掉编号与括号注，再 `tokenize()`；
        3. 该句 token 与**任一条**证据的覆盖率 `len(交集) / len(句子token)` ≥ threshold 即算被支撑
           （命中一条就提前结束，取的是各证据里的最大覆盖率）；
        4. 覆盖率 < threshold 的句子进 unsupported。
    注：无有效句子时返回 1.0（没有可校验的断言，不制造假阴性）。

    参数：answer 答案正文（通常是 `claim_body()` 的结果）；
          evidence_texts 证据原文序列；threshold 单句覆盖率阈值，默认 0.34。
    返回：Tuple[float, List[str]] —— 支撑率 = 被支撑句数 / 参与统计的句数，以及未被支撑的原句列表。
    副作用/异常：无；证据 token 在函数内预计算一次，只读不写。
    """
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
    """答案与问题的 token 级 F1（不依赖任何模型，可重复、可回归）。

    参数：answer 答案正文；question 原问题。
    返回：float —— 以 token **集合**算 P/R 后的 F1；问题或答案为空、或交集为 0 时返回 0.0
          （不返回 1.0：答非所问不会被算成"相关"）。
    副作用/异常：无。
    """
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
    """评测集给的要点覆盖情况（例如必须提到"合格投资者""50 万元"）。

    参数：answer 答案正文；keyphrases 评测集提供的必答要点列表。
    返回：float —— 命中数 / 要点总数；keyphrases 为空时返回 1.0（不设要点即无要求）。
    副作用/异常：无；判据是**原文子串包含**，不做分词或同义改写匹配（口径偏严，宁缺毋滥）。
    """
    if not keyphrases:
        return 1.0
    hits = sum(1 for phrase in keyphrases if phrase and phrase in answer)
    return hits / len(keyphrases)


@dataclass
class FaithfulnessReport:
    """一次答案的忠实度体检报告。

    字段：
        numbers                数字校验结果（`NumberCheck`）
        support_rate           句子支撑率（`sentence_support` 的返回值）
        unsupported_sentences  未被证据支撑的原句
        relevance              与问题的 token 级 F1（`answer_relevance`）
        coverage               评测要点覆盖率（`keyphrase_coverage`）

    关键派生属性：`faithful` = `numbers.ok and support_rate >= 0.8`——
    **数字错一个就整体不通过**，句子支撑率则允许 20% 的余量（模型改写措辞是正常的）。
    """

    numbers: NumberCheck = field(default_factory=NumberCheck)
    support_rate: float = 1.0
    unsupported_sentences: List[str] = field(default_factory=list)
    relevance: float = 0.0
    coverage: float = 1.0

    @property
    def faithful(self) -> bool:
        """是否达到门禁。参数：无。

        返回：bool —— 数字全部有据可查（`numbers.ok`）且句子支撑率 ≥ 0.8。
        副作用/异常：无。
        """
        return self.numbers.ok and self.support_rate >= 0.8

    def to_dict(self) -> Dict[str, object]:
        """导出为 dict，供轨迹落盘、评测汇总与接口展示。

        参数：无。
        返回：Dict[str, object] —— faithful / numbers / support_rate / unsupported_sentences /
              relevance / coverage；三个比例都保留 4 位小数，
              **unsupported_sentences 只取前 5 条**（原句可能很长，全量落盘会撑爆轨迹）。
        副作用/异常：无。
        """
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

    注意 `body = claim_body(answer) or answer` 这一行：当答案**以出处标记开头**
    （`claim_body` 返回空串）时会回退成整段答案——否则会拿一条空文本去算指标，
    得到"支撑率 1.0、忠实"的假绿灯。

    参数：answer 生成层产出的答案正文；question 原问题；
          evidence_texts 证据原文序列（`Evidence.text`）；keyphrases 评测要点，默认空（覆盖率为 1.0）。
    返回：`FaithfulnessReport`（数字校验、句子支撑率、未支撑句、相关性、要点覆盖）。
    副作用/异常：无；不写盘、不改入参、不抛业务异常（除 regex / 类型错误外无异常路径）。
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
