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

在 RAG 全链路中的位置
--------------------
    切分 / 索引 → 三路召回 → 去重重排 → 【本模块：作答 + 引用校验】 → 缓存 / 接口 / 轨迹

输入：`question` + `Sequence[Evidence]`（子块精确命中 + 父块上下文）+ `plan_note`（检索说明，可空）。
输出：`GeneratedAnswer`（正文 / 引用列表 / `CitationCheck` / refused / notes / latency_ms / llm 元信息）。
被谁调用：`src.engine.RAGEngine.ask()` 第 5 步；`tests/test_answer.py`、`tests/conftest.py` 直接构造使用。

对外关键对象
-----------
    `AnswerGenerator`   本模块唯一入口类，`generate()` 是主方法
    `GeneratedAnswer`   结果载体，`traceable` 属性即「引用是否全部可溯源」
    `best_sentence()`   抽取式作答的最小单元：从一段文本里挑出与问题最相关的一句
    `REFUSAL_TEXT`      证据不足时的标准拒答文案（**不要在别处另写一份**，否则前端要靠关键词猜拒答）

四条出口路径（由可用性与证据量决定，见 `generate()`）
--------------------------------------------------
    refused              证据条数 < `min_evidence` → 直接拒答，不检索不调模型
    mock-extractive      无 API Key / 强制 mock → 抽取式作答（只搬运原文，结构性无数字幻觉）
    openai               真实模型正常返回
    fallback-extractive  真实模型报错或返回空文本 → 降级为抽取式，并在 notes 里写明原因

引用编号的唯一签发方仍是 `citations.CitationLedger`：抽取式路径按需分配，真实路径先备齐全部编号。
输出前的 `ledger.validate(text)` 会把账本外的编号（模型编的 `[7]`）**当场剔除**并记入 notes。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..retrieve.pipeline import Evidence
from ..utils.text import split_sentences, tokenize
from .citations import Citation, CitationCheck, CitationLedger
from .llm import SYSTEM_PROMPT, LLMClient, build_user_prompt
# 注：`__all__` 紧贴 import 之后（无空行）是既有排版，这里保持原样不动。
__all__ = ["GeneratedAnswer", "AnswerGenerator", "best_sentence", "REFUSAL_TEXT"]

# 拒答文案：证据不足时的唯一出口。措辞刻意"给下一步"（补充哪类资料），
# 而不是只说"不知道"——金融场景里用户需要知道去哪里补材料。
REFUSAL_TEXT = "现有资料不足以回答该问题。请补充相关制度文件、产品说明书或风险案例后再试。"


def best_sentence(text: str, query_tokens: Sequence[str], fallback_limit: int = 160) -> str:
    """从一段文本里挑出与问题最相关的一句（抽取式作答的基本单元）。

    打分只看「查询词在句子里出现了多少」，不做任何生成，因此
    **选出来的句子一定是原文**，不会引入幻觉。

    入参的 query_tokens 允许是「词」也允许是「已经切好的 token」：内部统一再切一次，
    否则传入「合格投资者」这种整词时会与句子切出来的二元组对不上，永远判为不相关。

    打分口径：查询词与句子的 token 交集个数；下标 0 的句子额外加 0.05（同类句子里靠前的通常是定义/结论）。

    参数：text 候选文本（可多句，可为空串）；query_tokens 查询词或已切好的 token 序列（可为空）；
          fallback_limit 无句子可切时的截断长度，默认 160 字符。
    返回：str —— 得分最高的**原句**（不做生成，所以不会引入幻觉）；
          切不出句子时返回截断后的原文（超长加「…」）；query_tokens 为空时返回第一句。
    副作用/异常：无；不修改入参（query_tokens 为 None 时按空序列处理）。
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
    """一次生成的完整结果：答案正文 + 引用账本 + 校验结论。

    字段：
        question     原始问题
        text         答案正文（**已被 `CitationCheck.cleaned_text` 清洗**，悬空引用不在其中）
        mode         出口路径：mock-extractive / fallback-extractive / openai；拒答分支取 `LLMClient.mode`
        citations    只含「答案里真正用到」的引用（按 `check.used` 过滤后的账本子集）
        check        引用校验结论（None = 从未校验，此时 `traceable` 为 False）
        evidence     本次作答使用的证据（拒答分支为空列表）
        refused      是否走的拒答策略
        notes        人可读备注（降级原因、剔除的悬空引用条数等），随轨迹落盘
        latency_ms   生成耗时（毫秒）
        llm          LLM 元信息：`{"mode": ..., "model": ...}`
    """

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
        """答案用到的引用编号。参数：无；返回：List[int]（保持 citations 顺序，不重排）。副作用/异常：无。"""
        return [c.citation_no for c in self.citations]

    @property
    def source_ids(self) -> List[str]:
        """答案涉及的资料编号（去重保序）。参数：无；返回：List[str]。副作用/异常：无。"""
        seen: List[str] = []
        for c in self.citations:
            if c.source_id not in seen:
                seen.append(c.source_id)
        return seen

    @property
    def traceable(self) -> bool:
        """引用是否全部可追溯：每个 [n] 都能回到真实证据，且至少用到一条。

        参数：无。
        返回：bool —— `check` 为 None（没校验过）时 False；有校验时要求
              `not check.dangling`（没有编造的编号）且 `check.used` 非空（至少引用了一条）。
        副作用/异常：无。
        """
        if self.check is None:
            return False
        return not self.check.dangling and bool(self.check.used)

    def to_dict(self) -> Dict[str, object]:
        """导出为 dict（供缓存回填、接口响应与轨迹落盘）。

        参数：无。
        返回：Dict[str, object] —— `answer` 键对应 `self.text`，另含 traceable / citation_numbers /
              source_ids / citations / check / notes / latency_ms（保留 3 位小数）/ llm。
        副作用/异常：无。
        """
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
    """按可用性选择作答路径，并对结果做引用与忠实度校验。

    关键属性（构造时定，运行期不改）：
        llm            `LLMClient`；`llm.available` 决定走真实模型还是抽取式
        max_bullets    抽取式答案【依据】段最多列几条（默认 3）
        min_evidence   证据条数下限，低于它直接拒答（默认 1）

    职责边界：只负责「怎么把给定证据组织成带出处的答案」+「校验编号可溯源」。
    检索质量、主体闸门（在 `src.engine`）、忠实度打分（在 `faithfulness`）都不在这里做。
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        max_bullets: int = 3,
        min_evidence: int = 1,
    ) -> None:
        """装配生成器。

        参数：llm 可选客户端（None 时新建 `LLMClient()`，它自己会判断有没有 Key）；
              max_bullets 【依据】段条数上限，默认 3；min_evidence 证据条数下限，默认 1。
        返回：无（构造函数）。
        副作用/异常：无网络请求；入参会被 `int()` 归一（传字符串数字也能用）。
        """
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
        """作答主入口：选路径 → 生成 → 校验引用 → 组装结果。

        四条出口路径（判定顺序就是代码顺序）：
            1. `len(evidence) < min_evidence` → 拒答（`REFUSAL_TEXT`，refused=True，citations=[]）；
            2. `llm.available` → 先 `allocate_all()` 备齐编号，再调真实模型；
               若返回 error 或文本为空 → 降级 `_extractive()`，mode = "fallback-extractive"；
            3. `llm.available` 且正常返回 → mode = "openai"；
            4. `llm.available` 为假 → `_extractive()`，mode = "mock-extractive"。
        之后无条件执行结构性校验：`ledger.validate(text)` 会把账本外的编号从正文里**当场剔除**，
        被剔除的条数写进 notes（这就是"反幻觉不靠提示词"的落点）。

        参数：question 用户原问题；evidence 检索层给出的证据序列（按相关性降序）；
              plan_note 检索说明（问题类型 / 候选数 / 过滤条件），会进用户提示词。
        返回：`GeneratedAnswer`；其 `text` 已是清洗后的文本，`check` 为校验结论。
        副作用：走真实模型时发起一次网络请求（异常已在 `LLMClient.complete` 内吞掉）；
                其余为纯内存操作（耗时用 `time.perf_counter()` 统计）。
        异常：无预期异常；下游 `LLMClient` 的任何失败都表现为降级而不是抛出。
        """
        started = time.perf_counter()
        items = list(evidence)
        # 每次问答都新建账本：编号不跨问答复用，避免 [3] 指向上一次提问的资料。
        ledger = CitationLedger()

        if len(items) < self.min_evidence:
            # 拒答分支用空串做校验：得到 ok=True、used=[]、dangling=[]，
            # 也就是「拒答不是引用失败」，前端不必把拒答显示成错误。
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
        # 无论走哪条路径都过这一关：模型编造的编号在这里被抹掉，抽取式路径天然不会触发。
        check = ledger.validate(text)
        text = check.cleaned_text
        if check.dangling:
            notes.append(f"已剔除 {len(check.dangling)} 处悬空引用（模型编造了不存在的出处编号）")
        if mode != "mock-extractive":
            # 真实模型路径：补上出处清单，保证格式统一
            # 判据三重：清单非空、正文里本来就有「出处与时效」标题、且这段清单还没被追加过——
            # 避免模型自己写了清单时出现两份，或把同一段清单重复粘贴。
            rendered = ledger.render(check.used)
            if rendered and "出处与时效" in text and rendered not in text:
                text = text.rstrip() + "\n\n" + rendered

        # 只把「答案里真正引用到」的引用放进结果：没被用上的编号不上报，避免清单虚胖。
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

        参数：question 原问题（切成 token 后用于给句子打分）；
              evidence 证据序列（调用方已保证非空，因为空证据在 `generate()` 里就被拒答拦下）；
              ledger 本次问答的引用账本，函数内会**写入**它（按需 allocate）。
        返回：str —— 拼好的三段式答案文本：【结论】【依据】【出处与时效】。
        副作用：向 `ledger` 分配引用编号（因此调用方随后必须 `validate()`）；
                不做任何计算或改写，正文里的每句话都是证据原文。
        异常：无。
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
        # 上限判定放在追加之前，且 emitted 里已经含结论句，
        # 因此【依据】最多正好追加 max_bullets 条（默认 3），不会多出一条"凑数"的来源。
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
        """按 child_id 取回引用；不存在则新分配（保证 `[n]` 与账本一一对应）。

        参数：ledger 本次问答的引用账本；item 待引用的证据。
        返回：`Citation` —— 已有则返回原对象（编号不变），没有才新签发。
        副作用：命中新分配分支时**修改 ledger**（追加编号）。
        异常：无。
        """
        for citation in ledger.citations:
            if citation.child_id == item.child_id:
                return citation
        return ledger.allocate(item)
