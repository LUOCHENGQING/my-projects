"""引用分配与溯源校验。

「回答可溯源」在金融场景不是加分项，是准入条件：业务人员要拿这条回答去复核，
复核的第一步就是**顺着编号回到原文**。因此引用编号必须有唯一签发方，
并且答案里出现的每个编号都必须能被验证。

本模块的三条规矩：

1. **编号只由 CitationLedger 分配**，其他任何地方都不许自己拼 `[1]`。
   多写一处就多一处编号冲突，最终表现为"点了 [3] 跳到不相干的文档"。
2. **同一子块只占一个编号**。三路召回是冗余的，同一段证据被多个查询命中很常见，
   不去重编号会让参考文献里出现 5 条一模一样的出处。
3. **悬空引用当场剔除**。模型偶尔会编出 `[7]` 这种不存在的编号（幻觉引用），
   校验不通过就把它从答案里抹掉并记录，而不是留在那里等着被面试官挑出来。

ID 链路：`[n]` → Citation → 子块 child_id → 父块 parent_id → 文档 source_id → 文件路径。

在 RAG 全链路中的位置
--------------------
    切分 / 索引 → 三路召回 → 去重重排 → 【本模块：签发编号 + 校验编号】 → 答案渲染 → 缓存 / 接口

上游签发方**只有** `answer/generator.py` 的 `AnswerGenerator`：
抽取式路径按需 `allocate()`，真实模型路径先用 `allocate_all()` 把全部证据编号备好。
下游消费方：`src/engine.py` 把 `Citation` 列表放进 `AnswerResult`（随 `to_dict()` 写缓存并作为接口响应返回），
引用校验结论另会进轨迹的 `validate` 步骤；接口/前端凭这些编号回到原文做人工复核。

输入：`retrieve.pipeline.Evidence`（自带 child_id / parent_id / source_id 与出处元数据）
输出：`Citation`（编号 + 出处 + 片段）、`CitationCheck`（used / dangling / unused / cleaned_text）、
      `CitationLedger.render()` 产出的「出处与时效」多行文本。

三条口径（与代码一致，改动前请先看 `validate()` 与 `allocate()`）
--------------------------------------------------------------
    编号的形态      `[n]`，n 为 1~2 位数字（`_CITE_RE`），从 1 起顺序签发、不回收、不重排；
    同一子块        `_by_child[child_id]` 复用同一编号（三路召回反复命中同一段是常态）；
    片段长度        `allocate(..., snippet_limit=200)` 截断，超出以「…」结尾。

本模块不做任何模型调用、不做 IO，全部是确定性计算，因此可在 CI 里逐次回归。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..retrieve.pipeline import Evidence

__all__ = ["Citation", "CitationLedger", "CitationCheck"]

# 答案正文里的引用编号形态：`[n]`，n 为 1~2 位数字。
# 注：3 位及以上的 `[123]` 不会被识别——账本超过 99 条时该编号既不会被判悬空、也不会被计入 used。
_CITE_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass
class Citation:
    """一条引用记录：回答里的 `[n]` 能一路回溯到原文与版本。

    字段含义（构造时从 `Evidence` 按值拷贝，之后不再回查 Evidence，故本对象可独立落盘）：

        citation_no      引用编号 `[n]`，由 `CitationLedger.allocate()` 从 1 起顺序签发
        evidence_id      证据编号（E1、E2…），与提示词里的 `[E1]` 对应
        child_id         子块 ID —— 去重与「同一子块只占一个编号」的判据
        parent_id        父块 ID，用于回溯完整章节上下文
        source_id        来源资料编号，业务复核时凭它定位原文件
        title / section_title / institution   标题、章节、机构
        effective_date / version / doc_type   时效与版本信息
        snippet          证据正文截断（`allocate` 按 `snippet_limit` 截断）

    关键派生属性：`label`（人可读出处）、`updated_at`（无日期时给「未标注」）。
    """

    citation_no: int
    evidence_id: str
    child_id: str
    parent_id: str
    source_id: str
    title: str
    section_title: str
    institution: str
    effective_date: str
    version: str
    doc_type: str
    snippet: str

    @property
    def label(self) -> str:
        """人可读出处（机构 · 标题 · 章节），空字段自动省略，用于答案里的出处清单。

        参数：无。
        返回：str，形如「示例监管机构 · 资产管理产品管理办法 · 四、适当性匹配规则」；
              机构或标题为空时只拼接非空部分，首尾的「 · 」与空格会被 strip 掉。
        副作用/异常：无。
        """
        head = " · ".join(p for p in (self.institution, self.title) if p)
        return f"{head} · {self.section_title}".strip(" ·")

    @property
    def updated_at(self) -> str:
        """出处的时效信息：业务方据此判断依据是不是过期了。

        参数：无。
        返回：str —— `effective_date` 原值；为空时返回「未标注」（刻意不用空串，避免清单出现空白项）。
        副作用/异常：无。
        """
        return self.effective_date or "未标注"

    def to_dict(self) -> Dict[str, object]:
        """导出为纯 dict（12 个 Citation 字段 + 派生字段 label），供轨迹落盘与接口响应。

        参数：无。
        返回：Dict[str, object]；**不含 updated_at**（它派生自 effective_date，需要时由调用方自行取值）。
        副作用/异常：无。
        """
        return {
            "citation_no": self.citation_no,
            "evidence_id": self.evidence_id,
            "child_id": self.child_id,
            "parent_id": self.parent_id,
            "source_id": self.source_id,
            "title": self.title,
            "section_title": self.section_title,
            "institution": self.institution,
            "doc_type": self.doc_type,
            "effective_date": self.effective_date,
            "version": self.version,
            "snippet": self.snippet,
            "label": self.label,
        }


@dataclass
class CitationCheck:
    """一次引用校验的结果（由 `CitationLedger.validate()` 产出）。

    字段（全部是确定性判定的产物，没有模型参与）：

        ok           校验是否通过：**至少剩一个有效编号（或原文本来就为空）且没有悬空引用**
        used         清理后答案里实际出现、且都在账本内的编号（升序去重）
        dangling     答案里出现但账本里没有的编号 → 幻觉引用，已在 `cleaned_text` 中被剔除
        unused       账本里分配了、但答案里没引用的编号（合法但没用上，不算错误）
        cleaned_text 剔除悬空引用并压缩多余空格后的答案文本，**上层必须用它替换原答案**

    注意 ok 与 used 的关系：一处编号都不标的非空答案 ok=False（无法溯源即不算通过）。
    """

    ok: bool = True
    used: List[int] = field(default_factory=list)
    dangling: List[int] = field(default_factory=list)   # 答案里出现但无对应证据 → 幻觉引用
    unused: List[int] = field(default_factory=list)     # 分配了但答案里没用到
    cleaned_text: str = ""

    def to_dict(self) -> Dict[str, object]:
        """导出为 dict（4 个字段），供轨迹与接口响应展示「引用是否可溯源」。

        参数：无。
        返回：Dict[str, object] —— ok / used / dangling / unused；**刻意不含 cleaned_text**
              （那是答案正文，由调用方单独使用，重复落盘会让轨迹体积翻倍）。
        副作用/异常：无。
        """
        return {
            "ok": self.ok,
            "used": list(self.used),
            "dangling": list(self.dangling),
            "unused": list(self.unused),
        }


class CitationLedger:
    """一次问答的引用账本：编号的**唯一签发方**，同时也是校验方。

    关键属性（全部私有，对外只暴露副本或只读视图）：

        _citations   已签发的 Citation 列表，其下标 +1 就等于 `citation_no`（只追加，不重排、不回收）
        _by_child    child_id -> Citation 映射，保证「同一子块只占一个编号」
        _dropped     被剔除的幻觉引用记录（`citation_no` + `reason`），会随 `to_dict()` 进轨迹

    生命周期：每次 `AnswerGenerator.generate()` 新建一个，**不跨问答复用**——
    跨问答复用会让编号串到别的答案里，这是最难查的一类溯源 bug。
    """

    def __init__(self) -> None:
        """初始化空账本。

        参数：无。
        返回：无（构造函数）。
        副作用/异常：无。
        """
        self._citations: List[Citation] = []
        self._by_child: Dict[str, Citation] = {}
        self._dropped: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # 分配
    # ------------------------------------------------------------------
    def allocate(self, evidence: Evidence, snippet_limit: int = 200) -> Citation:
        """为一条证据分配引用编号；同一子块复用已有编号。

        参数：
            evidence      一条 `retrieve.pipeline.Evidence`（三路召回反复命中同一子块是常态）
            snippet_limit 片段最大字符数（默认 200），超出部分以「…」结尾——
                          出处清单不该带整章正文，那是父块上下文的职责
        返回：`Citation`；命中复用分支返回**已存在的那条**，编号保持不变。
        副作用：首次分配时追加进 `_citations`、登记 `_by_child`，编号 = 当前条数 + 1。
        异常：无（不校验 evidence 字段是否为空）。
        """
        existing = self._by_child.get(evidence.child_id)
        if existing is not None:
            return existing

        snippet = evidence.text.strip()
        if len(snippet) > snippet_limit:
            snippet = snippet[:snippet_limit] + "…"
        citation = Citation(
            citation_no=len(self._citations) + 1,
            evidence_id=evidence.evidence_id,
            child_id=evidence.child_id,
            parent_id=evidence.parent_id,
            source_id=evidence.source_id,
            title=evidence.title,
            section_title=evidence.section_title,
            institution=evidence.institution,
            effective_date=evidence.effective_date,
            version=evidence.version,
            doc_type=evidence.doc_type,
            snippet=snippet,
        )
        self._citations.append(citation)
        self._by_child[evidence.child_id] = citation
        return citation

    def allocate_all(self, evidence_list: Sequence[Evidence]) -> List[Citation]:
        """批量分配编号（按入参顺序），供真实模型路径「先把全部证据编号备好」使用。

        参数：evidence_list 证据序列（可为空）。
        返回：List[Citation]，与入参等长且顺序一一对应；重复子块会返回同一条对象。
        副作用/异常：同 `allocate()`；不抛异常。
        """
        return [self.allocate(e) for e in evidence_list]

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """账本里已签发的引用条数。参数：无；返回：int。副作用/异常：无。"""
        return len(self._citations)

    @property
    def citations(self) -> List[Citation]:
        """已签发引用的**副本**列表。参数：无；返回：List[Citation]（外部改动不会污染账本）。"""
        return list(self._citations)

    @property
    def known_numbers(self) -> set:
        """账本内全部合法编号集合——`validate()` 判定「悬空引用」的唯一依据。

        参数：无；返回：set[int]；副作用/异常：无。
        """
        return {c.citation_no for c in self._citations}

    def get(self, number: int) -> Optional[Citation]:
        """按编号取回引用。

        参数：number 引用编号 `[n]` 里的 n。
        返回：匹配的 `Citation`；编号不存在时返回 None（不抛异常——模型可能引用了不存在的编号）。
        副作用/异常：无。
        """
        for c in self._citations:
            if c.citation_no == number:
                return c
        return None

    def by_source(self, source_id: str) -> List[Citation]:
        """取某个来源文档下的全部引用（同一份资料可能被多个子块引用）。

        参数：source_id 资料编号。
        返回：List[Citation]（可能为空；保持账本内的原始顺序）。
        副作用/异常：无。
        """
        return [c for c in self._citations if c.source_id == source_id]

    @property
    def dropped(self) -> List[Dict[str, str]]:
        """被剔除的幻觉引用记录，写进轨迹供人工抽查。

        参数：无。
        返回：List[Dict[str, str]] 副本，每项为 `{"citation_no": "7", "reason": "答案引用了不存在的出处编号"}`。
        副作用/异常：无。
        """
        return list(self._dropped)

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def validate(self, text: str) -> CitationCheck:
        """校验答案里的引用编号：悬空的剔除、未使用的标注出来。

        判定口径（全部确定性计算，不调模型、不依赖提示词）：

            used      用 `_CITE_RE` 从原文抓出的编号，去重升序
            dangling  used 里不在 `known_numbers` 中的编号（模型自己编的）
            unused    `known_numbers` 中清理后仍未出现的编号（合法但没被引用，不算错）
            ok        「清理后仍有编号 或 原文本来就为空」且**没有悬空引用**

        剔除悬空引用是**结构性的反幻觉措施**：不依赖提示词叮嘱模型"别编编号"，
        而是在输出前做一次确定性校验，编出来的编号过不了这一关。

        参数：text 待校验的答案正文（`None` 会被当成空串处理）。
        返回：`CitationCheck`；其中 cleaned_text 才是可对外展示的文本，调用方必须用它替换原答案。
        副作用：把每个被剔除的编号追加进 `_dropped`（编号 + 原因），并压缩连续空格后 strip；
                本方法**不修改** `self._citations`，账本仍是只增不改。
        异常：无。
        """
        used = sorted({int(m.group(1)) for m in _CITE_RE.finditer(text or "")})
        known = self.known_numbers
        dangling = [n for n in used if n not in known]

        cleaned = text or ""
        for n in dangling:
            cleaned = re.sub(rf"\[{n}\]", "", cleaned)
            self._dropped.append({"citation_no": str(n), "reason": "答案引用了不存在的出处编号"})
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

        remaining = sorted({int(m.group(1)) for m in _CITE_RE.finditer(cleaned)})
        unused = [n for n in sorted(known) if n not in remaining]
        return CitationCheck(
            ok=(bool(remaining) or not text.strip()) and not dangling,
            used=remaining,
            dangling=dangling,
            unused=unused,
            cleaned_text=cleaned,
        )

    def render(self, numbers: Optional[Iterable[int]] = None) -> str:
        """渲染「出处与时效」清单。每条都带更新时间与版本，方便业务复核。

        参数：numbers 只渲染这些编号（None 表示账本内全部编号）；传入空集合会得到空串。
        返回：str，多行，每行形如
              `[1] 示例监管机构 · 资产管理产品管理办法 · 四、适当性匹配规则（资料编号 SRC-001，更新/生效日期 2024-01-01，版本 V2）`；
              相应字段为空时，机构/标题被省略，「，版本 …」整段被省掉。
        副作用/异常：无（只读账本）。
        """
        wanted = set(numbers) if numbers is not None else self.known_numbers
        rows: List[str] = []
        for c in self._citations:
            if c.citation_no not in wanted:
                continue
            ver = f"，版本 {c.version}" if c.version else ""
            rows.append(
                f"[{c.citation_no}] {c.label}（资料编号 {c.source_id}，更新/生效日期 {c.updated_at}{ver}）"
            )
        return "\n".join(rows)

    def to_dict(self) -> Dict[str, object]:
        """导出账本快照：count / citations / dropped。

        参数：无。
        返回：Dict[str, object] —— `count` 为已签发条数，`citations` 为逐条 `Citation.to_dict()`，
              `dropped` 为被剔除的幻觉引用记录（有了它才能证明"确实拦下了编号编造"）。
        副作用/异常：无。
        """
        return {
            "count": len(self._citations),
            "citations": [c.to_dict() for c in self._citations],
            "dropped": self.dropped,
        }
