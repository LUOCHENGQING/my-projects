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
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..retrieve.pipeline import Evidence

__all__ = ["Citation", "CitationLedger", "CitationCheck"]

_CITE_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass
class Citation:
    """一条引用记录：回答里的 `[n]` 能一路回溯到原文与版本。"""

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
        head = " · ".join(p for p in (self.institution, self.title) if p)
        return f"{head} · {self.section_title}".strip(" ·")

    @property
    def updated_at(self) -> str:
        return self.effective_date or "未标注"

    def to_dict(self) -> Dict[str, object]:
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
    """一次引用校验的结果。"""

    ok: bool = True
    used: List[int] = field(default_factory=list)
    dangling: List[int] = field(default_factory=list)   # 答案里出现但无对应证据 → 幻觉引用
    unused: List[int] = field(default_factory=list)     # 分配了但答案里没用到
    cleaned_text: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "used": list(self.used),
            "dangling": list(self.dangling),
            "unused": list(self.unused),
        }


class CitationLedger:
    """一次问答的引用账本。"""

    def __init__(self) -> None:
        self._citations: List[Citation] = []
        self._by_child: Dict[str, Citation] = {}
        self._dropped: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # 分配
    # ------------------------------------------------------------------
    def allocate(self, evidence: Evidence, snippet_limit: int = 200) -> Citation:
        """为一条证据分配引用编号；同一子块复用已有编号。"""
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
        return [self.allocate(e) for e in evidence_list]

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._citations)

    @property
    def citations(self) -> List[Citation]:
        return list(self._citations)

    @property
    def known_numbers(self) -> set:
        return {c.citation_no for c in self._citations}

    def get(self, number: int) -> Optional[Citation]:
        for c in self._citations:
            if c.citation_no == number:
                return c
        return None

    def by_source(self, source_id: str) -> List[Citation]:
        return [c for c in self._citations if c.source_id == source_id]

    @property
    def dropped(self) -> List[Dict[str, str]]:
        """被剔除的幻觉引用记录，写进轨迹供人工抽查。"""
        return list(self._dropped)

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def validate(self, text: str) -> CitationCheck:
        """校验答案里的引用编号：悬空的剔除、未使用的标注出来。

        剔除悬空引用是**结构性的反幻觉措施**：不依赖提示词叮嘱模型"别编编号"，
        而是在输出前做一次确定性校验，编出来的编号过不了这一关。
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
        """渲染「出处与时效」清单。每条都带更新时间与版本，方便业务复核。"""
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
        return {
            "count": len(self._citations),
            "citations": [c.to_dict() for c in self._citations],
            "dropped": self.dropped,
        }
