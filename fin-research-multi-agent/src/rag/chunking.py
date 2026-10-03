"""父子块切分（Parent-Child Chunking）。

为什么要父子块
--------------
* **子块入索引**：块越小，向量方向越集中、BM25 的词频统计越干净，召回更准。
  但块太小会丢上下文，模型拿到「净利润 96,480 万元」却不知道是哪一年、哪个口径。
* **父块存上下文**：命中子块后，把它的父块（= 一个完整章节，如「六、主要风险因素」）
  一起塞给 LLM。既保住了检索精度，又保住了生成时的上下文完整性。

实现：
    1. 以 Markdown 章节作为父块（天然语义边界，且便于引用时标注"出处章节"）
    2. 父块内部按句子累积成 ~CHILD_CHUNK_CHARS 字符的子块，相邻子块保留
       CHILD_CHUNK_OVERLAP 字符重叠，避免句子被硬切断导致语义断裂
    3. 子块记录 parent_id / 章节标题 / 文档元数据，供重排时做元数据过滤

引用可追溯性正建立在这套 ID 体系上：Writer 引用的编号 -> 子块 ID -> 父块 -> 文档 source_id。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..config import CHILD_CHUNK_CHARS, CHILD_CHUNK_OVERLAP, PARENT_MAX_CHARS
from ..utils.text import split_sentences
from .corpus import Document, Section

__all__ = ["ParentChunk", "ChildChunk", "split_document", "build_chunks"]


@dataclass
class ParentChunk:
    """父块：进入上下文窗口的完整章节。"""

    parent_id: str
    doc_id: str
    source_id: str
    order: int
    section_title: str
    text: str
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def citation_label(self) -> str:
        """人可读的出处标签，例如「示例科技股份有限公司 2024年年度 · 六、主要风险因素」。"""
        company = self.meta.get("company", "")
        period = self.meta.get("period", "")
        return f"{company} {period} · {self.section_title}".strip(" ·")


@dataclass
class ChildChunk:
    """子块：真正进入倒排索引与向量索引的最小检索单元。"""

    child_id: str
    parent_id: str
    doc_id: str
    source_id: str
    order: int
    text: str
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def section_title(self) -> str:
        return str(self.meta.get("section_title", ""))


def _split_parent_into_children(text: str, max_chars: int, overlap: int) -> List[str]:
    """把一个父块的正文切成带重叠的子块。"""
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for sent in sentences:
        # 单句就超过上限：硬切，保证子块不会无限大
        if len(sent) > max_chars:
            if buf:
                chunks.append("".join(buf))
                buf, buf_len = [], 0
            step = max(1, max_chars - overlap)
            for i in range(0, len(sent), step):
                piece = sent[i : i + max_chars]
                if piece.strip():
                    chunks.append(piece)
            continue

        if buf_len + len(sent) > max_chars and buf:
            chunks.append("".join(buf))
            # 保留尾部 overlap 字符作为下一块的前缀，维持跨块语义连续
            tail = "".join(buf)[-overlap:] if overlap > 0 else ""
            buf = [tail] if tail else []
            buf_len = len(tail)

        buf.append(sent)
        buf_len += len(sent)

    if buf:
        tail_text = "".join(buf).strip()
        if tail_text:
            chunks.append(tail_text)
    return chunks


def split_document(
    doc: Document,
    child_chars: int = CHILD_CHUNK_CHARS,
    overlap: int = CHILD_CHUNK_OVERLAP,
    parent_max_chars: int = PARENT_MAX_CHARS,
) -> tuple[List[ParentChunk], List[ChildChunk]]:
    """把一篇文档切成一棵「父块 -> 子块」二级结构。"""
    parents: List[ParentChunk] = []
    children: List[ChildChunk] = []
    doc_meta = doc.metadata_dict()

    for section in doc.sections:
        body = section.text
        if not body.strip():
            continue

        # 超长章节按 parent_max_chars 再切成多个父块，避免单块撑爆上下文
        slices = _slice_section(body, parent_max_chars)
        for sub_idx, sub_text in enumerate(slices):
            suffix = f"-p{sub_idx}" if len(slices) > 1 else ""
            parent_id = f"{doc.source_id}#{section.order}{suffix}"
            meta = dict(doc_meta)
            meta["section_title"] = section.title
            meta["section_order"] = section.order
            parents.append(
                ParentChunk(
                    parent_id=parent_id,
                    doc_id=doc.doc_id,
                    source_id=doc.source_id,
                    order=section.order,
                    section_title=section.title,
                    text=sub_text,
                    meta=meta,
                )
            )
            for c_idx, child_text in enumerate(
                _split_parent_into_children(sub_text, child_chars, overlap)
            ):
                children.append(
                    ChildChunk(
                        child_id=f"{parent_id}-c{c_idx}",
                        parent_id=parent_id,
                        doc_id=doc.doc_id,
                        source_id=doc.source_id,
                        order=c_idx,
                        text=child_text,
                        meta=meta,
                    )
                )
    return parents, children


def _slice_section(text: str, max_chars: int) -> List[str]:
    """章节过长时按段落聚合切成多段。"""
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p for p in text.split("\n") if p.strip()]
    out: List[str] = []
    buf: List[str] = []
    size = 0
    for para in paragraphs:
        if size + len(para) > max_chars and buf:
            out.append("\n".join(buf))
            buf, size = [], 0
        buf.append(para)
        size += len(para) + 1
    if buf:
        out.append("\n".join(buf))
    return out


def build_chunks(
    documents: Sequence[Document],
    child_chars: int = CHILD_CHUNK_CHARS,
    overlap: int = CHILD_CHUNK_OVERLAP,
    parent_max_chars: int = PARENT_MAX_CHARS,
) -> tuple[List[ParentChunk], List[ChildChunk]]:
    """批量切分多篇文档。"""
    all_parents: List[ParentChunk] = []
    all_children: List[ChildChunk] = []
    for doc in documents:
        parents, children = split_document(doc, child_chars, overlap, parent_max_chars)
        all_parents.extend(parents)
        all_children.extend(children)
    return all_parents, all_children
