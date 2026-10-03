"""父子块切分（Parent-Child Chunking）。

这是决定 RAG 效果的关键环节，也是最容易做砸的地方。
切分的两难是公开的：**切太碎**每块信息不完整、丢上下文，模型拿到半句话没法作答；
**切太大**一块里塞了很多无关内容，命中精度下降，还白白吃掉上下文窗口。

本模块的答案：**两级结构 + 按文档结构切 + 表格单独处理**。

    ┌─ 父块（= 语义完整的章节，进 LLM 上下文）───────────────────────┐
    │  ## 四、适当性匹配规则                                          │
    │  ┌─ 子块 c0 ─────────┐  ┌─ 子块 c1 ─────────┐                  │
    │  │ 第X条……（进索引） │  │ 第Y条……（进索引） │                  │
    │  └───────────────────┘  └───────────────────┘                  │
    └────────────────────────────────────────────────────────────────┘

三条硬规矩：

1. **切分依据是文档自身的结构**（标题层级 / 段落 / 表格），不是固定字数。
   按固定字数硬切会出现「一句话被切成两半」，命中了也读不出完整意思。
2. **表格必须成组切**：表头 + N 行数据为一组。把表头和数据行拆到不同块里，
   「R4 产品不得推荐给 C3 客户」这条信息就永远拼不回来。
3. **子块带重叠**：相邻子块保留 CHILD_CHUNK_OVERLAP 字符，避免跨块语义断裂。

引用可追溯性建立在 ID 体系上：
    答案里的 [1] -> 引用记录 -> 子块 ID -> 父块 ID -> 文档 source_id -> 原文件路径。
这条链路任何一环断了，「回答可溯源」就是空话。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import CHILD_CHUNK_CHARS, CHILD_CHUNK_OVERLAP, PARENT_MAX_CHARS, TABLE_ROWS_PER_CHUNK
from ..ingest.loader import Block, RawSection, SourceDocument, render_table_rows
from ..utils.text import split_sentences
from .metadata import extract_metadata

__all__ = ["ParentChunk", "ChildChunk", "ChunkStats", "split_document", "build_chunks", "BLOCK_PARAGRAPH", "BLOCK_TABLE"]

BLOCK_PARAGRAPH = "paragraph"
BLOCK_TABLE = "table"


@dataclass
class ParentChunk:
    """父块：进入 LLM 上下文窗口的完整语义单元。不直接参与命中。"""

    parent_id: str
    doc_id: str
    source_id: str
    order: int
    section_title: str
    text: str
    meta: Dict[str, object] = field(default_factory=dict)
    quality_score: float = 1.0

    @property
    def citation_label(self) -> str:
        """人可读出处标签，例如「示例监管机构 · 资产管理产品管理办法 · 四、适当性匹配规则」。"""
        institution = str(self.meta.get("institution", ""))
        title = str(self.meta.get("title", ""))
        head = " · ".join(p for p in (institution, title) if p)
        return f"{head} · {self.section_title}".strip(" ·")

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> Dict[str, object]:
        return {
            "parent_id": self.parent_id,
            "source_id": self.source_id,
            "section_title": self.section_title,
            "char_count": self.char_count,
            "quality_score": round(self.quality_score, 4),
        }


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
    kind: str = BLOCK_PARAGRAPH
    quality_score: float = 1.0
    row_span: Optional[Tuple[int, int]] = None   # 表格子块覆盖的数据行区间（含表头为 0）

    @property
    def section_title(self) -> str:
        return str(self.meta.get("section_title", ""))

    @property
    def is_table(self) -> bool:
        return self.kind == BLOCK_TABLE

    @property
    def citation_label(self) -> str:
        institution = str(self.meta.get("institution", ""))
        title = str(self.meta.get("title", ""))
        head = " · ".join(p for p in (institution, title) if p)
        return f"{head} · {self.section_title}".strip(" ·")

    def to_dict(self) -> Dict[str, object]:
        return {
            "child_id": self.child_id,
            "parent_id": self.parent_id,
            "source_id": self.source_id,
            "section_title": self.section_title,
            "kind": self.kind,
            "text": self.text,
            "row_span": list(self.row_span) if self.row_span else None,
            "quality_score": round(self.quality_score, 4),
        }


@dataclass
class ChunkStats:
    """切分统计，用于体检切分参数是否合理。"""

    documents: int = 0
    parents: int = 0
    children: int = 0
    table_children: int = 0
    paragraph_children: int = 0
    min_child_chars: int = 0
    max_child_chars: int = 0
    avg_child_chars: float = 0.0
    avg_children_per_parent: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "documents": self.documents,
            "parents": self.parents,
            "children": self.children,
            "table_children": self.table_children,
            "paragraph_children": self.paragraph_children,
            "min_child_chars": self.min_child_chars,
            "max_child_chars": self.max_child_chars,
            "avg_child_chars": round(self.avg_child_chars, 2),
            "avg_children_per_parent": round(self.avg_children_per_parent, 2),
        }


# ---------------------------------------------------------------------------
# 段落块 -> 子块
# ---------------------------------------------------------------------------
def split_paragraph_into_children(text: str, max_chars: int, overlap: int) -> List[str]:
    """把一个段落实体按句子累积成带重叠的子块。"""
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for sent in sentences:
        # 单句就超过上限（表格行、长条款）：硬切，保证子块不会无限大
        if len(sent) > max_chars:
            if buf:
                chunks.append("".join(buf))
                buf, buf_len = [], 0
            step = max(1, max_chars - overlap)
            for i in range(0, len(sent), step):
                piece = sent[i : i + max_chars]
                if piece.strip():
                    chunks.append(piece.strip())
            continue

        if buf_len + len(sent) > max_chars and buf:
            chunks.append("".join(buf))
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


# ---------------------------------------------------------------------------
# 表格块 -> 子块
# ---------------------------------------------------------------------------
def split_table_into_children(
    rows: Sequence[Sequence[str]],
    rows_per_chunk: int = TABLE_ROWS_PER_CHUNK,
) -> List[Tuple[str, Tuple[int, int]]]:
    """表格成组切块：每组都**重复带上表头**，返回 (文本, 行区间)。

    重复表头是刻意的：表格命中的是某几行数据，但交给 LLM 时必须让它知道
    「这一列叫什么」。否则模型只能靠猜，数字口径必然漂移。
    """
    if not rows:
        return []
    header = [str(h).strip() for h in rows[0]]
    body = list(rows[1:])
    if not body:
        return [(" | ".join(header), (0, 0))]

    step = max(1, rows_per_chunk)
    out: List[Tuple[str, Tuple[int, int]]] = []
    for start in range(0, len(body), step):
        group = body[start : start + step]
        # render_table_rows 会把表头拼进每一行（"列名: 值"），因此不必再单独带一行表头
        text = render_table_rows([header] + list(group))
        out.append((text, (start + 1, start + len(group))))
    return out


# ---------------------------------------------------------------------------
# 章节 -> 父块
# ---------------------------------------------------------------------------
def _slice_blocks(blocks: Sequence[Block], max_chars: int) -> List[List[Block]]:
    """把章节内的块聚合成若干父块：段落块可拆分边界，表格块尽量完整保留。"""
    groups: List[List[Block]] = []
    current: List[Block] = []
    size = 0

    for block in blocks:
        block_len = len(block.text) or len(render_table_rows(block.rows))
        if block_len == 0:
            continue
        # 超大表格：按行拆成多个父块，每块都带表头
        if block.is_table and block_len > max_chars:
            if current:
                groups.append(current)
                current, size = [], 0
            rows = block.rows
            header = rows[0]
            body = rows[1:]
            if not body:
                groups.append([block])
                continue
            buf_rows: List[List[str]] = []
            buf_size = len(" | ".join(header))
            for row in body:
                row_len = len(" | ".join(str(c) for c in row))
                if buf_rows and buf_size + row_len > max_chars:
                    groups.append([Block(kind="table", rows=[header] + buf_rows)])
                    buf_rows, buf_size = [], len(" | ".join(header))
                buf_rows.append(list(row))
                buf_size += row_len
            if buf_rows:
                groups.append([Block(kind="table", rows=[header] + buf_rows)])
            continue

        if size + block_len > max_chars and current:
            groups.append(current)
            current, size = [], 0
        current.append(block)
        size += block_len

    if current:
        groups.append(current)
    return [g for g in groups if g]


def split_document(
    doc: SourceDocument,
    child_chars: int = CHILD_CHUNK_CHARS,
    overlap: int = CHILD_CHUNK_OVERLAP,
    parent_max_chars: int = PARENT_MAX_CHARS,
    rows_per_chunk: int = TABLE_ROWS_PER_CHUNK,
    quality_score: float = 1.0,
) -> Tuple[List[ParentChunk], List[ChildChunk]]:
    """把一篇文档切成一棵「父块 -> 子块」二级结构。"""
    parents: List[ParentChunk] = []
    children: List[ChildChunk] = []

    for section in doc.sections:
        if not section.text.strip():
            continue
        groups = _slice_blocks(section.blocks, parent_max_chars)
        for sub_idx, group in enumerate(groups):
            suffix = f"-p{sub_idx}" if len(groups) > 1 else ""
            parent_id = f"{doc.source_id}#s{section.order}{suffix}"
            kinds = {BLOCK_TABLE if b.is_table else BLOCK_PARAGRAPH for b in group}
            kind = BLOCK_TABLE if kinds == {BLOCK_TABLE} else BLOCK_PARAGRAPH
            parent_text = "\n".join(
                render_table_rows(b.rows) if b.is_table else b.text.strip() for b in group
            ).strip()
            if not parent_text:
                continue
            parent_meta = extract_metadata(doc, section, block_kind=kind)
            parents.append(
                ParentChunk(
                    parent_id=parent_id,
                    doc_id=doc.doc_id,
                    source_id=doc.source_id,
                    order=section.order,
                    section_title=section.title,
                    text=parent_text,
                    meta=parent_meta,
                    quality_score=quality_score,
                )
            )

            c_idx = 0
            for block in group:
                if block.is_table:
                    for text, span in split_table_into_children(block.rows, rows_per_chunk):
                        if not text.strip():
                            continue
                        meta = extract_metadata(doc, section, block_kind=BLOCK_TABLE)
                        children.append(
                            ChildChunk(
                                child_id=f"{parent_id}-c{c_idx}",
                                parent_id=parent_id,
                                doc_id=doc.doc_id,
                                source_id=doc.source_id,
                                order=c_idx,
                                text=text,
                                meta=meta,
                                kind=BLOCK_TABLE,
                                quality_score=quality_score,
                                row_span=span,
                            )
                        )
                        c_idx += 1
                    continue

                for text in split_paragraph_into_children(block.text, child_chars, overlap):
                    meta = extract_metadata(doc, section, block_kind=BLOCK_PARAGRAPH)
                    children.append(
                        ChildChunk(
                            child_id=f"{parent_id}-c{c_idx}",
                            parent_id=parent_id,
                            doc_id=doc.doc_id,
                            source_id=doc.source_id,
                            order=c_idx,
                            text=text,
                            meta=meta,
                            kind=BLOCK_PARAGRAPH,
                            quality_score=quality_score,
                        )
                    )
                    c_idx += 1
    return parents, children


def build_chunks(
    documents: Sequence[SourceDocument],
    child_chars: int = CHILD_CHUNK_CHARS,
    overlap: int = CHILD_CHUNK_OVERLAP,
    parent_max_chars: int = PARENT_MAX_CHARS,
    rows_per_chunk: int = TABLE_ROWS_PER_CHUNK,
    quality_scores: Optional[Dict[str, float]] = None,
) -> Tuple[List[ParentChunk], List[ChildChunk]]:
    """批量切分多篇文档，并汇总切分统计。"""
    scores = quality_scores or {}
    all_parents: List[ParentChunk] = []
    all_children: List[ChildChunk] = []
    for doc in documents:
        parents, children = split_document(
            doc,
            child_chars=child_chars,
            overlap=overlap,
            parent_max_chars=parent_max_chars,
            rows_per_chunk=rows_per_chunk,
            quality_score=float(scores.get(doc.source_id, 1.0)),
        )
        all_parents.extend(parents)
        all_children.extend(children)
    return all_parents, all_children


def chunk_stats(
    parents: Sequence[ParentChunk],
    children: Sequence[ChildChunk],
    document_count: int = 0,
) -> ChunkStats:
    """汇总切分统计，用于判断切分参数是否合理（块太小 / 太大 / 表格被拆散）。"""
    lengths = [len(c.text) for c in children] or [0]
    table_children = sum(1 for c in children if c.is_table)
    stats = ChunkStats(
        documents=document_count,
        parents=len(parents),
        children=len(children),
        table_children=table_children,
        paragraph_children=len(children) - table_children,
        min_child_chars=min(lengths),
        max_child_chars=max(lengths),
        avg_child_chars=sum(lengths) / len(lengths),
        avg_children_per_parent=(len(children) / len(parents)) if parents else 0.0,
    )
    return stats
