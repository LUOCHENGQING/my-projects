"""父子块切分（Parent-Child Chunking）。

层次与职责
----------
RAG 证据层的第一道工序：把 corpus.load_documents() 得到的 Document 变成
「父块（ParentChunk，整章节，回填给 LLM 的上下文单位）+ 子块（ChildChunk，
真正进 BM25 与向量索引的最小检索单元）」的二级结构。它只切分、不编码、不打分。

关键类 / 函数：
    ParentChunk / ChildChunk  —— 两种粒度的数据载体（dataclass），带元数据与出处
    split_document            —— 单篇文档 -> (父块列表, 子块列表)
    build_chunks              —— 多篇文档批量切分，返回 (全部父块, 全部子块)

主要输入：Document（已解析出 sections 与 meta），以及三个长度参数（默认取自
config：CHILD_CHUNK_CHARS / CHILD_CHUNK_OVERLAP / PARENT_MAX_CHARS）。
主要输出：(List[ParentChunk], List[ChildChunk])，并由 orchestrator 交给
             HybridRetriever 建索引。
被谁调用：src/orchestrator.py（ResearchPipeline.__init__）、tests/ 与 eval/。

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

ID 与参数口径
-------------
    parent_id = "{source_id}#{section.order}"，章节被 PARENT_MAX_CHARS 再切开时后缀 "-p{sub_idx}"
    child_id  = "{parent_id}-c{c_idx}"
参数默认值来自 config（可用环境变量覆盖）：CHILD_CHUNK_CHARS=160、
CHILD_CHUNK_OVERLAP=40、PARENT_MAX_CHARS=1600。
本模块是纯函数式的：不读写磁盘、不改动传入的 Document，输出顺序完全由输入决定。
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
    """父块：进入上下文窗口的完整章节（超长章节会按 PARENT_MAX_CHARS 再切）。

    关键属性：
        parent_id / doc_id / source_id —— 三级标识，引用溯源与回填都依赖它
        order         —— 所属章节在文档内的序号（= corpus.Section.order）
        section_title —— 章节标题，如「六、主要风险因素」
        text          —— 父块正文，检索时作为 context 回填给 LLM
        meta          —— 扁平元数据（doc_id / source_id / title / company / doc_type /
                         period / year / path，另加 section_title / section_order），
                         供 HybridRetriever 做元数据加权与 strict 硬过滤
    状态流转：构造后只读。子块通过 parent_id 指回本对象，回填关系在
    HybridRetriever._parent_by_id 中建立。
    """

    parent_id: str
    doc_id: str
    source_id: str
    order: int
    section_title: str
    text: str
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def citation_label(self) -> str:
        """人可读的出处标签，例如「示例科技股份有限公司 2024年年度 · 六、主要风险因素」。

        参数：无（属性访问）。
        返回：str —— 由 meta 里的 company / period 与 section_title 拼接，末端的
            " ·" 会被 strip 掉，因此缺字段时不会留下孤立的间隔符。
        副作用：无。
        """
        company = self.meta.get("company", "")
        period = self.meta.get("period", "")
        return f"{company} {period} · {self.section_title}".strip(" ·")


@dataclass
class ChildChunk:
    """子块：真正进入倒排索引与向量索引的最小检索单元。

    关键属性：
        child_id —— 全局唯一检索主键（"{parent_id}-c{order}"），也是 Writer 引用编号的落点
        parent_id —— 回填父块上下文的指针
        doc_id / source_id —— 出处标识（source_id 即资料编号，如 EX-TECH-2024-AR）
        order —— 子块在父块内部的序号 c_idx
        text —— 子块正文，BM25 分词与哈希向量的直接输入
        meta —— 元数据；与同父块的 ParentChunk 是**同一个对象引用**（非副本），
                 检索期元数据过滤读的就是它
    状态流转：构造后只读。
    """

    child_id: str
    parent_id: str
    doc_id: str
    source_id: str
    order: int
    text: str
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def section_title(self) -> str:
        """章节标题的快捷访问（从 meta 取，缺失时返回空串）。

        参数：无（属性访问）。
        返回：str。副作用：无。
        """
        return str(self.meta.get("section_title", ""))


def _split_parent_into_children(text: str, max_chars: int, overlap: int) -> List[str]:
    """把一个父块的正文切成带重叠的子块。

    参数：
        text: 父块正文（已经是单段，不再含标题）。
        max_chars: 子块字符上限（config.CHILD_CHUNK_CHARS）。
        overlap: 相邻子块的重叠字符数（config.CHILD_CHUNK_OVERLAP）。
    返回：子块文本列表，按原文顺序；空文本返回 []。
    副作用：无。整句超出 max_chars 时会硬切（不保证句界完整）。
    """
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
            # 为什么步进取 max_chars - overlap：硬切也沿用与滑窗子块相同的重叠粒度，
            # 避免硬切段和相邻块之间出现语义断层；max(1, ...) 防 overlap >= max_chars 时死循环。
            step = max(1, max_chars - overlap)
            for i in range(0, len(sent), step):
                piece = sent[i : i + max_chars]
                if piece.strip():
                    chunks.append(piece)
            continue

        if buf_len + len(sent) > max_chars and buf:
            chunks.append("".join(buf))
            # 保留尾部 overlap 字符作为下一块的前缀，维持跨块语义连续
            # 为什么把 tail 的长度也计入 buf_len：下一块是从这段重叠文本开始累积的，
            # 不预扣长度就会让实际子块超出 max_chars。
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
    """把一篇文档切成一棵「父块 -> 子块」二级结构。

    参数：
        doc: 已解析的 Document（用其 sections、source_id、doc_id 与元数据）。
        child_chars / overlap: 子块长度与重叠，透传给 _split_parent_into_children。
        parent_max_chars: 父块上限，超长章节先按它再切（config.PARENT_MAX_CHARS）。
    返回：(parents, children) —— 顺序与 doc.sections 一致，ID 规则见模块 docstring。
    副作用：无（不修改 doc）。空白章节与空白子块会被跳过。
    """
    parents: List[ParentChunk] = []
    children: List[ChildChunk] = []
    doc_meta = doc.metadata_dict()

    for section in doc.sections:
        body = section.text
        if not body.strip():
            continue

        # 超长章节按 parent_max_chars 再切成多个父块，避免单块撑爆上下文
        # 为什么要给父块也设上限：context 是回填给 LLM 的预算，单块过大不仅挤占其他
        # 证据的位置，也会稀释这一块自身的信噪比。
        slices = _slice_section(body, parent_max_chars)
        for sub_idx, sub_text in enumerate(slices):
            # 为什么只在多段时加后缀：单段父块的 parent_id 保持「文档#章节序号」的干净形态，
            # 人读 trace 与引用时更直观。
            suffix = f"-p{sub_idx}" if len(slices) > 1 else ""
            parent_id = f"{doc.source_id}#{section.order}{suffix}"
            # 为什么父块与子块共用同一份 meta 对象：两者是同一段文本的两种粒度，
            # 元数据必然一致，没必要复制；注意这是同一个 dict 引用，下游请勿原地改写。
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
    """章节过长时按段落聚合切成多段。

    参数：text 章节正文；max_chars 单段上限（config.PARENT_MAX_CHARS）。
    返回：段落聚合后的文本列表；未超限时原样返回 [text]。
    副作用：无。段内不做二次切分——超长单段落会自成一个超限父块（保持段落完整优先）。
    """
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
        # 为什么 +1：最终用 "\n".join 拼回，每个段落之间会多出一个换行符，先算进去。
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
    """批量切分多篇文档。

    参数：documents 文档序列；其余长度参数与 split_document 同义。
    返回：(全部父块, 全部子块)，按 documents 顺序拼接，供 HybridRetriever 直接建索引。
    副作用：无。
    """
    all_parents: List[ParentChunk] = []
    all_children: List[ChildChunk] = []
    for doc in documents:
        parents, children = split_document(doc, child_chars, overlap, parent_max_chars)
        all_parents.extend(parents)
        all_children.extend(children)
    return all_parents, all_children
