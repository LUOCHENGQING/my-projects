"""父子块切分（Parent-Child Chunking）：把一篇文档变成「父块 -> 子块」两级结构。

本模块在 RAG 全链路中的位置
---------------------------
    ingest 解析出的 SourceDocument（章节 RawSection + 段落/表格 Block）
        -> **本模块：结构感知切分 + 逐块打元数据**
            -> src/index（BM25 倒排、稠密向量、稀疏权重都建在**子块**上）
                -> src/retrieve/hybrid.py（召回子块，再按 parent_id 回捞父块喂给 LLM）
                    -> src/engine.py（生成答案与可追溯引用）

输入：`SourceDocument` 序列（章节里已区分段落块与表格块）。
输出：`(List[ParentChunk], List[ChildChunk])`——父块给 LLM 读，子块给索引命中。
副作用：无 IO、无网络、无全局状态，纯内存计算，因此可被单测穷举。
主要调用方：`src.engine.RAGEngine`（经 `build_chunks`）、`tests/conftest.py`。

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

ID 命名规则（由本模块生成，检索层与缓存层都依赖它做回链）：
    父块 `{source_id}#s{section.order}`，同一章节被拆成多个父块时追加 `-p{sub_idx}`；
    子块 `{parent_id}-c{c_idx}`，`c_idx` 在**父块内**从 0 递增。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import CHILD_CHUNK_CHARS, CHILD_CHUNK_OVERLAP, PARENT_MAX_CHARS, TABLE_ROWS_PER_CHUNK
from ..ingest.loader import Block, RawSection, SourceDocument, render_table_rows
from ..utils.text import split_sentences
from .metadata import extract_metadata

__all__ = ["ParentChunk", "ChildChunk", "ChunkStats", "split_document", "build_chunks", "BLOCK_PARAGRAPH", "BLOCK_TABLE"]

# 块类型标记：写入 ChildChunk.kind 与元数据 block_kind，检索层据此区分「条款文本」与「表格数据」
# （表格块必须按成组策略切、且要重复表头，段落块按句子累积，两者的切分器完全不同）
BLOCK_PARAGRAPH = "paragraph"
BLOCK_TABLE = "table"


@dataclass
class ParentChunk:
    """父块：进入 LLM 上下文窗口的完整语义单元。不直接参与命中。

    典型大小由 `PARENT_MAX_CHARS` 约束，一个父块一般对应文档里的一个完整章节
    （或超大章节被 `_slice_blocks` 切开后的一段），保证送给模型的是「能独立读懂的一段制度」。

    关键属性：
        parent_id       形如 `{source_id}#s{order}`（同章节多父块时带 `-p{sub_idx}`），子块靠它回链
        doc_id          文档内部 ID，对应 `SourceDocument.doc_id`
        source_id       原始文件级 ID，引用与去重都认它
        order           章节序号（取自 `RawSection.order`），用于保持文档原序
        section_title   章节标题，检索结果展示与引用标签都用它
        text            父块正文；表格在拼接时已渲染为「列名: 值」的可读文本
        meta            从 `extract_metadata()` 拿到的扁平元数据（机构/年份/版本/密级等）
        quality_score   文档质量分（默认 1.0），由调用方经 `build_chunks` 传入，只影响排序权重
    """

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
        """人可读出处标签，例如「示例监管机构 · 资产管理产品管理办法 · 四、适当性匹配规则」。

        参数：无。
        返回：由元数据 institution、title 与本块 section_title 用「 · 」拼接的字符串；
              前两项缺失时自动跳过（`strip(" ·")` 收拾多余分隔符），因此**不会**返回开头的空段。
        副作用/异常：无。
        """
        institution = str(self.meta.get("institution", ""))
        title = str(self.meta.get("title", ""))
        head = " · ".join(p for p in (institution, title) if p)
        return f"{head} · {self.section_title}".strip(" ·")

    @property
    def char_count(self) -> int:
        """本父块的字符数（`len(self.text)`，即 Python 字符粒度而非字节数）。

        参数：无。
        返回：非负整数，0 表示空文本。
        副作用/异常：无。
        """
        return len(self.text)

    def to_dict(self) -> Dict[str, object]:
        """导出用于日志 / 体检报告 / API 响应的精简字典（**不含**全文 `text`，避免响应体膨胀）。

        参数：无。
        返回：含 parent_id、source_id、section_title、char_count、quality_score 的 dict；
              `quality_score` 四舍五入到 4 位小数。
        副作用/异常：无。
        """
        return {
            "parent_id": self.parent_id,
            "source_id": self.source_id,
            "section_title": self.section_title,
            "char_count": self.char_count,
            "quality_score": round(self.quality_score, 4),
        }


@dataclass
class ChildChunk:
    """子块：真正进入倒排索引与向量索引的最小检索单元。

    子块「小」是为了命中准（一块只讲一件事），父块「大」是为了让模型读得懂——
    两者用 parent_id 绑定，命中子块后再回捞父块，是本项目控制召回精度的核心手法。

    关键属性：
        child_id      形如 `{parent_id}-c{c_idx}`，`c_idx` 在父块内从 0 递增；引用编号最终落到它身上
        parent_id     所属父块 ID，回捞上下文用
        doc_id        文档内部 ID
        source_id     原始文件级 ID
        order         子块在**父块内**的序号（注意：不是章节号，章节号在 meta["section_order"]）
        text          子块正文；表格子块已按「列名: 值」渲染，便于 BM25 与向量各自命中
        meta          扁平元数据，召回前过滤条件的唯一来源
        kind          BLOCK_PARAGRAPH 或 BLOCK_TABLE，检索层据此做不同的重排/展示
        quality_score 文档质量分，参与排序加权
        row_span      表格子块覆盖的数据行区间（行号从 1 起，表头不算在内）；段落子块为 None
    """

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
        """所属章节标题（读自 `self.meta["section_title"]`，缺字段时返回空串而**不抛异常**）。

        参数：无。
        返回：章节标题字符串，可能为空串。
        副作用/异常：无；元数据缺失被静默降级为空串。
        """
        return str(self.meta.get("section_title", ""))

    @property
    def is_table(self) -> bool:
        """本子块是否来自表格（等价于 `kind == BLOCK_TABLE`）。

        参数：无。
        返回：True 表示表格子块（row_span 有值、正文是「列名: 值」形式）；否则 False。
        副作用/异常：无。
        """
        return self.kind == BLOCK_TABLE

    @property
    def citation_label(self) -> str:
        """人可读出处标签，格式与 `ParentChunk.citation_label` 一致（机构 · 标题 · 章节）。

        参数：无。
        返回：拼接后的出处字符串；元数据缺失时自动省略空段。
        副作用/异常：无。
        """
        institution = str(self.meta.get("institution", ""))
        title = str(self.meta.get("title", ""))
        head = " · ".join(p for p in (institution, title) if p)
        return f"{head} · {self.section_title}".strip(" ·")

    def to_dict(self) -> Dict[str, object]:
        """导出子块字典，供 API / 评测 / 引用展示使用。

        与父块不同，这里**保留全文 `text`**，因为子块是命中的最小单元，展示原文是必要的。
        参数：无。
        返回：含 child_id、parent_id、source_id、section_title、kind、text、row_span、
              quality_score 的 dict；`row_span` 由 tuple 转成 list 以便 JSON 序列化。
        副作用/异常：无。
        """
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
    """切分统计，用于体检切分参数是否合理。

    字段含义（数值越小/越大往往直接指向某个参数没调好）：
        documents               参与切分的文档数（由调用方传入，不是本类算出来的）
        parents / children      父块数、子块总数
        table_children          其中表格子块数（表格被拆散时通常异常偏高）
        paragraph_children      其中段落子块数（= children - table_children）
        min_child_chars         最短子块字符数（过小说明切太碎或表格行过短）
        max_child_chars         最长子块字符数（远超 CHILD_CHUNK_CHARS 说明遇到不可切的超长单句或大表格）
        avg_child_chars         子块平均字符数
        avg_children_per_parent 每个父块平均挂多少子块（衡量父块粒度与召回冗余度）

    由 `chunk_stats()` 构造，最终由 `src.engine.RAGEngine` 写进体检报告。
    """

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
        """导出统计字典（供体检报告 / 评测脚本落盘），两个均值保留 2 位小数。

        参数：无。
        返回：字段名与 dataclass 属性同名的 dict，`avg_child_chars` 与
              `avg_children_per_parent` 已 round 到 2 位。
        副作用/异常：无。
        """
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
    """把一个段落实体按句子累积成带重叠的子块。

    这是「按结构切而不是按字数切」的落地点：累积单位是**句子**，只有遇到不可切的
    超长单句（表格行、长条款）才退化成按字符硬切。

    参数：
        text       段落原文（可以是若干自然段的拼接；空串会直接得到空列表）
        max_chars  单个子块的字符上限，通常传 `CHILD_CHUNK_CHARS`（默认 180）
        overlap    相邻子块的重叠字符数，通常传 `CHILD_CHUNK_OVERLAP`（默认 48）；
                   仅对「句子累积形成的边界」生效，超长单句的硬切窗口同样按它留重叠
    返回：
        List[str]：切好的子块文本，按原文顺序排列；每块已 strip 掉首尾空白。
        无有效句子时返回 `[]`（而不是 `[""]`），保证调用方不会产出空块。
    副作用/异常：
        无副作用（不修改入参、无 IO）；不抛异常，`split_sentences` 返回空即视作无可切内容。
    """
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
            # 步长取 max_chars - overlap 而不是 max_chars，是为了让相邻硬切片仍共享 overlap 个字符，
            # 否则被切断的条款号/数字会只落在其中一片里，另一片永远检索不到
            step = max(1, max_chars - overlap)
            for i in range(0, len(sent), step):
                piece = sent[i : i + max_chars]
                if piece.strip():
                    chunks.append(piece.strip())
            continue

        if buf_len + len(sent) > max_chars and buf:
            chunks.append("".join(buf))
            # 新块以旧块尾部 overlap 个字符开头：跨块语义不断裂，代价是少量重复文本
            tail = "".join(buf)[-overlap:] if overlap > 0 else ""
            buf = [tail] if tail else []
            buf_len = len(tail)

        buf.append(sent)
        buf_len += len(sent)

    if buf:
        tail_text = "".join(buf).strip()
        # 纯空白（常见于重叠尾巴单独成块）直接丢弃，避免索引里出现无内容的块
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

    参数：
        rows          表格的二维文本，`rows[0]` 约定为表头行，其余为数据行；
                      每个单元格会被 `str()` 并 strip
        rows_per_chunk 每组包含的数据行数，默认取 `TABLE_ROWS_PER_CHUNK`（默认 2）。
                      注意这是**上界**：行数不足时最后一组会更短，但绝不会为 0 行
    返回：
        List[Tuple[str, Tuple[int, int]]]：(渲染后的表格文本, (起始行号, 结束行号))。
        行号从 1 起、**不含表头**；只有表头没有数据行时返回 `[("表头文本", (0, 0))]`；
        `rows` 为空时返回 `[]`。
    副作用/异常：
        无副作用；不抛异常（默认参数在函数定义时求值，因此该默认值在导入时即固定）。
    """
    if not rows:
        return []
    header = [str(h).strip() for h in rows[0]]
    body = list(rows[1:])
    if not body:
        # 空表也要留下表头：表头本身可能承载「这张表在统计什么」的信息
        return [(" | ".join(header), (0, 0))]

    step = max(1, rows_per_chunk)
    out: List[Tuple[str, Tuple[int, int]]] = []
    for start in range(0, len(body), step):
        group = body[start : start + step]
        # render_table_rows 会把表头拼进每一行（"列名: 值"），因此不必再单独带一行表头
        text = render_table_rows([header] + list(group))
        # 行号 +1 是因为 body 剔掉了表头，而对外汇报行号要按「第几行数据」算，便于引用定位
        out.append((text, (start + 1, start + len(group))))
    return out


# ---------------------------------------------------------------------------
# 章节 -> 父块
# ---------------------------------------------------------------------------
def _slice_blocks(blocks: Sequence[Block], max_chars: int) -> List[List[Block]]:
    """把章节内的块聚合成若干父块：段落块可拆分边界，表格块尽量完整保留。

    参数：
        blocks     同一章节内的块序列（段落块 `Block.text` / 表格块 `Block.rows`）
        max_chars  单个父块的字符上限，通常传 `PARENT_MAX_CHARS`（默认 1400）
    返回：
        List[List[Block]]：父块分组。空白块被丢弃，因此返回的每个分组都非空；
        原文顺序保持不变。
    副作用/异常：
        无副作用（不修改传入的 `Block`；拆超大表格时用 `Block(...)` 新建对象）；
        不抛异常。
    实现要点：
        分组长度的口径是 `len(block.text) or len(render_table_rows(block.rows))`——
        表格块通常没有 text，所以用渲染后的文本长度估算，否则大表格会被误判成 0 长度。
    """
    groups: List[List[Block]] = []
    current: List[Block] = []
    size = 0

    for block in blocks:
        # 表格块往往 text 为空、真正的内容在 rows 里，必须回退到渲染长度，否则会当成空块丢掉
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
                # 只有表头却仍超长：无法再拆，单列为一个父块
                groups.append([block])
                continue
            buf_rows: List[List[str]] = []
            # 初始长度算上表头，因为每个父块都会重复渲染表头
            buf_size = len(" | ".join(header))
            for row in body:
                row_len = len(" | ".join(str(c) for c in row))
                if buf_rows and buf_size + row_len > max_chars:
                    # 新父块重建为「表头 + 已缓冲数据行」：父块同样不能丢表头
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
    """把一篇文档切成一棵「父块 -> 子块」二级结构。

    这是本模块的主入口（`build_chunks` 的多文档版，实际逐篇调用它）。

    参数：
        doc              已解析的文档对象 `SourceDocument`；只有 `section.text` 非空的章节参与切分
        child_chars      子块字符上限，默认 `CHILD_CHUNK_CHARS`（默认 180）
        overlap          子块重叠字符数，默认 `CHILD_CHUNK_OVERLAP`（默认 48）
        parent_max_chars 父块字符上限，默认 `PARENT_MAX_CHARS`（默认 1400）
        rows_per_chunk   表格每组数据行数，默认 `TABLE_ROWS_PER_CHUNK`（默认 2）
        quality_score    写进每个父块/子块的质量分，默认 1.0
    返回：
        Tuple[List[ParentChunk], List[ChildChunk]]：父块与子块列表。
        两者按文档顺序生成；父块文本为空时该父块及其子块一起被跳过，
        因此**子块的 parent_id 一定能在父块列表里找到对应项**（回捞链路不会断）。
    副作用/异常：
        无副作用、无 IO；不抛异常（个别章节结构异常只会被跳过）。
    """
    parents: List[ParentChunk] = []
    children: List[ChildChunk] = []

    for section in doc.sections:
        # 空章节（往往只有标题或只有空白）不产生任何块，防止索引里出现无内容可命中的噪声
        if not section.text.strip():
            continue
        groups = _slice_blocks(section.blocks, parent_max_chars)
        for sub_idx, group in enumerate(groups):
            # 只有章节被拆成多段时才加 -p 后缀，避免单父块场景下 ID 里出现无意义的下标
            suffix = f"-p{sub_idx}" if len(groups) > 1 else ""
            parent_id = f"{doc.source_id}#s{section.order}{suffix}"
            # 一个分组里只要混有段落，就按段落口径归为 paragraph；
            # 因为父块是给人/模型读的混合文本，标成 table 会误导展示与重排逻辑
            kinds = {BLOCK_TABLE if b.is_table else BLOCK_PARAGRAPH for b in group}
            kind = BLOCK_TABLE if kinds == {BLOCK_TABLE} else BLOCK_PARAGRAPH
            # 表格块在父块里同样走 render_table_rows 渲染，保证与子块看到的文本形式一致
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

            # c_idx 在父块内连续递增：子块 ID 因此可读、可复现，也方便按父块做引用聚合
            c_idx = 0
            for block in group:
                if block.is_table:
                    # 表格子块 = 表头 + N 行数据（每行重复列名），并附带行区间用于引用定位
                    for text, span in split_table_into_children(block.rows, rows_per_chunk):
                        if not text.strip():
                            continue
                        # 每个子块都重新抽一次元数据（而不是复用父块的），保证块级过滤条件齐全
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
                    # 表格块已按上面的成组逻辑产出子块，不能再走段落切分，否则表头会与数据行分离
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
    """批量切分多篇文档，并汇总切分统计。

    参数：
        documents       文档序列（通常是 ingest 清洗后的全量语料）
        child_chars     子块字符上限，透传给 `split_document`，默认 `CHILD_CHUNK_CHARS`
        overlap         子块重叠字符数，透传，默认 `CHILD_CHUNK_OVERLAP`
        parent_max_chars 父块字符上限，透传，默认 `PARENT_MAX_CHARS`
        rows_per_chunk  表格每组数据行数，透传，默认 `TABLE_ROWS_PER_CHUNK`
        quality_scores  可选的 {source_id: 质量分} 映射；缺省时所有文档按 1.0 处理
    返回：
        Tuple[List[ParentChunk], List[ChildChunk]]：全量父块与子块，按文档顺序拼接。
        **注**：函数名虽然叫 build，本身不写索引也不落盘，只做纯内存切分；
        建 BM25 / 向量索引由 `src/retrieve/hybrid.py` 在拿到这两个列表后进行。
    副作用/异常：
        无副作用；不抛异常。
    """
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
    """汇总切分统计，用于判断切分参数是否合理（块太小 / 太大 / 表格被拆散）。

    参数：
        parents        父块序列
        children       子块序列（统计口径全部基于子块，因为子块才是检索单元）
        document_count 文档数。**注：实际实现为**直接写入结果，不由本函数统计，
                       调用方需自行传入（`RAGEngine` 传的是语料文档数）
    返回：
        ChunkStats：各项统计。`children` 为空时长度类字段按 [0] 兜底（min=max=0、
        avg=0.0），`parents` 为空时 `avg_children_per_parent` 为 0.0，均不会除零。
    副作用/异常：
        无副作用、无 IO；不抛异常。
    """
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
