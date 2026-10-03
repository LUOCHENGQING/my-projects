"""多源文档加载与解析层。

为什么要自己写解析
------------------
金融资料的特点是「同一个语义、十种写法」：制度是扫描件（OCR 文本带错字）、
产品说明是 Excel 导出的 CSV、监管政策是 Word 转的 Markdown、FAQ 是运维维护的 JSON。
一个 RAG 引擎如果只吃干净的 Markdown，上线第一天就会卡在解析上。

因此本层做三件事：

1. **格式适配**：把 .md / .txt(OCR) / .csv / .json 统一解析成同一种
   `SourceDocument` 结构（章节 → 段落块 / 表格块），下游切分逻辑完全不需要知道
   原文是什么格式。
2. **字段映射**：不同业务系统的字段名各不相同（`机构名称` / `institution_name` /
   `org`），统一映射到规范字段（`institution`）。映射表显式配置、可扩展，
   而不是靠模型猜。
3. **表格结构化**：Markdown 表格与 CSV 都解析成 `rows`，切分阶段才能按
   「表头 + N 行数据」成组切块，而不是把一张表按字数硬切断。

数据格式约定（.md）：
    ---
    source_id: POL-2024-07
    doc_type: 监管政策
    institution: 示例监管机构
    effective_date: 2024-07-01
    version: v2
    ---
    # 标题
    ## 一、总则
    正文……（可含 | 表格 |）
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import DATA_DIR
from ..utils.text import mask_sensitive, normalize_text

__all__ = [
    "Block",
    "RawSection",
    "SourceDocument",
    "FAQEntry",
    "Corpus",
    "load_corpus",
    "load_source_document",
    "parse_front_matter",
    "parse_markdown_sections",
    "parse_markdown_table",
    "map_fields",
]

# ---------------------------------------------------------------------------
# 字段映射：业务系统的五花八门 → 规范字段
# ---------------------------------------------------------------------------
FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "source_id": ("source_id", "资料编号", "文档编号", "编号", "id", "doc_no"),
    "title": ("title", "文档标题", "标题", "名称"),
    "doc_type": ("doc_type", "资料类型", "文档类型", "类型", "category"),
    "institution": ("institution", "机构名称", "机构", "公司", "org", "organization", "issuer"),
    "product": ("product", "产品名称", "产品", "product_name"),
    "product_code": ("product_code", "产品代码", "代码", "code"),
    "industry": ("industry", "行业", "所属行业", "sector"),
    "effective_date": ("effective_date", "生效日期", "发布日期", "日期", "date", "publish_date"),
    "version": ("version", "版本", "版本号", "ver"),
    "confidentiality": ("confidentiality", "密级", "保密级别", "security_level"),
    "region": ("region", "地区", "区域", "area"),
}

_ALIAS_LOOKUP: Dict[str, str] = {
    alias.lower(): canonical for canonical, aliases in FIELD_ALIASES.items() for alias in aliases
}

# 规范字段的顺序，写进元数据时保持稳定，便于 diff 与测试断言
CANONICAL_FIELDS: Tuple[str, ...] = tuple(FIELD_ALIASES)

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|[\s:\-|]*$")


def map_fields(raw: Dict[str, object]) -> Dict[str, str]:
    """把任意外部字段名映射成规范字段名。

    未识别的字段原样保留（前缀 `x_`），这样下游元数据过滤仍能用，
    同时不会污染规范字段空间。
    """
    out: Dict[str, str] = {}
    for key, value in raw.items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        canonical = _ALIAS_LOOKUP.get(str(key).strip().lower())
        if canonical:
            out[canonical] = text
        else:
            out.setdefault(f"x_{str(key).strip()}", text)
    return out


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class Block:
    """章节内的一个内容块。

    区分段落块与表格块是**为了切分服务**：表格必须按行组成组切，段落按句累积切，
    两者混在一起切必然把表头与数据行拆散。
    """

    kind: str                     # "paragraph" | "table"
    text: str = ""                # 段落原文 / 表格的 Markdown 原文
    rows: List[List[str]] = field(default_factory=list)  # 表格：第一行为表头

    @property
    def is_table(self) -> bool:
        return self.kind == "table"

    @property
    def header(self) -> List[str]:
        return self.rows[0] if self.rows else []


@dataclass
class RawSection:
    """文档中的一个章节（切分阶段会变成父块）。"""

    order: int
    title: str
    level: int
    blocks: List[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        """章节纯文本（表格渲染成 `列: 值` 行，保证表格内容也能被字面检索命中）。"""
        parts: List[str] = []
        for block in self.blocks:
            if block.is_table:
                parts.append(render_table_rows(block.rows))
            elif block.text.strip():
                parts.append(block.text.strip())
        return "\n".join(p for p in parts if p)

    @property
    def char_count(self) -> int:
        return len(self.text)


def render_table_rows(rows: Sequence[Sequence[str]]) -> str:
    """把表格行渲染成「表头: 值」的文本，让 BM25 也能命中表格里的单元格。"""
    if not rows:
        return ""
    header = [str(h).strip() for h in rows[0]]
    lines: List[str] = []
    for row in rows[1:]:
        cells = [str(c).strip() for c in row]
        pairs = [
            f"{header[i]}: {cells[i]}"
            for i in range(min(len(header), len(cells)))
            if cells[i]
        ]
        if pairs:
            lines.append("；".join(pairs) + "。")
    return "\n".join(lines)


@dataclass
class SourceDocument:
    """一篇解析完成的文档。"""

    doc_id: str
    source_id: str
    title: str
    path: str
    fmt: str
    meta: Dict[str, str] = field(default_factory=dict)
    sections: List[RawSection] = field(default_factory=list)
    raw: str = ""
    # 解析阶段发现的问题（OCR 可疑字符、空章节、表格列数不齐等），进轨迹便于治理
    issues: List[str] = field(default_factory=list)

    # ---- 常用元数据快捷访问 ----
    @property
    def doc_type(self) -> str:
        return self.meta.get("doc_type", "")

    @property
    def institution(self) -> str:
        return self.meta.get("institution", "")

    @property
    def product(self) -> str:
        return self.meta.get("product", "")

    @property
    def effective_date(self) -> str:
        return self.meta.get("effective_date", "")

    @property
    def year(self) -> Optional[int]:
        """从生效日期里抠出年份（用于「只看 2024 年之后」这类过滤）。"""
        raw = self.meta.get("effective_date", "")
        m = re.search(r"(19|20)\d{2}", raw)
        return int(m.group(0)) if m else None

    @property
    def version(self) -> str:
        return self.meta.get("version", "")

    def metadata_dict(self) -> Dict[str, object]:
        """给检索过滤用的扁平元数据。"""
        return {
            "doc_id": self.doc_id,
            "source_id": self.source_id,
            "title": self.title,
            "fmt": self.fmt,
            "doc_type": self.doc_type,
            "institution": self.institution,
            "product": self.product,
            "product_code": self.meta.get("product_code", ""),
            "industry": self.meta.get("industry", ""),
            "effective_date": self.effective_date,
            "year": self.year,
            "version": self.version,
            "confidentiality": self.meta.get("confidentiality", ""),
            "region": self.meta.get("region", ""),
            "path": self.path,
        }

    @property
    def table_count(self) -> int:
        return sum(1 for s in self.sections for b in s.blocks if b.is_table)


@dataclass
class FAQEntry:
    """FAQ 条目：高频问题直出，不走完整链路。"""

    faq_id: str
    question: str
    answer: str
    category: str = ""
    updated_at: str = ""
    source_id: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "faq_id": self.faq_id,
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "updated_at": self.updated_at,
            "source_id": self.source_id,
        }


@dataclass
class Corpus:
    """一个资料库目录解析后的全部内容。"""

    documents: List[SourceDocument] = field(default_factory=list)
    faq: List[FAQEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)

    def get(self, source_id: str) -> Optional[SourceDocument]:
        for doc in self.documents:
            if doc.source_id == source_id:
                return doc
        return None

    def require(self, source_id: str) -> SourceDocument:
        doc = self.get(source_id)
        if doc is None:
            raise KeyError(f"未找到资料编号：{source_id}")
        return doc

    @property
    def source_ids(self) -> List[str]:
        return sorted(d.source_id for d in self.documents)

    @property
    def doc_types(self) -> List[str]:
        return sorted({d.doc_type for d in self.documents if d.doc_type})

    @property
    def institutions(self) -> List[str]:
        return sorted({d.institution for d in self.documents if d.institution})

    @property
    def issues(self) -> List[str]:
        return [f"{d.source_id}: {msg}" for d in self.documents for msg in d.issues]

    def stats(self) -> Dict[str, object]:
        return {
            "documents": len(self.documents),
            "sections": sum(len(d.sections) for d in self.documents),
            "tables": sum(d.table_count for d in self.documents),
            "faq": len(self.faq),
            "doc_types": self.doc_types,
            "formats": sorted({d.fmt for d in self.documents}),
        }


# ---------------------------------------------------------------------------
# Markdown / 表格解析
# ---------------------------------------------------------------------------
def parse_front_matter(text: str) -> Tuple[Dict[str, str], str]:
    """解析 front-matter，返回 (元数据字典, 正文)。刻意不引入 PyYAML。"""
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    meta: Dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, text[match.end():]


def _split_table_row(line: str) -> List[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def parse_markdown_table(lines: Sequence[str]) -> List[List[str]]:
    """把连续的表格行解析成 rows（含表头）。首行是表头，第二行是分隔行。"""
    rows: List[List[str]] = []
    for idx, line in enumerate(lines):
        if idx == 1 and _TABLE_SEP_RE.match(line):
            continue
        cells = _split_table_row(line)
        if any(c for c in cells):
            rows.append(cells)
    return rows


def parse_markdown_sections(body: str) -> List[RawSection]:
    """按 Markdown 标题切章节，章节内部再区分段落块与表格块。"""
    sections: List[RawSection] = []
    title, level, order = "正文", 1, 0
    buffer: List[str] = []

    def flush() -> None:
        nonlocal order, buffer
        blocks = _blocks_from_lines(buffer)
        if blocks:
            sections.append(RawSection(order=order, title=title, level=level, blocks=blocks))
            order += 1
        buffer = []

    for line in body.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            continue
        buffer.append(line)
    flush()
    return sections


def _blocks_from_lines(lines: Sequence[str]) -> List[Block]:
    """把一串原始行切成段落块 / 表格块。"""
    blocks: List[Block] = []
    para: List[str] = []
    table: List[str] = []

    def flush_para() -> None:
        nonlocal para
        text = "\n".join(para).strip()
        if text:
            blocks.append(Block(kind="paragraph", text=text))
        para = []

    def flush_table() -> None:
        nonlocal table
        if table:
            rows = parse_markdown_table(table)
            if rows:
                blocks.append(Block(kind="table", text="\n".join(table), rows=rows))
        table = []

    for line in lines:
        if "|" in line and line.strip().startswith("|"):
            flush_para()
            table.append(line)
            continue
        flush_table()
        if not line.strip():
            flush_para()
            continue
        para.append(line)
    flush_table()
    flush_para()
    return blocks


# ---------------------------------------------------------------------------
# 各格式加载器
# ---------------------------------------------------------------------------
def _document_from_markdown(path: Path) -> SourceDocument:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    meta = map_fields(meta)
    source_id = meta.get("source_id") or path.stem
    title = meta.get("title") or path.stem
    for line in body.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m and len(m.group(1)) == 1:
            title = m.group(2).strip()
            break
    sections = parse_markdown_sections(body)
    return SourceDocument(
        doc_id=source_id,
        source_id=source_id,
        title=title,
        path=str(path),
        fmt="markdown",
        meta=meta,
        sections=sections,
        raw=raw,
    )


def _document_from_text(path: Path) -> SourceDocument:
    """纯文本（典型来源：扫描件 OCR 输出、系统导出的制度 txt）。"""
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    meta = map_fields(meta)
    meta.setdefault("doc_type", "OCR文本")
    source_id = meta.get("source_id") or path.stem
    sections = parse_markdown_sections(body)
    return SourceDocument(
        doc_id=source_id,
        source_id=source_id,
        title=meta.get("title") or path.stem,
        path=str(path),
        fmt="text",
        meta=meta,
        sections=sections,
        raw=raw,
    )


def _document_from_csv(path: Path) -> SourceDocument:
    """CSV（典型来源：产品要素表 / 指标横表）。整表作为一章的表格块。"""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = [[cell.strip() for cell in row] for row in csv.reader(fh) if any(c.strip() for c in row)]
    if not rows:
        raise ValueError(f"CSV 为空：{path}")
    source_id = path.stem
    title = path.stem
    # 允许前两行是 `# key: value` 形式的注释头（业务系统导出常见的写法）
    meta_raw: Dict[str, str] = {}
    while rows and rows[0] and rows[0][0].startswith("#"):
        key, _, value = rows[0][0].lstrip("#").strip().partition(":")
        if key.strip():
            meta_raw[key.strip()] = value.strip()
        rows.pop(0)
    meta = map_fields(meta_raw)
    meta.setdefault("doc_type", "产品要素表")
    source_id = meta.get("source_id") or source_id
    title = meta.get("title") or title
    section = RawSection(order=0, title=title, level=1, blocks=[Block(kind="table", rows=rows)])
    return SourceDocument(
        doc_id=source_id,
        source_id=source_id,
        title=title,
        path=str(path),
        fmt="csv",
        meta=meta,
        sections=[section],
        raw=path.read_text(encoding="utf-8-sig"),
    )


def _document_from_json_records(path: Path, payload: Dict[str, object]) -> SourceDocument:
    """JSON 结构化记录（典型来源：产品库导出）。

    把记录列表渲染成一张表 + 每条记录一段可读文本：表供结构化过滤，
    文本供语义检索。二者并存，避免「结构化字段检索不到、文本字段过滤不了」。
    """
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"JSON 缺少 records 列表：{path}")
    meta = map_fields(
        {k: v for k, v in payload.items() if k != "records" and not isinstance(v, (list, dict))}
    )
    meta.setdefault("doc_type", "结构化记录")
    source_id = meta.get("source_id") or path.stem
    title = meta.get("title") or path.stem

    keys: List[str] = []
    for rec in records:
        if isinstance(rec, dict):
            for k in rec:
                if k not in keys:
                    keys.append(k)
    rows: List[List[str]] = [keys]
    for rec in records:
        if isinstance(rec, dict):
            rows.append([str(rec.get(k, "")).strip() for k in keys])

    blocks: List[Block] = [Block(kind="table", rows=rows)]
    for rec in records:
        if not isinstance(rec, dict):
            continue
        text = "；".join(f"{k}: {rec.get(k)}" for k in keys if str(rec.get(k, "")).strip())
        if text:
            blocks.append(Block(kind="paragraph", text=text + "。"))

    section = RawSection(order=0, title=title, level=1, blocks=blocks)
    return SourceDocument(
        doc_id=source_id,
        source_id=source_id,
        title=title,
        path=str(path),
        fmt="json",
        meta=meta,
        sections=[section],
        raw=json.dumps(payload, ensure_ascii=False),
    )


def load_source_document(path: Path) -> Optional[SourceDocument]:
    """按后缀加载单篇文档；FAQ 文件返回 None（由 load_corpus 单独处理）。"""
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown"):
        return _document_from_markdown(path)
    if suffix == ".txt":
        return _document_from_text(path)
    if suffix == ".csv":
        return _document_from_csv(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "faq" in payload:
            return None
        if isinstance(payload, dict) and "records" in payload:
            return _document_from_json_records(path, payload)
        return None
    return None


def _load_faq(path: Path) -> List[FAQEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("faq") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    out: List[FAQEntry] = []
    for idx, item in enumerate(entries):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        out.append(
            FAQEntry(
                faq_id=str(item.get("faq_id") or f"FAQ-{idx + 1:03d}"),
                question=question,
                answer=answer,
                category=str(item.get("category", "")).strip(),
                updated_at=str(item.get("updated_at", "")).strip(),
                source_id=str(item.get("source_id", "")).strip(),
            )
        )
    return out


def load_corpus(data_dir: Optional[Path] = None) -> Corpus:
    """加载资料库目录下全部支持的文件（按文件名排序，保证顺序确定）。"""
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"资料目录不存在：{directory}")

    corpus = Corpus()
    for path in sorted(p for p in directory.iterdir() if p.is_file()):
        suffix = path.suffix.lower()
        if suffix not in (".md", ".markdown", ".txt", ".csv", ".json"):
            continue
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "faq" in payload:
                corpus.faq.extend(_load_faq(path))
                continue
        try:
            doc = load_source_document(path)
        except Exception as exc:  # noqa: BLE001 - 单篇脏文档不应拖垮整个资料库
            failed = SourceDocument(
                doc_id=path.stem,
                source_id=path.stem,
                title=path.stem,
                path=str(path),
                fmt=suffix.lstrip("."),
                issues=[f"解析失败：{type(exc).__name__}: {exc}"],
            )
            corpus.documents.append(failed)
            continue
        if doc is not None:
            corpus.documents.append(doc)
    return corpus


def apply_masking(documents: Iterable[SourceDocument]) -> int:
    """对全部文档正文做脱敏（在切分之前执行），返回被改动的文档数。

    单独抽成函数而不是在 loader 里直接做，是为了让解析层保持「如实反映原文」，
    脱敏作为可单测、可开关的一步显式执行。
    """
    changed = 0
    for doc in documents:
        original = doc.raw
        masked = mask_sensitive(original)
        if masked != original:
            changed += 1
            doc.raw = masked
            for section in doc.sections:
                for block in section.blocks:
                    block.text = mask_sensitive(block.text)
                    block.rows = [[mask_sensitive(c) for c in row] for row in block.rows]
    return changed


__all__ += ["apply_masking", "render_table_rows"]
