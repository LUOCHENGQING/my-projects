"""文档加载与解析层。

data/*.md 采用「YAML 风格 front-matter + Markdown」格式，便于人读也便于机器解析：

    ---
    source_id: EX-TECH-2024-AR
    company: 示例科技股份有限公司
    doc_type: 年度报告摘要
    year: 2024
    ---
    # 标题
    ## 一、公司基本情况
    正文……

刻意不引入 PyYAML：front-matter 只有扁平的 key: value，手写解析足够，
也让依赖树保持精简。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..config import DATA_DIR

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Section:
    """文档中的一个章节（= 一个父块）。"""

    doc_id: str
    order: int
    title: str
    level: int
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class Document:
    """一篇解析完成的文档。"""

    doc_id: str
    source_id: str
    title: str
    path: str
    meta: Dict[str, str] = field(default_factory=dict)
    sections: List[Section] = field(default_factory=list)
    raw: str = ""

    # ---- 常用元数据快捷访问 ----
    @property
    def company(self) -> str:
        return self.meta.get("company", "")

    @property
    def doc_type(self) -> str:
        return self.meta.get("doc_type", "")

    @property
    def period(self) -> str:
        return self.meta.get("period", "")

    @property
    def year(self) -> Optional[int]:
        raw = self.meta.get("year", "")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def metadata_dict(self) -> Dict[str, object]:
        """给检索结果用的扁平元数据。"""
        return {
            "doc_id": self.doc_id,
            "source_id": self.source_id,
            "title": self.title,
            "company": self.company,
            "doc_type": self.doc_type,
            "period": self.period,
            "year": self.year,
            "path": self.path,
        }


def parse_front_matter(text: str) -> tuple[Dict[str, str], str]:
    """解析 front-matter，返回 (元数据字典, 正文)。"""
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


def split_sections(doc_id: str, body: str) -> List[Section]:
    """按 Markdown 标题把正文切成章节。"""
    sections: List[Section] = []
    current_title = "正文"
    current_level = 1
    buffer: List[str] = []
    order = 0

    def flush() -> None:
        nonlocal order, buffer
        text = "\n".join(buffer).strip()
        if text:
            sections.append(
                Section(doc_id=doc_id, order=order, title=current_title, level=current_level, text=text)
            )
            order += 1
        buffer = []

    for line in body.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            flush()
            current_level = len(m.group(1))
            current_title = m.group(2).strip()
            continue
        buffer.append(line)
    flush()
    return sections


def load_document(path: Path) -> Document:
    """加载单篇文档。"""
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    source_id = meta.get("source_id") or path.stem
    doc_id = source_id
    title = path.stem
    for line in body.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m and len(m.group(1)) == 1:
            title = m.group(2).strip()
            break
    sections = split_sections(doc_id, body)
    return Document(
        doc_id=doc_id,
        source_id=source_id,
        title=title,
        path=str(path),
        meta=meta,
        sections=sections,
        raw=raw,
    )


class DocumentStore:
    """一个目录下全部文档的只读集合。"""

    def __init__(self, documents: Sequence[Document]) -> None:
        self.documents: List[Document] = list(documents)
        self._by_source: Dict[str, Document] = {d.source_id: d for d in self.documents}

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)

    def get(self, source_id: str) -> Optional[Document]:
        return self._by_source.get(source_id)

    def require(self, source_id: str) -> Document:
        doc = self.get(source_id)
        if doc is None:
            raise KeyError(f"未找到资料编号：{source_id}")
        return doc

    @property
    def companies(self) -> List[str]:
        return sorted({d.company for d in self.documents if d.company})

    @property
    def source_ids(self) -> List[str]:
        return sorted(self._by_source)

    def section_text(self, source_id: str, title_keyword: str) -> Optional[str]:
        """按章节标题关键词取原文，供事实抽取 / 引用定位使用。"""
        doc = self.get(source_id)
        if doc is None:
            return None
        for section in doc.sections:
            if title_keyword in section.title:
                return section.text
        return None

    def locate(self, source_id: str, needle: str) -> Optional[Section]:
        """定位包含某段文本的章节，用于让引用带上「出处章节」。"""
        doc = self.get(source_id)
        if doc is None:
            return None
        for section in doc.sections:
            if needle and needle in section.text:
                return section
        return None


def load_documents(data_dir: Optional[Path] = None) -> DocumentStore:
    """加载 data/ 下全部 .md 文档（按文件名排序，保证顺序确定）。"""
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"资料目录不存在：{directory}")
    paths = sorted(p for p in directory.glob("*.md") if p.is_file())
    return DocumentStore([load_document(p) for p in paths])
