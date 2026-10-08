"""文档加载与解析层（data/*.md -> Document）。

层次与职责
----------
RAG 证据层的入口：把磁盘上的 Markdown 原文读成结构化的 Document（front-matter
元数据 + 按标题切好的 Section 列表），供 chunking 切块、facts 抽表和 DocumentStore
做「按资料编号定位出处」。本模块只做读取与解析，不切块、不编码、不打分。

关键类 / 函数：
    Section / Document —— 章节与文档的数据载体（dataclass）
    DocumentStore      —— 目录级只读集合，提供 get / require / section_text / locate
    parse_front_matter / split_sections / load_document / load_documents —— 解析流水线

主要输入：data/*.md（或显式传入 data_dir）。
主要输出：DocumentStore（可迭代出 Document）。
被谁调用：src/orchestrator.py（ResearchPipeline.__init__ 调 load_documents），
          src/rag/facts.py 与 src/tools/builtin.py 消费其中的 Document。

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

解析口径与边界
--------------
    * front-matter 只在文件开头匹配（^--- / --- 包裹）；当前 data/ 里的键为
      source_id / company / doc_type / period / year / publisher / disclaimer，
      值一律按 str 保存，不做类型转换（数值转换交给 facts._parse_number）。
    * 章节以 Markdown 标题为界，标题行本身不进正文；任何标题出现之前的正文
      归入默认章节「正文」。没有一级标题时，Document.title 退回文件名词干。
    * 只读：不写回文件。load_documents 只扫描目录**第一层**的 *.md，且按文件名
      排序，保证构建顺序与运行结果可复现。
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
    """文档中的一个章节（= 一个父块）。

    关键属性：
        doc_id —— 所属文档编号（注：实际实现为 doc_id == source_id）
        order  —— 章节在文档内的 0 基序号；chunking 用它拼 parent_id
        title  —— 标题文本（不含 # 号）
        level  —— 标题层级（# 的个数，1~6）
        text   —— 该章节正文，不含标题行
    状态流转：构造后只读，不做二次加工。
    """

    doc_id: str
    order: int
    title: str
    level: int
    text: str

    @property
    def char_count(self) -> int:
        """章节正文字符数（供统计与调试观察块大小）。

        参数：无（属性访问）。返回：int。副作用：无。
        """
        return len(self.text)


@dataclass
class Document:
    """一篇解析完成的文档。

    关键属性：
        doc_id / source_id —— 文档编号；注：实际实现为 load_document 里 doc_id = source_id
            （同一个 source_id 既是资料编号，也是文档编号，facts.FactStore 的索引键用 company）
        title  —— 一级标题文本；没有一级标题时退回文件名词干
        path   —— 源文件绝对路径（也写进 meta，便于回溯）
        meta   —— front-matter 的原始 key/value（值全是 str）
        sections —— 章节列表，顺序即文中出现顺序
        raw    —— 文件原始全文；facts.parse_tables 直接按行扫描它来抽表格
    状态流转：构造后只读，检索与抽取阶段都只是消费者。
    """

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
        """公司全称（meta["company"]）；缺失时返回空串。

        参数：无（属性访问）。返回：str。副作用：无。
        """
        return self.meta.get("company", "")

    @property
    def doc_type(self) -> str:
        """文档类型（meta["doc_type"]），如「年度报告摘要」「三季度报告」；缺失返回空串。

        参数：无（属性访问）。返回：str。副作用：无。
        """
        return self.meta.get("doc_type", "")

    @property
    def period(self) -> str:
        """期间标签原文（meta["period"]），如「2024年年度」「2024年前三季度」；缺失返回空串。

        参数：无（属性访问）。返回：str。副作用：无。
        """
        return self.meta.get("period", "")

    @property
    def year(self) -> Optional[int]:
        """年份（meta["year"] 转 int）；缺失或无法转换（TypeError / ValueError）时返回 None。

        参数：无（属性访问）。返回：Optional[int]。副作用：无——解析失败被吞掉并降级为 None。
        """
        raw = self.meta.get("year", "")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def metadata_dict(self) -> Dict[str, object]:
        """给检索结果用的扁平元数据。

        参数：无。
        返回：dict —— 固定 8 个键：doc_id / source_id / title / company / doc_type /
            period / year / path；chunking.split_document 会在此基础上补
            section_title 与 section_order，HybridRetriever 的元数据打分就读这些键
            （未做数值/枚举约束，year 可能是 None）。
        副作用：无（每次返回新字典）。
        """
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
    """解析 front-matter，返回 (元数据字典, 正文)。

    参数：text —— 文件全文。
    返回：(meta, body)。meta 为 Dict[str, str]：空行、以 # 开头的行、不含冒号的行
        一律跳过；键值用 partition(":") 切分后各自 strip，所以值里允许出现冒号。
        没有 front-matter 时返回 ({}, text) 原样放行。
    副作用：无（纯函数）。
    """
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
    """按 Markdown 标题把正文切成章节。

    参数：doc_id 文档编号（写进每个 Section）；body 已剥离 front-matter 的正文。
    返回：List[Section]，顺序即文中出现顺序。标题行本身不进正文；第一个标题之前的
        正文归入标题为「正文」的默认章节；只有标题而正文为空的分组不产出 Section。
    副作用：无（对输入只读）。
    """
    sections: List[Section] = []
    current_title = "正文"
    current_level = 1
    buffer: List[str] = []
    order = 0

    def flush() -> None:
        """把当前缓冲区落成一个 Section（空缓冲则跳过），并复位缓冲。

        参数：无（闭包，读取外层的 current_title / current_level / buffer）。
        返回：None。
        副作用：向 sections 追加元素；order 自增；buffer 清空（通过 nonlocal 写回）。
        """
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
            # 为什么先 flush 再更新标题：flush 用的是「上一个标题」的状态，
            # 顺序颠倒会把旧正文错误地挂到新标题下面。
            flush()
            current_level = len(m.group(1))
            current_title = m.group(2).strip()
            continue
        buffer.append(line)
    # 收尾：最后一节没有后继标题来触发 flush，必须在这里显式落一次。
    flush()
    return sections


def load_document(path: Path) -> Document:
    """加载单篇文档。

    参数：path —— 单个 .md 文件路径。
    返回：Document。source_id 取 front-matter 的 source_id；title 取正文第一个
        一级标题，找不到则退回文件名词干。
    副作用：读磁盘。异常：文件缺失抛 FileNotFoundError，非 UTF-8 抛 UnicodeDecodeError
        （不吞异常——静默少读资料比启动失败更危险）。
    """
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    # 为什么留 path.stem 兜底：没有 source_id 的文档也必须有个稳定编号，
    # 否则整条「引用 -> 子块 -> 父块 -> source_id」的溯源链会断。
    source_id = meta.get("source_id") or path.stem
    doc_id = source_id
    title = path.stem
    for line in body.splitlines():
        m = _HEADING_RE.match(line.strip())
        # 只认一级标题（#）：## 及以下属于章节，继续往下找，找不到就用文件名词干。
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
    """一个目录下全部文档的只读集合。

    关键属性：
        documents  —— List[Document]，按加载顺序（load_documents 里已排序）
        _by_source —— source_id -> Document 索引，get / require / source_ids 的基础
    状态流转：构造后只读，只提供查询与出处定位，不提供增删改。
    """

    def __init__(self, documents: Sequence[Document]) -> None:
        """建立集合与 source_id 索引。

        参数：documents 文档序列（会被复制成新列表）。
        返回：None。
        副作用：无——只读入参，不修改传入对象。
        注：source_id 重复时后出现的文档会覆盖索引项（当前 data/ 内编号唯一）。
        """
        self.documents: List[Document] = list(documents)
        self._by_source: Dict[str, Document] = {d.source_id: d for d in self.documents}

    def __len__(self) -> int:
        """文档数量。

        参数：无。返回：int。副作用：无。
        """
        return len(self.documents)

    def __iter__(self):
        """按加载顺序迭代 Document（支持 for doc in store）。

        参数：无。返回：Iterator[Document]。副作用：无。
        """
        return iter(self.documents)

    def get(self, source_id: str) -> Optional[Document]:
        """按资料编号取文档。

        参数：source_id 资料编号（如 EX-TECH-2024-AR）。
        返回：Document 或 None（不存在时不抛异常）。副作用：无。
        """
        return self._by_source.get(source_id)

    def require(self, source_id: str) -> Document:
        """按资料编号取文档，取不到就报错。

        参数：source_id 资料编号。
        返回：Document。
        副作用：无。异常：不存在时抛 KeyError（消息里带编号，便于定位引用错误）。
        """
        doc = self.get(source_id)
        if doc is None:
            raise KeyError(f"未找到资料编号：{source_id}")
        return doc

    @property
    def companies(self) -> List[str]:
        """全部公司全称，去重后排序（空公司名被忽略）。

        参数：无（属性访问）。返回：List[str]。副作用：无。
        """
        return sorted({d.company for d in self.documents if d.company})

    @property
    def source_ids(self) -> List[str]:
        """全部资料编号，排序后返回（也可用来做确定性遍历）。

        参数：无（属性访问）。返回：List[str]。副作用：无。
        """
        return sorted(self._by_source)

    def section_text(self, source_id: str, title_keyword: str) -> Optional[str]:
        """按章节标题关键词取原文，供事实抽取 / 引用定位使用。

        参数：source_id 资料编号；title_keyword 标题关键词（子串匹配）。
        返回：首个标题包含该关键词的章节正文；文档不存在或无匹配时返回 None。
        副作用：无。注意匹配是「包含」而非精确相等，关键词过短可能命中意外章节。
        """
        doc = self.get(source_id)
        if doc is None:
            return None
        for section in doc.sections:
            if title_keyword in section.title:
                return section.text
        return None

    def locate(self, source_id: str, needle: str) -> Optional[Section]:
        """定位包含某段文本的章节，用于让引用带上「出处章节」。

        参数：source_id 资料编号；needle 待定位的文本片段。
        返回：首个正文包含 needle 的 Section；文档不存在或 needle 为空/未命中时返回 None。
        副作用：无。按文档顺序返回第一个命中，不做多命中消歧。
        """
        doc = self.get(source_id)
        if doc is None:
            return None
        for section in doc.sections:
            if needle and needle in section.text:
                return section
        return None


def load_documents(data_dir: Optional[Path] = None) -> DocumentStore:
    """加载 data/ 下全部 .md 文档（按文件名排序，保证顺序确定）。

    参数：data_dir 可选目录；缺省用 config.DATA_DIR。
    返回：DocumentStore。
    副作用：读磁盘。异常：目录不存在时抛 FileNotFoundError；单篇解析失败会向上冒泡，
        不做跳过——宁可启动失败，也不要静默少读资料导致证据缺口。
    """
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"资料目录不存在：{directory}")
    paths = sorted(p for p in directory.glob("*.md") if p.is_file())
    return DocumentStore([load_document(p) for p in paths])
