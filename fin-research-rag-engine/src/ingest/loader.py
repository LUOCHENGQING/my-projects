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

多格式如何统一解析
------------------
四类来源最终都收敛到同一种结构（`SourceDocument` → `RawSection` → `Block`），
下游切分/索引/检索只需要认识这套结构：

| 来源后缀 | 典型来源 | 解析方式 | `fmt` |
| --- | --- | --- | --- |
| `.md` / `.markdown` | Word 转 Markdown、监管政策正文 | front-matter + 标题分章节 + 表格成块 | `markdown` |
| `.txt` | 扫描件 OCR 输出、系统导出制度 | front-matter + 标题分章节，缺 `doc_type` 时补 `OCR文本` | `text` |
| `.csv` | Excel 导出的产品要素表 / 指标横表 | 整表作为一章的一个表格块，支持前导 `# key: value` 注释头 | `csv` |
| `.json` | 产品库导出的结构化记录 | `records` 列表渲染成「一张表 + 每条记录一段文本」 | `json` |

注：实际实现**不直接解析 PDF / Word / Excel / 扫描图**——本模块只按 `path.suffix`
分派上述四种文本格式；扫描图需要先用 `ingest/ocr.py` 的 `extract_text()`（或人工校对的旁挂 `.txt`）得到文本，再作为 `.txt` 进入本模块。PDF/Word/Excel 同理需要先转文本。

脱敏与字段映射的口径
--------------------
- **字段映射**（`map_fields`）：以 `FIELD_ALIASES` 为唯一口径，`key.strip().lower()`
  查表命中则写入规范字段，未命中则保留原名并加 `x_` 前缀；空值与全空白值直接丢弃。
  `None` 被跳过，因此「没有这个字段」与「字段值为空」在结果里都表现为键不存在。
- **脱敏**（`apply_masking`）：委托 `..utils.text.mask_sensitive`，按身份证
  （`\\b\\d{17}[\\dXx]\\b`）、手机号、银行卡号、邮箱四类正则命中即替换为
  `[类型:****后四位]`。本模块**刻意不在解析时顺手脱敏**，而是留成显式一步：
  解析层如实反映原文，脱敏可单测、可开关、可在切分前统一执行。
- **OCR 与形近字口径**：OCR 只产出/旁挂 `.txt` 文本，本模块不做字形纠正；
  「己经→已经」这类形近字与数字上下文里的字母归正在 `ingest/cleaning.py` 完成。

六处「同接口换实现」降级开关在摄取层的体现
------------------------------------------
工程里共有六处可替换实现（向量库 Milvus↔内存、向量模型 BGE-M3↔确定性哈希、
重排 bge-reranker↔本地交叉编码器、缓存 Redis↔内存 LRU、OCR PaddleOCR↔旁挂文本、
生成 OpenAI↔确定性抽取），摄取层体现的是 **OCR 这一处**：`ingest/ocr.py` 用
`paddleocr` / `sidecar` / `null` 三种后端保证 `OCRResult` 结构一致，没装引擎时
退化为「旁挂校对文本」再退化为「如实报缺 + warnings」，因此本模块里不存在
「因为没装 OCR 所以跑不了」的分支。其余五处的切换点在各自模块，不在本文件。
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import DATA_DIR      # 默认资料目录（load_corpus 未显式传参时使用）
from ..utils.text import mask_sensitive, normalize_text

# 注：实际实现仅 apply_masking 用到 mask_sensitive，normalize_text 在本模块内未被调用
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
# 每个规范字段（左）对应一组可识别的外部写法（右，含中文别名与英文别名）。
# 这是字段映射的**唯一口径**：新增来源系统时在这里补别名，而不是在代码里加 if。
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

# 反查表：别名小写 → 规范字段。别名统一小写是为了让 `Org` / `ORG` / `org` 都能命中
_ALIAS_LOOKUP: Dict[str, str] = {
    alias.lower(): canonical for canonical, aliases in FIELD_ALIASES.items() for alias in aliases
}

# 规范字段的顺序，写进元数据时保持稳定，便于 diff 与测试断言
CANONICAL_FIELDS: Tuple[str, ...] = tuple(FIELD_ALIASES)

# front-matter：文件开头的 `--- ... ---` 块（用 DOTALL 让 `.` 跨行匹配）
_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# Markdown 标题：抓取 `#`~`######` 的级别与标题文本
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# 表格分隔行（如 `| --- | :--: |`）：只含空格、冒号、连字符与竖线
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|[\s:\-|]*$")


def map_fields(raw: Dict[str, object]) -> Dict[str, str]:
    """把任意外部字段名映射成规范字段名。

    参数：
        raw: 外部来源的原始键值对（键名可能是 `机构名称` / `institution_name` / `org`，
             值可能是任意类型，统一 `str()` 后 strip）。

    返回：
        新字典；命中 `FIELD_ALIASES` 的键改写为规范字段名，未命中的键改写为 `x_原键名`。

    副作用 / 异常：
        纯函数，不修改入参，不抛异常（除 `str(value)` 自身可能抛出的极端情况）。

    未识别的字段原样保留（前缀 `x_`），这样下游元数据过滤仍能用，
    同时不会污染规范字段空间。
    """
    out: Dict[str, str] = {}
    for key, value in raw.items():
        if value is None:          # None 视为「无此字段」，避免写出 "None" 这种脏值
            continue
        text = str(value).strip()
        if not text:               # 全空白同样丢弃，保持元数据稀疏干净
            continue
        canonical = _ALIAS_LOOKUP.get(str(key).strip().lower())
        if canonical:
            out[canonical] = text
        else:
            # setdefault 而非直接赋值：保留首次出现的写法，避免多个同义未知键互相覆盖
            out.setdefault(f"x_{str(key).strip()}", text)
    return out


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class Block:
    """章节内的一个内容块。

    职责：承载「段落」或「整张表格」两种最小内容单元，是切分阶段的输入。
    关键属性：
        kind  块类型，取值 `"paragraph"` | `"table"`（由解析器写入，不改动）
        text  段落原文；表格块则存该表的 Markdown 原文
        rows  表格的二维单元格列表，**第一行为表头**；段落块为空列表

    区分段落块与表格块是**为了切分服务**：表格必须按行组成组切，段落按句累积切，
    两者混在一起切必然把表头与数据行拆散。
    """

    kind: str                     # "paragraph" | "table"
    text: str = ""                # 段落原文 / 表格的 Markdown 原文
    rows: List[List[str]] = field(default_factory=list)  # 表格：第一行为表头

    @property
    def is_table(self) -> bool:
        """是否为表格块（下游据此决定按行成组切还是按句切）。"""
        return self.kind == "table"

    @property
    def header(self) -> List[str]:
        """表格表头行；非表格块或空表返回空列表。"""
        return self.rows[0] if self.rows else []


@dataclass
class RawSection:
    """文档中的一个章节（切分阶段会变成父块）。

    职责：保留「标题 + 层级 + 块列表」，让切分阶段能按章节组织父子块。
    关键属性：
        order   章节在文档中的出现序号（从 0 递增，由解析顺序决定）
        title   标题文本；无 Markdown 标题时解析器填默认值 `正文`
        level   标题层级 1~6（`#` 的个数）；无标题时默认 1
        blocks  该章节下的内容块列表
    """

    order: int
    title: str
    level: int
    blocks: List[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        """章节纯文本（表格渲染成 `列: 值` 行，保证表格内容也能被字面检索命中）。

        返回：块文本按顺序用换行拼接；表格块走 `render_table_rows()`，
        空段落块被跳过。
        """
        parts: List[str] = []
        for block in self.blocks:
            if block.is_table:
                parts.append(render_table_rows(block.rows))
            elif block.text.strip():
                parts.append(block.text.strip())
        return "\n".join(p for p in parts if p)

    @property
    def char_count(self) -> int:
        """章节纯文本字符数（`text` 的长度，供质量评分与切分统计使用）。"""
        return len(self.text)


def render_table_rows(rows: Sequence[Sequence[str]]) -> str:
    """把表格行渲染成「表头: 值」的文本，让 BM25 也能命中表格里的单元格。

    参数：
        rows: 二维单元格序列，第一行视为表头；短行按两者较短长度对齐，
              缺列不报错（`min(len(header), len(cells))`）。

    返回：
        每条数据行渲染为 `表头: 值；表头: 值。`，再用换行连接成字符串；
        `rows` 为空、或所有数据行都无有效单元格时返回空串。

    副作用 / 异常：
        无副作用；对非字符串单元格做 `str()` 转换，不抛异常。
    """
    if not rows:
        return ""
    header = [str(h).strip() for h in rows[0]]
    lines: List[str] = []
    for row in rows[1:]:
        cells = [str(c).strip() for c in row]
        pairs = [
            f"{header[i]}: {cells[i]}"
            for i in range(min(len(header), len(cells)))
            if cells[i]                   # 跳过空单元格，避免渲染出「列: 」噪音
        ]
        if pairs:
            lines.append("；".join(pairs) + "。")
    return "\n".join(lines)


@dataclass
class SourceDocument:
    """一篇解析完成的文档。

    职责：解析层的最终产物，也是切分层的输入单位。
    关键属性：
        doc_id / source_id  文档标识（解析时取 front-matter 的 source_id，缺失则用文件名）
        title / path / fmt  标题、原始路径、来源格式（`markdown` / `text` / `csv` / `json`）
        meta                已做字段映射的元数据（规范字段 + `x_` 前缀的未知字段）
        sections            章节列表（章节 → 块）
        raw                 原始全文（脱敏与清洗会就地改写它）
        issues              解析/清洗阶段发现的问题，进轨迹便于治理
    """

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
        """资料类型（如 `监管政策` / `产品要素表`）；缺失返回空串。"""
        return self.meta.get("doc_type", "")

    @property
    def institution(self) -> str:
        """机构名称；缺失返回空串。"""
        return self.meta.get("institution", "")

    @property
    def product(self) -> str:
        """产品名称；缺失返回空串。"""
        return self.meta.get("product", "")

    @property
    def effective_date(self) -> str:
        """生效/发布日期原文；缺失返回空串。"""
        return self.meta.get("effective_date", "")

    @property
    def year(self) -> Optional[int]:
        """从生效日期里抠出年份（用于「只看 2024 年之后」这类过滤）。

        返回：`effective_date` 中第一个四位年份（`19xx` / `20xx`）的 int；
        找不到年份时返回 `None`。
        """
        raw = self.meta.get("effective_date", "")
        m = re.search(r"(19|20)\d{2}", raw)
        return int(m.group(0)) if m else None

    @property
    def version(self) -> str:
        """版本号；缺失返回空串。"""
        return self.meta.get("version", "")

    def metadata_dict(self) -> Dict[str, object]:
        """给检索过滤用的扁平元数据。

        返回：把文档级标识、类型、机构、产品、行业、日期与年份（`year` 为 int 或 None）
        以及密级、地区、路径拍平成一个字典，供索引层写入 payload。
        副作用：无。
        """
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
        """全文表格块总数（`stats()` 与质量体检的统计口径）。"""
        return sum(1 for s in self.sections for b in s.blocks if b.is_table)


@dataclass
class FAQEntry:
    """FAQ 条目：高频问题直出，不走完整链路。

    职责：FAQ 走「近似匹配 → 命中直接返回」的快车道，不需要切分、索引与生成。
    关键属性：
        faq_id      条目标识；JSON 里缺失时由 `_load_faq()` 生成 `FAQ-001` 形式
        question    问题原文（空则整条被丢弃）
        answer      答案原文（空则整条被丢弃）
        category    分类（可空）
        updated_at  最近更新时间（字符串，原样保留不做日期解析）
        source_id   关联的资料编号（可空）
    """

    faq_id: str
    question: str
    answer: str
    category: str = ""
    updated_at: str = ""
    source_id: str = ""

    def to_dict(self) -> Dict[str, str]:
        """拍平成字典输出（供 API / 轨迹展示）；字段名与入参保持一致，不做改名。"""
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
    """一个资料库目录解析后的全部内容。

    职责：资料库级容器，聚合文档与 FAQ，并提供按编号取用与统计的便捷入口。
    关键属性：
        documents  `SourceDocument` 列表（含解析失败的占位文档，其 `issues` 记录了原因）
        faq        `FAQEntry` 列表（来自含 `faq` 键的 JSON 文件，与 documents 分流）
    """

    documents: List[SourceDocument] = field(default_factory=list)
    faq: List[FAQEntry] = field(default_factory=list)

    def __len__(self) -> int:
        """文档篇数（**不含** FAQ 条目数）。"""
        return len(self.documents)

    def __iter__(self):
        """按顺序迭代文档，便于 `for doc in corpus` 直接使用。"""
        return iter(self.documents)

    def get(self, source_id: str) -> Optional[SourceDocument]:
        """按 `source_id` 线性查找文档；找不到返回 `None`（不抛异常）。"""
        for doc in self.documents:
            if doc.source_id == source_id:
                return doc
        return None

    def require(self, source_id: str) -> SourceDocument:
        """按 `source_id` 取文档，缺失即失败。

        异常：找不到时抛 `KeyError`（消息形如 `未找到资料编号：XXX`）。
        """
        doc = self.get(source_id)
        if doc is None:
            raise KeyError(f"未找到资料编号：{source_id}")
        return doc

    @property
    def source_ids(self) -> List[str]:
        """全部资料编号，已排序（顺序确定，便于 diff 与测试断言）。"""
        return sorted(d.source_id for d in self.documents)

    @property
    def doc_types(self) -> List[str]:
        """出现过的资料类型去重后排序（空类型不计入）。"""
        return sorted({d.doc_type for d in self.documents if d.doc_type})

    @property
    def institutions(self) -> List[str]:
        """出现过的机构去重后排序（空值不计入，用于主体闸门）。"""
        return sorted({d.institution for d in self.documents if d.institution})

    @property
    def issues(self) -> List[str]:
        """把每篇文档的 issues 加上编号前缀展开成一维列表，便于治理汇总。"""
        return [f"{d.source_id}: {msg}" for d in self.documents for msg in d.issues]

    def stats(self) -> Dict[str, object]:
        """资料库概览统计：文档数、章节数、表格数、FAQ 数、类型与格式清单。

        返回：全为纯 Python 值（列表已排序），可直接序列化。
        副作用：无。
        """
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
    """解析 front-matter，返回 (元数据字典, 正文)。刻意不引入 PyYAML。

    参数：
        text: 完整文件文本。

    返回：
        `(meta, body)`。命中开头的 `--- ... ---` 块时，`meta` 为按行 `key: value`
        拆出的字典（跳过空行、以 `#` 开头的注释行、以及不含 `:` 的行；键值均 strip），
        `body` 为分隔行之后的正文；未命中时返回 `({}, text)`（正文原样返回）。

    副作用 / 异常：无。
    """
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    meta: Dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        # partition 而非 split：值里再出现冒号（如时间 12:00）时不会被截断
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, text[match.end():]


def _split_table_row(line: str) -> List[str]:
    """把一行 Markdown 表格文本拆成单元格列表（去掉首尾竖线并逐个 strip）。

    参数：line — 形如 `| A | B |` 的单行文本。
    返回：单元格字符串列表；列数由竖线个数决定，不校验列数是否齐整。
    """
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def parse_markdown_table(lines: Sequence[str]) -> List[List[str]]:
    """把连续的表格行解析成 rows（含表头）。首行是表头，第二行是分隔行。

    参数：lines — 连续的表格行文本（调用方已保证每行以 `|` 开头）。
    返回：二维单元格列表，**不含分隔行**；整行全空的行被跳过。
    副作用 / 异常：无；分隔行若不在第 2 行则不会被识别（实际实现只判 `idx == 1`）。
    """
    rows: List[List[str]] = []
    for idx, line in enumerate(lines):
        if idx == 1 and _TABLE_SEP_RE.match(line):
            continue
        cells = _split_table_row(line)
        if any(c for c in cells):
            rows.append(cells)
    return rows


def parse_markdown_sections(body: str) -> List[RawSection]:
    """按 Markdown 标题切章节，章节内部再区分段落块与表格块。

    参数：
        body: 去掉 front-matter 之后的正文（调用方负责剥离）。

    返回：
        章节列表，`order` 从 0 递增；**只有含非空块的章节才会入列**。
        首个标题之前的内容归入默认章节（`title="正文"`、`level=1`）；
        标题文本取自 `#`~`######` 的分组 2，层级取 `#` 的个数。

    副作用 / 异常：无。标题行本身不进入块文本（只作章节标题）。
    """
    sections: List[RawSection] = []
    title, level, order = "正文", 1, 0
    buffer: List[str] = []

    def flush() -> None:
        """把当前缓冲区结算成一个章节：有块才追加，并推进 order。"""
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
    """把一串原始行切成段落块 / 表格块。

    参数：lines — 单个章节内的原始行。
    返回：`Block` 列表，工具类块与空块都不产生输出。

    归块规则（实际实现）：`strip()` 后以 `|` 开头且行内含 `|` 的行视为表格行，
    连续表格行合成一个表格块；其余非空行累积成段落块；空行会同时切断段落与表格。
    """
    blocks: List[Block] = []
    para: List[str] = []
    table: List[str] = []

    def flush_para() -> None:
        """结算当前段落缓冲；全空白则不产出块。"""
        nonlocal para
        text = "\n".join(para).strip()
        if text:
            blocks.append(Block(kind="paragraph", text=text))
        para = []

    def flush_table() -> None:
        """结算当前表格缓冲；解析不到任何行则不产出块（避免生成空表）。"""
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
    """加载 `.md` / `.markdown` 文档。

    参数：path — 文件路径（读取时强制 UTF-8，不做编码探测）。
    返回：`fmt="markdown"` 的 `SourceDocument`；元数据来自 front-matter 且已做字段映射。

    标题口径（实际实现）：front-matter 的 `title` 优先；否则取正文中**第一个一级标题**
    （`#`）；再否则用文件名（`path.stem`）。
    副作用：读取文件；读取失败（不存在 / 编码错误）时异常向上抛出，由 `load_corpus` 兜住。
    """
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    meta = map_fields(meta)
    source_id = meta.get("source_id") or path.stem   # 无编号时退化为文件名，保证 doc_id 不空
    title = meta.get("title") or path.stem
    for line in body.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m and len(m.group(1)) == 1:               # 只认一级标题，二级标题不当文档名
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
    """加载 `.txt` 文档（纯文本；典型来源：扫描件 OCR 输出、系统导出的制度 txt）。

    参数：path — 文件路径，UTF-8 读取。
    返回：`fmt="text"` 的 `SourceDocument`；元数据同样读 front-matter 并做字段映射。

    与 Markdown 的差异（实际实现）：标题**不**从正文一级标题推断（纯文本无标题语法），
    且缺 `doc_type` 时补默认值 `OCR文本`——该默认值会进入质量评分，使 `.txt` 文档
    被按 OCR 来源轻微降权。
    副作用 / 异常：同 `_document_from_markdown`。
    """
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
    """CSV（典型来源：产品要素表 / 指标横表）。整表作为一章的表格块。

    参数：path — CSV 文件路径。
    返回：`fmt="csv"` 的 `SourceDocument`，单章节单表格块（`order=0`、`level=1`）。

    口径说明：
        用 `utf-8-sig` 打开，顺带吃掉 Excel 导出常见的 BOM；
        首行即表头，**不做类型推断**，所有单元格 `strip()` 后原样保留为字符串；
        开头连续的 `# key: value` 行被抽出作为元数据（不留在表里），缺 `doc_type` 时补
        `产品要素表`。
    副作用：读取文件。
    异常：文件为空（或全是空行）时抛 `ValueError`。
    """
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
        rows.pop(0)                                   # 注释头消费掉，避免混进表格数据
    meta = map_fields(meta_raw)
    meta.setdefault("doc_type", "产品要素表")
    source_id = meta.get("source_id") or source_id    # 注释头里的编号优先于文件名
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

    参数：
        path    文件路径（仅用于取文件名兜底与写入 `path` 字段）
        payload 已解析的 JSON 顶层字典，必须含非空 `records` 列表

    返回：
        `fmt="json"` 的 `SourceDocument`，单章节内先放一张表块（表头为各记录键的并集，
        键顺序按首次出现确定），再为每条记录追加一段 `k: v；k: v。` 的段落块。

    副作用：无（不读文件）。
    异常：`records` 缺失、不是列表或为空时抛 `ValueError`；
          顶层非字典的入参在 `load_source_document` 阶段就被拦下，不会走到这里。

    把记录列表渲染成一张表 + 每条记录一段可读文本：表供结构化过滤，
    文本供语义检索。二者并存，避免「结构化字段检索不到、文本字段过滤不了」。
    """
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"JSON 缺少 records 列表：{path}")
    # 标量元数据才纳入映射：list / dict 类型的顶层键（如 records 本身）不是元数据
    meta = map_fields(
        {k: v for k, v in payload.items() if k != "records" and not isinstance(v, (list, dict))}
    )
    meta.setdefault("doc_type", "结构化记录")
    source_id = meta.get("source_id") or path.stem
    title = meta.get("title") or path.stem

    # 表头 = 各记录键的并集，按首次出现顺序排列（保证同一份文件解析结果稳定）
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
    """按后缀加载单篇文档；FAQ 文件返回 None（由 load_corpus 单独处理）。

    参数：path — 待加载的文件路径（只用 `path.suffix.lower()` 分派，不探测文件内容）。

    返回：
        - `.md` / `.markdown` / `.txt` / `.csv` → 对应 `SourceDocument`
        - 含 `faq` 键的 `.json` → `None`（FAQ 由 `_load_faq()` 单独解析）
        - `.json` 含 `records` → JSON 记录文档
        - 其他后缀，或 JSON 结构既不认识 `faq` 也不认识 `records` → `None`

    副作用：读取（CSV/JSON 还会解析）文件。
    异常：文件不存在、编码错误、内容不合法（空 CSV、缺 `records`）时异常向上抛，
          由 `load_corpus()` 捕获并转成「解析失败」占位文档。
    """
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
            return None                       # 由 load_corpus 分流到 _load_faq
        if isinstance(payload, dict) and "records" in payload:
            return _document_from_json_records(path, payload)
        return None
    return None


def _load_faq(path: Path) -> List[FAQEntry]:
    """从 JSON 文件解析 FAQ 条目。

    参数：path — 含顶层 `faq` 列表的 JSON 文件。
    返回：`FAQEntry` 列表。

    过滤与兜底口径（实际实现）：
        - 顶层不是字典、或 `faq` 不是列表时返回空列表（不抛异常）；
        - 非字典条目、以及 question 或 answer 为空白的条目被跳过；
        - `faq_id` 缺失时按当前枚举序号生成 `FAQ-001` 形式（序号基于原始列表下标，
          因此前面被跳过的条目会留下编号空档）。
    副作用：读取并解析 JSON 文件；文件损坏时 `json.loads` 的异常向上抛。
    """
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
        if not question or not answer:        # 缺问题或缺答案的条目直接丢弃，不猜内容
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
    """加载资料库目录下全部支持的文件（按文件名排序，保证顺序确定）。

    参数：data_dir — 资料目录；省略时用 `..config.DATA_DIR`。

    返回：`Corpus`（`documents` + `faq`）；未识别的后缀被静默跳过。

    副作用 / 异常：
        - 只扫**顶层文件**，不递归子目录（`directory.iterdir()` + `p.is_file()`）；
        - 目录不存在时抛 `FileNotFoundError`；
        - 单篇文档解析失败**不会**中断整库加载，而是生成一篇字段取自文件名的占位
          文档、把原因写进 `issues`（形如 `解析失败：ValueError: CSV 为空：…`）；
        - 注：实际实现中 `.json` 的 FAQ 分流判断位于 `try` **之外**，这一步的
          `json.loads` / 读文件失败不会被兜住，会直接让整个 `load_corpus()` 抛出。
    """
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
                corpus.faq.extend(_load_faq(path))   # FAQ 与文档分流，不进 documents
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

    参数：documents — 可迭代的 `SourceDocument`（通常是 `corpus.documents`）。

    返回：**正文 `raw` 实际发生变化**的文档数（不是被替换的敏感串条数）。
          注：实际实现只在 `raw` 被判出敏感内容时，才顺带处理该文档的块文本与表格单元格；
          若某文档的敏感串只出现在块里而 `raw` 中未被匹配到，则该文档既不会被改写、
          也不会被计入返回值。

    副作用（就地修改，无返回值携带改写结果）：
        - 改写 `doc.raw`（原文被替换为掩码文本，原始敏感串不再保留）；
        - 改写各 `Block.text` 与 `Block.rows` 中的单元格；
        - 不改动 `doc.sections` 结构与 `doc.meta`，也不写 `doc.issues`。

    掩码口径由 `..utils.text.mask_sensitive` 决定：身份证 / 手机号 / 银行卡号 / 邮箱
    命中即替换为 `[类型:****后四位]`。

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
