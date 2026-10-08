"""解析清洗子包。

    loader      多源文档解析（md / txt(OCR) / csv / json）+ 字段映射 + 表格结构化
    cleaning    文本清洗（形近字、模板行、水印）+ 文档质量评分
    ocr         OCR 可插拔后端（paddleocr / sidecar / null）

本层是整个链路的第一道工序，也是「脏数据能不能救回来」的分水岭：
解析错了，后面切得再漂亮、召回再准，答案也一定是错的。

在 RAG 全链路中的位置
---------------------
    资料目录 → 【ingest：解析 → 脱敏 → 清洗 → 质量评分】
             → chunking（父块/子块切分） → index（向量化 + 倒排）
             → retrieve（召回 + 重排） → generate（作答 + 引用）

本包只负责「把磁盘上的字节变成结构化的、可信的文本」，不做切分与索引。

职责
----
1. 格式适配：把 .md / .markdown、.txt、.csv、.json 解析成同一种结构
   （`SourceDocument` → `RawSection` → `Block`），下游不必关心来源格式。
   注：实际实现不直接解析 PDF / Word / Excel / 扫描图——这几类需要先转成上述
   文本格式（或用 ocr 子模块产出 `.txt`）再进入本层，`loader` 只用 `path.suffix`
   做分派。
2. 字段映射：外部系统的五花八门字段名经 `map_fields()` 收敛到规范字段
   （`FIELD_ALIASES`），未识别字段保留为 `x_` 前缀，映射口径显式配置而非靠模型猜。
3. 脱敏：`apply_masking()` 把身份证 / 手机号 / 银行卡号 / 邮箱替换为
   `[ID:****后四位]` 形式的掩码（正则口径在 `..utils.text.mask_sensitive`），
   在切分之前执行，因此索引、日志、轨迹里都不会出现原始敏感串。
4. 清洗与打分：`cleaning` 做 OCR 形近字纠正、模板行与水印剔除，并给出 0~1 的
   文档质量分；分数会随子块进入检索层参与降权。
5. OCR 兜底：`ocr` 提供 paddleocr / sidecar / null 三种同接口后端，
   即「六处同接口换实现」降级开关中**摄取层所体现的那一处**：没装 paddleocr 时
   自动退化为旁挂校对文本，再退化为如实的空结果 + warnings，绝无「跑不了的分支」。

对外关键对象
------------
    load_corpus(data_dir)        加载整个资料目录 → Corpus（文档 + FAQ）
    load_source_document(path)   加载单篇文档 → SourceDocument | None
    apply_masking(documents)     就地脱敏 → 被改动的文档数
    clean_corpus(documents)      就地清洗并打分 → List[QualityReport]
    extract_text(path, prefer)   OCR 抽取 → OCRResult（三种后端结构一致）

输入 / 输出
-----------
    输入：资料目录下的 .md / .txt / .csv / .json 文件（只扫顶层，不递归子目录）
    输出：`Corpus`（`SourceDocument` 列表 + `FAQEntry` 列表）与 `QualityReport` 列表

调用方
------
    `src/engine.py`：`load_corpus()` → `apply_masking()` → `clean_corpus()` 三步固定顺序；
    `src/faq.py`、`src/chunking/*`、`src/retrieve/*` 复用这里的数据结构；
    `tests/test_ingest.py` 与 `tests/conftest.py` 直接在此层做回归断言。
"""

from __future__ import annotations

# 包级再导出：下游统一从 `src.ingest` 取公开对象，避免各自 import 内部模块路径
from .cleaning import OCR_CONFUSIONS, QualityReport, clean_corpus, clean_document, clean_text, score_quality, strip_boilerplate
from .loader import (
    Block,
    Corpus,
    FAQEntry,
    RawSection,
    SourceDocument,
    apply_masking,
    load_corpus,
    load_source_document,
    map_fields,
    parse_front_matter,
    parse_markdown_sections,
    parse_markdown_table,
    render_table_rows,
)
from .ocr import OCREngine, OCRResult, extract_text, get_engine

# 显式声明公开面；`loader` 内部会把 apply_masking / render_table_rows 追加进它自己的 __all__
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
    "render_table_rows",
    "map_fields",
    "apply_masking",
    "clean_text",
    "clean_document",
    "clean_corpus",
    "strip_boilerplate",
    "score_quality",
    "QualityReport",
    "OCR_CONFUSIONS",
    "OCREngine",
    "OCRResult",
    "get_engine",
    "extract_text",
]
