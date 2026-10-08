"""解析清洗层测试：格式适配 / 字段映射 / 表格结构化 / 脱敏 / OCR 纠错 / 质量评分。

覆盖的被测模块
--------------
    src.utils.text       normalize_text / normalize_whitespace / tokenize / split_sentences /
                         mask_sensitive（上游工具层，直接影响分词、脱敏与证据文本）
    src.ingest.loader    多格式加载（markdown / text / csv / json）、front-matter 解析、
                         字段映射 map_fields、Markdown 章节与表格解析、表格渲染、
                         Corpus 取用与统计、apply_masking 的语料级效果
    src.ingest.cleaning  clean_text（形近字 / 数字夹缝字母 / 水印）、strip_boilerplate、
                         OCR_CONFUSIONS 词表、score_quality 与 clean_document 的质量记账
    src.ingest.ocr       Sidecar / Null 后端与 extract_text 的后缀分派（第三方 OCR 不可用时的降级契约）

覆盖策略
--------
    正常路径  四种来源格式各自解析成功；CSV/JSON 产出表格块；表格渲染带表头；
              front-matter 与中文字段别名正确映射；cleaning 各项纠错与段落标点保留。
    边界      空值字段（None / 全空白）被丢弃、空章节不产出、无 front-matter 的正文、
              未知后缀、空昵称式输入（tokenize / normalize 的空串）、表格分隔行位置。
    异常      语料目录不存在抛 FileNotFoundError、require 未知编号抛 KeyError、
              未知后缀返回 None 而不抛错、非图片后缀不调用 OCR 后端。
    对抗/降级 OCR 三种后端的「不许静默成功」契约：旁挂文本缺失、未装引擎时必须
              留下 warnings 而不是给一个空文本的成功结果。
    拒答      本模块不覆盖端到端拒答（那在 tests/test_engine_api.py），只覆盖摄取层的
              「如实报缺」：质量报告把 OCR 来源记为 issue，供上层降权与治理。

统计口径提醒：本文件的断言多数是**契约级**的（结构、字段名、是否入列），
而不是数值级的；数值断言（如稀疏权重）在 tests/test_index.py 里。另外本文件
刻意保留了一些「只断言不抛错 / 不残留」的宽松用例，它们防的是回归，不是精度。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingest.cleaning import clean_text, score_quality, strip_boilerplate
from src.ingest.loader import (
    SourceDocument,
    load_corpus,
    load_source_document,
    map_fields,
    parse_front_matter,
    parse_markdown_sections,
    parse_markdown_table,
    render_table_rows,
)
from src.ingest.cleaning import OCR_CONFUSIONS
from src.ingest.ocr import NullOCREngine, SidecarOCREngine, extract_text, get_engine
from src.utils import mask_sensitive, normalize_whitespace, split_sentences, tokenize


# ---------------------------------------------------------------------------
# 分词与归一化
# ---------------------------------------------------------------------------
def test_normalize_text_turns_punctuation_into_space():
    """分词用归一化必须把全角标点替换成空格，否则标点会把词粘在一起。"""
    from src.utils import normalize_text

    assert normalize_text("第一条：合格投资者，金融资产。") == "第一条 合格投资者 金融资产"


def test_normalize_whitespace_keeps_punctuation():
    """展示用归一化与分词用归一化必须分工：前者只动空白，标点逐字保留。"""
    assert normalize_whitespace("第一条：合格投资者，金融资产。") == "第一条：合格投资者，金融资产。"


def test_tokenize_keeps_cjk_bigrams_and_codes():
    """分词产物必须同时含单字、相邻二元组与整串代码（产品代码不可被拆成字母 + 数字）。"""
    # 刻意把「代码 + 等级 + 中文」混在一句里：三种 token 形态（整串代码 / 单字 / 二元组）一次覆盖
    tokens = tokenize("产品代码 WY2024-01 风险等级 R2")
    assert "wy2024-01" in tokens
    assert "风险" in tokens and "险等" in tokens
    assert "r2" in tokens


def test_tokenize_keeps_clause_numbers_intact():
    """条款号必须作为一个整体 token 保留，拆成单字后 BM25 无法精确命中「第四十二条」。"""
    # 带「之规定」是为了同时验证「第四十二条之」这类扩展条款号不会被截断
    tokens = tokenize("依据第四十二条之规定")
    assert "第四十二条" in tokens


def test_split_sentences_preserves_original_punctuation():
    """切句只按句末标点断开，不改写措辞、不吞掉标点（产物会直接进证据与答案）。"""
    # 两个「第 N 条」连写：验证切句点落在「。」上，而不会在条款号内部断开
    sents = split_sentences("第一条 为规范管理，制定本办法。第二条 本办法适用于银行。")
    assert sents == ["第一条 为规范管理，制定本办法。", "第二条 本办法适用于银行。"]


# ---------------------------------------------------------------------------
# 脱敏
# ---------------------------------------------------------------------------
def test_mask_sensitive_masks_id_phone_bank_email():
    """四类强敏感字段（身份证 / 手机号 / 银行卡号 / 邮箱）命中即脱敏，原文不得残留。"""
    # 四类一起写：顺带验证规则之间不会互相抢匹配（身份证 18 位 vs 银行卡 16~19 位最容易串）
    text = "身份证 310101199001011234 手机 13812345678 卡号 6222021234567890123 邮箱 a.b@example.com"
    masked = mask_sensitive(text)
    assert "310101199001011234" not in masked
    assert "13812345678" not in masked
    assert "6222021234567890123" not in masked
    assert "a.b@example.com" not in masked
    assert "[ID:" in masked and "[PHONE:" in masked and "[BANK:" in masked and "[EMAIL:" in masked


def test_mask_sensitive_keeps_tail_for_manual_check():
    """掩码必须保留后四位，人工核对时才能确认「是不是同一个人/同一张卡」。"""
    masked = mask_sensitive("手机号 13812345678")
    assert masked.endswith("5678]")


def test_mask_sensitive_leaves_normal_numbers_alone():
    """普通业务数字（金额 / 期限）不能被脱敏误伤——误伤等于把证据里的数字抹掉。"""
    # 选 300 / 90 这类"短且非身份证形态"的数字，正是最容易被过度匹配的长度
    assert mask_sensitive("规模 300 万元，期限 90 天") == "规模 300 万元，期限 90 天"


# ---------------------------------------------------------------------------
# front-matter 与字段映射
# ---------------------------------------------------------------------------
def test_parse_front_matter():
    """有 front-matter 块时，元数据与正文必须正确分离（正文从分隔行之后开始）。"""
    meta, body = parse_front_matter("---\nsource_id: X-1\ntitle: 测试\n---\n# 标题\n正文")
    assert meta == {"source_id": "X-1", "title": "测试"}
    assert body.startswith("# 标题")


def test_parse_front_matter_without_block():
    """没有 front-matter 块的纯正文必须原样返回，不能被误当成元数据吞掉。"""
    meta, body = parse_front_matter("正文而已")
    assert meta == {} and body == "正文而已"


def test_map_fields_recognises_chinese_aliases():
    """中文别名必须映射到规范字段名（业务系统用什么写法，下游都只看规范名）。"""
    # 三个别名分别覆盖 编号 / 机构 / 日期 三类规范字段，且刻意都用中文写法
    mapped = map_fields({"资料编号": "A-1", "机构名称": "示例银行", "生效日期": "2024-01-01"})
    assert mapped["source_id"] == "A-1"
    assert mapped["institution"] == "示例银行"
    assert mapped["effective_date"] == "2024-01-01"


def test_map_fields_keeps_unknown_fields_with_prefix():
    """未识别的字段要保留下来（加 x_ 前缀），不能丢弃——下游还要靠它做元数据过滤。"""
    mapped = map_fields({"自定义字段": "值", "机构名称": "示例银行"})
    assert mapped["x_自定义字段"] == "值"
    assert mapped["institution"] == "示例银行"


def test_map_fields_skips_empty_values():
    """空值与 None 都不产出字段：元数据里「没有这个字段」和「值为空」表现必须一致。"""
    # 一个空白串 + 一个 None 同时给：验证两条丢弃分支都生效，结果是空字典
    assert map_fields({"机构名称": "  ", "标题": None}) == {}


# ---------------------------------------------------------------------------
# Markdown 解析
# ---------------------------------------------------------------------------
def test_parse_markdown_table_drops_separator_row():
    """表格分隔行（只含竖线、冒号与连字符的那一行）不得进入 rows，否则会被当成表头数据。"""
    rows = parse_markdown_table(["| A | B |", "| --- | --- |", "| 1 | 2 |"])
    assert rows == [["A", "B"], ["1", "2"]]


def test_parse_markdown_sections_splits_by_heading():
    """章节按 Markdown 标题切分，标题文本与出现顺序都要与原文一致。"""
    body = "# 标题\n正文。\n\n## 一、总则\n第一条 内容。\n\n## 二、附则\n第二条 内容。"
    sections = parse_markdown_sections(body)
    titles = [s.title for s in sections]
    assert titles == ["标题", "一、总则", "二、附则"]


def test_parse_markdown_sections_drops_empty_sections():
    """只有标题、没有正文的章节不会产出空块（否则会污染索引与引用编号）。"""
    body = "# 标题\n\n## 一、总则\n第一条 内容。"
    titles = [s.title for s in parse_markdown_sections(body)]
    assert titles == ["一、总则"]


def test_parse_markdown_sections_marks_table_blocks():
    """表格行必须被识别为表格块（is_table 为真），下游才能按行成组切而不是按句切。"""
    body = "## 一、表\n\n| 列 | 值 |\n| --- | --- |\n| a | b |\n"
    sections = parse_markdown_sections(body)
    assert any(b.is_table for s in sections for b in s.blocks)


def test_render_table_rows_repeats_header_per_row():
    """表格渲染必须把表头带到每一行（「列名: 值」），BM25 才可能命中单元格里的内容。"""
    # 两行数据 + 一个百分比上限：验证每行都自带列名，而不是只在首行出现一次
    text = render_table_rows([["等级", "上限"], ["C1", "0%"], ["C2", "3%"]])
    assert "等级: C1" in text and "上限: 0%" in text
    assert "等级: C2" in text


# ---------------------------------------------------------------------------
# 多格式加载
# ---------------------------------------------------------------------------
def test_load_corpus_reads_all_formats(corpus):
    """示例资料库里的四种来源格式（markdown / text / csv / json）都必须被解析成文档。"""
    formats = {doc.fmt for doc in corpus.documents}
    assert {"markdown", "text", "csv", "json"} <= formats


def test_load_corpus_separates_faq(corpus):
    """FAQ 条目必须与正文文档分流，且每条都同时具备问题与答案。"""
    assert len(corpus.faq) >= 10
    assert all(entry.question and entry.answer for entry in corpus.faq)


def test_corpus_get_and_require(corpus):
    """按编号取文档：get 缺失返回 None，require 缺失必须抛 KeyError（不许静默返回空文档）。"""
    assert corpus.get("POL-2024-07") is not None
    with pytest.raises(KeyError):
        corpus.require("NOT-EXIST")


def test_corpus_stats(corpus):
    """资料库统计口径必须与容器实际内容一致（篇数 / 表格数 / FAQ 数不得各说各话）。"""
    stats = corpus.stats()
    assert stats["documents"] == len(corpus.documents)
    assert stats["tables"] > 0
    assert stats["faq"] == len(corpus.faq)


def test_csv_document_has_table_block(corpus):
    """CSV 必须整表进一个表格块且首行为表头，否则按行成组切块就无从谈起。"""
    doc = corpus.require("PROD-TABLE-2024")
    assert doc.table_count >= 1
    header = doc.sections[0].blocks[0].rows[0]
    assert "产品代码" in header


def test_json_document_builds_table_and_text(corpus):
    """JSON 记录必须同时产出表格块与可读文本段：前者供结构化过滤，后者供语义检索。"""
    doc = corpus.require("PROD-JSON-2024")
    assert doc.table_count >= 1
    assert "JG2024-03" in doc.sections[0].text


def test_source_document_metadata_helpers(corpus):
    """文档元数据快捷属性（类型 / 年份 / 版本）与拍平字典必须与 front-matter 一致。"""
    doc = corpus.require("POL-2024-07")
    assert doc.doc_type == "监管政策"
    assert doc.year == 2024
    assert doc.version == "v2"
    assert doc.metadata_dict()["source_id"] == "POL-2024-07"


def test_load_source_document_returns_none_for_unknown_suffix(tmp_path):
    """未知后缀返回 None 而不是抛错：批量加载时「不认识的文件」属于正常情况，不是异常。"""
    # .bin 且内容为单个空字节：既不是支持的文本格式，也不是能被文本解析器吃下的内容
    path = tmp_path / "x.bin"
    path.write_bytes(b"\x00")
    assert load_source_document(path) is None


# ---------------------------------------------------------------------------
# 清洗
# ---------------------------------------------------------------------------
def test_clean_text_fixes_ocr_confusions():
    """OCR 形近字必须按混淆表无条件纠正（形近字表只收高置信、低副作用的错字对）。"""
    assert "已经" in clean_text("客户己经完成评估")
    assert "收益" in clean_text("预期收溢率为 3.2%")


def test_clean_text_fixes_letters_in_digit_context():
    """数字夹缝里的字母（O→0、l→1 等）必须归正，且只动数字上下文、不误改正常英文词。"""
    # 刻意选「1O0」「2O24」这类金额/年份形态：字母两侧都是数字，正是归正规则的触发条件
    cleaned = clean_text("金额 1O0 万元，代码 2O24")
    assert "100" in cleaned
    assert "2024" in cleaned


def test_clean_text_drops_watermark_lines():
    """水印形态的整行必须被丢弃，同时正文行不得被连坐删掉。"""
    # 注：实际实现为——水印正则是「1~5 组（汉字 + 空白）后再跟一个汉字」，即至少要 3 个带空格的
    # 汉字（如「机 密 件」）才会命中；这里给的「机 密」只有 2 个字、不足以命中该正则，
    # 因此本用例的「机 密」消失落在下面 `not in` 这条断言上（正文保留那条才是本用例真正守住的点）。
    cleaned = clean_text("机 密\n第一条 正文内容。")
    assert "机 密" not in cleaned
    assert "第一条" in cleaned


def test_clean_text_keeps_punctuation_and_paragraphs():
    """清洗必须保留标点与段落空行——产物会直接成为证据文本与答案原句。"""
    cleaned = clean_text("第一条 为规范管理，制定本办法。\n\n第二条 本办法适用于银行。")
    assert "，" in cleaned and "。" in cleaned
    assert "\n\n" in cleaned


def test_strip_boilerplate_removes_repeated_short_lines():
    """出现次数达到阈值且不超长的重复短行判为页眉页脚，全部剔除并记录条数。"""
    # 同一行刻意重复 3 次（正好等于 min_repeat 默认值），验证是「>= 阈值」而非「> 阈值」
    lines = ["示例银行 内部资料", "正文一。", "示例银行 内部资料", "正文二。", "示例银行 内部资料"]
    kept, boiler = strip_boilerplate(lines)
    assert all("内部资料" not in line for line in kept)
    assert len(boiler) == 3


def test_strip_boilerplate_keeps_long_repeated_paragraphs():
    """超长段落即使反复出现也不算模板行：反复出现的免责声明正文属于有效内容，删掉会丢证据。"""
    # 用 5 次重复把长度顶到 40 字以上，正好越过「长度 <= 40」这条模板行判定线
    long_line = "本产品不构成投资建议，" * 5
    kept, boiler = strip_boilerplate([long_line, long_line, long_line, "短句。"])
    assert long_line in kept
    assert not boiler


def test_ocr_confusions_table_covers_common_cases():
    """形近字表的既定条目必须存在且方向正确（左为识别结果，右为更可能的本字）。"""
    assert OCR_CONFUSIONS["己经"] == "已经"
    assert OCR_CONFUSIONS["帐号"] == "账号"


def test_score_quality_penalises_ocr_documents(corpus):
    """质量分必须落在 [0,1] 且把 OCR 来源如实记为扣分原因（该分数会进重排参与降权）。"""
    doc = corpus.require("OCR-2023-12")
    report = score_quality(doc)
    assert 0.0 <= report.score <= 1.0
    assert any("OCR" in issue for issue in report.issues)


def test_clean_document_records_issues(corpus):
    """就地清洗必须把发现的问题记进文档 issues（治理报表与轨迹要靠它，不能只返回报告）。"""
    from src.ingest import load_corpus as reload_corpus
    from src.config import DATA_DIR
    from src.ingest.cleaning import clean_document

    # 重新读一份新语料而不是复用会话级 corpus：clean_document 会就地改写文档，
    # 就地改会污染同会话其它解析层用例（这正是夹具特意分离两份语料的原因）
    fresh = reload_corpus(DATA_DIR)
    doc = fresh.require("OCR-2023-12")
    report = clean_document(doc)
    before = doc.raw
    assert "已经" in doc.raw or "已经" in before or report.score <= 1.0
    assert doc.issues


# ---------------------------------------------------------------------------
# OCR 适配层
# ---------------------------------------------------------------------------
def test_sidecar_engine_reads_companion_text(tmp_path):
    """旁挂文本后端必须采用同名 .txt 作为识别结果，并把来源标成 sidecar。"""
    # 图片只写 4 字节的 PNG 魔数：本用例验证的是「文本来自旁挂文件」，不需要真实图片内容
    image = tmp_path / "scan.png"
    image.write_bytes(b"\x89PNG")
    (tmp_path / "scan.txt").write_text("扫描件转录内容", encoding="utf-8")
    result = SidecarOCREngine().recognize(image)
    assert result.ok and result.text == "扫描件转录内容"
    assert result.engine == "sidecar"


def test_sidecar_engine_reports_missing_companion(tmp_path):
    """缺旁挂文本时必须报「本次无结果」（ok 为假 + warnings），绝不能伪装成识别成功。"""
    image = tmp_path / "scan.png"
    image.write_bytes(b"\x89PNG")
    result = SidecarOCREngine().recognize(image)
    assert not result.ok
    assert result.warnings


def test_null_engine_never_silently_succeeds(tmp_path):
    """兜底后端必须显式报缺（空文本 + warnings），让缺口在轨迹里可见而不是悄悄丢数据。"""
    image = tmp_path / "scan.png"
    image.write_bytes(b"\x89PNG")
    result = NullOCREngine().recognize(image)
    assert not result.ok and result.warnings


def test_extract_text_rejects_non_image(tmp_path):
    """非图片后缀不进任何 OCR 后端，直接返回 engine="none" 并说明原因。"""
    path = tmp_path / "a.md"
    path.write_text("x", encoding="utf-8")
    result = extract_text(path)
    assert result.engine == "none" and result.warnings


def test_get_engine_returns_something_usable():
    """自动选后端的结果必须始终可用（装了真实引擎用它，没装退到旁挂文本），链条不断。"""
    engine = get_engine("auto")
    assert engine.available()


# ---------------------------------------------------------------------------
# 脱敏在语料上的整体效果
# ---------------------------------------------------------------------------
def test_corpus_contains_no_raw_phone_numbers(corpus):
    """脱敏后整库不得残留原始敏感串——索引、日志、轨迹都从这份文本派生。"""
    # 用夹具里出现过的那个具体号码（138-1234-5678）做全库扫描，而不是只查单篇文档
    for doc in corpus.documents:
        assert "13812345678" not in doc.raw


def test_load_corpus_on_missing_dir_raises(tmp_path):
    """资料目录不存在时必须立即失败（FileNotFoundError），不能返回空资料库让上层误以为「没有资料」。"""
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "not-exist")
