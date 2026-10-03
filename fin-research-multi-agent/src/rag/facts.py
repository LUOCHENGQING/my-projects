"""结构化财务事实抽取（Markdown 表格 -> 指标事实库）。

检索解决「找到相关段落」，但财务分析需要的是**结构化数值**：净利润到底是多少、
单位是万元还是元、是 2024 年还是 2023 年。如果让 LLM 从自然语言里"读"数字，
既不稳定也无法审计。

因此这里把 data/*.md 中所有形如：

    | 指标 | 2024年 | 2023年 | 单位 |
    | --- | --- | --- | --- |
    | 营业收入 | 1286400.00 | 1102300.00 | 万元 |

的表格解析成 (公司, 指标, 年份, 期间) -> 数值 的结构化事实，
由 get_financial_metric 工具对外提供。每一条事实都带 source_id 与所属章节，
保证了「数字 -> 出处」的可追溯。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .corpus import Document

# 表头：| 指标 | 2024年 | 2023年 | 单位 |
_COL_YEAR_RE = re.compile(r"^(\d{4})\s*年\s*(.*)$")
_NUM_RE = re.compile(r"^-?[\d,]+(?:\.\d+)?$")

# 指标别名：把用户 / LLM 的口语说法映射到报表口径
METRIC_ALIASES: Dict[str, str] = {
    "营收": "营业收入",
    "收入": "营业收入",
    "主营业务收入": "营业收入",
    "销售毛利": "毛利润",
    "毛利": "毛利润",
    "净利": "净利润",
    "归母净利": "归母净利润",
    "归属于母公司股东的净利润": "归母净利润",
    "总资产": "资产总额",
    "资产": "资产总额",
    "总负债": "负债总额",
    "负债": "负债总额",
    "净资产": "所有者权益",
    "股东权益": "所有者权益",
    "股东权益合计": "所有者权益",
    "经营现金流": "经营活动现金流净额",
    "经营活动产生的现金流量净额": "经营活动现金流净额",
    "经营性现金流": "经营活动现金流净额",
    "研发投入": "研发费用",
    "应收款": "应收账款",
    "应收": "应收账款",
    "不良率": "不良贷款率",
    "拨备率": "拨备覆盖率",
    "资本充足率": "资本充足率",
    "息差": "净息差",
    "质押比例": "控股股东股权质押比例",
    "担保余额": "对外担保余额",
    "诉讼金额": "未决诉讼涉案金额",
    "客户集中度": "前五大客户销售占比",
    "商誉": "商誉账面价值",
    "股本": "总股本",
}


def canonical_metric(name: str) -> str:
    """把指标名归一化到报表口径的标准名称。"""
    if not name:
        return name
    cleaned = name.strip()
    return METRIC_ALIASES.get(cleaned, cleaned)


def _parse_number(raw: str) -> Optional[float]:
    """解析表格里的数字，支持千分位、负号、百分号。"""
    text = raw.strip().replace(",", "").replace("％", "%")
    if not text or text in {"-", "--", "—", "不适用", "N/A", "n/a"}:
        return None
    if text.endswith("%"):
        text = text[:-1]
    if not _NUM_RE.match(text.replace(",", "")):
        # 允许 "1.42%" 之外的少量噪声（如全角数字），失败即放弃该单元格
        try:
            return float(text)
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


@dataclass
class MetricFact:
    """一条结构化财务事实。"""

    company: str
    metric: str
    year: int
    period: str          # 列头原文，如 "2024年" / "2024年前三季度"
    value: float
    unit: str
    source_id: str
    doc_id: str
    section_title: str
    period_suffix: str = ""   # "年" 之后的附加说明，如 "前三季度"

    def to_dict(self) -> Dict[str, object]:
        return {
            "company": self.company,
            "metric": self.metric,
            "year": self.year,
            "period": self.period,
            "value": self.value,
            "unit": self.unit,
            "source_id": self.source_id,
            "doc_id": self.doc_id,
            "section_title": self.section_title,
        }


def _is_separator_row(cells: Sequence[str]) -> bool:
    return all(set(c.strip()) <= set("-: ") for c in cells) and any(cells)


def _split_row(line: str) -> List[str]:
    """切分 Markdown 表格行。"""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [c.strip() for c in stripped.split("|")]


def parse_tables(doc: Document) -> List[MetricFact]:
    """解析一篇文档里的全部指标表格。"""
    facts: List[MetricFact] = []
    lines = doc.raw.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if "|" not in line or not line.strip().startswith("|"):
            i += 1
            continue

        header = _split_row(line)
        # 需要紧随其后的分隔行，才算真正的 Markdown 表格
        if i + 1 >= n or "|" not in lines[i + 1] or not _is_separator_row(_split_row(lines[i + 1])):
            i += 1
            continue

        # 表头解析：第一列 = 指标名，最后一列若为「单位」则是单位列，其余为年份列
        unit_col: Optional[int] = None
        if header and header[-1].strip() in {"单位", "币种", "计量单位"}:
            unit_col = len(header) - 1

        year_cols: List[Tuple[int, int, str, str]] = []  # (列下标, 年份, 列头原文, 附加说明)
        for idx, cell in enumerate(header):
            if idx == 0 or idx == unit_col:
                continue
            m = _COL_YEAR_RE.match(cell.strip())
            if m:
                year_cols.append((idx, int(m.group(1)), cell.strip(), m.group(2).strip()))

        if not year_cols:
            i += 1
            continue

        section_title = _section_title_at(doc, i)
        j = i + 2
        while j < n and "|" in lines[j] and lines[j].strip().startswith("|"):
            cells = _split_row(lines[j])
            if _is_separator_row(cells):
                j += 1
                continue
            if cells:
                metric_name = canonical_metric(cells[0])
                unit = cells[unit_col].strip() if unit_col is not None and unit_col < len(cells) else ""
                for col_idx, year, period_label, suffix in year_cols:
                    if col_idx >= len(cells):
                        continue
                    value = _parse_number(cells[col_idx])
                    if value is None or not metric_name:
                        continue
                    facts.append(
                        MetricFact(
                            company=doc.company,
                            metric=metric_name,
                            year=year,
                            period=period_label,
                            value=value,
                            unit=unit,
                            source_id=doc.source_id,
                            doc_id=doc.doc_id,
                            section_title=section_title,
                            period_suffix=suffix,
                        )
                    )
            j += 1
        i = j

    return facts


def _section_title_at(doc: Document, line_index: int) -> str:
    """找出某一行所属的章节标题。"""
    title = "正文"
    count = 0
    for line in doc.raw.splitlines():
        if count >= line_index:
            break
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip() or title
        count += 1
    return title


class FactStore:
    """全部文档的结构化事实库。"""

    def __init__(self, facts: Sequence[MetricFact]) -> None:
        self.facts: List[MetricFact] = list(facts)
        self._index: Dict[Tuple[str, str, int, str], MetricFact] = {}
        for fact in self.facts:
            self._index[(fact.company, fact.metric, fact.year, fact.period)] = fact

    def __len__(self) -> int:
        return len(self.facts)

    @property
    def companies(self) -> List[str]:
        return sorted({f.company for f in self.facts})

    def resolve_company(self, name: str) -> Optional[str]:
        """把用户 / LLM 写出的公司简称（如「示例科技」）解析成全称。"""
        if not name:
            return None
        if name in self.companies:
            return name
        for candidate in self.companies:
            if name in candidate or candidate in name:
                return candidate
        stripped = name.replace("股份有限公司", "").replace("有限公司", "").strip()
        if stripped:
            for candidate in self.companies:
                if stripped in candidate:
                    return candidate
        return None

    def years(self, company: str) -> List[int]:
        return sorted({f.year for f in self.facts if f.company == company})

    def metrics(self, company: str, year: int, period: str = "年度") -> List[str]:
        """列出某公司某年某期间下已有的全部指标名。"""
        return sorted(
            {
                f.metric
                for f in self.facts
                if f.company == company and f.year == year and _period_matches(f, period)
            }
        )

    def get(
        self,
        company: str,
        metric: str,
        year: Optional[int] = None,
        period: str = "年度",
    ) -> Optional[MetricFact]:
        """查询一条事实。year 为空时取该公司最新年份。"""
        target_metric = canonical_metric(metric)
        if year is None:
            candidates = [f.year for f in self.facts if f.company == company and f.metric == target_metric]
            if not candidates:
                return None
            year = max(candidates)

        # 1) 精确命中（含期间）
        for fact in self.facts:
            if (
                fact.company == company
                and fact.metric == target_metric
                and fact.year == year
                and _period_matches(fact, period)
            ):
                return fact

        # 2) 放宽公司名（允许简称匹配，例如只写「示例科技」）
        for fact in self.facts:
            if (
                fact.metric == target_metric
                and fact.year == year
                and _period_matches(fact, period)
                and (company in fact.company or fact.company in company)
            ):
                return fact
        return None

    def by_period(self, company: str, year: int, period: str = "年度") -> Dict[str, MetricFact]:
        """取某公司某期间的全部指标，返回 {指标名: 事实}。"""
        out: Dict[str, MetricFact] = {}
        for fact in self.facts:
            if fact.company == company and fact.year == year and _period_matches(fact, period):
                out.setdefault(fact.metric, fact)
        return out


def _period_matches(fact: MetricFact, period: str) -> bool:
    """期间匹配：年度 / 三季度 等口语说法映射到表格列头。"""
    p = (period or "").strip()
    if p in {"", "年度", "年", "全年", "年报"}:
        return fact.period_suffix == ""
    if p in {"三季度", "三季报", "前三季度", "Q3"}:
        return "三季" in fact.period_suffix or "三季" in fact.period
    if p in {"半年度", "中报", "半年报"}:
        return "半年" in fact.period_suffix or "中" in fact.period_suffix
    if p in {"一季度", "一季报", "Q1"}:
        return "一季" in fact.period_suffix or "一季" in fact.period
    return p in fact.period or p in fact.period_suffix


def build_fact_store(documents: Sequence[Document]) -> FactStore:
    """从全部文档构建事实库。"""
    facts: List[MetricFact] = []
    for doc in documents:
        facts.extend(parse_tables(doc))
    return FactStore(facts)
