"""结构化财务事实抽取（Markdown 表格 -> 指标事实库）——投研数字的**唯一合法来源**。

层次与职责
----------
RAG 证据层的结构化通路：把 data/*.md 里的指标表格解析成「(公司, 指标, 年份, 期间)
-> 数值」的事实记录，供 FactStore 建立索引。它只做解析与查询，不做比率计算
（比率在 tools/builtin.py 里由 RATIO_DEFS / calc_ratio 的纯函数完成）。

关键类 / 函数：
    MetricFact   —— 一条事实（含 company / metric / year / period / value / unit /
                    source_id / section_title）
    FactStore    —— 事实库与查询入口（get / metrics / by_period / resolve_company）
    parse_tables / build_fact_store —— 表格解析与批量建库
    canonical_metric / METRIC_ALIASES —— 口语指标名 -> 报表口径的归一化

主要输入：corpus.Document 列表（直接扫描 doc.raw 的表格行）。
主要输出：FactStore。
被谁调用：src/orchestrator.py（build_fact_store 建库后注册进工具上下文）、
          src/tools/builtin.py 的 get_financial_metric / assess_risk 等工具。

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

这套结构为什么能从根上掐掉「数字幻觉」
--------------------------------------
* **数值不经过 LLM 的笔**：报告里出现的每个数字都必须先由 get_financial_metric 从
  本模块的表格解析结果里取出来；模型只被允许解释与串联，不允许生成数值。
* **口径被结构化锁死**：company / metric / year / period 四个维度共同定位一个值，
  「2024 年的净利润」和「2024 年前三季度的净利润」在库里是两条不同的记录
  （由 period_suffix 区分，见 _period_matches），不会张冠李戴。
* **比率是确定性纯函数**：派生指标（净利率、资产负债率、同比增长率……）由
  tools/builtin.py 的 RATIO_DEFS / calc_ratio 用固定公式算出，同样不经过模型。
* **缺失显式暴露**：查不到指标时工具抛 ValueError 并列出该期间可用指标，
  让"没有数据"变成一个可见的错误，而不是让模型顺手编一个数字补上。
* **可审计**：每条事实都带 source_id / doc_id / section_title，报告中任一数字都能
  顺着这条链回到具体章节。

解析口径与边界
--------------
* 只把「标题行 + 紧随其后的分隔行（| --- |）+ 至少一个年份列」的区块当作表格，
  普通正文里出现的 | 不会误判（见 _is_separator_row）。
* 表头第一列 = 指标名；末列若为「单位 / 币种 / 计量单位」则当单位列；其余列用
  _COL_YEAR_RE 识别为年份列，4 位年份之后的文字进 period_suffix（如「前三季度」）。
* 单元格解析失败（空、"-"、"不适用"、非法数字）直接跳过，不抛异常也不写占位值——
  宁可少一条事实，也不要写入一个错误的数值。
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
    """把指标名归一化到报表口径的标准名称。

    参数：name —— 用户 / LLM 写出的指标名（如「营收」「净利」「归母净利」）。
    返回：命中的标准名（METRIC_ALIASES 的值，如「营业收入」「净利润」）；未收录时原样
        返回 strip 后的名字。空串 / None 直接原样返回（不做 strip），便于上层判空。
    副作用：无。这是"口径统一"的第一道闸门：别名表之外的说法不会被硬猜。
    """
    if not name:
        return name
    cleaned = name.strip()
    return METRIC_ALIASES.get(cleaned, cleaned)


def _parse_number(raw: str) -> Optional[float]:
    """解析表格里的数字，支持千分位、负号、百分号。

    参数：raw —— 单元格原文（可能带逗号、％ 或 %）。
    返回：float；空串，以及代码里明确列出的 "-" / "--" / "—" / "不适用" / "N/A" / "n/a"
        这些占位符返回 None（= 该单元格没有可用数值）。
        百分号会被剥掉但**不做 /100 换算**，即 "1.42%" 返回 1.42（单位由事实的 unit 字段承载）。
    副作用：无。解析失败一律返回 None，绝不抛异常——表格里偶发的排版噪声不该中断建库。
    """
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
    """一条结构化财务事实（= 表格里的一个数值单元格）。

    关键属性：
        company      公司全称（来自文档 front-matter）
        metric       已归一化的报表口径指标名（经 canonical_metric）
        year         4 位年份（从列头解析，int）
        period       列头原文，如 "2024年" / "2024年前三季度"
        value        数值（百分号已剥、未做 /100）
        unit         单位原文，如 "万元" / "%"；无单位列时为空串
        source_id / doc_id / section_title —— 出处三元组，引用可追溯的依据
        period_suffix "年"之后的附加说明，如 "前三季度"；年度报告为空串
    状态流转：构造后只读。同 (company, metric, year, period) 在 FactStore 里是唯一键。
    """

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
        """转成可 JSON 序列化的字典（get_financial_metric 工具的返回体）。

        参数：无。
        返回：dict，含 company / metric / year / period / value / unit / source_id /
            doc_id / section_title 共 9 个键。
        副作用：无。注：实际实现里**不返回** period_suffix，工具层只暴露 period 原文。
        """
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
    """判断一行是不是 Markdown 表格的分隔行（如 | --- | :--: |）。

    参数：cells —— 已切分的单元格列表。
    返回：bool。所有单元格去掉空白后只由 "-: " 组成，且至少有一个非空单元格时为 True。
    副作用：无。它是 parse_tables 判定"这确实是表格"的关键闸门，可挡掉正文里的裸 |。
    """
    return all(set(c.strip()) <= set("-: ") for c in cells) and any(cells)


def _split_row(line: str) -> List[str]:
    """切分 Markdown 表格行。

    参数：line —— 表格原始行（建议以 | 开头结尾，缺失时也能处理）。
    返回：单元格文本列表，每格已 strip。行首行尾的单个 | 被剥掉，内部按 | 切分；
        单元格内出现被转义的竖线时不做特殊处理（当前语料不涉及这种情况）。
    副作用：无。
    """
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [c.strip() for c in stripped.split("|")]


def parse_tables(doc: Document) -> List[MetricFact]:
    """解析一篇文档里的全部指标表格。

    参数：doc —— Document（实现上直接按行扫描 doc.raw，不用 sections）。
    返回：List[MetricFact]，按表格出现顺序；单元格无有效数字时该条事实不产出。
    副作用：无（对 doc 只读）。
    异常：不抛异常——格式不完整的区块会被整体跳过并继续向后扫描。
    识别条件（三者缺一不可）：当前行以 | 开头且含 |；下一行是分隔行（见
        _is_separator_row）；表头里至少有一个 _COL_YEAR_RE 命中的年份列。
    表体从表头后第二行开始，连续读取所有以 | 开头的行；中途再遇分隔行会跳过。
    """
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
        # 为什么必须校验分隔行：正文里也会出现竖线（如"A | B"），只看竖线会把普通
        # 文本误判成表格并抽出垃圾数值，所以要求格式三要素齐全。
        if i + 1 >= n or "|" not in lines[i + 1] or not _is_separator_row(_split_row(lines[i + 1])):
            i += 1
            continue

        # 表头解析：第一列 = 指标名，最后一列若为「单位」则是单位列，其余为年份列
        # 为什么只认末列：当前语料的单位都放在最后一列；认不出就当无单位（unit 为空串），
        # 不做任何猜测，避免给数值配上错误量纲。
        unit_col: Optional[int] = None
        if header and header[-1].strip() in {"单位", "币种", "计量单位"}:
            unit_col = len(header) - 1

        year_cols: List[Tuple[int, int, str, str]] = []  # (列下标, 年份, 列头原文, 附加说明)
        for idx, cell in enumerate(header):
            if idx == 0 or idx == unit_col:
                continue
            # 为什么严格按「4 位年份 + 年」匹配：宁可少抽一张表，也不要把「序号」这类
            # 数字列误当成时间维度——错配年份比缺数据更危险。
            m = _COL_YEAR_RE.match(cell.strip())
            if m:
                year_cols.append((idx, int(m.group(1)), cell.strip(), m.group(2).strip()))

        if not year_cols:
            i += 1
            continue

        # 为什么用行号回溯标题：表格行属于 doc.raw，不在 Section 对象里，
        # 只能回到原始全文按行号找最近的标题，才能给事实挂上「出处章节」。
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
                    # 为什么判越界：Markdown 允许短行（缺格子），缺的列直接跳过，
                    # 绝不能顺延补位——错位的数值会静默污染事实库。
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
    """找出某一行所属的章节标题。

    参数：doc 文档；line_index 目标行在 doc.raw 中的 0 基下标。
    返回：该行之前最近一个 Markdown 标题文本；若之前没有任何标题，返回默认的「正文」。
    副作用：无。按行扫描（而非查 sections）是为了与 parse_tables 的行号口径严格对齐。
    """
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
    """全部文档的结构化事实库（数值型结论的唯一取数口）。

    关键属性：
        facts    —— List[MetricFact]，按文档与表格出现顺序
        _index   —— (company, metric, year, period) -> MetricFact 的唯一键索引
                    （注：实际实现里它只在构造期建立，get / metrics / by_period 都是
                    线性扫描 self.facts，并未读取该索引；语料规模小，可读性优先）

    状态流转：构造后只读，只提供查询；不做任何写入、缓存或跨公司推断。
    典型用法（tools/builtin.py）：resolve_company 归一公司名 -> get 取一条事实 ->
    calc_ratio 用事实里的数值算比率。
    """

    def __init__(self, facts: Sequence[MetricFact]) -> None:
        """建立事实库与 (公司, 指标, 年份, 期间) 索引。

        参数：facts —— 事实序列（会被复制成新列表）。
        返回：None。
        副作用：无（不改动传入对象）。
        注：四个维度完全相同时，后写入的事实会覆盖索引项（同表重复列头时以最后一条为准）；
            by_period 则相反，用 setdefault 让先出现的那条胜出。
        """
        self.facts: List[MetricFact] = list(facts)
        self._index: Dict[Tuple[str, str, int, str], MetricFact] = {}
        for fact in self.facts:
            self._index[(fact.company, fact.metric, fact.year, fact.period)] = fact

    def __len__(self) -> int:
        """事实条数（demo / eval 用它展示语料覆盖度）。

        参数：无。返回：int。副作用：无。
        """
        return len(self.facts)

    @property
    def companies(self) -> List[str]:
        """库里出现过的公司全称，去重后排序。

        参数：无（属性访问）。返回：List[str]。副作用：无。
        """
        return sorted({f.company for f in self.facts})

    def resolve_company(self, name: str) -> Optional[str]:
        """把用户 / LLM 写出的公司简称（如「示例科技」）解析成全称。

        参数：name —— 任意公司名写法。
        返回：库中匹配到的全称；无法匹配返回 None。
        副作用：无。
        匹配顺序（先严后宽，避免误伤）：
            1) 与某个全称完全相等；
            2) 与某个全称互为子串（双向包含）；
            3) 去掉「股份有限公司」「有限公司」后缀后再做子串包含。
        注：companies 已排序，因此同分情况下命中的是全称字典序最靠前的那个。
        """
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
        """该公司在库里有数据的年份，升序去重（get_financial_metric 会把它回给模型，
        让模型知道哪些年份可查，而不是自己猜一个）。

        参数：company 公司全称（需先经 resolve_company 归一）。
        返回：List[int]；公司不存在时返回空列表。
        副作用：无。
        """
        return sorted({f.year for f in self.facts if f.company == company})

    def metrics(self, company: str, year: int, period: str = "年度") -> List[str]:
        """列出某公司某年某期间下已有的全部指标名。

        参数：company 公司全称；year 年份；period 期间口语说法（见 _period_matches）。
        返回：排序去重后的指标名列表；无匹配时返回空列表。
        副作用：无。用途：查不到指标时报错信息里列出"该期间可用指标"，把缺口讲清楚。
        """
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
        """查询一条事实。year 为空时取该公司最新年份。

        参数：
            company 公司全称（内部先经 canonical_metric 处理 metric，company 不再归一）；
            metric 指标名（支持 METRIC_ALIASES 里的口语说法）；
            year 年份，None 时取该公司该指标的最大年份；
            period 期间口语说法。
        返回：命中的 MetricFact；没命中返回 None（**不抛异常、不返回默认值**，
            由调用方决定如何暴露缺口）。
        副作用：无。两轮匹配：先要求公司全称精确相等，再放宽为双向子串匹配
            （因此传简称也能命中）；metric / year / period 两轮都必须满足。
        """
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
        """取某公司某期间的全部指标，返回 {指标名: 事实}。

        参数：company 公司全称；year 年份；period 期间口语说法。
        返回：dict；无匹配时返回空字典（不抛异常）。同一指标重复出现时**先出现的胜出**
            （用 setdefault 写入），与 __init__ 索引的"后写覆盖"相反。
        副作用：无。
        """
        out: Dict[str, MetricFact] = {}
        for fact in self.facts:
            if fact.company == company and fact.year == year and _period_matches(fact, period):
                out.setdefault(fact.metric, fact)
        return out


def _period_matches(fact: MetricFact, period: str) -> bool:
    """期间匹配：年度 / 三季度 等口语说法映射到表格列头。

    参数：fact 待判定的事实；period 调用方给出的期间说法。
    返回：bool。
    副作用：无。
    规则：
        "年度/年/全年/年报/空串" —— 只认 period_suffix == "" 的年度记录
            （为什么必须严格：否则「2024年前三季度」的数据会被当成年度数，直接算错同比）；
        "三季度/三季报/前三季度/Q3" —— period_suffix 或 period 含「三季」；
        "半年度/中报/半年报"      —— period_suffix 含「半年」或「中」；
        "一季度/一季报/Q1"        —— period_suffix 或 period 含「一季」；
        其它说法                  —— 退化为子串包含（period 或 period_suffix 里能找到即可）。
    """
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
    """从全部文档构建事实库。

    参数：documents —— 文档序列（通常来自 DocumentStore 的迭代）。
    返回：FactStore（内部已按 (公司, 指标, 年份, 期间) 建好索引）。
    副作用：无（对文档只读）。解析失败的单元格被静默跳过，因此这里不会抛异常。
    """
    facts: List[MetricFact] = []
    for doc in documents:
        facts.extend(parse_tables(doc))
    return FactStore(facts)
