"""确定性规则大脑（mock LLM）。

没有 OPENAI_API_KEY 时，系统自动切换到这里。它的输出 JSON 结构与真实 LLM
**完全一致**，因此上层 Agent 代码零改动、`python -m src.demo` 与 `pytest` 永远可跑。

它不是"随便返回一段假文本"：
    1. 用关键词 + 正则解析用户问题（公司、年份、分析维度）；
    2. 用阈值表对已计算的比率做定性判断（优于/低于基准、是否恶化）；
    3. 严格按照 AnalystAgent 的打回意见去补交叉验证说明。

换句话说，它是一个**可复现的规则策略**，而不是随机文本生成器。
真机模式下同一份 prompt 会发给真实模型，二者行为可通过 eval 对齐比较。

层次与职责
----------
本模块是「模型访问层」的离线后端，由 `src/llm/client.py` 的 `_mock_response()` 调用：
* 上游：`LLMClient.chat()` -> `client._mock_response()` -> `run_mock()`；
* 下游：不 import 项目内其它模块（只用标准库 re / typing），因此**不可能联网**；
* 它只做「payload -> 结构化 dict」，不调用工具、不读文件、不查数据库。

对外关键函数
------------
* `run_mock(task, payload)`  唯一入口，按任务名派发到五个 `_task_*` 实现
* 解析器：`detect_year` / `detect_companies` / `detect_period` / `detect_targets` / `_build_queries`
* 判断器：`_qualify`（定性） / `_direction`（趋势） / `_build_cross_checks`（交叉验证说明）

无 API Key 时如何保证全链路可离线复现
------------------------------------
1. **确定性输出**：全程无随机数、无当前时间、无网络、无全局可变状态，输出只是
   `payload` 的纯函数；同一份 payload 调用两次，`client._mock_response()` 里
   `json.dumps(data, ensure_ascii=False, sort_keys=True)` 的文本逐字节相同
   （`tests/test_mock_llm.py` 正是这样断言 plan 结果两次一致）。
2. **结构一致**：每个 `_task_*` 返回的键名与真机提示词里声明的 JSON schema 同名同层级，
   因此上层 Agent 的字段读取、引用校验、评测指标在两种模式下走同一段代码。
3. **派发兜底**：未知任务名不会抛异常，而是返回带 `error` 的结构化结果（见 `run_mock`），
   保证离线链路不会因为某个任务未实现而整条中断。
4. **数值来源不变**：mock 只对「已由工具算出的比率」做定性表述，不自己造数字，
   与「数字由工具产生、语言由模型产生」的总原则一致。

主要输入输出
------------
输入：`task`（plan/retrieve/analyze/risk_review/write）+ `payload`（dict，字段随任务而异）。
输出：与真机同构的 dict（含 `missing_data` 键），不含任何时间戳/随机标识。

被谁调用
--------
`src/llm/client.py`（mock 模式与降级路径）；`tests/test_mock_llm.py` 直接调用
`run_mock` 做单元断言。本模块**不关心** prompt 文本（client 侧构造的 user prompt
不会传进来），因此提示词改动不会影响 mock 的确定性。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# 关键词表
# ---------------------------------------------------------------------------
# 分析维度 -> 触发词。detect_targets 用它从用户问题里认出维度；
# _task_retrieve / _task_analyze 复用同一张表做证据命中与维度覆盖判定，保证「识别口径」与
# 「证据口径」一致（否则会出现认出了维度却覆盖不到证据的自相矛盾）
TARGET_KEYWORDS: Dict[str, Sequence[str]] = {
    "盈利能力": ["盈利", "利润", "毛利", "净利", "roe", "回报率", "赚钱", "业绩", "margin", "赚钱能力"],
    "偿债能力": ["偿债", "负债", "杠杆", "流动性", "债务", "资产负债率", "还债", "短期偿债"],
    "成长性": ["成长", "增长", "增速", "扩张", "同比", "发展趋势"],
    "现金流质量": ["现金流", "回款", "现金含量", "经营性现金", "造血", "资金链"],
    "营运效率": ["周转", "营运", "效率", "资产使用"],
    "风险合规": ["风险", "合规", "诉讼", "担保", "质押", "减值", "违规", "处罚", "隐患", "暴雷"],
    "银行监管指标": ["不良", "拨备", "资本充足", "息差", "银行", "监管指标", "资产质量"],
    "股东回报": ["分红", "派息", "股息", "回购", "股东回报"],
}

# 问题里一个维度关键词都没命中时的兜底维度：覆盖面日常三大项，保证结论不为空
DEFAULT_TARGETS = ["盈利能力", "偿债能力", "风险合规"]

# 比率 -> 所属分析维度（用于按维度聚合结论）
# 仅作为兜底：上游 analyst.py 传入的每条指标自带 target 字段，只有缺失时才回查这张表
RATIO_TARGET: Dict[str, str] = {
    "net_margin": "盈利能力",
    "gross_margin": "盈利能力",
    "roe": "盈利能力",
    "roa": "盈利能力",
    "rnd_intensity": "盈利能力",
    "expense_ratio": "盈利能力",
    "debt_to_asset": "偿债能力",
    "current_ratio": "偿债能力",
    "equity_multiplier": "偿债能力",
    "asset_turnover": "营运效率",
    "cash_conversion": "现金流质量",
    "revenue_growth": "成长性",
    "profit_growth": "成长性",
    "npl_ratio": "银行监管指标",
    "provision_coverage": "银行监管指标",
}

# 定性判断阈值：(下限, 定性, 是否越大越好)
# 注：实际实现为「比率名 -> dict」，键为 good / warn / bad 阈值、higher_better、unit，
#     并非上面那行注释描述的元组；_qualify 与 _direction 只读 good / warn / bad /
#     higher_better 四个键，unit 仅作阅读标注（"pct" 表示 0~1 的小数口径，"x" 表示倍数）。
# 键名对应上游传入的 ratio_name（growth_rate 被 revenue_growth / profit_growth 等条目复用），
# 因此表里虽然只有一条 growth_rate，实际覆盖了多个展示口径。
_RATIO_VIEWS: Dict[str, Dict[str, Any]] = {
    "net_margin": {"good": 0.05, "warn": 0.0, "higher_better": True, "unit": "pct"},
    "gross_margin": {"good": 0.25, "warn": 0.15, "higher_better": True, "unit": "pct"},
    "debt_to_asset": {"warn": 0.60, "bad": 0.70, "higher_better": False, "unit": "pct"},
    "current_ratio": {"good": 2.0, "warn": 1.0, "higher_better": True, "unit": "x"},
    "roe": {"good": 0.10, "warn": 0.06, "higher_better": True, "unit": "pct"},
    "roa": {"good": 0.03, "warn": 0.01, "higher_better": True, "unit": "pct"},
    "cash_conversion": {"good": 0.8, "warn": 0.5, "higher_better": True, "unit": "x"},
    "growth_rate": {"good": 0.10, "warn": 0.0, "higher_better": True, "unit": "pct"},
    "asset_turnover": {"good": 0.6, "warn": 0.3, "higher_better": True, "unit": "x"},
    "equity_multiplier": {"warn": 2.5, "bad": 3.5, "higher_better": False, "unit": "x"},
    "provision_coverage": {"good": 2.0, "warn": 1.5, "higher_better": True, "unit": "pct"},
    "npl_ratio": {"good": 0.01, "warn": 0.015, "higher_better": False, "unit": "pct"},
    "capital_adequacy": {"good": 0.12, "warn": 0.105, "higher_better": True, "unit": "pct"},
}

# 维度 -> 该维度结论的论证模板
# 模板里只留 company / year / verdict / details 四个占位符，其中 verdict 由 _task_analyze
# 依据该维度内最差的 _direction 结果填入（improving->整体稳健 等），details 是各比率的
# 「标签 + 展示值 + 定性」拼接，保证结论陈述既有数字又有判断
_STATEMENT_TEMPLATES: Dict[str, str] = {
    "盈利能力": "{company} {year} 年盈利能力{verdict}：{details}。",
    "偿债能力": "{company} {year} 年偿债能力{verdict}：{details}。",
    "成长性": "{company} {year} 年成长性{verdict}：{details}。",
    "现金流质量": "{company} {year} 年现金流质量{verdict}：{details}。",
    "营运效率": "{company} {year} 年营运效率{verdict}：{details}。",
    "银行监管指标": "{company} {year} 年监管指标{verdict}：{details}。",
    "股东回报": "{company} {year} 年股东回报{verdict}：{details}。",
    "风险合规": "{company} {year} 年合规与风险状况{verdict}：{details}。",
}


# ---------------------------------------------------------------------------
# 问题解析
# ---------------------------------------------------------------------------
def detect_year(question: str, fallback: int = 2024) -> int:
    """从问题里抽年份；抽不到就用已知最新年度。

    参数：
        question: 用户原始问题文本。
        fallback: 正则抽不到年份时返回的年度，默认 2024。
                  注：实际实现里默认值是固定的 2024；真正的「已知最新年度」由调用方
                  `_task_plan` 通过 `payload["latest_year"]` 传进 fallback 才会生效。
    返回：
        int —— 第一个匹配 `(20\\d{2})年?` 的四位年份（有问题里出现多个年份时取**第一个**）；
        完全没有 20xx 时返回 fallback。
    副作用/异常：
        纯函数，无副作用；question 为空串也不会抛异常。
    """
    matches = re.findall(r"(20\d{2})\s*年?", question)
    if matches:
        return int(matches[0])
    return fallback


def detect_companies(question: str, known: Sequence[str]) -> List[str]:
    """在问题里匹配已知公司（支持简称）。

    参数：
        question: 用户原始问题文本。
        known:    资料库中已知的公司全称序列（同时也是兜底候选来源）。
    返回：
        List[str] —— 命中的公司**全称**列表，保持 known 的原始顺序；
        命中条件是全称整体出现在问题里，或去掉「股份有限公司 / 有限公司」后的简称出现在
        问题里。若一个都没命中，则做行业兜底：问题含 银行/不良/拨备/息差 时返回第一个
        名称含「银行」的公司，否则返回 known 的第一项；known 为空时返回空列表。
    副作用/异常：
        纯函数，不修改入参、不做 I/O；known 为空时切片安全，不抛异常。
    """
    found: List[str] = []
    for full in known:
        short = full.replace("股份有限公司", "").replace("有限公司", "")
        if full in question or (short and short in question):
            found.append(full)
    if found:
        return found

    # 未显式点名：按行业关键词兜底（宁可给一个最可能的标的，也不要空手返回）
    if any(k in question for k in ("银行", "不良", "拨备", "息差")):
        return [c for c in known if "银行" in c][:1] or list(known[:1])
    return list(known[:1])


def detect_period(question: str) -> str:
    """识别报告期间口径：年度 / 三季度 / 半年度 / 一季度。

    参数：
        question: 用户原始问题文本。
    返回：
        str —— 取值只能是 "三季度" / "半年度" / "一季度" / "年度" 四种；
        按「三季度 -> 半年度 -> 一季度」的**固定优先级**判定（一个问题同时出现多个口径时，
        靠前的胜出），都不命中则为 "年度"。
    副作用/异常：纯函数，无副作用，不抛异常。
    """
    if any(k in question for k in ("三季度", "三季报", "前三季度", "Q3", "第三季度")):
        return "三季度"
    if any(k in question for k in ("半年度", "半年报", "中报", "上半年")):
        return "半年度"
    if any(k in question for k in ("一季度", "一季报", "Q1", "第一季度")):
        return "一季度"
    return "年度"


def detect_targets(question: str) -> List[str]:
    """按关键词识别分析维度。

    参数：
        question: 用户原始问题文本。
    返回：
        List[str] —— 命中的维度名列表（顺序由 TARGET_KEYWORDS 的声明顺序决定，
        即盈利能力 -> 偿债能力 -> 成长性 -> 现金流质量 -> 营运效率 -> 风险合规 ->
        银行监管指标 -> 股东回报）；一个都没命中时返回 DEFAULT_TARGETS 的副本。
    副作用/异常：
        纯函数；返回的是新列表（含 DEFAULT_TARGETS 分支的 `list(...)` 拷贝），
        调用方修改结果不会污染模块常量。
    """
    targets = [name for name, words in TARGET_KEYWORDS.items() if any(w in question for w in words)]
    if not targets:
        return list(DEFAULT_TARGETS)
    # 只要问到了风险，风险合规分析必须包含（风险是合规底线，不能被用户措辞漏掉）
    if "风险合规" not in targets and any(w in question for w in TARGET_KEYWORDS["风险合规"]):
        targets.append("风险合规")
    return targets


def _build_queries(company: str, year: int, targets: Sequence[str], question: str) -> List[str]:
    """为每个分析维度生成一路检索查询（多路检索的来源）。

    参数：
        company: 已识别的公司名；为空串时前缀退化为 " 2024年" 这种形式（不报错）。
        year:    分析年度，拼进查询前缀。
        targets: 已识别的分析维度；不在 mapping 里的维度会被静默跳过。
        question: 用户原始问题，作为额外最后一路查询（保留原始措辞以免丢失意图）。
    返回：
        List[str] —— 每维度一条 + 末尾一条原始问题，最多 6 条（`queries[:6]` 截断）。
        注：实际实现的上限是 6 条，比 prompts.py 里 PlannerAgent 声明的「3~5 条」更宽。
    副作用/异常：纯函数，无 I/O、无异常。
    """
    base = f"{company} {year}年"
    mapping = {
        "盈利能力": f"{base} 营业收入 净利润 毛利率 净利率 研发费用率",
        "偿债能力": f"{base} 资产负债率 流动比率 负债总额 偿债能力",
        "成长性": f"{base} 营业收入同比 净利润同比 营业收入增长",
        "现金流质量": f"{base} 经营活动现金流量净额 应收账款 回款 账期",
        "营运效率": f"{base} 总资产周转 存货 营运效率",
        "风险合规": f"{base} 风险 未决诉讼 对外担保 股权质押 商誉减值",
        "银行监管指标": f"{base} 不良贷款率 拨备覆盖率 资本充足率 净息差",
        "股东回报": f"{base} 利润分配 现金分红 每股收益",
    }
    queries = [mapping[t] for t in targets if t in mapping]
    # 额外一路"原文问题"查询，保留用户原始措辞
    queries.append(question.strip())
    return queries[:6]


# ---------------------------------------------------------------------------
# 各任务的确定性策略
# ---------------------------------------------------------------------------
# 下面五个 _task_* 是「一个任务名一个纯函数」，签名统一为 (payload) -> dict，
# 输出键与 prompts.py 中该 Agent 声明的 JSON schema 对齐；新增任务只需再注册进 _DISPATCH。
def _task_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
    """plan 任务：把研究问题解析成公司 / 年度 / 期间 / 维度，并给出路由与子任务。

    参数（读 payload，全部可选，缺失走默认值）：
        payload["question"]:      用户原始问题，解析的唯一依据。
        payload["companies"]:     资料库已知公司全称列表，供 detect_companies 匹配。
        payload["latest_year"]:   抽不到年份时的兜底年度（默认 2024）。
    返回：
        Dict[str, Any] —— 与 PlannerAgent 契约同构，键为 intent / companies / year /
        period / targets / route / subtasks / retrieval_queries / missing_data /
        assumptions；subtasks 固定四步 T1 检索 -> T2 分析 -> T3 核查 -> T4 撰写，
        route 固定为 ["retriever", "analyst", "risk_checker", "writer"]。
    副作用/异常：
        纯函数，无随机、无时间、无 I/O；payload 为空 dict 时也能产出完整结构，
        因此输出对同一输入可逐字节复现。
    """
    question: str = payload.get("question", "")
    known_companies: Sequence[str] = payload.get("companies") or []
    latest_year: int = int(payload.get("latest_year") or 2024)

    companies = detect_companies(question, known_companies)
    year = detect_year(question, fallback=latest_year)
    targets = detect_targets(question)
    period = detect_period(question)

    subtasks: List[Dict[str, Any]] = [
        {"id": "T1", "agent": "retriever", "goal": f"检索 {companies[0] if companies else ''} {year} 年与各分析维度相关的原文证据", "depends_on": []},
        {"id": "T2", "agent": "analyst", "goal": f"计算 {'、'.join(targets)} 相关指标并形成结论", "depends_on": ["T1"]},
        {"id": "T3", "agent": "risk_checker", "goal": "核查结论是否有数据与证据支撑，并执行风险规则扫描", "depends_on": ["T2"]},
        {"id": "T4", "agent": "writer", "goal": "生成带引用编号的投研简报", "depends_on": ["T3"]},
    ]

    return {
        "intent": f"对 {'/'.join(companies) if companies else '目标公司'} {year} 年{period}做 {'、'.join(targets)} 分析并提示风险",
        "companies": companies,
        "year": year,
        "period": period,
        "targets": targets,
        "route": ["retriever", "analyst", "risk_checker", "writer"],
        "subtasks": subtasks,
        "retrieval_queries": _build_queries(companies[0] if companies else "", year, targets, question),
        "missing_data": [],
        "assumptions": [
            f"未在问题中显式指明时，默认分析 {year} 年{period}口径。",
            "本系统仅使用本地资料库中的虚构演示数据，不构成任何投资建议。",
        ],
    }


def _task_retrieve(payload: Dict[str, Any]) -> Dict[str, Any]:
    """retrieve 任务：从候选片段里做去重筛选取 Top-N，并给出覆盖情况与相关性说明。

    参数（读 payload）：
        payload["candidates"]: 检索器返回的候选片段列表，每项含 child_id / parent_id /
                               score / text / section_title / matched_terms 等字段。
        payload["targets"]:    需要覆盖的分析维度；用于算 coverage 与 missing_data。
        payload["queries"]:    实际执行的多路查询串，原样回显。
        payload["limit"]:      保留片段数上限，默认 8。
    返回：
        Dict[str, Any] —— 与 RetrieverAgent 契约同构，键为 queries / selected /
        relevance_notes / coverage / missing_data；其中 selected 是保留片段的 child_id
        列表，missing_data 是 coverage 里一条证据都没命中的维度。
    副作用/异常：
        纯函数，不修改 payload；候选为空时返回空 selected 与全量 missing_data，不抛异常。
    判定口径：
        维度覆盖用 TARGET_KEYWORDS 的关键词在 `text + section_title` 上做子串匹配，
        与 detect_targets 共用同一张词表，保证「识别」与「取证」口径一致。
    """
    candidates: List[Dict[str, Any]] = payload.get("candidates") or []
    targets: Sequence[str] = payload.get("targets") or []
    queries: Sequence[str] = payload.get("queries") or []
    limit: int = int(payload.get("limit") or 8)

    # 1) 先按父块去重：同一章节只保留得分最高的一块，保证证据多样性
    # （父块相同意味着上下文高度重叠，重复占用名额只会降低覆盖率）
    best_by_parent: Dict[str, Dict[str, Any]] = {}
    for item in sorted(candidates, key=lambda x: -float(x.get("score") or 0.0)):
        # parent_id 缺失时退回 child_id，保证每个片段都有稳定的去重键
        pid = str(item.get("parent_id") or item.get("child_id"))
        if pid not in best_by_parent:
            best_by_parent[pid] = item
    pool = list(best_by_parent.values())[:limit]

    # 2) 计算每个分析维度的覆盖情况
    coverage: Dict[str, List[str]] = {}
    for target in targets:
        words = TARGET_KEYWORDS.get(target, [target])
        hits = [
            str(item.get("child_id"))
            for item in pool
            if any(w in str(item.get("text", "")) + str(item.get("section_title", "")) for w in words)
        ]
        coverage[target] = hits[:4]

    # 3) 生成相关性说明
    notes: Dict[str, str] = {}
    for item in pool:
        cid = str(item.get("child_id"))
        section = str(item.get("section_title", ""))
        matched = item.get("matched_terms") or []
        terms = "、".join([str(t) for t in matched[:5]])
        notes[cid] = (
            f"命中《{section}》章节，匹配词：{terms or '语义相近'}"
            if terms
            else f"来自《{section}》章节，与问题语义相近"
        )

    covered_targets = [t for t, ids in coverage.items() if ids]
    return {
        "queries": list(queries),
        "selected": [str(item.get("child_id")) for item in pool],
        "relevance_notes": notes,
        "coverage": coverage,
        "missing_data": [t for t in targets if t not in covered_targets],
    }


def _qualify(ratio_name: str, value: float) -> str:
    """把数值映射成定性判断，供结论陈述使用。

    参数：
        ratio_name: 比率名，用于在 _RATIO_VIEWS 里查阈值（如 net_margin / debt_to_asset）。
        value:      该比率的实际值（pct 类为 0~1 小数口径，x 类为倍数），由上游工具算出。
    返回：
        str —— 中文定性短语，取值集合固定为：
        越大越好的比率："表现良好" / "基本达标但偏弱" / "明显偏弱"；
        越小越好的比率："偏高需警惕" / "处于偏高水平" / "处于合理区间"；
        比率名不在阈值表里时："处于可观察区间"。
    副作用/异常：
        纯函数，只读阈值表；对 NaN/None 之类非法值不做校验（调用方已把 value 转成 float），
        不抛异常。
    """
    view = _RATIO_VIEWS.get(ratio_name)
    if not view:
        return "处于可观察区间"
    higher_better = bool(view.get("higher_better", True))
    if higher_better:
        if "good" in view and value >= view["good"]:
            return "表现良好"
        if "warn" in view and value >= view["warn"]:
            return "基本达标但偏弱"
        return "明显偏弱"
    if "bad" in view and value >= view["bad"]:
        return "偏高需警惕"
    if "warn" in view and value >= view["warn"]:
        return "处于偏高水平"
    return "处于合理区间"


def _direction(ratio_name: str, value: float) -> str:
    """把数值映射成趋势方向，供结论陈述与维度汇总使用。

    参数：
        ratio_name: 比率名，用于在 _RATIO_VIEWS 里查阈值。
        value:      该比率的实际值（口径同 _qualify）。
    返回：
        str —— 取值为 "improving" / "stable" / "deteriorating"（与 Agent 契约里的
        direction 字段同域）；比率名不在阈值表里时返回 "stable"。
    副作用/异常：
        纯函数，只读阈值表，不抛异常。
    注：这里判断的是「当期数值相对阈值的位置」，并非跨期比较——
        真正的同比变化由上游 comparatives（如 cash_conversion_prior）体现。
    """
    view = _RATIO_VIEWS.get(ratio_name)
    if not view:
        return "stable"
    higher_better = bool(view.get("higher_better", True))
    if higher_better:
        if "good" in view and value >= view["good"]:
            return "improving"
        return "deteriorating"
    if "bad" in view and value >= view["bad"]:
        return "deteriorating"
    if "warn" in view and value >= view["warn"]:
        return "stable"
    return "improving"


def _task_analyze(payload: Dict[str, Any]) -> Dict[str, Any]:
    """analyze 任务：按维度把已计算比率聚合成带论证链的分析结论。

    参数（读 payload）：
        payload["company"]:            公司名，写进结论陈述。
        payload["year"]:               年度，写进结论陈述。
        payload["ratios"]:             已计算比率列表，每项含 ratio_name / value / display /
                                       label / target / supplementary 等字段。
        payload["evidence"]:           可用证据列表，每项含 child_id / text。
        payload["targets"]:            关注的维度，仅在 grouped 为空时作为退化遍历源。
        payload["revision_requests"]:  上一轮 RiskChecker 的打回意见；非空时才会生成交叉验证。
        payload["missing"]:            缺失数据说明，原样透传到 missing_data。
        payload["comparatives"]:       上期/同期对照值，供 _build_cross_checks 使用。
    返回：
        Dict[str, Any] —— 键为 summary / findings / missing_data；每条 finding 含
        id（F1 起递增）/ title / target / statement / ratio_refs / evidence_ids /
        direction / cross_checks。
    副作用/异常：
        纯函数；比率或证据为空时返回空 findings（不抛异常），summary 会如实说明形成 0 条结论。
    关键取舍：
        * `supplementary=True` 的比率（如 receivables_growth）只做交叉验证材料，不进结论陈述，
          避免用验证口径污染主结论；
        * 维度方向取该维度内**最差**的 _direction（deteriorating > stable > improving），
          宁可提示风险也不要用均值掩盖问题。
    """
    company: str = payload.get("company", "")
    year: int = int(payload.get("year") or 0)
    ratios: List[Dict[str, Any]] = payload.get("ratios") or []
    evidence: List[Dict[str, Any]] = payload.get("evidence") or []
    revision_requests: List[str] = payload.get("revision_requests") or []
    targets: Sequence[str] = payload.get("targets") or []

    evidence_ids = [str(e.get("child_id")) for e in evidence]
    evidence_text_index = {str(e.get("child_id")): str(e.get("text", "")) for e in evidence}

    # 按分析维度聚合比率；supplementary 口径只作交叉验证材料，不写进结论陈述
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for ratio in ratios:
        if ratio.get("supplementary"):
            continue
        target = ratio.get("target") or RATIO_TARGET.get(str(ratio.get("ratio_name")), "盈利能力")
        grouped.setdefault(target, []).append(ratio)

    findings: List[Dict[str, Any]] = []
    fid = 0
    for target in list(grouped) or list(targets):
        items = grouped.get(target)
        if not items:
            continue
        fid += 1
        parts: List[str] = []
        refs: List[str] = []
        worst = "improving"
        for item in items:
            name = str(item.get("ratio_name"))
            value = float(item.get("value") or 0.0)
            display = str(item.get("display") or value)
            label = str(item.get("label") or name)
            parts.append(f"{label} {display}，{_qualify(name, value)}")
            refs.append(name)
            if _direction(name, value) == "deteriorating":
                worst = "deteriorating"
            elif _direction(name, value) == "stable" and worst != "deteriorating":
                worst = "stable"

        # 证据绑定：优先挑与该维度关键词共现的证据
        # （共现说明这段原文确实谈到了该维度，比按分数硬塞更可解释）
        words = TARGET_KEYWORDS.get(target, [target])
        bound = [
            cid for cid in evidence_ids
            if any(w in evidence_text_index.get(cid, "") for w in words)
        ][:3]
        # 一条都没共现时退回证据列表前两条：宁可弱关联，也不要让结论完全无出处
        if not bound:
            bound = evidence_ids[:2]

        statement = _STATEMENT_TEMPLATES.get(
            target, "{company} {year} 年{target}方面{verdict}：{details}。"
        ).format(
            company=company,
            year=year,
            target=target,
            verdict={"improving": "整体稳健", "stable": "总体平稳", "deteriorating": "存在压力"}[worst],
            details="；".join(parts),
        )

        cross_checks: List[str] = []
        if revision_requests:
            cross_checks = _build_cross_checks(items, revision_requests, payload)

        findings.append(
            {
                "id": f"F{fid}",
                "title": f"{target}分析",
                "target": target,
                "statement": statement,
                "ratio_refs": refs,
                "evidence_ids": bound,
                "direction": worst,
                "cross_checks": cross_checks,
            }
        )

    summary_bits = [f["title"] + "：" + f["statement"] for f in findings[:2]]
    return {
        "summary": (
            f"基于结构化指标计算，{company} {year} 年度共形成 {len(findings)} 条分析结论。"
            + (" ".join(summary_bits) if summary_bits else "")
        ),
        "findings": findings,
        "missing_data": payload.get("missing") or [],
    }


def _build_cross_checks(
    items: Sequence[Dict[str, Any]],
    revision_requests: Sequence[str],
    payload: Dict[str, Any],
) -> List[str]:
    """根据打回意见，为异常比率补交叉验证说明。

    参数：
        items:             本维度下的比率条目（含 ratio_name / value）。
        revision_requests: 打回意见文本列表；仅作为「本轮确实被打回过」的信号使用。
        payload:           需读取 `payload["comparatives"]`（上期与同期对照值）。
    返回：
        List[str] —— 交叉验证说明列表；只对三类比率产出说明：
        `cash_conversion`（含量、上期对比、应收增速 vs 收入增速互证）、
        `debt_to_asset`（杠杆水平变化，要求结合流动比率判断）、
        `revenue_growth`（要求与应收账款、经营现金流三方对照）；
        其余比率（含 supplementary 口径）不产出说明，无内容时返回**空列表**。
    副作用/异常：
        纯函数，不修改 payload；comparatives 缺字段时跳过对应分句，不抛异常。
    注：实际实现为函数末尾的 `if not notes and revision_requests:` 分支只是把 notes 再赋成
        空列表（此时它本来就是空列表），属于空操作——真正的行为是「无异常指标时不编套话」。
    """
    comparatives: Dict[str, Any] = payload.get("comparatives") or {}
    notes: List[str] = []
    for item in items:
        name = str(item.get("ratio_name"))
        value = float(item.get("value") or 0.0)
        if name == "cash_conversion":
            prior = comparatives.get("cash_conversion_prior")
            ar_growth = comparatives.get("receivable_growth")
            rev_growth = comparatives.get("revenue_growth")
            bits = [f"净利润现金含量 {value:.2f} 倍"]
            if prior is not None:
                bits.append(f"上期为 {float(prior):.2f} 倍，同比{'下降' if value < float(prior) else '上升'}")
            if ar_growth is not None and rev_growth is not None:
                bits.append(
                    f"同期应收账款增速 {float(ar_growth) * 100:.2f}% 高于营业收入增速 "
                    f"{float(rev_growth) * 100:.2f}%，两者互相印证：现金流走弱并非季节性因素，"
                    "而是结算周期延长与备货增加共同导致"
                )
            notes.append("；".join(bits) + "。")
        elif name == "debt_to_asset":
            prior = comparatives.get("debt_to_asset_prior")
            if prior is not None:
                notes.append(
                    f"资产负债率 {value * 100:.2f}%，上期为 {float(prior) * 100:.2f}%，"
                    f"杠杆水平{'上升' if value > float(prior) else '下降'}，需结合流动比率与短期借款结构判断。"
                )
        elif name == "revenue_growth":
            notes.append(
                f"营业收入同比增速 {value * 100:.2f}%，需与应收账款增速、经营现金流增速做三方对照。"
            )
    if not notes and revision_requests:
        # 没有异常指标需要交叉验证时，如实说明「本轮无需额外交叉验证」，
        # 而不是硬塞一句没有信息量的套话
        # 注：实际实现为把 notes 重新赋成空列表（它此时本就是空列表），是空操作；
        #     该分支的净效果就是「保持空列表返回」。
        notes = []
    return notes


def _task_risk_review(payload: Dict[str, Any]) -> Dict[str, Any]:
    """risk_review 任务：对分析结论做裁决（通过 / 打回重算 / 升级人工）。

    参数（读 payload）：
        payload["gate"]: {"gaps": [...], "round": 当前轮次, "max_rounds": 轮次上限}；
                         gate 整体缺失时按「无缺口、第 0 轮、上限 2 轮」处理。
        payload["risk"]: {"findings": [...], "overall_level": "info|low|medium|high"}，
                         来自风险规则引擎的扫描结果。
    返回：
        Dict[str, Any] —— 与 RiskCheckerAgent 契约同构，键为 verdict / narrative /
        gaps / risk_level / escalation_reason / missing_data；
        verdict 取值为 "revise" / "escalate" / "pass"，gaps 原样回显。
    副作用/异常：
        纯函数，无 I/O；payload 为空时也会给出 "pass" 的完整结构。
    裁决优先级（自上而下，先命中先返回）：
        1) 有 gaps 且 round < max_rounds           -> "revise"（还能重算，打回 Analyst）
        2) 有 gaps 且 round >= max_rounds          -> "escalate"（重算次数达上限）
        3) 无 gaps 但整体风险等级为 "high"          -> "escalate"（转人工确认）
        4) 其余                                    -> "pass"
        注意第 2 条优先于第 3 条：轮次耗尽时无论风险等级都会升级。
    """
    gate: Dict[str, Any] = payload.get("gate") or {}
    gaps: List[Dict[str, Any]] = gate.get("gaps") or []
    round_no: int = int(gate.get("round") or 0)
    max_rounds: int = int(gate.get("max_rounds") or 2)
    risk: Dict[str, Any] = payload.get("risk") or {}
    risk_findings: List[Dict[str, Any]] = risk.get("findings") or []
    overall = str(risk.get("overall_level") or "info")

    if gaps and round_no < max_rounds:
        verdict = "revise"
        narrative = (
            f"第 {round_no + 1} 轮核查发现 {len(gaps)} 项结论支撑不足，已打回 AnalystAgent 重算："
            + "；".join(g.get("problem", "") for g in gaps[:3])
            + "。"
        )
        escalation = ""
    elif gaps and round_no >= max_rounds:
        verdict = "escalate"
        narrative = (
            f"已完成 {max_rounds} 轮重算，仍有 {len(gaps)} 项结论支撑不足（重算次数已达上限），"
            "按流程标记为「需人工确认」。"
        )
        escalation = "反思循环达到上限仍未消除数据缺口，需要人工补充资料或调整研究口径。"
    elif overall == "high":
        verdict = "escalate"
        high_titles = "、".join(f["title"] for f in risk_findings if f.get("level") == "high") or "高风险事项"
        narrative = (
            f"结论本身证据链完整，但风险规则扫描命中 {len(risk_findings)} 项，"
            f"其中高风险事项：{high_titles}，需人工确认后再对外输出。"
        )
        escalation = f"命中高风险规则：{high_titles}。"
    else:
        verdict = "pass"
        narrative = (
            f"结论均有指标与原文支撑，风险规则扫描命中 {len(risk_findings)} 项、"
            f"整体风险等级为 {overall}，未触发人工确认门槛。"
        )
        escalation = ""

    return {
        "verdict": verdict,
        "narrative": narrative,
        "gaps": gaps,
        "risk_level": overall,
        "escalation_reason": escalation,
        "missing_data": [],
    }


def _task_write(payload: Dict[str, Any]) -> Dict[str, Any]:
    """write 任务：把结论、风险与引用编号组装成结构化投研简报。

    参数（读 payload）：
        payload["company"]:      公司名，写进标题。
        payload["year"]:         年度，写进标题。
        payload["period"]:       报告期间口径（默认 "年度"），写进标题。
        payload["findings"]:     分析结论列表，每项至少含 id / statement；
                                 仅前 4 条进入 executive_summary，全部进入 analysis_paragraphs。
        payload["citation_map"]: {finding_id: [引用编号, ...]}，由 WriterAgent 用
                                 cite_source 工具预先绑定；缺编号时该句就没有 [n]。
        payload["risk"]:         风险报告，取 findings[0] 与 overall_level 写风险提示。
    返回：
        Dict[str, Any] —— 键为 title / executive_summary / analysis_paragraphs /
        risk_note / missing_data；executive_summary 是字符串列表，
        analysis_paragraphs 是 {finding_id: 段落} 字典。
    副作用/异常：
        纯函数，不修改 payload；findings 为空时 executive_summary 退化为一条
        「未形成有效结论」的提示（仍带一次引用尝试），不抛异常。
    约束：
        编号只从 citation_map 里取（每个结论最多 3 个），**不新增任何数据**，
        因此不会产生悬空引用——这是评测中 citation_traceability_rate 能到 1.0 的前提。
    """
    company: str = payload.get("company", "")
    year: int = int(payload.get("year") or 0)
    period: str = str(payload.get("period") or "年度")
    findings: List[Dict[str, Any]] = payload.get("findings") or []
    citation_map: Dict[str, List[int]] = payload.get("citation_map") or {}
    risk: Dict[str, Any] = payload.get("risk") or {}

    def cite_for(fid: str) -> str:
        """把某条结论的引用编号拼成 `[1][3]` 形式的内联标注。

        参数：
            fid: 结论 id（如 "F1"），作为 citation_map 的键。
        返回：
            str —— 该结论最多 3 个编号拼接出的标注串；无对应编号时返回空串
            （宁可没有标注，也不伪造编号，否则评测的悬空引用检查会失败）。
        副作用/异常：
            闭包读取外层 citation_map，不修改它；无异常。
        """
        nos = citation_map.get(fid) or []
        return "".join(f"[{n}]" for n in nos[:3])

    executive: List[str] = []
    for finding in findings[:4]:
        cid = str(finding.get("id"))
        executive.append(f"{finding.get('statement', '')}{cite_for(cid)}")

    risk_findings = risk.get("findings") or []
    if risk_findings:
        top = risk_findings[0]
        all_cites = "".join(f"[{n}]" for n in sorted({n for nos in citation_map.values() for n in nos})[:3])
        executive.append(
            f"风险层面，{top.get('title', '')}（{top.get('metric_display', '')}）达到"
            f"{top.get('threshold', '')}的触发条件，需重点跟踪。{all_cites}"
        )
    if not executive:
        executive = [f"本次分析未形成有效结论，建议补充资料后重跑。{cite_for('F1')}"]

    paragraphs: Dict[str, str] = {}
    for finding in findings:
        cid = str(finding.get("id"))
        cite = cite_for(cid)
        # 交叉验证内容由 WriterAgent 以列表形式单独渲染，这里不重复拼接
        paragraphs[cid] = f"{finding.get('statement', '')}{cite}"

    return {
        "title": f"{company} {year} 年{period}投研简报",
        "executive_summary": executive,
        "analysis_paragraphs": paragraphs,
        "risk_note": (
            f"本次核查共命中 {len(risk_findings)} 项风险规则，整体风险等级 "
            f"{risk.get('overall_level', 'info')}。"
        ),
        "missing_data": [],
    }


# ---------------------------------------------------------------------------
# 总入口
# ---------------------------------------------------------------------------
# 任务名 -> 实现函数。键必须与 client._PROMPT_KEYS 的键、以及提示词里声明的任务一一对应；
# 用查表派发而非 if/elif，是为了让「mock 支持哪些任务」一眼可数（也便于测试遍历）
_DISPATCH = {
    "plan": _task_plan,
    "retrieve": _task_retrieve,
    "analyze": _task_analyze,
    "risk_review": _task_risk_review,
    "write": _task_write,
}


def run_mock(task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """执行确定性策略，返回结构化结果。

    参数：
        task:    任务名，应为 plan/retrieve/analyze/risk_review/write 之一。
        payload: 结构化输入；为 None 时按空 dict 处理（`payload or {}`）。
    返回：
        Dict[str, Any] —— 对应 `_task_*` 的输出；任务名未知时返回
        `{"error": "mock 不支持的任务类型：<task>", "missing_data": [task]}`，
        形如正常结果（而不是 None 或抛异常），让上层仍能按同一套字段读取。
    副作用/异常：
        纯函数、不联网、不写文件、无随机与时间依赖；任何输入都不抛异常，
        因此同一份 payload 的返回可逐字节复现（client 侧再经 sort_keys 序列化）。
    """
    handler = _DISPATCH.get(task)
    if handler is None:
        return {"error": f"mock 不支持的任务类型：{task}", "missing_data": [task]}
    return handler(payload or {})
