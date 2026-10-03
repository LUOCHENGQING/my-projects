"""确定性规则大脑（mock LLM）。

没有 OPENAI_API_KEY 时，系统自动切换到这里。它的输出 JSON 结构与真实 LLM
**完全一致**，因此上层 Agent 代码零改动、`python -m src.demo` 与 `pytest` 永远可跑。

它不是"随便返回一段假文本"：
    1. 用关键词 + 正则解析用户问题（公司、年份、分析维度）；
    2. 用阈值表对已计算的比率做定性判断（优于/低于基准、是否恶化）；
    3. 严格按照 AnalystAgent 的打回意见去补交叉验证说明。

换句话说，它是一个**可复现的规则策略**，而不是随机文本生成器。
真机模式下同一份 prompt 会发给真实模型，二者行为可通过 eval 对齐比较。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# 关键词表
# ---------------------------------------------------------------------------
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

DEFAULT_TARGETS = ["盈利能力", "偿债能力", "风险合规"]

# 比率 -> 所属分析维度（用于按维度聚合结论）
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
    """从问题里抽年份；抽不到就用已知最新年度。"""
    matches = re.findall(r"(20\d{2})\s*年?", question)
    if matches:
        return int(matches[0])
    return fallback


def detect_companies(question: str, known: Sequence[str]) -> List[str]:
    """在问题里匹配已知公司（支持简称）。"""
    found: List[str] = []
    for full in known:
        short = full.replace("股份有限公司", "").replace("有限公司", "")
        if full in question or (short and short in question):
            found.append(full)
    if found:
        return found

    # 未显式点名：按行业关键词兜底
    if any(k in question for k in ("银行", "不良", "拨备", "息差")):
        return [c for c in known if "银行" in c][:1] or list(known[:1])
    return list(known[:1])


def detect_period(question: str) -> str:
    """识别报告期间口径：年度 / 三季度 / 半年度 / 一季度。"""
    if any(k in question for k in ("三季度", "三季报", "前三季度", "Q3", "第三季度")):
        return "三季度"
    if any(k in question for k in ("半年度", "半年报", "中报", "上半年")):
        return "半年度"
    if any(k in question for k in ("一季度", "一季报", "Q1", "第一季度")):
        return "一季度"
    return "年度"


def detect_targets(question: str) -> List[str]:
    """按关键词识别分析维度。"""
    targets = [name for name, words in TARGET_KEYWORDS.items() if any(w in question for w in words)]
    if not targets:
        return list(DEFAULT_TARGETS)
    # 只要问到了风险，风险合规分析必须包含
    if "风险合规" not in targets and any(w in question for w in TARGET_KEYWORDS["风险合规"]):
        targets.append("风险合规")
    return targets


def _build_queries(company: str, year: int, targets: Sequence[str], question: str) -> List[str]:
    """为每个分析维度生成一路检索查询（多路检索的来源）。"""
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
def _task_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
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
    candidates: List[Dict[str, Any]] = payload.get("candidates") or []
    targets: Sequence[str] = payload.get("targets") or []
    queries: Sequence[str] = payload.get("queries") or []
    limit: int = int(payload.get("limit") or 8)

    # 1) 先按父块去重：同一章节只保留得分最高的一块，保证证据多样性
    best_by_parent: Dict[str, Dict[str, Any]] = {}
    for item in sorted(candidates, key=lambda x: -float(x.get("score") or 0.0)):
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
    """把数值映射成定性判断，供结论陈述使用。"""
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
        words = TARGET_KEYWORDS.get(target, [target])
        bound = [
            cid for cid in evidence_ids
            if any(w in evidence_text_index.get(cid, "") for w in words)
        ][:3]
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
    """根据打回意见，为异常比率补交叉验证说明。"""
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
        notes = []
    return notes


def _task_risk_review(payload: Dict[str, Any]) -> Dict[str, Any]:
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
    company: str = payload.get("company", "")
    year: int = int(payload.get("year") or 0)
    period: str = str(payload.get("period") or "年度")
    findings: List[Dict[str, Any]] = payload.get("findings") or []
    citation_map: Dict[str, List[int]] = payload.get("citation_map") or {}
    risk: Dict[str, Any] = payload.get("risk") or {}

    def cite_for(fid: str) -> str:
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
_DISPATCH = {
    "plan": _task_plan,
    "retrieve": _task_retrieve,
    "analyze": _task_analyze,
    "risk_review": _task_risk_review,
    "write": _task_write,
}


def run_mock(task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """执行确定性策略，返回结构化结果。"""
    handler = _DISPATCH.get(task)
    if handler is None:
        return {"error": f"mock 不支持的任务类型：{task}", "missing_data": [task]}
    return handler(payload or {})
