"""AnalystAgent —— 财务指标计算与分析。

职责边界（本项目最重要的一条设计原则）：
    **数字由工具产生，语言由模型产生。**

    * Agent 自己决定「算哪些指标、用哪两个科目、用什么口径」——这是领域策略，
      必须确定、可审计、可复现，不能交给模型即兴发挥。
    * 所有数值一律经由 `get_financial_metric`（结构化事实库）与 `calc_ratio`（比率计算器）
      产生；模型只负责解释这些数字之间的关系，并给出结论措辞。
    * 结论里出现的每一个 `ratio_refs` / `evidence_ids` 都会被二次校验，
      不在清单内的（= 幻觉出来的）一律剔除，并记入 errors。

它只持有 `get_financial_metric` + `calc_ratio`，拿不到 write 权限，
从权限层面杜绝"分析师顺手改数据/写文件"。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..state import ResearchState
from ..tools.registry import PermissionLevel
from .base import BaseAgent

#: 分析维度 -> 指标计算计划。
#: kind=ratio  : 计算 分子科目 / 分母科目
#: kind=growth : 计算 (本期 - 上期) / |上期|
#: kind=direct : 直接取披露值（监管指标等）
RATIO_PLAN: Dict[str, List[Dict[str, Any]]] = {
    "盈利能力": [
        {"key": "net_margin", "kind": "ratio", "ratio_name": "net_margin",
         "numerator": "净利润", "denominator": "营业收入"},
        {"key": "gross_margin", "kind": "ratio", "ratio_name": "gross_margin",
         "numerator": "毛利润", "denominator": "营业收入"},
        {"key": "roe", "kind": "ratio", "ratio_name": "roe",
         "numerator": "净利润", "denominator": "所有者权益"},
        {"key": "rnd_intensity", "kind": "ratio", "ratio_name": "rnd_intensity",
         "numerator": "研发费用", "denominator": "营业收入"},
    ],
    "偿债能力": [
        {"key": "debt_to_asset", "kind": "ratio", "ratio_name": "debt_to_asset",
         "numerator": "负债总额", "denominator": "资产总额"},
        {"key": "current_ratio", "kind": "ratio", "ratio_name": "current_ratio",
         "numerator": "流动资产", "denominator": "流动负债"},
    ],
    "成长性": [
        {"key": "revenue_growth", "kind": "growth", "ratio_name": "growth_rate",
         "metric": "营业收入", "label": "营业收入同比增速"},
        {"key": "profit_growth", "kind": "growth", "ratio_name": "growth_rate",
         "metric": "净利润", "label": "净利润同比增速"},
    ],
    "现金流质量": [
        {"key": "cash_conversion", "kind": "ratio", "ratio_name": "cash_conversion",
         "numerator": "经营活动现金流净额", "denominator": "净利润"},
    ],
    "营运效率": [
        {"key": "asset_turnover", "kind": "ratio", "ratio_name": "asset_turnover",
         "numerator": "营业收入", "denominator": "资产总额"},
    ],
    "银行监管指标": [
        {"key": "net_margin", "kind": "ratio", "ratio_name": "net_margin",
         "numerator": "净利润", "denominator": "营业收入"},
        {"key": "npl_ratio", "kind": "direct", "ratio_name": "npl_ratio",
         "metric": "不良贷款率", "pct": True},
        {"key": "provision_coverage", "kind": "direct", "ratio_name": "provision_coverage",
         "metric": "拨备覆盖率", "pct": True},
        {"key": "capital_adequacy", "kind": "direct", "ratio_name": "capital_adequacy",
         "metric": "资本充足率", "pct": True},
    ],
    "股东回报": [],
    "风险合规": [],
}

#: 当问题只涉及风险合规、没有任何会产出比率的维度时使用的**基线指标集**。
#: 理由：如果只回答"有诉讼、有担保"而没有任何杠杆与现金流口径，
#: 核查门会判定「结论缺乏量化支撑」并把人卡在反思循环里。
#: 注意它是"兜底"而不是"叠加"：当其它维度已经算过这些口径时不会重复计算，
#: 避免同一指标在多个结论里被反复陈述。
RISK_BASELINE_PLAN: List[Dict[str, Any]] = [
    {"key": "debt_to_asset", "kind": "ratio", "ratio_name": "debt_to_asset",
     "numerator": "负债总额", "denominator": "资产总额", "target": "风险合规"},
    {"key": "cash_conversion", "kind": "ratio", "ratio_name": "cash_conversion",
     "numerator": "经营活动现金流净额", "denominator": "净利润", "target": "风险合规"},
    {"key": "revenue_growth", "kind": "growth", "ratio_name": "growth_rate",
     "metric": "营业收入", "label": "营业收入同比增速", "target": "风险合规"},
]

#: 收到打回意见后额外补充的计算（用于交叉验证）。
#: supplementary=True 表示这些口径是"验证材料"，不进入结论陈述本身，
#: 只在交叉验证与比率表里出现，避免把"应收增速 35%"说成"表现良好"这类误读。
REVISION_PLAN: List[Dict[str, Any]] = [
    {"key": "receivable_growth", "kind": "growth", "ratio_name": "growth_rate",
     "metric": "应收账款", "label": "应收账款同比增速", "target": "现金流质量",
     "supplementary": True},
    {"key": "total_asset_growth", "kind": "growth", "ratio_name": "growth_rate",
     "metric": "资产总额", "label": "资产总额同比增速", "target": "偿债能力",
     "supplementary": True},
]

#: 交叉验证需要的历史口径（上一年度同一比率）
PRIOR_COMPARATIVES: List[Dict[str, Any]] = [
    {"key": "cash_conversion_prior", "ratio_name": "cash_conversion",
     "numerator": "经营活动现金流净额", "denominator": "净利润"},
    {"key": "debt_to_asset_prior", "ratio_name": "debt_to_asset",
     "numerator": "负债总额", "denominator": "资产总额"},
    {"key": "net_margin_prior", "ratio_name": "net_margin",
     "numerator": "净利润", "denominator": "营业收入"},
]


class AnalystAgent(BaseAgent):
    name = "analyst"
    role = "财务指标计算与分析"
    allowed_tools = ("get_financial_metric", "calc_ratio")
    permissions = frozenset({PermissionLevel.PUBLIC_READ, PermissionLevel.COMPUTE})

    def _execute(self, state: ResearchState) -> ResearchState:
        new_state = self._copy(state)
        plan = state.get("plan") or {}
        company = (plan.get("companies") or [""])[0]
        year = int(plan.get("year") or 0)
        period = str(plan.get("period") or "年度")
        targets: List[str] = list(plan.get("targets") or [])
        revision_round = int(state.get("revision_round") or 0)
        revision_requests: List[str] = list(state.get("revision_requests") or [])

        facts: Dict[str, Any] = {}
        metrics: Dict[str, Any] = {}
        missing: List[str] = []

        # ---- 1) 按维度执行指标计算计划 ----
        entries: List[Dict[str, Any]] = []
        for target in targets:
            for entry in RATIO_PLAN.get(target, []):
                item = dict(entry)
                item.setdefault("target", target)
                entries.append(item)

        # 纯风险类问题没有任何比率口径 -> 用基线指标兜底，保证结论有量化支撑
        used_baseline = False
        if not entries:
            used_baseline = True
            entries = [dict(item) for item in RISK_BASELINE_PLAN]

        seen_keys = set()
        for entry in entries:
            if entry["key"] in seen_keys:
                continue
            seen_keys.add(entry["key"])
            self._compute_entry(entry, company, year, period, facts, metrics, missing)

        # ---- 2) 被打回时：补充交叉验证所需的口径 ----
        comparatives: Dict[str, Any] = {}
        if revision_round > 0:
            for entry in REVISION_PLAN:
                if entry["key"] in metrics:
                    continue
                item = dict(entry)
                self._compute_entry(item, company, year, period, facts, metrics, missing)

            for entry in PRIOR_COMPARATIVES:
                value = self._compute_ratio_for_year(
                    company, year - 1, period, str(entry["ratio_name"]),
                    str(entry["numerator"]), str(entry["denominator"]),
                )
                if value is not None:
                    comparatives[entry["key"]] = value

            comparatives["revenue_growth"] = _value_of(metrics.get("revenue_growth"))
            comparatives["receivable_growth"] = _value_of(metrics.get("receivable_growth"))

        # ---- 3) LLM 生成结论（数字来自工具，语言来自模型） ----
        payload = {
            "company": company,
            "year": year,
            "period": period,
            "targets": targets,
            "revision_round": revision_round,
            "revision_requests": revision_requests,
            "used_risk_baseline": used_baseline,
            "ratios": [
                {
                    "key": item.get("key"),
                    "ratio_name": item.get("ratio_name"),
                    "label": item.get("label"),
                    "target": item.get("target"),
                    "value": item.get("value"),
                    "value_pct": item.get("value_pct"),
                    "display": item.get("display"),
                    "formula": item.get("formula"),
                    "benchmark": item.get("benchmark"),
                    "source_id": item.get("source_id"),
                    "supplementary": bool(item.get("supplementary")),
                }
                for item in metrics.values()
            ],
            "evidence": [
                {
                    "child_id": e.get("child_id"),
                    "source_id": e.get("source_id"),
                    "section_title": e.get("section_title"),
                    "text": e.get("text"),
                }
                for e in state.get("evidence", [])
            ],
            "comparatives": comparatives,
            "missing": missing,
        }
        response = self.ctx.llm.chat("analyze", payload)
        data: Dict[str, Any] = dict(response.data or {})

        # ---- 4) 反幻觉校验：结论引用必须真实存在 ----
        valid_ratios = set(metrics.keys())
        ratio_names = {str(m.get("ratio_name")) for m in metrics.values()}
        valid_evidence = {str(e.get("child_id")) for e in state.get("evidence", [])}
        findings, rejected = self._validate_findings(data.get("findings") or [], valid_ratios, ratio_names, valid_evidence)

        new_state["facts"] = facts
        new_state["metrics"] = metrics
        new_state["findings"] = findings
        new_state["analysis_summary"] = str(data.get("summary") or "")
        new_state["llm_stats"] = {**(state.get("llm_stats") or {}), "analyst": self._llm_stats_fragment(response)}
        self._merge_errors(new_state, "analyst", rejected + [f"缺失指标: {m}" for m in missing[:5]])
        return new_state

    # ------------------------------------------------------------------
    def _compute_entry(
        self,
        entry: Dict[str, Any],
        company: str,
        year: int,
        period: str,
        facts: Dict[str, Any],
        metrics: Dict[str, Any],
        missing: List[str],
    ) -> None:
        """执行一条指标计算计划。"""
        target = str(entry.get("target") or "")
        key = str(entry["key"])
        kind = str(entry["kind"])

        if kind == "ratio":
            num = self._fetch(company, str(entry["numerator"]), year, facts, period)
            den = self._fetch(company, str(entry["denominator"]), year, facts, period)
            if num is None or den is None:
                missing.append(f"{key}（缺少 {entry['numerator']} 或 {entry['denominator']}）")
                return
            result = self.call_tool(
                "calc_ratio",
                {
                    "ratio_name": str(entry["ratio_name"]),
                    "numerator": float(num["value"]),
                    "denominator": float(den["value"]),
                    "precision": 4,
                },
            )
            if not result.ok:
                missing.append(f"{key}（计算失败：{(result.error or {}).get('message', '')}）")
                return
            item = dict(result.data)
            item.update({"key": key, "target": target, "source_id": num.get("source_id"),
                         "inputs_detail": {entry["numerator"]: num["value"], entry["denominator"]: den["value"]},
                         "unit_of_inputs": num.get("unit", "")})
            metrics[key] = item

        elif kind == "growth":
            cur = self._fetch(company, str(entry["metric"]), year, facts, period)
            prev = self._fetch(company, str(entry["metric"]), year - 1, facts, period)
            if cur is None or prev is None:
                missing.append(f"{key}（缺少 {entry['metric']} 的上期数据）")
                return
            result = self.call_tool(
                "calc_ratio",
                {
                    "ratio_name": "growth_rate",
                    "numerator": float(cur["value"]),
                    "denominator": float(prev["value"]),
                    "precision": 4,
                    "label": str(entry.get("label") or f"{entry['metric']}同比增速"),
                },
            )
            if not result.ok:
                missing.append(f"{key}（计算失败：{(result.error or {}).get('message', '')}）")
                return
            item = dict(result.data)
            item.update({"key": key, "target": target or "成长性", "source_id": cur.get("source_id"),
                         "supplementary": bool(entry.get("supplementary"))})
            metrics[key] = item

        elif kind == "direct":
            fact = self._fetch(company, str(entry["metric"]), year, facts, period)
            if fact is None:
                missing.append(f"{key}（未披露 {entry['metric']}）")
                return
            raw = float(fact["value"])
            value = raw / 100.0 if entry.get("pct") else raw
            metrics[key] = {
                "key": key,
                "target": target,
                "ratio_name": str(entry["ratio_name"]),
                "label": _direct_label(str(entry["metric"])),
                "formula": "直接取披露值",
                "value": value,
                "value_pct": round(value * 100.0, 4) if entry.get("pct") else None,
                "display": f"{raw:.2f}%" if entry.get("pct") else f"{raw:.4f}",
                "source_id": fact.get("source_id"),
                "benchmark": "",
            }

    def _fetch(
        self,
        company: str,
        metric: str,
        year: int,
        facts: Dict[str, Any],
        period: str = "年度",
    ) -> Optional[Dict[str, Any]]:
        """取结构化指标（带本地缓存，避免同一指标重复调用工具）。"""
        cache_key = f"{metric}|{year}|{period}"
        if cache_key in facts:
            return facts[cache_key] or None
        result = self.call_tool(
            "get_financial_metric",
            {"company": company, "metric": metric, "year": year, "period": period},
        )
        if not result.ok:
            facts[cache_key] = None
            return None
        facts[cache_key] = result.data
        return result.data

    def _compute_ratio_for_year(
        self,
        company: str,
        year: int,
        period: str,
        ratio_name: str,
        numerator_metric: str,
        denominator_metric: str,
    ) -> Optional[float]:
        """计算上一年度的同一比率，用于同比对照。"""
        num = self._fetch(company, numerator_metric, year, {}, period)
        den = self._fetch(company, denominator_metric, year, {}, period)
        if num is None or den is None or float(den["value"]) == 0:
            return None
        result = self.call_tool(
            "calc_ratio",
            {
                "ratio_name": ratio_name,
                "numerator": float(num["value"]),
                "denominator": float(den["value"]),
                "precision": 6,
            },
        )
        return float(result.data["value"]) if result.ok else None

    # ------------------------------------------------------------------
    @staticmethod
    def _validate_findings(
        raw_findings: List[Dict[str, Any]],
        valid_ratio_keys: set,
        valid_ratio_names: set,
        valid_evidence: set,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """剔除引用不存在指标 / 证据的结论（防幻觉）。"""
        findings: List[Dict[str, Any]] = []
        rejected: List[str] = []
        for idx, raw in enumerate(raw_findings, 1):
            if not isinstance(raw, dict):
                continue
            refs = [str(r) for r in (raw.get("ratio_refs") or [])]
            good_refs = [r for r in refs if r in valid_ratio_keys or r in valid_ratio_names]
            bad_refs = [r for r in refs if r not in good_refs]

            evs = [str(e) for e in (raw.get("evidence_ids") or [])]
            good_evs = [e for e in evs if e in valid_evidence]
            bad_evs = [e for e in evs if e not in good_evs]

            if bad_refs:
                rejected.append(f"结论 {raw.get('id', idx)} 引用了不存在的指标 {bad_refs}，已剔除这些引用")
            if bad_evs:
                rejected.append(f"结论 {raw.get('id', idx)} 引用了不存在的证据 {bad_evs}，已剔除这些引用")

            statement = str(raw.get("statement") or "").strip()
            if not statement:
                continue

            findings.append(
                {
                    "id": str(raw.get("id") or f"F{idx}"),
                    "title": str(raw.get("title") or f"结论{idx}"),
                    "target": str(raw.get("target") or ""),
                    "statement": statement,
                    "ratio_refs": good_refs,
                    "evidence_ids": good_evs,
                    "direction": str(raw.get("direction") or "stable"),
                    "cross_checks": [str(c) for c in (raw.get("cross_checks") or [])],
                }
            )
        return findings, rejected

    # ------------------------------------------------------------------
    def _trace_input(self, state: ResearchState) -> Dict[str, Any]:
        plan = state.get("plan") or {}
        return {
            "company": (plan.get("companies") or [""])[0],
            "year": plan.get("year"),
            "targets": plan.get("targets", []),
            "revision_round": state.get("revision_round", 0),
            "revision_requests": state.get("revision_requests", []),
        }

    def _trace_output(self, state: ResearchState) -> Dict[str, Any]:
        return {
            "metrics": {
                k: {"label": v.get("label"), "display": v.get("display"), "value": v.get("value")}
                for k, v in (state.get("metrics") or {}).items()
            },
            "findings": [
                {
                    "id": f.get("id"),
                    "title": f.get("title"),
                    "ratio_refs": f.get("ratio_refs"),
                    "evidence_ids": f.get("evidence_ids"),
                    "cross_checks": len(f.get("cross_checks") or []),
                }
                for f in (state.get("findings") or [])
            ],
            "summary": state.get("analysis_summary", ""),
        }


def _value_of(item: Optional[Dict[str, Any]]) -> Optional[float]:
    if not item:
        return None
    value = item.get("value")
    return float(value) if isinstance(value, (int, float)) else None


def _direct_label(metric: str) -> str:
    return {
        "不良贷款率": "不良贷款率",
        "拨备覆盖率": "拨备覆盖率",
        "资本充足率": "资本充足率",
        "净息差": "净息差",
    }.get(metric, metric)
