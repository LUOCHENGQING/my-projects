"""RiskCheckerAgent —— 风险与合规核查（反思循环的驱动者）。

这是整个系统里唯一有权「打回上游」的 Agent，也是唯一能触发人机协同的 Agent。

它做两件事，且刻意分开：
    1. **确定性核查门（gate）** —— 结论是否有证据、是否有量化支撑、异常指标是否做了
       交叉验证、引用能否回溯到原文。这部分是合规闸门，**不交给模型判断**。
    2. **风险规则扫描** —— 调用 check_risk_rules 工具，按阈值规则给出风险等级与命中明细。

反思循环（最多 MAX_REVISION_ROUNDS 轮）：
    gate 发现缺口 且 未超轮次  -> verdict = revise    -> 条件边回到 AnalystAgent 重算
    gate 仍有缺口 且 已超轮次  -> verdict = escalate  -> 转人工确认（需人工确认）
    无缺口 但整体风险等级 high -> verdict = escalate  -> 转人工确认
    其余                       -> verdict = pass      -> 走到 WriterAgent

为什么循环不会死循环：
    * 轮次计数 revision_round 由状态持有，每次 revise 递增；
    * 图本身还设了 recursion_limit 作为最后一道保险；
    * 超限后不是继续重算，而是降级为「需人工确认」，把不确定性交给人，而不是让模型
      无休止地自我说服。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..state import ResearchState, evidence_ids
from ..tools.registry import PermissionLevel
from .base import BaseAgent

#: 触发「必须做交叉验证」的异常指标规则
OUTLIER_RULES: List[Dict[str, Any]] = [
    {"key": "cash_conversion", "test": lambda v: v < 0.50,
     "reason": "净利润现金含量低于 0.50 倍，盈利的现金支撑可能不足"},
    {"key": "debt_to_asset", "test": lambda v: v > 0.65,
     "reason": "资产负债率高于 65%，杠杆水平需要结合趋势判断"},
    {"key": "revenue_growth", "test": lambda v: v < 0.0,
     "reason": "营业收入出现负增长，需要解释原因"},
    {"key": "gross_margin", "test": lambda v: v < 0.15,
     "reason": "毛利率低于 15%，需要判断是否可持续"},
    {"key": "current_ratio", "test": lambda v: v < 1.20,
     "reason": "流动比率低于 1.20，短期偿债能力需要交叉验证"},
]


class RiskCheckerAgent(BaseAgent):
    name = "risk_checker"
    role = "风险与合规核查、结论打回"
    allowed_tools = ("check_risk_rules",)
    permissions = frozenset({PermissionLevel.RESTRICTED_READ})

    def _execute(self, state: ResearchState) -> ResearchState:
        new_state = self._copy(state)
        plan = state.get("plan") or {}
        company = (plan.get("companies") or [""])[0]
        year = int(plan.get("year") or 0)
        period = str(plan.get("period") or "年度")
        metrics: Dict[str, Any] = dict(state.get("metrics") or {})
        findings: List[Dict[str, Any]] = list(state.get("findings") or [])
        revision_round = int(state.get("revision_round") or 0)
        max_rounds = int(state.get("max_revision_rounds") or self.ctx.config.max_revision_rounds)

        # ---- 1) 确定性核查门 ----
        numeric = {k: float(v["value"]) for k, v in metrics.items() if isinstance(v.get("value"), (int, float))}
        gaps = self._gate(state, findings, metrics, numeric)

        # ---- 2) 风险规则扫描（工具） ----
        risk_result = self.call_tool(
            "check_risk_rules",
            {
                "company": company,
                "year": year,
                "period": period,
                "extra_facts": numeric,
                "min_level": "low",
            },
            )
        risk_data: Dict[str, Any] = dict(risk_result.data or {}) if risk_result.ok else {}
        if not risk_result.ok:
            self._merge_errors(
                new_state, "risk_checker",
                [f"风险规则工具失败：{(risk_result.error or {}).get('message', '')}"],
            )
        overall = str(risk_data.get("overall_level") or "info")

        # ---- 3) 确定性裁决（合规闸门不委托给模型） ----
        verdict = self._decide(gaps, revision_round, max_rounds, overall)

        # ---- 4) LLM 生成核查叙述（与裁决交叉校验） ----
        payload = {
            "company": company,
            "year": year,
            "period": period,
            "gate": {
                "gaps": gaps,
                "round": revision_round,
                "max_rounds": max_rounds,
                "deterministic_verdict": verdict,
            },
            "risk": {
                "overall_level": overall,
                "triggered_count": risk_data.get("triggered_count", 0),
                "findings": risk_data.get("findings", []),
                "facts_used": risk_data.get("facts_used", {}),
            },
            "findings": [
                {
                    "id": f.get("id"),
                    "title": f.get("title"),
                    "statement": f.get("statement"),
                    "ratio_refs": f.get("ratio_refs"),
                    "evidence_ids": f.get("evidence_ids"),
                    "cross_checks": len(f.get("cross_checks") or []),
                }
                for f in findings
            ],
            "metrics": {k: v.get("display") for k, v in metrics.items()},
        }
        response = self.ctx.llm.chat("risk_review", payload)
        llm_data: Dict[str, Any] = dict(response.data or {})
        llm_verdict = str(llm_data.get("verdict") or verdict)

        # ---- 5) 写回状态 ----
        narrative = str(llm_data.get("narrative") or "")
        escalation_reason = str(llm_data.get("escalation_reason") or "")
        if not escalation_reason and verdict == "escalate":
            escalation_reason = "核查判定需要人工确认，但模型未给出原因，请人工复核风险明细。"

        # 追加本轮核查记录：最终状态只保留「当前缺口」，但历史必须留痕，
        # 否则"为什么被打回"这个信息会在循环收敛后丢失，事后无法复盘。
        history = list((state.get("risk_report") or {}).get("gate_history") or [])
        history.append(
            {
                "round": revision_round,
                "verdict": verdict,
                "risk_level": overall,
                "gaps": gaps,
                "narrative": narrative,
            }
        )

        new_state["risk_report"] = {
            **risk_data,
            "narrative": narrative,
            "escalation_reason": escalation_reason,
            "gaps": gaps,
            "gate_history": history,
            "gate_round": revision_round,
            "gate_max_rounds": max_rounds,
            "deterministic_verdict": verdict,
            "llm_verdict": llm_verdict,
            "verdict_override": llm_verdict != verdict,
        }
        new_state["risk_verdict"] = verdict
        new_state["risk_level"] = overall
        new_state["needs_human"] = verdict == "escalate"
        new_state["revision_gaps"] = gaps
        new_state["llm_stats"] = {**(state.get("llm_stats") or {}), "risk_checker": self._llm_stats_fragment(response)}

        if verdict == "revise":
            new_state["revision_requests"] = [str(g.get("required_fix", "")) for g in gaps if g.get("required_fix")]
            new_state["revision_round"] = revision_round + 1
        else:
            new_state["revision_requests"] = []
            new_state["revision_round"] = revision_round

        return new_state

    # ------------------------------------------------------------------
    @staticmethod
    def _decide(gaps: List[Dict[str, Any]], revision_round: int, max_rounds: int, overall: str) -> str:
        """确定性裁决。"""
        if gaps and revision_round < max_rounds:
            return "revise"
        if gaps and revision_round >= max_rounds:
            return "escalate"
        if overall == "high":
            return "escalate"
        return "pass"

    # ------------------------------------------------------------------
    def _gate(
        self,
        state: ResearchState,
        findings: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        numeric: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """确定性核查门：数据是否足够、结论是否被支撑。"""
        gaps: List[Dict[str, Any]] = []
        valid_evidence = evidence_ids(state)
        doc_store = self.ctx.document_store
        known_sources = set(doc_store.source_ids) if doc_store is not None else set()
        evidence_index = {str(e.get("child_id")): e for e in state.get("evidence", [])}

        if not findings:
            gaps.append(
                {
                    "code": "GAP-NO-FINDING",
                    "target": "-",
                    "problem": "AnalystAgent 未产出任何分析结论。",
                    "required_fix": "基于已检索证据重新计算指标并给出至少一条有量化支撑的结论。",
                    "severity": "blocker",
                }
            )
            return gaps

        for finding in findings:
            fid = str(finding.get("id"))
            evs = [str(e) for e in (finding.get("evidence_ids") or [])]

            # (1) 证据支撑
            if not evs:
                gaps.append(
                    {
                        "code": "GAP-EVIDENCE",
                        "target": fid,
                        "problem": f"结论「{finding.get('title')}」没有任何原文证据支撑。",
                        "required_fix": "为该结论补挂至少一条来源于资料库的证据（child_id）。",
                        "severity": "blocker",
                    }
                )
            elif not set(evs) & valid_evidence:
                gaps.append(
                    {
                        "code": "GAP-EVIDENCE",
                        "target": fid,
                        "problem": f"结论「{finding.get('title')}」引用的证据不在本次检索结果中。",
                        "required_fix": "重新检索并绑定真实存在的证据片段。",
                        "severity": "blocker",
                    }
                )

            # (2) 引用可回溯：证据必须来自资料库内的真实文档
            for eid in evs:
                src = str((evidence_index.get(eid) or {}).get("source_id", ""))
                if known_sources and src and src not in known_sources:
                    gaps.append(
                        {
                            "code": "GAP-TRACE",
                            "target": fid,
                            "problem": f"证据 {eid} 的资料编号 {src} 无法在资料库中回溯。",
                            "required_fix": "替换为可回溯的资料来源。",
                            "severity": "high",
                        }
                    )

            # (3) 量化支撑
            if not (finding.get("ratio_refs") or []):
                gaps.append(
                    {
                        "code": "GAP-METRIC",
                        "target": fid,
                        "problem": f"结论「{finding.get('title')}」没有任何量化指标支撑。",
                        "required_fix": "用 calc_ratio / get_financial_metric 补齐量化指标后再下结论。",
                        "severity": "high",
                    }
                )

        # (4) 异常指标必须做交叉验证（反思循环的主要触发点）
        cross_checked_keys = {
            str(ref)
            for finding in findings
            if (finding.get("cross_checks") or [])
            for ref in (finding.get("ratio_refs") or [])
        }
        for rule in OUTLIER_RULES:
            key = str(rule["key"])
            value: Optional[float] = numeric.get(key)
            if value is None:
                continue
            try:
                triggered = bool(rule["test"](value))
            except Exception:  # noqa: BLE001
                continue
            if not triggered:
                continue
            if key in cross_checked_keys:
                continue
            gaps.append(
                {
                    "code": "GAP-XCHECK",
                    "target": key,
                    "problem": f"指标 {key}={value:.4f} 属于异常值：{rule['reason']}，但结论未做交叉验证。",
                    "required_fix": (
                        f"针对 {key} 补充上期（上年同期）对比，并与相关科目（如应收账款增速、"
                        "经营活动现金流）做交叉印证后再下结论。"
                    ),
                    "severity": "medium",
                }
            )

        # (5) 收入质量：应收账款增速显著高于收入增速时，同样要求交叉验证
        ar_growth = numeric.get("receivable_growth")
        rev_growth = numeric.get("revenue_growth")
        if ar_growth is not None and rev_growth is not None and (ar_growth - rev_growth) > 0.10:
            if "cash_conversion" not in cross_checked_keys:
                gaps.append(
                    {
                        "code": "GAP-XCHECK",
                        "target": "cash_conversion",
                        "problem": (
                            f"应收账款增速 {ar_growth * 100:.2f}% 显著高于营业收入增速 "
                            f"{rev_growth * 100:.2f}%，收入质量存疑，但结论未做交叉验证。"
                        ),
                        "required_fix": "用应收账款增速与经营现金流对照，验证收入确认质量。",
                        "severity": "high",
                    }
                )

        return gaps

    # ------------------------------------------------------------------
    def _trace_input(self, state: ResearchState) -> Dict[str, Any]:
        return {
            "findings": [f.get("id") for f in (state.get("findings") or [])],
            "revision_round": state.get("revision_round", 0),
            "max_revision_rounds": state.get("max_revision_rounds"),
        }

    def _trace_output(self, state: ResearchState) -> Dict[str, Any]:
        report = state.get("risk_report") or {}
        return {
            "verdict": state.get("risk_verdict"),
            "risk_level": state.get("risk_level"),
            "gaps": report.get("gaps", []),
            "gate_history": [
                {"round": h.get("round"), "verdict": h.get("verdict"),
                 "gaps": [g.get("code") for g in (h.get("gaps") or [])]}
                for h in (report.get("gate_history") or [])
            ],
            "risk_findings": [
                {"rule_id": f.get("rule_id"), "level": f.get("level"), "title": f.get("title")}
                for f in (report.get("findings") or [])
            ],
            "narrative": report.get("narrative", ""),
            "escalation_reason": report.get("escalation_reason", ""),
            "verdict_override": report.get("verdict_override", False),
        }

    def _trace_extra(self, state: ResearchState) -> Dict[str, Any]:
        report = state.get("risk_report") or {}
        return {
            "verdict": state.get("risk_verdict"),
            "gaps_count": len(report.get("gaps") or []),
            "risk_level": state.get("risk_level"),
            "next_revision_round": state.get("revision_round"),
        }
