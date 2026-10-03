"""WriterAgent —— 结构化投研简报生成。

职责边界：
    * 唯一持有 `cite_source`（write 权限）的 Agent：只有它能把资料编号解析成正式引用编号。
      这也意味着引用编号的分配是**单一职责**的，不会出现多个 Agent 各自编号导致错乱。
    * 报告骨架（章节、表格、引用清单）由 Agent 确定性地拼装，
      LLM 只负责核心结论与论证段落的语言组织。
      这样做的好处：无论模型怎么发挥，**引用编号格式与可追溯性都不会被破坏**。

引用可追溯的三段式：
    finding.evidence_ids -> evidence.source_id -> cite_source 分配的引用编号 [n]
    最终报告里的每个 [n] 都能在「五、引用来源」里找到对应的 source_id 与文件路径。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..state import ResearchState
from ..tools.registry import PermissionLevel
from .base import BaseAgent


def _fmt_num(value: Any) -> str:
    """数值格式化：大数加千分位，小数保留合理位数。"""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        if abs(value) >= 1:
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


class WriterAgent(BaseAgent):
    name = "writer"
    role = "结构化投研简报撰写与引用绑定"
    allowed_tools = ("cite_source",)
    permissions = frozenset({PermissionLevel.WRITE})

    def _execute(self, state: ResearchState) -> ResearchState:
        new_state = self._copy(state)
        plan = state.get("plan") or {}
        companies = plan.get("companies") or [""]
        company = companies[0] if companies else ""
        year = int(plan.get("year") or 0)
        period = str(plan.get("period") or "年度")
        findings: List[Dict[str, Any]] = list(state.get("findings") or [])
        evidence: List[Dict[str, Any]] = list(state.get("evidence") or [])
        risk_report: Dict[str, Any] = dict(state.get("risk_report") or {})

        evidence_index = {str(e.get("child_id")): e for e in evidence}

        # ---- 1) 为所有被引用的资料分配引用编号（write 权限工具） ----
        # 排序策略：被结论引用次数最多的资料优先拿到小编号（主要来源 = [1]），
        # 次数相同时按首次出现顺序，保证结果确定、可复现。
        usage: Dict[str, int] = {}
        first_seen: Dict[str, int] = {}
        order = 0
        for finding in findings:
            for eid in finding.get("evidence_ids") or []:
                src = str((evidence_index.get(str(eid)) or {}).get("source_id", ""))
                if not src:
                    continue
                usage[src] = usage.get(src, 0) + 1
                if src not in first_seen:
                    first_seen[src] = order
                    order += 1
        for item in evidence:
            src = str(item.get("source_id", ""))
            if src and src not in first_seen:
                first_seen[src] = order
                order += 1
                usage.setdefault(src, 0)

        used_source_ids: List[str] = sorted(first_seen, key=lambda s: (-usage.get(s, 0), first_seen[s]))

        citations: List[Dict[str, Any]] = []
        citation_index: Dict[str, int] = {}
        citation_failures: List[str] = []
        for source_id in used_source_ids:
            section_keyword = next(
                (str(e.get("section_title")) for e in evidence if str(e.get("source_id")) == source_id),
                "",
            )
            result = self.call_tool(
                "cite_source",
                {"source_id": source_id, "section_keyword": section_keyword or None},
            )
            if not result.ok:
                citation_failures.append(f"{source_id}: {(result.error or {}).get('message', '')}")
                continue
            data = dict(result.data or {})
            citation_index[source_id] = int(data.get("citation_no"))
            citations.append(data)

        # ---- 2) 结论 -> 引用编号映射 ----
        citation_map: Dict[str, List[int]] = {}
        for finding in findings:
            nos: List[int] = []
            for eid in finding.get("evidence_ids") or []:
                src = str((evidence_index.get(str(eid)) or {}).get("source_id", ""))
                no = citation_index.get(src)
                if no is not None and no not in nos:
                    nos.append(no)
            if not nos and citations:
                nos = [citations[0]["citation_no"]]
            citation_map[str(finding.get("id"))] = sorted(nos)

        # ---- 3) LLM 组织语言 ----
        payload = {
            "question": state.get("question", ""),
            "company": company,
            "year": year,
            "period": period,
            "findings": findings,
            "citation_map": citation_map,
            "citations": citations,
            "risk": risk_report,
            "analysis_summary": state.get("analysis_summary", ""),
            "human_decision": state.get("human_decision"),
        }
        response = self.ctx.llm.chat("write", payload)
        data: Dict[str, Any] = dict(response.data or {})

        # ---- 4) 确定性拼装报告 ----
        report = self._assemble(state, data, findings, citation_map, citations, company, year, period, risk_report)

        new_state["citations"] = citations
        new_state["citation_index"] = citation_index
        new_state["report"] = report
        new_state["llm_stats"] = {**(state.get("llm_stats") or {}), "writer": self._llm_stats_fragment(response)}
        self._merge_errors(new_state, "writer", citation_failures)
        return new_state

    # ------------------------------------------------------------------
    def _assemble(
        self,
        state: ResearchState,
        llm_data: Dict[str, Any],
        findings: List[Dict[str, Any]],
        citation_map: Dict[str, List[int]],
        citations: List[Dict[str, Any]],
        company: str,
        year: int,
        period: str,
        risk_report: Dict[str, Any],
    ) -> str:
        title = str(llm_data.get("title") or f"{company} {year} 年{period}投研简报")
        mode = (state.get("config") or {}).get("mock_llm", True)
        mode_desc = "确定性 mock 大脑（离线模式）" if mode else str((state.get("config") or {}).get("model_name", "LLM"))
        risk_level = str(risk_report.get("overall_level") or state.get("risk_level") or "info")
        human_decision = state.get("human_decision") or {}

        lines: List[str] = []
        lines.append(f"# {title}")
        lines.append("")
        counter = _SectionCounter()
        lines.append(f"> **研究问题**：{state.get('question', '')}")
        lines.append(
            f"> **运行编号**：{state.get('run_id', '')} ｜ **生成方式**：{mode_desc} ｜ **整体风险等级**：{risk_level.upper()}"
        )
        lines.append(
            "> **数据来源**：本地演示资料库（公司名称、财务数据与事件均为虚构，"
            "仅用于技术演示，不构成任何投资建议）"
        )
        if human_decision:
            lines.append(
                f"> **人工确认**：{human_decision.get('decision', '')}"
                f"（来源：{human_decision.get('source', '')}；说明：{human_decision.get('reason', '')}）"
            )
        lines.append("")

        # ---- 一、核心结论 ----
        lines.append(f"## {counter.next()}、核心结论")
        lines.append("")
        executive: List[str] = list(llm_data.get("executive_summary") or [])
        if not executive:
            executive = [
                f"{f.get('statement', '')}{''.join(f'[{n}]' for n in citation_map.get(str(f.get('id')), []))}"
                for f in findings
            ]
        for idx, item in enumerate(executive, 1):
            text = str(item).strip()
            if not _has_citation(text):
                fallback = citation_map.get(str(findings[0].get("id")) if findings else "", [])
                text = text + "".join(f"[{n}]" for n in fallback[:2])
            lines.append(f"{idx}. {text}")
        lines.append("")

        # ---- 二、关键财务指标 ----
        lines.append(f"## {counter.next()}、关键财务指标")
        lines.append("")
        lines.append("| 指标 | 数值 | 单位 | 出处 |")
        lines.append("| --- | --- | --- | --- |")
        rows = self._fact_rows(state.get("facts") or {})
        if rows:
            for metric, value, unit, source_id in rows:
                lines.append(f"| {metric} | {_fmt_num(value)} | {unit} | {source_id} |")
        else:
            lines.append("| （无） | - | - | - |")
        lines.append("")

        # ---- 三、关键比率 ----
        metrics: Dict[str, Any] = state.get("metrics") or {}
        if metrics:
            lines.append(f"## {counter.next()}、关键比率指标")
            lines.append("")
            lines.append("| 比率 | 数值 | 计算口径 | 出处 |")
            lines.append("| --- | --- | --- | --- |")
            for key, item in metrics.items():
                label = item.get("label") or key
                display = item.get("display") or _fmt_num(item.get("value"))
                lines.append(
                    f"| {label} | {display} | {item.get('formula', '')} | {item.get('source_id', '') or '-'} |"
                )
            lines.append("")

        # ---- 分析与论证 ----
        lines.append(f"## {counter.next()}、分析与论证")
        lines.append("")
        paragraphs: Dict[str, str] = dict(llm_data.get("analysis_paragraphs") or {})
        for finding in findings:
            fid = str(finding.get("id"))
            lines.append(f"### {fid} {finding.get('title', '')}")
            lines.append("")
            text = str(paragraphs.get(fid) or finding.get("statement") or "").strip()
            if not _has_citation(text):
                text += "".join(f"[{n}]" for n in citation_map.get(fid, [])[:2])
            lines.append(text)
            lines.append("")
            for extra in finding.get("cross_checks") or []:
                lines.append(f"- 交叉验证：{extra}")
            if finding.get("cross_checks"):
                lines.append("")

        # ---- 风险提示 ----
        lines.append(f"## {counter.next()}、风险提示")
        lines.append("")
        narrative = str(risk_report.get("narrative") or "").strip()
        if narrative:
            lines.append(narrative)
            lines.append("")
        risk_findings = risk_report.get("findings") or []
        if risk_findings:
            lines.append("| 等级 | 风险项 | 指标值 | 触发阈值 | 说明 | 出处 |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for item in risk_findings:
                lines.append(
                    f"| {str(item.get('level', '')).upper()} | {item.get('title', '')} | "
                    f"{item.get('metric_display', '')} | {item.get('threshold', '')} | "
                    f"{item.get('detail', '')} | {item.get('source_id', '') or '-'} |"
                )
            lines.append("")
        gaps = risk_report.get("gaps") or []
        if gaps:
            lines.append("**未消除的数据缺口（需人工确认）**")
            lines.append("")
            for gap in gaps:
                lines.append(f"- `{gap.get('code', '')}` {gap.get('problem', '')} → 要求补正：{gap.get('required_fix', '')}")
            lines.append("")

        # ---- 引用来源 ----
        lines.append(f"## {counter.next()}、引用来源")
        lines.append("")
        if citations:
            for item in sorted(citations, key=lambda x: int(x.get("citation_no", 0))):
                lines.append(
                    f"[{item.get('citation_no')}] {item.get('company', '')} {item.get('period', '')} · "
                    f"{item.get('section', '')} —— `{item.get('source_id', '')}`"
                    f"（{item.get('doc_type', '')}，文件：{item.get('path', '')}）"
                )
        else:
            lines.append("（本次运行未产生引用）")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append(
            "> 免责声明：本简报由多智能体系统自动生成，资料库内容为虚构演示数据，"
            "不构成任何投资建议；请勿据此做出投资决策。"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    @staticmethod
    def _fact_rows(facts: Dict[str, Any]) -> List[Tuple[str, Any, str, str]]:
        """把事实缓存整理成表格行（同一年度的同一指标只保留一条）。"""
        seen = set()
        rows: List[Tuple[str, Any, str, str]] = []
        for key in sorted(facts.keys()):
            item = facts.get(key)
            if not isinstance(item, dict):
                continue
            metric = str(item.get("metric") or "")
            year = item.get("year")
            dedup = (metric, year, item.get("period"))
            if not metric or dedup in seen:
                continue
            seen.add(dedup)
            rows.append(
                (
                    f"{metric}（{item.get('period', year)}）",
                    item.get("value"),
                    str(item.get("unit") or ""),
                    str(item.get("source_id") or "-"),
                )
            )
        return rows[:24]

    # ------------------------------------------------------------------
    def _trace_input(self, state: ResearchState) -> Dict[str, Any]:
        return {
            "findings": [f.get("id") for f in (state.get("findings") or [])],
            "evidence": [e.get("child_id") for e in (state.get("evidence") or [])],
            "risk_verdict": state.get("risk_verdict"),
        }

    def _trace_output(self, state: ResearchState) -> Dict[str, Any]:
        return {
            "citations": [
                {"no": c.get("citation_no"), "source_id": c.get("source_id"), "section": c.get("section")}
                for c in (state.get("citations") or [])
            ],
            "report_chars": len(state.get("report") or ""),
            "report_head": (state.get("report") or "")[:400],
        }


def _has_citation(text: str) -> bool:
    """判断一段文本里是否已经包含 [n] 形式的引用编号。"""
    import re

    return bool(re.search(r"\[\d+\]", text or ""))


_CN_NUM = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]


class _SectionCounter:
    """章节序号计数器，保证章节编号连续且不会因为可选章节而错位。"""

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> str:
        label = _CN_NUM[self._n] if self._n < len(_CN_NUM) else str(self._n + 1)
        self._n += 1
        return label
