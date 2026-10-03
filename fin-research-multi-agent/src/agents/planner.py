"""PlannerAgent —— 任务分解与路由。

职责边界（刻意收窄）：
    * 只做「把问题拆成子任务」和「决定走哪些 Agent」，**不检索、不计算、不下结论**。
    * 不持有任何工具：它没有能力篡改数据，从结构上杜绝"规划者顺手把活干了"。
    * 输出 route 直接驱动编排层的条件边，是整张图的入口决策。

它拿到的上下文是"资料库里有哪些公司 / 哪些文档 / 哪些分析维度"，而不是全量原文——
让规划器看到原文会诱导它提前下结论，这是从工程上防越权的做法。
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..state import ResearchState
from .base import BaseAgent

DEFAULT_ROUTE = ["retriever", "analyst", "risk_checker", "writer"]
VALID_ROUTE = ["retriever", "analyst", "risk_checker", "writer"]


class PlannerAgent(BaseAgent):
    name = "planner"
    role = "任务分解与路由"
    allowed_tools = ()
    permissions = frozenset()

    def _execute(self, state: ResearchState) -> ResearchState:
        new_state = self._copy(state)
        plan_state = state.get("plan") or {}
        question = state.get("question", "")

        known_companies: List[str] = []
        latest_year = 2024
        docs: List[Dict[str, Any]] = []

        store = self.ctx.fact_store
        doc_store = self.ctx.document_store
        if store is not None:
            known_companies = list(store.companies)
            years = [y for c in known_companies for y in store.years(c)]
            if years:
                latest_year = max(years)
        if doc_store is not None:
            docs = [
                {
                    "source_id": d.source_id,
                    "company": d.company,
                    "doc_type": d.doc_type,
                    "year": d.year,
                    "sections": [s.title for s in d.sections],
                }
                for d in doc_store
            ]

        payload = {
            "question": question,
            "companies": known_companies,
            "latest_year": latest_year,
            "documents": docs,
        }
        response = self.ctx.llm.chat("plan", payload)
        data: Dict[str, Any] = dict(response.data or {})

        # ---- 输出校验与兜底（LLM 可能返回不合规的 route） ----
        companies = [c for c in (data.get("companies") or []) if c in known_companies] or known_companies[:1]
        route = [r for r in (data.get("route") or DEFAULT_ROUTE) if r in VALID_ROUTE]
        if "writer" not in route:
            route.append("writer")
        route = sorted(set(route), key=VALID_ROUTE.index)

        year = data.get("year")
        if not isinstance(year, int):
            year = latest_year

        period = str(data.get("period") or "年度").strip() or "年度"
        if period not in {"年度", "三季度", "半年度", "一季度"}:
            period = "年度"

        queries = [q for q in (data.get("retrieval_queries") or []) if isinstance(q, str) and q.strip()]
        if not queries:
            queries = [question]

        data.update(
            {
                "companies": companies,
                "year": year,
                "period": period,
                "route": route,
                "retrieval_queries": queries,
                "targets": data.get("targets") or ["盈利能力", "偿债能力", "风险合规"],
            }
        )

        new_state["plan"] = data
        new_state["route"] = route
        new_state["retrieval_queries"] = queries
        new_state["max_revision_rounds"] = int(state.get("max_revision_rounds") or self.ctx.config.max_revision_rounds)
        new_state["revision_round"] = int(state.get("revision_round") or 0)
        new_state["revision_requests"] = list(state.get("revision_requests") or [])
        new_state["llm_stats"] = {**(state.get("llm_stats") or {}), "planner": self._llm_stats_fragment(response)}
        return new_state

    def _trace_input(self, state: ResearchState) -> Dict[str, Any]:
        return {"question": state.get("question", "")}

    def _trace_output(self, state: ResearchState) -> Dict[str, Any]:
        plan = state.get("plan") or {}
        return {
            "intent": plan.get("intent", ""),
            "companies": plan.get("companies", []),
            "year": plan.get("year"),
            "period": plan.get("period"),
            "targets": plan.get("targets", []),
            "route": plan.get("route", []),
            "subtasks": plan.get("subtasks", []),
            "retrieval_queries": plan.get("retrieval_queries", []),
        }
