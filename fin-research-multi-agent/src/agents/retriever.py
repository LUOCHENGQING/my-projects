"""RetrieverAgent —— 多路检索与证据筛选。

职责边界：
    * 只持有 `search_filings` 一个工具，只做检索与筛选，**不做任何计算与结论**。
    * 对 Planner 给出的多路查询**逐一执行**（这就是"多路检索"），按 child_id 去重后
      取各路最高分，再交给 LLM 做证据筛选与覆盖度评估。
    * 输出带出处的证据片段（子块原文 + 父块上下文 + source_id + 章节标题），
      下游 Analyst 的每一条结论都必须挂在这里的 child_id 上。

防幻觉：LLM 选出的 child_id 会被强制与真实候选集合求交集，编造的 ID 一律丢弃。
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..state import ResearchState
from ..tools.registry import PermissionLevel
from .base import BaseAgent


class RetrieverAgent(BaseAgent):
    name = "retriever"
    role = "多路资料检索与证据筛选"
    allowed_tools = ("search_filings",)
    permissions = frozenset({PermissionLevel.PUBLIC_READ})

    #: 单路查询召回条数
    PER_QUERY_TOPK = 5
    #: 最终保留的证据条数上限
    MAX_EVIDENCE = 8

    def _execute(self, state: ResearchState) -> ResearchState:
        new_state = self._copy(state)
        plan = state.get("plan") or {}
        company = (plan.get("companies") or [""])[0]
        year = plan.get("year")
        targets = plan.get("targets") or []
        queries = state.get("retrieval_queries") or [state.get("question", "")]

        # ---- 1) 多路检索：每路查询一次，按 child_id 合并取最高分 ----
        merged: Dict[str, Dict[str, Any]] = {}
        executed: List[str] = []
        failures: List[str] = []

        for query in queries:
            result = self.call_tool(
                "search_filings",
                {
                    "query": query,
                    "top_k": self.PER_QUERY_TOPK,
                    "company": company or None,
                    "year": year if isinstance(year, int) else None,
                    "strict": False,
                },
            )
            if not result.ok:
                failures.append(f"{query} -> {(result.error or {}).get('message', '')}")
                continue
            executed.append(query)
            for item in (result.data or {}).get("results", []):
                cid = str(item.get("child_id"))
                current = merged.get(cid)
                if current is None or float(item.get("score") or 0.0) > float(current.get("score") or 0.0):
                    merged[cid] = item

        candidates = sorted(merged.values(), key=lambda x: -float(x.get("score") or 0.0))

        # ---- 2) LLM 做证据筛选与覆盖度评估 ----
        payload = {
            "question": state.get("question", ""),
            "company": company,
            "year": year,
            "targets": targets,
            "queries": executed,
            "limit": self.MAX_EVIDENCE,
            "candidates": [
                {
                    "child_id": item.get("child_id"),
                    "parent_id": item.get("parent_id"),
                    "source_id": item.get("source_id"),
                    "section_title": item.get("section_title"),
                    "score": item.get("score"),
                    "matched_terms": item.get("matched_terms"),
                    "text": item.get("text"),
                }
                for item in candidates[:20]
            ],
        }
        response = self.ctx.llm.chat("retrieve", payload)
        data: Dict[str, Any] = dict(response.data or {})

        # ---- 3) 反幻觉：选中的 ID 必须真实存在于候选中 ----
        valid_ids = {str(item.get("child_id")): item for item in candidates}
        selected_ids = [str(cid) for cid in (data.get("selected") or []) if str(cid) in valid_ids]
        if not selected_ids:
            selected_ids = [str(item.get("child_id")) for item in candidates[: self.MAX_EVIDENCE]]
        selected_ids = selected_ids[: self.MAX_EVIDENCE]

        evidence: List[Dict[str, Any]] = []
        for cid in selected_ids:
            item = dict(valid_ids[cid])
            item["child_id"] = cid
            evidence.append(item)

        new_state["evidence"] = evidence
        new_state["retrieval_meta"] = {
            "queries": executed,
            "query_failures": failures,
            "candidate_count": len(candidates),
            "selected_count": len(evidence),
            "coverage": data.get("coverage") or {},
            "relevance_notes": data.get("relevance_notes") or {},
            "missing_data": data.get("missing_data") or [],
        }
        new_state["llm_stats"] = {**(state.get("llm_stats") or {}), "retriever": self._llm_stats_fragment(response)}
        return new_state

    def _trace_input(self, state: ResearchState) -> Dict[str, Any]:
        plan = state.get("plan") or {}
        return {
            "queries": state.get("retrieval_queries", []),
            "company": (plan.get("companies") or [""])[0],
            "year": plan.get("year"),
        }

    def _trace_output(self, state: ResearchState) -> Dict[str, Any]:
        meta = state.get("retrieval_meta") or {}
        return {
            "selected": [e.get("child_id") for e in state.get("evidence", [])],
            "sources": sorted({str(e.get("source_id")) for e in state.get("evidence", [])}),
            "candidate_count": meta.get("candidate_count"),
            "coverage": meta.get("coverage"),
            "missing_data": meta.get("missing_data"),
        }
