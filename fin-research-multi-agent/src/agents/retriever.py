"""RetrieverAgent —— 多路检索与证据筛选。

层次与职责：
    位于 agents 层，是 planning 之后的第一段执行环节（图中的 ``retriever`` 节点，
    由 planner 的条件边决定是否进入；固定边 ``retriever -> analyst`` 把它交给 Analyst）。
    它把 Planner 写下的 ``retrieval_queries`` 变成一份**可追溯的证据清单** ``evidence``，
    这是整个系统「防幻觉」的地基：后面 Analyst 的每条结论都必须挂在这里的 child_id 上。

职责边界：
    * 只持有 `search_filings` 一个工具，只做检索与筛选，**不做任何计算与结论**。
    * 对 Planner 给出的多路查询**逐一执行**（这就是"多路检索"），按 child_id 去重后
      取各路最高分，再交给 LLM 做证据筛选与覆盖度评估。
    * 输出带出处的证据片段（子块原文 + 父块上下文 + source_id + 章节标题），
      下游 Analyst 的每一条结论都必须挂在这里的 child_id 上。
      注：实际实现为 —— 证据条目里**没有父块上下文**。``search_filings`` 返回的是
      ``RetrievedSnippet.to_dict()``，其键为 child_id / parent_id / source_id / company /
      period / year / section_title / text / score / components / matched_terms；
      数据类上虽有 ``context``（父块原文）字段，但 ``to_dict()`` 并未把它带出来，
      本 Agent 也只在调 LLM 时透传了其中一部分字段（不含 context / company / period / year）。

互斥点：
    * vs PlannerAgent：Planner 只写查询、不做检索；Retriever 只执行查询、不改 route。
    * vs AnalystAgent：Retriever **只召回不判断**——它不调用任何计算工具，不产生 metrics，
      也绝不对数据下结论；打分排序是检索相关度，不是财务评价。
    * vs RiskCheckerAgent：Retriever 不校验结论是否有证据（那是核查门的事），
      它只保证"证据池里的东西真实存在"。
    * vs WriterAgent：Retriever 不接触引用编号（cite_source 属 write 权限，它没有）。

防幻觉：LLM 选出的 child_id 会被强制与真实候选集合求交集，编造的 ID 一律丢弃。

对外关键类 / 函数：
    * ``RetrieverAgent``（``name = "retriever"``）—— 唯一对外类；
    * 类常量 ``PER_QUERY_TOPK`` / ``MAX_EVIDENCE`` —— 召回与保留条数的旋钮。

主要输入输出：
    输入：``state["retrieval_queries"]``（缺失时退回 ``state["question"]``）、
    ``state["plan"]`` 里的 companies / year / targets。
    输出（写回新 state）：``evidence``（证据列表）、``retrieval_meta``（执行过的查询、
    失败查询、候选数、入选数、覆盖度评估、相关性说明、缺失数据）、``llm_stats["retriever"]``。

被谁调用：
    * ``src/orchestrator.py`` 注册为节点 ``"retriever"``；
    * ``tests/test_citation_traceability.py`` 直接实例化做单步测试。
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..state import ResearchState
from ..tools.registry import PermissionLevel
from .base import BaseAgent


class RetrieverAgent(BaseAgent):
    """多路检索与证据筛选 Agent：只负责"找到并留下证据"，不负责"评价"。

    职责：
        1. 逐条执行 Planner 给出的检索查询（多路召回）；
        2. 按 child_id 归并去重，同一条证据只保留各路中的最高分；
        3. 请 LLM 在候选池中挑选证据并评估覆盖度；
        4. 用候选池对 LLM 的选择做交集校验，产出最终的 ``evidence``。

    关键属性（类属性）：
        name = "retriever" —— 写进 trace 与 steps。
        role = "多路资料检索与证据筛选"。
        allowed_tools = ("search_filings",) —— 只有这一个工具，结构上无法计算或写文件。
        permissions = {PermissionLevel.PUBLIC_READ} —— 只读公开资料，无 compute / write。
        PER_QUERY_TOPK = 5 —— 每路查询向工具要的条数（传给 search_filings 的 top_k）。
        MAX_EVIDENCE = 8 —— 最终 evidence 的长度上限。

    状态流转：
        入口态：plan / retrieval_queries 已由 Planner 写好，evidence 为空。
        出口态：evidence 与 retrieval_meta 被填充，供 Analyst 引用、RiskChecker 校验。
        本 Agent 处于固定边上（retriever -> analyst），不会被条件边回指，
        因此每次 run 最多执行一次（Analyst 被打回时只会回 Analyst，不会回到这里）。
    """

    name = "retriever"
    role = "多路资料检索与证据筛选"
    allowed_tools = ("search_filings",)
    permissions = frozenset({PermissionLevel.PUBLIC_READ})

    #: 单路查询召回条数（作为 search_filings 的 top_k 传入）
    PER_QUERY_TOPK = 5
    #: 最终保留的证据条数上限（同时作为送给 LLM 的 limit 提示与硬截断长度）
    MAX_EVIDENCE = 8

    def _execute(self, state: ResearchState) -> ResearchState:
        """执行多路检索 -> 归并去重 -> LLM 筛选 -> 反幻觉校验，产出 evidence。

        参数：
            state: 上游共享状态。读取 ``plan``（取第一家公司、年份、targets）、
                ``retrieval_queries``（缺失时退回 ``question``）、``question``、``llm_stats``。

        返回：
            新 state，写入：
                * ``evidence`` —— 最终证据列表（字典原样保留 search_filings 的字段，
                  并把 ``child_id`` 归一化为字符串）；
                * ``retrieval_meta`` —— 检索过程与质量元数据；
                * ``llm_stats["retriever"]`` —— 本次 LLM 调用的精简统计。

        副作用 / 异常：
            * 每条查询调用一次 ``search_filings`` 工具（有检索开销，但不改任何数据）；
              单路失败不会中断流程，而是记入 ``retrieval_meta["query_failures"]``；
            * 调用一次 LLM（``task="retrieve"``）；
            * 本方法不主动抛异常；``call_tool`` 越权时返回失败结果而非异常。
        """
        new_state = self._copy(state)
        plan = state.get("plan") or {}
        # 只取第一家公司：单次运行聚焦一家主体，多主体对比不在本轮设计范围内。
        company = (plan.get("companies") or [""])[0]
        year = plan.get("year")
        targets = plan.get("targets") or []
        # 兜底：Planner 没给查询时用原问题当唯一一路，保证"至少召回一次"。
        queries = state.get("retrieval_queries") or [state.get("question", "")]

        # ---- 1) 多路检索：每路查询一次，按 child_id 合并取最高分 ----
        merged: Dict[str, Dict[str, Any]] = {}
        executed: List[str] = []
        failures: List[str] = []

        for query in queries:
            # strict=False：元数据过滤只做"软约束"，命中不到就放宽，避免多路召回被元数据卡成空集。
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
                # 单路失败只登记不抛错：多路召回互为冗余，剩几路就够下游用，
                # 失败原因写进 meta 让人能从 trace 里复盘。
                failures.append(f"{query} -> {(result.error or {}).get('message', '')}")
                continue
            executed.append(query)
            for item in (result.data or {}).get("results", []):
                cid = str(item.get("child_id"))
                current = merged.get(cid)
                # 同一条证据被多路命中时保留最高分（并因此保留该路的 matched_terms 等上下文）。
                if current is None or float(item.get("score") or 0.0) > float(current.get("score") or 0.0):
                    merged[cid] = item

        # 按分数降序：LLM 与兜底截断都优先看到最相关的证据。
        candidates = sorted(merged.values(), key=lambda x: -float(x.get("score") or 0.0))

        # ---- 2) LLM 做证据筛选与覆盖度评估 ----
        # 只把候选的前 20 条交给模型（控制 prompt 体积）；这里的 20 是硬编码上限，
        # 不跟随 PER_QUERY_TOPK / MAX_EVIDENCE 变化。
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
        # 与 Planner 的 route 过滤同理：模型输出只是"建议"，必须落在真实集合内才生效。
        valid_ids = {str(item.get("child_id")): item for item in candidates}
        selected_ids = [str(cid) for cid in (data.get("selected") or []) if str(cid) in valid_ids]
        if not selected_ids:
            # 模型全选了幻觉 ID（或压根没选）时的兜底：按分数取前 MAX_EVIDENCE 条，
            # 宁可给出"分数最高的一批"也不让证据池为空——空证据会让下游核查门直接不通过。
            selected_ids = [str(item.get("child_id")) for item in candidates[: self.MAX_EVIDENCE]]
        selected_ids = selected_ids[: self.MAX_EVIDENCE]

        evidence: List[Dict[str, Any]] = []
        for cid in selected_ids:
            item = dict(valid_ids[cid])
            item["child_id"] = cid  # 归一化为字符串，保证下游按 str 索引（evidence_ids）时能对上
            evidence.append(item)

        new_state["evidence"] = evidence
        # retrieval_meta 是"检索质量"的观测面：覆盖度与缺失数据不参与裁决，
        # 但会随 trace 一起留痕，用于人工确认时判断"是没查到还是真没有"。
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
        """trace 输入摘要：记录待执行的查询与检索约束。

        参数：state —— 进入本步时的状态，读 ``retrieval_queries`` 与 ``plan``。
        返回：``{"queries": [...], "company": str, "year": int|None}``；
            ``plan`` 缺失时 company 取空字符串。
        副作用 / 异常：无。
        """
        plan = state.get("plan") or {}
        return {
            "queries": state.get("retrieval_queries", []),
            "company": (plan.get("companies") or [""])[0],
            "year": plan.get("year"),
        }

    def _trace_output(self, state: ResearchState) -> Dict[str, Any]:
        """trace 输出摘要：记录入选证据 ID、来源集合与召回质量。

        参数：state —— 本步执行后的状态，读 ``evidence`` 与 ``retrieval_meta``。
        返回：``{"selected": [child_id...], "sources": [source_id...]（去重升序）,
            "candidate_count": int|None, "coverage": dict|None, "missing_data": list|None}``。
        副作用 / 异常：无；对 ``e.get("source_id")`` 为 None 的情况会转成字符串 "None" 参与去重，
            这是既有实现的既有行为，仅在注释中说明、不作改动。
        """
        meta = state.get("retrieval_meta") or {}
        return {
            "selected": [e.get("child_id") for e in state.get("evidence", [])],
            "sources": sorted({str(e.get("source_id")) for e in state.get("evidence", [])}),
            "candidate_count": meta.get("candidate_count"),
            "coverage": meta.get("coverage"),
            "missing_data": meta.get("missing_data"),
        }
