"""PlannerAgent —— 任务分解与路由。

层次与职责：
    位于 agents 层，是整张图的**入口节点**（``src/orchestrator.py`` 的 ``GraphSpec.entry``）。
    它不产出生意结论，只产出「往下怎么走」：候选公司、年份、报告期、分析维度（targets）、
    检索查询（retrieval_queries），以及决定分岔的 ``route``。

职责边界（刻意收窄）：
    * 只做「把问题拆成子任务」和「决定走哪些 Agent」，**不检索、不计算、不下结论**。
    * 不持有任何工具：它没有能力篡改数据，从结构上杜绝"规划者顺手把活干了"。
    * 输出 route 直接驱动编排层的条件边，是整张图的入口决策。

互斥点（与其它四个 Agent）：
    * vs RetrieverAgent：Planner 只**提出**查询，绝不执行检索；召回与筛选是 Retriever 的事。
    * vs AnalystAgent：Planner 不调用任何计算工具，也不产生 metrics / findings。
    * vs WriterAgent：Planner 不写报告、不碰引用编号。
    * vs RiskCheckerAgent：Planner 不做合规判断；反而是 RiskChecker 有权把 Analyst 打回。

它拿到的上下文是"资料库里有哪些公司 / 哪些文档 / 哪些分析维度"，而不是全量原文——
让规划器看到原文会诱导它提前下结论，这是从工程上防越权的做法。
    注：实际实现为 —— 送给 LLM 的 payload 里确实没有原文（只有 source_id / company /
    doc_type / year / 章节标题），但 ``_trace_output`` 会另外把 plan 摘要写进 trace，二者不要混淆。

对外关键类 / 函数：
    * ``PlannerAgent``（``name = "planner"``）—— 唯一对外类，实现 ``_execute``；
    * ``DEFAULT_ROUTE`` / ``VALID_ROUTE`` —— 路由白名单常量。

主要输入输出：
    输入：``state["question"]``（必需）、``state["plan"]``、``state["max_revision_rounds"]``
    等已有字段，以及 ``ctx.fact_store`` / ``ctx.document_store`` 提供的**元数据**。
    输出（写回新 state）：``plan`` / ``route`` / ``retrieval_queries`` /
    ``max_revision_rounds`` / ``revision_round`` / ``revision_requests`` / ``llm_stats``。

被谁调用：
    * ``src/orchestrator.py`` 把它注册为节点 ``"planner"``（图入口）；
    * ``tests/test_citation_traceability.py`` 直接实例化做单步测试。
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..state import ResearchState
from .base import BaseAgent

#: LLM 未给出 route（或给出的全是不合法项）时的兜底路由：完整四段流程。
DEFAULT_ROUTE = ["retriever", "analyst", "risk_checker", "writer"]
#: 路由白名单，同时也是**执行顺序**的权威定义（末尾用 VALID_ROUTE.index 排序）。
#: 不在其中的名字一律被过滤掉，防止 LLM 编造出图里不存在的节点导致运行期报错。
VALID_ROUTE = ["retriever", "analyst", "risk_checker", "writer"]


class PlannerAgent(BaseAgent):
    """任务分解与路由 Agent：图入口，只规划不执行。

    职责：
        把自然语言研究问题翻译成一份结构化执行计划（``plan``），并给出 ``route``。
        它是「多智能体」中负责**调度语义**的一环，而非执行语义。

    关键属性（类属性，见基类说明）：
        name = "planner" —— 写进 trace 与 steps 的 agent 名。
        role = "任务分解与路由"。
        allowed_tools = () —— **空元组**：一个工具都调不了，这是本类最重要的结构性约束。
        permissions = frozenset() —— 不持有任何权限等级。

    状态流转：
        入口态：state 里只有 question / run_id / config 等初始字段。
        出口态：写入 plan（含 companies / year / period / targets / route /
        retrieval_queries 等）与 route，并把反思循环所需的计数器
        （max_revision_rounds / revision_round / revision_requests）**初始化**好，
        供后续 Analyst <-> RiskChecker 往返使用。
        本 Agent 不会被任何条件边回指（拓扑上无环回到 planner），因此每个 run 只执行一次。
    """

    name = "planner"
    role = "任务分解与路由"
    allowed_tools = ()
    permissions = frozenset()

    def _execute(self, state: ResearchState) -> ResearchState:
        """产出 plan 与 route（本 Agent 的全部业务逻辑）。

        参数：
            state: 上游共享状态。只读使用 ``question`` / ``plan`` /
                ``max_revision_rounds`` / ``revision_round`` / ``revision_requests``。

        返回：
            浅拷贝后的新 state，额外写入：
                * ``plan`` —— 经**白名单过滤与兜底**后的结构化计划；
                * ``route`` —— 已排序、去重、且保证含 "writer" 的路由列表；
                * ``retrieval_queries`` —— 非空字符串查询列表；
                * ``max_revision_rounds`` / ``revision_round`` / ``revision_requests``
                  —— 反思循环的初始化（保留上游已有值，不覆盖）；
                * ``llm_stats["planner"]`` —— 本次 LLM 调用的精简统计。

        副作用 / 异常：
            * 调用一次 LLM（``task="plan"``）；若模型未返回合法 JSON，LLM 层会走
              mock / 降级路径，``response.data`` 可能为空字典，此时下面每个 ``data.get``
              的兜底分支都会生效——这正是「LLM 说了不算，契约说了算」的体现。
            * 除 LLM 外无外部副作用（不调工具、不写文件）。
            * 本方法不主动抛异常；读 fact_store / document_store 前都做了 None 判空。
        """
        new_state = self._copy(state)
        # 注：实际实现为 —— 这个局部变量被赋值后**从未被读取**（本方法后续统一用 `data`
        # 承载计划内容）。保留它不影响行为，仅作为「读入上游已有 plan」的痕迹。
        plan_state = state.get("plan") or {}
        question = state.get("question", "")

        # 只收集「元数据」而非原文：公司清单 + 最新年份 + 文档目录（章节标题级别）。
        # 目的是让规划器知道"有哪些料"，但不给它提前下结论的素材。
        known_companies: List[str] = []
        latest_year = 2024
        docs: List[Dict[str, Any]] = []

        store = self.ctx.fact_store
        doc_store = self.ctx.document_store
        if store is not None:
            known_companies = list(store.companies)
            # 把所有公司的所有年份摊平后取最大值，作为"最新年度"的默认值。
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
        # 公司必须落在事实库已知清单内；模型若报了个不存在的公司名，宁可退回列表第一家，
        # 也不允许把幻觉实体带进后续所有工具调用（否则检索/取指标必然全线失败）。
        companies = [c for c in (data.get("companies") or []) if c in known_companies] or known_companies[:1]
        # route 过滤：只保留图里真实存在的节点名。
        route = [r for r in (data.get("route") or DEFAULT_ROUTE) if r in VALID_ROUTE]
        # writer 是产物出口，任何路线都必须以它收尾；模型漏了就在这里补上。
        if "writer" not in route:
            route.append("writer")
        # 去重 + 按 VALID_ROUTE 的固定次序重排：保证同一问题多次运行得到同一条路径（可复现）。
        route = sorted(set(route), key=VALID_ROUTE.index)

        # 年份类型校验：LLM 常把年份返回成字符串，这里只接受 int，否则退回资料库最新年度。
        year = data.get("year")
        if not isinstance(year, int):
            year = latest_year

        # 报告期白名单：限定四种口径，避免模型自造"上半年""Q3"之类的非规范说法影响取数。
        period = str(data.get("period") or "年度").strip() or "年度"
        if period not in {"年度", "三季度", "半年度", "一季度"}:
            period = "年度"

        # 检索查询必须是非空字符串；一条都没有时用原问题兜底，保证 Retriever 至少有一路召回。
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
                # 分析维度兜底：覆盖盈利能力 / 偿债能力 / 风险合规，对应 analyst.py 的 RATIO_PLAN 键。
                "targets": data.get("targets") or ["盈利能力", "偿债能力", "风险合规"],
            }
        )

        new_state["plan"] = data
        new_state["route"] = route
        new_state["retrieval_queries"] = queries
        # 这三个字段是反思循环的"仪表盘"：上限来自配置，当前轮次从 0 起，
        # 打回意见列表初始为空。用 `or` 保留上游已有值，是为了兼容"从快照恢复后继续跑"的场景。
        new_state["max_revision_rounds"] = int(state.get("max_revision_rounds") or self.ctx.config.max_revision_rounds)
        new_state["revision_round"] = int(state.get("revision_round") or 0)
        new_state["revision_requests"] = list(state.get("revision_requests") or [])
        # 合并而非覆盖 llm_stats：同一 run 内多个 Agent 各自往里塞自己的调用统计。
        new_state["llm_stats"] = {**(state.get("llm_stats") or {}), "planner": self._llm_stats_fragment(response)}
        return new_state

    def _trace_input(self, state: ResearchState) -> Dict[str, Any]:
        """trace 输入摘要：只记原始问题。

        参数：state —— 进入本步时的状态。
        返回：``{"question": str}``。
        副作用 / 异常：无。
        """
        return {"question": state.get("question", "")}

    def _trace_output(self, state: ResearchState) -> Dict[str, Any]:
        """trace 输出摘要：记录计划的关键决策项。

        参数：state —— 本步执行后的状态（读 ``plan``）。
        返回：含 intent / companies / year / period / targets / route / subtasks /
            retrieval_queries 的字典。
            注：实际实现为 —— 其中 ``intent`` 与 ``subtasks`` 是**可选透传**字段：
            只有 LLM 在 plan 里显式返回它们时才有值，本模块的兜底逻辑不会构造这两个键，
            因此缺失时取到的是空字符串 / 空列表。
        副作用 / 异常：无；``plan`` 缺失时按空字典处理。
        """
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
