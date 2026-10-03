"""Blackboard 共享状态定义。

多智能体协作的核心是**共享状态（blackboard）**：所有 Agent 读写同一份状态，
通过状态而非直接函数调用来传递信息与触发下一步。这样带来三个好处：
    1. 可观测：任何时刻的状态快照就是「系统当前知道什么」的完整答案；
    2. 可回放：状态 + 条件边函数 = 完全可复现的执行路径；
    3. 可测试：单测可以直接构造状态、调用单个 Agent，不需要跑整张图。

字段分组（用注释划清边界，避免 Agent 之间"偷偷"耦合）：
    [输入]      run_id / question / config
    [Planner]   plan / route / retrieval_queries / max_revision_rounds
    [Retriever] evidence / retrieval_meta
    [Analyst]   facts / metrics / findings / analysis_summary
    [Risk]      risk_report / risk_verdict / revision_round / revision_requests / needs_human
    [HITL]      human_decision
    [Writer]    citations / citation_index / report
    [观测]      steps / errors / llm_stats
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # pragma: no cover - 仅用于类型标注，运行时不强依赖
    from typing import TypedDict
except ImportError:  # pragma: no cover
    TypedDict = dict  # type: ignore[assignment,misc]


class ResearchState(TypedDict, total=False):
    """投研任务的共享状态（LangGraph 的 State 通道定义）。"""

    # ------------- [输入] -------------
    run_id: str
    question: str
    started_at: str
    config: Dict[str, Any]

    # ------------- [PlannerAgent] -------------
    plan: Dict[str, Any]
    route: List[str]
    retrieval_queries: List[str]
    max_revision_rounds: int

    # ------------- [RetrieverAgent] -------------
    evidence: List[Dict[str, Any]]
    retrieval_meta: Dict[str, Any]

    # ------------- [AnalystAgent] -------------
    facts: Dict[str, Any]          # 原始指标：{"营业收入": {"value":..., "source_id":...}}
    metrics: Dict[str, Any]        # 计算得到的比率：{"net_margin": {...}}
    findings: List[Dict[str, Any]]  # 分析结论
    analysis_summary: str

    # ------------- [RiskCheckerAgent] -------------
    risk_report: Dict[str, Any]
    risk_verdict: str              # pass / revise / escalate
    revision_round: int
    revision_requests: List[str]   # 打回时给 Analyst 的具体整改要求
    revision_gaps: List[Dict[str, Any]]  # 打回意见的结构化原文（供人工确认界面展示）
    needs_human: bool
    risk_level: str

    # ------------- [HITL] -------------
    human_decision: Optional[Dict[str, Any]]

    # ------------- [WriterAgent] -------------
    citations: List[Dict[str, Any]]
    citation_index: Dict[str, int]
    report: str

    # ------------- [观测] -------------
    steps: List[Dict[str, Any]]
    errors: List[str]
    llm_stats: Dict[str, Any]


#: 状态字段清单，供测试断言「状态契约」是否被破坏
STATE_FIELDS: List[str] = list(ResearchState.__annotations__.keys())


def new_state(
    question: str,
    run_id: str,
    *,
    started_at: str = "",
    config: Optional[Dict[str, Any]] = None,
    max_revision_rounds: int = 2,
) -> ResearchState:
    """构造一次运行的初始黑board状态。"""
    return ResearchState(
        run_id=run_id,
        question=question,
        started_at=started_at,
        config=config or {},
        plan={},
        route=[],
        retrieval_queries=[],
        max_revision_rounds=max_revision_rounds,
        evidence=[],
        retrieval_meta={},
        facts={},
        metrics={},
        findings=[],
        analysis_summary="",
        risk_report={},
        risk_verdict="",
        revision_round=0,
        revision_requests=[],
        revision_gaps=[],
        needs_human=False,
        risk_level="info",
        human_decision=None,
        citations=[],
        citation_index={},
        report="",
        steps=[],
        errors=[],
        llm_stats={},
    )


def evidence_by_id(state: ResearchState) -> Dict[str, Dict[str, Any]]:
    """证据索引：child_id -> 证据片段。"""
    return {str(e.get("child_id")): e for e in state.get("evidence", [])}


def evidence_ids(state: ResearchState) -> set:
    return set(evidence_by_id(state))


def snapshot(state: ResearchState) -> Dict[str, Any]:
    """生成用于 trace 的状态摘要（只保留规模信息，不落全量文本）。"""
    return {
        "plan_companies": (state.get("plan") or {}).get("companies", []),
        "route": state.get("route", []),
        "evidence_count": len(state.get("evidence", [])),
        "metrics_count": len(state.get("metrics", {})),
        "findings_count": len(state.get("findings", [])),
        "revision_round": state.get("revision_round", 0),
        "risk_verdict": state.get("risk_verdict", ""),
        "risk_level": state.get("risk_level", ""),
        "needs_human": state.get("needs_human", False),
        "citations_count": len(state.get("citations", [])),
        "report_chars": len(state.get("report", "") or ""),
    }
