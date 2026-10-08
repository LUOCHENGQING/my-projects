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
    [Risk]      risk_report / risk_verdict / revision_round / revision_requests /
                revision_gaps / needs_human / risk_level
    [HITL]      human_decision
    [Writer]    citations / citation_index / report
    [观测]      steps / errors / llm_stats

架构层次与职责：
    本模块属于最底层的「领域模型层」，只定义共享状态的**形状**与若干纯函数工具，
    不含任何流程控制逻辑（流程控制全部在 orchestrator + 各 Agent 里）。它同时承担两个角色：
        1. LangGraph 的 `state_schema`（见 orchestrator.build_engine / _build_spec）；
        2. 自研引擎里 dict blackboard 的书面契约（字段清单可被测试直接断言）。
    换句话说：本模块回答「系统当前知道什么」，不回答「下一步做什么」。

对外关键对象：
    ResearchState            共享状态 schema（TypedDict，total=False）
    STATE_FIELDS             字段清单，测试用它断言「状态契约」未被破坏
    new_state()              构造初始状态（把全部字段显式填上初值）
    evidence_by_id()         证据索引：child_id -> 证据片段
    evidence_ids()           证据 ID 集合（供 RiskChecker 校验引用真实性）
    snapshot()               生成 trace 用的轻量规模摘要

主要输入输出：
    输入：new_state(question, run_id, ...) 的运行参数；运行期各 Agent 直接读写状态字典。
    输出：一份贯穿全流程、可 JSON 化的状态字典（blackboard）；snapshot() 输出小体积摘要。

被谁调用：
    * orchestrator：ResearchPipeline.run() 用 new_state() 初始化，_build_spec() 把
      ResearchState 作为图的状态 schema，_summary() 用 snapshot() 写进 trace 汇总行；
    * agents/base.py：BaseAgent.run() 的入参与返回值都是 ResearchState（模板方法负责落 trace）；
    * agents/risk_checker.py：用 evidence_ids() 判断结论引用的 child_id 是否真实存在；
    * tests/test_state_flow.py：用 STATE_FIELDS / new_state() 断言状态契约与流转顺序。

状态流转（粗粒度，字段填充顺序 == 流程顺序）：
    planner 写 plan/route/retrieval_queries -> retriever 写 evidence/retrieval_meta ->
    analyst 写 facts/metrics/findings/analysis_summary -> risk_checker 写 risk_report/
    risk_verdict/revision_round（revise 时回到 analyst 重算）-> 仅在 escalate 时
    human_review 写 human_decision -> writer 写 citations/citation_index/report。
    每一步都会由 BaseAgent.run() 向 steps 追加一条留痕。

状态膨胀的规避（为什么状态里看不到长文本）：
    * evidence 只存「命中的子块」及其定位元数据（child_id / parent_id / source_id /
      section_title / score / matched_terms），条数由 RetrieverAgent.MAX_EVIDENCE 截断。
      注：实际实现为 RetrievedSnippet.to_dict()（src/rag/retriever.py）并不输出 context 字段，
      因此回填的父块正文不会进入共享状态，父块只以 parent_id 标识、需要时再回查检索器——
      这正是「只保留命中子块与必要父块」的落地方式。
    * snapshot() 只落 count / chars 这类规模信息，不落全量文本。
    * risk_checker 只把每轮缺口的留痕放进 risk_report.gate_history，当前缺口单独放
      revision_gaps，避免整份历史被反复复制进每一步的状态快照。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # pragma: no cover - 仅用于类型标注，运行时不强依赖
    from typing import TypedDict
except ImportError:  # pragma: no cover
    TypedDict = dict  # type: ignore[assignment,misc]


class ResearchState(TypedDict, total=False):
    """投研任务的共享状态（LangGraph 的 State 通道定义）。

    职责：定义所有 Agent 共享读写的那一份 blackboard 的字段与类型。Agent 之间不直接
    互相调用，只通过往这份状态里写字段来驱动下一步——状态即通信媒介。

    total=False 的含义：所有键都是可选的，因此「LangGraph 的增量更新（节点只返回自己
    改动的键）」与「自研引擎的整体覆盖」两种语义可以共用同一份 schema；而 new_state()
    会显式把全部字段填上初值，保证「初始状态字段齐全」这条契约成立。

    关键属性（按写入者分组，见下方分区注释）：输入 3 项、Planner 4 项、Retriever 2 项、
    Analyst 4 项、Risk 7 项、HITL 1 项、Writer 3 项、观测 3 项。

    状态流转：字段的填充顺序就是流程顺序 —— planner -> retriever -> analyst ->
    risk_checker（verdict 为 revise 则回到 analyst 重算）-> human_review（仅 escalate 时）
    -> writer；每一步都会由 BaseAgent.run() 向 steps 追加留痕。

    注：本类只有类型声明、没有行为；字段级语义见下方每个分区的行内注释。
    """

    # ------------- [输入] -------------
    run_id: str
    question: str
    started_at: str
    config: Dict[str, Any]

    # ------------- [PlannerAgent] -------------
    plan: Dict[str, Any]
    route: List[str]                 # Planner 给出的 Agent 顺序，条件边按它挑第一个下游
    retrieval_queries: List[str]     # 拆出的多路检索查询，由 RetrieverAgent 逐一执行
    max_revision_rounds: int         # 反思循环轮次上限（来自 config.MAX_REVISION_ROUNDS）

    # ------------- [RetrieverAgent] -------------
    evidence: List[Dict[str, Any]]   # 只保留命中的子块 + 定位元数据，见模块 docstring
    retrieval_meta: Dict[str, Any]

    # ------------- [AnalystAgent] -------------
    facts: Dict[str, Any]          # 原始指标：{"营业收入": {"value":..., "source_id":...}}
    metrics: Dict[str, Any]        # 计算得到的比率：{"net_margin": {...}}
    findings: List[Dict[str, Any]]  # 分析结论
    analysis_summary: str

    # ------------- [RiskCheckerAgent] -------------
    risk_report: Dict[str, Any]    # findings / gaps / gate_history / escalation_reason 等
    risk_verdict: str              # pass / revise / escalate
    revision_round: int            # 已发生的重算轮次，每次 revise 递增（防死循环的三重保险之一）
    revision_requests: List[str]   # 打回时给 Analyst 的具体整改要求
    revision_gaps: List[Dict[str, Any]]  # 打回意见的结构化原文（供人工确认界面展示）
    needs_human: bool              # risk_verdict == "escalate" 时为 True
    risk_level: str                # info / low / medium / high

    # ------------- [HITL] -------------
    human_decision: Optional[Dict[str, Any]]  # None 表示未触发人工确认

    # ------------- [WriterAgent] -------------
    citations: List[Dict[str, Any]]
    citation_index: Dict[str, int]
    report: str                    # 带 [n] 引用的简报；被 HITL 驳回时为空串

    # ------------- [观测] -------------
    steps: List[Dict[str, Any]]    # 每步留痕 {step, agent, status, latency_ms, tool_calls}
    errors: List[str]              # 单步异常被 BaseAgent.run() 兜住后记在这里，不炸整张图
    llm_stats: Dict[str, Any]


#: 状态字段清单，供测试断言「状态契约」是否被破坏
#: 注：实际实现为取 ResearchState.__annotations__ 的键序，因此新增字段会自动进入清单，
#: tests/test_state_flow.py 会据此校验 new_state() 是否把每个字段都填了初值。
STATE_FIELDS: List[str] = list(ResearchState.__annotations__.keys())


def new_state(
    question: str,
    run_id: str,
    *,
    started_at: str = "",
    config: Optional[Dict[str, Any]] = None,
    max_revision_rounds: int = 2,
) -> ResearchState:
    """构造一次运行的初始黑board状态。

    参数：
        question             研究问题原文（Planner 的输入，也是最终 summary 的回显字段）。
        run_id               运行编号（与 runs/<run_id>.jsonl 同名，串起 trace 与 replay）。
        started_at           开始时间字符串（由 orchestrator 用 datetime.now().isoformat() 传入）。
        config               RuntimeConfig.to_dict() 的快照，写进状态便于事后复现；
                             缺省为 {}。
        max_revision_rounds  反思循环轮次上限，缺省 2；orchestrator 用
                             config.max_revision_rounds 覆盖它（即 MAX_REVISION_ROUNDS）。

    返回：
        一份字段齐全的 ResearchState：所有列表为 []、字典为 {}、字符串为 ""、
        revision_round=0、risk_level="info"、needs_human=False、human_decision=None。

    副作用 / 异常：
        纯函数，不落盘、不打印、不抛异常（仅做字典构造）。
    """
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
    """证据索引：child_id -> 证据片段。

    参数：
        state  共享状态；只读其中 evidence 字段。

    返回：
        {child_id 字符串: 该条证据 dict}，用于 O(1) 按 ID 回查原文。

    副作用 / 异常：
        纯函数，无副作用。注：实际实现为 str(e.get("child_id"))，因此某条证据缺少
        child_id 字段时会落入字符串键 "None"（真实链路上 child_id 恒存在）。
    """
    return {str(e.get("child_id")): e for e in state.get("evidence", [])}


def evidence_ids(state: ResearchState) -> set:
    """当前所有证据的 child_id 集合。

    参数：
        state  共享状态；只读 evidence。

    返回：
        证据 ID 的 set，供 RiskCheckerAgent 做「结论引用的证据是否真实存在」的校验
        （编造的 child_id 会被判定为无证据支撑）。

    副作用 / 异常：
        纯函数，无副作用；空 evidence 时返回空 set。
    """
    return set(evidence_by_id(state))


def snapshot(state: ResearchState) -> Dict[str, Any]:
    """生成用于 trace 的状态摘要（只保留规模信息，不落全量文本）。

    这是「状态膨胀规避」在观测侧的落点：trace 每行都要能安全落盘，所以既不写证据正文、
    也不写简报全文，只写 count / chars 与路由等小字段。

    参数：
        state  共享状态（可为运行中或运行结束后的任意快照）。

    返回：
        含 plan_companies / route / evidence_count / metrics_count / findings_count /
        revision_round / risk_verdict / risk_level / needs_human / citations_count /
        report_chars 的扁平字典，会被 orchestrator._summary() 放进 trace 汇总行。

    副作用 / 异常：
        纯函数，无副作用；对缺失字段一律用 get 兜底，不会抛 KeyError。
    """
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
