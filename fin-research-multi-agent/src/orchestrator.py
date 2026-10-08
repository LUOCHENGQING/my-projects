"""编排层：把语料、工具、Agent 和引擎装配成一张可运行的图。

Agent 拓扑（与 README 的 mermaid 图一致）：

                    ┌──────────┐
                    │ planner  │
                    └────┬─────┘
              route[] ┌──┴───────────────┐
                      v                  v
                ┌───────────┐      ┌───────────┐
                │ retriever │      │  writer   │  (仅寒暄类问题)
                └─────┬─────┘      └─────┬─────┘
                      v                  │
                ┌───────────┐            │
                │  analyst  │<─────┐     │
                └─────┬─────┘      │     │
                      v            │     │
                ┌─────────────┐    │     │
                │risk_checker │────┘ revise（最多 2 轮）
                └──────┬──────┘          │
              escalate │ pass            │
                       v                 │
              ┌────────────────┐         │
              │  human_review  │─────────┤ approved
              └────────┬───────┘         │
                   rejected               │
                       v                  v
                     [END] <──────── writer
                                        │
                                       END

注：实际实现为 risk_checker 裁决 pass 时**直接**走 writer，只有 escalate 才进 human_review
（见 _make_routers() 里的 route_after_risk 与 _build_spec() 里的 conditional_edges 映射表）；
上面 ASCII 图中 "escalate │ pass" 共用一条下行箭头的画法容易被误读成「pass 也去人工确认」，
以代码为准。

三条条件边（这是"多智能体"和"一条链"的本质区别）：
    1. planner  -> 根据 route 决定是否需要检索；
    2. risk_checker -> pass / revise（回到 analyst）/ escalate（转人工）；
    3. human_review -> approved（继续写简报）/ rejected（终止）。

反思循环的防死循环三保险：
    (a) revision_round 计数，超过 max_revision_rounds 一律 escalate；
    (b) 循环只允许 analyst <-> risk_checker 之间往返，拓扑上不会绕回 planner；
    (c) 引擎层 recursion_limit 兜底。

架构层次与职责：
    本模块是「编排层」，夹在引擎（engine_langgraph / engine_native）与 Agent（agents/*）
    之间：向上给 demo / eval / 测试提供 ResearchPipeline 与 run_research 两个入口；向下把
    语料、工具、LLM、Agent 与人机协同装配成一张图，并选择引擎执行。所有「拓扑决策」
    （节点顺序、条件边路由、反思回边、HITL 落点）都集中在这里，Agent 只关心自己的单步业务。

对外关键对象：
    GraphSpec         引擎无关的图声明（state_schema / nodes / entry / edges /
                      conditional_edges / end），是两种引擎的共同输入
    build_engine()    按偏好实例化引擎：auto 优先 LangGraph，构建失败（或运行期异常）则降级自研
    ResearchPipeline  一条可复用的流水线（构造时装载语料与工具，run() 执行一次）
    run_research()    一次性执行的便捷函数（demo / eval / 测试复用）
    make_run_id()     生成 run-时间戳-问题指纹 形式的运行编号

主要输入输出：
    run(question, run_id=None) ->
        {run_id, engine, trace_path,
         state（最终共享状态：report / metrics / findings / risk_report / human_decision …）,
         report（简报正文；HITL 驳回时为空串）, summary, llm_stats}
    也就是「状态 + 简报 + 轨迹路径」三件套；轨迹由 tracing 落成 runs/<run_id>.jsonl，
    可用 `python -m src.replay <run_id>` 完整回放。

被谁调用：
    * src/demo.py：CLI 演示（构造 pipeline、打印过程与简报、提示 replay 命令）；
    * eval/run_eval.py：评测脚本逐条 case 调 pipeline.run() 采集四项指标；
    * tests/conftest.py：pipeline 夹具（auto=True、独立 runs_dir、quiet=True，避免污染 runs/）；
    * tests/test_state_flow.py 等：直接断言 result["state"] 的字段与访问顺序。

三重保险在本文件里的具体落点：
    (a) run() 把 config.max_revision_rounds（即 config.MAX_REVISION_ROUNDS，默认 2）写进
        初始状态；RiskCheckerAgent 据此裁决 revise / escalate，human_review_node 也会读它
        生成「反思循环已达上限（N 轮）」的升级文案；
    (b) 回边只有 risk_checker -> analyst 这一条（映射见 conditional_edges），
        retriever / planner 不在环上，因此重算不会重新检索、更不会回到入口；
    (c) run() 构造 graph_config 时显式传 GRAPH_RECURSION_LIMIT 作 recursion_limit，
        两种引擎都实现了「步数超限即抛错」，即使有人误改条件边也不会把进程挂死。

HITL 的触发与降级：
    触发：risk_checker 的 verdict 为 escalate（高风险，或反思循环达上限仍有缺口）时，
    条件边 after_risk 把流程送进 human_review 节点；该节点用 hitl.render_payload() 组装
    摘要、HumanReviewer.review() 取决策，并把决策写回 state["human_decision"]。
    降级：HumanReviewer 在 --auto / 非 TTY 下自动放行（source 记为 auto / non-interactive），
    保证 CI、评测与演示不被交互卡死；人工驳回时 after_human 走 END，不产出简报。

JSONL 轨迹与 replay：
    每次 run() 新建一个 TraceRecorder（run_id 同时充当 checkpointer 的 thread_id）；
    各 Agent 的每一步由 BaseAgent.run() 写一行，human_review 节点手写一行，引擎降级时
    追加一行 engine_fallback，最后由 recorder.close(summary) 写汇总行；
    replay 只读这份 JSONL，就能还原整条执行路径（耗时、digest、工具调用、业务留痕）。

状态膨胀的规避（与 state.py / agents/retriever.py 协作）：
    evidence 只保留命中的子块与定位元数据（父块正文不落状态），条数由
    RetrieverAgent.MAX_EVIDENCE 截断；_summary() 里的 snapshot 只写规模信息，不写正文。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .agents import build_agents
from .agents.base import AgentContext
from .config import GRAPH_RECURSION_LIMIT, RUNS_DIR, RuntimeConfig, runtime_config
from .engine_langgraph import LANGGRAPH_AVAILABLE, build_langgraph_runner
from .engine_native import END, StateGraph
from .hitl import HumanReviewer, render_payload
from .llm.client import LLMClient
from .rag import HybridRetriever, build_chunks, build_fact_store, load_documents
from .state import ResearchState, new_state, snapshot
from .tools import build_default_registry
from .tracing import TraceRecorder
from .utils.digest import digest

__all__ = ["GraphSpec", "ResearchPipeline", "build_engine", "run_research"]


# ---------------------------------------------------------------------------
# 图规格（引擎无关的声明式描述）
# ---------------------------------------------------------------------------
@dataclass
class GraphSpec:
    """一张图的完整声明，可被 LangGraph 或自研引擎编译执行。

    职责：把「图长什么样」与「用什么引擎跑」解耦——本类只描述拓扑，不含任何执行逻辑，
    因此 engine_langgraph.build_langgraph_runner() 与 build_engine() 的自研分支可以共用它。

    关键属性：
        state_schema       共享状态 schema；本项目固定传 ResearchState（TypedDict）
        nodes              节点名 -> 节点函数（(state) -> state），共 6 个：
                           5 个 Agent（planner/retriever/analyst/risk_checker/writer）
                           + 编排层自带的 human_review
        entry              入口节点名；本项目固定 "planner"
        edges              无条件边列表 [(src, dst)]，dst 可为 END
        conditional_edges  条件边列表 [(src, router, mapping)]，router 返回 key，
                           mapping 把 key 翻成目标节点名（缺失时自研引擎会按 key 原样当节点名）
        end                终止标记，默认 engine_native.END（"__end__"，与 LangGraph 同值）

    状态流转：本类只在装配期存在（ResearchPipeline._build_spec() 产出、last_spec 留档），
    运行期不再被修改——每次 run() 都重新装配，节点函数闭包绑定当次的 recorder。
    """

    state_schema: Any
    nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]
    entry: str
    edges: List[Tuple[str, str]] = field(default_factory=list)
    conditional_edges: List[Tuple[str, Callable[[Dict[str, Any]], str], Dict[str, str]]] = field(
        default_factory=list
    )
    end: str = END


# ---------------------------------------------------------------------------
# 引擎选择
# ---------------------------------------------------------------------------
def build_engine(spec: GraphSpec, preference: str = "auto", thread_id: str = "default") -> Any:
    """按偏好构建执行引擎；auto 优先 LangGraph，失败自动降级到自研引擎。

    参数：
        spec        图声明（GraphSpec），两种引擎都按同一份声明装配。
        preference  引擎偏好：auto / langgraph / native（大小写不敏感，空值按 auto 处理）。
        thread_id   checkpointer 的会话 ID；本项目传 run_id，使「一次运行 = 一个 thread」，
                    从而状态历史与轨迹文件一一对应。

    返回：
        一个 runner 对象，统一暴露 engine_name 与 invoke(state, config)；实际类型为
        LangGraphRunner（engine_name="langgraph"）或 CompiledGraph（engine_name="native"）。

    副作用 / 异常：
        纯装配，不执行图。异常策略刻意不对称：preference="langgraph" 时构建失败会**抛出**
        （用户明确要求该引擎，不该悄悄换掉）；auto 时静默吞掉异常并降级到自研引擎，
        保证任何环境都能跑通。
        注：实际实现为 InMemoryCheckpointer 在自研分支内部才 import（模块顶部只导入了
        END 与 StateGraph）；自研引擎每次 compile 都新建一个内存 checkpointer（不落盘）。
    """
    pref = (preference or "auto").lower()

    if pref in {"auto", "langgraph"} and LANGGRAPH_AVAILABLE:
        try:
            return build_langgraph_runner(spec, thread_id=thread_id)
        except Exception:  # noqa: BLE001 - 版本不兼容时静默降级
            if pref == "langgraph":
                raise
            # auto 模式下继续走自研引擎

    graph = StateGraph(spec.state_schema)
    for name, fn in spec.nodes.items():
        graph.add_node(name, fn)
    graph.set_entry_point(spec.entry)
    for src, dst in spec.edges:
        graph.add_edge(src, dst)
    for src, router, mapping in spec.conditional_edges:
        graph.add_conditional_edges(src, router, mapping)
    from .engine_native import InMemoryCheckpointer

    return graph.compile(checkpointer=InMemoryCheckpointer())


# ---------------------------------------------------------------------------
# 管线
# ---------------------------------------------------------------------------
class ResearchPipeline:
    """投研多智能体管线。

    职责：把「一次性装配」与「多次运行」分开——构造时完成重活（装载 data/ 语料、切父子块、
    建混合检索器与事实库、注册工具、初始化 LLM 与人机协同），run() 时只做轻量装配
    （建 recorder、建图、选引擎、执行）。因此评测里可以复用一个 pipeline 跑多条 case。

    关键属性：
        config        RuntimeConfig（engine 可被构造参数覆盖），其 max_revision_rounds
                      最终写进初始状态，作为反思循环的上限；
        documents / parents / children   语料与父子块（demo 的 banner 会打印数量）；
        retriever     HybridRetriever（工具 search_filings 的后端）；
        fact_store    结构化事实库（get_financial_metric 等工具的后端）；
        registry      工具注册中心（Agent 只能调用各自白名单内的工具）；
        llm           LLMClient（无 key 时自动进入确定性 mock 模式）；
        reviewer      HumanReviewer（HITL 决策来源，quiet 时把 print_fn 换成空函数）；
        runs_dir      轨迹目录（默认 config.RUNS_DIR，测试会传临时目录）；
        quiet         是否静音（CI/评测用）；
        engine_name   最近一次 run() 实际使用的引擎名（"langgraph" / "native"）；
        last_spec     最近一次 run() 装配出的 GraphSpec（留档，便于调试拓扑）。

    状态流转：构造（只读语料） -> run()（每次重建 recorder 与图，互不干扰） ->
    结果字典返回后本对象仍可复用于下一次 run()。
    """

    def __init__(
        self,
        *,
        auto: bool = False,
        engine: Optional[str] = None,
        runs_dir: Optional[Path] = None,
        config: Optional[RuntimeConfig] = None,
        data_dir: Optional[Path] = None,
        reviewer: Optional[HumanReviewer] = None,
        quiet: bool = False,
    ) -> None:
        """装配管线（重活在构造期一次做完）。

        参数（签名里有 *，因此全部为关键字参数）：
            auto      是否让人工确认环节自动放行（传给 HumanReviewer）；评测/测试用 True。
            engine    引擎偏好覆盖：非空则写入 self.config.engine（auto / langgraph / native）。
            runs_dir  轨迹目录；None 表示 config.RUNS_DIR。
            config    RuntimeConfig；None 表示调用 runtime_config() 现取（会读环境变量）。
            data_dir  语料目录；None 表示 config.DATA_DIR（详见 rag.load_documents）。
            reviewer  自定义人工确认器；None 表示按 auto/quiet 构造默认 HumanReviewer。
            quiet     是否静音；True 时把 HumanReviewer 的 print_fn 换成空函数（评测/测试不刷屏）。

        返回：无。
        副作用（较重）：
            读取并解析 data/ 下的全部文档、切分父子块、构建 BM25/向量索引与事实库、
            注册全部内置工具；装载过程本身不打印，但有明显的 CPU / 内存开销。
            注：实际实现为 self.quiet 只影响 HITL 摘要与提示的打印，不影响其它日志。
        异常：
            语料目录缺失或不可解析时由 rag 层抛出（构造失败比运行到一半失败更容易定位）。
        """
        self.config = config or runtime_config()
        if engine:
            self.config.engine = engine
        self.runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
        self.quiet = quiet

        # ---- 1) 语料 -> 父子块 -> 混合检索器 + 事实库 ----
        self.documents = load_documents(data_dir)
        self.parents, self.children = build_chunks(
            list(self.documents),
            child_chars=self.config.child_chunk_chars,
            overlap=self.config.child_chunk_overlap,
        )
        self.retriever = HybridRetriever(
            self.parents, self.children, weights=self.config.rerank_weights, embed_dim=self.config.embed_dim
        )
        self.fact_store = build_fact_store(list(self.documents))

        # ---- 2) 工具注册中心 ----
        self.registry = build_default_registry(self.retriever, self.fact_store, self.documents)

        # ---- 3) LLM（无 key 自动 mock） ----
        self.llm = LLMClient(self.config)

        # ---- 4) 人机协同 ----
        self.reviewer = reviewer or HumanReviewer(auto=auto, print_fn=print if not quiet else (lambda *a, **k: None))

        self.engine_name = "native"
        self.last_spec: Optional[GraphSpec] = None

    # ------------------------------------------------------------------
    # 图装配
    # ------------------------------------------------------------------
    def _make_routers(self) -> Dict[str, Callable[[Dict[str, Any]], str]]:
        """三条条件边的路由函数。

        参数：无（路由只读传入的 state，不依赖实例状态）。
        返回：
            {"after_planner": ..., "after_risk": ..., "after_human": ...}，
            键名与 _build_spec() 里 conditional_edges 的引用一一对应。

        副作用 / 异常：
            纯函数（只构造闭包），无 IO；路由函数本身对缺字段一律用默认值兜底，
            因此不会因 state 不完整而抛异常——更重要的是不会把图卡住（必须有明确出口）。

        为什么要独立成一层：自研引擎与 LangGraph 都要求 router 是纯函数（只吃 state、
        吐下一个节点名），把三条判断集中在这里，可以单测路由而不必真的跑图。
        """

        def route_after_planner(state: Dict[str, Any]) -> str:
            """按 Planner 给出的 route 决定第一个下游 Agent。

            参数：state —— 只读 route（Planner 写的 Agent 顺序列表）。
            返回：命中的第一个节点名；一个都没命中时返回 "writer"
                  （寒暄类问题不需要检索与分析，直接写简报）。
            副作用 / 异常：纯函数，无副作用、不抛异常。
            注：实际实现为按 ("retriever", "analyst", "risk_checker") 的固定优先级取
            「第一个出现在 route 里的节点」，而不是照 route 的顺序取首个元素——
            这样即使 Planner 把 route 写成 ["writer", "retriever"]，检索也仍会执行。
            """
            route = state.get("route") or []
            for candidate in ("retriever", "analyst", "risk_checker"):
                if candidate in route:
                    return candidate
            return "writer"

        def route_after_risk(state: Dict[str, Any]) -> str:
            """反思循环的核心分支：打回重算 / 转人工 / 通过。

            参数：state —— 只读 risk_verdict（RiskCheckerAgent 的确定性裁决）。
            返回：
                "revise"   -> "analyst"（回到 AnalystAgent 重算，这是唯一的回边）；
                "escalate" -> "human_review"（转人工确认）；
                其余       -> "writer"（pass 或字段缺失都放行到写作，避免流程卡死）。
            副作用 / 异常：纯函数，无副作用、不抛异常。
            注：实际实现为对 risk_verdict 做字符串相等判断，未知取值一律当 pass 处理；
            真正的「最多重算 N 轮」约束在 RiskCheckerAgent._decide() 里（达上限即 escalate）。
            """
            verdict = str(state.get("risk_verdict") or "pass")
            if verdict == "revise":
                return "analyst"
            if verdict == "escalate":
                return "human_review"
            return "writer"

        def route_after_human(state: Dict[str, Any]) -> str:
            """人工确认后的分支：批准则继续写作，否则终止。

            参数：state —— 只读 human_decision.decision（HumanDecision.to_dict() 的产物）。
            返回："writer"（approved）或 "end"（其余，含 rejected 与字段缺失）。
            副作用 / 异常：纯函数，无副作用、不抛异常。
            注：实际实现为「默认 rejected」——取不到决策时按驳回处理，宁可不出简报，
            也不在未获批准的情况下输出高风险结论。
            """
            decision = (state.get("human_decision") or {}).get("decision", "rejected")
            return "writer" if decision == "approved" else "end"

        return {
            "after_planner": route_after_planner,
            "after_risk": route_after_risk,
            "after_human": route_after_human,
        }

    def _build_spec(self, recorder: TraceRecorder) -> GraphSpec:
        """为一次运行装配图（Agent 与 recorder 绑定）。

        参数：
            recorder  本次运行的轨迹记录器；会被塞进 AgentContext，让每个 Agent 的每一步
                      都能写进同一份 JSONL（这是「每个 Agent 的输入输出都落到 trace」的来源）。

        返回：
            GraphSpec —— entry="planner"，5 个 Agent 节点 + human_review 节点，
            3 条无条件边（retriever->analyst、analyst->risk_checker、writer->END）
            与 3 条条件边（planner / risk_checker / human_review 之后）。

        副作用：
            创建 AgentContext 与全部 Agent 实例（其中 LLM 客户端、工具注册中心是构造期
            传入的共享依赖）；不改 self 上的其它状态。每次 run() 都会重新调用一次，
            因此不同 run 的 Agent 实例与 recorder 严格一一对应，不会串线。
        """
        ctx = AgentContext(
            registry=self.registry,
            llm=self.llm,
            recorder=recorder,
            config=self.config,
            extras={"pipeline": self},
        )
        agents = build_agents(ctx)
        routers = self._make_routers()

        def human_review_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """人机协同节点：高风险 / 循环超限时转人工确认。

            注：实际实现为就地阻塞读取输入（交互式环境下），而不是把图挂起等外部恢复；
            详见 hitl 模块 docstring 与 HumanReviewer.review()。

            参数：
                state  当前共享状态；只读 risk_report / risk_level / revision_gaps /
                       max_revision_rounds / question，并基于它复制出新状态。

            返回：
                新的状态字典（浅拷贝 + 覆写 human_decision 与 steps），不修改入参 state。

            副作用 / 异常：
                1) 打印待确认摘要并向 stdin 阻塞读一行（交互模式下；auto / 非 TTY 直接放行，
                   详见 hitl.HumanReviewer.review）；
                2) 手动写一行 agent="human_review" 的轨迹（extra 里带 needs_human 与
                   risk_level），并追加一条 steps 留痕——这两件事本该由 BaseAgent.run()
                   模板方法负责，但本节点不是 BaseAgent 子类，所以在此显式补上。
                异常：不影响流程——review() 内部已把 EOF / 中断吞成「驳回」。

            升级原因的取值顺序（为什么这样写）：优先用 RiskCheckerAgent 给出的
            escalation_reason（模型叙述，信息最全）；为空时按「有缺口 = 循环达上限」
            「无缺口 = 风险等级 high」两种情形现编文案，保证人工看到的一定是有意义的原因，
            而不会出现空白。
            """
            started = time.perf_counter()
            new_state = dict(state)
            risk_report = state.get("risk_report") or {}
            reason = str(risk_report.get("escalation_reason") or "")
            gaps = risk_report.get("gaps") or []
            if not reason:
                if gaps:
                    reason = (
                        f"反思循环已达上限（{state.get('max_revision_rounds')} 轮），"
                        f"仍有 {len(gaps)} 项数据缺口未消除。"
                    )
                else:
                    reason = "风险等级为 high，按流程需要人工确认。"

            payload = render_payload(state, reason)
            decision = self.reviewer.review(payload)
            new_state["human_decision"] = decision.to_dict()

            entry = recorder.record(
                agent="human_review",
                input_obj=payload,
                output_obj=decision.to_dict(),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                status="ok",
                extra={"needs_human": True, "risk_level": state.get("risk_level", "")},
            )
            steps = list(new_state.get("steps") or [])
            steps.append({"step": entry["step"], "agent": "human_review", "status": "ok",
                          "latency_ms": entry["latency_ms"], "tool_calls": []})
            new_state["steps"] = steps
            return new_state

        # 节点名即 trace 里的 agent 字段；这里绑定的是 BaseAgent.run（模板方法），
        # 而不是各 Agent 的 _execute —— 只有走 run() 才能保证「执行 + 落 trace + 记 steps」。
        nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "planner": agents["planner"].run,
            "retriever": agents["retriever"].run,
            "analyst": agents["analyst"].run,
            "risk_checker": agents["risk_checker"].run,
            "human_review": human_review_node,
            "writer": agents["writer"].run,
        }

        return GraphSpec(
            state_schema=ResearchState,
            nodes=nodes,
            entry="planner",
            # 无条件边只保留「必然发生」的三段；分支一律交给下面的条件边，
            # 这样拓扑里唯一的环就是 risk_checker -> analyst 这一条反思回边。
            edges=[
                ("retriever", "analyst"),   # 检索完必然进入分析
                ("analyst", "risk_checker"),  # 分析完必然接受核查（可能被打回重算）
                ("writer", END),            # 简报写完即结束
            ],
            # 条件边：router 返回 key，再由 mapping 翻成目标节点名（"end" 翻成 END 标记）。
            conditional_edges=[
                (
                    "planner",
                    routers["after_planner"],
                    # 四个出口都必须列出：route 命中谁就走谁，都没命中时 router 已兜底 "writer"
                    {"retriever": "retriever", "analyst": "analyst",
                     "risk_checker": "risk_checker", "writer": "writer"},
                ),
                (
                    "risk_checker",
                    routers["after_risk"],
                    # 反思循环的三个出口：回 analyst 重算 / 转人工 / 通过后写作
                    {"analyst": "analyst", "human_review": "human_review", "writer": "writer"},
                ),
                (
                    "human_review",
                    routers["after_human"],
                    # 人工批准才继续写作，否则直接 END（不产出简报）
                    {"writer": "writer", "end": END},
                ),
            ],
        )

    # ------------------------------------------------------------------
    # 运行
    # ------------------------------------------------------------------
    def run(self, question: str, run_id: Optional[str] = None) -> Dict[str, Any]:
        """执行一次完整投研流程。

        参数：
            question  研究问题原文（Planner 的输入）。
            run_id    运行编号；None 时用 make_run_id(question) 生成。
                      同一个 run_id 会**覆盖**同名轨迹文件，因此复用编号 = 有意重跑比对。

        返回：
            {run_id, engine（实际使用的引擎名）, trace_path（JSONL 绝对路径）,
             state（最终共享状态）, report（简报正文，HITL 驳回时为空串）,
             summary（运行汇总，与轨迹汇总行同源）, llm_stats（LLM 调用统计）}。

        副作用：
            1) 在 runs_dir 下落一份 <run_id>.jsonl 轨迹（含汇总行）；
            2) 交互式环境下可能阻塞等待人工确认（见 human_review_node）；
            3) 若 LangGraph 运行期异常，会把 self.config.engine 改成 "native"（**永久改实例
               配置**，后续 run 默认走自研引擎），并向轨迹补一行 engine_fallback。
        异常：
            非 LangGraph 引擎抛出的异常直接向上抛（此时轨迹只有已完成的步骤，没有汇总行）；
            LangGraph 的异常被降级逻辑吞掉，除非降级后的自研引擎再抛。
        """
        run_id = run_id or make_run_id(question)
        recorder = TraceRecorder(run_id, runs_dir=self.runs_dir)
        spec = self._build_spec(recorder)
        self.last_spec = spec

        # 让 checkpointer 的 thread_id 与 run_id 对齐：一次运行 = 一个 thread，
        # 于是「轨迹文件」「状态历史」「replay 命令」三者可以用同一个编号互相索引。
        thread_id = run_id
        runner = build_engine(spec, self.config.engine, thread_id=thread_id)
        self.engine_name = getattr(runner, "engine_name", "unknown")

        state: ResearchState = new_state(
            question=question,
            run_id=run_id,
            started_at=datetime.now().isoformat(timespec="seconds"),
            config=self.config.to_dict(),
            max_revision_rounds=self.config.max_revision_rounds,
        )
        # recursion_limit 是防死循环的最后一道保险（第三重）：即使条件边被误改成自环，
        # 引擎也会在 GRAPH_RECURSION_LIMIT 步内抛错，而不是把进程挂死。
        graph_config = {
            "recursion_limit": GRAPH_RECURSION_LIMIT,
            "configurable": {"thread_id": thread_id},
        }

        started = time.perf_counter()
        try:
            final = runner.invoke(state, config=graph_config)
        except Exception as exc:  # noqa: BLE001 - LangGraph 运行期异常时降级重跑
            # 为什么能安全重跑：state 是纯字典、Agent 侧没有需要回滚的外部写入（只读语料 +
            # 写轨迹），且 checkpointer 按 thread_id 隔离；重跑一次比让整条流程失败更划算。
            if self.engine_name == "langgraph":
                recorder.record(
                    agent="engine_fallback",
                    input_obj={"engine": "langgraph"},
                    output_obj={"error": f"{type(exc).__name__}: {exc}"},
                    latency_ms=0.0,
                    status="error",
                )
                self.config.engine = "native"
                runner = build_engine(spec, "native", thread_id=thread_id)
                self.engine_name = getattr(runner, "engine_name", "native")
                final = runner.invoke(state, config=graph_config)
            else:
                raise
        total_ms = (time.perf_counter() - started) * 1000.0

        summary = self._summary(final, total_ms, run_id)
        recorder.close(summary)

        return {
            "run_id": run_id,
            "engine": self.engine_name,
            "trace_path": str(recorder.path),
            "state": final,
            "report": final.get("report", "") or "",
            "summary": summary,
            "llm_stats": self.llm.stats(),
        }

    # ------------------------------------------------------------------
    def _summary(self, state: Dict[str, Any], total_ms: float, run_id: str) -> Dict[str, Any]:
        """运行汇总（写入 trace 最后一行，也是 eval 的主要数据来源）。

        参数：
            state     最终共享状态（只读；各类计数用 get 兜底）。
            total_ms  端到端耗时（毫秒），round 到 3 位小数。
            run_id    运行编号，回填进汇总便于单行自证身份。

        返回：
            扁平汇总字典：status（有简报为 "ok"，否则 "no_report"）、engine、mode
            （mock / openai-compatible）、question、total_latency_ms、steps、
            agents_visited（执行路径）、revision_round、risk_verdict、risk_level、
            needs_human、human_decision、findings / citations / evidence 计数、
            report_chars、errors，以及 snapshot（state.snapshot() 的规模摘要）。

        副作用 / 异常：
            纯函数；注意它读的是 self.engine_name（引擎降级后已更新为 "native"）。
        注：实际实现为 status 只区分 "ok" / "no_report" 两种，被 HITL 驳回（report 为空）
        也会记成 "no_report"；replay 的 _status_mark 对未知 status 显示 "?"。
        """

        findings = state.get("findings") or []
        citations = state.get("citations") or []
        report = state.get("report") or ""
        return {
            "run_id": run_id,
            "status": "ok" if report else "no_report",
            "engine": getattr(self, "engine_name", "unknown"),
            "mode": "mock" if self.llm.is_mock else "openai-compatible",
            "question": state.get("question", ""),
            "total_latency_ms": round(total_ms, 3),
            "steps": len(state.get("steps") or []),
            "agents_visited": [s.get("agent") for s in (state.get("steps") or [])],
            "revision_round": state.get("revision_round", 0),
            "risk_verdict": state.get("risk_verdict", ""),
            "risk_level": state.get("risk_level", ""),
            "needs_human": state.get("needs_human", False),
            "human_decision": (state.get("human_decision") or {}).get("decision", ""),
            "findings": len(findings),
            "citations": len(citations),
            "report_chars": len(report),
            "evidence": len(state.get("evidence") or []),
            "errors": state.get("errors") or [],
            "snapshot": snapshot(state),  # type: ignore[arg-type]
        }


def make_run_id(question: str) -> str:
    """生成可读且唯一的运行编号：run-时间戳-问题指纹。

    参数：
        question  研究问题原文；只用于算指纹，不做任何清洗。

    返回：
        形如 "run-20261003-011813-349dce" 的字符串：时间戳精确到秒，指纹是问题文本的
        sha256 前 6 位（utils.digest.digest）。同一秒内跑不同问题不会撞号，
        同一问题在不同秒跑也会得到新编号（便于反复重跑比对）。

    副作用 / 异常：
        纯函数（读系统时钟），无副作用、不抛异常。
    注：实际实现为按 `%Y%m%d-%H%M%S` 取本地时间，因此该编号同时充当轨迹文件名与
    checkpointer 的 thread_id（见 ResearchPipeline.run）。
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"run-{stamp}-{digest(question, 6)}"


def run_research(
    question: str,
    *,
    auto: bool = False,
    engine: Optional[str] = None,
    run_id: Optional[str] = None,
    runs_dir: Optional[Path] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """一次性执行投研流程的便捷入口（供 demo / eval / 测试复用）。

    参数：
        question  研究问题原文。
        auto      是否自动放行人工确认（评测 / CI 传 True）。
        engine    引擎偏好 auto / langgraph / native；None 表示沿用环境配置。
        run_id    指定运行编号；None 表示自动生成。
        runs_dir  轨迹目录；None 表示 config.RUNS_DIR。
        quiet     是否静音（不打印 HITL 摘要）。

    返回：
        ResearchPipeline.run() 的结果字典（run_id / engine / trace_path / state /
        report / summary / llm_stats）。

    副作用：
        每次调用都**新建一个 ResearchPipeline**，即重新装载语料、重建检索索引与工具注册中心
        （较慢）；需要连续跑多条问题时，请直接复用 ResearchPipeline 实例。
    异常：
        语料装载失败或引擎执行失败会向上抛（本函数不做任何兜底）。
    """
    pipeline = ResearchPipeline(auto=auto, engine=engine, runs_dir=runs_dir, quiet=quiet)
    return pipeline.run(question, run_id=run_id)
