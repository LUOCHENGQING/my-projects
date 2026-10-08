"""投顾流水线：节点实现 + 共享状态 + 人机协同。

层次定位
--------
本模块位于**编排层**（`src/engine/` 之下、`src/agents/` 与各确定性求解器之上）：
- 向上：被 `src/engine/langgraph_engine.py`（正式引擎）与 `src/engine/native.py`
  （零依赖降级引擎）调用，两个引擎都只注册/串联**本模块提供的同一批节点函数**，
  因此两条路径的**状态流转与结果必然一致**，并有单测做路径一致性对照；
  也被 `eval/run_eval.py`、`src/demo.py` 等入口通过 `run_pipeline` / `build_pipeline` 调用。
- 向下：每个节点把活分派给 `src/agents/` 的五个 Agent（画像 / 筛选 / 组合 / 适当性 / 建议书），
  Agent 再调用 `src/constraints.py`、`src/optimizer.py`、`src/suitability/` 等确定性求解器。

解决什么问题
------------
把「客户约束 → 可行域 → 组合 → 适当性硬闸门 → 人工确认 → 建议书 → 版本落盘」串成一条
**可重放、可留痕、可打回重配**的确定性主流程，并保证合规闸门拥有真实的否决权。

对外暴露的关键对象
------------------
- `AdvisoryState`：两个引擎共用的共享状态 schema（TypedDict）。
- `PipelineConfig`：流水线配置（引擎、是否自动放行、打回轮次上限、豁免规则等）。
- `AdvisoryPipeline`：节点实现 + 路由 + 人机协同判定的载体（核心类）。
- `build_pipeline()` / `run_pipeline()`：装配流水线 / 一次跑完的顶层入口。
- `run_native()`：直接调用降级引擎的函数式入口（薄封装）。
- `state_summary()`：把终态压缩成可比较、可展示的摘要。
- `STATUS_*` 常量：终态取值，供 `state_summary()` 与各测试断言使用。

主要输入 / 输出
---------------
输入：`client_id` + `PipelineConfig`（可选注入 `DataBundle`、`BaseLLM`、`Tracer`、`VersionStore`）。
输出：跑完后的 `AdvisoryState`（终态见 `status` 字段）+ `Tracer`（trace 落盘）+ `AdvisoryPipeline`
（可从中读 `store` 拿到建议版本链）。

共享状态 `AdvisoryState`（LangGraph 与降级引擎共用同一份 schema 与同一批节点函数，
因此两条路径的**状态流转与结果必然一致**，并有单测做路径一致性对照）。

流转：
    profile → screen → optimize → suitability
                          ↑            │
                          └── 打回重配 ─┤（最多 max_repair_rounds 轮）
                                       ├─ 通过 ─────→ human_gate → narrative → END
                                       └─ 超限 → escalate → human_gate → narrative → END

状态如何流转
------------
节点**不原地改 state**，而是返回「增量字典」，由引擎合并回共享状态（LangGraph 由
`StateGraph` 合并，native 引擎由 `state.update(...)` 合并）。跨节点的关键状态：
- `effective_client`：被 `TightenSpec` 逐轮收紧后的客户约束，是 screen / optimize 的实际输入；
- `round`：打回轮次计数，初值 0，每次 `reoptimize` 由 `node_suitability` 加一；
- `gate`：适当性结论（`pass` / `reoptimize` / `reject`），决定条件路由；
- `human_review` / `status`：由 `node_human_gate` 写入，`node_narrative` 读取后落盘版本；
- `change_log`：只追加的变更流水，node 名依次可对照执行顺序。

人机协同：命中 block 后被豁免、单一产品集中度超过内控预警线、高龄客户三类情形，
必须经理财经理确认才能定稿；`--auto` 用于演示自动放行，非交互环境自动降级并留痕。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, TypedDict

from .agents import (
    AdvisorNarrativeAgent,
    ClientProfilingAgent,
    PortfolioOptimizerAgent,
    ProductScreeningAgent,
    SuitabilityOfficerAgent,
)
from .constraints import TightenSpec
from .dataset import DataBundle, load_data
from .llm import BaseLLM, build_llm, load_dotenv
from .observability import Tracer, new_run_id
from .schemas import (
    AdviceRecord,
    ClientProfile,
    GateDecision,
    HumanReview,
    Portfolio,
    Product,
    ScreeningResult,
)
from .utils import now_iso, pct
from .versioning import (
    AdviceSnapshot,
    VersionStore,
    build_payload,
    chain_rows,
)

#: 流程状态取值
#: 注意区分：STATUS_FINAL（正常定稿）、STATUS_FINAL_DEGRADED（非交互环境降级放行）、
#: STATUS_FINAL_ESCALATED（打回重配次数用尽转人工后定稿）、
#: STATUS_REJECTED（闸门直接拒绝，含命中 veto 规则）、STATUS_REJECTED_BY_HUMAN（人工否决）、
#: STATUS_BLOCKED 不是终态：它是「被闸门拦截的那一版」写进版本链时使用的快照状态。
STATUS_FINAL = "final"
STATUS_FINAL_DEGRADED = "final_degraded"
STATUS_FINAL_ESCALATED = "final_after_escalation"
STATUS_REJECTED = "rejected"
STATUS_REJECTED_BY_HUMAN = "rejected_by_human"
STATUS_BLOCKED = "blocked"


class AdvisoryState(TypedDict, total=False):
    """投顾流水线共享状态（两个引擎共用）。

    所有键均为可选（`total=False`）：节点只返回自己负责的增量，由引擎合并。
    字段按流水线阶段分组如下，注释标注「谁写入 / 谁读取」，是理解状态流转的主线索。
    """

    #: 本次运行的唯一标识（由 `new_run_id("advisory")` 生成，也可由调用方显式指定）
    run_id: str
    #: 客户号（`initial_state` 写入；各节点据此取客户档案）
    client_id: str
    #: 实际生效的编排引擎名："langgraph" 或 "native"（可能已被 `resolve_engine` 降级）
    engine: str

    # ---- 阶段一：客户画像（node_profile 写入）----
    #: 原始客户档案（未被收紧，作为留痕与对照基准）
    client: ClientProfile
    #: 经 TightenSpec 逐轮收紧后的有效客户档案，screen / optimize 的实际输入
    effective_client: ClientProfile
    #: 画像结果字典：硬约束 / 软偏好 / 问卷折算 / 保护标记等（供建议书渲染）
    profile: dict[str, Any]

    # ---- 阶段二：产品筛选（node_screen / node_optimize 写入）----
    #: 本轮筛选结果（含被剔除产品的原因码与说明）
    screening: ScreeningResult
    #: 当前候选产品号列表（等于 `screening.included`）
    candidates: list[str]
    #: 筛选口径说明（LLM 或 mock 生成）
    screening_note: str
    #: 各轮筛选结果快照（打回重配时每轮追加一条）
    screening_rounds: list[dict[str, Any]]

    # ---- 阶段三：组合构建（node_optimize 写入）----
    #: 候选组合（权重 + 产品要素快照 + 指标）
    portfolio: Portfolio
    #: 组合权衡说明（LLM 或 mock 生成）
    portfolio_note: str
    #: 本轮紧约束清单（由 `binding_constraints` 给出）
    binding: list[str]
    #: 各候选产品的打分（优化器输入）
    product_scores: dict[str, float]
    #: 各轮组合快照（含当轮约束、权重、现金比例、指标）
    portfolio_rounds: list[dict[str, Any]]
    #: 打回轮次计数：初值 0，每次 `reoptimize` 由 node_suitability 加一
    round: int

    # ---- 阶段四：适当性闸门（node_suitability 写入）----
    #: 闸门结论：directive 为 pass / reoptimize / reject
    gate: GateDecision
    #: 复核意见（LLM 或 mock 生成）
    suitability_comment: str
    #: 打回时下发的约束收紧指令（`TightenSpec.to_dict()` 的普通字典）
    tighten: dict[str, Any]
    #: 各轮闸门结论快照（轮次 / directive / blocks / warns / exempted）
    gate_rounds: list[dict[str, Any]]
    #: 超限转人工的原因列表（node_escalate 写入）
    escalation_reasons: list[str]

    # ---- 阶段五：建议书与留痕（node_human_gate / node_narrative 写入）----
    #: 情景压力测试报告（`StressReport`）
    stress: Any
    #: 反事实解释报告（`CounterfactualReport`）
    counterfactual: Any
    #: 建议书正文（Markdown）
    narrative: str
    #: 12 项必备要素的达标标记：要素名 -> 是否具备
    elements: dict[str, bool]
    #: 结构化最终建议（含组合 / 闸门 / 人工确认 / 压力测试 / 反事实）
    advice: AdviceRecord
    #: 人工确认记录（required / decision / operator / note 等）
    human_review: HumanReview
    #: 本版建议的版本号（由 `VersionStore.next_version` 分配）
    version: int
    #: 上一版版本号（无前序版本时为 None）
    parent_version: int | None
    #: 终态，取值见模块顶部的 STATUS_* 常量
    status: str
    #: 只追加的变更流水：node 名 + 轮次 + 该步关键事实
    change_log: list[dict[str, Any]]


@dataclass(frozen=True)
class PipelineConfig:
    """流水线配置（冻结，运行期不可变，保证一次运行的口径一致）。

    关键字段：
    - `engine`：请求的引擎名（"langgraph" / "native"）；`build_pipeline` 会用
      `resolve_engine` 解析后写入实际生效值，因此 `--engine langgraph` 在缺依赖时可能落到 native。
    - `auto`：True 时人机协同闸门自动放行，仅用于演示；留痕 decision="auto_approved"。
    - `max_repair_rounds`：**打回重配的轮次上限**（默认 2）；当闸门的 `round_index`
      达到该值时不再打回，改为 directive="reject" 且 `escalated=True` 转人工。
    - `exempt_rules`：经人工豁免的 block 级规则号，命中后降级为 warn 并强制人工确认。
    - `interactive`：None 表示按 stdin 是否 TTY 自动判定；True/False 为显式指定。
    - `operator`：人工确认留痕中的操作人标识。
    """

    engine: str = "langgraph"
    auto: bool = False
    max_repair_rounds: int = 2
    exempt_rules: tuple[str, ...] = ()
    interactive: bool | None = None
    operator: str = "理财经理（示例工号 A001）"

    def resolve_interactive(self) -> bool:
        """是否具备交互式人工确认条件（未显式指定时按 stdin 是否 TTY 判断）。

        返回：
            True 表示可以 `input()` 交互确认；False 表示必须走降级放行路径。

        异常：不抛出——底层 `isatty()` 在无可用 stdin 的环境会抛异常，此处捕获后返回 False。
        """
        if self.interactive is not None:
            return bool(self.interactive)
        try:
            return bool(sys.stdin and sys.stdin.isatty())
        except Exception:  # noqa: BLE001 - 某些运行环境没有可用的 stdin
            return False


class AdvisoryPipeline:
    """约束驱动的投顾流水线（节点函数同时服务两个引擎）。

    职责
    ----
    集中实现**全部节点函数**（`node_*`）、**条件路由**（`route_after_suitability`）、
    **人机协同判定**（`human_review_reasons` / `decide_human_review`）与**版本落盘**
    （`_save_snapshot`）。两个引擎只负责串联这些函数，不重复实现任何业务逻辑。

    关键属性
    --------
    - `data`：`DataBundle`，冻结的样例数据（客户 / 产品 / 问卷 / 情景）。
    - `config`：`PipelineConfig`，本次运行配置（含打回轮次上限）。
    - `llm`：`BaseLLM`，无 Key 时自动为 mock 实现，仅用于措辞生成。
    - `tracer`：`Tracer | None`，逐步落盘执行痕迹；`build_pipeline` 会注入并下发给各 Agent。
    - `store`：`VersionStore`，建议版本链（不可变快照，可回溯「当时按什么约束配的」）。
    - `agents`：`dict[str, Any]`，五个 Agent 的实例表，键为类名。
    - `engine_note`：引擎解析说明（例如「未检测到 langgraph，已自动降级到 native 引擎」）。

    状态机要点
    ----------
    profile → screen → optimize → suitability，随后按 `gate.directive` 三分支：
    - `pass` → human_gate（→ narrative）；
    - `reoptimize` → 回到 optimize，但 `effective_client` 已被 `TightenSpec` 单调收紧、
      `round` 已加一，故重配一定在更小的可行域内进行；
    - `reject` → escalate（→ human_gate → narrative）。
    **循环边界**：`round_index`（= state["round"]）达到 `config.max_repair_rounds` 时闸门不再
    下发 reoptimize，直接 reject + escalated，由 `node_escalate` 记录原因；
    native 引擎另有 `MAX_STEPS = 64` 的步数上限做死循环防御。

    被谁使用：`src/engine/langgraph_engine.py`、`src/engine/native.py`、
    `src/demo.py`、`eval/run_eval.py`，以及 `tests/conftest.py` 的 `make_pipeline` 夹具。
    """

    def __init__(
        self,
        data: DataBundle,
        *,
        llm: BaseLLM | None = None,
        tracer: Tracer | None = None,
        store: VersionStore | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        """装配流水线：装载五个 Agent、确定配置、准备版本链。

        参数：
            data：已装载的 `DataBundle`（必填）。
            llm：语言模型；为 None 时调用 `build_llm()`（无 Key 自动退化为 mock）。
            tracer：执行痕迹记录器；为 None 时该阶段不落盘（`build_pipeline` 之后会重新注入）。
            store：版本链存储；为 None 时使用默认 `VersionStore()`（落盘到仓库 runs/）。
            config：流水线配置；为 None 时使用 `PipelineConfig()` 默认值。

        返回：无。

        副作用：构造五个 Agent（共享同一 llm 与 tracer）。不读盘、不写盘。
        """
        self.data = data
        self.config = config or PipelineConfig()
        self.llm = llm or build_llm()
        self.tracer = tracer
        self.store = store or VersionStore()
        self.agents: dict[str, Any] = {
            "ClientProfilingAgent": ClientProfilingAgent(llm=self.llm, tracer=tracer),
            "ProductScreeningAgent": ProductScreeningAgent(llm=self.llm, tracer=tracer),
            "PortfolioOptimizerAgent": PortfolioOptimizerAgent(llm=self.llm, tracer=tracer),
            "SuitabilityOfficerAgent": SuitabilityOfficerAgent(llm=self.llm, tracer=tracer),
            "AdvisorNarrativeAgent": AdvisorNarrativeAgent(llm=self.llm, tracer=tracer),
        }
        self.engine_note: str = ""

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @property
    def products(self) -> dict[str, Product]:
        """产品要素表。

        返回：`data.products`（产品号 -> `Product`），即全市场产品池，
        是筛选与适当性规则共同的 universe。
        """
        return self.data.products

    def initial_state(self, client_id: str, run_id: str) -> AdvisoryState:
        """构造初始共享状态。

        参数：
            client_id：客户号，后续 `node_profile` 据此取客户档案。
            run_id：本次运行标识，写入 trace 与建议快照。

        返回：只填了「起点字段」的 `AdvisoryState`——两个引擎的初始化逻辑共用此处，
        因此两条路径的初始状态完全一致。

        说明：`round=0`，三个 `*_rounds` 与 `change_log` 置空列表，`tighten={}`，
        `escalation_reasons=[]`，`status="running"`（"running" 不是终态常量，仅表示进行中）。
        """
        return AdvisoryState(
            run_id=run_id,
            client_id=client_id,
            engine=self.config.engine,
            round=0,
            screening_rounds=[],
            portfolio_rounds=[],
            gate_rounds=[],
            change_log=[],
            tighten={},
            escalation_reasons=[],
            status="running",
        )

    # ------------------------------------------------------------------
    # 节点：客户画像
    # ------------------------------------------------------------------
    def node_profile(self, state: AdvisoryState) -> dict[str, Any]:
        """客户画像与硬约束提取（流水线第一个节点）。

        参数：
            state：至少含 `client_id` 的共享状态。

        返回：增量字典 `{client, effective_client, profile}`。

        副作用：通过 ClientProfilingAgent 写一条 profile trace（Agent 内部完成）。

        异常：客户号不存在时 `DataBundle.client` 抛 `KeyError`（异常会直接冒泡终止本次运行）。

        说明：此时 `effective_client` 在数值上等于 `client`（仅风险等级可能按「从严原则」
        被问卷折算值下调），真正的约束收紧发生在打回重配时的 `node_suitability`。
        """
        client = self.data.client(state["client_id"])
        result = self.agents["ClientProfilingAgent"].run(
            client=client, questionnaire=self.data.questionnaire
        )
        return {
            "client": result["client"],
            "effective_client": result["effective_client"],
            "profile": result["profile"],
        }

    # ------------------------------------------------------------------
    # 节点：产品筛选（初始可行域）
    # ------------------------------------------------------------------
    def node_screen(self, state: AdvisoryState) -> dict[str, Any]:
        """按有效客户约束筛选初始候选池（只跑第 0 轮）。

        参数：
            state：需含 `effective_client`（由 node_profile 写入）。

        返回：增量字典 `{screening, candidates, screening_note, screening_rounds}`；
        `screening_rounds` 在此处重置为「只含本轮」的列表。

        说明：本节点固定传 `round_index=0`；打回后的重新筛选由 `node_optimize` 负责，
        因此`node_screen` 全程只会执行一次。
        """
        effective_client = state["effective_client"]
        result = self.agents["ProductScreeningAgent"].run(
            client=effective_client, products=self.products, round_index=0
        )
        screening: ScreeningResult = result["screening"]
        return {
            "screening": screening,
            "candidates": result["candidates"],
            "screening_note": result["screening_note"],
            "screening_rounds": [screening.model_dump()],
        }

    # ------------------------------------------------------------------
    # 节点：组合构建（打回重配时会再次进入）
    # ------------------------------------------------------------------
    def node_optimize(self, state: AdvisoryState) -> dict[str, Any]:
        """在（可能已被收紧的）可行域内求解组合。

        参数：
            state：需含 `effective_client`；第 0 轮还会直接复用 `screening`。

        返回：增量字典，写入 `screening` / `candidates` / `screening_rounds` /
        `portfolio` / `portfolio_note` / `binding` / `product_scores` /
        `portfolio_rounds` / `change_log`。

        是否为重配轮：按 `state["round"]` 判定——
        - `round_index == 0` 且 state 里已有 `screening`：直接复用 node_screen 的结果，
          不重复筛选（保证第 0 轮的 trace 里 screen 只出现一次）；
        - 否则（即打回后的第 1 轮起）：用**收紧后的** `effective_client` 重新筛选，
          并把新结果追加进 `screening_rounds`。

        副作用：
        - 通过 PortfolioOptimizerAgent 写一条 optimize trace；
        - 追加一条 `portfolio_rounds` 快照（含当轮 `constraint_snapshot()`、权重、
          现金比例、指标），是「按什么约束配出什么组合」的直接证据；
        - 追加 `change_log`：`{"node": "optimize", "round", "candidates", "holdings", "violations"}`；
          注：其中 `"violations": 0` 是**固定的字面量 0**，并非本节点实时统计的违反数，
          真实违反数由各测试与 `eval/run_eval.py` 用 `check_portfolio` 独立复核。

        异常：候选池为空时仍会继续调用优化器（由 `src/optimizer.py` 内部决定产出空持仓），
        不在此处提前报错。
        """
        round_index = int(state.get("round", 0))
        effective_client = state["effective_client"]

        # 第 0 轮：直接复用 node_screen 已算出的候选池，避免重复筛选与重复 trace
        screening_rounds = list(state.get("screening_rounds") or [])
        if round_index == 0 and state.get("screening") is not None:
            screening: ScreeningResult = state["screening"]
        else:
            # 打回重配后的第 1 轮起：用被 TightenSpec 收紧过的 effective_client 重建可行域
            screening = self.agents["ProductScreeningAgent"].run(
                client=effective_client, products=self.products, round_index=round_index
            )["screening"]
            screening_rounds.append(screening.model_dump())

        result = self.agents["PortfolioOptimizerAgent"].run(
            client=effective_client,
            candidates=screening.included,
            products=self.products,
            round_index=round_index,
        )
        portfolio: Portfolio = result["portfolio"]
        portfolio_rounds = list(state.get("portfolio_rounds") or [])
        portfolio_rounds.append(
            {
                "round": round_index,
                "constraints": effective_client.constraint_snapshot(),
                "weights": dict(portfolio.weights),
                "cash_weight": portfolio.cash_weight,
                "metrics": dict(portfolio.metrics),
            }
        )

        change_log = list(state.get("change_log") or [])
        change_log.append(
            {
                "node": "optimize",
                "round": round_index,
                "candidates": len(screening.included),
                "holdings": len(portfolio.held_ids()),
                "violations": 0,
            }
        )

        return {
            "screening": screening,
            "candidates": list(screening.included),
            "screening_rounds": screening_rounds,
            "portfolio": portfolio,
            "portfolio_note": result["portfolio_note"],
            "binding": result["binding"],
            "product_scores": result["product_scores"],
            "portfolio_rounds": portfolio_rounds,
            "change_log": change_log,
        }

    # ------------------------------------------------------------------
    # 节点：适当性闸门
    # ------------------------------------------------------------------
    def node_suitability(self, state: AdvisoryState) -> dict[str, Any]:
        """适当性复核；不通过时下发收紧指令并记录被拦截版本。

        参数：
            state：需含 `client`、`portfolio`（`candidates` 可选）。

        返回：增量字典 `{gate, suitability_comment, tighten, gate_rounds}`，
        打回时额外写 `{effective_client, round}`。

        副作用：
        - 通过 SuitabilityOfficerAgent 写一条 suitability trace；
        - 追加一条 `gate_rounds` 快照（轮次 / directive / blocks / warns / exempted）；
        - 追加 `change_log`：`{"node": "suitability", "round", "directive", "blocks"}`；
        - **`directive == "reoptimize"` 时额外落盘一条 `STATUS_BLOCKED` 版本快照**
          （建议版本链的第一个节点），并追加 `change_log`：`{"node": "repair", ...}`。

        打回重配的闭环（关键）：
        1. `TightenSpec.from_dict(gate.tighten)` 还原收紧指令；
        2. `tighten.apply(state["effective_client"])` 生成**单调收紧**后的新约束
           （等级取更小、上限取更低、流动性下限取更高、禁止项取并集）；
        3. `round` 加一，路由回 `optimize`——因此在更小的可行域里重建候选池与组合。

        轮次上限：本节点不自行判断上限，只把 `round_index` 与
        `config.max_repair_rounds` 一起交给 SuitabilityOfficerAgent；当
        `round_index >= max_repair_rounds` 时闸门返回 directive="reject" 且
        `gate.escalated=True`，于是路由进入 escalate 分支，循环到此终止。
        """
        round_index = int(state.get("round", 0))
        client: ClientProfile = state["client"]
        portfolio: Portfolio = state["portfolio"]

        result = self.agents["SuitabilityOfficerAgent"].run(
            client=client,
            portfolio=portfolio,
            universe=self.products,
            candidates=state.get("candidates") or [],
            round_index=round_index,
            exempt_rules=self.config.exempt_rules,
            max_repair_rounds=self.config.max_repair_rounds,
        )
        gate: GateDecision = result["gate"]

        gate_rounds = list(state.get("gate_rounds") or [])
        gate_rounds.append(
            {
                "round": round_index,
                "directive": gate.directive,
                "blocks": [v.rule_id for v in gate.blocks],
                "warns": [v.rule_id for v in gate.warns],
                "exempted": gate.exempted_rules,
            }
        )

        updates: dict[str, Any] = {
            "gate": gate,
            "suitability_comment": result["suitability_comment"],
            "tighten": result["tighten"],
            "gate_rounds": gate_rounds,
        }

        change_log = list(state.get("change_log") or [])
        change_log.append(
            {
                "node": "suitability",
                "round": round_index,
                "directive": gate.directive,
                "blocks": [v.rule_id for v in gate.blocks],
            }
        )
        updates["change_log"] = change_log

        if gate.directive == "reoptimize":
            # ── 打回重配分支：先留痕被拦截的版本，再收紧约束并把轮次 +1 ──
            # 记录被拦截版本（建议版本链的第一个节点）
            self._save_snapshot(
                client=client,
                portfolio=portfolio,
                run_id=state["run_id"],
                status=STATUS_BLOCKED,
                directive=gate.directive,
                gate=gate,
                screening=state.get("screening"),
                change_reason=(
                    "适当性闸门打回：命中 "
                    + "、".join(dict.fromkeys(v.rule_id for v in gate.blocks))
                    + "，已下发约束收紧指令"
                ),
            )
            tighten = TightenSpec.from_dict(gate.tighten)
            # 单调收紧：只往更严格一侧走，保证多轮打回一定收敛而不是来回震荡
            updates["effective_client"] = tighten.apply(state["effective_client"])
            # 轮次 +1：与 config.max_repair_rounds 比较，构成打回循环的边界
            updates["round"] = round_index + 1
            change_log.append(
                {
                    "node": "repair",
                    "round": round_index,
                    "tighten": gate.tighten,
                    "next_round": round_index + 1,
                }
            )
            updates["change_log"] = change_log

        return updates

    # ------------------------------------------------------------------
    # 节点：超限转人工
    # ------------------------------------------------------------------
    def node_escalate(self, state: AdvisoryState) -> dict[str, Any]:
        """打回重配次数用尽：标记转人工并保留失败原因。

        参数：
            state：需含 `gate`（`GateDecision`）。

        返回：增量字典 `{status: "escalated", escalation_reasons, change_log}`。

        说明：注意此处写入的 `status` 是**中间态字符串 "escalated"**（不是 STATUS_* 常量），
        它表示「已转人工」；最终终态由随后的 `node_human_gate` 覆盖（可修复但超限的情形
        会因 `human_review_reasons` 带入 `escalation_reasons` 而升级为 STATUS_FINAL_ESCALATED）。

        两条失败分支（互斥）：
        - 命中不可修复的 `veto` 规则（例如可行域为空 S-FEASIBLE-POOL）→ 闸门直接拒绝，
          原因为「命中不可修复的 veto 规则」，措辞中不含「打回重配次数用尽」；
        - 否则为 block 级规则在 `round_index + 1` 轮仍未消除（打回次数用尽）。
        """
        gate: GateDecision = state["gate"]
        if gate.veto_rules:
            reasons = [
                "命中不可修复的 veto 规则，闸门直接拒绝："
                + "、".join(gate.veto_rules)
            ]
        else:
            reasons = [
                f"第 {gate.round_index + 1} 轮仍命中 block 级规则（打回重配次数用尽）："
                + "、".join(dict.fromkeys(v.rule_id for v in gate.blocks))
            ]
        change_log = list(state.get("change_log") or [])
        change_log.append({"node": "escalate", "reasons": reasons})
        return {
            "status": "escalated",
            "escalation_reasons": reasons,
            "change_log": change_log,
        }

    # ------------------------------------------------------------------
    # 节点：建议书
    # ------------------------------------------------------------------
    def node_narrative(self, state: AdvisoryState) -> dict[str, Any]:
        """生成压力测试、反事实解释与建议书正文，并落盘最终版本快照。

        参数：
            state：需含 `client`、`portfolio`、`run_id`；可选 `human_review`（缺省视为
            一张「无需确认」的空记录）、`status`（缺省 STATUS_FINAL）、`screening`、
            `gate`、`effective_client`、`engine`。

        返回：增量字典 `{narrative, elements, advice, stress, counterfactual,
        version, parent_version, change_log}`。

        执行顺序要点：本节点在 `node_human_gate` **之后**执行，所以
        `status` / `human_review` 已确定，建议书与快照都能带上人工确认结论。

        副作用：
        - 通过 AdvisorNarrativeAgent 写一条 narrative trace；
        - `store.next_version(client_id)` 分配版本号，`parent_version` 指向上一版；
        - `_save_snapshot(...)` 落盘一条**不可变**版本快照（含约束、组合、规则命中、
          人工确认、压力测试与反事实），变更原因由 `_final_change_reason` 生成；
        - 追加 `change_log`：`{"node": "narrative", "version", "elements_ok", "elements_total"}`。

        异常/降级：无前序版本时 `parent_version` 为 None；`directive` 的取值为
        `state["gate"].directive if state.get("gate") else "pass"`——注：实际实现先按下标取
        `state["gate"]` 再判别真值，在 `total=False` 的 schema 下若 state 里完全没有该键
        会先抛 KeyError；两条引擎路径在进入本节点前都必然已有 `gate`，正常流程不会触发。
        """
        client: ClientProfile = state["client"]
        portfolio: Portfolio = state["portfolio"]
        review: HumanReview = state.get("human_review") or HumanReview()
        status: str = state.get("status") or STATUS_FINAL
        # 先分配版本号（同一节点内的 _save_snapshot 会复用该号，避免重复自增）
        version, parent_version = self.store.next_version(client.client_id)
        # 前序版本摘要：让建议书能展示"相比上一版改了什么"
        prior = chain_rows(self.store.chain(client.client_id))

        result = self.agents["AdvisorNarrativeAgent"].run(
            client=client,
            portfolio=portfolio,
            products=self.products,
            stress_config=self.data.stress_config,
            screening=state.get("screening"),
            gate=state.get("gate"),
            human_review=review,
            status=status,
            version=version,
            engine=state.get("engine", self.config.engine),
            run_id=state["run_id"],
            prior_versions=prior,
            counterfactual_client=state.get("effective_client"),
        )

        change_log = list(state.get("change_log") or [])
        change_log.append(
            {
                "node": "narrative",
                "version": version,
                "elements_ok": sum(1 for ok in result["elements"].values() if ok),
                "elements_total": len(result["elements"]),
            }
        )

        self._save_snapshot(
            client=client,
            portfolio=portfolio,
            run_id=state["run_id"],
            status=status,
            directive=(state["gate"].directive if state.get("gate") else "pass"),
            gate=state.get("gate"),
            screening=state.get("screening"),
            human_review=review,
            stress=result["stress"],
            counterfactual=result["counterfactual"],
            narrative=result["narrative"],
            change_reason=self._final_change_reason(client.client_id, state),
            version=version,
        )

        return {
            "narrative": result["narrative"],
            "elements": result["elements"],
            "advice": result["advice"],
            "stress": result["stress"],
            "counterfactual": result["counterfactual"],
            "version": version,
            "parent_version": parent_version,
            "change_log": change_log,
        }

    # ------------------------------------------------------------------
    # 节点：人机协同闸门
    # ------------------------------------------------------------------
    def node_human_gate(self, state: AdvisoryState) -> dict[str, Any]:
        """人工确认（或按配置放行）。

        参数：
            state：需含 `client`、`portfolio`；可选 `gate`、`escalation_reasons`。

        返回：增量字典 `{human_review, status, change_log}`。

        闸门在建议书定稿**之前**执行：未通过人工确认（或被否决）时，
        最终状态不会是正常定稿，从而保证「必须人工确认后才能定稿」。

        位置与语义（重要）：
        - 在拓扑上位于 `suitability`/`escalate` 之后、`narrative` 之前；
        - 因此 `node_narrative` 能读到已确定的人工确认结论，把它同时写进结构化建议
          （`AdviceRecord.human_review`）与建议书正文（「人工确认记录」章节）；
        - 本节点**不阻塞流程**：无论放行、降级还是否决，都会继续走 narrative，
          只是终态 `status` 不同（否决记为 STATUS_REJECTED_BY_HUMAN）。

        状态判定优先级（后者覆盖前者）：
        1. 默认按 `review.decision` 查 `status_map`
           （not_required/approved/auto_approved → STATUS_FINAL，
           auto_degraded → STATUS_FINAL_DEGRADED，rejected → STATUS_REJECTED_BY_HUMAN）；
        2. 若 `gate.directive == "reject"`（含 veto 与超限转人工）→ 覆盖为 STATUS_REJECTED，
           即**合规拒绝优先于人工结论**；
        3. 若存在 `escalation_reasons` 且当前仍为 STATUS_FINAL → 升级为 STATUS_FINAL_ESCALATED
           （表明这份建议是「打回次数用尽转人工后」才定稿的）。

        副作用：追加 `change_log`：`{"node": "human_gate", "required", "decision", "status"}`。
        """
        gate: GateDecision | None = state.get("gate")
        review = self.decide_human_review(state)

        # 人工结论 → 终态映射；合规拒绝（directive == "reject"）具有更高优先级，见下
        status_map = {
            "not_required": STATUS_FINAL,
            "approved": STATUS_FINAL,
            "auto_approved": STATUS_FINAL,
            "auto_degraded": STATUS_FINAL_DEGRADED,
            "rejected": STATUS_REJECTED_BY_HUMAN,
        }
        if gate is not None and gate.directive == "reject":
            # 合规拒绝优先于人工结论：即使理财经理同意放行，也不能定稿
            status = STATUS_REJECTED
        else:
            status = status_map.get(review.decision, STATUS_FINAL)
        if state.get("escalation_reasons") and status == STATUS_FINAL:
            # 打回次数用尽后转人工、且最终放行定稿 → 单独标记，便于事后追溯
            status = STATUS_FINAL_ESCALATED

        change_log = list(state.get("change_log") or [])
        change_log.append(
            {
                "node": "human_gate",
                "required": review.required,
                "decision": review.decision,
                "status": status,
            }
        )
        return {"human_review": review, "status": status, "change_log": change_log}

    # ------------------------------------------------------------------
    # 人机协同判定
    # ------------------------------------------------------------------
    def human_review_reasons(self, state: AdvisoryState) -> list[str]:
        """判断是否需要人工确认，并给出全部触发原因。

        参数：
            state：需含 `client`、`portfolio`；可选 `gate`、`escalation_reasons`。

        返回：去重（保持首次出现顺序）后的原因列表；**列表为空即表示无需人工确认**。

        三类触发情形（可同时命中，故用列表而非布尔）：
        1. `gate.exempted_rules` 非空——命中 block 级规则后被人工豁免，豁免必须留痕；
        2. `state["escalation_reasons"]` 非空——打回重配次数用尽转人工；
        3. 任一持仓的 `weight > client.warning_ratio + 1e-9`——单一产品集中度超过内控预警线
           （比较时带 1e-9 容差以避免浮点误差误判；预警线见 `ClientProfile.warning_ratio`，
           未显式配置时取 `max_single_product_ratio * 0.8`）；
        另外：`client.is_elderly`（年满 65 周岁）时追加「特别保护确认」原因。

        副作用：无（纯计算，不写 state、不落盘）。
        """
        reasons: list[str] = []
        gate: GateDecision | None = state.get("gate")
        client: ClientProfile = state["client"]
        portfolio: Portfolio = state["portfolio"]

        if gate is not None and gate.exempted_rules:
            reasons.append(
                "命中 block 级适当性规则并经人工豁免：" + "、".join(gate.exempted_rules)
            )
        if state.get("escalation_reasons"):
            reasons.extend(state["escalation_reasons"])
        for pid in portfolio.held_ids():
            weight = portfolio.weights[pid]
            if weight > client.warning_ratio + 1e-9:
                reasons.append(
                    f"单一产品集中度 {pct(weight)} 超过内控预警线 {pct(client.warning_ratio)}"
                    f"（{portfolio.products[pid].name}）"
                )
        if client.is_elderly:
            reasons.append(f"客户为高龄客户（{client.age} 周岁），须执行特别保护确认")
        return list(dict.fromkeys(reasons))

    def decide_human_review(self, state: AdvisoryState) -> HumanReview:
        """执行人工确认流程：`--auto` 自动放行；非交互环境自动降级并留痕。

        参数：
            state：需含 `client`、`portfolio`（内部转交 `human_review_reasons`）。

        返回：`HumanReview` 记录，三条互斥路径：
        - 无任何触发原因 → `required=False, decision="not_required", operator="system"`；
        - `config.auto` 为真 → `decision="auto_approved"`，operator 记为 `"auto(--auto)"`，
          用于演示，仍写出完整 reasons 留痕；
        - 否则若 `config.resolve_interactive()` 为真 → 交互式询问
          `input("检测到需人工确认事项，是否放行？(y/N) ")`，
          首尾去空白并转小写，落在 {"y", "yes"} 记 `"approved"`，其余（含直接回车）记 `"rejected"`；
        - 否则（非交互环境）→ `decision="auto_degraded"` 且 `degraded=True`，
          即「自动降级放行 + 留痕 + 提示需后续人工复核」。

        三条路径都会把 `gate.exempted_rules` 抄进 `HumanReview.exempted_rules`，
        并写入 `decided_at=now_iso()`。

        副作用：交互路径会阻塞等待 stdin；`resolve_interactive()` 抛出的异常已在其内部吞掉。
        """
        reasons = self.human_review_reasons(state)
        if not reasons:
            return HumanReview(required=False, decision="not_required", operator="system")

        if self.config.auto:
            return HumanReview(
                required=True,
                reasons=reasons,
                decision="auto_approved",
                operator="auto(--auto)",
                note="演示模式（--auto）自动放行，已记录确认留痕",
                decided_at=now_iso(),
                exempted_rules=list((state.get("gate").exempted_rules if state.get("gate") else []) or []),
            )

        if self.config.resolve_interactive():
            answer = input("检测到需人工确认事项，是否放行？(y/N) ").strip().lower()
            approved = answer in {"y", "yes"}
            return HumanReview(
                required=True,
                reasons=reasons,
                decision="approved" if approved else "rejected",
                operator=self.config.operator,
                note="交互式人工确认" + ("：同意放行" if approved else "：否决"),
                decided_at=now_iso(),
                exempted_rules=list((state.get("gate").exempted_rules if state.get("gate") else []) or []),
            )

        return HumanReview(
            required=True,
            reasons=reasons,
            decision="auto_degraded",
            operator=self.config.operator,
            degraded=True,
            note="非交互环境无法完成人工确认，按配置自动降级放行并留痕，需后续人工复核",
            decided_at=now_iso(),
            exempted_rules=list((state.get("gate").exempted_rules if state.get("gate") else []) or []),
        )

    # ------------------------------------------------------------------
    # 版本链
    # ------------------------------------------------------------------
    def _final_change_reason(self, client_id: str, state: AdvisoryState) -> str:
        """生成最终版本的变更原因。

        参数：
            client_id：客户号，用于取该客户的建议版本链。
            state：共享状态（只读 `escalation_reasons`）。

        返回：四种中文变更原因之一，按下列优先级判定：
        - 版本链为空 → "首次生成建议"；
        - 链尾状态为 `STATUS_BLOCKED` → "按适当性闸门下发的约束收紧指令重新配置后通过并定稿"
          （即打回重配路径；注：实际实现的判定顺序是**先看链尾是否 blocked、后看是否转人工**，
          因此同一份建议既被打回过又走了转人工时，会记成「按收紧指令重配」）；
        - 否则若 `state["escalation_reasons"]` 非空 → "超限转人工处理后定稿"；
        - 其余 → "重新生成建议（重跑流水线）"。

        副作用：无（只读版本链）。
        """
        chain = self.store.chain(client_id)
        if not chain:
            return "首次生成建议"
        last = chain[-1]
        if last.status == STATUS_BLOCKED:
            return "按适当性闸门下发的约束收紧指令重新配置后通过并定稿"
        if state.get("escalation_reasons"):
            return "超限转人工处理后定稿"
        return "重新生成建议（重跑流水线）"

    def _save_snapshot(
        self,
        *,
        client: ClientProfile,
        portfolio: Portfolio,
        run_id: str,
        status: str,
        directive: str,
        gate: GateDecision | None,
        screening: ScreeningResult | None = None,
        human_review: HumanReview | None = None,
        stress: Any = None,
        counterfactual: Any = None,
        narrative: str = "",
        change_reason: str = "",
        version: int | None = None,
    ) -> AdviceSnapshot:
        """保存一条不可变的建议版本快照。

        参数（除 version 外均为关键字参数）：
            client：客户档案（写入快照的约束部分）。
            portfolio：方案组合（写入快照的权重与指标部分）。
            run_id：本次运行标识。
            status：该快照的状态（终态常量，或打回时的 `STATUS_BLOCKED`）。
            directive：闸门结论字符串（"pass" / "reoptimize" / "reject"）。
            gate：闸门结论对象，可为 None。
            screening：筛选结果，可为 None。
            human_review：人工确认记录，可为 None。
            stress / counterfactual：压力测试与反事实报告，可为 None。
            narrative：建议书正文，默认空串。
            change_reason：变更原因说明，默认空串。
            version：显式指定版本号；为 None 时自动分配。

        返回：写入后的 `AdviceSnapshot`（含 version / parent_version / payload / status）。

        版本号与父子关系：
        - `version is None` 时调用 `store.next_version()`，同时拿到自增版本号与父版本号；
        - 显式传入 `version` 时（`node_narrative` 的用法，版本号已在同一节点内分配过一次，
          避免重复自增），父版本取当前链尾的 `version`，链为空则 None。

        副作用：`store.append(snapshot)` 追加落盘一条 JSONL 记录；快照本身不可变，
        用于事后回溯「当时按什么约束、什么规则命中配出的这份建议」。
        """
        if version is None:
            version, parent = self.store.next_version(client.client_id)
        else:
            chain = self.store.chain(client.client_id)
            parent = chain[-1].version if chain else None
        payload = build_payload(
            client,
            portfolio,
            run_id=run_id,
            engine=self.config.engine,
            status=status,
            directive=directive,
            screening=screening,
            gate=gate,
            human_review=human_review,
            stress=stress,
            counterfactual=counterfactual,
            narrative=narrative,
        )
        snapshot = AdviceSnapshot.create(
            version=version,
            parent_version=parent,
            client_id=client.client_id,
            run_id=run_id,
            change_reason=change_reason,
            payload=payload,
            status=status,
        )
        self.store.append(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------
    def route_after_suitability(self, state: AdvisoryState) -> str:
        """闸门后的条件路由（两个引擎共用同一判定）。

        参数：
            state：需含 `gate`。

        返回：路由键（也是 LangGraph `ROUTE_MAP` 的键、native 引擎的分支条件）：
        - `"pass"`（directive == "pass"）→ `"human_gate"`；
        - `"reoptimize"` → `"optimize"`（打回重配，回到组合构建节点）；
        - 其余（即 `"reject"`，含 veto 直接拒绝与超限转人工）→ `"escalate"`。

        说明：这是全流程唯一的循环出口判定，两个引擎都调用它，
        因此「LangGraph 与 native 结果一致」由同一段代码保证。
        """
        gate: GateDecision = state["gate"]
        if gate.directive == "pass":
            # 闸门放行 → 先过人工确认闸门，再由 narrative 定稿
            return "human_gate"
        if gate.directive == "reoptimize":
            # 打回重配 → 回到 optimize（effective_client 已被收紧、round 已 +1）
            return "optimize"
        # reject：命中 veto 规则，或 block 规则在 max_repair_rounds 轮后仍未消除
        return "escalate"


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------
def build_pipeline(
    *,
    data_dir: Any = None,
    engine: str = "langgraph",
    auto: bool = False,
    max_repair_rounds: int = 2,
    exempt_rules: tuple[str, ...] = (),
    interactive: bool | None = None,
    runs_dir: Any = None,
    store: VersionStore | None = None,
    llm: BaseLLM | None = None,
) -> tuple[AdvisoryPipeline, Tracer]:
    """构建流水线与 tracer。

    参数（全部为关键字参数）：
        data_dir：样例数据目录；None 表示用 `src/dataset.py` 里的默认 `DATA_DIR`。
        engine：请求的引擎名，默认 "langgraph"；交给 `resolve_engine` 解析（可能降级）。
        auto：是否演示模式自动放行人工确认，默认 False。
        max_repair_rounds：**打回重配轮次上限**，默认 2，写入 `PipelineConfig`。
        exempt_rules：人工豁免的 block 规则号元组，默认空。
        interactive：人工确认是否交互；None 表示按 stdin TTY 自动判定。
        runs_dir：trace 输出目录；None 表示用 `Tracer` 的默认 RUNS_DIR。
        store：版本链存储；None 表示用默认 `VersionStore()`。
        llm：语言模型；None 表示 `build_llm()`（无 Key 自动 mock）。

    返回：`(AdvisoryPipeline, Tracer)`。

    副作用：
    - `load_dotenv()` 读取 .env（不覆盖已存在的环境变量）；
    - `load_data(data_dir)` 读盘装载样例数据；
    - 创建 `Tracer(run_id=new_run_id("advisory"), runs_dir=runs_dir)`，即此刻就生成了 run_id；
    - 把 tracer 注入到 pipeline 与**全部五个 Agent**，保证各步骤都往同一个 run 落 trace。

    异常：`resolve_engine` 对不支持的引擎名抛 `ValueError`；数据文件缺失/主键重复由
    `load_data` 抛异常。
    """
    load_dotenv()
    from .engine import resolve_engine

    resolved_engine, engine_note = resolve_engine(engine)
    data = load_data(data_dir)
    config = PipelineConfig(
        engine=resolved_engine,
        auto=auto,
        max_repair_rounds=max_repair_rounds,
        exempt_rules=tuple(exempt_rules),
        interactive=interactive,
    )
    pipeline = AdvisoryPipeline(data, llm=llm, tracer=None, store=store, config=config)
    pipeline.engine_note = engine_note
    tracer = Tracer(run_id=new_run_id("advisory"), runs_dir=runs_dir)
    pipeline.tracer = tracer
    for agent in pipeline.agents.values():
        agent.tracer = tracer
    return pipeline, tracer


def run_pipeline(
    client_id: str,
    *,
    engine: str = "langgraph",
    auto: bool = True,
    data_dir: Any = None,
    runs_dir: Any = None,
    store: VersionStore | None = None,
    max_repair_rounds: int = 2,
    exempt_rules: tuple[str, ...] = (),
    run_id: str | None = None,
    interactive: bool | None = False,
) -> tuple[AdvisoryState, Tracer, AdvisoryPipeline]:
    """按指定引擎跑完一位客户的完整投顾流水线。

    参数（除 client_id 外均为关键字参数）：
        client_id：客户号。
        engine：请求的引擎名，默认 "langgraph"；注意此处的分支判断用的是**传入值**
            （`engine == "native"` 走降级引擎，其余一律走 LangGraph），而
            `build_pipeline` 内部可能已把实际引擎降级为 native 并写在 `pipeline.engine_note` 里。
        auto：默认 **True**（与 `build_pipeline` 的默认 False 不同），便于评估脚本无人值守跑通。
        data_dir / runs_dir / store：分别透传给 `build_pipeline`。
        max_repair_rounds：打回重配轮次上限，默认 2。
        exempt_rules：人工豁免的 block 规则号元组。
        run_id：显式指定本次 run_id；为 None 时用 `build_pipeline` 新建的那个。
        interactive：是否交互确认，默认 **False**（评估/CI 环境不阻塞）。

    返回：`(AdvisoryState, Tracer, AdvisoryPipeline)`——终态、执行痕迹、流水线对象
    （可从 `pipeline.store` 读取版本链）。

    副作用：读样例数据、写 trace jsonl、追加建议版本快照；若 `run_id` 非空会覆盖
    `tracer.run_id`（后续 trace 文件名随之改变）。
    """
    pipeline, tracer = build_pipeline(
        data_dir=data_dir,
        engine=engine,
        auto=auto,
        max_repair_rounds=max_repair_rounds,
        exempt_rules=exempt_rules,
        interactive=interactive,
        runs_dir=runs_dir,
        store=store,
    )
    if run_id:
        tracer.run_id = run_id

    if engine == "native":
        state = run_native(pipeline, client_id, tracer.run_id)
    else:
        from .engine.langgraph_engine import run_langgraph

        state = run_langgraph(pipeline, client_id, tracer.run_id)
    return state, tracer, pipeline


def run_native(pipeline: AdvisoryPipeline, client_id: str, run_id: str) -> AdvisoryState:
    """降级引擎（零第三方依赖）：手动状态机，与 LangGraph 路径完全一致。

    参数：
        pipeline：已装配的 `AdvisoryPipeline`（提供节点函数与路由）。
        client_id：客户号。
        run_id：运行标识，写入初始状态与 trace。

    返回：跑完后的 `AdvisoryState`。

    说明：本函数是**薄封装**，真正的状态机在 `src/engine/native.py`（含 `MAX_STEPS = 64`
    步数上限防御死循环）。函数体内延迟 import，避免与引擎包形成循环依赖。
    """
    from .engine.native import run_native as _run

    return _run(pipeline, client_id, run_id)


def state_summary(state: AdvisoryState) -> dict[str, Any]:
    """把最终状态压缩成可比较、可展示的摘要（评估与双引擎对照共用）。

    参数：
        state：跑完后的共享状态（各键均可缺失）。

    返回：扁平字典，字段与 `tests/test_pipeline.py::test_state_summary_shape` 的
    `expected_keys` 一一对应，主要换算规则：
    - `rounds = round + 1`（轮次展示为人类从 1 开始的计数）；
    - `weights` / `metrics` 为 `Portfolio` 的拷贝，无组合时为空/默认值；
    - `block_rules` 保留原始顺序，`warn_rules` 做去重；
    - `escalated` 取 `gate.escalated`（注意与终态 STATUS_FINAL_ESCALATED 不是同一含义：
      前者表示「闸门因打回次数用尽而升级」，后者表示「最终建议是转人工后定稿的」）；
    - `human_required` / `human_decision` 来自 `human_review`，无记录时为 False / None；
    - `elements_ok` 统计 12 项要素中达标项数；`stress_scenarios` / `counterfactual_variants`
      分别取报告条数；`narrative_length` 为正文长度；
    - `trace_steps` 恒为 None（trace 步数需另取 `tracer.step_count()`，此处不持有 tracer）。

    副作用：无；`weights` / `metrics` 是拷贝，调用方修改不会影响 state。
    """
    gate: GateDecision | None = state.get("gate")
    portfolio: Portfolio | None = state.get("portfolio")
    advice: AdviceRecord | None = state.get("advice")
    return {
        "client_id": state.get("client_id"),
        "engine": state.get("engine"),
        "status": state.get("status"),
        "rounds": int(state.get("round", 0)) + 1,
        "candidates": list(state.get("candidates") or []),
        "weights": dict(portfolio.weights) if portfolio else {},
        "cash_weight": portfolio.cash_weight if portfolio else 0.0,
        "metrics": dict(portfolio.metrics) if portfolio else {},
        "directive": gate.directive if gate else None,
        "block_rules": [v.rule_id for v in gate.blocks] if gate else [],
        "warn_rules": list(dict.fromkeys(v.rule_id for v in gate.warns)) if gate else [],
        "escalated": bool(gate.escalated) if gate else False,
        "human_required": bool(state.get("human_review") and state["human_review"].required),
        "human_decision": state["human_review"].decision if state.get("human_review") else None,
        "version": state.get("version"),
        "elements_ok": sum(1 for ok in (state.get("elements") or {}).values() if ok),
        "stress_scenarios": len(state["stress"].scenarios) if state.get("stress") else 0,
        "counterfactual_variants": (
            len(state["counterfactual"].variants) if state.get("counterfactual") else 0
        ),
        "narrative_length": len(state.get("narrative") or ""),
        "advice_status": advice.status if advice else None,
        "trace_steps": None,
    }
