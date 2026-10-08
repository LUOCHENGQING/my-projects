"""Agent 工具白名单与权限集合。

层级
----
工具契约层（`src/agents/tools.py`）：位于 Agent 实现（`base.py` 与五个 Agent）
与确定性业务模块（`..constraints` / `..optimizer` / `..suitability` / `..stress` /
`..counterfactual`）之间，是 Agent 触碰业务能力的**唯一通道**。

每个 Agent 有：
- **独立 system prompt**（角色与输出边界）
- **独立工具白名单**（`tools`）：只能调用白名单内的工具
- **独立权限集合**（`permissions`）：工具本身还带权限标签，双重校验
- **可写 state 键白名单**（`can_write_state`）：最小权限原则，越权写共享状态直接报错

这样做的目的是让"谁能做什么"变成可断言的数据，而不是散落在代码里的约定。

对外暴露
--------
- 权限常量：`PERM_*`（11 个，见下）
- `Tool` / `ToolRegistry` / `TOOL_REGISTRY`（默认注册表）/ `ALL_TOOLS`（全量工具表）
- `AgentSpec`：Agent 能力契约
- 便捷函数：`tool_names()` / `registry_tools()` / `tighten_from_payload()`
- 转出 `binding_constraints`（来自 `..optimizer`，供调用方免于直接依赖优化器模块）

工具注册表怎么工作
------------------
`ALL_TOOLS` 是 17 个 `Tool` 的不可变元组，`TOOL_REGISTRY = ToolRegistry(ALL_TOOLS)`
在导入时按 `name` 建索引。调用链固定为
`BaseAgent.call_tool(name)` → **先查 `AgentSpec.tools` 白名单**（不在则 `PermissionError`）
→ **再查 `AgentSpec.permissions` 是否含该工具的 `permission` 标签**（不含则 `PermissionError`）
→ 最后才执行 `tool.handler(**kwargs)`。「双重校验」指的就是这两道独立检查。

权限标签（`Tool.permission`，名称与取值必须一致）
-------------------------------------------------
`profile:read`（读客户档案/解析问卷）、`profile:derive`（派生约束/画像文案）、
`screening:execute`（执行筛选）、`screening:read`（读剔除原因）、
`optimize:compute`（打分与求解权重）、`optimize:read`（读优化结果，本模块未分配给任何工具）、
`suitability:judge`（求值规则/构造收紧）、`suitability:veto`（出具闸门结论，可否决）、
`narrative:write`（产出文案）、`narrative:analyze`（压力测试与反事实）、
`trace:write`（写 trace）。
其中 `PERM_OPTIMIZE_READ` 与 `PERM_TRACE_WRITE` 被导出但**没有**绑定到任何工具：
`trace:write` 是给 Agent 写 trace 的权限声明，`optimize:read` 目前为预留。

幂等与超时约束
--------------
- **幂等**：所有 `_tool_*` 处理器都是对确定性函数的薄封装，不写文件、不改全局状态，
  同一入参重复调用得到**相同结果**（`profile.read_client` 还显式返回
  `model_copy(deep=True)` 深拷贝，避免下游误改缓存后让后续调用结果漂移）。
  因此多轮打回重配时可以安全地重复调用同一工具，不需要去重或补偿逻辑。
- **无隐式重试**：`ToolRegistry.invoke` / `get` 只做一次调用，失败即原样抛错，不重试、不吞异常；
  适当性打回轮次的上限由 `SuitabilityGate(max_repair_rounds=...)` 控制，而不是工具层。
- **无工具级超时控制**：工具层不设超时。唯一的超时在模型接入层——
  `src/llm.py` 的 `LLMConfig.timeout`（默认 30 秒，环境变量 `LLM_TIMEOUT`），
  由 `urllib.request.urlopen` 施加；`_tool_compose_text` 是唯一可能访问外部的工具，
  其失败由 `FallbackLLM` 降级到确定性 mock。

被谁使用
--------
`src/agents/base.py`（`BaseAgent` 持有 `registry` 并做双重校验）与五个 Agent 模块
（各自声明 `SPEC = AgentSpec(...)`）；`src/demo.py --catalog` 打印工具与权限清单。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from ..constraints import TightenSpec, screen_products
from ..counterfactual import analyze_counterfactual
from ..dataset import level_label, score_to_level
from ..optimizer import binding_constraints, build_portfolio, product_score
from ..schemas import (
    ClientProfile,
    GateDecision,
    Portfolio,
    Product,
    ScreeningResult,
    StressReport,
)
from ..stress import run_stress
from ..suitability import RuleContext, RuleEvaluation, SuitabilityGate, build_tighten, evaluate_rules

# ---------------------------------------------------------------------------
# 权限标签
# 每个标签用 `<资源>:<动作>` 命名，由 `Tool.permission` 携带，并在
# `BaseAgent.call_tool` 中与 `AgentSpec.permissions` 比对（第二道校验）。
# 注意：标签字符串是权限体系的契约值，改动会直接导致 Agent 越权报错。
# ---------------------------------------------------------------------------
PERM_PROFILE_READ = "profile:read"
PERM_PROFILE_DERIVE = "profile:derive"
PERM_SCREEN_EXECUTE = "screening:execute"
PERM_SCREEN_READ = "screening:read"
PERM_OPTIMIZE_COMPUTE = "optimize:compute"
PERM_OPTIMIZE_READ = "optimize:read"
PERM_SUIT_JUDGE = "suitability:judge"
PERM_SUIT_VETO = "suitability:veto"
PERM_NARRATIVE_WRITE = "narrative:write"
PERM_NARRATIVE_ANALYZE = "narrative:analyze"
PERM_TRACE_WRITE = "trace:write"


@dataclass(frozen=True)
class Tool:
    """一个可被 Agent 调用的工具。

    不可变（`frozen=True`），在模块导入时随 `ALL_TOOLS` 一次性建好。

    关键属性：
        name：工具名，形如 `screening.filter_universe`；`ToolRegistry` 的索引键，
            也是 `AgentSpec.tools` 白名单里书写的名字。
        permission：权限标签，取值必须是上面某个 `PERM_*` 常量；调用前会被校验。
        description：中文用途说明（供文档与 `--catalog` 展示，不参与逻辑）。
        handler：真正执行的函数，必须能接受关键字参数调用（`handler(**kwargs)`）；
            本模块的处理器一律用关键字-only 参数声明。

    被谁使用：`ToolRegistry`（建索引与调用）、`AgentSpec.tools`（白名单引用其 `name`）。
    """

    name: str
    permission: str
    description: str
    handler: Callable[..., Any]


class ToolRegistry:
    """工具注册表：按名字调用，未登记即报错。

    关键属性：
        _tools：`{工具名: Tool}` 字典，构造时由 `Iterable[Tool]` 一次性建立
            （同名工具后者覆盖前者）。

    被谁使用：`BaseAgent.registry`（默认取模块级 `TOOL_REGISTRY`）；
        适当性打回时重复调用同一工具不会产生副作用，见模块头"幂等与超时约束"。
    """

    def __init__(self, tools: Iterable[Tool]) -> None:
        """初始化注册表。

        参数：
            tools：工具可迭代对象；按 `tool.name` 建索引，重复名以最后一个为准。
        """
        self._tools: dict[str, Tool] = {tool.name: tool for tool in tools}

    def get(self, name: str) -> Tool:
        """取工具；不存在时抛 KeyError。

        参数：
            name：工具名。
        返回：
            对应的 `Tool` 对象。
        异常：名字未登记时抛 `KeyError("未登记的工具：<name>")`。
        """
        if name not in self._tools:
            raise KeyError(f"未登记的工具：{name}")
        return self._tools[name]

    def invoke(self, name: str, **kwargs: Any) -> Any:
        """调用工具。

        参数：
            name：工具名。
            **kwargs：原样透传给 `Tool.handler` 的关键字参数。
        返回：
            处理器的返回值（原始对象，不做包装或校验）。
        异常：名字未登记抛 `KeyError`；参数不匹配或业务校验失败由处理器自身抛出，
            **不重试、不吞异常**。
        说明：`BaseAgent.call_tool` 走的是白名单 + 权限双重校验后的路径，
            本方法本身不做权限检查，只做名字解析与调用。
        """
        return self.get(name).handler(**kwargs)

    def names(self) -> list[str]:
        """全部工具名（排序）。

        返回：按字典序排序的工具名列表（新列表，修改不影响注册表）。
        """
        return sorted(self._tools)

    def __contains__(self, name: object) -> bool:
        """支持 `name in registry` 判断工具是否已登记。

        参数：name：任意对象（非字符串也安全，返回 False）。
        返回：是否在注册表中。
        """
        return name in self._tools


# ---------------------------------------------------------------------------
# 工具实现：全部是对确定性能力的薄封装（工具不是装饰）
# 约定：处理器只做参数整理与转发，不在这里做业务判断，也不写文件/全局状态，
#       因此同一入参重复调用结果一致（幂等），便于适当性打回后重复执行。
# ---------------------------------------------------------------------------
def _tool_read_client(*, client: ClientProfile) -> ClientProfile:
    """读取客户档案（返回深拷贝，避免下游误改缓存）。

    参数：client：客户档案对象。
    返回：`client.model_copy(deep=True)` 的**深拷贝**；调用方修改返回值不会影响原对象，
        这是本工具幂等性的关键（否则下游改动会让后续读取漂移）。
    副作用：无。
    """
    return client.model_copy(deep=True)


def _tool_parse_questionnaire(
    *, client: ClientProfile, questionnaire: Mapping[str, Any]
) -> dict[str, Any]:
    """解析风险测评问卷，折算等级。

    参数：
        client：客户档案，只用到 `client_id` 以定位该客户的作答。
        questionnaire：问卷原始数据，读取 `responses`（`{客户号: {题号: 得分}}`）
            与 `items`（题目列表）。
    返回：
        `{client_id, item_count, answers, score, level, level_label}`：
        `answers` 为该客户的作答字典副本；`score` 为各题得分之和（无作答记录时为 None）；
        `level` 由 `score_to_level()` 折算（score 为 None 时该函数自行处理）；
        `level_label` 为等级中文标签。
    副作用：无；不修改 `questionnaire`。
    """
    responses = questionnaire.get("responses", {}).get(client.client_id) or {}
    score = int(sum(responses.values())) if responses else None
    level = score_to_level(score, dict(questionnaire))
    return {
        "client_id": client.client_id,
        "item_count": len(questionnaire.get("items", [])),
        "answers": dict(responses),
        "score": score,
        "level": level,
        "level_label": level_label(level, dict(questionnaire)),
    }


def _tool_derive_constraints(
    *, client: ClientProfile, questionnaire: Mapping[str, Any]
) -> dict[str, Any]:
    """抽取硬约束与软偏好，并按从严原则给出有效风险等级。

    参数：
        client：客户档案（只读）。
        questionnaire：问卷原始数据，内部转交给 `_tool_parse_questionnaire`。

    返回：
        含以下键的字典：
        `questionnaire`（解析结果）、`archived_level`（档案登记等级）、
        `effective_level`（有效等级 = `min(档案等级, 问卷折算等级)`，问卷无记录时取档案等级）、
        `tightened_by_questionnaire`（问卷是否确实收紧了等级）、
        `hard_constraints`（风险等级/期限/流动性下限/三项集中度上限/可投金额/禁止项/
        经验品类/合格投资者/币种/税收优惠额度）、
        `soft_preferences`（收益目标/最大回撤容忍/费率预算）、
        `protection_flags`（`is_elderly` / `dual_record_completed` / `internal_warning_ratio`）。

    副作用：无（不修改 `client`；返回的列表字段是新列表）。
    说明：本函数内部会**直接调用** `_tool_parse_questionnaire`（而非经注册表），
        因此 `ClientProfilingAgent` 的 trace 中会同时登记
        `profile.parse_questionnaire` 与 `profile.derive_constraints` 两个工具名。
    """
    parsed = _tool_parse_questionnaire(client=client, questionnaire=questionnaire)
    scored = parsed["level"]
    effective = client.risk_capacity if scored is None else min(client.risk_capacity, int(scored))
    return {
        "questionnaire": parsed,
        "archived_level": client.risk_capacity,
        "effective_level": effective,
        "tightened_by_questionnaire": bool(scored is not None and int(scored) < client.risk_capacity),
        "hard_constraints": {
            "risk_capacity": effective,
            "investment_horizon_years": client.investment_horizon_years,
            "liquidity_floor_ratio": client.liquidity_floor_ratio,
            "max_single_product_ratio": client.max_single_product_ratio,
            "max_single_class_ratio": client.max_single_class_ratio,
            "max_single_issuer_ratio": client.max_single_issuer_ratio,
            "investable_amount": client.investable_amount,
            "prohibited_categories": list(client.prohibited_categories),
            "prohibited_product_ids": list(client.prohibited_product_ids),
            "experienced_categories": list(client.experienced_categories),
            "qualified_investor": client.qualified_investor,
            "currency": client.currency,
            "tax_advantaged_quota": client.tax_advantaged_quota,
        },
        "soft_preferences": {
            "return_target": client.return_target,
            "max_drawdown_tolerance": client.max_drawdown_tolerance,
            "annual_fee_budget_ratio": client.annual_fee_budget_ratio,
        },
        "protection_flags": {
            "is_elderly": client.is_elderly,
            "dual_record_completed": client.dual_record_completed,
            "internal_warning_ratio": client.warning_ratio,
        },
    }


def _tool_filter_universe(
    *, client: ClientProfile, products: Mapping[str, Product], round_index: int = 0
) -> ScreeningResult:
    """在硬约束可行域内筛选候选池。

    参数：
        client：客户档案（通常已按从严原则收紧）。
        products：全市场产品表 `{产品号: Product}`。
        round_index：适当性打回轮次，默认 0；用于让剔除记录带上轮次信息。
    返回：
        `ScreeningResult`（含 `universe_size` / `included` / `excluded` 等）。
    副作用：无（纯计算，转发给 `constraints.screen_products`）。
    """
    return screen_products(client, products, round_index)


def _tool_explain_exclusion(*, screening: ScreeningResult, limit: int = 3) -> list[str]:
    """给出主要剔除原因（用于向客户解释）。

    参数：
        screening：筛选结果。
        limit：最多返回条数，默认 3（按 `screening.excluded` 的既有顺序取前 N 条）。
    返回：
        `["产品名：剔除说明", ...]` 形式的字符串列表；不足 limit 条时返回实际条数，不报错。
    副作用：无。
    """
    return [f"{item.product_name}：{item.detail}" for item in screening.excluded[:limit]]


def _tool_score_products(
    *, client: ClientProfile, products: Mapping[str, Product]
) -> dict[str, float]:
    """按多目标口径给产品打分。

    参数：
        client：客户档案（提供风险等级、期限等打分依据）。
        products：待打分产品表。
    返回：
        `{产品号: 分数}`，分数按 `round(..., 12)` 规整（与 `utils.QUANTIZE` 口径一致，
        保证摘要与比较稳定）；产品号按字典序遍历，因此键顺序确定。
    副作用：无。
    """
    return {pid: round(product_score(products[pid], client), 12) for pid in sorted(products)}


def _tool_solve_weights(
    *, client: ClientProfile, candidates: Iterable[str], products: Mapping[str, Product]
) -> Portfolio:
    """在候选池内求解权重。

    参数：
        client：客户档案（约束来源）。
        candidates：候选产品号可迭代对象（内部转为元组）。
        products：全市场产品表。
    返回：
        `Portfolio` 组合对象（含权重、现金权重、指标与 `rationale`）。
    副作用：无；候选集不可行时由 `optimizer.build_portfolio` 自行决定返回值/异常。
    """
    return build_portfolio(client, tuple(candidates), products)


def _tool_evaluate_rules(
    *,
    client: ClientProfile,
    portfolio: Portfolio,
    universe: Mapping[str, Product],
    candidates: Iterable[str],
) -> RuleEvaluation:
    """求值全部适当性规则。

    参数：
        client：客户档案（适当性判定的依据）。
        portfolio：待复核组合。
        universe：全市场产品表（用于查产品要素）。
        candidates：候选产品号可迭代对象。
    返回：
        `RuleEvaluation`（含 `blocks` / `warns` 等命中结果）。
    副作用：无（先 `RuleContext.build` 组装上下文，再交给 `evaluate_rules`）。
    """
    context = RuleContext.build(client, portfolio, universe, tuple(candidates))
    return evaluate_rules(context)


def _tool_issue_directive(
    *,
    client: ClientProfile,
    portfolio: Portfolio,
    universe: Mapping[str, Product],
    candidates: Iterable[str],
    round_index: int = 0,
    exempt_rules: Iterable[str] = (),
    max_repair_rounds: int = 2,
) -> GateDecision:
    """出具适当性闸门结论（含打回指令）。

    参数：
        client / portfolio / universe / candidates：同 `_tool_evaluate_rules`。
        round_index：当前轮次，默认 0。
        exempt_rules：人工豁免的 block 规则号，默认空。
        max_repair_rounds：允许打回重配的最大轮次，默认 2；超过后闸门不再要求重配。
    返回：
        `GateDecision`，`directive` 为 `pass` / `reoptimize` / `reject` 之一，
        并带上 `blocks` / `warns` 与约束收紧指令 `tighten`。
    副作用：无（`SuitabilityGate` 是无状态判定器，每次新建）。
    说明：本工具是流程中唯一带"否决"语义的工具，对应 `PERM_SUIT_VETO`。
    """
    context = RuleContext.build(client, portfolio, universe, tuple(candidates))
    gate = SuitabilityGate(max_repair_rounds=max_repair_rounds)
    return gate.review(context, round_index=round_index, exempt_rules=exempt_rules)


def _tool_build_tighten(
    *,
    client: ClientProfile,
    portfolio: Portfolio,
    universe: Mapping[str, Product],
    candidates: Iterable[str],
    rule_ids: Iterable[str],
) -> dict[str, Any]:
    """由指定规则命中构造约束收紧指令。

    参数：
        client / portfolio / universe / candidates：同 `_tool_evaluate_rules`。
        rule_ids：需要转成收紧动作的规则号集合；只会取 `evaluation.blocks` 中
            规则号命中的那部分（warn 级命中不参与）。
    返回：
        `TightenSpec.to_dict()` 的普通字典（可直接写入 state / trace）；
        没有任何匹配时返回空指令的字典形式，不抛异常。
    副作用：无。
    说明：收紧语义是**单调收紧**（等级取更小、上限取更低、流动性下限取更高、
        禁止项取并集），因此多轮打回必然收敛，不会来回震荡。
    """
    context = RuleContext.build(client, portfolio, universe, tuple(candidates))
    evaluation = evaluate_rules(context)
    wanted = set(rule_ids)
    blocks = [v for v in evaluation.blocks if v.rule_id in wanted]
    return build_tighten(blocks, context).to_dict()


def _tool_run_stress(
    *, portfolio: Portfolio, client: ClientProfile, config: Mapping[str, Any]
) -> StressReport:
    """执行情景压力测试。

    参数：
        portfolio：待测算组合。
        client：客户档案（提供回撤容忍度，用于判断是否超出）。
        config：压力情景配置（情景定义与敏感性系数表）。
    返回：
        `StressReport`（含 `scenarios`、`formula`、`worst_scenario_id`、`worst_impact`）。
    副作用：无；测算为确定性推算，不含随机与模型调用。
    """
    return run_stress(portfolio, client, config)


def _tool_build_counterfactual(
    *,
    client: ClientProfile,
    portfolio: Portfolio,
    products: Mapping[str, Product],
    rule_ids: Iterable[str],
    version_label: str,
    status: str,
    rule_client: ClientProfile | None = None,
):
    """生成反事实解释报告（基线为生效约束，规则判定针对真实客户档案）。

    参数：
        client：**基线客户档案**（已按生效约束收紧后的那份，构成反事实的对照基线）。
        portfolio：待解释的组合。
        products：全市场产品表。
        rule_ids：基线方案实际依据的规则号（调用方传 `gate.blocks` 与 `gate.warns`
            的规则号拼接结果）。
        version_label：基线版本标签，形如 `v1`。
        status：基线状态，取闸门 `directive`（无闸门时为 `"unknown"`）。
        rule_client：真实客户档案；None 时退化为 `client`。用于保证"适当性判定始终
            针对客户真实档案"，不随约束收紧而改变。
    返回：
        `CounterfactualReport`（含 `variants` 与 `coverage`）；无返回类型注解，
        实际返回类型由 `counterfactual.analyze_counterfactual` 决定。
    副作用：无（每个变体都是"改约束 → 重新求解 → 结构化比对"的确定性重算）。
    """
    return analyze_counterfactual(
        client,
        products,
        portfolio,
        baseline_rule_ids=tuple(rule_ids),
        baseline_version=version_label,
        baseline_status=status,
        rule_client=rule_client,
    )


def _tool_compose_text(*, llm: Any, task: str, context: Mapping[str, Any]) -> dict[str, Any]:
    """调用 LLM（或 mock 大脑）生成结构化文案。

    参数：
        llm：LLM 客户端（`src/llm.py` 的 `BaseLLM` 实现）；由 Agent 从自身 `llm`
            属性透传进来，**是全部工具中唯一可能访问外部的依赖**。
        task：任务名（`src/mock_brain.py` 的 `TASK_SCHEMAS` key）。
        context：结构化事实字典。

    返回：
        `{"task": 任务名, "mode": LLMResult.mode 取值, "data": 文案字典}`；
        注意返回值**丢弃**了 `LLMResult.raw` 与 `LLMResult.error`。

    副作用：可能发起一次 LLM 网络调用（超时由 `LLMConfig.timeout` 控制）；
        真实调用失败或字段缺失时由 `FallbackLLM` 降级为确定性 mock，
        因此正常情况下不会把异常抛给 Agent。

    说明：同一个处理器被 5 个工具名复用（`profile.compose_profile_text` /
        `screening.compose_screening_text` / `optimize.compose_rationale` /
        `suitability.compose_comment` / `narrative.compose_text`），区别只在各自传入的
        `task` 与调起它的 Agent 白名单；`llm` 为 None 时本函数不做兜底，
        会因 `None.compose` 抛 `AttributeError`。
    """
    result = llm.compose(task, context)
    return {"task": result.task, "mode": result.mode, "data": result.data}


#: 全量工具表
#: 共 17 个工具，按 Agent 视角分组排列：profile / screening / optimize / suitability / narrative。
#: 每个 Tool 的第一个字段是工具名（`AgentSpec.tools` 白名单引用它），第二个是权限标签。
#: 注意 `screening.compose_screening_text` / `optimize.compose_rationale` 的权限标签是
#: `narrative:write`（文案产出统一归口），而非 screening / optimize 前缀。
ALL_TOOLS: tuple[Tool, ...] = (
    Tool("profile.read_client", PERM_PROFILE_READ, "读取客户档案", _tool_read_client),
    Tool("profile.parse_questionnaire", PERM_PROFILE_READ, "解析风险测评问卷", _tool_parse_questionnaire),
    Tool("profile.derive_constraints", PERM_PROFILE_DERIVE, "抽取硬约束与软偏好", _tool_derive_constraints),
    Tool("profile.compose_profile_text", PERM_PROFILE_DERIVE, "生成客户画像文案（LLM/mock）", _tool_compose_text),
    Tool("screening.filter_universe", PERM_SCREEN_EXECUTE, "在硬约束内筛选候选池", _tool_filter_universe),
    Tool("screening.explain_exclusion", PERM_SCREEN_READ, "解释产品剔除原因", _tool_explain_exclusion),
    Tool("screening.compose_screening_text", PERM_NARRATIVE_WRITE, "生成筛选说明文案", _tool_compose_text),
    Tool("optimize.score_products", PERM_OPTIMIZE_COMPUTE, "多目标打分", _tool_score_products),
    Tool("optimize.solve_weights", PERM_OPTIMIZE_COMPUTE, "求解组合权重", _tool_solve_weights),
    Tool("optimize.compose_rationale", PERM_NARRATIVE_WRITE, "生成权衡说明文案", _tool_compose_text),
    Tool("suitability.evaluate_rules", PERM_SUIT_JUDGE, "求值适当性规则", _tool_evaluate_rules),
    Tool("suitability.issue_directive", PERM_SUIT_VETO, "出具闸门结论与打回指令", _tool_issue_directive),
    Tool("suitability.build_tighten", PERM_SUIT_JUDGE, "构造约束收紧指令", _tool_build_tighten),
    Tool("suitability.compose_comment", PERM_SUIT_JUDGE, "生成复核意见文案", _tool_compose_text),
    Tool("narrative.run_stress", PERM_NARRATIVE_ANALYZE, "执行情景压力测试", _tool_run_stress),
    Tool("narrative.build_counterfactual", PERM_NARRATIVE_ANALYZE, "生成反事实解释", _tool_build_counterfactual),
    Tool("narrative.compose_text", PERM_NARRATIVE_WRITE, "生成文案（LLM/mock）", _tool_compose_text),
)

#: 默认工具注册表
#: 进程内单例，被 `BaseAgent.__init__` 作为 `registry` 的默认值使用。
TOOL_REGISTRY = ToolRegistry(ALL_TOOLS)


@dataclass(frozen=True)
class AgentSpec:
    """Agent 的能力契约。

    不可变（`frozen=True`）；每个 Agent 模块在模块级声明一份 `SPEC`，
    由 `BaseAgent.__init__` 接收并落到 `self.spec`。

    关键属性：
        name：Agent 名称（与 `AGENT_ORDER` 中的字符串一致，也是流水线 state 的标识）。
        role：中文角色说明（展示用）。
        system_prompt：该 Agent 的角色提示词与输出边界。
        tools：工具白名单，元素必须是 `ALL_TOOLS` 中已登记的 `Tool.name`；
            `BaseAgent.call_tool` 的第一道校验。
        permissions：权限标签集合，必须包含所调用工具 `Tool.permission` 的标签；
            `BaseAgent.call_tool` 的第二道校验（取值来自 `PERM_*` 常量）。
        can_write_state：允许写入的共享 state 键；`BaseAgent.guard_output` 据此
            拦截越权写入，默认空元组（表示不允许写任何键）。

    被谁使用：五个 Agent 模块（各自声明 `SPEC`）、`BaseAgent`、`demo --catalog`。
    """

    name: str
    role: str
    system_prompt: str
    tools: tuple[str, ...]
    permissions: frozenset[str]
    can_write_state: tuple[str, ...] = field(default=())

    def describe(self) -> dict[str, Any]:
        """能力清单（demo / README 使用）。

        返回：
            `{name, role, tools, permissions, can_write_state}`；
            `tools` / `can_write_state` 转成列表，`permissions` 排序后转列表
            （集合本身无序，排序保证输出稳定、可写进快照与文档）。
        副作用：无。
        """
        return {
            "name": self.name,
            "role": self.role,
            "tools": list(self.tools),
            "permissions": sorted(self.permissions),
            "can_write_state": list(self.can_write_state),
        }


def tool_names() -> list[str]:
    """全部工具名。

    返回：默认注册表中按字典序排序的工具名列表（当前共 17 个）。
    副作用：无。
    """
    return TOOL_REGISTRY.names()


def registry_tools() -> tuple[Tool, ...]:
    """全部工具对象。

    返回：模块级 `ALL_TOOLS` 元组本身（**不是副本**，调用方不应修改；
        由于 `Tool` 为 frozen dataclass 且元组不可变，实际也无法就地改动）。
    副作用：无。
    """
    return ALL_TOOLS


def tighten_from_payload(payload: Mapping[str, Any] | None) -> TightenSpec:
    """便捷函数：从 state 中的收紧指令字典还原对象。

    参数：payload：`TightenSpec.to_dict()` 产出的字典，或 None / 空字典。
    返回：`TightenSpec`；payload 为假值（None、空字典）时返回**空指令**而非报错。
    副作用：无。
    """
    return TightenSpec.from_dict(payload)


#: 导出清单：权限常量 11 个 + 工具/契约类型 + 便捷函数 + 转出的 binding_constraints。
#: 注意 `binding_constraints` 来自 `..optimizer`，此处转出以便调用方少依赖一个模块。
__all__ = [
    "AgentSpec",
    "ALL_TOOLS",
    "PERM_NARRATIVE_ANALYZE",
    "PERM_NARRATIVE_WRITE",
    "PERM_OPTIMIZE_COMPUTE",
    "PERM_OPTIMIZE_READ",
    "PERM_PROFILE_DERIVE",
    "PERM_PROFILE_READ",
    "PERM_SCREEN_EXECUTE",
    "PERM_SCREEN_READ",
    "PERM_SUIT_JUDGE",
    "PERM_SUIT_VETO",
    "PERM_TRACE_WRITE",
    "TOOL_REGISTRY",
    "Tool",
    "ToolRegistry",
    "binding_constraints",
    "registry_tools",
    "tighten_from_payload",
    "tool_names",
]
