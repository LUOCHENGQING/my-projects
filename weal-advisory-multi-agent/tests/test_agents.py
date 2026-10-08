"""五个 Agent 的能力契约测试。

覆盖对象
--------
`src.agents`：`BaseAgent` 基类、五个具体 Agent（画像 / 筛选 / 求解 / 适当性复核 /
建议书），以及 `build_agents` / `agent_catalog` / `AGENT_ORDER` / `tool_names`
和它们背后的工具与权限体系。

覆盖策略（按关注点分层）
------------------------
- **结构契约**：五个 Agent 必须被构建出来，且各自拥有**互不相同**的 system prompt、
  工具白名单与权限集合（防止"能力同质化"，也防止某个岗位悄悄拿到别人的工具）。
- **最小权限（对抗路径）**：白名单外的工具、白名单内但缺少权限标签的工具、
  未注册的工具、越权写共享状态键，四种越权都必须抛 `PermissionError`。
- **职责输出（正常路径）**：画像提取硬约束与软偏好、筛选给出可追溯的剔除原因、
  求解器产出 0 违反的组合、闸门能拦下并下发收紧指令、建议书要素全部齐全。
- **不变式**：Agent 只读输入，运行前后不得修改传入的 `ClientProfile` / `Product`。

夹具：`mock_llm`（session，确定性大脑）、`agents`（本文件的 function 夹具）、
`data`（session，全量样例数据）。
"""

from __future__ import annotations

import pytest

from src.agents import (
    AGENT_ORDER,
    AdvisorNarrativeAgent,
    AgentSpec,
    BaseAgent,
    ClientProfilingAgent,
    PortfolioOptimizerAgent,
    ProductScreeningAgent,
    SuitabilityOfficerAgent,
    build_agents,
    tool_names,
)
from src.constraints import screen_products
from src.optimizer import build_portfolio
from tests.helpers import make_client


@pytest.fixture
def agents(mock_llm):
    """构建五个 Agent 的映射 `name -> BaseAgent`（function 作用域）。

    逐用例重建：部分用例会把 Agent 当作有状态对象使用（如覆写 tracer），
    重建可避免用例之间相互串扰。
    """
    return build_agents(mock_llm)


def test_five_agents_are_built(agents):
    """不变式：`build_agents` 恰好产出 `AGENT_ORDER` 声明的五个 Agent，不多不少。"""
    assert set(agents) == set(AGENT_ORDER)
    assert len(agents) == 5


def test_each_agent_has_its_own_system_prompt(agents):
    """契约：五个 Agent 的 system prompt 两两不同，且都不是敷衍的短句（长度 > 80）。"""
    prompts = {name: agent.spec.system_prompt for name, agent in agents.items()}
    assert len(set(prompts.values())) == 5
    for prompt in prompts.values():
        assert len(prompt) > 80


def test_tool_whitelists_are_distinct(agents):
    """契约：工具白名单两两不同、非空，且只能引用注册表中真实存在的工具名。"""
    whitelists = {name: tuple(agent.spec.tools) for name, agent in agents.items()}
    assert len(set(whitelists.values())) == 5
    for name, tools in whitelists.items():
        assert tools, f"{name} 没有工具白名单"
        assert set(tools) <= set(tool_names())


def test_permission_sets_are_independent(agents):
    """契约：权限集合两两不同，且"否决权"只归属适当性复核岗。"""
    permissions = {name: agent.spec.permissions for name, agent in agents.items()}
    assert len(set(frozenset(p) for p in permissions.values())) == 5
    # 否决权只有适当性复核岗拥有
    veto_holders = [name for name, p in permissions.items() if "suitability:veto" in p]
    assert veto_holders == ["SuitabilityOfficerAgent"]


def test_can_write_state_keys_do_not_overlap_much(agents):
    """最小权限：任何 Agent 的可写状态键都不是全集，且关键键互不越权。"""
    profiling = agents["ClientProfilingAgent"].spec.can_write_state
    officer = agents["SuitabilityOfficerAgent"].spec.can_write_state
    # 画像岗写不了组合、也写不了闸门结论；闸门结论只能由适当性复核岗落笔
    assert "portfolio" not in profiling
    assert "portfolio" not in officer
    assert "gate" not in profiling
    assert "gate" in officer


def test_call_tool_outside_whitelist_raises(agents):
    """对抗路径：调用不在白名单内的工具必须立刻被拒（即使该工具真实存在）。"""
    profiling = agents["ClientProfilingAgent"]
    with pytest.raises(PermissionError):
        profiling.call_tool("optimize.solve_weights", client=make_client(), candidates=(), products={})


def test_call_tool_without_permission_raises(mock_llm):
    """工具在白名单内但缺少权限标签时同样必须被拦下。"""
    # 刻意只给 trace:write：白名单放行 optimize.solve_weights，但权限标签缺失
    spec = AgentSpec(
        name="BadAgent",
        role="越权测试",
        system_prompt="测试用 Agent",
        tools=("optimize.solve_weights",),
        permissions=frozenset({"trace:write"}),
        can_write_state=("portfolio",),
    )
    agent = BaseAgent(spec, llm=mock_llm)
    with pytest.raises(PermissionError):
        agent.call_tool("optimize.solve_weights", client=make_client(), candidates=(), products={})


def test_guard_output_rejects_illegal_state_key(agents):
    """契约：`guard_output` 放行自己负责的状态键，拦截他人负责的状态键。"""
    profiling = agents["ClientProfilingAgent"]
    assert profiling.guard_output({"profile": {}}) == {"profile": {}}
    with pytest.raises(PermissionError):
        profiling.guard_output({"portfolio": {}})


def test_unknown_tool_is_rejected_before_registry_lookup(agents):
    """契约：白名单校验先于注册表查找，因此连"不存在的工具名"也报 PermissionError。"""
    profiling = agents["ClientProfilingAgent"]
    with pytest.raises(PermissionError):
        profiling.call_tool("not.a.tool")


def test_client_profiling_agent_extracts_hard_constraints(data, agents):
    """正常路径：画像 Agent 从档案 + 问卷中提取硬约束、软偏好、摘要与生效客户。"""
    client = data.client("C001")
    result = agents["ClientProfilingAgent"].run(client=client, questionnaire=data.questionnaire)
    hard = result["profile"]["hard_constraints"]
    # 断言值即 data/clients.json 中 C001 的登记值：R2、5 年、单一产品上限 30%
    assert hard["risk_capacity"] == 2
    assert hard["investment_horizon_years"] == 5.0
    assert hard["max_single_product_ratio"] == pytest.approx(0.3)
    assert result["profile"]["soft_preferences"]["return_target"] == pytest.approx(0.045)
    assert result["profile"]["summary"]
    assert result["effective_client"].client_id == "C001"


def test_client_profiling_applies_strict_lower_risk_level(data, agents):
    """从严原则：问卷折算等级低于档案等级时取孰低。"""
    client = data.client("C002")
    # C002 档案登记 R4，问卷得分 60 折算为 R3，因此生效等级必须被压到 R3
    assert client.risk_capacity == 4
    result = agents["ClientProfilingAgent"].run(client=client, questionnaire=data.questionnaire)
    assert result["profile"]["questionnaire"]["level"] == 3
    assert result["effective_client"].risk_capacity == 3
    assert result["profile"]["tightened_by_questionnaire"] is True
    assert "从严" in result["profile"]["consistency_note"]


def test_client_profiling_does_not_mutate_client(data, agents):
    """不变式：画像过程不修改传入的客户档案（纯读）。"""
    client = data.client("C001")
    before = client.model_dump()
    agents["ClientProfilingAgent"].run(client=client, questionnaire=data.questionnaire)
    assert client.model_dump() == before


def test_product_screening_agent_reports_exclusions(data, agents):
    """正常路径：筛选结果同时给出入选池与剔除清单，且剔除项带可追溯的原因码。"""
    client = data.client("C002")
    result = agents["ProductScreeningAgent"].run(client=client, products=data.products)
    screening = result["screening"]
    assert screening.included
    assert screening.excluded
    # C002 未具备"权益"投资经验，因此剔除原因里必然出现 C-EXPERIENCE
    reasons = {code for item in screening.excluded for code in item.reasons}
    assert "C-EXPERIENCE" in reasons
    assert result["screening_note"]


def test_portfolio_optimizer_agent_produces_feasible_portfolio(data, agents):
    """不变式：求解器接受的组合，硬约束违反数恒为 0（并给出绑定约束与说明）。"""
    from src.constraints import check_portfolio

    client = data.client("C001")
    # 与流水线保持一致的两步顺序：先按硬约束筛出可行域，再在可行域内求解权重
    screening = screen_products(client, data.products, 0)
    result = agents["PortfolioOptimizerAgent"].run(
        client=client, candidates=screening.included, products=data.products
    )
    portfolio = result["portfolio"]
    assert check_portfolio(portfolio, client) == []
    assert result["portfolio_note"]
    assert result["binding"] is not None


def test_suitability_officer_blocks_and_issues_tighten(data, agents):
    """职责：对高龄客户必须拦截（S-ELDERLY）并要求重配，同时下发风险等级收紧指令。"""
    client = data.client("C004")
    # C004 为 68 岁高龄且未完成双录，必然触发 S-ELDERLY 阻断
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    result = agents["SuitabilityOfficerAgent"].run(
        client=client,
        portfolio=portfolio,
        universe=data.products,
        candidates=screening.included,
        round_index=0,
    )
    gate = result["gate"]
    assert gate.directive == "reoptimize"
    assert "S-ELDERLY" in [v.rule_id for v in gate.blocks]
    assert result["suitability_comment"]
    assert result["tighten"]["risk_cap"] == 3


def test_suitability_officer_passes_compliant_portfolio(data, agents):
    """正常路径：合规组合必须被闸门放行（passed=True），不得误杀。"""
    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    result = agents["SuitabilityOfficerAgent"].run(
        client=client,
        portfolio=portfolio,
        universe=data.products,
        candidates=screening.included,
        round_index=0,
    )
    assert result["gate"].passed is True


def test_advisor_narrative_agent_emits_all_required_elements(data, agents):
    """职责：建议书 Agent 一次产出全部 12 项要素（且非空）+ 压力测试 + 反事实 + 留痕。"""
    from src.narrative import NARRATIVE_ELEMENTS

    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    # 先拿到闸门结论，再作为输入交给建议书 Agent（与流水线的数据流一致）
    gate = agents["SuitabilityOfficerAgent"].run(
        client=client,
        portfolio=portfolio,
        universe=data.products,
        candidates=screening.included,
    )["gate"]
    result = agents["AdvisorNarrativeAgent"].run(
        client=client,
        portfolio=portfolio,
        products=data.products,
        stress_config=data.stress_config,
        screening=screening,
        gate=gate,
        version=1,
        engine="native",
        run_id="test-run",
    )
    assert set(result["elements"]) == set(NARRATIVE_ELEMENTS)
    assert all(result["elements"].values())
    assert result["stress"].scenario_count >= 3
    assert result["counterfactual"].variants
    assert result["advice"].narrative
    assert result["advice"].run_id == "test-run"


def test_agents_never_mutate_input_state(data, agents):
    """不变式：Agent 运行不修改传入的产品对象（样例数据被多模块复用，不能被污染）。"""
    client = data.client("C003")
    screening_before = data.products["P-BOND-02"].model_dump()
    agents["ProductScreeningAgent"].run(client=client, products=data.products)
    assert data.products["P-BOND-02"].model_dump() == screening_before


def test_agent_catalog_describes_all_agents():
    """契约：`agent_catalog()` 为五个 Agent 各给出非空的能力清单（工具 / 权限 / 可写状态）。"""
    from src.agents import agent_catalog

    catalog = agent_catalog()
    assert len(catalog) == 5
    for item in catalog:
        assert item["name"] and item["role"]
        assert item["tools"] and item["permissions"] and item["can_write_state"]


def test_agents_share_registry_but_not_capabilities(agents):
    """架构约束：五个 Agent 共用同一工具注册表实例，但能力契约（spec）彼此独立。"""
    registries = {id(agent.registry) for agent in agents.values()}
    assert len(registries) == 1
    assert len({agent.spec.name for agent in agents.values()}) == 5


def test_profiling_agent_classes_are_exported():
    """导出契约：五个 Agent 类都能从 `src.agents` 导入，且都是 `BaseAgent` 子类。"""
    for cls in (
        ClientProfilingAgent,
        ProductScreeningAgent,
        PortfolioOptimizerAgent,
        SuitabilityOfficerAgent,
        AdvisorNarrativeAgent,
    ):
        assert issubclass(cls, BaseAgent)
