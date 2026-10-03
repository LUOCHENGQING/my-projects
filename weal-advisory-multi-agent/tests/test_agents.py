"""五个 Agent 的能力契约测试：独立 prompt / 白名单 / 权限 / 越权拦截 / 职责输出。"""

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
    return build_agents(mock_llm)


def test_five_agents_are_built(agents):
    assert set(agents) == set(AGENT_ORDER)
    assert len(agents) == 5


def test_each_agent_has_its_own_system_prompt(agents):
    prompts = {name: agent.spec.system_prompt for name, agent in agents.items()}
    assert len(set(prompts.values())) == 5
    for prompt in prompts.values():
        assert len(prompt) > 80


def test_tool_whitelists_are_distinct(agents):
    whitelists = {name: tuple(agent.spec.tools) for name, agent in agents.items()}
    assert len(set(whitelists.values())) == 5
    for name, tools in whitelists.items():
        assert tools, f"{name} 没有工具白名单"
        assert set(tools) <= set(tool_names())


def test_permission_sets_are_independent(agents):
    permissions = {name: agent.spec.permissions for name, agent in agents.items()}
    assert len(set(frozenset(p) for p in permissions.values())) == 5
    # 否决权只有适当性复核岗拥有
    veto_holders = [name for name, p in permissions.items() if "suitability:veto" in p]
    assert veto_holders == ["SuitabilityOfficerAgent"]


def test_can_write_state_keys_do_not_overlap_much(agents):
    """最小权限：任何 Agent 的可写状态键都不是全集，且关键键互不越权。"""
    profiling = agents["ClientProfilingAgent"].spec.can_write_state
    officer = agents["SuitabilityOfficerAgent"].spec.can_write_state
    assert "portfolio" not in profiling
    assert "portfolio" not in officer
    assert "gate" not in profiling
    assert "gate" in officer


def test_call_tool_outside_whitelist_raises(agents):
    profiling = agents["ClientProfilingAgent"]
    with pytest.raises(PermissionError):
        profiling.call_tool("optimize.solve_weights", client=make_client(), candidates=(), products={})


def test_call_tool_without_permission_raises(mock_llm):
    """工具在白名单内但缺少权限标签时同样必须被拦下。"""
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
    profiling = agents["ClientProfilingAgent"]
    assert profiling.guard_output({"profile": {}}) == {"profile": {}}
    with pytest.raises(PermissionError):
        profiling.guard_output({"portfolio": {}})


def test_unknown_tool_is_rejected_before_registry_lookup(agents):
    profiling = agents["ClientProfilingAgent"]
    with pytest.raises(PermissionError):
        profiling.call_tool("not.a.tool")


def test_client_profiling_agent_extracts_hard_constraints(data, agents):
    client = data.client("C001")
    result = agents["ClientProfilingAgent"].run(client=client, questionnaire=data.questionnaire)
    hard = result["profile"]["hard_constraints"]
    assert hard["risk_capacity"] == 2
    assert hard["investment_horizon_years"] == 5.0
    assert hard["max_single_product_ratio"] == pytest.approx(0.3)
    assert result["profile"]["soft_preferences"]["return_target"] == pytest.approx(0.045)
    assert result["profile"]["summary"]
    assert result["effective_client"].client_id == "C001"


def test_client_profiling_applies_strict_lower_risk_level(data, agents):
    """从严原则：问卷折算等级低于档案等级时取孰低。"""
    client = data.client("C002")
    assert client.risk_capacity == 4
    result = agents["ClientProfilingAgent"].run(client=client, questionnaire=data.questionnaire)
    assert result["profile"]["questionnaire"]["level"] == 3
    assert result["effective_client"].risk_capacity == 3
    assert result["profile"]["tightened_by_questionnaire"] is True
    assert "从严" in result["profile"]["consistency_note"]


def test_client_profiling_does_not_mutate_client(data, agents):
    client = data.client("C001")
    before = client.model_dump()
    agents["ClientProfilingAgent"].run(client=client, questionnaire=data.questionnaire)
    assert client.model_dump() == before


def test_product_screening_agent_reports_exclusions(data, agents):
    client = data.client("C002")
    result = agents["ProductScreeningAgent"].run(client=client, products=data.products)
    screening = result["screening"]
    assert screening.included
    assert screening.excluded
    reasons = {code for item in screening.excluded for code in item.reasons}
    assert "C-EXPERIENCE" in reasons
    assert result["screening_note"]


def test_portfolio_optimizer_agent_produces_feasible_portfolio(data, agents):
    from src.constraints import check_portfolio

    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    result = agents["PortfolioOptimizerAgent"].run(
        client=client, candidates=screening.included, products=data.products
    )
    portfolio = result["portfolio"]
    assert check_portfolio(portfolio, client) == []
    assert result["portfolio_note"]
    assert result["binding"] is not None


def test_suitability_officer_blocks_and_issues_tighten(data, agents):
    client = data.client("C004")
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
    from src.narrative import NARRATIVE_ELEMENTS

    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
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
    client = data.client("C003")
    screening_before = data.products["P-BOND-02"].model_dump()
    agents["ProductScreeningAgent"].run(client=client, products=data.products)
    assert data.products["P-BOND-02"].model_dump() == screening_before


def test_agent_catalog_describes_all_agents():
    from src.agents import agent_catalog

    catalog = agent_catalog()
    assert len(catalog) == 5
    for item in catalog:
        assert item["name"] and item["role"]
        assert item["tools"] and item["permissions"] and item["can_write_state"]


def test_agents_share_registry_but_not_capabilities(agents):
    registries = {id(agent.registry) for agent in agents.values()}
    assert len(registries) == 1
    assert len({agent.spec.name for agent in agents.values()}) == 5


def test_profiling_agent_classes_are_exported():
    for cls in (
        ClientProfilingAgent,
        ProductScreeningAgent,
        PortfolioOptimizerAgent,
        SuitabilityOfficerAgent,
        AdvisorNarrativeAgent,
    ):
        assert issubclass(cls, BaseAgent)
