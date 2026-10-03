"""建议书要素完整性测试。"""

from __future__ import annotations

import pytest

from src.constraints import screen_products
from src.narrative import (
    NARRATIVE_ELEMENTS,
    build_narrative,
    dual_record_required,
    narrative_completeness,
)
from src.optimizer import build_portfolio
from src.schemas import CounterfactualReport, GateDecision, HumanReview, StressReport
from src.stress import run_stress
from tests.helpers import make_client, make_product


def _compose(task, context):
    """测试用确定性文案注入。"""
    from src.mock_brain import mock_compose

    return mock_compose(task, context)


@pytest.fixture
def narrative_case(data):
    client = data.client("C004")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    stress = run_stress(portfolio, client, data.stress_config)
    gate = GateDecision(
        round_index=0,
        passed=False,
        directive="reoptimize",
        blocks=[],
        warns=[],
        comment="测试用闸门结论。",
    )
    human = HumanReview(
        required=True,
        reasons=["客户为高龄客户"],
        decision="auto_approved",
        operator="auto(--auto)",
        note="演示模式自动放行",
    )
    return client, portfolio, screening, gate, stress, human


def test_narrative_contains_every_required_element(narrative_case):
    client, portfolio, screening, gate, stress, human = narrative_case
    text, elements = build_narrative(
        client=client,
        portfolio=portfolio,
        screening=screening,
        gate=gate,
        human_review=human,
        stress=stress,
        counterfactual=CounterfactualReport(),
        version=2,
        engine="native",
        run_id="test-run",
        compose=_compose,
    )
    assert set(elements) == set(NARRATIVE_ELEMENTS)
    assert all(elements.values())
    assert narrative_completeness(elements) == 1.0


def test_narrative_mentions_dual_record_marker(narrative_case):
    client, portfolio, screening, gate, stress, human = narrative_case
    text, _ = build_narrative(
        client=client,
        portfolio=portfolio,
        screening=screening,
        gate=gate,
        human_review=human,
        stress=stress,
        counterfactual=CounterfactualReport(),
        version=1,
        engine="native",
        run_id="r",
        compose=_compose,
    )
    assert "双录留痕" in text
    assert "补录" in text  # C004 未完成双录


def test_dual_record_detection_rules(data):
    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    required, reasons = dual_record_required(client, portfolio)
    assert required is False or reasons

    elderly = make_client(age=70)
    product = make_product(risk_level=4)
    from src.schemas import Portfolio

    risky = Portfolio(
        weights={product.product_id: 0.5},
        products={product.product_id: product},
        cash_weight=0.5,
    )
    required_elderly, reasons_elderly = dual_record_required(elderly, risky)
    assert required_elderly is True
    assert any("高龄" in item for item in reasons_elderly)
    assert any("R4" in item for item in reasons_elderly)


def test_narrative_completeness_handles_empty():
    assert narrative_completeness({}) == 0.0
    assert narrative_completeness({key: False for key in NARRATIVE_ELEMENTS}) == 0.0


def test_narrative_contains_no_real_institution_names(narrative_case):
    from tests.test_data_hygiene import REAL_NAME_BLACKLIST

    client, portfolio, screening, gate, stress, human = narrative_case
    text, _ = build_narrative(
        client=client,
        portfolio=portfolio,
        screening=screening,
        gate=gate,
        human_review=human,
        stress=stress,
        counterfactual=CounterfactualReport(),
        version=1,
        engine="native",
        run_id="r",
        compose=_compose,
    )
    for name in REAL_NAME_BLACKLIST:
        assert name not in text, f"建议书出现疑似真实机构名：{name}"


def test_empty_portfolio_narrative_still_complete(data):
    client = data.client("C006")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    stress = run_stress(portfolio, client, data.stress_config)
    gate = GateDecision(directive="reject", passed=False, comment="可行域为空，拒绝出具建议。")
    text, elements = build_narrative(
        client=client,
        portfolio=portfolio,
        screening=screening,
        gate=gate,
        human_review=HumanReview(),
        stress=stress,
        counterfactual=CounterfactualReport(),
        version=1,
        engine="native",
        run_id="r",
        compose=_compose,
    )
    assert all(elements.values())
    assert "不出具配置建议" in text
