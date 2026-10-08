"""建议书（Advisor Narrative）要素完整性与合规措辞测试。

被测模块：`src.narrative`
- `NARRATIVE_ELEMENTS`：建议书 12 项必备要素的定义（章节标题 → 要素键）；
- `build_narrative`：把客户档案 / 组合 / 筛选结果 / 闸门结论 / 人工确认 / 压力测试 /
  反事实报告 / 版本号拼装成建议书正文与要素命中表；
- `dual_record_required`：双录留痕触发判定（高龄客户、R4 及以上产品、衍生品结构）；
- `narrative_completeness`：要素完整率口径（评估脚本按此指标打分）。

覆盖策略
- 正常路径：C004 高龄客户走完"筛选 → 组合 → 压力 → 闸门 → 人工"全链路后，
  12 项要素必须全部命中且完整率为 1.0；
- 边界路径：可行域为空（C006）被闸门 reject 时，仍须出具结构完整、明确写
  "不出具配置建议"的建议书，不能塌缩成残缺文档；
- 边界口径：空要素表与"全 False 要素表"的完整率都必须为 0.0（不得按缺失项数计分或除零）；
- 对抗探针：建议书正文不得出现任何真实金融机构名称（复用 data hygiene 黑名单）。

注：本模块全部为模块级测试函数，未定义测试类，故无类级 docstring。
"""

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
    """构造一条完整的建议书输入链（C004 高龄客户）。

    刻意覆盖"需要人工确认 + 首轮被闸门打回"的组合：要素渲染不应依赖闸门是否通过，
    因此这里固定传入 passed=False 的结论，用来证明建议书依旧能完整渲染。
    """
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
    """不变式：输入链路完整时，12 项必备要素必须全部非空且完整率为 1.0。"""
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
    """不变式：warn 级的双录留痕要求必须落到正文——C004 未完成双录，须出现"补录"字样。"""
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
    """不变式：双录结论自洽——required 为假时必须无原因，为真时必须给出原因。

    单点变量法：先验干净客户 C001，再只改「年龄 → 70」与「风险等级 → 4」两个字段，
    隔离出"高龄 + R4"这两个独立触发条件，避免其它约束干扰判定。
    """
    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    required, reasons = dual_record_required(client, portfolio)
    assert required is False or reasons

    elderly = make_client(age=70)
    product = make_product(risk_level=4)
    from src.schemas import Portfolio

    # 手工装配组合（不经优化器），确保"高龄 + R4"是唯一变量
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
    """口径：空要素表与全 False 要素表的完整率均为 0.0（评估指标不允许出现除零或虚高）。"""
    assert narrative_completeness({}) == 0.0
    assert narrative_completeness({key: False for key in NARRATIVE_ELEMENTS}) == 0.0


def test_narrative_contains_no_real_institution_names(narrative_case):
    """对抗探针：建议书正文不得出现任何真实机构名，逐条比对 data hygiene 黑名单。"""
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
    """边界：可行域为空被 reject 时，建议书仍须 12 项要素齐全并明确写明"不出具配置建议"。"""
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
