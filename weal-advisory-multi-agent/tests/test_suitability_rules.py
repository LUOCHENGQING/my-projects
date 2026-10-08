"""适当性规则库与硬闸门测试：22 条规则逐条可触发、block 必拦、warn 不拦。

被测模块：`src.suitability`（规则库 `rules.py` + 闸门 `engine.py`）
- `ALL_RULES` / `RULES_BY_ID` / `get_rule` / `rule_catalog` / `format_rule_table`：规则注册表与清单视图；
- `evaluate_rules`：按固定顺序确定性求值，输出 hits / blocks / warns；
- `SuitabilityGate.review`：闸门结论（pass / reoptimize / reject）、豁免降级、升级转人工；
- `build_tighten`：把 block 命中翻译成 `TightenSpec`（单调收紧指令）。

规则清单（编号与名称严格取自 `src/suitability/rules.py`，级别与闸门行为如下）
    block 级 —— 命中即**拦截**（打回重配；轮次用尽转人工）
        S-RISK-MATCH            风险等级匹配
        S-HORIZON               投资期限匹配
        S-CONC-SINGLE           单一产品集中度上限
        S-CONC-CLASS            单一资产类别上限
        S-CONCENTRATION-ISSUER  同一发行人集中度上限
        S-LIQUIDITY             流动性需求满足
        S-ENTRY                 起投金额匹配
        S-ELDERLY               高龄客户特别保护
        S-EXPERIENCE            投资经验匹配
        S-DERIVATIVE-BAN        衍生品禁止项
        S-CURRENCY              币种匹配
        S-QUALIFIED             合格投资者准入
        S-PROHIBITED            禁止项清单
        S-WEIGHT                权重合法性
        S-FEASIBLE-POOL         可行域非空（veto：**直接拒绝**，不存在修复路径）
    warn 级 —— 命中即**放行**，仅在建议书中揭示 / 转人工提示
        S-DUAL-RECORD           双录留痕
        S-FEE-DISCLOSE          费率揭示
        S-TAX                   税收优惠额度
        S-COOLING               冷静期提示
        S-CONCENTRATION-WARNING 内控集中度预警线
        S-CASH-RESERVE          现金缓冲建议
        S-DIVERSIFICATION       分散度建议

覆盖策略
- 规则库自身：编号唯一、字段完备、需求点名的规则号齐全、block 必有修复策略（veto 除外）、
  仅 S-FEASIBLE-POOL 为 veto、三个入口视图一致、未知规则号抛 KeyError；
- 逐条可触发：为 22 条规则各配一个"单点触发"工厂（TRIGGERS），参数化遍历断言命中；
- 闸门行为：block → 打回重配并下发收紧；轮次用尽 → 转人工；仅 warn → 放行；
  veto → 直接拒绝；人工豁免 → 降级为 warn 后放行；
- 收紧构造：S-ELDERLY / S-EXPERIENCE / S-CONC-SINGLE / S-FEASIBLE-POOL 四种修复语义的翻译结果；
- 不变式：规则求值确定、不改动组合对象、命中级别与规则定义一致。

注：本模块全部为模块级测试函数，未定义测试类，故无类级 docstring。
"""

from __future__ import annotations

from typing import Callable

import pytest

from src.constraints import TightenSpec
from src.schemas import ClientProfile, Product
from src.suitability import (
    ALL_RULES,
    RULES_BY_ID,
    RuleContext,
    SuitabilityGate,
    build_tighten,
    evaluate_rules,
    format_rule_table,
    get_rule,
)
from src.suitability.rules import has_block_rule, rule_catalog
from tests.helpers import build_portfolio_obj, make_client, make_product, singleton_products

#: 触发器协议：返回 (客户档案, 产品表, 权重, 候选产品号) 四元组，由 _context 组装成 RuleContext
TriggerFactory = Callable[[], tuple[ClientProfile, dict[str, Product], dict[str, float], tuple[str, ...]]]


def _trigger_risk_match():
    """触发 S-RISK-MATCH「风险等级匹配」（block → 拦截）：客户承受能力 R2 却持有 R4 产品。"""
    product = make_product(risk_level=4)
    return make_client(risk_capacity=2), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_horizon():
    """触发 S-HORIZON「投资期限匹配」（block → 拦截）：产品期限 5 年超过客户 1 年投资期限。"""
    product = make_product(horizon_years=5.0)
    return make_client(investment_horizon_years=1.0), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_conc_single():
    """触发 S-CONC-SINGLE「单一产品集中度上限」（block → 拦截）：单只权重 0.5 超过上限 0.3。

    同时把类别上限与发行主体上限抬到 0.95，做成"单点触发"，避免连带命中其它集中度规则。
    """
    product = make_product()
    return (
        make_client(max_single_product_ratio=0.3, max_single_class_ratio=0.95, max_single_issuer_ratio=0.95),
        singleton_products(product),
        {product.product_id: 0.5},
        (product.product_id,),
    )


def _trigger_conc_class():
    """触发 S-CONC-CLASS「单一资产类别上限」（block → 拦截）：两只固收产品各 0.30，类别合计 0.60 > 上限 0.50。

    刻意用不同发行主体、且单只权重不越限，保证命中的是"类别合计"而非单只或单主体。
    """
    first = make_product(product_id="A", issuer="甲")
    second = make_product(product_id="B", issuer="乙")
    return (
        make_client(max_single_product_ratio=0.35, max_single_class_ratio=0.5, max_single_issuer_ratio=0.95),
        singleton_products(first, second),
        {"A": 0.3, "B": 0.3},
        ("A", "B"),
    )


def _trigger_conc_issuer():
    """触发 S-CONCENTRATION-ISSUER「同一发行人集中度上限」（block → 拦截）：两只产品同属一个主体，合计 0.60 > 0.50。

    类别上限抬到 0.95 以隔离变量（两只同类别但主体相同，命中主体口径）。
    """
    first = make_product(product_id="A", issuer="同一主体")
    second = make_product(product_id="B", issuer="同一主体")
    return (
        make_client(max_single_product_ratio=0.35, max_single_class_ratio=0.95, max_single_issuer_ratio=0.5),
        singleton_products(first, second),
        {"A": 0.3, "B": 0.3},
        ("A", "B"),
    )


def _trigger_liquidity():
    """触发 S-LIQUIDITY「流动性需求满足」（block → 拦截）：产品可即时变现比例 0，组合流动性低于客户下限 0.5。

    集中度上限抬到 0.95 以隔离变量（此处权重 0.9 只用于压低流动性，不该被集中度规则抢先命中）。
    """
    product = make_product(liquidity_ratio=0.0)
    return (
        make_client(liquidity_floor_ratio=0.5, max_single_product_ratio=0.95, max_single_class_ratio=0.95),
        singleton_products(product),
        {product.product_id: 0.9},
        (product.product_id,),
    )


def _trigger_entry():
    """触发 S-ENTRY「起投金额匹配」（block → 拦截）：起投 500 万远高于客户可投 10 万。"""
    product = make_product(min_investment=5000000.0)
    return make_client(investable_amount=100000.0), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_elderly():
    """触发 S-ELDERLY「高龄客户特别保护」（block → 拦截）：客户 70 岁（≥65 即高龄）持有 R4 产品。"""
    product = make_product(risk_level=4)
    return make_client(age=70), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_experience():
    """触发 S-EXPERIENCE「投资经验匹配」（block → 拦截）：产品要求"权益"经验，客户仅有"货币"经验。"""
    product = make_product(requires_experience=["权益"])
    return make_client(experienced_categories=["货币"]), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_derivative_ban():
    """触发 S-DERIVATIVE-BAN「衍生品禁止项」（block → 拦截）：客户禁止"衍生品"类别，而产品含衍生品结构。"""
    product = make_product(asset_class="混合", is_derivative=True)
    return (
        make_client(prohibited_categories=["衍生品"]),
        singleton_products(product),
        {product.product_id: 0.3},
        (product.product_id,),
    )


def _trigger_currency():
    """触发 S-CURRENCY「币种匹配」（block → 拦截）：产品为 USD，客户偏好 CNY。"""
    product = make_product(currency="USD")
    return make_client(), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_qualified():
    """触发 S-QUALIFIED「合格投资者准入」（block → 拦截）：产品仅面向合格投资者，客户不具备该资格。"""
    product = make_product(qualified_investor_only=True)
    return make_client(qualified_investor=False), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_prohibited():
    """触发 S-PROHIBITED「禁止项清单」（block → 拦截）：客户禁止"固收"类别，产品恰属该类别（命中 C-PROHIBITED-CLASS）。"""
    product = make_product(asset_class="固收")
    return make_client(prohibited_categories=["固收"]), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_weight():
    """触发 S-WEIGHT「权重合法性」（block → 拦截）：两只产品各 0.70，合计 1.40 超过 100%（命中 C-WEIGHT-SUM）。"""
    first = make_product(product_id="A", issuer="甲")
    second = make_product(product_id="B", issuer="乙")
    return (
        make_client(max_single_product_ratio=0.95, max_single_class_ratio=0.95, max_single_issuer_ratio=0.95),
        singleton_products(first, second),
        {"A": 0.7, "B": 0.7},
        ("A", "B"),
    )


def _trigger_feasible_pool():
    """触发 S-FEASIBLE-POOL「可行域非空」（block 且 veto=True → 直接拒绝）：候选池为空。

    veto 语义：不存在"收紧后再试"的修复路径，因此闸门必须直接 reject，而不是打回重配。
    """
    product = make_product()
    return make_client(), singleton_products(product), {}, ()


def _trigger_dual_record():
    """触发 S-DUAL-RECORD「双录留痕」（warn → 放行，仅揭示）：客户未完成双录且持有 R4 产品。"""
    product = make_product(risk_level=4)
    return (
        make_client(dual_record_completed=False),
        singleton_products(product),
        {product.product_id: 0.3},
        (product.product_id,),
    )


def _trigger_fee():
    """触发 S-FEE-DISCLOSE「费率揭示」（warn → 放行，仅揭示）：产品费率 2% 远超客户费率预算 0.1%。"""
    product = make_product(fee_rate=0.02)
    return (
        make_client(annual_fee_budget_ratio=0.001),
        singleton_products(product),
        {product.product_id: 0.3},
        (product.product_id,),
    )


def _trigger_tax():
    """触发 S-TAX「税收优惠额度」（warn → 放行，仅揭示）：税优产品配置额 0.3×100 万 = 30 万，远超额度 1000 元。"""
    product = make_product(tax_advantaged=True)
    return (
        make_client(tax_advantaged_quota=1000.0),
        singleton_products(product),
        {product.product_id: 0.3},
        (product.product_id,),
    )


def _trigger_cooling():
    """触发 S-COOLING「冷静期提示」（warn → 放行，仅揭示）：期限 3 年 ≥ 1 年且即时变现比例 0.1 < 0.3。"""
    product = make_product(horizon_years=3.0, liquidity_ratio=0.1)
    return make_client(), singleton_products(product), {product.product_id: 0.3}, (product.product_id,)


def _trigger_concentration_warning():
    """触发 S-CONCENTRATION-WARNING「内控集中度预警线」（warn → 放行，但须转人工确认）：权重 0.3 越过预警线 0.2。"""
    product = make_product()
    return (
        make_client(internal_single_product_warning_ratio=0.2),
        singleton_products(product),
        {product.product_id: 0.3},
        (product.product_id,),
    )


def _trigger_cash_reserve():
    """触发 S-CASH-RESERVE「现金缓冲建议」（warn → 放行，仅揭示）：权重吃满 100%，现金为 0 低于建议下限 3%。

    集中度上限抬到 1.0 以隔离变量（否则满仓会先撞上集中度上限，命中另一条规则）。
    """
    product = make_product()
    return (
        make_client(max_single_product_ratio=1.0, max_single_class_ratio=1.0, max_single_issuer_ratio=1.0),
        singleton_products(product),
        {product.product_id: 1.0},
        (product.product_id,),
    )


def _trigger_diversification():
    """触发 S-DIVERSIFICATION「分散度建议」（warn → 放行，仅揭示）：仅持有 1 只产品，低于建议下限 3 只。"""
    product = make_product()
    return make_client(), singleton_products(product), {product.product_id: 0.5}, (product.product_id,)


# 规则号 -> 单点触发工厂。键集合必须与 ALL_RULES 完全一致，
# 由 test_triggers_cover_every_rule 守护，防止新增规则后测试静默失效。
TRIGGERS: dict[str, TriggerFactory] = {
    "S-RISK-MATCH": _trigger_risk_match,
    "S-HORIZON": _trigger_horizon,
    "S-CONC-SINGLE": _trigger_conc_single,
    "S-CONC-CLASS": _trigger_conc_class,
    "S-CONCENTRATION-ISSUER": _trigger_conc_issuer,
    "S-LIQUIDITY": _trigger_liquidity,
    "S-ENTRY": _trigger_entry,
    "S-ELDERLY": _trigger_elderly,
    "S-EXPERIENCE": _trigger_experience,
    "S-DERIVATIVE-BAN": _trigger_derivative_ban,
    "S-CURRENCY": _trigger_currency,
    "S-QUALIFIED": _trigger_qualified,
    "S-PROHIBITED": _trigger_prohibited,
    "S-WEIGHT": _trigger_weight,
    "S-FEASIBLE-POOL": _trigger_feasible_pool,
    "S-DUAL-RECORD": _trigger_dual_record,
    "S-FEE-DISCLOSE": _trigger_fee,
    "S-TAX": _trigger_tax,
    "S-COOLING": _trigger_cooling,
    "S-CONCENTRATION-WARNING": _trigger_concentration_warning,
    "S-CASH-RESERVE": _trigger_cash_reserve,
    "S-DIVERSIFICATION": _trigger_diversification,
}


def _context(factory: TriggerFactory) -> RuleContext:
    """把触发器输出组装成 RuleContext（内部复用 constraints 的硬约束判定结果）。"""
    client, products, weights, candidates = factory()
    portfolio = build_portfolio_obj(products, weights)
    return RuleContext.build(client, portfolio, products, candidates)


# ---------------------------------------------------------------------------
# 规则库自身
# ---------------------------------------------------------------------------
def test_rule_catalog_has_at_least_fifteen_rules():
    """不变式：规则库规模不少于 15 条（当前 22 条），防止规则被误删导致合规覆盖塌陷。"""
    assert len(ALL_RULES) >= 15


def test_rule_ids_are_unique():
    """不变式：规则编号全局唯一——编号是审计追踪与人工豁免清单的主键。"""
    ids = [rule.id for rule in ALL_RULES]
    assert len(ids) == len(set(ids))


def test_triggers_cover_every_rule():
    """触发器必须覆盖全部规则，避免新增规则后测试悄悄失效。"""
    assert set(TRIGGERS) == {rule.id for rule in ALL_RULES}


def test_every_rule_has_required_fields():
    """不变式：每条规则必须具备 S- 前缀编号、名称、分类、合规依据、合法级别（block/warn）与可调用 handler。"""
    for rule in ALL_RULES:
        assert rule.id.startswith("S-")
        assert rule.name
        assert rule.category
        assert rule.basis
        assert rule.severity in {"block", "warn"}
        assert callable(rule.handler)


def test_required_rule_ids_present():
    """硬性要求中逐条点名的规则号必须齐全。"""
    required = {
        "S-RISK-MATCH",
        "S-HORIZON",
        "S-CONC-SINGLE",
        "S-CONC-CLASS",
        "S-LIQUIDITY",
        "S-ENTRY",
        "S-ELDERLY",
        "S-EXPERIENCE",
        "S-DERIVATIVE-BAN",
        "S-DUAL-RECORD",
        "S-FEE-DISCLOSE",
        "S-CURRENCY",
        "S-TAX",
        "S-CONCENTRATION-ISSUER",
        "S-COOLING",
    }
    assert required <= {rule.id for rule in ALL_RULES}


def test_block_rules_have_repair_strategy_except_veto():
    """不变式：非 veto 的 block 规则必须定义修复策略——否则闸门无法在更小可行域内完成重配。"""
    for rule in ALL_RULES:
        if rule.severity == "block" and not rule.veto:
            assert rule.repair, f"{rule.id} 是 block 级但未定义修复策略"


def test_only_feasible_pool_is_veto():
    """不变式：唯一不可修复（veto）规则只能是 S-FEASIBLE-POOL——veto 意味着直接拒绝、无重配路径。"""
    veto_ids = {rule.id for rule in ALL_RULES if rule.veto}
    assert veto_ids == {"S-FEASIBLE-POOL"}


def test_rule_catalog_and_get_rule_agree():
    """不变式：rule_catalog / RULES_BY_ID / get_rule 三个入口对同一规则库的视图必须一致。

    同时锚定级别判定本身：S-RISK-MATCH 为 block，S-COOLING 不是 block。
    """
    catalog = {item["id"] for item in rule_catalog()}
    assert catalog == set(RULES_BY_ID)
    assert get_rule("S-RISK-MATCH").severity == "block"
    assert has_block_rule("S-ELDERLY")
    assert not has_block_rule("S-COOLING")


def test_format_rule_table_lists_all_ids():
    """口径：规则清单表格必须列出全部规则号（demo / README 展示口径）。"""
    table = format_rule_table()
    for rule in ALL_RULES:
        assert rule.id in table


def test_get_rule_unknown_raises():
    """异常路径：未知规则号必须抛 KeyError，不得静默返回 None 让调用方继续跑。"""
    with pytest.raises(KeyError):
        get_rule("S-NOT-EXIST")


# ---------------------------------------------------------------------------
# 逐条规则可触发
# ---------------------------------------------------------------------------
# 参数化数据直接取自 ALL_RULES：规则库新增一条，用例集自动扩展
# （配合 test_triggers_cover_every_rule 保证新规则必须先补上单点触发器）。
@pytest.mark.parametrize("rule_id", [rule.id for rule in ALL_RULES])
def test_every_rule_is_triggerable(rule_id: str):
    """逐条验证：该规则号必须能被对应单点场景命中（出现在 hit_rule_ids 中）。"""
    evaluation = evaluate_rules(_context(TRIGGERS[rule_id]))
    assert rule_id in evaluation.hit_rule_ids


def test_rule_severity_classification_is_consistent():
    """不变式：命中条目的 severity 必须与规则定义一致——handler 不得私自降级或升级。"""
    for rule_id, factory in TRIGGERS.items():
        evaluation = evaluate_rules(_context(factory))
        rule = RULES_BY_ID[rule_id]
        matched = [v for v in evaluation.hits if v.rule_id == rule_id]
        assert matched, rule_id
        assert all(v.severity == rule.severity for v in matched)


# ---------------------------------------------------------------------------
# 闸门行为
# ---------------------------------------------------------------------------
def test_gate_blocks_on_block_rule_and_requests_reoptimize():
    """闸门铁律：block 级 S-RISK-MATCH 命中必须打回重配（directive=reoptimize）并下发非空收紧指令。"""
    gate = SuitabilityGate(max_repair_rounds=2)
    decision = gate.review(_context(_trigger_risk_match), round_index=0)
    assert decision.passed is False
    assert decision.directive == "reoptimize"
    assert "S-RISK-MATCH" in [v.rule_id for v in decision.blocks]
    assert decision.tighten  # 必须下发收紧指令


def test_gate_escalates_when_repair_rounds_exhausted():
    """闸门铁律：重配轮次用尽（max_repair_rounds=0）仍不合规 → reject + escalated=True，且不再下发收紧指令。

    后一条很关键：既然要转人工，就不该再给出"重配指令"制造歧义。
    """
    gate = SuitabilityGate(max_repair_rounds=0)
    decision = gate.review(_context(_trigger_risk_match), round_index=0)
    assert decision.directive == "reject"
    assert decision.escalated is True
    assert decision.tighten == {}


def test_gate_passes_when_only_warn_rules_hit():
    """放行语义：仅命中 warn 级 S-COOLING 时必须 pass，且该告警必须出现在 warns 中待揭示。"""
    gate = SuitabilityGate()
    decision = gate.review(_context(_trigger_cooling), round_index=0)
    assert decision.passed is True
    assert decision.directive == "pass"
    assert "S-COOLING" in [v.rule_id for v in decision.warns]


def test_gate_veto_rule_rejects_directly():
    """闸门铁律：veto 规则 S-FEASIBLE-POOL 命中必须直接 reject，且 escalated=False——
    这不是"轮次用尽"，而是根本不存在修复路径。"""
    gate = SuitabilityGate()
    decision = gate.review(_context(_trigger_feasible_pool), round_index=0)
    assert decision.directive == "reject"
    assert decision.veto_rules == ["S-FEASIBLE-POOL"]
    assert decision.escalated is False


def test_gate_exemption_downgrades_block_to_warn():
    """豁免路径：人工豁免 S-RISK-MATCH 后该命中降级为 warn（detail 带【经人工豁免】），闸门放行且豁免清单留痕。"""
    gate = SuitabilityGate()
    decision = gate.review(_context(_trigger_risk_match), round_index=0, exempt_rules=["S-RISK-MATCH"])
    assert decision.passed is True
    assert decision.directive == "pass"
    assert decision.exempted_rules == ["S-RISK-MATCH"]
    assert any("经人工豁免" in v.detail for v in decision.warns)


def test_gate_comment_mentions_directive():
    """口径：闸门结论文案必须写明"打回"与命中的规则号，保证审计人员不查代码也能读懂结论。"""
    gate = SuitabilityGate()
    decision = gate.review(_context(_trigger_conc_single), round_index=0)
    assert "打回" in decision.comment
    assert "S-CONC-SINGLE" in decision.comment


def test_evaluate_rules_is_deterministic():
    """不变式：同一上下文两次求值结果完全一致（规则顺序与条目排序都稳定）。"""
    ctx = _context(_trigger_conc_class)
    first = [v.model_dump() for v in evaluate_rules(ctx).hits]
    second = [v.model_dump() for v in evaluate_rules(ctx).hits]
    assert first == second


def test_evaluate_rules_does_not_mutate_portfolio():
    """不变式：构建 RuleContext 是纯函数——求值前后组合对象必须逐字段不变。"""
    client, products, weights, candidates = _trigger_conc_class()
    portfolio = build_portfolio_obj(products, weights)
    before = portfolio.model_dump()
    RuleContext.build(client, portfolio, products, candidates)
    assert portfolio.model_dump() == before


# ---------------------------------------------------------------------------
# 打回重配的约束收紧构造
# ---------------------------------------------------------------------------
def test_build_tighten_for_elderly_rule():
    """收紧构造：S-ELDERLY（repair=elderly_protect）必须下发风险等级上限 3 与单一产品上限 20%。"""
    ctx = _context(_trigger_elderly)
    evaluation = evaluate_rules(ctx)
    blocks = [v for v in evaluation.blocks if v.rule_id == "S-ELDERLY"]
    spec = build_tighten(blocks, ctx)
    assert spec.risk_cap == 3
    assert spec.max_single_product_ratio == pytest.approx(0.20)


def test_build_tighten_excludes_offending_products():
    """收紧构造：S-EXPERIENCE（repair=exclude_products）必须把违规产品（默认 T-PROD-01）加入排除清单。"""
    ctx = _context(_trigger_experience)
    evaluation = evaluate_rules(ctx)
    blocks = [v for v in evaluation.blocks if v.rule_id == "S-EXPERIENCE"]
    spec = build_tighten(blocks, ctx)
    assert "T-PROD-01" in spec.excluded_product_ids


def test_build_tighten_for_concentration_scales_cap_down():
    """收紧构造：S-CONC-SINGLE（repair=single_cap_down）按 CAP_TIGHTEN_FACTOR=0.8 下调客户单一产品上限。"""
    ctx = _context(_trigger_conc_single)
    evaluation = evaluate_rules(ctx)
    blocks = [v for v in evaluation.blocks if v.rule_id == "S-CONC-SINGLE"]
    spec = build_tighten(blocks, ctx)
    assert spec.max_single_product_ratio == pytest.approx(0.3 * 0.8)


def test_build_tighten_is_empty_for_veto_rule():
    """收紧构造：S-FEASIBLE-POOL 无修复策略（repair 为空），收紧指令必须为空——它是拒绝而非重配。"""
    ctx = _context(_trigger_feasible_pool)
    evaluation = evaluate_rules(ctx)
    spec = build_tighten(evaluation.blocks, ctx)
    assert spec.is_empty()


def test_tighten_spec_from_gate_is_applicable():
    """口径：闸门下发的 tighten 字典必须能还原为 TightenSpec 并应用到客户档案，
    且结果只能收紧（risk_capacity 不得高于原值 2）。"""
    gate = SuitabilityGate()
    decision = gate.review(_context(_trigger_risk_match), round_index=0)
    client = make_client(risk_capacity=2)
    tightened = TightenSpec.from_dict(decision.tighten).apply(client)
    assert tightened.risk_capacity <= 2
