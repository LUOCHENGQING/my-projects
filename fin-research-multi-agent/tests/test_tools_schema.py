"""工具层测试：JSON Schema 校验、权限分级、超时重试、幂等缓存、统一异常。

被测行为（src.tools 的 ToolSpec / ToolRegistry / PermissionLevel 与内置业务工具）：
1. 声明完整性：5 个业务工具齐备，且描述、JSON Schema、权限等级、超时、幂等、重试次数等属性齐全；
2. 入参校验：缺必填、类型错、超范围、多余参数、枚举越界都必须被 SCHEMA_INVALID 拒绝，schema 的 default 必须生效；
3. 权限分级：四档权限被区分使用，未授权调用 PERMISSION_DENIED，授权后放行；
4. 统一异常：未知工具与业务异常都收敛成错误对象（TOOL_NOT_FOUND / EXECUTION_ERROR），调用方不会崩；
5. 超时与重试：超时被强制中断并按 max_retries 重试，瞬时故障可自愈，业务异常不重试；
6. 幂等缓存：幂等工具命中缓存只执行一次，非幂等工具（cite_source）不得被缓存吞掉；
7. 业务正确性：指标查询的数值与来源、比率公式、风险规则命中与银行口径豁免。

覆盖策略：正常（工具主路径与业务规则）、边界（default 生效、恰好用满重试、空参数）、
异常（超时 / 业务异常 / 未知工具 / 未授权）、对抗（多余参数、类型欺骗、越权调用、缓存穿透）。
"""

from __future__ import annotations

import time

import pytest

from src.tools import PermissionLevel, ToolRegistry, ToolSpec


# ---------------------------------------------------------------------------
# 1. 工具声明属性完整性
# ---------------------------------------------------------------------------
def test_all_required_tools_registered_with_full_attributes(registry):
    """验证规则：题目要求的 5 个工具必须齐备，且每个都带齐描述、schema、权限、超时、幂等与重试属性（参数需有 type）。"""
    expected = {"search_filings", "get_financial_metric", "calc_ratio", "check_risk_rules", "cite_source"}
    assert expected.issubset(set(registry.names()))

    for spec in registry.specs():
        assert spec.name
        assert spec.description and len(spec.description) > 10
        assert isinstance(spec.schema, dict) and spec.schema.get("type") == "object"
        assert isinstance(spec.permission_level, PermissionLevel)
        assert spec.timeout_s > 0
        assert isinstance(spec.idempotent, bool)
        assert spec.max_retries >= 0
        # 参数必须有 JSON Schema 描述，才能既做校验又当 function-calling 声明
        for prop in (spec.schema.get("properties") or {}).values():
            assert "type" in prop


def test_permission_levels_are_differentiated(registry):
    """验证规则：权限分级确实被用起来——四个工具分属四档不同等级，而不是统统一个等级。"""
    # 先摊平成 name -> level，逐个断言分工后，再用去重档数兜底
    levels = {spec.name: spec.permission_level for spec in registry.specs()}
    assert levels["search_filings"] == PermissionLevel.PUBLIC_READ
    assert levels["calc_ratio"] == PermissionLevel.COMPUTE
    assert levels["check_risk_rules"] == PermissionLevel.RESTRICTED_READ
    assert levels["cite_source"] == PermissionLevel.WRITE
    assert len(set(levels.values())) == 4


# ---------------------------------------------------------------------------
# 2. JSON Schema 入参校验
# ---------------------------------------------------------------------------
def test_schema_rejects_missing_required_argument(registry):
    """验证规则：缺少必填参数必须被 schema 拦截，且错误细节要点名缺的是哪个参数。"""
    # 只传 top_k、故意漏掉 query：验证的是「必填」而不是「类型」错误
    result = registry.call("search_filings", {"top_k": 3})
    assert result.ok is False
    assert result.error["code"] == "SCHEMA_INVALID"
    assert any("query" in e for e in result.error["detail"]["errors"])


def test_schema_rejects_wrong_type_and_range(registry):
    """验证规则：参数类型错误与数值越界都必须被 schema 拒绝（同一校验通道的两种失败形态）。"""
    # 用中文字符串冒充整数、用 999 突破上限，分别覆盖 type 与 maximum 两条约束
    wrong_type = registry.call("search_filings", {"query": "净利润", "top_k": "五条"})
    assert wrong_type.ok is False and wrong_type.error["code"] == "SCHEMA_INVALID"

    out_of_range = registry.call("search_filings", {"query": "净利润", "top_k": 999})
    assert out_of_range.ok is False
    assert any("不得大于" in e for e in out_of_range.error["detail"]["errors"])


def test_schema_rejects_unknown_extra_argument(registry):
    """验证规则：additionalProperties=False 时多余参数必须被拒，防止调用方拼错字段名却被静默忽略。"""
    # bogus 不在 schema 里，专门用来验证多余参数没有被放开
    result = registry.call("calc_ratio", {"ratio_name": "net_margin", "numerator": 1,
                                          "denominator": 2, "bogus": 1})
    assert result.ok is False
    assert any("额外参数" in e for e in result.error["detail"]["errors"])


def test_schema_rejects_enum_violation(registry):
    """验证规则：ratio_name 不在枚举白名单内必须被拒，避免未定义的比率名进入计算层。"""
    result = registry.call("calc_ratio", {"ratio_name": "not_a_ratio", "numerator": 1, "denominator": 2})
    assert result.ok is False
    assert result.error["code"] == "SCHEMA_INVALID"


def test_schema_defaults_are_applied(registry):
    """验证规则：未传 top_k 时应按 schema 的 default 生效返回结果，而不是报缺参。"""
    # 只给 query 一个参数，验证 default 生效这条路
    result = registry.call("search_filings", {"query": "营业收入 净利润"})
    assert result.ok is True
    assert result.data["count"] <= 5  # default top_k = 5


# ---------------------------------------------------------------------------
# 3. 权限分级
# ---------------------------------------------------------------------------
def test_permission_denied_blocks_call(registry):
    """验证规则：未授予 write 权限时必须拒绝执行，并回报所需等级（最小权限的强制点）。"""
    # 只授予读与计算两档权限，缺少 cite_source 所需的 write
    result = registry.call(
        "cite_source",
        {"source_id": "EX-TECH-2024-AR"},
        granted={PermissionLevel.PUBLIC_READ, PermissionLevel.COMPUTE},
    )
    assert result.ok is False
    assert result.error["code"] == "PERMISSION_DENIED"
    assert result.error["detail"]["required"] == "write"


def test_granted_permission_allows_call(registry):
    """验证规则：显式授予 write 后同一调用必须放行，且结果回传被引用的来源。"""
    result = registry.call(
        "cite_source",
        {"source_id": "EX-TECH-2024-AR"},
        granted={PermissionLevel.WRITE},
    )
    assert result.ok is True
    assert result.data["source_id"] == "EX-TECH-2024-AR"


# ---------------------------------------------------------------------------
# 4. 统一异常处理
# ---------------------------------------------------------------------------
def test_unknown_tool_returns_error_not_exception(registry):
    """验证规则：调用不存在的工具必须返回 TOOL_NOT_FOUND 错误对象，而不是抛异常打断调用方。"""
    result = registry.call("does_not_exist", {})
    assert result.ok is False
    assert result.error["code"] == "TOOL_NOT_FOUND"


def test_business_error_is_wrapped(registry):
    """验证规则：工具内部抛出的业务异常被收敛成 EXECUTION_ERROR，并保留原始提示，调用方不会崩。"""
    # 构造除零场景，触发工具内部的业务校验异常
    result = registry.call("calc_ratio", {"ratio_name": "net_margin", "numerator": 1, "denominator": 0})
    assert result.ok is False
    assert result.error["code"] == "EXECUTION_ERROR"
    assert "分母不能为 0" in result.error["message"]


# ---------------------------------------------------------------------------
# 5. 超时与重试
# ---------------------------------------------------------------------------
def test_timeout_is_enforced_and_retried():
    """验证规则：执行超时被强制中断，并按 max_retries 重试（总尝试次数 = 1 + max_retries）。"""
    registry = ToolRegistry()

    def slow(**_kwargs):
        """桩工具：睡眠时长超过 timeout_s，用于验证超时与重试计数。"""
        # 睡眠时长远大于 timeout_s，确保第一次尝试必然超时
        time.sleep(0.4)
        return {"done": True}

    registry.register(
        ToolSpec(
            name="slow_tool",
            description="用于验证超时的慢工具",
            schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=slow,
            timeout_s=0.08,
            max_retries=1,
            idempotent=False,
        )
    )
    # 超时 + 重试后仍然失败：用 attempts 验证重试次数确实按 max_retries 生效
    result = registry.call("slow_tool", {})
    assert result.ok is False
    assert result.error["code"] == "TIMEOUT"
    assert result.attempts == 2  # 首次 + 1 次重试


def test_transient_failure_recovers_on_retry():
    """验证规则：瞬时故障（超时 / 连接断开）被判定为可重试，第二次尝试成功后正常返回数据。"""
    registry = ToolRegistry()
    # 用计数器实现「第一次抛超时、第二次成功」，同时可断言只重试了一次
    calls = {"n": 0}

    def flaky(**_kwargs):
        """桩工具：首次调用抛超时异常、之后成功，用于验证瞬时故障重试。"""
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("模拟第一次网络超时")
        return {"value": 42}

    registry.register(
        ToolSpec(
            name="flaky_tool",
            description="第一次抛超时异常、第二次成功",
            schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=flaky,
            timeout_s=1.0,
            max_retries=2,
        )
    )
    result = registry.call("flaky_tool", {})
    assert result.ok is True
    assert result.attempts == 2
    assert calls["n"] == 2
    assert result.data == {"value": 42}


def test_business_error_is_not_retried():
    """验证规则：业务异常（ValueError）不属于瞬时故障，不得重试，避免浪费配额并掩盖错误。"""
    registry = ToolRegistry()
    # max_retries 故意设成 3：若实现误把业务异常当可重试，attempts 会立刻暴露
    calls = {"n": 0}

    def boom(**_kwargs):
        """桩工具：每次调用都抛业务异常，用于验证业务异常不重试。"""
        calls["n"] += 1
        raise ValueError("业务规则不满足")

    registry.register(
        ToolSpec(
            name="boom_tool",
            description="始终抛业务异常",
            schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=boom,
            timeout_s=1.0,
            max_retries=3,
        )
    )
    result = registry.call("boom_tool", {})
    assert result.ok is False
    assert result.attempts == 1
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# 6. 幂等键缓存
# ---------------------------------------------------------------------------
def test_idempotent_tool_is_cached():
    """验证规则：声明为幂等的工具第二次调用命中缓存，handler 只执行一次且两次结果相同。"""
    registry = ToolRegistry()
    # 用计数器当 handler 的副作用探针：调用次数是「缓存是否真的生效」的唯一证据
    counter = {"n": 0}

    def count(**_kwargs):
        """桩工具：返回累计调用次数，用于验证幂等缓存命中与不缓存的分支。"""
        counter["n"] += 1
        return {"n": counter["n"]}

    registry.register(
        ToolSpec(
            name="counter",
            description="计数工具",
            schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=count,
            idempotent=True,
        )
    )
    first = registry.call("counter", {})
    second = registry.call("counter", {})
    assert first.ok and second.ok
    assert first.cached is False and second.cached is True
    assert counter["n"] == 1
    assert second.data == first.data


def test_non_idempotent_tool_is_not_cached(registry):
    """验证规则：cite_source 声明为非幂等（每次分配新引用序号），因此不得被缓存吞掉。"""
    # 同一份资料引用两次：第二次必须真的执行，序号继续递增
    first = registry.call("cite_source", {"source_id": "EX-TECH-2024-AR"}, granted={PermissionLevel.WRITE})
    second = registry.call("cite_source", {"source_id": "EX-TECH-2024-AR"}, granted={PermissionLevel.WRITE})
    assert first.ok and second.ok
    assert second.cached is False
    assert second.data["citation_no"] > first.data["citation_no"]


def test_call_log_records_every_call(registry):
    """验证契约：每次调用（含失败）都按序进调用日志，drain 取出后日志清空。"""
    # 一成一败两次调用：验证失败也会留痕，并带上 error_code
    registry.call("search_filings", {"query": "净利润"})
    registry.call("nope", {})
    log = registry.drain_call_log()
    assert len(log) == 2
    assert log[0]["ok"] is True
    assert log[1]["ok"] is False and log[1]["error_code"] == "TOOL_NOT_FOUND"
    assert registry.drain_call_log() == []  # 取出后清空


# ---------------------------------------------------------------------------
# 7. 业务工具正确性
# ---------------------------------------------------------------------------
def test_get_financial_metric_returns_value_and_provenance(registry):
    """验证规则：指标查询返回归一后的指标名、数值与来源文件（结论可回溯到资料）。"""
    # 公司传简称、指标传别名，一次覆盖两条归一化路径
    result = registry.call("get_financial_metric",
                           {"company": "示例科技", "metric": "营收", "year": 2024})
    assert result.ok is True
    assert result.data["metric"] == "营业收入"      # 别名归一化
    assert result.data["value"] == pytest.approx(1286400.0)
    assert result.data["source_id"] == "EX-TECH-2024-AR"


def test_calc_ratio_growth_formula(registry):
    """验证规则：growth_rate = (numerator - denominator) / denominator，且展示格式化为一位小数百分比。"""
    # 110 与 100 便于心算校验 10% 这一期望值
    result = registry.call("calc_ratio",
                           {"ratio_name": "growth_rate", "numerator": 110, "denominator": 100})
    assert result.ok is True
    assert result.data["value"] == pytest.approx(0.10)
    assert result.data["display"] == "10.0%"


def test_check_risk_rules_finds_high_risk(registry):
    """验证规则：工商企业口径下「净利润现金含量过低」与「应收增速显著高于收入增速」必须同时命中，整体风险判为 high。"""
    result = registry.call("check_risk_rules", {"company": "示例科技股份有限公司", "year": 2024})
    assert result.ok is True
    assert result.data["overall_level"] == "high"
    assert result.data["entity_type"] == "non_financial"
    rule_ids = {f["rule_id"] for f in result.data["findings"]}
    assert "R-CASH-01" in rule_ids          # 净利润现金含量过低
    assert "R-AR-01" in rule_ids            # 应收增速显著高于收入增速


def test_check_risk_rules_skips_corporate_rules_for_bank(registry):
    """验证规则：银行不套用工商企业的杠杆警戒线（银行资产负债率天然 >90%），因此 R-DEBT* 规则必须被跳过。"""
    result = registry.call("check_risk_rules", {"company": "示例智造银行股份有限公司", "year": 2024})
    assert result.ok is True
    assert result.data["entity_type"] == "financial"
    rule_ids = {f["rule_id"] for f in result.data["findings"]}
    assert not any(r.startswith("R-DEBT") for r in rule_ids)
