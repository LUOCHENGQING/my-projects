"""工具层测试：JSON Schema 校验、权限分级、超时重试、幂等缓存、统一异常。"""

from __future__ import annotations

import time

import pytest

from src.tools import PermissionLevel, ToolRegistry, ToolSpec


# ---------------------------------------------------------------------------
# 1. 工具声明属性完整性
# ---------------------------------------------------------------------------
def test_all_required_tools_registered_with_full_attributes(registry):
    """题目要求的 5 个工具必须齐备，且每个都带齐全部属性。"""
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
    """权限分级确实被用起来了，而不是全都一个等级。"""
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
    result = registry.call("search_filings", {"top_k": 3})
    assert result.ok is False
    assert result.error["code"] == "SCHEMA_INVALID"
    assert any("query" in e for e in result.error["detail"]["errors"])


def test_schema_rejects_wrong_type_and_range(registry):
    wrong_type = registry.call("search_filings", {"query": "净利润", "top_k": "五条"})
    assert wrong_type.ok is False and wrong_type.error["code"] == "SCHEMA_INVALID"

    out_of_range = registry.call("search_filings", {"query": "净利润", "top_k": 999})
    assert out_of_range.ok is False
    assert any("不得大于" in e for e in out_of_range.error["detail"]["errors"])


def test_schema_rejects_unknown_extra_argument(registry):
    result = registry.call("calc_ratio", {"ratio_name": "net_margin", "numerator": 1,
                                          "denominator": 2, "bogus": 1})
    assert result.ok is False
    assert any("额外参数" in e for e in result.error["detail"]["errors"])


def test_schema_rejects_enum_violation(registry):
    result = registry.call("calc_ratio", {"ratio_name": "not_a_ratio", "numerator": 1, "denominator": 2})
    assert result.ok is False
    assert result.error["code"] == "SCHEMA_INVALID"


def test_schema_defaults_are_applied(registry):
    """未传 top_k 时应按 schema 的 default 生效，而不是报错。"""
    result = registry.call("search_filings", {"query": "营业收入 净利润"})
    assert result.ok is True
    assert result.data["count"] <= 5  # default top_k = 5


# ---------------------------------------------------------------------------
# 3. 权限分级
# ---------------------------------------------------------------------------
def test_permission_denied_blocks_call(registry):
    result = registry.call(
        "cite_source",
        {"source_id": "EX-TECH-2024-AR"},
        granted={PermissionLevel.PUBLIC_READ, PermissionLevel.COMPUTE},
    )
    assert result.ok is False
    assert result.error["code"] == "PERMISSION_DENIED"
    assert result.error["detail"]["required"] == "write"


def test_granted_permission_allows_call(registry):
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
    result = registry.call("does_not_exist", {})
    assert result.ok is False
    assert result.error["code"] == "TOOL_NOT_FOUND"


def test_business_error_is_wrapped(registry):
    """工具内部抛出的业务异常被收敛成 EXECUTION_ERROR，调用方不会崩。"""
    result = registry.call("calc_ratio", {"ratio_name": "net_margin", "numerator": 1, "denominator": 0})
    assert result.ok is False
    assert result.error["code"] == "EXECUTION_ERROR"
    assert "分母不能为 0" in result.error["message"]


# ---------------------------------------------------------------------------
# 5. 超时与重试
# ---------------------------------------------------------------------------
def test_timeout_is_enforced_and_retried():
    registry = ToolRegistry()

    def slow(**_kwargs):
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
    result = registry.call("slow_tool", {})
    assert result.ok is False
    assert result.error["code"] == "TIMEOUT"
    assert result.attempts == 2  # 首次 + 1 次重试


def test_transient_failure_recovers_on_retry():
    """瞬时故障（超时/连接断开）应当被判定为可重试并自动恢复。"""
    registry = ToolRegistry()
    calls = {"n": 0}

    def flaky(**_kwargs):
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
    """业务异常不属于瞬时故障，不应浪费重试。"""
    registry = ToolRegistry()
    calls = {"n": 0}

    def boom(**_kwargs):
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
    registry = ToolRegistry()
    counter = {"n": 0}

    def count(**_kwargs):
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
    """cite_source 声明为非幂等（每次分配新引用序号），不能被缓存吞掉。"""
    first = registry.call("cite_source", {"source_id": "EX-TECH-2024-AR"}, granted={PermissionLevel.WRITE})
    second = registry.call("cite_source", {"source_id": "EX-TECH-2024-AR"}, granted={PermissionLevel.WRITE})
    assert first.ok and second.ok
    assert second.cached is False
    assert second.data["citation_no"] > first.data["citation_no"]


def test_call_log_records_every_call(registry):
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
    result = registry.call("get_financial_metric",
                           {"company": "示例科技", "metric": "营收", "year": 2024})
    assert result.ok is True
    assert result.data["metric"] == "营业收入"      # 别名归一化
    assert result.data["value"] == pytest.approx(1286400.0)
    assert result.data["source_id"] == "EX-TECH-2024-AR"


def test_calc_ratio_growth_formula(registry):
    result = registry.call("calc_ratio",
                           {"ratio_name": "growth_rate", "numerator": 110, "denominator": 100})
    assert result.ok is True
    assert result.data["value"] == pytest.approx(0.10)
    assert result.data["display"] == "10.0%"


def test_check_risk_rules_finds_high_risk(registry):
    result = registry.call("check_risk_rules", {"company": "示例科技股份有限公司", "year": 2024})
    assert result.ok is True
    assert result.data["overall_level"] == "high"
    assert result.data["entity_type"] == "non_financial"
    rule_ids = {f["rule_id"] for f in result.data["findings"]}
    assert "R-CASH-01" in rule_ids          # 净利润现金含量过低
    assert "R-AR-01" in rule_ids            # 应收增速显著高于收入增速


def test_check_risk_rules_skips_corporate_rules_for_bank(registry):
    """银行不套用工商企业的杠杆警戒线（银行资产负债率天然 >90%）。"""
    result = registry.call("check_risk_rules", {"company": "示例智造银行股份有限公司", "year": 2024})
    assert result.ok is True
    assert result.data["entity_type"] == "financial"
    rule_ids = {f["rule_id"] for f in result.data["findings"]}
    assert not any(r.startswith("R-DEBT") for r in rule_ids)
