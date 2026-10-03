"""适当性规则库与硬闸门。

- `rules.py`：22 条适当性规则（block / warn），规则即数据 + 纯函数
- `engine.py`：规则求值器与 `SuitabilityGate` 硬闸门
"""

from __future__ import annotations

from .engine import (
    DEFAULT_MAX_REPAIR_ROUNDS,
    RuleEvaluation,
    SuitabilityGate,
    build_tighten,
    evaluate_rules,
    format_rule_table,
    gate_summary,
)
from .rules import (
    ALL_RULES,
    RULES_BY_ID,
    Rule,
    RuleContext,
    get_rule,
    rule_catalog,
)

__all__ = [
    "ALL_RULES",
    "RULES_BY_ID",
    "Rule",
    "RuleContext",
    "RuleEvaluation",
    "SuitabilityGate",
    "build_tighten",
    "evaluate_rules",
    "format_rule_table",
    "gate_summary",
    "get_rule",
    "rule_catalog",
    "DEFAULT_MAX_REPAIR_ROUNDS",
]
