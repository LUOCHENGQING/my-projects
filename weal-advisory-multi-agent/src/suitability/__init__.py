"""适当性规则库与硬闸门。

所属层次
--------
领域规则层（`src/suitability/`）：不依赖 Agent、不调用模型、不写流程状态；
向上被 Agent 层（`src/agents/suitability_officer.py` 及其工具封装）使用，
向下只依赖纯工具层（`src/constraints.py`、`src/schemas.py`、`src/utils.py`）。

解决什么问题
------------
把"适当性"做成一个**可枚举、可单测、结论可复现**的独立子系统：
规则是数据（22 条），求值与闸门是纯函数，命中后下发的是**约束收紧指令**
而不是让模型重新表述——这样合规结论不会随模型波动而改变。

对外暴露（`__all__` 即公开契约）
--------------------------------
- 规则层（来自 `rules`）：`Rule`、`RuleContext`、`ALL_RULES`、`RULES_BY_ID`、
  `get_rule`、`rule_catalog`
- 引擎层（来自 `engine`）：`RuleEvaluation`、`SuitabilityGate`、
  `evaluate_rules`、`build_tighten`、`gate_summary`、`format_rule_table`、
  `DEFAULT_MAX_REPAIR_ROUNDS`
（注：`engine.liquidity_line` 未列入本包 `__all__`，需从 `engine` 直接导入。）

被谁调用
--------
`src/agents/tools.py`（`suitability.*` 系列工具）、
`eval/run_eval.py`（`RuleContext` + `SuitabilityGate` 统计规则命中）、
`tests/test_suitability_rules.py`。

模块分工
--------
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

#: 本包的公开契约：外部（Agent 工具层、评估脚本、单测）只应通过这些名字访问，
#: 这样规则表与闸门的内部实现可以独立演进。
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
