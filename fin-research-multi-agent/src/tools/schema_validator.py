"""轻量 JSON Schema 校验器。

为什么自己写而不用 jsonschema 包：
    1. 工具入参 schema 只用到一个很小的子集（type/properties/required/enum/数值区间/
       长度/数组项），自己实现约 100 行，零依赖、报错信息可控、可读性强；
    2. 报错信息是给 LLM 看的——需要精确到「哪个字段、期望什么、实际得到什么」，
       自定义实现能保证这种格式，方便 Agent 自我纠错后重试。

支持的关键字：
    type, properties, required, enum, const, default,
    minimum, maximum, exclusiveMinimum, exclusiveMaximum,
    minLength, maxLength, minItems, maxItems, items,
    additionalProperties, oneOf, description
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

__all__ = ["validate", "SchemaValidationError"]

_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


class SchemaValidationError(ValueError):
    """schema 校验失败，携带全部错误路径。"""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors: List[str] = list(errors)
        super().__init__("; ".join(self.errors))


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    return {
        bool: "boolean",
        int: "integer",
        float: "number",
        str: "string",
        list: "array",
        dict: "object",
    }.get(type(value), type(value).__name__)


def _match_type(value: Any, expected: str) -> bool:
    python_type = _TYPE_MAP.get(expected)
    if python_type is None:
        return True
    # bool 是 int 的子类，必须单独排除，否则 True 会被判成合法 integer
    if expected in {"integer", "number"} and isinstance(value, bool):
        return False
    if expected == "integer" and isinstance(value, float):
        return value.is_integer()
    return isinstance(value, python_type)


def _validate_node(value: Any, schema: Dict[str, Any], path: str, errors: List[str]) -> None:
    if not isinstance(schema, dict) or not schema:
        return

    # oneOf：至少满足一个分支
    if "oneOf" in schema:
        branches = schema["oneOf"]
        sub_errors: List[List[str]] = []
        for branch in branches:
            branch_errors: List[str] = []
            _validate_node(value, branch, path, branch_errors)
            if not branch_errors:
                return
            sub_errors.append(branch_errors)
        errors.append(f"{path}: 不满足 oneOf 的任何一个分支")
        return

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: 期望常量 {schema['const']!r}，实际 {value!r}")
        return

    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(x) for x in schema["enum"])
        errors.append(f"{path}: 取值必须是 [{allowed}] 之一，实际为 {value!r}")

    expected_type = schema.get("type")
    if expected_type is not None:
        types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(_match_type(value, t) for t in types):
            errors.append(f"{path}: 期望类型 {'/'.join(types)}，实际为 {_type_name(value)}")
            return  # 类型不对，后续长度/区间检查没有意义

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: 长度至少 {schema['minLength']}，实际 {len(value)}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: 长度至多 {schema['maxLength']}，实际 {len(value)}")
        if schema.get("minLength", 0) and not value.strip():
            errors.append(f"{path}: 不允许为空白字符串")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: 不得小于 {schema['minimum']}，实际 {value}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: 不得大于 {schema['maximum']}，实际 {value}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: 必须大于 {schema['exclusiveMinimum']}，实际 {value}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: 必须小于 {schema['exclusiveMaximum']}，实际 {value}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: 元素个数至少 {schema['minItems']}，实际 {len(value)}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: 元素个数至多 {schema['maxItems']}，实际 {len(value)}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                _validate_node(item, item_schema, f"{path}[{idx}]", errors)

    if isinstance(value, dict):
        properties: Dict[str, Any] = schema.get("properties", {}) or {}
        for name in schema.get("required", []) or []:
            if name not in value:
                errors.append(f"{path}.{name}: 缺少必填参数")
        for name, sub in properties.items():
            if name in value:
                _validate_node(value[name], sub, f"{path}.{name}", errors)
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append(f"{path}.{name}: 不接受的额外参数")


def apply_defaults(value: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """按 schema 的 default 填充缺省值（仅顶层 properties，工具入参足够）。"""
    out = dict(value)
    for name, sub in (schema.get("properties") or {}).items():
        if name not in out and isinstance(sub, dict) and "default" in sub:
            out[name] = sub["default"]
    return out


def validate(value: Any, schema: Optional[Dict[str, Any]]) -> None:
    """校验失败抛 SchemaValidationError。"""
    errors: List[str] = []
    _validate_node(value, schema or {}, "$", errors)
    if errors:
        raise SchemaValidationError(errors)


def check(value: Any, schema: Optional[Dict[str, Any]]) -> List[str]:
    """校验但只返回错误列表，不抛异常。"""
    errors: List[str] = []
    _validate_node(value, schema or {}, "$", errors)
    return errors
