"""轻量 JSON Schema 校验器（最小权限工具层的第二道闸门）。

层次与职责：
    tools 包内的纯工具模块，零第三方依赖，只被 src/tools/registry.py 使用：
    ToolRegistry.call() 先 apply_defaults 填默认值，再 validate 做**强校验**，
    未通过就返回 code=SCHEMA_INVALID 的失败结果，绝不让幻觉参数落到业务逻辑。

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
    注：实际实现为——default 不在校验阶段生效，_validate_node 会忽略它；
    缺省值填充由 apply_defaults 在校验前完成，且仅覆盖顶层 properties。

对外关键函数：
    validate(value, schema)      校验失败抛 SchemaValidationError（注册中心走这条）
    check(value, schema)         校验但只返回错误列表（便于单测/预检）
    apply_defaults(value, schema) 按 schema 填顶层默认值，返回新 dict
    SchemaValidationError        携带 errors: List[str] 的异常类型
    注：__all__ 只列了 validate 与 SchemaValidationError，
    check / apply_defaults 属公开函数但未列入 __all__（注册中心按名直接导入）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

__all__ = ["validate", "SchemaValidationError"]

# JSON Schema 类型名 -> Python 类型；"number" 同时接受 int 与 float
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
    """schema 校验失败，携带全部错误路径。

    继承 ValueError，便于调用方用宽泛 except 兜住。
    关键属性：
        errors: List[str] —— 每条形如 "$.top_k: 期望类型 integer，实际为 string"，
                路径用 $.字段 表示，直接可读，适合回灌给 LLM 自纠。
    状态流转：
        一次 validate 会累积**全部**错误后再抛出，因此不是"首错即停"；
        唯一提前 return 的情形是类型不匹配（后续长度/区间检查已无意义）。
    """

    def __init__(self, errors: Sequence[str]) -> None:
        """构造校验异常：errors 为逐条错误信息，并拼成异常消息供 LLM 自纠。"""
        # 复制一份并保留为公开属性，避免调用方拿到可变的入参序列
        self.errors: List[str] = list(errors)
        super().__init__("; ".join(self.errors))


def _type_name(value: Any) -> str:
    """把 Python 值映射回 JSON Schema 的类型名，仅用于拼装报错信息。

    参数：
        value：任意被校验的值。
    返回值：
        str，如 "null" / "boolean" / "integer" / "number" / "string" /
        "array" / "object"；自定义类型返回其类名。
    副作用 / 异常：无。
    """
    if value is None:
        return "null"
    # 先按精确类型查表，避免 isinstance 把 bool 当成 integer 造成误导性报错
    return {
        bool: "boolean",
        int: "integer",
        float: "number",
        str: "string",
        list: "array",
        dict: "object",
    }.get(type(value), type(value).__name__)


def _match_type(value: Any, expected: str) -> bool:
    """判断 value 是否满足 schema 声明的某个类型。

    参数：
        value：被校验的值。
        expected：JSON Schema 类型名（string/integer/number/boolean/array/object/null）。
    返回值：
        bool。注：实际实现为——未知类型名一律返回 True（宽松放行），
        以保证未来新增关键字不会让老工具集体校验失败。
    副作用 / 异常：无。
    """
    python_type = _TYPE_MAP.get(expected)
    if python_type is None:
        return True
    # bool 是 int 的子类，必须单独排除，否则 True 会被判成合法 integer
    if expected in {"integer", "number"} and isinstance(value, bool):
        return False
    # 3.0 这类"整数值的浮点"按整数放行，兼容 JSON 解析与 numpy 计算的差异
    if expected == "integer" and isinstance(value, float):
        return value.is_integer()
    return isinstance(value, python_type)


def _validate_node(value: Any, schema: Dict[str, Any], path: str, errors: List[str]) -> None:
    """递归校验单个节点，把错误描述追加进 errors（不做任何截断）。

    参数：
        value：当前节点的值。
        schema：当前节点对应的 JSON Schema 片段；空 dict / 非 dict 视为"无约束"。
        path：错误定位路径，从 "$" 起，如 "$.results[0].source_id"。
        errors：累积错误列表（原地追加，是本函数唯一的"输出通道"）。
    返回值：
        None。
    副作用 / 异常：
        原地修改 errors；自身不抛异常——是否失败由调用方（validate/check）决定。
        注：遇到类型不匹配会立即 return，跳过该节点的长度/区间/子结构检查，
        目的是让报错信息聚焦在根因上。
    """
    # 非 dict 或空 schema 表示该节点不设约束（对应 JSON Schema 的 true）
    if not isinstance(schema, dict) or not schema:
        return

    # oneOf：至少满足一个分支
    # 分支校验各自用独立列表，避免试探性错误污染最终报错
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

    # const 优先于 enum/type：常量不匹配就没必要继续报其它错
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: 期望常量 {schema['const']!r}，实际 {value!r}")
        return

    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(x) for x in schema["enum"])
        errors.append(f"{path}: 取值必须是 [{allowed}] 之一，实际为 {value!r}")

    # type 允许写成列表（联合类型），任一命中即通过
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
        # 声明了 minLength 的字段不接受纯空白：挡住了 " " 这类看似非空实则无效的输入
        if schema.get("minLength", 0) and not value.strip():
            errors.append(f"{path}: 不允许为空白字符串")

    # 区间检查前再次排除 bool：True/False 参与大小比较没有业务含义
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
        # 只支持 items 为单一 schema 的写法（不支持 tuple 形式），够工具入参使用
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                _validate_node(item, item_schema, f"{path}[{idx}]", errors)

    if isinstance(value, dict):
        properties: Dict[str, Any] = schema.get("properties", {}) or {}
        # 必填缺失是 LLM 最常见的错误，先把它们全部报出来再校验具体字段
        for name in schema.get("required", []) or []:
            if name not in value:
                errors.append(f"{path}.{name}: 缺少必填参数")
        for name, sub in properties.items():
            if name in value:
                _validate_node(value[name], sub, f"{path}.{name}", errors)
        # additionalProperties=False 是"最小权限"的体现：多余参数（幻觉字段）直接拒绝
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append(f"{path}.{name}: 不接受的额外参数")


def apply_defaults(value: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """按 schema 的 default 填充缺省值（仅顶层 properties，工具入参足够）。

    参数：
        value：调用方传入的原始参数 dict。
        schema：工具入参 schema（type=object）。
    返回值：
        dict —— 填好缺省值的**新字典**；已有键一律不覆盖。
    副作用 / 异常：
        不修改入参 value；schema 不合法（properties 非 dict）时按空处理，不抛异常。
    """
    out = dict(value)
    for name, sub in (schema.get("properties") or {}).items():
        # 只认 dict 形式的子 schema；"default": None 也算有效默认值（键存在即填充）
        if name not in out and isinstance(sub, dict) and "default" in sub:
            out[name] = sub["default"]
    return out


def validate(value: Any, schema: Optional[Dict[str, Any]]) -> None:
    """校验失败抛 SchemaValidationError。

    参数：
        value：待校验值（工具调用场景是 apply_defaults 之后的参数 dict）。
        schema：入参 schema；传 None 表示无约束（等价于放行）。
    返回值：
        None（通过时静默返回）。
    副作用 / 异常：
        无副作用；校验不通过时抛 SchemaValidationError，errors 为全部错误描述。
    """
    errors: List[str] = []
    # 根路径固定为 "$"，与 JSONPath/JSON Schema 的错误定位习惯一致
    _validate_node(value, schema or {}, "$", errors)
    if errors:
        raise SchemaValidationError(errors)


def check(value: Any, schema: Optional[Dict[str, Any]]) -> List[str]:
    """校验但只返回错误列表，不抛异常。

    参数：
        value：待校验值。
        schema：入参 schema；None 表示无约束。
    返回值：
        List[str]，空列表表示通过。
    副作用 / 异常：
        无副作用、不抛异常；适合需要"先收集再决策"的场景（如一次报告多个字段问题）。
    """
    errors: List[str] = []
    _validate_node(value, schema or {}, "$", errors)
    return errors
