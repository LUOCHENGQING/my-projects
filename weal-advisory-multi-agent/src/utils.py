"""通用工具：规范化序列化、摘要（digest）、时间戳、浮点比较。

设计要点
--------
1. `canonical_json` 对浮点做定点量化后再序列化，保证同一份数值在不同机器、
   不同调用顺序下产生**完全一致的摘要**（trace 与版本链的哈希都依赖它）。
2. 所有比较都带显式容差 `EPS`，避免浮点误差把可行组合误判为越界。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any

# 权重类比较容差（1e-9 足以覆盖 16 位双精度累加误差）
WEIGHT_TOL = 1e-9
# 数值比率类比较容差
RATIO_TOL = 1e-9
# 浮点量化精度
QUANTIZE = 12


def _normalize(obj: Any) -> Any:
    """把任意对象递归转换为可稳定序列化的结构。"""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        # 量化后再转 float，消除 0.30000000000000004 这类噪声；
        # 整数值统一折叠为 int（1.0 与 1 产生同一份摘要，提升跨模块摘要稳定性）
        value = round(obj, QUANTIZE)
        if value == 0:
            return 0
        if float(value).is_integer() and abs(value) < 1e15:
            return int(value)
        return value
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = [_normalize(v) for v in obj]
        if isinstance(obj, (set, frozenset)):
            items = sorted(items, key=lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True))
        return items
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _normalize(dataclasses.asdict(obj))
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return _normalize(model_dump())
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return _normalize(to_dict())
    return str(obj)


def canonical_json(obj: Any) -> str:
    """返回规范化 JSON 字符串（键排序、浮点量化、中文不转义）。"""
    return json.dumps(_normalize(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(obj: Any, length: int = 16) -> str:
    """对任意对象取 SHA-256 摘要前缀，用于 trace 的输入/输出指纹。"""
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def now_iso() -> str:
    """本地时区 ISO 时间戳（秒级），用于 trace 与版本快照。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ge(a: float, b: float, tol: float = RATIO_TOL) -> bool:
    """浮点意义上的 a >= b。"""
    return a + tol >= b


def le(a: float, b: float, tol: float = RATIO_TOL) -> bool:
    """浮点意义上的 a <= b。"""
    return a <= b + tol


def gt(a: float, b: float, tol: float = RATIO_TOL) -> bool:
    """浮点意义上的 a > b。"""
    return a > b + tol


def clamp(value: float, low: float, high: float) -> float:
    """把 value 限制在 [low, high]。"""
    return max(low, min(high, value))


def pct(value: float, digits: int = 2) -> str:
    """把比率格式化为百分比字符串（0.035 -> 3.50%）。"""
    return f"{value * 100:.{digits}f}%"
