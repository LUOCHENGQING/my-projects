"""通用工具：规范化序列化、摘要（digest）、时间戳、浮点比较。

所处层次
--------
本模块位于 **L0 公共基础层**：它不 import 项目内任何其它业务模块，而 `versioning`
（版本链快照哈希）、`observability`（trace 输入/输出指纹）、`constraints` 与
`optimizer`（浮点比较与夹取）、`narrative` 与 `mock_brain`（百分比格式化）等上层
模块都直接依赖它。因此这里任何一处行为变化，都会同时影响 trace 指纹、版本链哈希
与约束判定的结果。

解决什么问题
------------
1. `canonical_json` 对浮点做定点量化后再序列化，保证同一份数值在不同机器、
   不同调用顺序下产生**完全一致的摘要**（trace 与版本链的哈希都依赖它）。
2. 所有比较都带显式容差 `EPS`，避免浮点误差把可行组合误判为越界。
   注：实际实现中没有名为 `EPS` 的常量，容差由两个具名常量提供——
   比率比较用 `RATIO_TOL`、权重比较用 `WEIGHT_TOL`，两者取值均为 1e-9。

对外暴露的关键对象
------------------
- 常量：`WEIGHT_TOL`、`RATIO_TOL`、`QUANTIZE`
- 序列化与摘要：`canonical_json`、`digest`
- 时间戳：`now_iso`
- 数值工具：`ge`、`le`、`gt`、`clamp`、`pct`
- 私有函数：`_normalize`（仅由本模块的 `canonical_json`/`digest` 调用）

主要输入输出
------------
输入为任意 Python 对象（含 dataclass、Pydantic 模型、Enum、set 等）；输出为 str
（JSON 文本 / 十六进制摘要 / ISO 时间戳 / 百分比文本）、bool 或 float。
本模块不做文件与网络 I/O，除 `now_iso` 读取系统时钟外均为纯函数。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any

# 权重类比较容差（1e-9 足以覆盖 16 位双精度累加误差）
# 被 constraints（集中度等约束判定）与 optimizer（权重归零阈值）引用
WEIGHT_TOL = 1e-9
# 数值比率类比较容差：ge / le / gt 的默认 tol
RATIO_TOL = 1e-9
# 浮点量化精度：_normalize 以 round(obj, QUANTIZE) 消除累加噪声，
# 位数与 versioning.diff_versions 中 round(delta, 12) 保持一致
QUANTIZE = 12


def _normalize(obj: Any) -> Any:
    """把任意对象递归转换为可稳定序列化的结构。

    参数:
        obj: 任意 Python 对象。识别 None / bool / str / int / float / Enum / dict /
            list / tuple / set / frozenset / dataclass、带 `model_dump()` 的
            Pydantic 模型、带 `to_dict()` 的对象。

    返回:
        仅由 dict（键统一为 str）/ list / bool / int / float / str / None 构成的
        结构：dict 键已按 str 排序，set 已转为确定性顺序的 list。

    说明:
        该函数的转换规则就是"摘要规范化"的定义本身，任何调整都会让既有 trace 与
        版本链的哈希失效，因此视为不可随意变更的契约。未识别的类型回退为
        `str(obj)`（不抛异常）。
    """
    if obj is None or isinstance(obj, (bool, str, int)):
        # 注意：bool 是 int 的子类、但不是 float。若不在这一行短路，True/False
        # 会一路落到函数末尾的 str(obj)，变成 "True"/"False" 两个字符串，摘要
        # 语义随之改变，故该分支是摘要正确性的关键。
        return obj
    if isinstance(obj, float):
        # 量化后再转 float，消除 0.30000000000000004 这类噪声；
        # 整数值统一折叠为 int（1.0 与 1 产生同一份摘要，提升跨模块摘要稳定性）
        value = round(obj, QUANTIZE)
        if value == 0:
            # 把 0.0 / -0.0 / 量化后为 0 的极小值统一折叠为 int 0，避免符号零
            # 或残差噪声产生不同摘要
            return 0
        if float(value).is_integer() and abs(value) < 1e15:
            # 整数值折叠为 int（1.0 与 1 同摘要）；1e15 上界用于避开超出双精度
            # 精确整数表示范围时的错误折叠
            return int(value)
        return value
    if isinstance(obj, Enum):
        # 取枚举的 value，使 Enum 成员与其原始字面量得到同一份摘要
        return obj.value
    if isinstance(obj, dict):
        # 先按 str(key) 排序再递归：canonical_json 的 sort_keys 只作用于序列化阶段，
        # 这里排序能保证嵌套层级同样稳定
        return {str(k): _normalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = [_normalize(v) for v in obj]
        if isinstance(obj, (set, frozenset)):
            # set/frozenset 无固有顺序，改用元素 JSON 文本作为排序键，保证可复现
            items = sorted(items, key=lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True))
        return items
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # 第二个条件排除 "dataclass 类对象本身"（类也满足 is_dataclass）
        return _normalize(dataclasses.asdict(obj))
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        # Pydantic 模型的序列化入口
        return _normalize(model_dump())
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        # 鸭子类型兜底：自带 to_dict() 的对象（如引擎状态/运行摘要对象）
        return _normalize(to_dict())
    return str(obj)


def canonical_json(obj: Any) -> str:
    """返回规范化 JSON 字符串（键排序、浮点量化、中文不转义）。

    参数:
        obj: 任意可被 `_normalize` 处理的对象。

    返回:
        紧凑格式 JSON 文本：`separators=(",", ":")` 去掉多余空白，键已排序，
        中文按原字符输出（`ensure_ascii=False`）。逻辑内容相同的 dict，无论键的
        插入顺序如何，都得到同一字符串。

    说明:
        这是 trace 指纹与版本链哈希的公共入口，输出的确定性直接决定跨机器比对
        是否成立。
    """
    return json.dumps(_normalize(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(obj: Any, length: int = 16) -> str:
    """对任意对象取 SHA-256 摘要前缀，用于 trace 的输入/输出指纹。

    参数:
        obj: 任意对象，先经 `canonical_json` 规范化。
        length: 截取的十六进制字符数，默认 16（即 64 bit）；版本链快照传 32
            （见 `versioning.AdviceSnapshot.create` 的 `snapshot_hash`）。

    返回:
        小写十六进制摘要字符串，长度为 `length`（不超过 64）。

    说明:
        摘要不可逆，只用于一致性比对与防篡改自检（`AdviceSnapshot.verify`），
        不能还原原文。
    """
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def now_iso() -> str:
    """本地时区 ISO 时间戳（秒级），用于 trace 与版本快照。

    返回:
        `datetime.now().astimezone().isoformat(timespec="seconds")` 的结果，形如
        `2026-10-08T21:40:03+08:00`（带本地时区偏移，精确到秒）。

    说明:
        精度只到秒，同一秒内多次调用可能返回相同字符串；该值仅用于展示与排序，
        不参与任何摘要/哈希计算（trace 指纹与快照哈希都只覆盖载荷内容）。
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ge(a: float, b: float, tol: float = RATIO_TOL) -> bool:
    """浮点意义上的 a >= b。

    参数:
        a, b: 待比较的浮点数。
        tol: 容差，默认 `RATIO_TOL`（1e-9）。

    返回:
        `a + tol >= b`，即 b 最多比 a 大 tol 时仍算满足。

    副作用/异常:
        无；不修改入参。
    """
    return a + tol >= b


def le(a: float, b: float, tol: float = RATIO_TOL) -> bool:
    """浮点意义上的 a <= b。

    参数:
        a, b: 待比较的浮点数。
        tol: 容差，默认 `RATIO_TOL`（1e-9）。

    返回:
        `a <= b + tol`，即 a 最多比 b 大 tol 时仍算满足。

    说明:
        约束求解用它判断"是否超限"，避免浮点累加误差把可行组合误判为越界。
    """
    return a <= b + tol


def gt(a: float, b: float, tol: float = RATIO_TOL) -> bool:
    """浮点意义上的 a > b（严格大于，但仍留容差）。

    参数:
        a, b: 待比较的浮点数。
        tol: 容差，默认 `RATIO_TOL`（1e-9）。

    返回:
        `a > b + tol`；即仅当 a 明显大于 b（超出容差）时才为 True。

    说明:
        与 `ge` 的差别在于"相等"落在哪一侧：差值在容差内时 `ge` 为 True、`gt` 为 False。
    """
    return a > b + tol


def clamp(value: float, low: float, high: float) -> float:
    """把 value 限制在 [low, high]。

    参数:
        value: 待夹取的数值。
        low: 下界（含）。
        high: 上界（含）。

    返回:
        `max(low, min(high, value))`。

    边界条件:
        不做 `low <= high` 校验；若调用方传入 low > high，由于先取 min 再取 max，
        结果恒为 low。区间有效性由调用方保证（optimizer 用它做权重裁剪）。
    """
    return max(low, min(high, value))


def pct(value: float, digits: int = 2) -> str:
    """把比率格式化为百分比字符串（0.035 -> 3.50%）。

    参数:
        value: 比率值（0.035 表示 3.5%），不校验取值范围。
        digits: 保留的小数位数，默认 2。

    返回:
        `f"{value * 100:.{digits}f}%"` 形式的字符串，仅用于 CLI/建议书展示。
    """
    return f"{value * 100:.{digits}f}%"
