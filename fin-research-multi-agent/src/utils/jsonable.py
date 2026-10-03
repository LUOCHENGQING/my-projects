"""把任意对象转成「可序列化的纯 Python 对象」。

为什么需要：numpy 的标量（np.float64 / np.int64）是 Python float/int 的子类，
在普通运算里不会报错，但一旦进入 LangGraph 的 checkpointer（msgpack 序列化）
就会抛出 `Type is not msgpack serializable: numpy.float64`，
表现为「图跑到一半崩掉」。这类 bug 极难定位，因此在数据出场的地方统一做一次净化。
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__ = ["to_plain"]


def to_plain(obj: Any) -> Any:
    """递归地把 numpy 标量 / 数组 / 元组转成原生 Python 类型。"""
    # 基本类型直接返回（bool 要在 int 之前判断，避免被当成 int）
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_plain(v) for v in obj]

    # numpy 标量：有 item() 方法的都是 numpy 家族
    item = getattr(obj, "item", None)
    if callable(item) and getattr(obj, "shape", None) == ():
        try:
            return to_plain(item())
        except Exception:  # noqa: BLE001
            pass

    # numpy 数组：转成 list
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return to_plain(tolist())
        except Exception:  # noqa: BLE001
            pass

    return obj
