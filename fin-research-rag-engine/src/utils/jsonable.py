"""把任意对象转成「可序列化的纯 Python 对象」。

为什么需要：numpy 标量（np.float64 / np.int64）是 Python float/int 的子类，普通运算
不会报错，但一旦进入 JSON 序列化（轨迹落盘、FastAPI 响应、Redis 缓存）就会抛
`TypeError: Object of type float64 is not JSON serializable`。这类 bug 极难定位，
因此在数据出场的地方统一做一次净化。
"""

from __future__ import annotations

from typing import Any

__all__ = ["to_plain"]


def to_plain(obj: Any) -> Any:
    """递归地把 numpy 标量 / 数组 / 元组 / set 转成原生 Python 类型。"""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_plain(v) for v in obj]

    # numpy 标量：有 item() 且 shape 为 () 的都属于这一类
    item = getattr(obj, "item", None)
    if callable(item) and getattr(obj, "shape", None) == ():
        try:
            return to_plain(item())
        except Exception:  # noqa: BLE001
            pass

    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return to_plain(tolist())
        except Exception:  # noqa: BLE001
            pass

    return obj
