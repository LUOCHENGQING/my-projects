"""把任意对象转成「可序列化的纯 Python 对象」。

层次与职责：
    utils 层（零业务依赖）。在两个"数据出场"位置被调用：
        src/rag/retriever.py —— 检索结果进入 Agent 状态前净化；
        src/tools/registry.py —— 工具返回值写回上层状态前净化。

为什么需要：numpy 的标量（np.float64 / np.int64）是 Python float/int 的子类，
在普通运算里不会报错，但一旦进入 LangGraph 的 checkpointer（msgpack 序列化）
就会抛出 `Type is not msgpack serializable: numpy.float64`，
表现为「图跑到一半崩掉」。这类 bug 极难定位，因此在数据出场的地方统一做一次净化。

对外关键函数：to_plain（递归净化，幂等且不修改入参）。
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__ = ["to_plain"]


def to_plain(obj: Any) -> Any:
    """递归地把 numpy 标量 / 数组 / 元组转成原生 Python 类型。

    参数：
        obj：任意对象，常见为 dict / list / numpy 标量 / numpy 数组混排的结构。
    返回值：
        净化后的新对象：dict 的键统一转 str、list/tuple/set 统一转 list、
        numpy 标量取其 item()、numpy 数组取其 tolist()；无法识别的对象原样返回。
    副作用 / 异常：
        不修改入参（只构造新容器）；内部兜底不抛异常——
        对不认识的对象宁可原样返回，也不在上层状态里制造新的报错点。
        注：set 转 list 后元素顺序不保证稳定（集合本身无序）。
    """
    # 基本类型直接返回（bool 要在 int 之前判断，避免被当成 int）
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    # float 也重建一次：np.float64 等子类会被归一成真正的内置 float
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, dict):
        # 键必须转 str：msgpack 对非字符串键的处理与 JSON 不一致，统一成 str 最稳
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        # 统一成 list：tuple 与 set 进 checkpointer 后类型语义容易丢失
        return [to_plain(v) for v in obj]

    # numpy 标量：有 item() 方法的都是 numpy 家族
    # shape == () 是"零维标量"的判据，用来把标量与数组区分开
    item = getattr(obj, "item", None)
    if callable(item) and getattr(obj, "shape", None) == ():
        try:
            return to_plain(item())
        except Exception:  # noqa: BLE001
            pass

    # numpy 数组：转成 list
    # 走到这里说明不是零维：用 tolist() 再递归一次，兜住多维嵌套数组
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return to_plain(tolist())
        except Exception:  # noqa: BLE001
            pass

    # 兜底：dataclass 实例等自定义对象原样返回，交给上层各自的序列化逻辑处理
    return obj
