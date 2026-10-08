"""把任意对象转成「可序列化的纯 Python 对象」。

为什么需要：numpy 标量（np.float64 / np.int64）是 Python float/int 的子类，普通运算
不会报错，但一旦进入 JSON 序列化（轨迹落盘、FastAPI 响应、Redis 缓存）就会抛
`TypeError: Object of type float64 is not JSON serializable`。这类 bug 极难定位，
因此在数据出场的地方统一做一次净化。

注：实际实现里 `np.float64` **本身是 Python `float` 的子类**，会被 `to_plain()` 里的
float 分支提前处理；`item()` 那段逻辑兜的是其余 numpy 标量（如 `np.int64`、`np.bool_`）与零维数组。

在 RAG 全链路中的位置
--------------------
    生成 / 检索 → 【本模块：出场前净化】 → 轨迹落盘（`src/tracing.py`） / 接口响应 / 缓存写入

被谁调用：`src/engine.py`（`AnswerResult.to_dict()` 的返回值）、`src/tracing.py`、
`retrieve/pipeline.py`（`Evidence.to_dict()`）、`cache/redis_cache.py`（`set()` 序列化前）。

对外只有一个函数：`to_plain(obj)`，**永不抛异常**——序列化前的最后一道兜底不该成为新的故障点。
"""

from __future__ import annotations

from typing import Any

__all__ = ["to_plain"]


def to_plain(obj: Any) -> Any:
    """递归地把 numpy 标量 / 数组 / 元组 / set 转成原生 Python 类型。

    转换口径（按判断顺序）：
        None / str / bool / int  原样返回（bool 放在 int 之前虽无必要，但足以说明"布尔不转成 0/1"）；
        float                    统一 `float(obj)` 包一层，从而把 `np.float64` 变成纯 float；
        dict                     键统一 `str()`，值递归；
        list / tuple / set       统一变成 **list**（set 会因此失去"去重"的语义，但 JSON 本就不支持 set）；
        其余对象                 有 `item()` 且 `shape == ()` → 当 numpy 标量处理；
                                 有 `tolist()` → 当数组处理；
                                 两者都没有 → **原样返回**（不认识的类型不猜，交给调用方的序列化器报错）。

    参数：obj 任意对象（通常是各层 `to_dict()` 拼出来的嵌套 dict / list）。
    返回：Any —— 只含 dict / list / str / int / float / bool / None 的结构（未知类型可能原样透出）。
    副作用/异常：无副作用；`item()` / `tolist()` 的失败被静默吞掉并退回 `obj`，
                因此**本函数自身不会抛异常**（除递归深度超限外）。
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_plain(v) for v in obj]

    # numpy 标量：有 item() 且 shape 为 () 的都属于这一类
    # 先判 shape 再调 item()：普通对象也可能有 item 方法（如某些映射类型），
    # 少了 shape 判据会把它们误转成标量。
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
