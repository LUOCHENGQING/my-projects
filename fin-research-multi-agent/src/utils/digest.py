"""摘要（digest）工具：把任意输入 / 输出压成可比较、可审计的短指纹。

层次与职责：
    utils 层（零业务依赖）。被 src/tracing.py 与 src/llm/client.py 用于给
    prompt / 响应落指纹，被 src/orchestrator.py 用于派生 run_id 之类的稳定标识。

trace 里不直接落全量数据，而是落 digest + 长度，既保护隐私又便于比对
「同一次运行是否走了一样的路径」。

对外关键函数：digest（字符串指纹）、digest_obj（指纹 + 规模描述）、
short_id（由 seed 派生确定性 ID）；内部 _stable_json 负责把任意对象规范化成字符串。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# 超长内容的判定阈值（字符数）。注：实际实现为——它只参与 truncated 布尔标记，
# 本模块不会真的截断字符串，raw 始终保留完整内容参与哈希计算。
_MAX_REPR = 4000


def _stable_json(obj: Any) -> str:
    """尽量把对象转成稳定的 JSON 字符串；失败则退回 repr。

    参数：
        obj：任意 Python 对象（普通标量、dict、dataclass、numpy 标量等）。
    返回值：
        str。sort_keys=True 保证同一映射的不同插入顺序得到同一串；
        ensure_ascii=False 保证中文按原样入哈希；default=str 兜住不可序列化对象。
    副作用 / 异常：
        无副作用；内部 try 兜底，不向外抛异常（最坏情况返回 repr(obj)）。
    """
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        # 兜底路径：repr 可能不稳定（含内存地址时每次不同），仅在 JSON 化失败时使用
        return repr(obj)


def digest(value: Any, length: int = 16) -> str:
    """返回 value 的 sha256 十六进制前缀。

    参数：
        value：字符串按原样参与哈希；其它类型先经 _stable_json 规范化。
        length：保留的十六进制字符数，默认 16（碰撞概率足够低且便于阅读）。
    返回值：
        str，长度为 length 的小写十六进制串。
    副作用 / 异常：
        无副作用；不抛异常（由 _stable_json 兜底）。
    """
    raw = value if isinstance(value, str) else _stable_json(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def digest_obj(value: Any, length: int = 16) -> dict:
    """返回带长度信息的 digest 描述，便于在 trace 里快速判断规模。

    参数：
        value：任意对象，或已经是字符串的内容。
        length：传给 digest 的十六进制前缀长度。
    返回值：
        dict：{"digest": str, "chars": int 规范化后的字符数,
               "truncated": bool 是否超过 _MAX_REPR}。
    副作用 / 异常：
        无副作用；不抛异常。注：truncated 仅表示"内容偏长"，
        本函数不会裁剪原始内容。
    """
    raw = value if isinstance(value, str) else _stable_json(value)
    return {
        "digest": digest(raw, length),
        "chars": len(raw),
        "truncated": len(raw) > _MAX_REPR,
    }


def short_id(prefix: str, seed: str, length: int = 10) -> str:
    """生成确定性短 ID（同 seed 必得同 ID），用于 run_id / source_id 派生。

    参数：
        prefix：ID 前缀，形如 "run" / "src"，不含分隔符。
        seed：参与哈希的种子串（同一 seed 必得同一结果，跨进程稳定）。
        length：哈希截断长度，默认 10。
    返回值：
        str，形如 f"{prefix}-{hexdigest[:length]}"。
    副作用 / 异常：
        无副作用；不抛异常（seed 必须能 encode 为 utf-8）。
    """
    return f"{prefix}-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:length]}"
