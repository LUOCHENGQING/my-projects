"""摘要（digest）工具：把任意输入 / 输出压成可比较、可审计的短指纹。

trace 里不直接落全量数据，而是落 digest + 长度，既保护隐私又便于比对
「同一次运行是否走了一样的路径」。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

_MAX_REPR = 4000


def _stable_json(obj: Any) -> str:
    """尽量把对象转成稳定的 JSON 字符串；失败则退回 repr。"""
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return repr(obj)


def digest(value: Any, length: int = 16) -> str:
    """返回 value 的 sha256 十六进制前缀。"""
    raw = value if isinstance(value, str) else _stable_json(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def digest_obj(value: Any, length: int = 16) -> dict:
    """返回带长度信息的 digest 描述，便于在 trace 里快速判断规模。"""
    raw = value if isinstance(value, str) else _stable_json(value)
    return {
        "digest": digest(raw, length),
        "chars": len(raw),
        "truncated": len(raw) > _MAX_REPR,
    }


def short_id(prefix: str, seed: str, length: int = 10) -> str:
    """生成确定性短 ID（同 seed 必得同 ID），用于 run_id / source_id 派生。"""
    return f"{prefix}-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:length]}"
