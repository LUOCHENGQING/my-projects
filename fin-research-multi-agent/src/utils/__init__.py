"""通用工具函数集合（utils 层）。

层次与职责：
    位于最底层的「通用工具层」，不依赖项目内其它包（只用标准库），被上层全部模块复用：
    src/rag（分词、切句）、src/tools（返回值净化）、src/orchestrator、src/tracing、
    src/llm/client（digest 落痕）、src/demo、src/replay、eval/run_eval（控制台编码）。
    本文件只做「再导出」，不含任何业务逻辑。

对外关键函数（统一从此处导出，便于 `from ..utils import xxx`）：
    ensure_utf8_console  非 TTY 时把 stdout/stderr 切到 UTF-8（实现见 .console）
    digest / digest_obj  文本或对象的 sha256 短指纹，用于 trace 比对（实现见 .digest）
    short_id             由 seed 派生确定性短 ID（实现见 .digest）
    to_plain             递归净化 numpy 标量/数组为原生 Python 类型（实现见 .jsonable）
    normalize_text       文本归一化（实现见 .text）
    tokenize             中英混合的确定性分词，BM25 与向量侧共用（实现见 .text）
    split_sentences      按中文句末标点切句，供子块切分使用（实现见 .text）
"""

from __future__ import annotations

# 逐个具名导入而非 `from .x import *`：导入即校验子模块可用性，也便于静态检查工具追踪引用
from .console import ensure_utf8_console
from .digest import digest, digest_obj, short_id
from .jsonable import to_plain
from .text import normalize_text, tokenize, split_sentences

# 对外契约清单：只影响 `from src.utils import *` 的可见符号；
# 注：text.first_sentence 等子模块内公开函数刻意不在此列出（未被仓库其它模块使用）
__all__ = [
    "ensure_utf8_console",
    "digest",
    "digest_obj",
    "short_id",
    "to_plain",
    "normalize_text",
    "tokenize",
    "split_sentences",
]
