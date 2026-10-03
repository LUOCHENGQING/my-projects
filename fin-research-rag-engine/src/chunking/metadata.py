"""多维元数据与元数据过滤表达式。

元数据在 RAG 里的定位是**召回前的过滤器**，不是装饰：
    「只看 2024 年之后的产品说明」
    「只看某家机构的尽调档案」
    「剔除已被新版制度覆盖的旧条款」
这三类需求如果靠「召回一大堆再让模型挑」，既浪费上下文窗口又不稳定；
正确的做法是在**打分之前**把候选集收窄。

因此本模块提供两件东西：

1. `extract_metadata()`：把文档级元数据 + 章节信息摊平成每个块都携带的扁平字典
   （扁平是为了能直接喂给向量库的过滤表达式，嵌套结构很多向量库并不支持）。
2. `MetadataFilter`：一个 Milvus 风格的过滤表达式求值器，支持
   `doc_type = "监管政策" AND year >= 2024`、`institution in ["示例银行"]`、
   `version contains "v2"` 以及与或非与括号。手写而不依赖向量库客户端，
   是为了让**过滤逻辑可单测**——过滤条件写错会静默丢证据，是最难查的一类 bug。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..ingest.loader import RawSection, SourceDocument

__all__ = ["extract_metadata", "MetadataFilter", "FilterError", "META_FIELDS"]

# 参与过滤的规范字段（其余 x_ 前缀字段原样保留但只在 contains 里可用）
META_FIELDS: Tuple[str, ...] = (
    "doc_id",
    "source_id",
    "title",
    "doc_type",
    "institution",
    "product",
    "product_code",
    "industry",
    "effective_date",
    "year",
    "version",
    "confidentiality",
    "region",
    "section_title",
    "section_order",
    "block_kind",
    "fmt",
)

_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


class FilterError(ValueError):
    """过滤表达式非法。刻意抛异常而不是静默返回 True——
    静默放行会让「本该被版本过滤掉的旧条款」混进答案，属于合规事故。"""


def extract_metadata(
    doc: SourceDocument,
    section: Optional[RawSection] = None,
    block_kind: str = "paragraph",
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """把文档 + 章节信息摊平成块级元数据。"""
    meta: Dict[str, object] = dict(doc.metadata_dict())
    if section is not None:
        meta["section_title"] = section.title
        meta["section_order"] = section.order
    meta["block_kind"] = block_kind
    # 把文档级原始字段里的 x_ 扩展字段也带上，保证自定义过滤条件同样可用
    for key, value in doc.meta.items():
        meta.setdefault(key, value)
    if extra:
        meta.update(extra)
    return meta


# ---------------------------------------------------------------------------
# 过滤表达式
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (?P<lparen>\()
      | (?P<rparen>\))
      | (?P<lbracket>\[)
      | (?P<rbracket>\])
      | (?P<comma>,)
      | (?P<op>>=|<=|!=|==|=|>|<)
      | (?P<str>"[^"]*"|'[^']*')
      | (?P<num>-?\d+(?:\.\d+)?)
      | (?P<word>[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff\.\-]*)
    )
    """,
    re.VERBOSE,
)

_COMPARE_OPS = (">=", "<=", "!=", "==", "=", ">", "<")


@dataclass
class _Token:
    kind: str
    value: str


def _tokenize(expr: str) -> List[_Token]:
    tokens: List[_Token] = []
    pos = 0
    while pos < len(expr):
        if expr[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(expr, pos)
        if not m or m.end() == m.start():
            raise FilterError(f"无法解析的字符：{expr[pos]!r}（位置 {pos}）")
        pos = m.end()
        kind = m.lastgroup or ""
        tokens.append(_Token(kind=kind, value=m.group(0).strip()))
    return tokens


class MetadataFilter:
    """编译后的元数据过滤条件。

    表达式示例：
        doc_type = "监管政策" AND year >= 2024
        institution in ["示例银行", "示例证券"] OR doc_type = "风险案例"
        (doc_type = "内部制度" OR doc_type = "监管政策") AND version != "v1"
    """

    def __init__(self, expr: str = "", tree: Optional[Dict[str, Any]] = None, raw: str = "") -> None:
        self.expr: str = expr or raw
        self.tree: Optional[Dict[str, Any]] = tree

    # ---- 构造 ----
    @classmethod
    def parse(cls, expr: Optional[str]) -> "MetadataFilter":
        text = (expr or "").strip()
        if not text:
            return cls(expr="", tree=None)
        tokens = _tokenize(text)
        parser = _Parser(tokens)
        tree = parser.parse()
        return cls(expr=text, tree=tree)

    @property
    def is_empty(self) -> bool:
        return self.tree is None

    # ---- 求值 ----
    def matches(self, meta: Dict[str, object]) -> bool:
        if self.tree is None:
            return True
        return _eval(self.tree, meta)

    def __call__(self, meta: Dict[str, object]) -> bool:
        return self.matches(meta)

    def to_dict(self) -> Dict[str, object]:
        return {"expr": self.expr, "empty": self.is_empty}


def _eval(node: Dict[str, Any], meta: Dict[str, object]) -> bool:
    kind = node["kind"]
    if kind == "and":
        return _eval(node["left"], meta) and _eval(node["right"], meta)
    if kind == "or":
        return _eval(node["left"], meta) or _eval(node["right"], meta)
    if kind == "not":
        return not _eval(node["node"], meta)
    if kind == "cmp":
        return _compare(meta.get(node["field"]), node["op"], node["value"])
    if kind == "in":
        got = meta.get(node["field"])
        return any(_compare(got, "=", v) for v in node["values"])
    if kind == "contains":
        got = meta.get(node["field"])
        if got is None:
            return False
        if node["value"] is None:
            return False
        return str(node["value"]) in str(got)
    raise FilterError(f"未知节点：{kind}")


def _coerce(value: Any) -> Any:
    """把字符串形式的数字转成数字，便于 year >= 2024 这类比较。"""
    if isinstance(value, str):
        text = value.strip()
        if _NUM_RE.match(text):
            return float(text) if "." in text else int(text)
    return value


def _compare(got: Any, op: str, want: Any) -> bool:
    if got is None:
        return False
    if op in ("=", "=="):
        if isinstance(got, (int, float)) or isinstance(want, (int, float)):
            return _coerce(got) == _coerce(want)
        return str(got) == str(want)
    if op == "!=":
        return not _compare(got, "=", want)
    g, w = _coerce(got), _coerce(want)
    try:
        if op == ">":
            return g > w
        if op == ">=":
            return g >= w
        if op == "<":
            return g < w
        if op == "<=":
            return g <= w
    except TypeError:
        # 类型不可比时退化为字符串比较，避免整个检索因一条脏元数据而崩
        return _compare(str(got), op, str(want))
    raise FilterError(f"未知运算符：{op}")


class _Parser:
    """递归下降解析器：or -> and -> not -> primary。手写而非 eval()，
    因为过滤表达式可能来自外部请求，用 eval 就是一个远程代码执行漏洞。"""

    def __init__(self, tokens: Sequence[_Token]) -> None:
        self.tokens = list(tokens)
        self.pos = 0

    # ---- 基础 ----
    def peek(self) -> Optional[_Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> _Token:
        tok = self.peek()
        if tok is None:
            raise FilterError("表达式意外结束")
        self.pos += 1
        return tok

    def expect_word(self, word: str) -> None:
        tok = self.next()
        if tok.kind != "word" or tok.value.lower() != word:
            raise FilterError(f"期望 {word}，实际得到 {tok.value!r}")

    # ---- 文法 ----
    def parse(self) -> Dict[str, Any]:
        tree = self.parse_or()
        if self.peek() is not None:
            raise FilterError(f"表达式尾部有多余内容：{self.peek().value!r}")
        return tree

    def parse_or(self) -> Dict[str, Any]:
        node = self.parse_and()
        while True:
            tok = self.peek()
            if tok and tok.kind == "word" and tok.value.lower() == "or":
                self.next()
                node = {"kind": "or", "left": node, "right": self.parse_and()}
                continue
            if tok and tok.kind == "word" and tok.value.lower() in ("and", "not", "contains", "in"):
                # 常见笔误：`a = 1 b = 2`（漏了 and），必须报错而不是静默只取前半段
                raise FilterError(f"疑似缺少连接符，遇到 {tok.value!r}")
            return node

    def parse_and(self) -> Dict[str, Any]:
        node = self.parse_not()
        while True:
            tok = self.peek()
            if tok and tok.kind == "word" and tok.value.lower() == "and":
                self.next()
                node = {"kind": "and", "left": node, "right": self.parse_not()}
                continue
            return node

    def parse_not(self) -> Dict[str, Any]:
        tok = self.peek()
        if tok and tok.kind == "word" and tok.value.lower() == "not":
            self.next()
            return {"kind": "not", "node": self.parse_not()}
        return self.parse_primary()

    def parse_primary(self) -> Dict[str, Any]:
        tok = self.peek()
        if tok is None:
            raise FilterError("表达式为空")
        if tok.kind == "lparen":
            self.next()
            node = self.parse_or()
            closing = self.next()
            if closing.kind != "rparen":
                raise FilterError("括号未闭合")
            return node
        return self.parse_condition()

    def parse_condition(self) -> Dict[str, Any]:
        field_tok = self.next()
        if field_tok.kind != "word":
            raise FilterError(f"期望字段名，得到 {field_tok.value!r}")
        field_name = field_tok.value

        tok = self.peek()
        if tok and tok.kind == "word" and tok.value.lower() in ("in", "contains", "not"):
            keyword = tok.value.lower()
            if keyword == "not":
                self.next()
                inner = self.next()
                if inner.kind != "word" or inner.value.lower() != "in":
                    raise FilterError("仅支持 `not in` 形式的否定")
                values = self.parse_list(field_name)
                return {"kind": "not", "node": {"kind": "in", "field": field_name, "values": values}}
            self.next()
            if keyword == "in":
                values = self.parse_list(field_name)
                return {"kind": "in", "field": field_name, "values": values}
            value = self.parse_value()
            return {"kind": "contains", "field": field_name, "value": value}

        op_tok = self.next()
        if op_tok.kind != "op":
            raise FilterError(f"期望比较运算符，得到 {op_tok.value!r}")
        value = self.parse_value()
        return {"kind": "cmp", "field": field_name, "op": op_tok.value, "value": value}

    def parse_list(self, field_name: str) -> List[Any]:
        opener = self.next()
        if opener.kind != "lbracket":
            raise FilterError(f"`in` 后面需要列表，字段 {field_name}")
        values: List[Any] = []
        while True:
            tok = self.peek()
            if tok is None:
                raise FilterError("列表未闭合")
            if tok.kind == "rbracket":
                self.next()
                return values
            if tok.kind == "comma":
                self.next()
                continue
            values.append(self.parse_value())

    def parse_value(self) -> Any:
        tok = self.next()
        if tok.kind == "str":
            return tok.value[1:-1]
        if tok.kind == "num":
            return _coerce(tok.value)
        if tok.kind == "word":
            low = tok.value.lower()
            if low in ("true", "false"):
                return low == "true"
            if low in ("null", "none"):
                return None
            # 未加引号的裸词按字符串处理，兼容 `doc_type = 监管政策` 这种写法
            return tok.value
        raise FilterError(f"不合法的取值：{tok.value!r}")


def build_filter(expr: Optional[str]) -> MetadataFilter:
    """便捷函数：`build_filter('year >= 2024')`。"""
    return MetadataFilter.parse(expr)


def filter_chunks(items: Iterable[Any], expr: Optional[str], key=lambda item: item.meta) -> List[Any]:
    """按表达式过滤任意带 `.meta` 的对象。"""
    cond = MetadataFilter.parse(expr)
    if cond.is_empty:
        return list(items)
    return [item for item in items if cond.matches(key(item))]
