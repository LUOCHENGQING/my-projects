"""多维元数据与元数据过滤表达式：为每个块贴上可做「召回前过滤」的标签。

本模块在 RAG 全链路中的位置
---------------------------
    ingest 解析出文档级元数据（机构/产品/行业/年份/版本/密级/文号……）
        -> **本模块：摊平成块级扁平元数据 + 提供过滤表达式求值器**
            -> chunking.parent_child 把元数据挂到每个 ParentChunk / ChildChunk 上
                -> index（元数据随记录一起入库；vector_store 用本模块做标量过滤与删除）
                    -> retrieve.hybrid / pipeline（**打分前**先按表达式收窄候选集）
                        -> engine（答案里回显过滤条件，让用户知道证据被怎么筛过）

对外关键名字：
    extract_metadata()  文档 + 章节 -> 块级扁平元数据（本模块的输出格式就是索引库的字段格式）
    MetadataFilter      Milvus 风格过滤表达式（= / != / > / >= / < / <= / in / not in / contains / AND / OR / NOT / 括号）
    build_filter()      `MetadataFilter.parse()` 的便捷包装
    filter_chunks()     对任意带 `.meta` 的对象列表做过滤
    FilterError         表达式非法时抛出的异常（继承 ValueError）
    META_FIELDS         参与过滤的规范字段清单

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

副作用/异常总览：本模块纯内存、无 IO、无网络；`FilterError`（ValueError 子类）
是唯一的业务异常出口。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..ingest.loader import RawSection, SourceDocument

__all__ = ["extract_metadata", "MetadataFilter", "FilterError", "META_FIELDS"]

# 参与过滤的规范字段（其余 x_ 前缀字段原样保留但只在 contains 里可用）
#
# 字段语义速查（与 extract_metadata / ingest.loader 的产出保持一致）：
#   doc_id / source_id      文档内部 ID / 原始文件级 ID，定位与去重
#   title / doc_type        文档标题 / 文档类型（监管政策、内部制度、风险案例……）
#   institution             发文或归属机构，最常用的过滤维度
#   product / product_code  产品名称 / 产品代码（如 R2、C3 这类风险等级与产品编码）
#   industry                行业标签
#   effective_date / year   生效日期 / 生效年份——「只看 2024 年之后」就靠这两个
#   version                 制度版本，用于剔除被新版覆盖的旧条款
#   confidentiality         密级，权限隔离与合规展示的分界线
#   region                  适用地区
#   section_title / section_order  章节标题与序号，引用展示与范围限定
#   block_kind              段落 (paragraph) 还是表格 (table)，重排与展示据此分流
#   fmt                     原始格式（pdf / docx / md……），排障时用
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

# 只有「整串都是数字」的字符串才转数值，避免把 "2024年"、"R2" 这类混合串误判成数字
_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


class FilterError(ValueError):
    """过滤表达式非法。刻意抛异常而不是静默返回 True——
    静默放行会让「本该被版本过滤掉的旧条款」混进答案，属于合规事故。

    继承自 `ValueError`，被抛出点包括 `_tokenize`（无法解析的字符）、`_Parser` 的
    各文法方法（括号未闭合、缺少连接符、列表未闭合等）、`_eval` / `_compare`（未知节点或运算符）。
    """


def extract_metadata(
    doc: SourceDocument,
    section: Optional[RawSection] = None,
    block_kind: str = "paragraph",
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """把文档 + 章节信息摊平成块级元数据。

    这是元数据进入检索链路的唯一入口，被 `parent_child.split_document` 对**每个**
    父块/子块各调用一次（而不是复用父块那份），保证块级过滤字段齐全。

    参数：
        doc         已解析文档 `SourceDocument`；取其 `metadata_dict()`（规范字段）与
                    `meta`（原始字段，含 x_ 扩展字段）
        section     所属章节 `RawSection`；为空时**不写** section_title / section_order
                    （例如文档级摘要块），此时这两个键会缺省而不是空串
        block_kind  块类型标记，取值约定为 "paragraph" / "table"（默认 "paragraph"）
        extra       额外的元数据覆盖项，最后 update 进去，优先级最高

    返回：
        Dict[str, object]：扁平字典（无嵌套，可直接作为向量库标量字段）。
        合并顺序为「文档级 metadata_dict() -> 章节字段 -> block_kind -> 文档原始字段
        setdefault -> extra update」，因此越靠后的来源越高优先级，但原始字段
        不会覆盖规范字段（用 setdefault 而非 update）。

    注：实际实现为 —— 本函数**只搬运不做推断**：`META_FIELDS` 里机构/产品/行业/年份/版本/
        密级等字段是否存在，完全取决于 `doc.metadata_dict()` 与 `doc.meta` 是否给出了它们。
        字段缺失时该键**不会出现**（不是空串），此时过滤表达式里写 `year >= 2024`
        会把这条记录判为不匹配而剔除（`_compare()` 对 None 返回 False）。

    副作用/异常：
        无副作用（不改动 `doc`）；不抛异常。缺失来源只表现为键不存在。
    """
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
# 词法规则：每一种语法成分一个具名组（lparen/rparen/…/word），
# 靠 lastgroup 反查 token 类型，因此**组的顺序即优先级**——
# 必须让 >= <= != 这类双字符运算符排在单字符 < > = 前面，否则会被拆成两个 token。
# word 允许中文、字母数字、下划线以及 . 和 -，以覆盖 `WY2024-01`、`doc_type` 这类取值。
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

# 比较运算符的规范顺序（长符号在前，仅供阅读与校验时参照用）
# 注：实际实现为 —— `_compare()` 内部按 op 字符串逐个分支判断，
# 该常量目前未被任何函数读取，保留为运算符口径的单一事实来源。
_COMPARE_OPS = (">=", "<=", "!=", "==", "=", ">", "<")


@dataclass
class _Token:
    """词法单元（内部类型）。

    参数/属性：
        kind    token 类型，取自 `_TOKEN_RE` 的具名组名：
                lparen / rparen / lbracket / rbracket / comma / op / str / num / word
        value   token 文本；字符串字面量**保留引号**（由 `parse_value()` 去掉）
    返回/副作用：无；纯数据结构。
    """

    kind: str
    value: str


def _tokenize(expr: str) -> List[_Token]:
    """把过滤表达式切成 `_Token` 序列（词法分析）。

    参数：
        expr  过滤表达式原文，例如 `year >= 2024 AND doc_type = "监管政策"`；
              前后空白与任意位置空白都会被跳过
    返回：
        List[_Token]：按出现顺序排列的 token 列表；空表达式返回空列表。
    副作用/异常：
        无副作用。遇到无法匹配的字符（如 `@`、未闭合的引号）抛
        `FilterError`，并在消息里带上字符与位置 pos，便于用户改表达式。
    """
    tokens: List[_Token] = []
    pos = 0
    while pos < len(expr):
        if expr[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(expr, pos)
        # m.end() == m.start() 说明正则匹配了空串（只会匹配空白），必须视为失败，
        # 否则 while 循环不前进会死循环
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

    关键属性：
        expr  表达式原文（用于回显、写进缓存 key、日志留痕）；空串表示「无过滤」
        tree  编译后的语法树（dict 形式）；为 None 表示**无过滤条件**，此时
              `matches()` 恒为 True、`is_empty` 为 True

    对外方法：`parse()` 编译、`matches()` / `__call__()` 求值、`to_dict()` 导出。
    实例本身无状态可变的缓存，可安全跨线程复用。
    """

    def __init__(self, expr: str = "", tree: Optional[Dict[str, Any]] = None, raw: str = "") -> None:
        """直接构造编译结果（通常不手写，请用 `MetadataFilter.parse()`）。

        参数：
            expr  表达式原文；为空时回退取 `raw`
            tree  已编译的语法树；传 None 表示无过滤条件
            raw   兼容旧调用方的原文入参别名（仅当 expr 为空时生效）
        返回：无（构造函数）。
        副作用/异常：无；不做任何校验，`tree` 结构非法会在求值阶段才暴露。
        """
        self.expr: str = expr or raw
        self.tree: Optional[Dict[str, Any]] = tree

    # ---- 构造 ----
    @classmethod
    def parse(cls, expr: Optional[str]) -> "MetadataFilter":
        """把表达式字符串编译成 `MetadataFilter`（词法 + 递归下降语法分析）。

        参数：
            expr  过滤表达式；None 或纯空白 -> 返回「无过滤」实例（tree=None，matches 恒真），
                  这是刻意设计：调用方不必到处判空
        返回：
            MetadataFilter：`expr` 为 strip 后的原文（空白表达式存成空串），`tree` 为语法树
        副作用/异常：
            无副作用。表达式非法时抛 `FilterError`（含中文提示，如「括号未闭合」
            「疑似缺少连接符」「表达式尾部有多余内容」），**绝不静默放行**。
        """
        text = (expr or "").strip()
        if not text:
            return cls(expr="", tree=None)
        tokens = _tokenize(text)
        parser = _Parser(tokens)
        tree = parser.parse()
        return cls(expr=text, tree=tree)

    @property
    def is_empty(self) -> bool:
        """是否无过滤条件（`tree is None`）。

        参数：无。
        返回：True 表示不做任何过滤。
        副作用/异常：无。
        用途：`vector_store._candidate_indexes()` 与 `hybrid._mask()` 用它走「全量候选」快路径，
              避免为无过滤场景白算一遍 mask。
        """
        return self.tree is None

    # ---- 求值 ----
    def matches(self, meta: Dict[str, object]) -> bool:
        """判断一条元数据是否满足本过滤条件。

        参数：
            meta  块级扁平元数据（通常来自 `ChildChunk.meta`）
        返回：
            bool：无过滤条件时恒为 True；否则按语法树求值。
            字段缺失（`meta.get()` 为 None）时比较类节点**返回 False**——
            即「过滤条件写了某字段，但该块没有这个字段」会被剔除，宁可少召回不可错召回。
        副作用/异常：
            无副作用；语法树出现未知节点 kind 时抛 `FilterError`。
        """
        if self.tree is None:
            return True
        return _eval(self.tree, meta)

    def __call__(self, meta: Dict[str, object]) -> bool:
        """让过滤器实例可直接当谓词用，例如 `filter(cond, chunks)`。

        参数：meta 块级元数据。
        返回：同 `matches()`。
        副作用/异常：同 `matches()`。
        """
        return self.matches(meta)

    def to_dict(self) -> Dict[str, object]:
        """导出过滤条件摘要（写进检索计划/接口响应，让用户看到「证据被怎么筛过」）。

        参数：无。
        返回：`{"expr": 原文, "empty": 是否无过滤}`。
        副作用/异常：无。
        """
        return {"expr": self.expr, "empty": self.is_empty}


def _eval(node: Dict[str, Any], meta: Dict[str, object]) -> bool:
    """递归求值语法树（内部函数，被 `MetadataFilter.matches()` 调用）。

    参数：
        node  语法树节点，`node["kind"]` 取 and / or / not / cmp / in / contains，结构如下：
              and / or   -> {"left": 子节点, "right": 子节点}
              not        -> {"node": 子节点}
              cmp        -> {"field": 字段名, "op": 运算符, "value": 取值}
              in         -> {"field": 字段名, "values": [取值, ...]}（`not in` 由外层 not 包裹）
              contains   -> {"field": 字段名, "value": 取值}（子串匹配，大小写敏感）
    返回：
        bool：节点是否匹配。字段缺失（`meta.get()` 为 None）时 cmp 走 `_compare()`
        返回 False、contains 显式返回 False。
    副作用/异常：
        无副作用；`kind` 非法时抛 `FilterError`（说明语法树被绕过解析器手工构造了）。
    """
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
    """把字符串形式的数字转成数字，便于 year >= 2024 这类比较。

    参数：value 任意对象（常见是元数据里以字符串形式保存的年份/金额）。
    返回：可整串解析为数字的字符串转成 int（不含小数点）或 float（含小数点）；
          其余类型原样返回，**不抛异常**。
    副作用/异常：无。
    """
    if isinstance(value, str):
        text = value.strip()
        if _NUM_RE.match(text):
            return float(text) if "." in text else int(text)
    return value


def _compare(got: Any, op: str, want: Any) -> bool:
    """按运算符比较单个字段值（内部函数，`_eval` 与 `in` 列表匹配都走它）。

    参数：
        got   块元数据里的实际取值；为 None 时**一律返回 False**（字段缺失即不匹配）
        op    运算符，取值 = / == / != / > / >= / < / <=
        want  表达式里写的目标取值
    返回：
        bool。相等判断：任一侧是数值就先 `_coerce()` 再比，否则按 `str()` 比
        （所以 `2024` 与 `"2024"` 能匹配）；`!=` 实现为相等判断取反。
        大小比较：先 `_coerce()`；类型不可比时**退化为字符串比较**而不是抛错。
    副作用/异常：
        无副作用。`op` 不在支持列表时抛 `FilterError`。
    """
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
        """初始化解析器。

        参数：tokens 由 `_tokenize()` 产出的词法序列（内部会复制成 list）。
        返回：无（构造函数）。
        副作用/异常：无；不校验 token 是否合法，语法错误在 parse 阶段报出。
        """
        self.tokens = list(tokens)
        self.pos = 0

    # ---- 基础 ----
    def peek(self) -> Optional[_Token]:
        """查看当前 token 但**不消费**（前瞻一个符号）。

        参数：无。
        返回：当前 `_Token`；已到末尾时返回 None。
        副作用/异常：无（不改变 self.pos）。
        """
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> _Token:
        """消费并返回当前 token，游标前移一格。

        参数：无。
        返回：被消费的 `_Token`。
        副作用/异常：修改 `self.pos`；已到末尾时抛 `FilterError("表达式意外结束")`。
        """
        tok = self.peek()
        if tok is None:
            raise FilterError("表达式意外结束")
        self.pos += 1
        return tok

    def expect_word(self, word: str) -> None:
        """断言下一个 token 是指定关键字（大小写不敏感），主要用于 `not in`。

        参数：word 期望的关键字，如 "in"。
        返回：None。
        副作用/异常：消费一个 token；不匹配时抛 `FilterError`。
        """
        tok = self.next()
        if tok.kind != "word" or tok.value.lower() != word:
            raise FilterError(f"期望 {word}，实际得到 {tok.value!r}")

    # ---- 文法 ----
    def parse(self) -> Dict[str, Any]:
        """解析整个表达式并校验没有剩余内容（对外主入口）。

        参数：无（使用构造时传入的 tokens）。
        返回：语法树根节点（dict）。
        副作用/异常：消费 token 流；尾部有多余内容时抛 `FilterError`
                    （防止 `a = 1 b = 2` 这类漏写连接符的表达式被静默截断）。
        """
        tree = self.parse_or()
        if self.peek() is not None:
            raise FilterError(f"表达式尾部有多余内容：{self.peek().value!r}")
        return tree

    def parse_or(self) -> Dict[str, Any]:
        """文法层：or（优先级最低，左结合）。

        参数：无。
        返回：`{"kind": "or", ...}` 或更低层的节点。
        副作用/异常：消费 token；遇到 and / not / in / contains 等关键字却不在
                    合法位置时抛 `FilterError`（典型是漏写 or/and）。
        """
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
        """文法层：and（优先级高于 or，左结合）。

        参数：无。
        返回：`{"kind": "and", ...}` 或 not/primary 节点。
        副作用/异常：消费 token；不在此层抛错（错误交由更底层给出更精确的提示）。
        """
        node = self.parse_not()
        while True:
            tok = self.peek()
            if tok and tok.kind == "word" and tok.value.lower() == "and":
                self.next()
                node = {"kind": "and", "left": node, "right": self.parse_not()}
                continue
            return node

    def parse_not(self) -> Dict[str, Any]:
        """文法层：not（右结合，可连续叠加）。

        参数：无。
        返回：`{"kind": "not", "node": ...}`，或没有 not 时的 primary 节点。
        副作用/异常：消费 token；下游异常按原样抛出。
        """
        tok = self.peek()
        if tok and tok.kind == "word" and tok.value.lower() == "not":
            self.next()
            return {"kind": "not", "node": self.parse_not()}
        return self.parse_primary()

    def parse_primary(self) -> Dict[str, Any]:
        """文法层：括号分组或单个条件。

        参数：无。
        返回：括号内的 or 子树，或 `parse_condition()` 的条件节点。
        副作用/异常：
            消费 token；表达式为空、或左括号未用 `)` 闭合时抛 `FilterError`。
            注：括号只影响结合，不会在树里留下额外节点。
        """
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
        """解析一个原子条件：`字段 op 值` / `字段 in [..]` / `字段 not in [..]` / `字段 contains 值`。

        参数：无。
        返回：
            `{"kind": "cmp", ...}`、`{"kind": "in", ...}`、
            `{"kind": "contains", ...}`，或 `not in` 时外层再包一层
            `{"kind": "not", "node": {"kind": "in", ...}}`。
        副作用/异常：
            消费 token；字段名位置不是 word、缺少运算符、`not` 后不是 `in`
            （即仅支持 `not in`，不支持 `not contains`）时抛 `FilterError`。
        """
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
        """解析 `in` 后面的方括号列表（如 `["示例银行", "示例证券"]`）。

        参数：field_name 只用于拼错误提示，帮助用户定位是哪个字段的列表写坏了。
        返回：取值列表 `List[Any]`；`[]` 是合法写法（结果恒不匹配）。
        副作用/异常：
            消费 token；开头不是 `[`、列表未闭合（缺 `]`）时抛 `FilterError`。
            逗号被跳过，因此末尾多写逗号或写成 `[a,,b]` 都能容忍。
        """
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
        """解析一个取值字面量。

        参数：无。
        返回：
            带引号的字符串 -> 去掉首尾引号的 `str`；
            数字 -> `_coerce()` 后的 int / float；
            裸词 true/false（不分大小写）-> `bool`；
            裸词 null/none -> `None`；
            其余裸词按字符串处理（兼容 `doc_type = 监管政策` 不加引号的写法）。
        副作用/异常：消费 token；token 类型不是 str/num/word 时抛 `FilterError`。
        """
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
    """便捷函数：`build_filter('year >= 2024')`。

    参数：expr 过滤表达式；None/空白表示无过滤。
    返回：`MetadataFilter`（等价于 `MetadataFilter.parse(expr)`）。
    副作用/异常：无副作用；表达式非法时抛 `FilterError`。
    """
    return MetadataFilter.parse(expr)


def filter_chunks(items: Iterable[Any], expr: Optional[str], key=lambda item: item.meta) -> List[Any]:
    """按表达式过滤任意带 `.meta` 的对象。

    参数：
        items  待过滤对象（本项目里通常是 `ChildChunk` / `ParentChunk` 列表，也支持 dict 片段等）
        expr   过滤表达式；None/空白 -> 原样返回全部元素（拷贝成新 list）
        key    取元数据的取值函数，默认 `lambda item: item.meta`；
               传入 dict 列表时可改成 `lambda item: item`
    返回：
        List[Any]：命中的元素，**保持原顺序**；无过滤时返回 `list(items)` 的新列表。
    副作用/异常：
        无副作用（不修改元素，惰性入参会被完整消费）。
        表达式非法时抛 `FilterError`——**在过滤前就抛**，不会返回被悄悄放大的结果集。
        注：实际实现为 —— `src/retrieve/hybrid.py` 走的是 `MetadataFilter.parse()` +
        `matches()` 与向量库 expr 参数，本函数主要供离线评测与单测使用。
    """
    cond = MetadataFilter.parse(expr)
    if cond.is_empty:
        return list(items)
    return [item for item in items if cond.matches(key(item))]
