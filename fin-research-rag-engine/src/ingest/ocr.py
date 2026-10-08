"""OCR 适配层。

在 RAG 全链路中的位置：摄取层（ingest）的**入口兜底**——扫描件先在这里变成文本，
再作为 `.txt` 交给 `loader` 解析（`fmt="text"`，会被质量评分按 OCR 来源轻罚）。
它是「六处同接口换实现」降级开关中**摄取层所体现的那一处**。

金融场景里大量历史资料是**扫描件存档**（监管批复、盖章合同、纸质制度签发稿），
这批资料不进检索就等于知识库缺了一半。但 OCR 引擎的安装体积与推理成本都不小，
所以这里做成**可插拔后端**：

    paddleocr  真实后端，装了 paddleocr 才启用（生产环境用）
    sidecar    旁挂文本后端：扫描件 `xxx.png` 旁边放同名 `xxx.txt`
               （= 已经人工校对过的转录稿），离线演示与 CI 用这个
    null       没装也没旁挂文本时返回空结果，并在结果里留下 warnings

这样做的好处是：**引擎代码里不存在「因为没装 OCR 所以跑不了」的分支**，
`extract_text()` 的返回结构在三种后端下完全一致，上层解析逻辑不需要任何特判。

对外关键对象
------------
    OCRResult                统一结果结构（text / engine / confidence / page_count / warnings）
    get_engine(prefer)       按名字取后端（带实例缓存）：`paddleocr` / `sidecar` / `null` / 其他=auto
    extract_text(path, prefer)  对扫描件或图片抽文本；非图片后缀直接返回空结果并说明原因

调用关系（实际实现）
--------------------
`loader.py` **并不调用**本模块：`ingest/__init__.py` 只是把它再导出，实际调用入口是
`tests/test_ingest.py`（以及未来的批次处理脚本）。也就是说，扫描件目前是在**离线**
被转成 `.txt` 后才进入 `loader` 的，运行期没有隐式的 OCR 调用。

选后端口径说明（实际实现）
--------------------------
- `prefer="auto"`：先尝试 paddleocr，`available()` 为假则**直接退到 sidecar**——
  不会再探测 sidecar 是否存在旁挂文本；旁挂文本缺失时由 `SidecarOCREngine.recognize()`
  返回空结果 + warning，**不会**继续退到 null。
- `IMAGE_SUFFIXES` 里含 `.pdf`：即 PDF 也会被送进 OCR 后端（`PaddleOCR` 按页识别），
  因此本模块只识别扫描件 PDF；文本型 PDF 走 `loader` 的文本格式更合适。
- 置信度是各后端自报的：paddleocr 取所有文本行 score 的算术平均（无行则 0.0），
  sidecar 固定 0.99，null 保持默认 0.0；`NullOCREngine` 的 `page_count` 固定为 1。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["OCRResult", "OCREngine", "PaddleOCREngine", "SidecarOCREngine", "NullOCREngine", "get_engine", "extract_text"]

# 允许走 OCR 的后缀白名单；含 `.pdf`（扫描件 PDF 按页识别，文本型 PDF 更适合走 loader）
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pdf")


@dataclass
class OCRResult:
    """一次识别的结果。三种后端返回同一结构，上层无需特判。

    关键属性：
        path        被识别的文件路径（字符串形式，原样回填）
        text        识别出的纯文本；后端不可用时为空串（**不抛异常**）
        engine      产出该结果的后端名：`paddleocr` / `sidecar` / `null`；
                    `extract_text()` 对非图片后缀会写 `none`
        confidence  置信度 0~1，各后端自报（paddleocr 为行 score 均值，sidecar 固定 0.99）
        page_count  页数（paddleocr 取返回页数，sidecar 数分页符 `\\f`，null 固定 1）
        warnings    降级/异常说明，供轨迹展示；**不参与判定**，是否可用看 `ok`
    """

    path: str
    text: str
    engine: str
    confidence: float = 0.0
    page_count: int = 1
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """是否真的拿到了文本（`text.strip()` 非空）；空结果一律为 False。"""
        return bool(self.text.strip())

    def to_dict(self) -> Dict[str, object]:
        """拍平成字典供 API/轨迹输出；`confidence` 保留 4 位小数，`chars` 为文本长度。"""
        return {
            "path": self.path,
            "engine": self.engine,
            "confidence": round(self.confidence, 4),
            "page_count": self.page_count,
            "chars": len(self.text),
            "warnings": list(self.warnings),
        }


class OCREngine:
    """OCR 后端接口。

    职责：定义「可用性探测 + 识别」两个方法，让上层只依赖统一契约。
    关键属性（子类覆写）：
        name  后端名，会写进 `OCRResult.engine`
    """

    name = "base"

    def available(self) -> bool:  # pragma: no cover - 接口默认实现
        """后端是否可用；接口默认返回 False（子类覆写）。"""
        return False

    def recognize(self, path: Path) -> OCRResult:  # pragma: no cover - 接口默认实现
        """识别一张图 / 一份扫描 PDF；接口默认抛 `NotImplementedError`（子类覆写）。

        参数：path — 待识别文件路径。
        返回：`OCRResult`（子类实现约定：不可用时不抛异常，而是返回空文本 + warnings）。
        """
        raise NotImplementedError


class PaddleOCREngine(OCREngine):
    """PaddleOCR 后端。中文与表格识别表现稳定，生产环境推荐。

    职责：包装 paddleocr，把其嵌套返回结构拍平成统一文本。
    关键属性：
        _engine  惰性创建的 `PaddleOCR` 实例（首次 recognize 才构造，避免 import 即加载模型）
        _error   最近一次 `available()` 探测失败的原因（形如 `ModuleNotFoundError: …`）
    """

    name = "paddleocr"

    def __init__(self) -> None:
        """只初始化占位属性，**不**在此处加载模型或 import paddleocr（保持构造廉价）。"""
        self._engine = None
        self._error: Optional[str] = None

    def available(self) -> bool:
        """探测 paddleocr 是否可 import。

        返回：可 import 为 True；失败为 False 并把异常摘要记到 `self._error`。
        副作用：不缓存失败结果，每次调用都会重试 import（运行期装了包即可生效）。
        """
        try:
            import paddleocr  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            self._error = f"{type(exc).__name__}: {exc}"
            return False
        return True

    def recognize(self, path: Path) -> OCRResult:
        """调用 PaddleOCR 识别一份图片 / 扫描 PDF。

        参数：path — 待识别文件路径（原样透传给 paddleocr，并用 `str()` 写进结果）。

        返回：`OCRResult`。文本为**识别行按顺序用换行拼接**；`confidence` 为所有行
        score 的算术平均（一行都没有则为 0.0）；`page_count` 取返回的页数（为 0 时记 1）。
        识别不到内容时返回空文本结果，**不抛异常**。

        副作用 / 异常：
            - 未安装 paddleocr 时直接返回空结果 + warning（不做任何识别）；
            - 首次调用会构造并缓存 `PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)`；
            - 单行返回结构与预期不符时静默 `continue` 跳过该行（兼容不同版本），
              因此**不会**因个别脏行中断整篇识别。
        """
        if not self.available():
            return OCRResult(path=str(path), text="", engine=self.name, warnings=[f"未安装 paddleocr（{self._error}）"])
        from paddleocr import PaddleOCR  # type: ignore

        if self._engine is None:
            # use_angle_cls 打开：扫描件常带倾斜，不矫正会整行识别错
            self._engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        raw = self._engine.ocr(str(path), cls=True)
        lines: List[str] = []
        scores: List[float] = []
        for page in raw or []:
            for item in page or []:
                try:
                    text, score = item[1][0], float(item[1][1])
                except Exception:  # noqa: BLE001 - 不同版本返回结构有差异
                    continue
                lines.append(str(text))
                scores.append(score)
        confidence = sum(scores) / len(scores) if scores else 0.0
        return OCRResult(
            path=str(path),
            text="\n".join(lines),
            engine=self.name,
            confidence=confidence,
            page_count=len(raw or []) or 1,
        )


class SidecarOCREngine(OCREngine):
    """旁挂文本后端：读取同名 `.txt` 作为识别结果。

    职责：用「人工校对过的转录稿」替代真实推理，让离线与 CI 结果完全确定。
    关键属性：`name = "sidecar"`（写进 `OCRResult.engine`，便于区分结果来源）。

    这不是「假 OCR」，而是把**人工校对环节显式化**：现实中扫描件识别完，
    业务方一定会人工核一遍，核完的稿子就是 sidecar。CI 与离线演示用它，
    可以保证测试结果完全确定、不随 OCR 引擎版本漂移。
    """

    name = "sidecar"

    def available(self) -> bool:
        """恒为 True：后端本身永远可用，缺旁挂文本属于「本次无结果」而非「后端不可用」。"""
        return True

    def recognize(self, path: Path) -> OCRResult:
        """读取 `path` 同名 `.txt`（`with_suffix` 换后缀）作为识别文本。

        参数：path — 扫描件/图片路径。
        返回：`OCRResult`，`confidence` 固定 0.99，`page_count = max(1, 分页符\\f 个数 + 1)`，
              `warnings` 固定说明「来自 sidecar」。

        副作用 / 异常：读取 txt 文件（UTF-8，读取失败异常向上抛）；
              旁挂文本不存在时返回空文本 + warning（该文件不参与检索），不抛异常。
        """
        sidecar = path.with_suffix(".txt")
        if not sidecar.exists():
            return OCRResult(
                path=str(path),
                text="",
                engine=self.name,
                warnings=[f"未找到旁挂文本 {sidecar.name}，该扫描件不参与检索"],
            )
        text = sidecar.read_text(encoding="utf-8")
        return OCRResult(
            path=str(path),
            text=text,
            engine=self.name,
            confidence=0.99,
            page_count=max(1, text.count("\f") + 1),
            warnings=["识别结果来自人工校对后的旁挂文本（sidecar）"],
        )


class NullOCREngine(OCREngine):
    """兜底后端：明确返回空结果并说明原因，绝不静默丢数据。

    用途：显式表达「这份扫描件这次没被识别」——结果里带 warnings，
    轨迹与治理报表能看见缺口，而不是悄悄从语料里消失。
    """

    name = "null"

    def available(self) -> bool:
        """恒为 True：作为链条的最后一级，它必须「可用」，只是产不出文本。"""
        return True

    def recognize(self, path: Path) -> OCRResult:
        """永远返回空文本结果，`warnings` 说明无可用后端；`page_count`/`confidence` 保持默认值。"""
        return OCRResult(
            path=str(path),
            text="",
            engine=self.name,
            warnings=["无可用 OCR 后端（未安装 paddleocr 且无旁挂文本）"],
        )


# 后端实例缓存：键是 `prefer.lower()`，避免每次取引擎都重新构造（PaddleOCR 构造很贵）
_ENGINE_CACHE: Dict[str, OCREngine] = {}


def get_engine(prefer: str = "auto") -> OCREngine:
    """选择 OCR 后端：auto 时优先 paddleocr，其次 sidecar，最后 null。

    参数：prefer — 后端名（大小写不敏感）：`paddleocr` / `sidecar` / `null`；
          其他取值（含默认 `auto`）走自动选择。

    返回：`OCREngine` 实例（同一 `prefer` 复用缓存实例）。

    自动选择口径（实际实现）：new 一个 `PaddleOCREngine()`，`available()` 为真就用它，
    否则**直接返回 `SidecarOCREngine()`**——不再探测旁挂文本是否存在、也不会返回
    `NullOCREngine`（null 只在显式 `prefer="null"` 时被选中）。
    副作用：写 `_ENGINE_CACHE`（进程内长期驻留）。
    """
    key = prefer.lower()
    if key in _ENGINE_CACHE:
        return _ENGINE_CACHE[key]

    engine: OCREngine
    if key == "paddleocr":
        engine = PaddleOCREngine()
    elif key == "sidecar":
        engine = SidecarOCREngine()
    elif key == "null":
        engine = NullOCREngine()
    else:
        # 未指定后端时的自动降级：装了就上真实引擎，没装就用旁挂校对文本
        paddle = PaddleOCREngine()
        engine = paddle if paddle.available() else SidecarOCREngine()
    _ENGINE_CACHE[key] = engine
    return engine


def extract_text(path: Path, prefer: str = "auto") -> OCRResult:
    """对扫描件 / 图片抽取文本；非图片后缀直接返回空结果并标注原因。

    参数：
        path   待抽取的文件路径（可传 str，内部过一遍 `Path`）
        prefer 后端名，默认 `auto`（语义见 `get_engine`）

    返回：`OCRResult`。后缀不在 `IMAGE_SUFFIXES` 时不调用任何后端，直接返回
          `engine="none"` + warning（形如 `非图片格式，跳过 OCR：.txt`）。

    副作用：可能触发 paddleocr 的模型加载与推理；结果由所选后端决定。
    异常：本函数自身不抛业务异常；后端内部的文件读取/推理异常会向上抛出。
    """
    target = Path(path)
    if target.suffix.lower() not in IMAGE_SUFFIXES:
        return OCRResult(
            path=str(target),
            text="",
            engine="none",
            warnings=[f"非图片格式，跳过 OCR：{target.suffix}"],
        )
    engine = get_engine(prefer)
    return engine.recognize(target)
