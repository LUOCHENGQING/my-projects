"""OCR 适配层。

金融场景里大量历史资料是**扫描件存档**（监管批复、盖章合同、纸质制度签发稿），
这批资料不进检索就等于知识库缺了一半。但 OCR 引擎的安装体积与推理成本都不小，
所以这里做成**可插拔后端**：

    paddleocr  真实后端，装了 paddleocr 才启用（生产环境用）
    sidecar    旁挂文本后端：扫描件 `xxx.png` 旁边放同名 `xxx.txt`
               （= 已经人工校对过的转录稿），离线演示与 CI 用这个
    null       没装也没旁挂文本时返回空结果，并在结果里留下 warnings

这样做的好处是：**引擎代码里不存在「因为没装 OCR 所以跑不了」的分支**，
`extract_text()` 的返回结构在三种后端下完全一致，上层解析逻辑不需要任何特判。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["OCRResult", "OCREngine", "PaddleOCREngine", "SidecarOCREngine", "NullOCREngine", "get_engine", "extract_text"]

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pdf")


@dataclass
class OCRResult:
    """一次识别的结果。三种后端返回同一结构，上层无需特判。"""

    path: str
    text: str
    engine: str
    confidence: float = 0.0
    page_count: int = 1
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "engine": self.engine,
            "confidence": round(self.confidence, 4),
            "page_count": self.page_count,
            "chars": len(self.text),
            "warnings": list(self.warnings),
        }


class OCREngine:
    """OCR 后端接口。"""

    name = "base"

    def available(self) -> bool:  # pragma: no cover - 接口默认实现
        return False

    def recognize(self, path: Path) -> OCRResult:  # pragma: no cover - 接口默认实现
        raise NotImplementedError


class PaddleOCREngine(OCREngine):
    """PaddleOCR 后端。中文与表格识别表现稳定，生产环境推荐。"""

    name = "paddleocr"

    def __init__(self) -> None:
        self._engine = None
        self._error: Optional[str] = None

    def available(self) -> bool:
        try:
            import paddleocr  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            self._error = f"{type(exc).__name__}: {exc}"
            return False
        return True

    def recognize(self, path: Path) -> OCRResult:
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

    这不是「假 OCR」，而是把**人工校对环节显式化**：现实中扫描件识别完，
    业务方一定会人工核一遍，核完的稿子就是 sidecar。CI 与离线演示用它，
    可以保证测试结果完全确定、不随 OCR 引擎版本漂移。
    """

    name = "sidecar"

    def available(self) -> bool:
        return True

    def recognize(self, path: Path) -> OCRResult:
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
    """兜底后端：明确返回空结果并说明原因，绝不静默丢数据。"""

    name = "null"

    def available(self) -> bool:
        return True

    def recognize(self, path: Path) -> OCRResult:
        return OCRResult(
            path=str(path),
            text="",
            engine=self.name,
            warnings=["无可用 OCR 后端（未安装 paddleocr 且无旁挂文本）"],
        )


_ENGINE_CACHE: Dict[str, OCREngine] = {}


def get_engine(prefer: str = "auto") -> OCREngine:
    """选择 OCR 后端：auto 时优先 paddleocr，其次 sidecar，最后 null。"""
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
        paddle = PaddleOCREngine()
        engine = paddle if paddle.available() else SidecarOCREngine()
    _ENGINE_CACHE[key] = engine
    return engine


def extract_text(path: Path, prefer: str = "auto") -> OCRResult:
    """对扫描件 / 图片抽取文本；非图片后缀直接返回空结果并标注原因。"""
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
