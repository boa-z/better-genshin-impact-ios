"""可插拔 OCR。默认后端 RapidOCR（PaddleOCR 模型，与 BetterGI 原版同源）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OcrItem:
    text: str
    x: float
    y: float
    width: float
    height: float
    confidence: float


class NullOcr:
    _warned = False

    def recognize(self, bgr: np.ndarray) -> list[OcrItem]:
        if not NullOcr._warned:
            NullOcr._warned = True
            print("[ocr] 未安装 OCR 后端（pip install rapidocr-onnxruntime），OCR 识别返回空结果")
        return []


class RapidOcrBackend:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR  # 延迟导入，可选依赖

        self._engine = RapidOCR()

    def recognize(self, bgr: np.ndarray) -> list[OcrItem]:
        result, _ = self._engine(bgr)
        items: list[OcrItem] = []
        for box, text, conf in result or []:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            items.append(OcrItem(text=text, x=min(xs), y=min(ys),
                                 width=max(xs) - min(xs), height=max(ys) - min(ys),
                                 confidence=float(conf)))
        return items


_provider = None


def get_ocr():
    global _provider
    if _provider is None:
        try:
            _provider = RapidOcrBackend()
        except ImportError:
            _provider = NullOcr()
    return _provider


def set_ocr(provider) -> None:
    global _provider
    _provider = provider
