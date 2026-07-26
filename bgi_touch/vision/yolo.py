"""YOLOv8 ONNX 通用推理器（BetterGI 官方模型：bgi_tree/bgi_mine/bgi_fish 等）。

输出布局 [1, 4+nc, anchors]（xywh + 类别分数），letterbox 预处理 + NMS。
复用 rapidocr-onnxruntime 带来的 onnxruntime，无新增依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODELS = Path(__file__).resolve().parents[2] / "assets" / "models"


@dataclass
class Detection:
    x: float  # 左上角，输入图像坐标
    y: float
    width: float
    height: float
    confidence: float
    class_id: int

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2


class YoloPredictor:
    def __init__(self, model: str | Path):
        import onnxruntime as ort

        path = Path(model)
        if not path.exists():
            path = MODELS / f"{model}.onnx"
        if not path.exists():
            raise FileNotFoundError(f"模型不存在：{model}（运行 tools/fetch_map_assets.py --models）")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])

    def _letterbox(self, bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        h, w = bgr.shape[:2]
        r = min(self.size / w, self.size / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        dx, dy = (self.size - nw) // 2, (self.size - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(bgr, (nw, nh))
        return canvas, r, (dx, dy)

    def predict(self, bgr: np.ndarray, conf_threshold: float = 0.4,
                iou_threshold: float = 0.45) -> list[Detection]:
        img, r, (dx, dy) = self._letterbox(bgr)
        blob = img[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: np.ascontiguousarray(blob)})[0][0]  # (4+nc, N)
        boxes_xywh = out[:4].T          # (N,4) cx,cy,w,h
        scores_all = out[4:].T          # (N,nc)
        class_ids = scores_all.argmax(axis=1)
        scores = scores_all.max(axis=1)
        keep = scores >= conf_threshold
        if not keep.any():
            return []
        boxes_xywh, scores, class_ids = boxes_xywh[keep], scores[keep], class_ids[keep]
        # letterbox 逆变换
        xy = (boxes_xywh[:, :2] - [dx, dy]) / r
        wh = boxes_xywh[:, 2:] / r
        tl = xy - wh / 2
        rects = np.hstack([tl, wh])
        idx = cv2.dnn.NMSBoxes(rects.tolist(), scores.tolist(), conf_threshold, iou_threshold)
        result = []
        for i in np.array(idx).flatten():
            result.append(Detection(float(rects[i][0]), float(rects[i][1]),
                                    float(rects[i][2]), float(rects[i][3]),
                                    float(scores[i]), int(class_ids[i])))
        return sorted(result, key=lambda d: -d.confidence)
