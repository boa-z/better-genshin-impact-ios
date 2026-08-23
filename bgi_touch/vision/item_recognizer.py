"""BetterGI ItemV2 icon embedding recognizer.

The model and prototype CSV are distributed by BetterGI.Assets.Model and are
downloaded into the ignored assets/models directory by fetch_map_assets.py.
"""

from __future__ import annotations

import base64
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


MODELS = Path(__file__).resolve().parents[2] / "assets" / "models"


@dataclass(frozen=True)
class ItemIconMatch:
    name: str
    score: float
    quality_level: int = -1


class ItemIconRecognizer:
    """Port of BetterGI's ItemRecognizer with vectorized cosine matching."""

    def __init__(
        self,
        model_path: str | Path = MODELS / "item.onnx",
        prototypes_path: str | Path = MODELS / "item.csv",
    ):
        import onnxruntime as ort

        model_path = Path(model_path)
        prototypes_path = Path(prototypes_path)
        if not model_path.exists() or not prototypes_path.exists():
            raise FileNotFoundError(
                "缺少 BetterGI ItemV2 模型；请运行 "
                "`.venv/bin/python tools/fetch_map_assets.py --models`"
            )
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.names, self.quality_levels, self.prototypes = self._load_prototypes(
            prototypes_path
        )

    @staticmethod
    def _load_prototypes(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
        names: list[str] = []
        qualities: list[int] = []
        vectors: list[np.ndarray] = []
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                vector = np.frombuffer(
                    base64.b64decode(row["embedding"]), dtype="<f4"
                ).astype(np.float32, copy=True)
                norm = float(np.linalg.norm(vector))
                if norm <= 1e-12:
                    continue
                names.append(row["item_name"].strip())
                class_id = row["item_class_id"].strip().casefold()
                quality = row.get("quality_level", "").strip()
                qualities.append(
                    int(quality) if class_id.startswith("relic:") and quality else -1
                )
                vectors.append(vector / norm)
        if not vectors:
            raise ValueError(f"ItemV2 原型表为空：{path}")
        return names, np.asarray(qualities, dtype=np.int16), np.stack(vectors)

    def match(self, icon_bgr: np.ndarray) -> ItemIconMatch:
        if icon_bgr.shape[:2] != (125, 125):
            icon_bgr = cv2.resize(icon_bgr, (125, 125), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32)
        tensor = (tensor / 255.0 - 0.5) / 0.5
        outputs = self.session.run(None, {self.input_name: np.ascontiguousarray(tensor)})
        feature = np.asarray(outputs[0]).reshape(-1).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(feature))
        if norm <= 1e-12:
            return ItemIconMatch("", float("-inf"))
        scores = self.prototypes @ (feature / norm)
        index = int(np.argmax(scores))
        return ItemIconMatch(
            self.names[index],
            float(scores[index]),
            int(self.quality_levels[index]),
        )
