"""小地图定位器：GameContext 帧 → 世界坐标（PathingExecutor 的 Positioner 实现）。

裁剪位置来自布局 profile 的 minimapCenter；匹配走 MapLocator（SIFT 两级）。
注：原版还有小地图 alpha 渐晕补偿（MiniMapPreprocessor），实测朴素灰度在
SIFT 下已能稳定匹配，暂未移植；若弱纹理区域丢失率高再补。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..engine.context import GameContext
from .map_locator import MapLocator

MINIMAP_RADIUS_N = 0.042  # 半径 / 屏宽（实测 iPhone 13 Pro Max）


class MinimapPositioner:
    def __init__(self, ctx: GameContext, map_name: str = "Teyvat"):
        self.ctx = ctx
        self.locator = MapLocator(map_name)

    def crop_minimap(self, bgr: np.ndarray) -> np.ndarray | None:
        mm = self.ctx.layout.buttons.get("minimapCenter")
        if mm is None:
            return None
        h, w = bgr.shape[:2]
        cx, cy, r = mm[0] * w, mm[1] * h, MINIMAP_RADIUS_N * w
        x0, y0 = int(cx - r), int(cy - r)
        size = int(2 * r)
        if x0 < 0 or y0 < 0 or y0 + size > h or x0 + size > w:
            return None
        return cv2.cvtColor(bgr[y0:y0 + size, x0:x0 + size], cv2.COLOR_BGR2GRAY)

    def get_position(self, bgr: np.ndarray) -> tuple[float, float] | None:
        mm = self.crop_minimap(bgr)
        if mm is None:
            return None
        return self.locator.locate_world(mm)

    def get_position_pixel(self, bgr: np.ndarray) -> tuple[float, float] | None:
        mm = self.crop_minimap(bgr)
        if mm is None:
            return None
        return self.locator.locate_pixel(mm)

    def reset(self) -> None:
        self.locator.reset()
