"""小地图定位器：GameContext 帧 → 世界坐标（PathingExecutor 的 Positioner 实现）。

裁剪位置来自布局 profile 的 minimapCenter；匹配走 MapLocator（SIFT 两级）。
注：原版还有小地图 alpha 渐晕补偿（MiniMapPreprocessor），实测朴素灰度在
SIFT 下已能稳定匹配，暂未移植；若弱纹理区域丢失率高再补。
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from ..engine.context import GameContext
from .map_locator import MapLocator

MINIMAP_RADIUS_N = 0.042  # 半径 / 屏宽（实测 iPhone 13 Pro Max）


class MinimapPositioner:
    def __init__(self, ctx: GameContext, map_name: str = "Teyvat"):
        self.ctx = ctx
        self.locator = MapLocator(map_name)
        self.map_name = self.locator.map_name
        self._last_position: tuple[float, float] | None = None
        self._last_fix_at = 0.0

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

    def get_position_stable(
        self,
        bgr: np.ndarray,
        *,
        cache_time_ms: int = 120,
        max_jump: float = 260.0,
    ) -> tuple[float, float] | None:
        """Locate a frame while rejecting implausible SIFT jumps.

        BetterGI keeps a previous map position and retries with a global match
        when the local result is missing or suddenly jumps.  A short cache is
        useful on iOS because screenshot and input round trips are slower than
        the Windows capture loop, while ``max_jump`` prevents a single bad
        descriptor match from steering the character across the map.
        """
        now = time.monotonic()
        if (
            self._last_position is not None
            and cache_time_ms > 0
            and (now - self._last_fix_at) * 1000 < cache_time_ms
        ):
            return self._last_position

        position = self.get_position(bgr)
        if position is None:
            return None

        if self._last_position is not None:
            jump = math.hypot(
                position[0] - self._last_position[0],
                position[1] - self._last_position[1],
            )
            if jump > max_jump:
                # The local result may be a false match. Re-run from a clean
                # locator state before exposing it to the movement controller.
                self.locator.reset()
                retry = self.get_position(bgr)
                if retry is None:
                    return None
                retry_jump = math.hypot(
                    retry[0] - self._last_position[0],
                    retry[1] - self._last_position[1],
                )
                if retry_jump > max_jump:
                    return None
                position = retry

        self._last_position = position
        self._last_fix_at = now
        return position

    def get_position_pixel(self, bgr: np.ndarray) -> tuple[float, float] | None:
        mm = self.crop_minimap(bgr)
        if mm is None:
            return None
        return self.locator.locate_pixel(mm)

    def reset(self) -> None:
        self.locator.reset()
        self._last_position = None
        self._last_fix_at = 0.0

    def set_prior(self, wx: float, wy: float) -> None:
        """已知世界坐标（如刚传送到锚点）时设置局部搜索先验，避免全局匹配。"""
        if self.locator.config is None:
            self.reset()
            return
        self.locator.prev = self.locator.config.world_to_image(wx, wy)
        self._last_position = (float(wx), float(wy))
        self._last_fix_at = 0.0
