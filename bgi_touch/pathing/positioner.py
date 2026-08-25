"""小地图定位器：GameContext 帧 → 世界坐标（PathingExecutor 的 Positioner 实现）。

裁剪位置来自布局 profile 的 minimapCenter；匹配走 BetterGI 风格的
小地图预处理与 MapLocator（SIFT 两级）。
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from ..engine.context import GameContext
from .camera import orientation_with_confidence
from .map_locator import MapLocator
from .minimap import (
    MINIMAP_CAPTURE_RADIUS_N,
    MINIMAP_NORMALIZED_SIZE,
    preprocess_minimap_for_matching,
)

MINIMAP_RADIUS_N = MINIMAP_CAPTURE_RADIUS_N  # 半径 / 屏宽（实测 iPhone 13 Pro Max）
# BetterGI's MiniMapPreprocessor normalizes the usable minimap to 156x156
# before matching.  The same HUD occupies about 233x233 native pixels on an
# iPhone 13 Pro Max; feeding those pixels to SIFT directly changes descriptor
# scale enough to lose otherwise valid matches.
MINIMAP_MATCH_SIZE = 156


class MinimapPositioner:
    def __init__(self, ctx: GameContext, map_name: str = "Teyvat"):
        self.ctx = ctx
        self.locator = MapLocator(map_name)
        self.map_name = self.locator.map_name
        self._last_position: tuple[float, float] | None = None
        self._last_fix_at = 0.0

    def _crop_native_minimap(self, bgr: np.ndarray) -> np.ndarray | None:
        mm = self.ctx.layout.buttons.get("minimapCenter")
        if mm is None:
            return None
        if not isinstance(bgr, np.ndarray) or bgr.ndim < 2:
            return None
        h, w = bgr.shape[:2]
        cx, cy, r = mm[0] * w, mm[1] * h, MINIMAP_RADIUS_N * w
        x0, y0 = int(cx - r), int(cy - r)
        size = int(2 * r)
        if x0 < 0 or y0 < 0 or y0 + size > h or x0 + size > w:
            return None
        return bgr[y0:y0 + size, x0:x0 + size]

    def crop_minimap_color(self, bgr: np.ndarray) -> np.ndarray | None:
        """Return the native minimap normalized to BetterGI's 212px input."""
        minimap = self._crop_native_minimap(bgr)
        if minimap is None:
            return None
        if minimap.shape[:2] != (MINIMAP_NORMALIZED_SIZE, MINIMAP_NORMALIZED_SIZE):
            minimap = cv2.resize(
                minimap,
                (MINIMAP_NORMALIZED_SIZE, MINIMAP_NORMALIZED_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
        return minimap

    def crop_minimap(self, bgr: np.ndarray) -> np.ndarray | None:
        """Return the legacy 156px grayscale crop used by callers."""
        minimap = self._crop_native_minimap(bgr)
        if minimap is None:
            return None
        if minimap.shape[:2] != (MINIMAP_MATCH_SIZE, MINIMAP_MATCH_SIZE):
            minimap = cv2.resize(
                minimap,
                (MINIMAP_MATCH_SIZE, MINIMAP_MATCH_SIZE),
                # Keep BetterGI/OpenCV's linear resampling. INTER_AREA looks
                # smoother but changes the SIFT descriptors enough to drop
                # valid local matches on the native iPhone minimap.
                interpolation=cv2.INTER_LINEAR,
            )
        return cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)

    def _preprocessed_query(
        self,
        bgr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        minimap = self.crop_minimap_color(bgr)
        if minimap is None:
            return None
        angle, _confidence = orientation_with_confidence(minimap)
        return preprocess_minimap_for_matching(minimap, angle)

    def get_position(self, bgr: np.ndarray) -> tuple[float, float] | None:
        query = self._preprocessed_query(bgr)
        if query is not None:
            position = self.locator.locate_world(*query)
            if position is not None:
                return position
        # The BetterGI-preprocessed image is ideal for the desktop template
        # matcher, while the iOS SIFT asset path can still prefer the raw
        # 156px descriptor geometry.  Keep that proven query as a bounded
        # fallback when the new mask/alpha path has no match.
        legacy = self.crop_minimap(bgr)
        return None if legacy is None else self.locator.locate_world(legacy)

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
        query = self._preprocessed_query(bgr)
        if query is not None:
            position = self.locator.locate_pixel(*query)
            if position is not None:
                return position
        legacy = self.crop_minimap(bgr)
        return None if legacy is None else self.locator.locate_pixel(legacy)

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
