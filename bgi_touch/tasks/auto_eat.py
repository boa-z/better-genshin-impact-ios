"""AutoEat task and trigger for the mobile HUD.

BetterGI reads the current avatar's health through a Windows vision helper and
presses the quick-use gadget when the bar is low.  The iOS version keeps the
same contract but makes the health signal configurable because HUD scale and
safe areas differ between devices.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..engine.context import GameContext
from ..engine.recognition import RecognitionObject

DEFAULT_HEALTH_ROI = (720, 900, 480, 140)


def _ref_crop(bgr: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a centered 1920x1080 reference ROI from a landscape frame."""
    rx, ry, rw, rh = roi
    scale = bgr.shape[0] / 1080.0
    x = int(round((bgr.shape[1] - 1920 * scale) / 2 + rx * scale))
    y = int(round(ry * scale))
    w = max(1, int(round(rw * scale)))
    h = max(1, int(round(rh * scale)))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(bgr.shape[1], x + w), min(bgr.shape[0], y + h)
    return bgr[y0:y1, x0:x1]


def red_bar_components(
    bgr: np.ndarray,
    roi: tuple[int, int, int, int] = DEFAULT_HEALTH_ROI,
    *,
    min_width_ref: int = 55,
) -> list[tuple[int, int, int, int]]:
    """Return red horizontal bar components inside the health ROI."""
    crop = _ref_crop(bgr, roi)
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.array([0, 90, 70]), np.array([14, 255, 255]))
    red |= cv2.inRange(hsv, np.array([170, 90, 70]), np.array([180, 255, 255]))
    red = cv2.morphologyEx(
        red,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
    )
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Scale against the complete frame. Using the cropped height makes the
    # threshold change when a custom ROI is clipped by a safe area.
    min_width = max(10, round(min_width_ref * bgr.shape[0] / 1080.0))
    output = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_width and w / max(1, h) >= 2.5 and h >= 2:
            output.append((x, y, w, h))
    return sorted(output, key=lambda rect: rect[2], reverse=True)


def current_avatar_is_low_hp(
    bgr: np.ndarray,
    roi: tuple[int, int, int, int] = DEFAULT_HEALTH_ROI,
    *,
    min_width_ref: int = 55,
) -> bool:
    """Return true only when a substantial horizontal red health bar is found."""
    return bool(red_bar_components(bgr, roi, min_width_ref=min_width_ref))


class AutoEatTrigger:
    name = "AutoEat"

    def __init__(
        self,
        ctx: GameContext,
        *,
        health_roi: tuple[int, int, int, int] = DEFAULT_HEALTH_ROI,
        check_interval_ms: int = 150,
        eat_interval_s: float = 8.0,
        eat_interval_ms: int | None = None,
        min_width_ref: int = 55,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.enabled = True
        self.health_roi = tuple(int(value) for value in health_roi)
        if eat_interval_ms is not None:
            eat_interval_s = float(eat_interval_ms) / 1000.0
        self.check_interval_s = max(0.02, int(check_interval_ms) / 1000.0)
        self.eat_interval_s = max(0.5, float(eat_interval_s))
        self.min_width_ref = max(10, int(min_width_ref))
        self.log = log
        self._last_check_at = 0.0
        self._last_eat_at = 0.0

    def on_frame(self, region) -> None:
        now = time.monotonic()
        if now - self._last_check_at < self.check_interval_s:
            return
        self._last_check_at = now
        if now - self._last_eat_at < self.eat_interval_s:
            return
        if not current_avatar_is_low_hp(
            region.bgr, self.health_roi, min_width_ref=self.min_width_ref
        ):
            return
        self.ctx.input.key_press("Z")
        self._last_eat_at = now
        self.log("[AutoEat] 检测到低生命值，使用便携营养袋")


class AutoEatTask:
    """Monitor low HP or consume one named food through the inventory UI."""

    def __init__(
        self,
        ctx: GameContext,
        *,
        food_name: str | None = None,
        check_interval_ms: int = 500,
        eat_interval_s: float = 8.0,
        duration_s: float = 0.0,
        health_roi: tuple[int, int, int, int] = DEFAULT_HEALTH_ROI,
        min_width_ref: int = 55,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.food_name = str(food_name).strip() if food_name else None
        self.check_interval_ms = max(80, int(check_interval_ms))
        self.eat_interval_s = max(0.5, float(eat_interval_s))
        self.duration_s = max(0.0, float(duration_s))
        self.health_roi = tuple(int(value) for value in health_roi)
        self.min_width_ref = max(10, int(min_width_ref))
        self.log = log

    def _find_text(self, *words: str, timeout_s: float = 5.0):
        deadline = time.monotonic() + max(0.1, timeout_s)
        while time.monotonic() < deadline:
            hits = self.ctx.capture_region().find_multi(
                RecognitionObject.ocr(0, 0, 1920, 1080), limit=50
            )
            for hit in hits:
                text = hit.text.replace(" ", "")
                if any(str(word).replace(" ", "") in text for word in words):
                    return hit
            self.ctx.sleep(250)
        return None

    def _use_named_food(self) -> bool:
        self.ctx.input.key_press("B")
        self.ctx.sleep(1200)
        food = self._find_text(self.food_name or "", timeout_s=6)
        if food is None:
            self.ctx.input.key_press("ESCAPE")
            self.log(f"[AutoEat] 背包中未找到料理：{self.food_name}")
            return False
        food.click()
        self.ctx.sleep(300)
        used = self._find_text("使用", "Use", timeout_s=3)
        if used is not None:
            used.click()
        self.ctx.sleep(500)
        confirm = self._find_text("确认", "确定", "Confirm", timeout_s=3)
        if confirm is not None:
            confirm.click()
        self.ctx.sleep(800)
        self.ctx.input.key_press("ESCAPE")
        self.log(f"[AutoEat] 已尝试使用料理：{self.food_name}")
        return True

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        if self.food_name:
            if cancelled and cancelled():
                return False
            return self._use_named_food()

        deadline = (
            time.monotonic() + self.duration_s if self.duration_s > 0 else None
        )
        last_eat_at = 0.0
        self.log("[AutoEat] 自动吃药监控启动")
        try:
            while deadline is None or time.monotonic() < deadline:
                if cancelled and cancelled():
                    return False
                now = time.monotonic()
                if now - last_eat_at >= self.eat_interval_s:
                    frame = self.ctx.capture_bgr()
                    if current_avatar_is_low_hp(
                        frame, self.health_roi, min_width_ref=self.min_width_ref
                    ):
                        self.ctx.input.key_press("Z")
                        last_eat_at = now
                        self.log("[AutoEat] 检测到低生命值，使用便携营养袋")
                self.ctx.sleep(self.check_interval_ms)
            return True
        finally:
            self.ctx.input.release_all()
