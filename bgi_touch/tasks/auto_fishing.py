"""AutoFishing task migrated from BetterGI's image-driven fishing loop.

The iOS port keeps the deterministic part of the original implementation:
yellow fish-bar segmentation and cursor/target control.  Fish-pond YOLO
selection is intentionally optional; callers can start this task while a rod
is already equipped, which is also the safest behavior for touch devices.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..engine.context import GameContext
from ..engine.recognition import Mat, RecognitionObject

REF_WIDTH = 1920
REF_HEIGHT = 1080
ASSETS = Path(__file__).resolve().parents[2] / "assets" / "templates" / "autofishing"


def get_fish_bar_rects(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return horizontal yellow fish-bar components as `(x, y, w, h)`.

    BetterGI uses HSV_FULL with a hue around 60 degrees, low saturation and a
    bright value.  The contour filters below mirror its height/alignment rules
    and are independent of a live GameContext for offline testing.
    """
    if bgr is None or bgr.size == 0:
        return []
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV_FULL)
    low = np.array([39, 43, 245], dtype=np.uint8)
    high = np.array([46, 104, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        angle = float(cv2.minAreaRect(contour)[2])
        distance_to_horizontal = min(abs(angle) % 45, 45 - abs(angle) % 45)
        if distance_to_horizontal > 1.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w > 0 and h > 0:
            boxes.append((x, y, w, h))
    if not boxes:
        return []

    widest = max(boxes, key=lambda box: box[2])
    wx, wy, ww, wh = widest
    center_y = wy + wh / 2
    return [
        box
        for box in boxes
        if abs((box[1] + box[3] / 2) - center_y) < wh / 5
        and abs(wh - box[3]) < wh / 3
        and box[2] > wh / 4
    ]


def fish_bar_action(rects: list[tuple[int, int, int, int]]) -> str | None:
    """Return `hold`, `release`, or `None` for a detected fish bar."""
    if len(rects) == 2:
        cursor, target = sorted(rects, key=lambda box: box[2])
        if target[2] < cursor[2] * 10:
            return None
        return "hold" if cursor[0] < target[0] else "release"
    if len(rects) == 3:
        left, cursor, right = sorted(rects, key=lambda box: box[0])
        right_gap = right[0] + right[2] - (cursor[0] + cursor[2])
        left_gap = cursor[0] - left[0]
        return "release" if right_gap <= left_gap else "hold"
    return None


def match_fish_bite_words(bgr: np.ndarray, roi: tuple[int, int, int, int]) -> bool:
    """Detect the bright centered bite prompt used before the fish bar."""
    x, y, w, h = roi
    crop = bgr[max(0, y):min(bgr.shape[0], y + h), max(0, x):min(bgr.shape[1], x + w)]
    if crop.size == 0:
        return False
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(rgb, np.array([253, 253, 253], np.uint8), np.array([255, 255, 255], np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilated = cv2.dilate(mask, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    bx, by, bw, bh = max((cv2.boundingRect(c) for c in contours), key=lambda r: r[3])
    return bh < crop.shape[0] and bw / max(1, bh) >= 3 and w > bw * 3 and bx < w / 2 < bx + bw


class AutoFishingTask:
    def __init__(
        self,
        ctx: GameContext,
        target_catches: int = 1,
        timeout_s: float = 120,
        idle_timeout_s: float = 20,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.target_catches = max(1, int(target_catches))
        self.timeout_s = max(5.0, float(timeout_s))
        self.idle_timeout_s = max(2.0, float(idle_timeout_s))
        self.log = log
        self._templates: dict[str, Mat] = {}

    def _template(self, name: str) -> Mat:
        if name not in self._templates:
            self._templates[name] = Mat.from_file(str(ASSETS / f"{name}.png"))
        return self._templates[name]

    def _find(self, region, name: str, roi=None):
        ro = RecognitionObject.template_match(self._template(name), *(roi or (None,) * 4))
        ro.threshold = 0.70
        return region.find(ro)

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        deadline = time.monotonic() + self.timeout_s
        last_activity = time.monotonic()
        last_press = 0.0
        last_bar_seen = 0.0
        holding = False
        catches = 0
        self.log(f"[AutoFishing] 自动钓鱼启动（目标 {self.target_catches} 条）")
        try:
            while time.monotonic() < deadline:
                if cancelled and cancelled():
                    self.log("[AutoFishing] 已取消")
                    return False
                region = self.ctx.capture_region()
                frame = region.bgr
                now = time.monotonic()

                if not self._find(region, "exit_fishing", (150, 0, 480, 270)).is_empty():
                    if holding:
                        self.ctx.input.attack_up()
                    self.log("[AutoFishing] 检测到退出钓鱼按钮")
                    return catches >= self.target_catches

                bar = get_fish_bar_rects(frame[: max(1, frame.shape[0] // 2)])
                action = fish_bar_action(bar)
                if action == "hold":
                    if not holding:
                        self.ctx.input.attack_down()
                        holding = True
                    last_bar_seen = last_activity = now
                elif action == "release":
                    if holding:
                        self.ctx.input.attack_up()
                        holding = False
                    last_bar_seen = last_activity = now
                elif bar:
                    last_bar_seen = last_activity = now

                # A bite prompt is more reliable than OCR on small iPhone text.
                bite = match_fish_bite_words(frame, (frame.shape[1] // 3, 0, frame.shape[1] // 3, frame.shape[0] // 2))
                lift = not self._find(region, "lift_rod", (1440, 400, 480, 540)).is_empty()
                if bite or lift:
                    if now - last_press > 0.45:
                        self.ctx.input.attack()
                        last_press = now
                        last_activity = now
                        self.log("[AutoFishing] 自动提竿")
                elif not bar and not self._find(region, "wait_bite", (1440, 270, 480, 540)).is_empty():
                    last_activity = now
                elif not bar and not self._find(region, "Space", (960, 540, 960, 540)).is_empty():
                    if now - last_press > 0.8:
                        self.ctx.input.key_press("SPACE")
                        last_press = now
                        last_activity = now

                if last_bar_seen and not bar and now - last_bar_seen >= 1.0:
                    catches += 1
                    last_bar_seen = 0.0
                    last_activity = now
                    self.log(f"[AutoFishing] 完成第 {catches} 条")
                    if catches >= self.target_catches:
                        return True
                    if now - last_press > 0.8:
                        self.ctx.input.key_press("SPACE")
                        last_press = now

                if now - last_activity >= self.idle_timeout_s:
                    self.log("[AutoFishing] 未检测到钓鱼界面，结束任务")
                    return False
                self.ctx.sleep(120)
            self.log("[AutoFishing] 超时退出")
            return False
        finally:
            if holding:
                self.ctx.input.attack_up()
            self.ctx.input.release_all()
