"""AutoCook SoloTask migrated from BetterGI's color-bar state machine.

The original task does not need OCR or a model: it detects the cooking icon,
counts the exact warm-yellow pixels in the timing bar, and presses Space after
the stable peak drops.  Keeping the detector pure makes it testable without a
device and keeps the iPhone task small.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..engine.context import GameContext
from ..engine.recognition import Mat, RecognitionObject

REF_WIDTH = 1920
REF_HEIGHT = 1080
COOK_COLOR_RECT = (600, 660, 730, 190)
TARGET_COOK_COLOR_RGB = (255, 192, 64)

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "templates" / "autocook"


def count_target_color(
    bgr: np.ndarray,
    rect: tuple[int, int, int, int] = COOK_COLOR_RECT,
    scale: float = 1.0,
    centered: bool = True,
) -> int:
    """Count BetterGI's cooking-bar color in a reference-space rectangle.

    The original C# task converts BGR to RGB and matches `(255, 192, 64)`
    exactly.  The optional tolerance is deliberately omitted to preserve the
    original signal; screenshots are already rendered in the same color space.
    """
    x, y, w, h = rect
    if centered:
        x = int(round((bgr.shape[1] - REF_WIDTH * scale) / 2 + x * scale))
    else:
        x = int(round(x * scale))
    y = int(round(y * scale))
    w = int(round(w * scale))
    h = int(round(h * scale))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(bgr.shape[1], x + w), min(bgr.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return 0
    rgb = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    color = np.asarray(TARGET_COOK_COLOR_RGB, dtype=np.uint8)
    mask = cv2.inRange(rgb, color, color)
    return int(cv2.countNonZero(mask))


def update_peak(
    current: int,
    candidate: int | None,
    stable_frames: int,
    *,
    peak_min: int = 600,
    tolerance: int = 20,
    required_frames: int = 3,
) -> tuple[int | None, int, int | None]:
    """Advance peak tracking, returning `(candidate, stable, built_peak)`."""
    if current <= peak_min:
        return None, 0, None
    if candidate is None:
        return current, 1, None
    if abs(current - candidate) <= tolerance:
        candidate = max(candidate, current)
        stable_frames += 1
        if stable_frames >= required_frames:
            return None, 0, candidate
        return candidate, stable_frames, None
    return current, 1, None


class AutoCookTask:
    def __init__(
        self,
        ctx: GameContext,
        check_interval_ms: int = 400,
        stop_on_recover: bool = True,
        idle_timeout_s: float = 15,
        timeout_s: float = 900,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.check_interval_ms = max(20, int(check_interval_ms))
        self.stop_on_recover = bool(stop_on_recover)
        self.idle_timeout_s = max(1.0, float(idle_timeout_s))
        self.timeout_s = max(1.0, float(timeout_s))
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
        first_seen = time.monotonic()
        last_ui_check = 0.0
        in_cook_ui = False
        peak: int | None = None
        candidate: int | None = None
        stable = 0
        self.log("[AutoCook] 自动烹饪任务启动")
        try:
            while time.monotonic() < deadline:
                if cancelled and cancelled():
                    self.log("[AutoCook] 已取消")
                    return False
                region = self.ctx.capture_region()
                now = time.monotonic()
                if not in_cook_ui or now - last_ui_check >= self.check_interval_ms / 1000:
                    current_ui = not self._find(region, "ui_left_top_cook_icon", (0, 0, 300, 180)).is_empty()
                    if current_ui != in_cook_ui:
                        peak = candidate = None
                        stable = 0
                    in_cook_ui = current_ui
                    last_ui_check = now
                    if not in_cook_ui:
                        if now - first_seen >= self.idle_timeout_s:
                            self.log("[AutoCook] 未检测到烹饪界面，结束任务")
                            return False
                        self.ctx.sleep(self.check_interval_ms)
                        continue
                    if self.stop_on_recover and not self._find(
                        region, "btn_white_recover", (580, 950, 90, 95)
                    ).is_empty():
                        self.log("[AutoCook] 检测到自动烹饪按钮，结束任务")
                        return True
                    confirm = self._find(region, "btn_white_confirm")
                    if not confirm.is_empty():
                        confirm.click()
                        candidate = peak = None
                        stable = 0
                        self.log("[AutoCook] 自动确认")

                if in_cook_ui:
                    current = count_target_color(
                        region.bgr,
                        scale=region.bgr.shape[0] / REF_HEIGHT,
                    )
                    if peak is not None:
                        if current <= peak - int(300 * region.bgr.shape[0] / REF_HEIGHT):
                            self.ctx.input.key_press("SPACE")
                            self.log(f"[AutoCook] 烹饪条下降，按 Space（峰值 {peak}，当前 {current}）")
                            peak = candidate = None
                            stable = 0
                    else:
                        candidate, stable, built = update_peak(
                            current,
                            candidate,
                            stable,
                            peak_min=int(600 * region.bgr.shape[0] / REF_HEIGHT),
                        )
                        if built is not None:
                            peak = built
                            self.log(f"[AutoCook] 记录稳定峰值 {built}")
                self.ctx.sleep(self.check_interval_ms)
            self.log("[AutoCook] 超时退出")
            return False
        finally:
            self.ctx.input.release_all()
