"""AutoEat task and trigger for the mobile HUD.

BetterGI reads the current avatar's health through a Windows vision helper and
presses the quick-use gadget when the bar is low.  The iOS version keeps the
same contract but makes the health signal configurable because HUD scale and
safe areas differ between devices.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ..engine.context import GameContext
from ..engine.recognition import Mat, RecognitionObject
from .common_jobs import exclusive_realtime_triggers
from .inventory_grid import (
    InventoryGridScanner,
    normalize_grid_icon,
    recognize_inventory_count,
)

DEFAULT_HEALTH_ROI = (720, 900, 480, 140)
DEFAULT_RECOVERY_ROI = (1810, 778, 23, 23)
DEFAULT_RESURRECTION_ROI = (1810, 778, 18, 19)
FOOD_CONFIRM_ROI = (500, 550, 920, 480)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTO_EAT_ASSETS = PROJECT_ROOT / "assets" / "templates" / "autoeat"
FOOD_CONFIRM_TEMPLATE = PROJECT_ROOT / "assets" / "templates" / "autocook" / "btn_white_confirm.png"


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
        recovery_cache_s: float = 30.0,
        resurrection_cooldown_s: float = 2.0,
        recovery_roi: tuple[int, int, int, int] = DEFAULT_RECOVERY_ROI,
        resurrection_roi: tuple[int, int, int, int] = DEFAULT_RESURRECTION_ROI,
        recovery_detector: Callable[[Any], bool] | None = None,
        resurrection_detector: Callable[[Any], bool] | None = None,
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
        self.recovery_cache_s = max(0.0, float(recovery_cache_s))
        self.resurrection_cooldown_s = max(0.0, float(resurrection_cooldown_s))
        self.recovery_roi = tuple(int(value) for value in recovery_roi)
        self.resurrection_roi = tuple(int(value) for value in resurrection_roi)
        self.recovery_detector = recovery_detector
        self.resurrection_detector = resurrection_detector
        self.log = log
        self._last_check_at = 0.0
        self._last_eat_at = float("-inf")
        self._last_resurrection_at = float("-inf")
        self._last_recovery_check_at = float("-inf")
        self._recovery_detected = False
        self._templates: dict[str, RecognitionObject | None] = {}
        self._template_failures: set[str] = set()

    def _template(
        self,
        name: str,
        roi: tuple[int, int, int, int],
        threshold: float = 0.8,
    ) -> RecognitionObject | None:
        if name in self._templates:
            return self._templates[name]
        if name in self._template_failures:
            return None
        try:
            template = Mat.from_file(str(AUTO_EAT_ASSETS / f"{name}.png"))
            recognition = RecognitionObject.template_match(template, *roi)
            recognition.threshold = threshold
            self._templates[name] = recognition
            return recognition
        except (OSError, ValueError, TypeError) as error:
            self._template_failures.add(name)
            self.log(f"[AutoEat] {name} 模板不可用：{error}")
            return None

    @staticmethod
    def _detector_result(detector: Callable[[Any], bool], region: Any) -> bool:
        try:
            return bool(detector(region))
        except TypeError:
            # Small script/test hosts sometimes provide a zero-argument probe.
            return bool(detector())  # type: ignore[call-arg]

    def _icon_visible(
        self,
        region: Any,
        name: str,
        roi: tuple[int, int, int, int],
    ) -> bool:
        recognition = self._template(name, roi)
        if recognition is None:
            return False
        try:
            result = region.find(recognition)
            return bool(result.is_exist())
        except Exception as error:
            self.log(f"[AutoEat] {name} 识别失败：{error}")
            return False

    def _has_recovery(self, region: Any) -> bool:
        if self.recovery_detector is not None:
            return self._detector_result(self.recovery_detector, region)
        return self._icon_visible(region, "Recovery", self.recovery_roi)

    def _has_resurrection(self, region: Any) -> bool:
        if self.resurrection_detector is not None:
            return self._detector_result(self.resurrection_detector, region)
        return self._icon_visible(region, "Resurrection", self.resurrection_roi)

    def on_frame(self, region) -> None:
        now = time.monotonic()
        if now - self._last_check_at < self.check_interval_s:
            return
        self._last_check_at = now
        try:
            low_hp = current_avatar_is_low_hp(
                region.bgr, self.health_roi, min_width_ref=self.min_width_ref
            )
            if low_hp:
                if now - self._last_recovery_check_at >= self.recovery_cache_s:
                    self._recovery_detected = self._has_recovery(region)
                    self._last_recovery_check_at = now
                if (
                    self._recovery_detected
                    and now - self._last_eat_at >= self.eat_interval_s
                ):
                    self.ctx.input.key_press("Z")
                    self._last_eat_at = now
                    self.log("[AutoEat] 检测到红血，使用便携营养袋")

            if (
                self._has_resurrection(region)
                and now - self._last_resurrection_at >= self.resurrection_cooldown_s
            ):
                self.ctx.input.key_press("Z")
                self._last_resurrection_at = now
                self.log("[AutoEat] 检测到复活图标，使用便携营养袋")
        except Exception as error:
            # A transient OCR/template failure must not stop the shared trigger
            # loop or leave an input action half-completed.
            self.log(f"[AutoEat] 检测失败：{error}")


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
        max_pages: int = 100,
        scanner: Any | None = None,
        recognizer: Any | None = None,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.food_name = str(food_name).strip() if food_name else None
        self.check_interval_ms = max(80, int(check_interval_ms))
        self.eat_interval_s = max(0.5, float(eat_interval_s))
        self.duration_s = max(0.0, float(duration_s))
        self.health_roi = tuple(int(value) for value in health_roi)
        self.min_width_ref = max(10, int(min_width_ref))
        self.max_pages = max(1, min(100, int(max_pages)))
        self.scanner = scanner
        self.recognizer = recognizer
        self.log = log

    def _find_text(
        self,
        *words: str,
        timeout_s: float = 5.0,
        cancelled: Callable[[], bool] | None = None,
    ):
        attempts = max(1, int(float(timeout_s) * 1000 / 250) + 1)
        for attempt in range(attempts):
            if cancelled and cancelled():
                return None
            hits = self.ctx.capture_region().find_multi(
                RecognitionObject.ocr(0, 0, 1920, 1080), limit=50
            )
            for hit in hits:
                text = hit.text.replace(" ", "")
                if any(str(word).replace(" ", "") in text for word in words):
                    return hit
            if attempt + 1 < attempts:
                self.ctx.sleep(250)
        return None

    def _food_recognizer(self):
        if self.recognizer is None:
            from ..vision.item_recognizer import ItemIconRecognizer

            self.recognizer = ItemIconRecognizer()
        return self.recognizer

    @staticmethod
    def _food_names_match(expected: object, actual: object) -> bool:
        compact = lambda value: "".join(str(value or "").split()).casefold()
        left, right = compact(expected), compact(actual)
        return bool(left and right and left == right)

    def _find_food_confirm(
        self,
        *,
        timeout_s: float = 3.0,
        cancelled: Callable[[], bool] | None = None,
    ):
        attempts = max(1, int(float(timeout_s) * 1000 / 250) + 1)
        template: RecognitionObject | None = None
        try:
            template = RecognitionObject.template_match(
                Mat.from_file(str(FOOD_CONFIRM_TEMPLATE)), *FOOD_CONFIRM_ROI,
            )
            template.threshold = 0.74
        except (OSError, ValueError, TypeError) as error:
            self.log(f"[AutoEat] 确认按钮模板不可用，改用 OCR：{error}")

        for attempt in range(attempts):
            if cancelled and cancelled():
                return None
            region = self.ctx.capture_region()
            if template is not None:
                try:
                    result = region.find(template)
                    if result.is_exist():
                        return result
                except Exception:
                    pass
            try:
                hits = region.find_multi(
                    RecognitionObject.ocr(*FOOD_CONFIRM_ROI), limit=20,
                )
                for hit in hits:
                    text = str(getattr(hit, "text", "")).replace(" ", "")
                    if any(word in text for word in ("确认", "確定", "Confirm")):
                        return hit
            except Exception:
                pass
            if attempt + 1 < attempts:
                self.ctx.sleep(250)
        return None

    def _use_named_food(self, cancelled: Callable[[], bool] | None = None) -> int | bool:
        scanner = self.scanner or InventoryGridScanner(
            self.ctx, "Food", max_pages=self.max_pages, log=self.log,
        )
        recognizer = self._food_recognizer()
        opened = False
        with exclusive_realtime_triggers(self.ctx):
            try:
                if cancelled and cancelled():
                    return False
                opened = bool(scanner.open())
                if not opened:
                    self.log("[AutoEat] 无法打开食物背包")
                    return False

                for page, frame, cells in scanner.pages(cancelled=cancelled):
                    for cell in cells:
                        if cancelled and cancelled():
                            return False
                        icon = normalize_grid_icon(cell.crop(frame))
                        if icon.size == 0:
                            continue
                        match = recognizer.match(icon)
                        predicted_name = getattr(match, "name", "")
                        if not self._food_names_match(self.food_name, predicted_name):
                            continue

                        scanner.tap(cell)
                        count_result = recognize_inventory_count(cell.crop(frame))
                        self.ctx.sleep(300)
                        confirm = self._find_food_confirm(cancelled=cancelled)
                        if confirm is None:
                            # A few game versions show an intermediate “使用”
                            # label before the white confirmation button.
                            used = self._find_text(
                                "使用", "Use", timeout_s=0.75, cancelled=cancelled,
                            )
                            if used is not None:
                                used.click()
                                confirm = self._find_food_confirm(cancelled=cancelled)
                        if confirm is None:
                            self.log(f"[AutoEat] 未找到料理确认按钮：{self.food_name}")
                            return False
                        confirm.click()

                        remaining = count_result.count - 1 if count_result.count >= 0 else -2
                        if count_result.count < 0:
                            self.log(
                                f"[AutoEat] 无法识别料理数量（{count_result.reason or 'UNKNOWN'}），"
                                f"仍尝试使用：{predicted_name}"
                            )
                        else:
                            self.log(
                                f"[AutoEat] 已使用料理：{predicted_name}，剩余 {remaining}"
                            )
                        return remaining

                self.log(f"[AutoEat] 背包中未找到料理：{self.food_name}")
                return -1
            finally:
                if opened:
                    try:
                        scanner.close()
                    except Exception as error:
                        self.log(f"[AutoEat] 返回主界面失败：{error}")

    def run(self, cancelled: Callable[[], bool] | None = None) -> int | bool | None:
        try:
            if self.food_name:
                return self._use_named_food(cancelled)

            deadline = (
                time.monotonic() + self.duration_s if self.duration_s > 0 else None
            )
            last_eat_at = 0.0
            self.log("[AutoEat] 自动吃药监控启动")
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
