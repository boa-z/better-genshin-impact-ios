"""自动剧情触发器（原版 AutoSkip 的移动端适配）。

检测条件（防误触，先判定在对话中）：
- 对话选项图标 icon_option 模板命中 → 点选项（可偏好含指定文本/最上面一项）
- 或左上角自动播放指示（stop_auto 模板）命中 → 点屏幕中下部推进对话
- 对话结束后的有限窗口内，稳定识别并关闭普通页、道具页和初见角色横幅

模板来自原版 AutoSkip/Assets（1080p 基准，识别层自动缩放）。
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..engine.context import GameContext
from ..engine.recognition import ImageRegion, Mat, RecognitionObject
from ..vision.game_ui import is_big_map_ui, is_main_ui

TEMPLATES = Path(__file__).resolve().parents[2] / "assets" / "templates" / "autoskip"


class AutoSkipTrigger:
    name = "AutoSkip"

    # A Wi-Fi iPhone screenshot can take 3.5-5.6 seconds. Two-frame stable
    # confirmation therefore needs a slightly wider window than desktop's 10s.
    POPUP_WINDOW_S = 15.0
    PAGE_CLOSE_STABLE_S = 0.2
    BLACK_CLICK_INTERVAL_S = 1.2
    ITEM_CLICK_INTERVAL_S = 1.0

    def __init__(self, ctx: GameContext, prefer_text: str | None = None,
                 priority_texts: list[str] | None = None,
                 click_option: str = "优先选择第一个选项",
                 quickly_skip: bool = True,
                 skip_built_in_options: bool = False,
                 after_choose_delay_ms: int = 0,
                 before_confirm_delay_ms: int = 0,
                 close_popup_pages: bool = True,
                 log: Callable[[str], None] = print,
                 clock: Callable[[], float] = time.monotonic,
                 main_ui_detector: Callable[[object, np.ndarray], bool] = is_main_ui,
                 big_map_detector: Callable[[object, np.ndarray], bool] = is_big_map_ui):
        self.ctx = ctx
        self.enabled = True
        self.priority_texts = list(priority_texts or ([prefer_text] if prefer_text else []))
        self.click_option = click_option
        self.quickly_skip = quickly_skip
        self.skip_built_in_options = skip_built_in_options
        self.after_choose_delay_ms = max(0, int(after_choose_delay_ms))
        self.before_confirm_delay_ms = max(0, int(before_confirm_delay_ms))
        self.close_popup_pages = bool(close_popup_pages)
        self.log = log
        self._clock = clock
        self._main_ui_detector = main_ui_detector
        self._big_map_detector = big_map_detector
        self._last_dialogue_at: float | None = None
        self._page_close_seen_at: float | None = None
        self._last_black_click_at = float("-inf")
        self._last_item_click_at = float("-inf")
        # 选项图标出现在屏幕右侧偏下（ref 空间 ROI 收窄降误报）
        self.ro_option = RecognitionObject.template_match(
            Mat.from_file(str(TEMPLATES / "icon_option.png")), 1000, 280, 850, 700)
        self.ro_option.threshold = 0.75
        # 对话中的左上"自动播放"指示
        self.ro_auto = RecognitionObject.template_match(
            Mat.from_file(str(TEMPLATES / "stop_auto.png")), 0, 0, 400, 140)
        self.ro_auto.threshold = 0.75
        self.ro_page_close = RecognitionObject.template_match(
            Mat.from_file(str(TEMPLATES / "page_close.png")), 1600, 0, 320, 160)
        self.ro_page_close.threshold = 0.72
        self.ro_popup_guards = []
        for name in ("guiding_notes.png", "chat_history.png", "valiant_chronicles.png"):
            guard = RecognitionObject.template_match(
                Mat.from_file(str(TEMPLATES / name)), 0, 0, 320, 180)
            guard.threshold = 0.75
            self.ro_popup_guards.append(guard)

    def on_frame(self, region: ImageRegion) -> None:
        now = self._clock()
        options = region.find_multi(self.ro_option, limit=6)
        if options:
            self._last_dialogue_at = now
            self._page_close_seen_at = None
            self._choose_option(region, options)
            return

        auto_playing = region.find(self.ro_auto).is_exist()
        if auto_playing:
            self._last_dialogue_at = now
            self._page_close_seen_at = None
            if self.quickly_skip:
                # 对话进行中且无选项 → 点中下部推进
                if self.before_confirm_delay_ms:
                    self.ctx.sleep(self.before_confirm_delay_ms)
                self.ctx.input.click_ref(960, 820)
            return

        if (
            self.close_popup_pages
            and self._last_dialogue_at is not None
            and now - self._last_dialogue_at <= self.POPUP_WINDOW_S
        ):
            if self._handle_post_dialogue_popup(region, now):
                return
        else:
            self._page_close_seen_at = None

        self._click_black_screen(region, now)

    def _choose_option(self, region: ImageRegion, options: list) -> bool:
        """Apply BetterGI's custom-priority then default-order semantics."""

        options = sorted(options, key=lambda option: option.y)
        chosen = None
        if self.priority_texts:
            texts = []
            for option in options:
                line = region.find(RecognitionObject.ocr(
                    option.x + 30, option.y - 12, 800, 60
                ))
                texts.append(line.text if line.is_exist() else "")
            for preferred in self.priority_texts:
                chosen = next((
                    option for option, text in zip(options, texts)
                    if preferred in text
                ), None)
                if chosen is not None:
                    break

        # Upstream custom priorities still apply when the default policy is
        # "不选择选项". SkipBuiltInClickOptions only disables built-in keyword
        # lists; it must not disable custom/default selection entirely.
        if chosen is None:
            if self.click_option == "不选择选项":
                return False
            if self.click_option == "优先选择最后一个选项":
                chosen = options[-1]
            elif self.click_option == "随机选择选项":
                chosen = random.choice(options)
            else:
                chosen = options[0]

        self.log(f"[AutoSkip] 点击对话选项 @({chosen.x:.0f},{chosen.y:.0f})")
        if self.after_choose_delay_ms:
            self.ctx.sleep(self.after_choose_delay_ms)
        chosen.click()
        return True

    def _handle_post_dialogue_popup(self, region: ImageRegion, now: float) -> bool:
        page_close = region.find(self.ro_page_close)
        if page_close.is_exist():
            protected = (
                self._main_ui_detector(self.ctx, region.bgr)
                or self._big_map_detector(self.ctx, region.bgr)
                or any(region.find(guard).is_exist() for guard in self.ro_popup_guards)
            )
            if protected:
                self._page_close_seen_at = None
                return False
            if self._page_close_seen_at is None:
                self._page_close_seen_at = now
                return False
            if now - self._page_close_seen_at < self.PAGE_CLOSE_STABLE_S:
                return False
            page_close.click()
            self._page_close_seen_at = None
            self._last_dialogue_at = now
            self.log("[AutoSkip] 关闭剧情弹出页")
            return True
        self._page_close_seen_at = None

        is_main = self._main_ui_detector(self.ctx, region.bgr)
        is_big_map = self._big_map_detector(self.ctx, region.bgr)
        if is_main or is_big_map:
            return False
        if self._close_item_popup(region.bgr, now):
            return True
        return self._close_character_popup(region.bgr, now)

    def _reference_frame(self, bgr: np.ndarray) -> np.ndarray:
        """Center-crop wide iPhone frames into BetterGI's 1920x1080 space."""

        height, width = bgr.shape[:2]
        target_ratio = 16 / 9
        if width / max(1, height) >= target_ratio:
            content_width = max(1, round(height * target_ratio))
            left = max(0, (width - content_width) // 2)
            crop = bgr[:, left:left + content_width]
        else:
            content_height = max(1, round(width / target_ratio))
            top = max(0, (height - content_height) // 2)
            crop = bgr[top:top + content_height, :]
        return cv2.resize(crop, (1920, 1080), interpolation=cv2.INTER_AREA)

    def _close_item_popup(self, bgr: np.ndarray, now: float) -> bool:
        if now - self._last_item_click_at < self.ITEM_CLICK_INTERVAL_S:
            return False
        reference = self._reference_frame(bgr)
        crop = reference[980:1060, 945:975]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, np.array((0, 240, 229)), np.array((25, 255, 255)))
        blue = cv2.inRange(hsv, np.array((90, 156, 145)), np.array((99, 208, 253)))
        contours = []
        for mask in (yellow, blue):
            found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(found)
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if not (10 <= area <= 50 and len(approx) == 3):
                continue
            x, y, width, height = cv2.boundingRect(approx)
            self.ctx.input.click_ref(945 + x + width / 2, 980 + y + height / 2)
            self._last_item_click_at = now
            self._last_dialogue_at = now
            self.log(f"[AutoSkip] 关闭道具弹出页（三角面积 {area:.0f}）")
            return True
        return False

    def _close_character_popup(self, bgr: np.ndarray, now: float) -> bool:
        reference = self._reference_frame(bgr).copy()
        cv2.rectangle(reference, (240, 395), (540, 445), (229, 241, 245), -1)
        cv2.rectangle(reference, (290, 660), (500, 700), (101, 82, 74), -1)
        hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
        light = cv2.inRange(hsv, np.array((18, 16, 234)), np.array((27, 19, 250)))
        dark = cv2.inRange(hsv, np.array((101, 57, 95)), np.array((118, 85, 106)))
        combined = cv2.bitwise_or(light, dark)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image_area = reference.shape[0] * reference.shape[1]
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if height == 0:
                continue
            area_ratio = width * height / image_area
            aspect_ratio = width / height
            if not (0.24 < area_ratio < 0.3 and 5.6 <= aspect_ratio <= 7.2):
                continue
            if y <= 1080 * 0.3 or y + height >= 1080 * 0.7:
                continue
            if cv2.countNonZero(light[y:y + height, x:x + width]) == 0:
                continue
            if cv2.countNonZero(dark[y:y + height, x:x + width]) == 0:
                continue
            self.ctx.input.click_ref(100, 100)
            self._last_dialogue_at = now
            self.log("[AutoSkip] 关闭初见角色弹出页")
            return True
        return False

    def _click_black_screen(self, region: ImageRegion, now: float) -> bool:
        if now - self._last_black_click_at < self.BLACK_CLICK_INTERVAL_S:
            return False
        gray = cv2.cvtColor(region.bgr, cv2.COLOR_BGR2GRAY)
        band = gray[gray.shape[0] // 3:gray.shape[0] * 2 // 3]
        if band.size == 0:
            return False
        black_rate = float(np.count_nonzero(band == 0)) / band.size
        if not 0.5 <= black_rate < 0.98999:
            return False
        self.ctx.input.click_ref(960, 540)
        self._last_black_click_at = now
        self.log(f"[AutoSkip] 点击黑屏（黑色比例 {black_rate:.3f}）")
        return True
