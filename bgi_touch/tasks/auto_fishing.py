"""AutoFishing task migrated from BetterGI's image-driven fishing loop.

The iOS port includes BetterGI's fish YOLO labels, bait policy, HutaoFisher
rod-distance model, bite recognition, and yellow fish-bar controller.
"""

from __future__ import annotations

import math
import time
from enum import IntEnum
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..engine.context import GameContext
from ..engine.recognition import Mat, RecognitionObject
from ..vision.item_recognizer import ItemIconRecognizer
from .fishing_model import FishPond, FishPredictor, choose_bait, rod_state

REF_WIDTH = 1920
REF_HEIGHT = 1080
ASSETS = Path(__file__).resolve().parents[2] / "assets" / "templates" / "autofishing"


class FishingTimePolicy(IntEnum):
    """Numeric values match BetterGI's public SoloTask parameter contract."""

    ALL = 0
    DAYTIME = 1
    NIGHTTIME = 2
    DONT_CHANGE = 3


def parse_fishing_time_policy(value) -> FishingTimePolicy:
    if isinstance(value, FishingTimePolicy):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower().replace("_", "").replace("-", "")
        aliases = {
            "all": FishingTimePolicy.ALL,
            "全天": FishingTimePolicy.ALL,
            "day": FishingTimePolicy.DAYTIME,
            "daytime": FishingTimePolicy.DAYTIME,
            "白天": FishingTimePolicy.DAYTIME,
            "night": FishingTimePolicy.NIGHTTIME,
            "nighttime": FishingTimePolicy.NIGHTTIME,
            "夜晚": FishingTimePolicy.NIGHTTIME,
            "dontchange": FishingTimePolicy.DONT_CHANGE,
            "不调": FishingTimePolicy.DONT_CHANGE,
            "不调整": FishingTimePolicy.DONT_CHANGE,
        }
        if normalized in aliases:
            return aliases[normalized]
    try:
        return FishingTimePolicy(int(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无效的 fishingTimePolicy：{value}") from exc


def fishing_hours(policy: FishingTimePolicy, *, coop: bool = False) -> tuple[int | None, ...]:
    """Return BetterGI's 07:00/19:00 fishing phase schedule."""
    if coop or policy == FishingTimePolicy.DONT_CHANGE:
        return (None,)
    if policy == FishingTimePolicy.DAYTIME:
        return (7,)
    if policy == FishingTimePolicy.NIGHTTIME:
        return (19,)
    return (7, 19)


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
        target_catches: int = 0,
        timeout_s: float = 120,
        idle_timeout_s: float = 20,
        auto_throw_rod_enabled: bool = False,
        throw_rod_timeout_s: float = 15,
        fishing_time_policy: FishingTimePolicy | int | str = FishingTimePolicy.ALL,
        coop: bool = False,
        quit_on_finish: bool = True,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.target_catches = max(0, int(target_catches))
        self.timeout_s = max(5.0, float(timeout_s))
        self.idle_timeout_s = max(2.0, float(idle_timeout_s))
        self.auto_throw_rod_enabled = bool(auto_throw_rod_enabled)
        self.throw_rod_timeout_s = max(3.0, float(throw_rod_timeout_s))
        self.fishing_time_policy = parse_fishing_time_policy(fishing_time_policy)
        self.coop = bool(coop)
        self.quit_on_finish = bool(quit_on_finish)
        self.log = log
        self._templates: dict[str, Mat] = {}
        self._fish_predictor = None
        self._item_recognizer = None
        self._selected_bait: str | None = None

    def _template(self, name: str) -> Mat:
        if name not in self._templates:
            self._templates[name] = Mat.from_file(str(ASSETS / f"{name}.png"))
        return self._templates[name]

    def _find(self, region, name: str, roi=None):
        ro = RecognitionObject.template_match(self._template(name), *(roi or (None,) * 4))
        ro.threshold = 0.70
        return region.find(ro)

    def _in_fishing_mode(self, region=None) -> bool:
        region = region or self.ctx.capture_region()
        return self._find(region, "exit_fishing", (1780, 900, 140, 180)).is_exist()

    def _predict_fishpond(self, frame: np.ndarray, *, include_target: bool = False) -> FishPond:
        if self._fish_predictor is None:
            self._fish_predictor = FishPredictor()
        return self._fish_predictor.predict(frame, include_target=include_target)

    def _tap_ocr(self, *texts: str, timeout_s: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            hits = self.ctx.capture_region().find_multi(
                RecognitionObject.ocr(0, 0, 1920, 1080), limit=40
            )
            for hit in hits:
                normalized = hit.text.replace(" ", "")
                if any(text.replace(" ", "") in normalized for text in texts):
                    hit.click()
                    self.ctx.sleep(500)
                    return True
            self.ctx.sleep(200)
        return False

    def _enter_fishing_mode(self, cancelled=None) -> bool:
        if self._in_fishing_mode():
            return True
        self.log("[AutoFishing] 寻找鱼塘并进入钓鱼模式")
        self.ctx.input.move_camera_by(0, 480)
        self.ctx.sleep(400)
        for _ in range(24):
            if cancelled and cancelled():
                return False
            frame = self.ctx.capture_bgr()
            pond = self._predict_fishpond(frame)
            bounds = pond.bounds
            if bounds is None:
                self.ctx.input.move_camera_by(120, 0)
                self.ctx.sleep(250)
                continue
            center_x = bounds[0] + bounds[2] / 2
            if center_x < frame.shape[1] * 0.25:
                self.ctx.input.move_camera_by(-100, 0)
                self.ctx.sleep(250)
                continue
            if center_x > frame.shape[1] * 0.75:
                self.ctx.input.move_camera_by(100, 0)
                self.ctx.sleep(250)
                continue
            self.ctx.input.key_down("S")
            self.ctx.sleep(100)
            self.ctx.input.key_up("S")
            self.ctx.sleep(300)
            self.ctx.input.key_down("W")
            self.ctx.sleep(100)
            self.ctx.input.key_up("W")
            self.ctx.sleep(600)
            self.ctx.input.key_press("F")
            self.ctx.sleep(1000)
            self._tap_ocr("开始钓鱼", "确认", timeout_s=2.5)
            for _ in range(8):
                if self._in_fishing_mode():
                    return True
                self.ctx.sleep(350)
        self.log("[AutoFishing] 未能进入钓鱼模式")
        return False

    def _normalized_1080p(self, frame: np.ndarray) -> np.ndarray:
        scale = self.ctx.transform.scale
        width = round(REF_WIDTH * scale)
        left = max(0, round((frame.shape[1] - width) / 2))
        crop = frame[:round(REF_HEIGHT * scale), left:left + width]
        return cv2.resize(crop, (REF_WIDTH, REF_HEIGHT), interpolation=cv2.INTER_AREA)

    def _select_bait_icon(self, bait: str) -> bool:
        if self._item_recognizer is None:
            self._item_recognizer = ItemIconRecognizer()
        frame = self._normalized_1080p(self.ctx.capture_bgr())
        roi_x, roi_y, roi_w, roi_h = 538, 400, 864, 238
        grid = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        gray = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 20, 40)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width >= 100 and height and abs(width / height - 0.81) < 0.06:
                boxes.append((x, y, width, height))
        for x, y, width, height in sorted(boxes):
            card = grid[y:y + height, x:x + width]
            icon = card[:min(card.shape[0], card.shape[1]), :min(card.shape[0], card.shape[1])]
            match = self._item_recognizer.match(icon)
            if match.score >= 0.75 and match.name == bait:
                ref_x, ref_y = roi_x + x + width / 2, roi_y + y + height / 2
                dev_x, dev_y = self.ctx.transform.to_device(ref_x, ref_y, anchor="center")
                self.ctx.device.tap(
                    dev_x, dev_y,
                    image_width=self.ctx.transform.device_width,
                    image_height=self.ctx.transform.device_height,
                )
                self.ctx.sleep(600)
                self._tap_ocr("确认", timeout_s=1.5)
                return True
        return False

    def _choose_bait(self, pond: FishPond) -> str | None:
        bait = choose_bait(pond.fishes)
        if bait is None or bait == self._selected_bait:
            return bait
        region = self.ctx.capture_region()
        switch = self._find(region, "switch_bait", (960, 810, 960, 270))
        if not switch.is_exist():
            self.log("[AutoFishing] 未找到换饵按钮")
            return None
        self.log(f"[AutoFishing] 选择鱼饵 {bait}")
        switch.click()
        self.ctx.sleep(700)
        if not self._select_bait_icon(bait):
            self.ctx.input.key_press("ESCAPE")
            self.log(f"[AutoFishing] 未识别到鱼饵 {bait}")
            return None
        self._selected_bait = bait
        return bait

    def _cast_rod(self, cancelled=None) -> bool:
        frame = self.ctx.capture_bgr()
        pond = self._predict_fishpond(frame)
        bait = self._choose_bait(pond)
        if bait is None:
            return False
        self.ctx.input.move_camera_by(0, 480)
        self.ctx.sleep(300)
        self.ctx.input.attack_down()
        deadline = time.monotonic() + self.throw_rod_timeout_s
        try:
            while time.monotonic() < deadline:
                if cancelled and cancelled():
                    return False
                frame = self.ctx.capture_bgr()
                pond = self._predict_fishpond(frame, include_target=True)
                if pond.rod is None:
                    self.ctx.input.move_camera_by(0, 80)
                    self.ctx.sleep(180)
                    continue
                candidates = [fish for fish in pond.fishes if fish.kind.bait == bait]
                if not candidates:
                    self.ctx.sleep(180)
                    continue
                rod_center = (
                    pond.rod[0] + pond.rod[2] / 2,
                    pond.rod[1] + pond.rod[3] / 2,
                )
                fish = min(candidates, key=lambda item: math.hypot(
                    item.center[0] - rod_center[0], item.center[1] - rod_center[1]
                ))
                state = rod_state(pond.rod, fish, frame.shape[1], frame.shape[0])
                dx = fish.center[0] - rod_center[0]
                dy = fish.center[1] - rod_center[1]
                if state == 0:
                    self.log(f"[AutoFishing] 尝试钓取 {fish.kind.chinese_name}")
                    self.ctx.input.attack_up()
                    return True
                if state == 1:
                    self.ctx.input.move_camera_by(-dx / 1.5, -dy * 1.5)
                elif state == 2:
                    self.ctx.input.move_camera_by(dx / 1.5, dy * 1.5)
                else:
                    self.ctx.input.move_camera_by(dx * 0.35, dy * 0.35)
                self.ctx.sleep(max(80, min(500, int(math.hypot(dx, dy)))))
        finally:
            self.ctx.input.attack_up()
        self.log("[AutoFishing] 自动抛竿超时")
        return False

    def _quit_fishing_mode(self) -> None:
        if not self._in_fishing_mode():
            return
        self.ctx.input.key_press("ESCAPE")
        self.ctx.sleep(600)
        self._tap_ocr("确认", "退出", timeout_s=2)

    def _set_time(self, hour: int) -> bool:
        from ..engine.genshin_api import GenshinApi

        self.log(f"[AutoFishing] 调整游戏时间到 {hour:02d}:00")
        return bool(GenshinApi(self.ctx).setTime(hour, 0))

    def _run_fishing_round(
        self,
        deadline: float,
        target_remaining: int,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[bool, int]:
        last_activity = time.monotonic()
        last_press = 0.0
        last_bar_seen = 0.0
        holding = False
        catches = 0
        try:
            if self.auto_throw_rod_enabled:
                if not self._enter_fishing_mode(cancelled):
                    self.log("[AutoFishing] 当前时段没有可进入的鱼塘")
                    return True, 0
                if not self._cast_rod(cancelled):
                    self.log("[AutoFishing] 当前时段没有可钓目标")
                    return True, 0
            last_activity = time.monotonic()
            while time.monotonic() < deadline:
                if cancelled and cancelled():
                    self.log("[AutoFishing] 已取消")
                    return False, catches
                region = self.ctx.capture_region()
                frame = region.bgr
                now = time.monotonic()

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
                    if target_remaining and catches >= target_remaining:
                        return True, catches
                    if self.auto_throw_rod_enabled:
                        if not self._cast_rod(cancelled):
                            self.log("[AutoFishing] 当前时段鱼塘已清空")
                            return True, catches
                        last_activity = time.monotonic()
                    elif now - last_press > 0.8:
                        self.ctx.input.key_press("SPACE")
                        last_press = now

                if now - last_activity >= self.idle_timeout_s:
                    self.log("[AutoFishing] 未检测到钓鱼界面，结束任务")
                    return True, catches
                self.ctx.sleep(120)
            self.log("[AutoFishing] 超时退出")
            return False, catches
        finally:
            if holding:
                self.ctx.input.attack_up()

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        deadline = time.monotonic() + self.timeout_s
        catches = 0
        target = f"目标 {self.target_catches} 条" if self.target_catches else "清空鱼塘"
        phases = fishing_hours(self.fishing_time_policy, coop=self.coop)
        self.log(
            f"[AutoFishing] 自动钓鱼启动（{target}，"
            f"时间策略 {self.fishing_time_policy.name}）"
        )
        try:
            for phase_index, hour in enumerate(phases, start=1):
                if time.monotonic() >= deadline or (cancelled and cancelled()):
                    return False
                if phase_index > 1:
                    self._quit_fishing_mode()
                    self.ctx.sleep(600)
                if hour is not None and not self._set_time(hour):
                    # BetterGI's SetTimeTask logs and continues fishing when
                    # changing time fails (for example during a UI transition).
                    self.log("[AutoFishing] 调整时间失败，继续使用当前游戏时间")
                remaining = (
                    max(0, self.target_catches - catches)
                    if self.target_catches
                    else 0
                )
                success, round_catches = self._run_fishing_round(
                    deadline, remaining, cancelled
                )
                catches += round_catches
                if not success:
                    return False
                if self.target_catches and catches >= self.target_catches:
                    return True
            return self.target_catches == 0 or catches >= self.target_catches
        finally:
            if self.auto_throw_rod_enabled and self.quit_on_finish:
                self._quit_fishing_mode()
            self.ctx.input.release_all()
