"""Automatically follow the currently tracked quest marker."""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np

from ..engine.context import GameContext
from ..engine.recognition import RecognitionObject
from ..vision.game_ui import is_main_ui


_DISTANCE_RE = re.compile(r"(?<!\d)(\d{1,5})\s*[m米](?!\w)", re.IGNORECASE)


def extract_mission_distance(texts: Iterable[str]) -> int | None:
    values = []
    for text in texts:
        for match in _DISTANCE_RE.finditer(str(text or "").replace("Ｍ", "m")):
            values.append(int(match.group(1)))
    return min(values) if values else None


@dataclass(frozen=True)
class TrackMarker:
    x: float
    y: float
    width: float
    height: float
    score: float


def find_blue_track_marker(frame: np.ndarray) -> TrackMarker | None:
    """Find BetterGI's cyan quest marker without a resolution-bound template."""
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        return None
    height, width = frame.shape[:2]
    left, right = round(width * 300 / 1920), round(width * 1620 / 1920)
    crop = frame[:, left:right]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((90, 70, 75)), np.array((132, 255, 255)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scale = height / 1080
    candidates: list[TrackMarker] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if not (8 * scale <= w <= 75 * scale and 8 * scale <= h <= 75 * scale):
            continue
        aspect = w / max(1, h)
        fill = area / max(1, w * h)
        if not (0.45 <= aspect <= 2.1 and fill >= 0.12):
            continue
        # Prefer compact bright markers over thin blue scenery fragments.
        score = fill * min(w, h) - abs(1.0 - aspect) * 2
        candidates.append(TrackMarker(
            left + x + w / 2,
            y + h / 2,
            float(w),
            float(h),
            float(score),
        ))
    return max(candidates, key=lambda marker: marker.score, default=None)


class AutoTrackTask:
    """Track the selected quest with V and steer toward its blue marker."""

    def __init__(
        self,
        ctx: GameContext,
        *,
        timeout_s: float = 120.0,
        far_distance_m: int = 150,
        arrival_distance_m: int = 3,
        teleport_when_far: bool = True,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.timeout_s = max(5.0, float(timeout_s))
        self.far_distance_m = max(20, int(far_distance_m))
        self.arrival_distance_m = max(1, int(arrival_distance_m))
        self.teleport_when_far = bool(teleport_when_far)
        self.log = log
        self.distance_roi = RecognitionObject.ocr(0, 175, 560, 240)

    def _distance(self, region) -> int | None:
        hits = region.find_multi(self.distance_roi, limit=20)
        return extract_mission_distance(hit.text for hit in hits)

    @contextmanager
    def _exclusive_triggers(self):
        """Prevent realtime triggers from acting on stale tracking frames."""
        loop = getattr(self.ctx, "_trigger_loop", None)
        if loop is None or not getattr(loop, "active", False):
            yield
            return
        state = loop.pause()
        try:
            yield
        finally:
            loop.resume(state)

    def _teleport_near_quest(
        self,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        """Open the tracked quest map and choose its nearest visible anchor."""
        if cancelled and cancelled():
            return False
        self.log("[AutoTrack] 任务较远，打开任务页寻找附近传送点")
        self.ctx.input.key_press("J")
        self.ctx.sleep(900)
        # Upstream toggles tracking twice at the quest page's lower-right
        # action, which opens a map centered on the tracked objective.
        self.ctx.input.click_ref(1680, 1015)
        self.ctx.sleep(250)
        self.ctx.input.click_ref(1680, 1015)
        self.ctx.sleep(1500)
        try:
            from ..pathing.tp import TpTask

            teleport = TpTask(self.ctx, log=self.log)
            if not teleport._tap_anchor_icon_near_center():
                self.log("[AutoTrack] 任务地图中心附近未找到已解锁传送点")
                self.ctx.input.key_press("ESCAPE")
                return False
            if not teleport._find_and_tap_confirm():
                self.log("[AutoTrack] 未出现传送确认面板")
                self.ctx.input.key_press("ESCAPE")
                return False
            teleport._wait_for_teleport_completion()
            return True
        except Exception as error:
            self.log(f"[AutoTrack] 任务传送失败，继续尝试直接追踪：{error}")
            self.ctx.input.key_press("ESCAPE")
            return False

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        with self._exclusive_triggers():
            return self._run(cancelled)

    def _run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        deadline = time.monotonic() + self.timeout_s
        forward = False
        last_distance: int | None = None
        missing = 0
        try:
            initial = self.ctx.capture_region()
            if not is_main_ui(self.ctx, initial.bgr):
                self.log("[AutoTrack] 当前不在主界面")
                return False
            distance = self._distance(initial)
            if distance is not None:
                self.log(f"[AutoTrack] 当前任务距离 {distance}m")
            if (
                distance is not None
                and distance >= self.far_distance_m
                and self.teleport_when_far
            ):
                self._teleport_near_quest(cancelled)

            self.ctx.input.key_press("V")
            self.ctx.sleep(1200)
            while time.monotonic() < deadline:
                if cancelled and cancelled():
                    self.log("[AutoTrack] 已取消")
                    return False
                region = self.ctx.capture_region()
                distance = self._distance(region)
                if distance is not None:
                    last_distance = distance
                    if distance <= self.arrival_distance_m:
                        self.log(f"[AutoTrack] 到达任务目标（{distance}m）")
                        return True

                marker = find_blue_track_marker(region.bgr)
                if marker is None:
                    missing += 1
                    if last_distance is not None and last_distance <= 10 and missing >= 2:
                        self.log("[AutoTrack] 近距离任务标记消失，视为到达")
                        return True
                    if missing >= 6:
                        self.log("[AutoTrack] 连续未找到蓝色任务标记")
                        return False
                    self.ctx.sleep(350)
                    continue
                missing = 0
                height, width = region.bgr.shape[:2]
                error_x = marker.x - width / 2
                target_y = height * 0.38
                error_y = marker.y - target_y
                scale = max(0.5, height / 1080)
                turn_ref = max(-140.0, min(140.0, error_x / scale * 0.16))
                tilt_ref = max(-70.0, min(70.0, error_y / scale * 0.08))
                if abs(turn_ref) >= 2 or abs(tilt_ref) >= 2:
                    self.ctx.input.move_camera_by(turn_ref, tilt_ref)

                aligned = abs(error_x) <= width * 0.12
                if aligned and not forward:
                    self.ctx.input.key_down("W")
                    forward = True
                elif not aligned and abs(error_x) > width * 0.28 and forward:
                    self.ctx.input.key_up("W")
                    forward = False
                self.ctx.sleep(250)
            self.log("[AutoTrack] 追踪超时")
            return False
        finally:
            if forward:
                self.ctx.input.key_up("W")
            self.ctx.input.release_all()
