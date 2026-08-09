"""Pixel-driven AutoMusicGame implementation for the mobile touch layout."""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np

from ..engine.context import GameContext

DEFAULT_LANES = (417, 628, 844, 1061, 1277, 1493)
DEFAULT_KEYS = ("A", "S", "D", "J", "K", "L")
DEFAULT_LANE_Y = 921


def detect_music_lanes(
    bgr: np.ndarray,
    lane_x: Sequence[int] = DEFAULT_LANES,
    lane_y: int = DEFAULT_LANE_Y,
    *,
    threshold: int = 220,
    radius: int = 3,
) -> tuple[bool, ...]:
    """Return lanes whose timing marker is dark, matching BetterGI's B test."""
    if bgr is None or bgr.size == 0:
        return tuple(False for _ in lane_x)
    output = []
    for x in lane_x:
        x0, x1 = max(0, x - radius), min(bgr.shape[1], x + radius + 1)
        y0, y1 = max(0, lane_y - radius), min(bgr.shape[0], lane_y + radius + 1)
        if x1 <= x0 or y1 <= y0:
            output.append(False)
            continue
        # Keep the original blue-channel test, but use a small median patch so
        # JPEG noise and a one-pixel cursor do not generate duplicate taps.
        output.append(float(np.median(bgr[y0:y1, x0:x1, 0])) < threshold)
    return tuple(output)


class AutoMusicGameTask:
    def __init__(
        self,
        ctx: GameContext,
        *,
        lane_x: Sequence[int] = DEFAULT_LANES,
        lane_y: int = DEFAULT_LANE_Y,
        keys: Sequence[str] = DEFAULT_KEYS,
        threshold: int = 220,
        poll_interval_ms: int = 35,
        timeout_s: float = 900.0,
        idle_timeout_s: float = 8.0,
        log: Callable[[str], None] = print,
    ):
        if len(lane_x) != len(keys):
            raise ValueError("AutoMusicGame 的 laneX 与 keys 数量必须一致")
        self.ctx = ctx
        self.lane_x = tuple(int(x) for x in lane_x)
        self.lane_y = int(lane_y)
        self.keys = tuple(str(key) for key in keys)
        self.threshold = max(1, min(255, int(threshold)))
        self.poll_interval_s = max(0.02, int(poll_interval_ms) / 1000)
        self.timeout_s = max(1.0, float(timeout_s))
        self.idle_timeout_s = max(1.0, float(idle_timeout_s))
        self.log = log

    def run(
        self,
        cancelled: Callable[[], bool] | None = None,
        frame_observer: Callable[[object], bool] | None = None,
    ) -> bool:
        deadline = time.monotonic() + self.timeout_s
        last_cue = time.monotonic()
        held: set[str] = set()
        seen_cue = False
        self.log("[AutoMusicGame] 自动演奏启动")
        try:
            while time.monotonic() < deadline:
                if cancelled and cancelled():
                    return False
                region = self.ctx.capture_region()
                if frame_observer and frame_observer(region):
                    return True
                frame = region.bgr
                lane_points = [
                    self.ctx.transform.to_device(x, self.lane_y)
                    for x in self.lane_x
                ]
                dark = detect_music_lanes(
                    frame,
                    tuple(round(x) for x, _ in lane_points),
                    round(lane_points[0][1]) if lane_points else self.lane_y,
                    threshold=self.threshold,
                )
                desired = {key for key, is_dark in zip(self.keys, dark) if is_dark}
                for key in sorted(desired - held):
                    self.ctx.input.key_down(key)
                for key in sorted(held - desired):
                    self.ctx.input.key_up(key)
                if desired:
                    seen_cue = True
                    last_cue = time.monotonic()
                held = desired
                if seen_cue and time.monotonic() - last_cue >= self.idle_timeout_s:
                    return True
                self.ctx.sleep(self.poll_interval_s * 1000)
            return seen_cue
        finally:
            for key in held:
                self.ctx.input.key_up(key)
            self.ctx.input.release_all()
            self.log("[AutoMusicGame] 自动演奏结束")
