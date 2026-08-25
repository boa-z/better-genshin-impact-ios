"""AutoFishing realtime trigger migrated from BetterGI.

The standalone :class:`AutoFishingTask` can enter a fishing pond and manage a
whole session.  BetterGI also exposes a lightweight trigger that takes over
after the player has already entered the fishing UI.  This module implements
that latter contract for iOS: it only reacts to the fishing HUD, controls the
fish bar, and never moves the camera or starts a fishing session by itself.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from ..engine.context import GameContext
from ..engine.recognition import ImageRegion, Mat, RecognitionObject
from ..tasks.auto_fishing import (
    ASSETS,
    fish_bar_action,
    get_fish_bar_rects,
    match_fish_bite_words,
)


class AutoFishingTrigger:
    """Semi-automatic fishing controller for the realtime trigger loop."""

    name = "AutoFish"

    def __init__(
        self,
        ctx: GameContext,
        *,
        log: Callable[[str], None] = print,
        action_debounce_s: float = 0.45,
    ):
        self.ctx = ctx
        self.log = log
        self.enabled = True
        self.is_exclusive = False
        self._holding = False
        self._last_action_at = 0.0
        self._last_bar_at = 0.0
        self._action_debounce_s = max(0.1, float(action_debounce_s))
        self._templates: dict[str, Mat] = {}

    def _template(self, name: str) -> Mat:
        if name not in self._templates:
            self._templates[name] = Mat.from_file(str(ASSETS / f"{name}.png"))
        return self._templates[name]

    def _find(self, region: ImageRegion, name: str, roi) -> object:
        recognition = RecognitionObject.template_match(self._template(name), *roi)
        recognition.threshold = 0.70
        return region.find(recognition)

    @staticmethod
    def _exists(result: object) -> bool:
        method = getattr(result, "is_exist", None)
        return bool(method()) if callable(method) else bool(result)

    def _in_fishing_mode(self, region: ImageRegion) -> bool:
        return self._exists(self._find(region, "exit_fishing", (1780, 900, 140, 180)))

    def _release_bar(self) -> None:
        if self._holding:
            self.ctx.input.attack_up()
            self._holding = False

    def _reset_session(self) -> None:
        self._release_bar()
        self.is_exclusive = False
        self._last_bar_at = 0.0

    def _press_attack(self, now: float) -> None:
        if now - self._last_action_at < self._action_debounce_s:
            return
        self.ctx.input.attack()
        self._last_action_at = now
        self.log("[AutoFishing] 自动提竿")

    def on_frame(self, region: ImageRegion) -> None:
        in_fishing = self._in_fishing_mode(region)
        if not in_fishing:
            if self.is_exclusive:
                self.log("[AutoFishing] 退出钓鱼界面")
                self._reset_session()
            return

        if not self.is_exclusive:
            self.is_exclusive = True
            self._last_bar_at = 0.0
            self.log("[AutoFishing] 半自动钓鱼启动")

        frame = region.bgr
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return
        now = time.monotonic()
        bar = get_fish_bar_rects(frame[: max(1, frame.shape[0] // 2)])
        action = fish_bar_action(bar)
        if action == "hold":
            self._last_bar_at = now
            if not self._holding:
                self.ctx.input.attack_down()
                self._holding = True
        elif action == "release":
            self._last_bar_at = now
            self._release_bar()
        elif bar:
            self._last_bar_at = now

        bite = match_fish_bite_words(
            frame,
            (frame.shape[1] // 3, 0, frame.shape[1] // 3, frame.shape[0] // 2),
        )
        lift = self._exists(self._find(region, "lift_rod", (1440, 400, 480, 540)))
        if bite or lift:
            self._press_attack(now)
            return

        wait_bite = self._exists(self._find(region, "wait_bite", (1440, 270, 480, 540)))
        space = self._exists(self._find(region, "Space", (960, 540, 960, 540)))
        if not bar and (wait_bite or space):
            if now - self._last_action_at >= 0.8:
                self.ctx.input.key_press("SPACE")
                self._last_action_at = now

    def close(self) -> None:
        self._reset_session()
        self.enabled = False

