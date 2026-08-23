"""One-key reward claiming adapted from BetterGI's template loop."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from ..engine.context import GameContext
from ..engine.recognition import Mat, RecognitionObject


ASSETS = Path(__file__).resolve().parents[2] / "assets" / "templates" / "quick_claim"


class QuickClaimRewardTask:
    def __init__(
        self,
        ctx: GameContext,
        *,
        max_clicks: int = 30,
        scroll_down: bool = False,
        max_scrolls: int = 3,
        timeout_s: float = 30.0,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.max_clicks = max(1, min(100, int(max_clicks)))
        self.scroll_down = bool(scroll_down)
        self.max_scrolls = max(0, min(20, int(max_scrolls)))
        self.timeout_s = max(2.0, float(timeout_s))
        self.log = log
        self._templates = {
            name: Mat.from_file(str(ASSETS / f"{name}.png"))
            for name in ("claim_text", "claim_gift", "click_blank_continue")
        }

    def _find_multi(self, region, name: str, threshold: float):
        ro = RecognitionObject.template_match(self._templates[name])
        ro.threshold = threshold
        return region.find_multi(ro, limit=20)

    def _find_candidates(self, region):
        candidates = [
            *(('领取', hit) for hit in self._find_multi(region, "claim_text", 0.86)),
            *(('礼物领取', hit) for hit in self._find_multi(region, "claim_gift", 0.86)),
        ]
        return sorted(candidates, key=lambda item: (item[1].dy, item[1].dx))

    def _dismiss_continue(self) -> None:
        for _ in range(3):
            self.ctx.sleep(160)
            region = self.ctx.capture_region()
            ro = RecognitionObject.template_match(self._templates["click_blank_continue"])
            ro.threshold = 0.82
            if region.find(ro).is_exist():
                self.ctx.input.key_press("ESCAPE")
                self.ctx.sleep(220)
                return

    def _scroll(self) -> None:
        t = self.ctx.transform
        self.ctx.device.swipe(
            t.device_width * 0.72,
            t.device_height * 0.78,
            t.device_width * 0.72,
            t.device_height * 0.35,
            duration_ms=350,
            image_width=t.device_width,
            image_height=t.device_height,
        )
        self.ctx.sleep(300)

    def run(self, cancelled: Callable[[], bool] | None = None) -> int:
        deadline = time.monotonic() + self.timeout_s
        clicks = 0
        scrolls = 0
        while clicks < self.max_clicks and time.monotonic() < deadline:
            if cancelled and cancelled():
                break
            candidates = self._find_candidates(self.ctx.capture_region())
            if candidates:
                name, candidate = candidates[0]
                candidate.click()
                clicks += 1
                self.log(f"[QuickClaimReward] 点击{name}图标（{clicks}）")
                self._dismiss_continue()
                self.ctx.sleep(180)
                continue
            if self.scroll_down and scrolls < self.max_scrolls:
                self._scroll()
                scrolls += 1
                continue
            break
        self.log(f"[QuickClaimReward] 本次领取完成，共点击 {clicks} 个图标")
        return clicks

