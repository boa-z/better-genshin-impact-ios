"""Automatic game-door and login-popup handling from BetterGI GameLoading."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from ..engine.recognition import Mat, RecognitionObject
from ..vision.game_ui import is_main_ui

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "templates" / "game_loading"


class GameLoadingTrigger:
    name = "GameLoading"

    def __init__(
        self,
        ctx,
        *,
        timeout_s: float = 300.0,
        interval_s: float = 2.0,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
        main_ui_detector=is_main_ui,
    ):
        self.ctx = ctx
        self.enabled = True
        self.log = log
        self._clock = clock
        self._main_ui_detector = main_ui_detector
        self._started_at = clock()
        self._last_action_at = float("-inf")
        self._last_check_at = float("-inf")
        self._interval_s = max(0.1, float(interval_s))
        self._timeout_s = max(1.0, float(timeout_s))
        self._templates = {
            name: Mat.from_file(str(ASSETS / f"{name}.png"))
            for name in ("enter_game", "choose_enter_game", "welkin_moon_logo", "girl_moon")
        }

    def _template(self, name: str, roi, threshold: float = 0.75):
        ro = RecognitionObject.template_match(self._templates[name], *roi)
        ro.threshold = threshold
        return ro

    def on_frame(self, region) -> None:
        if not self.enabled:
            return
        now = self._clock()
        if now - self._started_at >= self._timeout_s:
            self.log("[GameLoading] 自动开门超时，停止处理")
            self._finish()
            return
        if self._main_ui_detector(self.ctx, region.bgr):
            self.log("[GameLoading] 已进入游戏主界面")
            self._finish()
            return
        if now - self._last_check_at < self._interval_s:
            return
        self._last_check_at = now

        # Official-server account conflict / account-selection confirmation.
        choose = region.find(self._template(
            "choose_enter_game", (685, 595, 549, 64), threshold=0.72,
        ))
        if choose.is_exist():
            choose.click()
            self._acted(now, "点击账号确认进入")
            return

        enter = region.find(self._template(
            "enter_game", (640, 540, 640, 540), threshold=0.70,
        ))
        if enter.is_exist():
            enter.click()
            self._acted(now, "点击进入游戏")
            return

        # Age/guardian notices and reward popups use OCR because their mobile
        # layout differs from the PC assets.  Only known confirmation/reward
        # words are actionable; arbitrary loading text is never clicked.
        hits = region.find_multi(RecognitionObject.ocr_this(), limit=40)
        texts = ["".join(str(hit.text).split()) for hit in hits]
        age_notice = any("适龄" in text or "监护" in text for text in texts)
        if age_notice:
            confirm = next((
                hit for hit, text in zip(hits, texts)
                if any(word in text for word in ("确认", "同意", "知道了"))
            ), None)
            if confirm is not None:
                confirm.click()
                self._acted(now, "关闭适龄提示")
                return

        lower_half = (0, 540, 1920, 540)
        welkin = region.find(self._template("welkin_moon_logo", lower_half, 0.70))
        girl = region.find(self._template("girl_moon", lower_half, 0.70))
        if welkin.is_exist() or girl.is_exist() or any("空月祝福" in text for text in texts):
            self.ctx.input.click_ref(960, 820)
            self._acted(now, "关闭空月祝福")
            return
        if any("原石" in text for text in texts):
            self.ctx.input.click_ref(960, 820)
            self._acted(now, "关闭原石奖励弹窗")

    def _acted(self, now: float, message: str) -> None:
        self._last_action_at = now
        self.log(f"[GameLoading] {message}")

    def _finish(self) -> None:
        self.enabled = False
        loop = getattr(self.ctx, "_trigger_loop", None)
        if loop is not None and loop.get(self.name) is self:
            loop.remove(self.name)

    def close(self) -> None:
        self.enabled = False
