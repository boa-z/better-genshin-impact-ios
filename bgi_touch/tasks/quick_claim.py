"""One-key reward claiming adapted from BetterGI's template loop."""

from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Callable

from ..engine.context import GameContext
from ..engine.recognition import Mat, RecognitionObject
from .common_jobs import exclusive_realtime_triggers


ASSETS = Path(__file__).resolve().parents[2] / "assets" / "templates" / "quick_claim"

CLICK_ONCE_MODE = "点按一次"
HOLD_MODE = "按住持续"
_MODE_ALIASES = {
    CLICK_ONCE_MODE: CLICK_ONCE_MODE,
    "click": CLICK_ONCE_MODE,
    "clickonce": CLICK_ONCE_MODE,
    "点按": CLICK_ONCE_MODE,
    HOLD_MODE: HOLD_MODE,
    "hold": HOLD_MODE,
    "continuous": HOLD_MODE,
    "按住": HOLD_MODE,
}


def _normalize_mode(value: object) -> str:
    text = str(value or CLICK_ONCE_MODE).strip()
    normalized = _MODE_ALIASES.get(text.casefold(), _MODE_ALIASES.get(text))
    if normalized is None:
        raise ValueError(f"不支持的一键领取奖励模式：{value}")
    return normalized


class QuickClaimRewardTask:
    def __init__(
        self,
        ctx: GameContext,
        *,
        max_clicks: int = 30,
        scroll_down: bool = False,
        max_scrolls: int = 3,
        scroll_amount: int = 1,
        timeout_s: float = 30.0,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.max_clicks = max(1, min(100, int(max_clicks)))
        self.scroll_down = bool(scroll_down)
        self.max_scrolls = max(0, min(20, int(max_scrolls)))
        self.scroll_amount = max(1, min(20, abs(int(scroll_amount))))
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
        for index in range(self.scroll_amount):
            self.ctx.device.swipe(
                t.device_width * 0.72,
                t.device_height * 0.78,
                t.device_width * 0.72,
                t.device_height * 0.35,
                duration_ms=350,
                image_width=t.device_width,
                image_height=t.device_height,
            )
            if index + 1 < self.scroll_amount:
                self.ctx.sleep(120)
        self.ctx.sleep(300)

    @staticmethod
    def _is_cancelled(cancelled: Callable[[], bool] | None) -> bool:
        if cancelled is None:
            return False
        try:
            return bool(cancelled())
        except Exception:
            # A disappearing JS/WebUI owner must fail closed so a lost hold
            # event cannot leave the reward worker running indefinitely.
            return True

    def _claim_candidate(self, candidate, clicks: int) -> int:
        name, region = candidate
        region.click()
        clicks += 1
        self.log(f"[QuickClaimReward] 点击{name}图标（{clicks}）")
        self._dismiss_continue()
        self.ctx.sleep(180)
        return clicks

    def run(self, cancelled: Callable[[], bool] | None = None) -> int:
        with exclusive_realtime_triggers(self.ctx):
            deadline = time.monotonic() + self.timeout_s
            clicks = 0
            scrolls = 0
            while clicks < self.max_clicks and time.monotonic() < deadline:
                if self._is_cancelled(cancelled):
                    break
                candidates = self._find_candidates(self.ctx.capture_region())
                if candidates:
                    clicks = self._claim_candidate(candidates[0], clicks)
                    continue
                if self.scroll_down and scrolls < self.max_scrolls:
                    self._scroll()
                    scrolls += 1
                    continue
                break
            self.log(f"[QuickClaimReward] 本次领取完成，共点击 {clicks} 个图标")
            return clicks

    def run_while_holding(
        self,
        cancelled: Callable[[], bool] | None = None,
    ) -> int:
        """Keep claiming until the owning hotkey is released.

        BetterGI's hold mode deliberately keeps polling an empty page: the
        user may be holding the key while scrolling through a long reward
        list.  The task therefore must not reuse ``run()``'s finite
        no-candidate exit condition.  The whole loop stays inside one
        realtime-exclusive scope so AutoPick/AutoSkip cannot regain the
        screenshot/input channel between two reward clicks.
        """
        with exclusive_realtime_triggers(self.ctx):
            clicks = 0
            scrolls = 0
            last_empty_log = float("-inf")
            while not self._is_cancelled(cancelled):
                candidates = self._find_candidates(self.ctx.capture_region())
                if candidates:
                    clicks = self._claim_candidate(candidates[0], clicks)
                    continue
                if self.scroll_down and scrolls < self.max_scrolls:
                    self._scroll()
                    scrolls += 1
                    continue

                now = time.monotonic()
                if now - last_empty_log >= 2.0:
                    self.log("[QuickClaimReward] 未找到领取图标，持续等待")
                    last_empty_log = now
                self.ctx.sleep(260)
            self.log(f"[QuickClaimReward] 持续领取停止，共点击 {clicks} 个图标")
            return clicks


class OneKeyClaimRewardTask:
    """DeviceHub-safe equivalent of BetterGI's global reward hotkey.

    BetterGI owns this task from its global hotkey service rather than from a
    JS ``dispatcher`` call.  On iOS the WebUI/KeyMouseHook bridge calls
    ``KeyDown``/``KeyUp``; the worker keeps the same click-once and hold modes
    while the actual image work is delegated to ``QuickClaimRewardTask``.
    """

    ClickOnceMode = CLICK_ONCE_MODE
    HoldMode = HOLD_MODE

    def __init__(
        self,
        ctx: GameContext,
        *,
        mode: str = CLICK_ONCE_MODE,
        scroll_down_enabled: bool = False,
        scroll_down_amount: int = 2,
        max_clicks: int = 30,
        max_scrolls: int = 3,
        timeout_s: float = 30.0,
        enabled: bool = True,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.mode = _normalize_mode(mode)
        self.scroll_down_enabled = bool(scroll_down_enabled)
        self.scroll_down_amount = max(1, abs(int(scroll_down_amount)))
        self.max_clicks = max(1, min(100, int(max_clicks)))
        self.max_scrolls = max(0, min(20, int(max_scrolls)))
        self.timeout_s = max(2.0, float(timeout_s))
        self.enabled = bool(enabled)
        self.log = log
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._key_down = False
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        with self._lock:
            return bool(self._worker and self._worker.is_alive())

    def _cancelled(self) -> bool:
        with self._lock:
            released = not self._key_down
        return self._stop.is_set() or (
            self.mode == HOLD_MODE and released
        )

    def key_down(self) -> bool:
        with self._lock:
            if self._key_down or not self.enabled:
                return False
            self._key_down = True
            if self._worker and self._worker.is_alive():
                return False
            self._stop.clear()
            self._worker = threading.Thread(
                target=self._run,
                daemon=True,
                name="one-key-claim-reward",
            )
            self._worker.start()
        return True

    def key_up(self) -> bool:
        with self._lock:
            was_down = self._key_down
            self._key_down = False
        if self.mode == HOLD_MODE:
            self._stop.set()
        return was_down

    def stop(self, *, wait: bool = True, timeout: float = 2.0) -> None:
        with self._lock:
            self._key_down = False
            worker = self._worker
        self._stop.set()
        if wait and worker and worker is not threading.current_thread():
            worker.join(timeout=max(0.0, float(timeout)))

    def wait(self, timeout: float | None = None) -> bool:
        with self._lock:
            worker = self._worker
        if worker:
            worker.join(timeout)
        return not self.running

    def _run(self) -> None:
        try:
            task = QuickClaimRewardTask(
                self.ctx,
                max_clicks=self.max_clicks,
                # Upstream only enables scrolling in hold mode.  The amount
                # is translated to the iOS swipe count by QuickClaimReward.
                scroll_down=self.mode == HOLD_MODE and self.scroll_down_enabled,
                max_scrolls=self.max_scrolls,
                scroll_amount=self.scroll_down_amount,
                timeout_s=self.timeout_s,
                log=self.log,
            )
            if self.mode == HOLD_MODE:
                task.run_while_holding(cancelled=self._cancelled)
            else:
                task.run(cancelled=self._cancelled)
        except Exception as error:
            self.log(f"[OneKeyClaimReward] 执行失败：{error}")
        finally:
            with self._lock:
                if self._worker is threading.current_thread():
                    self._worker = None

    KeyDown = key_down
    KeyUp = key_up
    Stop = stop


__all__ = [
    "CLICK_ONCE_MODE", "HOLD_MODE", "OneKeyClaimRewardTask",
    "QuickClaimRewardTask",
]
