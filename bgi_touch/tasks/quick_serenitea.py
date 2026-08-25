"""BetterGI QuickSereniteaPot hotkey flow adapted to touch input."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from ..engine.context import GameContext
from ..engine.genshin_api import GenshinApi
from ..engine.recognition import Mat, RecognitionObject
from .common_jobs import exclusive_realtime_triggers


ASSETS = Path(__file__).resolve().parents[2] / "assets" / "templates" / "quick_serenitea"


class QuickSereniteaPotTask:
    """Deploy the Serenitea Pot gadget and enter/leave it from the main UI."""

    def __init__(
        self,
        ctx: GameContext,
        *,
        timeout_s: float = 35.0,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.timeout_s = max(10.0, float(timeout_s))
        self.log = log
        self._pot = Mat.from_file(str(ASSETS / "SereniteaPotIcon.png"))

    @staticmethod
    def _clean(text: str) -> str:
        return str(text).replace(" ", "").replace("\u3000", "")

    def _find_text(self, *words: str, roi=(0, 0, 1920, 1080)):
        region = self.ctx.capture_region()
        hits = region.find_multi(RecognitionObject.ocr(*roi), limit=50)
        for hit in hits:
            text = self._clean(hit.text)
            if any(self._clean(word) in text for word in words):
                return hit
        return None

    def _find_pot(self):
        region = self.ctx.capture_region()
        ro = RecognitionObject.template_match(self._pot, 100, 100, 1190, 860)
        ro.threshold = 0.72
        return region.find(ro)

    def _wait_for_pot(self, deadline: float):
        while time.monotonic() < deadline:
            pot = self._find_pot()
            if pot.is_exist():
                return pot
            self.ctx.sleep(300)
        return None

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        with exclusive_realtime_triggers(self.ctx):
            return self._run_locked(cancelled)

    def _run_locked(self, cancelled: Callable[[], bool] | None = None) -> bool:
        deadline = time.monotonic() + self.timeout_s
        api = GenshinApi(self.ctx, log=self.log)
        if not api.returnMainUi():
            return False
        if cancelled and cancelled():
            return False

        self.log("[QuickSereniteaPot] 打开背包并选择小道具页")
        self.ctx.input.key_press("B")
        self.ctx.sleep(900)
        # BetterGI uses this reference point for the gadget category.
        self.ctx.input.click_ref(1050, 50)
        self.ctx.sleep(350)

        pot = self._wait_for_pot(min(deadline, time.monotonic() + 8.0))
        if pot is None:
            self.ctx.input.key_press("ESCAPE")
            self.log("[QuickSereniteaPot] 背包中未识别到尘歌壶")
            return False
        pot.click()
        self.ctx.sleep(300)

        place = self._find_text("放置", "Place", roi=(1100, 650, 820, 430))
        if place is not None:
            place.click()
        else:
            self.ctx.input.click_ref(1695, 1015)
        self.ctx.sleep(1000)

        while time.monotonic() < deadline:
            if cancelled and cancelled():
                return False
            option = self._find_text("进入尘歌壶", "离开尘歌壶", "进入", "离开",
                                     roi=(900, 350, 1020, 700))
            if option is not None:
                action = "进入" if "进入" in option.text else "离开"
                self.log(f"[QuickSereniteaPot] 识别到{action}尘歌壶")
                self.ctx.input.key_press("F")
                self.ctx.sleep(250)
                # Co-op builds show one extra enter/leave confirmation.
                confirm = self._find_text("进入", "离开", "确认", "Confirm",
                                          roi=(800, 500, 1120, 550))
                if confirm is not None:
                    confirm.click()
                return True
            self.ctx.sleep(300)
        self.log("[QuickSereniteaPot] 部署后未识别到进入/离开交互")
        return False
