"""BetterGI QuickBuy hotkey flow adapted to iOS touch sliders."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..engine.context import GameContext
from ..engine.recognition import Mat, RecognitionObject
from .common_jobs import exclusive_realtime_triggers

ASSET = Path(__file__).resolve().parents[2] / "assets" / "quickbuy" / "SereniteaPotCoin.png"


class QuickBuyTask:
    def __init__(
        self,
        ctx: GameContext,
        *,
        serenitea: bool | None = None,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.serenitea = serenitea
        self.log = log
        self._coin = Mat.from_file(str(ASSET)) if ASSET.is_file() else None

    def _is_serenitea_shop(self) -> bool:
        if self.serenitea is not None:
            return self.serenitea
        region = self.ctx.capture_region()
        if self._coin is not None:
            recognition = RecognitionObject.template_match(
                self._coin, 1610, 28, 160, 45
            )
            recognition.threshold = 0.76
            if region.find(recognition).is_exist():
                return True
        # The icon template is the reliable path. OCR is a safe fallback on
        # clients that show the currency name next to the balance.
        for hit in region.find_multi(
            RecognitionObject.ocr(1450, 0, 470, 150), limit=12
        ):
            if "洞天宝钱" in hit.text.replace(" ", ""):
                return True
        if self._coin is None:
            raise FileNotFoundError(
                f"快速购买缺少商店识别资产：{ASSET}；"
                "请运行 tools/fetch_map_assets.py --quick-buy，"
                "或显式配置 serenitea=true/false"
            )
        return False

    def _swipe_ref(
        self, x1: float, y1: float, x2: float, y2: float, duration_ms: int = 350
    ) -> None:
        transform = self.ctx.transform
        dx1, dy1 = transform.to_device(x1, y1)
        dx2, dy2 = transform.to_device(x2, y2)
        self.ctx.device.swipe(
            dx1,
            dy1,
            dx2,
            dy2,
            duration_ms=duration_ms,
            image_width=transform.device_width,
            image_height=transform.device_height,
        )

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        with exclusive_realtime_triggers(self.ctx):
            if cancelled and cancelled():
                return False
            if self._is_serenitea_shop():
                self.log("[QuickBuy] 尘歌壶商店：数量拉满并购买")
                self._swipe_ref(1450, 690, 1860, 690)
                self.ctx.sleep(200)
                self.ctx.input.click_ref(1600, 1020)
                self.ctx.sleep(250)
                self.ctx.input.click_ref(960, 850)
                return True

            self.log("[QuickBuy] 普通商店：打开数量页、拉满并确认")
            self.ctx.input.click_ref(1695, 1020)
            self.ctx.sleep(180)
            self._swipe_ref(742, 601, 1700, 601)
            self.ctx.sleep(180)
            self.ctx.input.click_ref(1100, 780)
            self.ctx.sleep(250)
            self.ctx.input.click_ref(1695, 1020)
            return True
