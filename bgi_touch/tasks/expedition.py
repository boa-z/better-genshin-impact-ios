"""Claim completed expeditions and dispatch the same assignments again."""

from __future__ import annotations

import time
from typing import Callable

from ..engine.context import GameContext
from ..engine.recognition import RecognitionObject
from .common_jobs import exclusive_realtime_triggers


class OneKeyExpeditionTask:
    """Mobile adaptation of BetterGI's expedition-page one-key action.

    The desktop implementation matches the button icons.  Mobile keeps the
    same labels, so OCR is used within the two small button areas and does not
    require device-specific binary templates.
    """

    def __init__(
        self,
        ctx: GameContext,
        *,
        collect_retries: int = 2,
        redispatch_retries: int = 3,
        timeout_s: float = 12.0,
        close_page: bool = True,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.collect_retries = max(1, min(5, int(collect_retries)))
        self.redispatch_retries = max(1, min(8, int(redispatch_retries)))
        self.timeout_s = max(2.0, float(timeout_s))
        self.close_page = bool(close_page)
        self.log = log
        # BetterGI reference-space equivalents of the upstream lower-left and
        # lower-right recognition areas.
        self.ro_collect = RecognitionObject.ocr_match(
            0, 720, 560, 360, "全部领取", "一键领取",
        )
        self.ro_redispatch = RecognitionObject.ocr_match(
            900, 780, 620, 300, "再次派遣", "重新派遣", "再次探索",
        )

    def _cancelled(
        self,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        return time.monotonic() >= deadline or bool(cancelled and cancelled())

    def _find(self, recognition):
        return self.ctx.capture_region().find(recognition)

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        with exclusive_realtime_triggers(self.ctx):
            deadline = time.monotonic() + self.timeout_s
            try:
                collect = None
                for attempt in range(self.collect_retries):
                    if self._cancelled(deadline, cancelled):
                        return False
                    collect = self._find(self.ro_collect)
                    if collect.is_exist():
                        break
                    self.log("[Expedition] 未找到全部领取按钮")
                    if attempt + 1 < self.collect_retries:
                        self.ctx.sleep(1000)
                if collect is None or not collect.is_exist():
                    return False

                collect.click()
                self.log("[Expedition] 全部领取")
                self.ctx.sleep(1100)

                for attempt in range(self.redispatch_retries):
                    if self._cancelled(deadline, cancelled):
                        return False
                    redispatch = self._find(self.ro_redispatch)
                    if redispatch.is_exist():
                        redispatch.click()
                        self.log("[Expedition] 再次派遣")
                        self.ctx.sleep(500)
                        if self.close_page:
                            self.ctx.input.key_press("ESCAPE")
                            self.ctx.sleep(250)
                        self.log("[Expedition] 完成")
                        return True
                    if attempt + 1 < self.redispatch_retries:
                        self.ctx.sleep(1000)
                self.log("[Expedition] 未检测到再次派遣按钮")
                return False
            except Exception as error:
                self.log(f"[Expedition] 执行失败：{error}")
                return False
            finally:
                self.ctx.input.release_all()
