"""Claim BetterGI's long-term encounter-point reward.

The desktop job opens the Adventurer's Handbook, selects the commission tab
when necessary, and clicks the claim control.  The iOS layout has no stable
desktop template in every safe-area variant, so this port uses small OCR
regions and keeps the whole flow inside the caller-owned trigger scope.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from ..engine.context import GameContext
from ..engine.genshin_api import GenshinApi
from ..engine.recognition import RecognitionObject
from .common_jobs import exclusive_realtime_triggers


_HANDBOOK_ROI = (0, 0, 720, 1080)
_CLAIM_ROI = (1180, 520, 740, 500)

_COMMISSION_MARKERS = (
    "每日委托奖励",
    "每日奖励",
    "DailyCommissionReward",
    "DailyReward",
)
_COMMISSION_TAB_MARKERS = ("委托", "委託", "Commission", "Commissions")
_CLAIM_MARKERS = ("领取", "領取", "Claim")
_CLAIMED_MARKERS = (
    "今日奖励已领取",
    "今日奖励已領取",
    "奖励已领取",
    "獎勵已領取",
    "Today'srewardsclaimed",
    "Dailyrewardsclaimed",
)


def _compact_text(value: Any) -> str:
    return (
        str(value or "")
        .replace(" ", "")
        .replace("\u3000", "")
        .replace("\n", "")
        .replace("\r", "")
        .replace("：", ":")
        .strip()
    )


def _contains(text: str, markers: tuple[str, ...]) -> bool:
    compact = _compact_text(text).casefold()
    return any(_compact_text(marker).casefold() in compact for marker in markers)


class ClaimEncounterPointsRewardsTask:
    """Open the handbook and claim the available encounter-point reward.

    ``run`` returns details for native callers.  The public genshin and
    dispatcher wrappers intentionally reduce that result to a bool, matching
    the existing BetterGI-compatible API surface.
    """

    def __init__(
        self,
        ctx: GameContext,
        *,
        timeout_s: float = 12.0,
        return_main_ui: Callable[[], bool] | None = None,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.timeout_s = max(4.0, min(60.0, float(timeout_s)))
        self.return_main_ui = return_main_ui or GenshinApi(ctx, log=log).returnMainUi
        self.log = log
        self._handbook_ocr = RecognitionObject.ocr(*_HANDBOOK_ROI)
        self._claim_ocr = RecognitionObject.ocr(*_CLAIM_ROI)

    @staticmethod
    def _cancelled(cancelled: Callable[[], bool] | None) -> bool:
        try:
            return bool(cancelled and cancelled())
        except Exception:
            return True

    @staticmethod
    def _find_marker(hits, markers: tuple[str, ...]):
        return next(
            (
                hit for hit in hits
                if _contains(getattr(hit, "text", ""), markers)
            ),
            None,
        )

    @classmethod
    def _find_claim_button(cls, hits):
        """Find a claim control without treating status text as a button."""
        for hit in hits:
            text = getattr(hit, "text", "")
            if _contains(text, _CLAIMED_MARKERS) or "未领取" in _compact_text(text):
                continue
            if _contains(text, _CLAIM_MARKERS):
                return hit
        return None

    @classmethod
    def _find_commission_tab(cls, hits):
        exact = {
            _compact_text(marker).casefold()
            for marker in _COMMISSION_TAB_MARKERS
        }
        for hit in hits:
            text = _compact_text(getattr(hit, "text", ""))
            folded = text.casefold()
            if folded in exact:
                return hit
            # Avoid clicking the larger page title "每日委托奖励" when OCR
            # attaches the title and tab into one result.
            if (
                any(marker.casefold() in folded for marker in _COMMISSION_TAB_MARKERS)
                and "每日" not in text
                and len(text) <= 12
            ):
                return hit
        return None

    def _scan(self):
        """Capture one frame and reuse its OCR results for all decisions."""
        region = self.ctx.capture_region()
        handbook_hits = region.find_multi(self._handbook_ocr, limit=40)
        claim_hits = region.find_multi(self._claim_ocr, limit=20)
        return region, handbook_hits, claim_hits

    def _notify_log(self, message: str) -> None:
        self.log(f"[ClaimEncounter] {message}")

    def _run_locked(
        self,
        cancelled: Callable[[], bool] | None,
        deadline: float,
    ) -> dict[str, Any]:
        page_opened = False
        claim_clicked = False
        tab_clicked = False
        try:
            if self._cancelled(cancelled):
                return {
                    "ok": False,
                    "opened": False,
                    "claimed": False,
                    "alreadyClaimed": False,
                    "cancelled": True,
                }

            if not self.return_main_ui():
                self._notify_log("无法回到主界面")
                return {
                    "ok": False,
                    "opened": False,
                    "claimed": False,
                    "alreadyClaimed": False,
                }

            self.ctx.input.key_press("F1")
            self.ctx.sleep(1000)

            while time.monotonic() < deadline:
                if self._cancelled(cancelled):
                    return {
                        "ok": False,
                        "opened": page_opened,
                        "claimed": claim_clicked,
                        "alreadyClaimed": False,
                        "cancelled": True,
                    }

                _region, handbook_hits, claim_hits = self._scan()
                page_hit = self._find_marker(handbook_hits, _COMMISSION_MARKERS)
                if page_hit is not None:
                    page_opened = True

                claimed_hit = self._find_marker(
                    (*handbook_hits, *claim_hits), _CLAIMED_MARKERS,
                )
                if claimed_hit is not None:
                    page_opened = True
                    self._notify_log("历练点奖励已领取")
                    return {
                        "ok": True,
                        "opened": True,
                        "claimed": False,
                        "alreadyClaimed": True,
                    }

                claim_hit = self._find_claim_button(claim_hits)
                if claim_hit is not None:
                    page_opened = True
                    claim_hit.click()
                    claim_clicked = True
                    self._notify_log("领取长效历练点奖励")
                    self.ctx.sleep(1000)
                    return {
                        "ok": True,
                        "opened": True,
                        "claimed": True,
                        "alreadyClaimed": False,
                    }

                tab = self._find_commission_tab(handbook_hits)
                if tab is not None and not tab_clicked:
                    page_opened = True
                    tab.click()
                    tab_clicked = True
                    self._notify_log("打开冒险之证委托页")
                    self.ctx.sleep(500)
                else:
                    # The page transition can take several frames on a
                    # device. Avoid another F1 press while it is settling.
                    self.ctx.sleep(350)

            self._notify_log("未找到领取按钮，可能未完成委托或页面未打开")
            return {
                "ok": False,
                "opened": page_opened,
                "claimed": claim_clicked,
                "alreadyClaimed": False,
            }
        except Exception as error:
            self._notify_log(f"执行失败：{error}")
            return {
                "ok": False,
                "opened": page_opened,
                "claimed": claim_clicked,
                "alreadyClaimed": False,
                "error": str(error),
            }
        finally:
            if page_opened and not self._cancelled(cancelled):
                self.ctx.sleep(200)
                try:
                    self.return_main_ui()
                except Exception as error:
                    self._notify_log(f"返回主界面失败：{error}")

    def run(self, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        """Run the task while the caller-owned realtime trigger is paused."""
        try:
            with exclusive_realtime_triggers(self.ctx):
                return self._run_locked(
                    cancelled,
                    time.monotonic() + self.timeout_s,
                )
        except Exception as error:
            self._notify_log(f"触发器隔离失败：{error}")
            return {
                "ok": False,
                "opened": False,
                "claimed": False,
                "alreadyClaimed": False,
                "error": str(error),
            }


__all__ = ["ClaimEncounterPointsRewardsTask"]
