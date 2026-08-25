"""Check the daily commission reward state and emit BetterGI notifications."""

from __future__ import annotations

import time
from typing import Any, Callable

from ..engine.context import GameContext
from ..engine.genshin_api import GenshinApi
from ..engine.recognition import RecognitionObject


DAILY_REWARD_EVENT = "daily.reward"
_PAGE_MARKERS = (
    "每日委托奖励",
    "每日奖励",
    "DailyCommission",
    "DailyReward",
)
_CLAIMED_MARKERS = (
    "今日奖励已领取",
    "今日奖励已領取",
    "Today'srewardsclaimed",
    "Dailyrewardsclaimed",
)
_COMMISSION_TAB_MARKERS = ("委托", "委託", "Commission", "Commissions")


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


class CheckRewardsTask:
    """Portable equivalent of BetterGI's ``CheckRewardsTask``.

    The task deliberately scans one OCR region per polling iteration and
    reuses its results for page/tab/status decisions. This keeps the check
    from creating a second screenshot loop beside the WebUI or realtime
    trigger stream.
    """

    def __init__(
        self,
        ctx: GameContext,
        *,
        timeout_s: float = 12.0,
        notification_service=None,
        return_main_ui: Callable[[], bool] | None = None,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.timeout_s = max(4.0, min(60.0, float(timeout_s)))
        self.notification_service = notification_service
        self.return_main_ui = return_main_ui or GenshinApi(ctx, log=log).returnMainUi
        self.log = log
        # Matches the upstream GetConfirmRa area (10%..40% width and
        # 10%..80% height) while keeping the reference coordinate contract.
        self._scan_ocr = RecognitionObject.ocr(192, 108, 576, 756)

    @staticmethod
    def _contains(text: str, markers: tuple[str, ...]) -> bool:
        compact = _compact_text(text).casefold()
        return any(_compact_text(marker).casefold() in compact for marker in markers)

    @classmethod
    def _find_commission_tab(cls, hits):
        exact = {marker.casefold() for marker in _COMMISSION_TAB_MARKERS}
        for hit in hits:
            text = _compact_text(getattr(hit, "text", ""))
            folded = text.casefold()
            if folded in exact:
                return hit
            # OCR sometimes attaches a short suffix to the tab label, but do
            # not click the larger "每日委托奖励" page title by mistake.
            if (
                any(marker.casefold() in folded for marker in _COMMISSION_TAB_MARKERS)
                and "每日" not in text
                and len(text) <= 12
            ):
                return hit
        return None

    def _scan(self):
        region = self.ctx.capture_region()
        return region, region.find_multi(self._scan_ocr, limit=80)

    @staticmethod
    def _has_marker(hits, markers: tuple[str, ...]) -> bool:
        return any(
            CheckRewardsTask._contains(getattr(hit, "text", ""), markers)
            for hit in hits
        )

    def _notify(self, message: str, *, success: bool) -> None:
        result = "Success" if success else "Fail"
        self.log(f"[CheckRewards] {message}")
        service = self.notification_service
        if service is None:
            return
        try:
            service.notify(DAILY_REWARD_EVENT, message, result=result)
        except Exception as error:
            # A broken Gotify endpoint must not prevent returning to the game.
            self.log(f"[CheckRewards] 通知发送失败：{error}")

    @staticmethod
    def _cancelled(cancelled: Callable[[], bool] | None) -> bool:
        try:
            return bool(cancelled and cancelled())
        except Exception:
            return True

    def _wait_for_page(
        self,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[bool, bool]:
        tab_clicked = False
        while time.monotonic() < deadline:
            if self._cancelled(cancelled):
                return False, tab_clicked
            _, hits = self._scan()
            if self._has_marker(hits, _PAGE_MARKERS):
                return True, tab_clicked
            tab = self._find_commission_tab(hits)
            if tab is not None and not tab_clicked:
                tab.click()
                tab_clicked = True
                self.log("[CheckRewards] 打开委托奖励页")
                self.ctx.sleep(500)
            else:
                self.ctx.sleep(400)
        return False, tab_clicked

    def _wait_for_claimed(
        self,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[bool, bool]:
        while time.monotonic() < deadline:
            if self._cancelled(cancelled):
                return False, True
            _, hits = self._scan()
            if self._has_marker(hits, _CLAIMED_MARKERS):
                return True, False
            self.ctx.sleep(400)
        return False, False

    def run(self, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_s
        page_opened = False
        try:
            if self._cancelled(cancelled):
                return {"checked": False, "claimed": None, "cancelled": True}
            if not self.return_main_ui():
                self._notify("检查每日奖励失败：无法回到主界面", success=False)
                return {"checked": False, "claimed": None, "pageOpened": False}

            self.ctx.input.key_press("F1")
            self.ctx.sleep(900)
            page_opened, _tab_clicked = self._wait_for_page(deadline, cancelled)
            if not page_opened:
                if self._cancelled(cancelled):
                    return {"checked": False, "claimed": None, "cancelled": True}
                self._notify("检查每日奖励失败：未打开冒险之证委托页", success=False)
                return {"checked": False, "claimed": None, "pageOpened": False}

            claimed, was_cancelled = self._wait_for_claimed(deadline, cancelled)
            if was_cancelled:
                return {"checked": False, "claimed": None, "cancelled": True}
            if claimed:
                self._notify("检查每日奖励：已领取", success=True)
            else:
                self._notify("检查到每日奖励未领取，请手动查看！", success=False)
            return {"checked": True, "claimed": claimed, "pageOpened": True}
        except Exception as error:
            self.log(f"[CheckRewards] 执行失败：{error}")
            return {
                "checked": False,
                "claimed": None,
                "pageOpened": page_opened,
                "error": str(error),
            }
        finally:
            if page_opened and not self._cancelled(cancelled):
                self.ctx.sleep(200)
                try:
                    self.return_main_ui()
                except Exception as error:
                    self.log(f"[CheckRewards] 返回主界面失败：{error}")
