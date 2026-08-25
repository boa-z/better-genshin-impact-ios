"""Stateful BetterGI-compatible reward claiming jobs.

The desktop jobs for the battle pass and mail screen are deliberately small,
but they are not a single OCR click: both jobs have to get back to the game
HUD, wait for a page transition, deal with reward popups, and leave a manual
selection dialog untouched.  This module keeps those transitions explicit so
the iOS implementation can be replayed with recorded/fake frames in tests.

All scans in a state use one caller-owned frame.  The job also pauses the
shared realtime trigger loop for its complete lifetime; AutoPick/AutoSkip
therefore cannot consume a menu transition frame or compete for input.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from ..engine.context import GameContext
from ..engine.genshin_api import GenshinApi
from ..engine.recognition import ImageRegion, Mat, RecognitionObject
from .common_jobs import exclusive_realtime_triggers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = PROJECT_ROOT.parent / "better-genshin-impact"

_FULL_ROI = (0, 0, 1920, 1080)
# BetterGI searches the lower-right part of the page for the one-click
# control.  Keep the region broad enough for iPhone safe-area variants while
# excluding the left-side page title and navigation labels.
_CLAIM_ROI = (1040, 440, 880, 600)

_BATTLE_PASS_MARKERS = (
    "纪行", "紀行", "BattlePass", "Battle Pass", "Battlepass",
)
_MAIL_MARKERS = ("邮件", "郵件", "Mail")
_CLAIM_ALL_MARKERS = (
    "一键领取", "一鍵領取", "全部领取", "全部領取", "领取全部", "領取全部",
    "ClaimAll", "Claim All", "CollectAll", "Collect All",
)
_CLAIM_GENERIC_MARKERS = ("领取", "領取", "Claim", "Collect")
_ALREADY_CLAIMED_MARKERS = (
    "已领取", "已領取", "奖励已领取", "獎勵已領取", "AlreadyClaimed",
    "Already Claimed", "Claimed",
)
_MANUAL_SELECTION_MARKERS = (
    "请选择", "請選擇", "选择奖励", "選擇獎勵", "请选择奖励", "請選擇獎勵",
    "选择一个", "選擇一個", "Select a reward", "Choose a reward",
)
_PRIMOGEM_MARKERS = ("原石", "Primogem", "Primogems")


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


def _contains(text: Any, markers: Iterable[str]) -> bool:
    compact = _compact_text(text).casefold()
    return any(_compact_text(marker).casefold() in compact for marker in markers)


@dataclass
class ClaimPageResult:
    """Result of one page's claim attempt."""

    clicked: bool = False
    manual_selection: bool = False
    popup_closed: bool = False


@dataclass
class ClaimRewardsResult:
    """Detailed result retained for native callers and offline diagnostics."""

    opened: bool = False
    claimed: bool = False
    manual_selection: bool = False
    returned: bool = False
    cancelled: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.opened and self.returned and self.error is None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "opened": self.opened,
            "claimed": self.claimed,
            "manualSelection": self.manual_selection,
            "returned": self.returned,
            "cancelled": self.cancelled,
            **({"error": self.error} if self.error else {}),
        }


class _ClaimRewardsTask:
    """Shared page/OCR/template mechanics for mail and battle pass jobs."""

    _task_name = "ClaimRewards"

    # These paths deliberately include the upstream checkout as a fallback:
    # the iOS repository already uses that arrangement for POI assets, and it
    # lets a developer test the full common asset set without copying binary
    # files into a device-specific checkout.
    _asset_paths = {
        "mail": (
            PROJECT_ROOT / "assets" / "templates" / "common" / "esc_mail_reward.png",
            PROJECT_ROOT / "assets" / "templates" / "mail" / "esc_mail_reward.png",
            UPSTREAM_ROOT / "BetterGenshinImpact" / "GameTask" / "Common" /
            "Element" / "Assets" / "1920x1080" / "esc_mail_reward.png",
        ),
        "primogem": (
            PROJECT_ROOT / "assets" / "templates" / "autoskip" / "primogem.png",
            UPSTREAM_ROOT / "BetterGenshinImpact" / "GameTask" / "AutoSkip" /
            "Assets" / "1920x1080" / "primogem.png",
        ),
        "white_cancel": (
            PROJECT_ROOT / "assets" / "templates" / "stygian" / "white_cancel.png",
            UPSTREAM_ROOT / "BetterGenshinImpact" / "GameTask" / "Common" /
            "Element" / "Assets" / "1920x1080" / "btn_white_cancel.png",
        ),
        "white_confirm": (
            PROJECT_ROOT / "assets" / "templates" / "stygian" / "white_confirm.png",
            UPSTREAM_ROOT / "BetterGenshinImpact" / "GameTask" / "Common" /
            "Element" / "Assets" / "1920x1080" / "btn_white_confirm.png",
        ),
        "black_confirm": (
            PROJECT_ROOT / "assets" / "templates" / "stygian" / "black_confirm.png",
            UPSTREAM_ROOT / "BetterGenshinImpact" / "GameTask" / "Common" /
            "Element" / "Assets" / "1920x1080" / "btn_black_confirm.png",
        ),
        "autoskip_confirm_1": (
            PROJECT_ROOT / "assets" / "templates" / "autoskip" / "comfirm_btn1.png",
        ),
        "autoskip_confirm_2": (
            PROJECT_ROOT / "assets" / "templates" / "autoskip" / "comfirm_btn2.png",
        ),
    }

    def __init__(
        self,
        ctx: GameContext,
        *,
        return_main_ui: Callable[[], bool] | None = None,
        timeout_s: float = 24.0,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.timeout_s = max(6.0, min(120.0, float(timeout_s)))
        self.log = log
        self.return_main_ui = return_main_ui or GenshinApi(ctx, log=log).returnMainUi
        self._full_ocr = RecognitionObject.ocr(*_FULL_ROI)
        self._claim_ocr = RecognitionObject.ocr(*_CLAIM_ROI)
        self._templates: dict[str, RecognitionObject | None] = {}

    @staticmethod
    def _cancelled(cancelled: Callable[[], bool] | None) -> bool:
        try:
            return bool(cancelled and cancelled())
        except Exception:
            # A disappearing cancellation token must still trigger the input
            # cleanup/finally path.
            return True

    def _log(self, message: str) -> None:
        self.log(f"[{self._task_name}] {message}")

    @classmethod
    def _asset_path(cls, name: str) -> Path | None:
        return next((path for path in cls._asset_paths.get(name, ()) if path.is_file()), None)

    def _template(self, name: str) -> RecognitionObject | None:
        if name in self._templates:
            return self._templates[name]
        path = self._asset_path(name)
        if path is None:
            self._templates[name] = None
            return None
        try:
            ro = RecognitionObject.template_match(Mat.from_file(str(path)))
            ro.threshold = 0.72
        except (OSError, ValueError, TypeError) as error:
            self._log(f"模板 {name} 不可用：{error}")
            ro = None
        self._templates[name] = ro
        return ro

    @staticmethod
    def _find_marker(hits: Iterable[Any], markers: tuple[str, ...]):
        return next(
            (hit for hit in hits if _contains(getattr(hit, "text", ""), markers)),
            None,
        )

    @classmethod
    def _find_claim_button(cls, hits: Iterable[Any]):
        """Prefer an explicit 'claim all' label over generic claim text."""
        candidates = []
        for hit in hits:
            text = _compact_text(getattr(hit, "text", ""))
            if not text or _contains(text, _ALREADY_CLAIMED_MARKERS):
                continue
            if _contains(text, _CLAIM_ALL_MARKERS):
                return hit
            if _contains(text, _CLAIM_GENERIC_MARKERS):
                candidates.append(hit)
        return candidates[0] if candidates else None

    def _scan(self):
        """Capture once and run all state probes against the same image."""
        region = self.ctx.capture_region()
        full_hits = region.find_multi(self._full_ocr, limit=60)
        claim_hits = region.find_multi(self._claim_ocr, limit=30)
        return region, full_hits, claim_hits

    def _template_exists(self, region: ImageRegion, name: str) -> bool:
        ro = self._template(name)
        if ro is None:
            return False
        try:
            return bool(region.find(ro).is_exist())
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            self._log(f"模板 {name} 识别失败：{error}")
            return False

    def _is_manual_selection(self, region, full_hits: Iterable[Any]) -> bool:
        if self._find_marker(full_hits, _MANUAL_SELECTION_MARKERS) is not None:
            return True

        # Match the upstream Bv prompt rule: a cancel and a confirm button in
        # the same frame are enough to identify a manual selection dialog.
        has_cancel = self._template_exists(region, "white_cancel")
        has_confirm = any(
            self._template_exists(region, name)
            for name in (
                "white_confirm", "black_confirm",
                "autoskip_confirm_1", "autoskip_confirm_2",
            )
        )
        return bool(has_cancel and has_confirm)

    def _has_primogem(self, region, full_hits: Iterable[Any]) -> bool:
        return bool(
            self._find_marker(full_hits, _PRIMOGEM_MARKERS) is not None
            or self._template_exists(region, "primogem")
        )

    def _wait_click_page(
        self,
        markers: tuple[str, ...],
        deadline: float,
        *,
        template_name: str | None = None,
    ) -> bool:
        while time.monotonic() < deadline:
            region, full_hits, _claim_hits = self._scan()
            hit = self._find_marker(full_hits, markers)
            if hit is None and template_name:
                ro = self._template(template_name)
                if ro is not None:
                    try:
                        candidate = region.find(ro)
                    except (AttributeError, TypeError, ValueError, RuntimeError):
                        candidate = None
                    if candidate is not None and candidate.is_exist():
                        hit = candidate
            if hit is not None:
                hit.click()
                self.ctx.sleep(700)
                return True
            self.ctx.sleep(350)
        return False

    def _claim_current_page(
        self,
        deadline: float,
        *,
        allow_empty: bool = True,
    ) -> ClaimPageResult:
        while time.monotonic() < deadline:
            region, full_hits, claim_hits = self._scan()
            if self._is_manual_selection(region, full_hits):
                return ClaimPageResult(manual_selection=True)
            hit = self._find_claim_button(claim_hits)
            if hit is not None:
                hit.click()
                self._log("点击一键领取")
                self.ctx.sleep(1000)

                # This post-click frame serves both popup and manual-dialog
                # detection.  It avoids the old implementation's two extra
                # screenshot producers during a transition.
                after, after_full, _after_claim = self._scan()
                manual = self._is_manual_selection(after, after_full)
                popup_closed = False
                if self._has_primogem(after, after_full):
                    self.ctx.input.key_press("ESCAPE")
                    self.ctx.sleep(300)
                    popup_closed = True
                    self._log("关闭原石奖励弹窗")
                return ClaimPageResult(
                    clicked=True,
                    manual_selection=manual,
                    popup_closed=popup_closed,
                )

            # A page with no available reward is a normal BetterGI outcome;
            # the caller decides whether another tab should still be visited.
            if allow_empty and self._find_marker(full_hits, _ALREADY_CLAIMED_MARKERS):
                return ClaimPageResult()
            self.ctx.sleep(350)
        return ClaimPageResult()

    def _run_locked(
        self,
        cancelled: Callable[[], bool] | None,
        deadline: float,
    ) -> ClaimRewardsResult:
        raise NotImplementedError

    def run(self, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        result = ClaimRewardsResult()
        try:
            with exclusive_realtime_triggers(self.ctx):
                result = self._run_locked(cancelled, time.monotonic() + self.timeout_s)
        except Exception as error:
            result.error = str(error)
            self._log(f"执行失败：{error}")
        return result.as_dict()


class ClaimBattlePassRewardsTask(_ClaimRewardsTask):
    """Mirror BetterGI's two-stage battle-pass claim flow."""

    _task_name = "ClaimBattlePass"

    def _run_locked(
        self,
        cancelled: Callable[[], bool] | None,
        deadline: float,
    ) -> ClaimRewardsResult:
        result = ClaimRewardsResult()
        try:
            if self._cancelled(cancelled):
                result.cancelled = True
                return result
            if not self.return_main_ui():
                self._log("无法回到主界面")
                return result

            self.ctx.input.key_press("ESCAPE")
            self.ctx.sleep(900)
            if not self._wait_click_page(_BATTLE_PASS_MARKERS, deadline):
                self._log("未找到纪行入口")
                return result
            result.opened = True

            # The upstream job first claims battle-pass points, then waits for
            # the level-up animation before visiting the reward tab.
            for index, tab_x in enumerate((960, 858)):
                if self._cancelled(cancelled):
                    result.cancelled = True
                    break
                if index:
                    self.ctx.sleep(2500)
                self.ctx.input.click_ref(tab_x, 45)
                self.ctx.sleep(500 if index == 0 else 1500)
                page = self._claim_current_page(deadline)
                result.claimed |= page.clicked
                result.manual_selection |= page.manual_selection
                if page.manual_selection:
                    self._log("检测到需手动选择的纪行奖励，停止后续点击")
                    break
        finally:
            if result.opened:
                self.ctx.sleep(200)
                try:
                    result.returned = bool(self.return_main_ui())
                except Exception as error:
                    self._log(f"返回主界面失败：{error}")
        return result


class ClaimMailRewardsTask(_ClaimRewardsTask):
    """Open the Paimon mail page and claim all available attachments."""

    _task_name = "ClaimMail"

    def _run_locked(
        self,
        cancelled: Callable[[], bool] | None,
        deadline: float,
    ) -> ClaimRewardsResult:
        result = ClaimRewardsResult()
        try:
            if self._cancelled(cancelled):
                result.cancelled = True
                return result
            if not self.return_main_ui():
                self._log("无法回到主界面")
                return result

            self.ctx.input.key_press("ESCAPE")
            self.ctx.sleep(1300)
            if not self._wait_click_page(
                _MAIL_MARKERS,
                deadline,
                template_name="mail",
            ):
                self._log("没有邮件奖励")
                return result
            result.opened = True

            if self._cancelled(cancelled):
                result.cancelled = True
                return result
            page = self._claim_current_page(deadline)
            result.claimed = page.clicked
            result.manual_selection = page.manual_selection
            if page.manual_selection:
                self._log("检测到需手动选择的邮件奖励，停止后续点击")
        finally:
            if result.opened:
                self.ctx.input.key_press("ESCAPE")
                self.ctx.sleep(400)
                try:
                    result.returned = bool(self.return_main_ui())
                except Exception as error:
                    self._log(f"返回主界面失败：{error}")
        return result


__all__ = [
    "ClaimBattlePassRewardsTask",
    "ClaimMailRewardsTask",
    "ClaimPageResult",
    "ClaimRewardsResult",
]
