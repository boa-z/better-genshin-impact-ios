"""领取尘歌壶奖励的 BetterGI Common Job 移动端实现。

原版 ``GoToSereniteaPotTask`` 同时负责进壶、寻找阿圆、领取好感/宝钱和
按配置购买洞天百宝。早期 iOS 移植只实现了 ``QuickSereniteaPot``（进出
尘歌壶），导致 OneDragon 中同名项目实际上没有领取奖励。本模块保留
原版任务的公共契约，但把 Windows 专用的鼠标寻路替换为：

* 复用背包小道具进壶流程；
* 在同一张截图中 OCR 阿圆/壶灵，按目标偏移转动相机并前进；
* 通过共享交互检测器和 ``genshin.chooseTalkOption`` 完成对话；
* 只有显式提供商店物品时才打开尘歌壶商店并购买，默认不产生购买动作。

任务会在整个流程期间暂停调用方的实时触发器，避免 AutoPick/AutoSkip
与本任务的截图和输入互相竞争。
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from ..engine.context import GameContext
from ..engine.genshin_api import GenshinApi
from ..engine.recognition import RecognitionObject
from .common_jobs import InteractionPromptDetector, _pause_realtime_triggers, _resume_realtime_triggers


def _value(raw: Any, *names: str, default: Any = None) -> Any:
    """Read JSON/ClearScript style properties case-insensitively."""

    if isinstance(raw, Mapping):
        folded = {
            str(key).replace("_", "").casefold(): value
            for key, value in raw.items()
        }
        for name in names:
            key = str(name).replace("_", "").casefold()
            if key in folded:
                return folded[key]
        return default
    if raw is None:
        return default
    wanted = {str(name).replace("_", "").casefold() for name in names}
    try:
        for candidate in dir(raw):
            if candidate.replace("_", "").casefold() in wanted:
                result = getattr(raw, candidate)
                return default if result is None else result
    except (AttributeError, TypeError):
        pass
    return default


def _compact(value: Any) -> str:
    return "".join(str(value or "").replace("\u3000", "").split()).casefold()


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on", "是"}:
            return True
        if normalized in {"0", "false", "no", "off", "否", ""}:
            return False
    return default


_DAY_NAMES = {
    "星期一": 0, "周一": 0, "一": 0, "monday": 0, "mon": 0,
    "星期二": 1, "周二": 1, "二": 1, "tuesday": 1, "tue": 1,
    "星期三": 2, "周三": 2, "三": 2, "wednesday": 2, "wed": 2,
    "星期四": 3, "周四": 3, "四": 3, "thursday": 3, "thu": 3,
    "星期五": 4, "周五": 4, "五": 4, "friday": 4, "fri": 4,
    "星期六": 5, "周六": 5, "六": 5, "saturday": 5, "sat": 5,
    "星期日": 6, "星期天": 6, "周日": 6, "周天": 6,
    "日": 6, "天": 6, "sunday": 6, "sun": 6,
}
_EVERY_DAY = frozenset({"", "每天重复", "每天", "daily", "everyday", "all"})


def _normalize_day(value: Any) -> int | None:
    text = _compact(value)
    if text in _EVERY_DAY:
        return None
    if text.isdigit():
        number = int(text)
        if 1 <= number <= 7:
            return number - 1
    return _DAY_NAMES.get(text)


def server_day(now: datetime | None = None) -> int:
    """Return BetterGI's day-of-week (the daily reset is at 04:00)."""

    current = now or datetime.now().astimezone()
    return (current - timedelta(hours=4)).weekday()


def is_shop_schedule_active(schedule: Any = None, *, now: datetime | None = None) -> bool:
    """Whether a ``SecretTreasureObjects`` weekly prefix is active."""

    if schedule is None or _normalize_day(schedule) is None:
        text = _compact(schedule)
        if text not in _EVERY_DAY:
            # Unknown schedule names should fail closed.  A typo must not
            # cause an unintended weekly purchase.
            return False
        return True
    return _normalize_day(schedule) == server_day(now)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values = value.replace("；", ";").replace("，", ",").replace("\n", ";")
        return [part.strip() for part in values.replace(",", ";").split(";") if part.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


@dataclass(frozen=True)
class SereniteaPotRewardConfig:
    """Safe, touch-oriented subset of the upstream reward configuration."""

    enter_with_pot: bool = True
    claim_trust: bool = True
    shop_items: tuple[str, ...] = ()
    shop_schedule: str = ""
    leave_after: bool = True
    timeout_s: float = 240.0
    search_turns: int = 30
    search_step: float = 180.0
    walk_timeout_s: float = 35.0
    npc_names: tuple[str, ...] = ("阿圆", "壶灵", "<壶灵>", "Tubby")

    @classmethod
    def from_mapping(cls, raw: Any) -> "SereniteaPotRewardConfig":
        raw_items = _value(
            raw,
            "shopItems",
            "buyItems",
            "secretTreasureObjects",
            "secretTreasureItems",
            default=(),
        )
        items = _string_list(raw_items)
        schedule = str(_value(
            raw, "shopSchedule", "shopDay", "buyDay", default=""
        ) or "").strip()
        # BetterGI stores the weekly selector as the first element of
        # SecretTreasureObjects (for example ["星期一", "布匹"]).
        if items and not schedule and (
            _compact(items[0]) in _EVERY_DAY or _normalize_day(items[0]) is not None
        ):
            schedule, items = items[0], items[1:]
        timeout = _value(raw, "timeoutSeconds", "timeout", default=240)
        search_turns = _value(raw, "searchTurns", "maxSearchTurns", default=30)
        search_step = _value(raw, "searchStep", "cameraStep", default=180)
        walk_timeout = _value(raw, "walkTimeoutSeconds", "walkTimeout", default=35)
        try:
            timeout = max(20.0, min(1800.0, float(timeout)))
        except (TypeError, ValueError):
            timeout = 240.0
        try:
            search_turns = max(1, min(120, int(search_turns)))
        except (TypeError, ValueError):
            search_turns = 30
        try:
            search_step = max(40.0, min(600.0, float(search_step)))
        except (TypeError, ValueError):
            search_step = 180.0
        try:
            walk_timeout = max(2.0, min(180.0, float(walk_timeout)))
        except (TypeError, ValueError):
            walk_timeout = 35.0
        names = tuple(_string_list(_value(raw, "npcNames", default=())) or cls.npc_names)
        return cls(
            enter_with_pot=_bool(_value(raw, "enterWithPot", "usePot", default=True), True),
            claim_trust=_bool(_value(raw, "claimTrust", "claimFriendship", default=True), True),
            shop_items=tuple(dict.fromkeys(items)),
            shop_schedule=schedule,
            leave_after=_bool(_value(raw, "leaveAfter", "returnToWorld", default=True), True),
            timeout_s=timeout,
            search_turns=search_turns,
            search_step=search_step,
            walk_timeout_s=walk_timeout,
            npc_names=names,
        )


class SereniteaPotRewardsTask:
    """Enter the Serenitea Pot, claim configured rewards, and leave safely."""

    NPC_ROI = (180, 70, 1560, 650)
    SHOP_ROI = (120, 120, 1680, 820)

    def __init__(
        self,
        ctx: GameContext,
        *,
        config: SereniteaPotRewardConfig | Mapping[str, Any] | None = None,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.config = (
            config if isinstance(config, SereniteaPotRewardConfig)
            else SereniteaPotRewardConfig.from_mapping(config or {})
        )
        self.log = log
        self._interaction = InteractionPromptDetector(ctx, log=log)

    @staticmethod
    def _cancelled(cancelled: Callable[[], bool] | None) -> bool:
        try:
            return bool(cancelled and cancelled())
        except Exception:
            return True

    def _find_text(self, region, words: Sequence[str], roi) -> Any | None:
        wanted = tuple(_compact(word) for word in words if _compact(word))
        if not wanted:
            return None
        hits = region.find_multi(RecognitionObject.ocr(*roi), limit=80)
        for hit in hits:
            text = _compact(getattr(hit, "text", ""))
            if any(word in text for word in wanted):
                return hit
        return None

    def _tap_text(
        self,
        words: Sequence[str],
        deadline: float,
        *,
        roi=(0, 0, 1920, 1080),
        interval_ms: int = 350,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        while time.monotonic() < deadline:
            if self._cancelled(cancelled):
                return False
            region = self.ctx.capture_region()
            hit = self._find_text(region, words, roi)
            if hit is not None:
                hit.click()
                self.ctx.sleep(450)
                return True
            self.ctx.sleep(interval_ms)
        return False

    def _enter(self, cancelled: Callable[[], bool] | None) -> bool:
        if not self.config.enter_with_pot:
            return True
        from .quick_serenitea import QuickSereniteaPotTask

        return QuickSereniteaPotTask(
            self.ctx,
            timeout_s=min(60.0, self.config.timeout_s),
            log=self.log,
        ).run(cancelled=cancelled)

    def _find_and_approach_npc(
        self,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        """Find 阿圆 with OCR and approach without a second capture loop."""

        names = self.config.npc_names
        search_deadline = min(
            deadline, time.monotonic() + self.config.walk_timeout_s
        )
        for _ in range(self.config.search_turns):
            if time.monotonic() >= search_deadline or self._cancelled(cancelled):
                return False
            region = self.ctx.capture_region()
            target = self._find_text(region, names, self.NPC_ROI)
            if target is None:
                self.ctx.input.move_camera_by(self.config.search_step, 0)
                self.ctx.sleep(350)
                continue

            center_x = float(getattr(target, "x", 0)) + float(getattr(target, "width", 0)) / 2
            center_y = float(getattr(target, "y", 0)) + float(getattr(target, "height", 0)) / 2
            dx = center_x - 960.0
            if abs(dx) > max(45.0, float(getattr(target, "width", 40)) * 1.4):
                self.ctx.input.move_camera_by(max(-300.0, min(300.0, dx * 0.65)), 0)
                self.ctx.sleep(300)
                continue
            if center_y > 360:
                self.ctx.input.move_camera_by(0, max(-180.0, min(180.0, (center_y - 280) * 0.55)))
                self.ctx.sleep(300)
                continue

            # Keep walking only for this iteration.  The finally block below
            # prevents a cancelled task from leaving W held on DeviceHub.
            self.ctx.input.key_down("W")
            try:
                if self._interaction.visible(region):
                    self.ctx.input.key_press("F")
                    self.log(f"[Serenitea] 已接近 {'/'.join(names[:2])}")
                    self.ctx.sleep(700)
                    return True
                self.ctx.sleep(280)
            finally:
                self.ctx.input.key_up("W")

        self.log("[Serenitea] 未找到阿圆/壶灵")
        return False

    def _choose(self, api: GenshinApi, words: Sequence[str], cancelled) -> bool:
        if self._cancelled(cancelled):
            return False
        option = next((str(word) for word in words if str(word).strip()), "")
        if not option:
            return False
        try:
            return bool(api.chooseTalkOption(option, skip_times=6))
        except Exception as error:
            self.log(f"[Serenitea] 对话选项「{option}」失败：{error}")
            return False

    def _dismiss_reward_popup(self, cancelled) -> None:
        # The affection page uses several localized labels.  Click a visible
        # button first; only fall back to Escape after the OCR grace period so
        # a slow reward animation is not mistaken for a dialogue exit.
        deadline = time.monotonic() + 5.0
        words = ("确认", "确定", "关闭", "知道了", "无法领取好感", "Confirm", "Close")
        self._tap_text(words, deadline, roi=(620, 380, 680, 500), cancelled=cancelled)
        if not self._cancelled(cancelled):
            self.ctx.input.key_press("ESCAPE")
            self.ctx.sleep(400)

    def _buy_shop_items(self, api: GenshinApi, deadline: float, cancelled) -> list[str]:
        items = self.config.shop_items
        if not items or not is_shop_schedule_active(self.config.shop_schedule):
            if items and self.config.shop_schedule:
                self.log(f"[Serenitea] 本日不购买洞天百宝：{self.config.shop_schedule}")
            return []
        if not self._choose(api, ("洞天百宝", "洞天百寶", "Realm Depot", "Shop"), cancelled):
            self.log("[Serenitea] 未找到洞天百宝对话选项")
            return []
        self.ctx.sleep(700)
        from .quick_buy import QuickBuyTask

        purchased: list[str] = []
        for item in items:
            if time.monotonic() >= deadline or self._cancelled(cancelled):
                break
            region = self.ctx.capture_region()
            hit = self._find_text(region, (item,), self.SHOP_ROI)
            if hit is None:
                self.log(f"[Serenitea] 商店未找到：{item}")
                continue
            hit.click()
            self.ctx.sleep(450)
            if QuickBuyTask(self.ctx, serenitea=True, log=self.log).run(cancelled=cancelled):
                purchased.append(item)
        self.ctx.input.key_press("ESCAPE")
        self.ctx.sleep(500)
        return purchased

    def run(self, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.timeout_s
        result: dict[str, Any] = {
            "status": "failed",
            "entered": False,
            "npcFound": False,
            "trustClaimed": False,
            "purchased": [],
            "left": False,
        }
        trigger_loop, trigger_state = _pause_realtime_triggers(self.ctx)
        api = GenshinApi(self.ctx, log=self.log)
        try:
            if self._cancelled(cancelled):
                result["status"] = "cancelled"
                return result
            if not self._enter(cancelled):
                return result
            result["entered"] = True
            if not self._find_and_approach_npc(deadline, cancelled):
                return result
            result["npcFound"] = True

            if self.config.claim_trust:
                result["trustClaimed"] = self._choose(
                    api, ("信任", "Trust", "领取好感", "好感"), cancelled
                )
                if result["trustClaimed"]:
                    self._dismiss_reward_popup(cancelled)

            result["purchased"] = self._buy_shop_items(api, deadline, cancelled)
            if self._cancelled(cancelled):
                result["status"] = "cancelled"
                return result
            if self.config.claim_trust and not result["trustClaimed"] and not result["purchased"]:
                return result

            if self.config.leave_after:
                result["left"] = api.clickChatExitUntilMainUi(retry_times=8)
                if result["left"]:
                    from .quick_serenitea import QuickSereniteaPotTask

                    result["left"] = QuickSereniteaPotTask(
                        self.ctx,
                        timeout_s=min(60.0, max(10.0, deadline - time.monotonic())),
                        log=self.log,
                    ).run(cancelled=cancelled)
            result["status"] = "completed" if not self.config.leave_after or result["left"] else "completed_with_warnings"
            return result
        except Exception as error:
            result["error"] = str(error)
            self.log(f"[Serenitea] 领取尘歌壶奖励失败：{error}")
            return result
        finally:
            try:
                self.ctx.input.release_all()
            except Exception:
                pass
            _resume_realtime_triggers(trigger_loop, trigger_state)


# Keep the upstream class spelling available to converted Python adapters.
GoToSereniteaPotTask = SereniteaPotRewardsTask
