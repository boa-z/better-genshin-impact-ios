"""BetterGI ``GoToAdventurersGuildTask`` 的 iOS 触控实现。

原版任务并不只是“走到凯瑟琳并按 F”：它还负责领取长效历练点奖励、
领取每日委托奖励、确认奖励弹窗、重新进入凯瑟琳对话以及一键领取/重派
探索任务。这里把这些步骤放在一个任务中，避免 OneDragon 或脚本调用时
在同一页重复打开冒险之证。

整个流程使用一个共享截图/输入所有权范围。实时 AutoPick/AutoSkip 在菜单
和对话切换期间暂停，路线文件中的默认 realtimeTriggers 也会被关闭；
否则路线结束帧很容易被实时触发器抢先按 F。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..engine.context import GameContext
from ..engine.genshin_api import GenshinApi
from ..engine.recognition import Mat, RecognitionObject
from ..pathing.executor import PathingExecutor
from ..pathing.model import PathingTask
from .common_jobs import InteractionPromptDetector, exclusive_realtime_triggers
from .expedition import OneKeyExpeditionTask


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_ASSETS = PROJECT_ROOT / "assets" / "templates"
BLACK_CONFIRM_ASSET = COMMON_ASSETS / "artifact_salvage" / "btn_black_confirm.png"


def _value(raw: Any, *names: str, default: Any = None) -> Any:
    """Read mapping/object properties using BetterGI's mixed casing."""

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


def _bool(value: Any, default: bool = False) -> bool:
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


def _compact(value: Any) -> str:
    return (
        str(value or "")
        .replace(" ", "")
        .replace("\u3000", "")
        .replace("\n", "")
        .replace("\r", "")
        .casefold()
    )


@dataclass(frozen=True)
class AdventurersGuildConfig:
    """Touch-safe subset of the upstream job parameters."""

    country: str
    daily_reward_party_name: str = ""
    only_do_once: bool = False
    timeout_s: float = 180.0
    encounter_timeout_s: float = 12.0
    talk_skip_times: int = 10
    interaction_retries: int = 3
    close_retries: int = 15

    @classmethod
    def from_mapping(
        cls,
        country: str | None = None,
        raw: Mapping[str, Any] | None = None,
    ) -> "AdventurersGuildConfig":
        raw = raw or {}
        configured_country = country or _value(raw, "country", "Country", default="")
        country_text = str(configured_country or "").strip()
        if not country_text:
            raise ValueError("前往冒险家协会需要 country")

        def bounded_float(value: Any, fallback: float, lower: float, upper: float) -> float:
            try:
                return max(lower, min(upper, float(value)))
            except (TypeError, ValueError):
                return fallback

        def bounded_int(value: Any, fallback: int, lower: int, upper: int) -> int:
            try:
                return max(lower, min(upper, int(value)))
            except (TypeError, ValueError):
                return fallback

        return cls(
            country=country_text,
            daily_reward_party_name=str(_value(
                raw,
                "dailyRewardPartyName",
                "daily_reward_party_name",
                "partyName",
                default="",
            ) or "").strip(),
            only_do_once=_bool(_value(
                raw, "onlyDoOnce", "only_do_once", default=False,
            )),
            timeout_s=bounded_float(
                _value(raw, "timeoutSeconds", "timeout", default=180),
                180.0,
                30.0,
                1800.0,
            ),
            encounter_timeout_s=bounded_float(
                _value(raw, "encounterTimeoutSeconds", "encounterTimeout", default=12),
                12.0,
                4.0,
                60.0,
            ),
            talk_skip_times=bounded_int(
                _value(raw, "talkSkipTimes", "skipTimes", default=10),
                10,
                1,
                30,
            ),
            interaction_retries=bounded_int(
                _value(raw, "interactionRetries", default=3),
                3,
                1,
                8,
            ),
            close_retries=bounded_int(
                _value(raw, "closeRetries", default=15),
                15,
                1,
                40,
            ),
        )


class AdventurersGuildTask:
    """Navigate to Catherine and complete the upstream daily-reward flow."""

    DAILY_OPTIONS = ("每日", "每日委托", "Daily Commissions", "Daily")
    EXPEDITION_OPTIONS = ("探索", "探索派遣", "Expeditions", "Expedition")

    def __init__(
        self,
        ctx: GameContext,
        country: str | None = None,
        *,
        config: AdventurersGuildConfig | Mapping[str, Any] | None = None,
        daily_reward_party_name: str = "",
        only_do_once: bool = False,
        timeout_s: float = 180.0,
        encounter_timeout_s: float = 12.0,
        talk_skip_times: int = 10,
        log: Callable[[str], None] = print,
        api: GenshinApi | None = None,
    ):
        self.ctx = ctx
        self.log = log
        if isinstance(config, AdventurersGuildConfig):
            self.config = config
        else:
            values = dict(config or {})
            if country is not None:
                values.setdefault("country", country)
            values.setdefault("dailyRewardPartyName", daily_reward_party_name)
            values.setdefault("onlyDoOnce", only_do_once)
            values.setdefault("timeoutSeconds", timeout_s)
            values.setdefault("encounterTimeoutSeconds", encounter_timeout_s)
            values.setdefault("talkSkipTimes", talk_skip_times)
            self.config = AdventurersGuildConfig.from_mapping(raw=values)
        self.api = api or GenshinApi(ctx, log=log)
        self.interaction = InteractionPromptDetector(ctx, log=log)
        self._black_confirm: RecognitionObject | None = None
        self._returned_to_main = False

    @staticmethod
    def _cancelled(cancelled: Callable[[], bool] | None) -> bool:
        try:
            return bool(cancelled and cancelled())
        except Exception:
            return True

    def _log(self, message: str) -> None:
        self.log(f"[AdventurersGuild] {message}")

    def _return_main_ui(self) -> bool:
        method = getattr(self.api, "returnMainUi", None)
        if not callable(method):
            return True
        result = bool(method())
        if result:
            self._returned_to_main = True
        return result

    def _route(self) -> bool:
        route_resolver = getattr(self.api, "_poi_route", None)
        if callable(route_resolver):
            route_path = route_resolver("冒险家协会", self.config.country)
        else:
            route_path = (
                PROJECT_ROOT
                / "assets"
                / "pathing"
                / "poi"
                / f"冒险家协会_{self.config.country}.json"
            )
        route = PathingTask.load(route_path)
        # The upstream route has AutoPick enabled by default. The task owns
        # the interaction key and must not let a realtime trigger consume the
        # last route frame or press F before the dialogue is visible.
        route.realtime_triggers = {}
        executor = PathingExecutor(
            self.ctx,
            party_slots=getattr(self.api, "_party_slots", None),
            log=self.log,
        )
        ok = bool(executor.run(route))
        if not ok:
            self._log("冒险家协会路线执行失败")
        return ok

    @staticmethod
    def _is_talk(api: Any, frame: Any) -> bool:
        detector = getattr(api, "_is_talk_ui_frame", None)
        if callable(detector):
            try:
                bgr = getattr(frame, "bgr", frame)
                return bool(detector(bgr))
            except (AttributeError, TypeError, ValueError):
                return False
        return False

    def _press_catherine_until_talk(
        self,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        """Press the mapped interaction key until the Catherine dialogue opens."""

        for attempt in range(self.config.interaction_retries):
            if self._cancelled(cancelled):
                return False
            region = self.ctx.capture_region()
            if self._is_talk(self.api, region):
                return True
            try:
                prompt_visible = self.interaction.visible(region)
            except Exception as error:
                prompt_visible = False
                self._log(f"交互提示识别失败（继续按键）：{error}")
            # Match the upstream three-attempt contract: the prompt detector
            # is diagnostic, while the mapped F action is safe and must still
            # be retried when OCR misses a transition frame.
            self.ctx.input.key_press("F")
            self._log("与凯瑟琳交互" if prompt_visible else "尝试与凯瑟琳交互")
            self.ctx.sleep(500)

        try:
            return self._is_talk(self.api, self.ctx.capture_region())
        except Exception:
            return False

    def _choose_option(
        self,
        options: tuple[str, ...],
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        if self._cancelled(cancelled):
            return False
        chooser = getattr(self.api, "chooseTalkOption", None)
        if not callable(chooser):
            return False

        # Chinese is the default game locale. If it is not found, one English
        # retry keeps scripts usable on an English client without running the
        # full retry loop once for every localization.
        candidates = [options[0]] if options else []
        english = next(
            (option for option in options[1:] if str(option).isascii()),
            None,
        )
        if english and english not in candidates:
            candidates.append(english)
        for option in candidates:
            if self._cancelled(cancelled):
                return False
            try:
                if bool(chooser(
                    option,
                    skip_times=self.config.talk_skip_times,
                    is_orange=True,
                )):
                    return True
            except TypeError:
                # Small host doubles and older facades may only accept
                # positional parameters.
                if bool(chooser(option, self.config.talk_skip_times, True)):
                    return True
        return False

    def _black_confirm_recognition(self) -> RecognitionObject | None:
        if self._black_confirm is not None:
            return self._black_confirm
        if not BLACK_CONFIRM_ASSET.is_file():
            return None
        try:
            recognition = RecognitionObject.template_match(
                Mat.from_file(str(BLACK_CONFIRM_ASSET)),
                500,
                650,
                920,
                400,
            )
            recognition.threshold = 0.72
            self._black_confirm = recognition
        except (OSError, ValueError, RuntimeError) as error:
            self._log(f"黑色确认模板加载失败：{error}")
            self._black_confirm = None
        return self._black_confirm

    def _click_black_confirm(self) -> bool:
        """Click the daily-reward confirmation prompt if it is visible."""

        try:
            region = self.ctx.capture_region()
            recognition = self._black_confirm_recognition()
            if recognition is not None:
                hit = region.find(recognition)
                if hit.is_exist():
                    hit.click()
                    self._log("确认每日委托奖励提示")
                    self.ctx.sleep(500)
                    return True

            # OCR fallback is useful on translated clients and on a HUD scale
            # where the retained desktop template is too small.
            hits = region.find_multi(
                RecognitionObject.ocr(850, 550, 1000, 500),
                limit=30,
            )
            for hit in hits:
                if any(word in _compact(getattr(hit, "text", "")) for word in (
                    "确认", "確定", "yes", "confirm",
                )):
                    hit.click()
                    self._log("通过 OCR 确认每日委托奖励提示")
                    self.ctx.sleep(500)
                    return True
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            self._log(f"确认每日委托奖励提示失败：{error}")
        return False

    def _close_dialogue(self, cancelled: Callable[[], bool] | None) -> bool:
        if self._cancelled(cancelled):
            return False
        closer = getattr(self.api, "clickChatExitUntilMainUi", None)
        if callable(closer):
            try:
                result = bool(closer(retry_times=self.config.close_retries))
            except TypeError:
                result = bool(closer(self.config.close_retries))
            if result:
                self._returned_to_main = True
            return result
        return self._return_main_ui()

    def _reopen_catherine(self, cancelled: Callable[[], bool] | None) -> bool:
        if self._cancelled(cancelled):
            return False
        self.ctx.sleep(1200)
        self._returned_to_main = False
        return self._press_catherine_until_talk(cancelled)

    def _run_locked(
        self,
        cancelled: Callable[[], bool] | None,
        deadline: float,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "country": self.config.country,
            "onlyDoOnce": self.config.only_do_once,
            "partySwitched": False,
            "encounterPoints": None,
            "routed": False,
            "talkOpened": False,
            "dailyReward": "not_found",
            "dailyConfirmation": False,
            "expedition": "not_found",
            "expeditionCompleted": False,
            "returnedToMainUi": False,
        }
        try:
            if self._cancelled(cancelled):
                result["cancelled"] = True
                return result
            if time.monotonic() >= deadline:
                result["timeout"] = True
                return result

            if not self._return_main_ui():
                raise RuntimeError("无法回到主界面")

            if self.config.daily_reward_party_name:
                switcher = getattr(self.api, "switchParty", None)
                if not callable(switcher) or not bool(
                    switcher(self.config.daily_reward_party_name)
                ):
                    raise RuntimeError(
                        f"无法切换每日奖励好感队伍：{self.config.daily_reward_party_name}"
                    )
                result["partySwitched"] = True

            if not self.config.only_do_once:
                claimer = getattr(self.api, "claimEncounterPointsRewards", None)
                if callable(claimer):
                    # This step is best-effort, matching the desktop claim job:
                    # no available reward must not prevent daily commission and
                    # expedition handling.
                    result["encounterPoints"] = bool(
                        claimer(self.config.encounter_timeout_s)
                    )

            if time.monotonic() >= deadline or self._cancelled(cancelled):
                result["cancelled"] = self._cancelled(cancelled)
                result["timeout"] = not result.get("cancelled", False)
                return result

            result["routed"] = self._route()
            if not result["routed"]:
                raise RuntimeError("冒险家协会路线执行失败")
            if not self._press_catherine_until_talk(cancelled):
                raise RuntimeError("与凯瑟琳对话失败")
            result["talkOpened"] = True

            if self._choose_option(self.DAILY_OPTIONS, cancelled):
                result["dailyReward"] = "selected"
                self.ctx.sleep(800)
                result["dailyConfirmation"] = self._click_black_confirm()
                if not self._close_dialogue(cancelled):
                    raise RuntimeError("领取每日委托奖励后无法退出对话")
                if not self._reopen_catherine(cancelled):
                    raise RuntimeError("领取每日委托奖励后无法重新进入凯瑟琳对话")
            else:
                result["dailyReward"] = "unavailable"
                self._log("每日委托奖励未完成或已领取")

            if self._choose_option(self.EXPEDITION_OPTIONS, cancelled):
                result["expedition"] = "selected"
                self.ctx.sleep(500)
                result["expeditionCompleted"] = bool(
                    OneKeyExpeditionTask(
                        self.ctx,
                        log=self.log,
                    ).run(cancelled=cancelled)
                )
            else:
                result["expedition"] = "unavailable"
                self._log("探索派遣未完成或已领取")

            if time.monotonic() >= deadline or self._cancelled(cancelled):
                result["cancelled"] = self._cancelled(cancelled)
                result["timeout"] = not result.get("cancelled", False)
                return result

            result["returnedToMainUi"] = self._close_dialogue(cancelled)
            result["ok"] = bool(result["returnedToMainUi"])
            if not result["ok"]:
                raise RuntimeError("冒险家协会任务结束时无法返回主界面")
            return result
        except Exception as error:
            result["error"] = str(error)
            self._log(f"执行失败：{error}")
            return result
        finally:
            # Cancellation and an OCR exception can leave the character in a
            # dialogue/menu. Perform bounded cleanup while this task still
            # owns the input channel; never let a realtime trigger see that
            # transition frame.
            needs_cleanup = bool(
                result.get("routed")
                or result.get("talkOpened")
                or result.get("dailyReward") != "not_found"
                or result.get("expedition") != "not_found"
            )
            if not result.get("returnedToMainUi") and needs_cleanup:
                try:
                    # Cleanup is deliberately independent of the cancellation
                    # callback. A cancelled task must still release a dialog
                    # or menu before the trigger loop is restored.
                    result["returnedToMainUi"] = self._close_dialogue(None)
                    if not result["returnedToMainUi"]:
                        result["returnedToMainUi"] = self._return_main_ui()
                except Exception as error:
                    self._log(f"安全退出对话失败：{error}")

    def run(self, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.timeout_s
        try:
            with exclusive_realtime_triggers(self.ctx):
                return self._run_locked(cancelled, deadline)
        except Exception as error:
            self._log(f"触发器隔离失败：{error}")
            return {
                "ok": False,
                "country": self.config.country,
                "onlyDoOnce": self.config.only_do_once,
                "error": str(error),
            }


# Keep an upstream-friendly spelling for Python adapters and converted jobs.
GoToAdventurersGuildTask = AdventurersGuildTask


__all__ = [
    "AdventurersGuildConfig",
    "AdventurersGuildTask",
    "GoToAdventurersGuildTask",
]
