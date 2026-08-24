"""Post-fight material pickup shared by AutoFight and pathing callers."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from .dsl import CombatCommand, CombatExecutor


def _value(raw: Any, key: str, default: Any = None) -> Any:
    if isinstance(raw, Mapping):
        wanted = key.replace("_", "").casefold()
        for candidate, value in raw.items():
            if str(candidate).replace("_", "").casefold() == wanted:
                return value
        return default
    try:
        value = getattr(raw, key)
    except (AttributeError, TypeError):
        wanted = key.replace("_", "").casefold()
        for candidate in dir(raw) if raw is not None else ():
            if str(candidate).replace("_", "").casefold() == wanted:
                return getattr(raw, candidate)
        return default
    return default if value is None else value


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().casefold()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    return default


@dataclass(frozen=True)
class PostFightPickupConfig:
    """Subset of BetterGI AutoFight pickup options usable on iOS."""

    kazuha_pickup_enabled: bool = True
    pick_drops_after_fight_enabled: bool = False
    pick_drops_after_fight_seconds: int = 15
    exp_based_pickup_enabled: bool = False
    battle_threshold_for_loot: int = -1
    kazuha_party_name: str = ""
    qin_double_pick_up: bool = False

    @classmethod
    def from_mapping(cls, raw: Any) -> "PostFightPickupConfig":
        seconds = _value(raw, "pickDropsAfterFightSeconds", 15)
        threshold = _value(raw, "battleThresholdForLoot", -1)
        try:
            seconds = max(0, min(120, int(seconds)))
        except (TypeError, ValueError):
            seconds = 15
        try:
            threshold = int(threshold)
        except (TypeError, ValueError):
            threshold = -1
        return cls(
            kazuha_pickup_enabled=_bool(
                _value(raw, "kazuhaPickupEnabled", True), True,
            ),
            pick_drops_after_fight_enabled=_bool(
                _value(raw, "pickDropsAfterFightEnabled", False), False,
            ),
            pick_drops_after_fight_seconds=seconds,
            exp_based_pickup_enabled=_bool(
                _value(raw, "expBasedPickupEnabled", False), False,
            ),
            battle_threshold_for_loot=threshold,
            kazuha_party_name=str(_value(raw, "kazuhaPartyName", "") or "").strip(),
            qin_double_pick_up=_bool(
                _value(raw, "qinDoublePickUp", False), False,
            ),
        )


@dataclass(frozen=True)
class PostFightPickupResult:
    elite_detected: bool | None
    picker: str | None
    scan_requested: bool
    completed: bool


class PostFightPickup:
    """Execute the safe iOS equivalent of BetterGI's post-fight pickup.

    The native AutoPick trigger is temporarily added to the existing shared
    ``TriggerLoop`` for scan pickup.  No task-local screenshot producer is
    created; if the trigger loop is unavailable, the fallback only replays
    the mapped interaction/camera inputs.
    """

    _PICKER_ALIASES = {
        "枫原万叶": {"枫原万叶", "万叶", "kazuha"},
        "琴": {"琴", "jean"},
    }

    def __init__(
        self,
        ctx,
        *,
        party_slots: Mapping[str, int] | None = None,
        config: PostFightPickupConfig | None = None,
        log: Callable[[str], None] = print,
        executor: CombatExecutor | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.ctx = ctx
        self.party_slots = dict(party_slots or getattr(ctx, "party_slots", {}) or {})
        self.config = config or PostFightPickupConfig()
        self.log = log
        self.executor = executor
        self.clock = clock

    @classmethod
    def _find_picker(cls, party_slots: Mapping[str, int]) -> str | None:
        for canonical, aliases in cls._PICKER_ALIASES.items():
            for name in party_slots:
                if str(name).strip().casefold() in {
                    value.casefold() for value in aliases
                }:
                    return str(name)
        return None

    def _executor(self) -> CombatExecutor:
        if self.executor is None:
            self.executor = CombatExecutor.for_context(
                self.ctx, party_slots=self.party_slots, log=self.log,
            )
        return self.executor

    def _use_picker(self, picker: str, cancelled: Callable[[], bool] | None) -> bool:
        if cancelled and cancelled():
            return False
        executor = self._executor()
        try:
            executor.switch_to(picker)
            # The profile maps E to the mobile elemental-skill button. A
            # charged Kazuha/Jean skill is the portable equivalent of the
            # upstream mouse/keyboard pickup macro.
            executor.exec(CombatCommand("e", ["hold"]))
            executor.exec(CombatCommand("attack", ["0.6"]))
            if picker.casefold() in {"琴", "jean"} and self.config.qin_double_pick_up:
                self.ctx.sleep(700)
                executor.exec(CombatCommand("e", ["hold"]))
                executor.exec(CombatCommand("attack", ["0.6"]))
            self.ctx.sleep(900)
            self.log(f"[AutoFight] 使用 {picker} 战技拾取战斗掉落")
            return True
        except Exception as error:
            self.log(f"[AutoFight] {picker} 战后拾取失败：{error}")
            return False
        finally:
            try:
                self.ctx.input.release_all()
            except Exception:
                pass

    def _scan_drops(self, cancelled: Callable[[], bool] | None) -> bool:
        seconds = self.config.pick_drops_after_fight_seconds
        if not self.config.pick_drops_after_fight_enabled or seconds <= 0:
            return False

        loop = None
        previous = None
        owns_trigger = False
        try:
            loop = getattr(self.ctx, "triggers", None)
            if loop is not None:
                previous = list(loop.triggers)
                if loop.get("AutoPick") is None:
                    self.ctx.enable_trigger("AutoPick")
                    owns_trigger = True
        except Exception as error:
            self.log(f"[AutoFight] 战后拾取触发器未能启用，使用直接交互扫描：{error}")

        deadline = self.clock() + seconds
        try:
            while self.clock() < deadline:
                if cancelled and cancelled():
                    break
                self.ctx.input.key_press("F")
                self.ctx.input.move_camera_by(180, 0)
                self.ctx.sleep(700)
            return True
        finally:
            if owns_trigger and loop is not None and previous is not None:
                try:
                    loop.replace(previous)
                    # enable_trigger() starts the shared loop even when no
                    # trigger was active before this scan. Do not leave an
                    # idle screenshot producer behind after restoring an empty
                    # trigger list.
                    if not previous:
                        loop.stop()
                except Exception as error:
                    self.log(f"[AutoFight] 恢复战后拾取触发器失败：{error}")

    def run(
        self,
        *,
        elite_detected: bool | None = None,
        battle_count: int = 1,
        cancelled: Callable[[], bool] | None = None,
    ) -> PostFightPickupResult:
        config = self.config
        if cancelled and cancelled():
            return PostFightPickupResult(elite_detected, None, False, False)

        eligible = True
        if config.exp_based_pickup_enabled:
            eligible = bool(elite_detected)
            if not eligible:
                self.log("[AutoFight] 未检测到精英经验图标，跳过战后聚怪拾取")
        if config.battle_threshold_for_loot >= 2 and (
            int(battle_count) < config.battle_threshold_for_loot
        ):
            eligible = False
            self.log(
                f"[AutoFight] 战斗人次（{battle_count}）低于拾取阈值 "
                f"（{config.battle_threshold_for_loot}），跳过战后聚怪拾取"
            )

        picker = self._find_picker(self.party_slots) if (
            config.kazuha_pickup_enabled and eligible
        ) else None
        completed = False
        if picker is not None:
            completed = self._use_picker(picker, cancelled)
        elif config.kazuha_pickup_enabled and config.kazuha_party_name and eligible:
            self.log(
                f"[AutoFight] 未找到万叶/琴槽位，暂不能自动切换拾取队伍："
                f"{config.kazuha_party_name}"
            )

        scan_requested = self._scan_drops(cancelled)
        return PostFightPickupResult(elite_detected, picker, scan_requested, completed)
