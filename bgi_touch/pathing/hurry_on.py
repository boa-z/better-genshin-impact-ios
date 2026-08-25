"""Screenshot-free scheduling for BetterGI's pathing hurry abilities.

The Windows implementation uses several character-specific HUD icon models.
Those models are not reliable on every iOS capture scale, and asking for a
second frame from the scheduler would compete with the shared DeviceHub
capture loop.  This module therefore keeps the deterministic part of the
feature separate: party selection, cooldown pacing, jump cadence, and the
safe dismount decision.  The executor consumes the resulting actions while
continuing to use its normal movement frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping

from .party_config import PathingPartyConfig


@dataclass(frozen=True)
class HurryProfile:
    """Timing contract for one supported travel character."""

    skill_interval_s: float
    skill_hold_ms: int = 80
    jump_enabled: bool = False


# These intervals match the current combat asset where the character has a
# normal E cooldown.  The action scheduler deliberately leaves a safety
# margin for input latency and server-side cooldown display lag.
HURRY_PROFILES: dict[str, HurryProfile] = {
    "玛薇卡": HurryProfile(15.0, 80, True),
    "闲云": HurryProfile(12.0, 80, True),
    "桑多涅": HurryProfile(10.0),
    "恰斯卡": HurryProfile(6.5),
    "流浪者": HurryProfile(6.0),
    "伊法": HurryProfile(7.5),
    "希诺宁": HurryProfile(7.0),
    "法尔伽": HurryProfile(8.0, 800),
    "夜兰": HurryProfile(10.0, 800),
}


@dataclass(frozen=True)
class HurryAction:
    """Actions to apply to the current pathing frame."""

    press_skill: bool = False
    press_jump: bool = False
    sprint_jump: bool = False
    switch_to_walk: bool = False
    suppress_sprint: bool = False

    @property
    def handled(self) -> bool:
        return (
            self.press_skill
            or self.press_jump
            or self.sprint_jump
            or self.switch_to_walk
        )


class HurryOnController:
    """Make deterministic hurry decisions without owning a screenshot loop."""

    def __init__(
        self,
        config: PathingPartyConfig,
        party_slots: Mapping[str, int] | None,
        *,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.party_slots = dict(party_slots or {})
        self.log = log
        self.clock = clock
        self.avatar = config.resolve_hurry_avatar(self.party_slots)
        self.profile = HURRY_PROFILES.get(self.avatar or "")
        self._last_skill_at = float("-inf")
        self._last_jump_at = float("-inf")
        self._last_tick_at = float("-inf")
        self._mavika_sprint_jump_count = 0
        self._walk_switched = False
        self._started = False

    @property
    def enabled(self) -> bool:
        return (
            self.config.enabled
            and self.avatar is not None
            and self.profile is not None
        )

    @property
    def frame_interval_ms(self) -> int:
        return self.config.effective_hurry_frame_interval_ms

    @property
    def walk_avatar(self) -> str | None:
        return self.config.resolve_walk_avatar(self.party_slots, self.avatar)

    def start(self) -> str | None:
        """Return the selected avatar once per movement segment."""

        if not self.enabled or self._started:
            return None
        self._started = True
        self.log(f"[pathing] 赶路角色：{self.avatar}")
        return self.avatar

    def tick(
        self,
        *,
        distance: float,
        move_mode: str,
        now: float | None = None,
    ) -> HurryAction:
        """Return the next action for one already-captured pathing frame.

        ``frame_interval`` throttles this decision function only.  It never
        causes a new capture, which is important when WebUI/AutoPick and the
        route runner share one DeviceHub stream.
        """

        if not self.enabled:
            return HurryAction()
        now = self.clock() if now is None else float(now)
        if now - self._last_tick_at < self.frame_interval_ms / 1000:
            return HurryAction(
                suppress_sprint=(
                    self.avatar == "玛薇卡"
                    and self.config.mwk_disable_sprint_enabled
                    and distance > self.config.approach_stop_distance
                )
            )
        self._last_tick_at = now

        mode = str(move_mode or "").casefold()
        if mode not in {"run", "dash"}:
            return HurryAction()

        approaching = distance <= self.config.approach_stop_distance
        if approaching and self.config.switch_to_walk_enabled and not self._walk_switched:
            self._walk_switched = True
            self.log(f"[pathing] {self.avatar} 接近路点，切换步行角色")
            return HurryAction(switch_to_walk=True)

        if distance <= self.config.distance:
            return HurryAction(
                suppress_sprint=(
                    self.avatar == "玛薇卡"
                    and self.config.mwk_disable_sprint_enabled
                )
            )

        press_skill = now - self._last_skill_at >= self.profile.skill_interval_s
        press_jump = False
        sprint_jump = False
        if press_skill:
            self._last_skill_at = now
            self.log(f"[pathing] {self.avatar} 赶路技能")
        if self.profile.jump_enabled and self.config.mwk_jump_fly_enabled:
            jump_interval = self.config.mwk_jump_fly_interval_seconds
            if distance >= self.config.mwk_jump_fly_distance:
                press_jump = now - self._last_jump_at >= jump_interval
                if press_jump:
                    self._last_jump_at = now
                    self.log(f"[pathing] {self.avatar} 跳跃赶路")
                    if (
                        self.avatar == "玛薇卡"
                        and not self.config.mwk_disable_sprint_enabled
                        and self._mavika_sprint_jump_count
                        < self.config.mwk_jump_fly_sprint_count
                        and distance > self.config.mwk_jump_fly_distance * 1.3
                    ):
                        self._mavika_sprint_jump_count += 1
                        sprint_jump = True
                        self.log(
                            "[pathing] 玛薇卡跳飞前冲刺 "
                            f"{self._mavika_sprint_jump_count}/"
                            f"{self.config.mwk_jump_fly_sprint_count}"
                        )

        return HurryAction(
            press_skill=press_skill,
            press_jump=press_jump,
            sprint_jump=sprint_jump,
            suppress_sprint=(
                self.avatar == "玛薇卡"
                and self.config.mwk_disable_sprint_enabled
            ),
        )


__all__ = ["HURRY_PROFILES", "HurryAction", "HurryOnController", "HurryProfile"]
