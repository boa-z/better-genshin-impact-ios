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

import math
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

FLIGHT_HURRY_AVATARS = frozenset({"恰斯卡", "伊法", "流浪者"})

# BetterGI lets vehicle-like hurry characters keep running through a dense
# route, but asks them to dismount before a sharp turn or a non-running
# waypoint.  Keep the same per-character thresholds in the mobile decision
# layer; the executor supplies the route context and the current frame only.
TURN_ANGLE_THRESHOLDS: dict[str, float] = {
    "桑多涅": 45.0,
    "恰斯卡": 45.0,
    "伊法": 45.0,
    "流浪者": 45.0,
    "玛薇卡": 60.0,
    "闲云": 120.0,
    "希诺宁": 120.0,
    "法尔伽": 120.0,
    "夜兰": 120.0,
}


@dataclass(frozen=True)
class HurryAction:
    """Actions to apply to the current pathing frame."""

    press_skill: bool = False
    press_jump: bool = False
    sprint_jump: bool = False
    switch_to_walk: bool = False
    suppress_sprint: bool = False
    hold_sprint: bool = False
    stop_flying: bool = False

    @property
    def handled(self) -> bool:
        return (
            self.press_skill
            or self.press_jump
            or self.sprint_jump
            or self.switch_to_walk
            or self.stop_flying
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
        self._flight_active = False
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
        next_distance: float | None = None,
        next_type: str | None = None,
        next_move_mode: str | None = None,
        current_type: str | None = None,
        current_action: str | None = None,
        turn_angle: float = 0.0,
        motion_status: str | None = None,
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

        motion = str(motion_status or "").strip().casefold()
        if self.avatar in FLIGHT_HURRY_AVATARS:
            if motion in {"fly", "flying", "1"}:
                self._flight_active = True
            elif motion in {"normal", "0", "landed"}:
                self._flight_active = False

        approaching = self._should_approach(
            distance,
            next_distance=next_distance,
            next_type=next_type,
            next_move_mode=next_move_mode,
            current_type=current_type,
            current_action=current_action,
            turn_angle=turn_angle,
        )
        if approaching and self._flight_active:
            self._flight_active = False
            switch_to_walk = (
                self.config.switch_to_walk_enabled and not self._walk_switched
            )
            if switch_to_walk:
                self._walk_switched = True
            self.log(f"[pathing] {self.avatar} 接近路点，结束飞行")
            return HurryAction(
                switch_to_walk=switch_to_walk,
                stop_flying=True,
            )
        if approaching and self.config.switch_to_walk_enabled and not self._walk_switched:
            self._walk_switched = True
            self.log(f"[pathing] {self.avatar} 接近路点，切换步行角色")
            return HurryAction(switch_to_walk=True)

        if self._flight_active:
            # Chasca/Ifa/Wanderer need a sustained sprint input while their
            # flight prompt is visible. The executor keeps W and this sprint
            # state alive through the same DeviceHub lease; no new frame is
            # requested here.
            return HurryAction(
                suppress_sprint=True,
                hold_sprint=True,
            )

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

    def _should_approach(
        self,
        distance: float,
        *,
        next_distance: float | None,
        next_type: str | None,
        next_move_mode: str | None,
        current_type: str | None,
        current_action: str | None,
        turn_angle: float,
    ) -> bool:
        """Mirror BetterGI's precise/continuous dismount boundary.

        The Windows implementation reads the next route point from the
        active path and uses it to decide whether a hurry character should
        pass the current point.  The old mobile controller only compared the
        current distance, which made ``连续赶路`` stop at every point and
        caused overshoot when a non-running segment followed.  All inputs are
        route metadata or values derived from the current frame; this method
        never captures a new screenshot.
        """
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(distance):
            return False

        effective_stop = min(
            self.config.approach_stop_distance,
            self.config.distance,
        )
        if self.config.travel_mode == "精准靠近":
            return distance < effective_stop

        if self.config.travel_mode != "连续赶路":
            # Unknown values should remain conservative and preserve the
            # mobile runner's original precise-stop behaviour.
            return distance < effective_stop

        try:
            angle = float(turn_angle)
        except (TypeError, ValueError):
            angle = 0.0
        if not math.isfinite(angle):
            angle = 0.0
        threshold = TURN_ANGLE_THRESHOLDS.get(self.avatar or "", 120.0)

        next_mode = str(next_move_mode or "").strip().casefold()
        next_kind = str(next_type or "").strip().casefold()
        current_kind = str(current_type or "").strip().casefold()
        action = str(current_action or "").strip().casefold()
        next_is_run = next_mode in {"run", "dash"}
        boundary = (
            (next_distance is not None and next_distance < 25)
            or next_kind == "target"
            or next_distance is None
            or not next_is_run
            or action in {"fight", "combat_script"}
            or current_kind == "target"
            or angle >= threshold
        )
        return distance < max(effective_stop, 15) and boundary


__all__ = [
    "HURRY_PROFILES",
    "FLIGHT_HURRY_AVATARS",
    "HurryAction",
    "HurryOnController",
    "HurryProfile",
    "TURN_ANGLE_THRESHOLDS",
]
