"""BetterGI ``PathingPartyConfig`` compatibility for the mobile path runner.

The desktop project passes ``Config.PathingConfig`` to every pathing project.
Older iOS code ignored that object because the first mobile pathing port only
needed the route JSON.  Keeping the config as a small, validated value object
lets route groups use the same settings without coupling the touch executor to
the WPF model.

The hurry implementation intentionally exposes only decisions that can be
made from the route state and the already-known party mapping.  It does not
request an extra screenshot for desktop-only vehicle/flight icon checks; the
executor falls back to ordinary movement when that evidence is unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..config_values import as_bool


HURRY_AVATARS = (
    "",
    "自动",
    "玛薇卡",
    "闲云",
    "桑多涅",
    "恰斯卡",
    "流浪者",
    "伊法",
    "希诺宁",
    "法尔伽",
    "夜兰",
)

# Keep the order from BetterGI's picker.  Empty/automatic entries are handled
# separately so they can never become an accidental character name.
AUTO_HURRY_PRIORITY = tuple(value for value in HURRY_AVATARS if value not in {"", "自动"})
RECOVER_TIMINGS = frozenset({"AnyWaypoint", "OnlyTeleport", "Never"})


def _folded_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key).replace("_", "").casefold(): item
        for key, item in value.items()
    }


def _value(raw: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    folded = _folded_mapping(raw)
    for name in names:
        key = name.replace("_", "").casefold()
        if key in folded:
            return folded[key]
    return default


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _slot_number(value: Any) -> int | None:
    try:
        slot = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return slot if 1 <= slot <= 4 else None


def _recover_timing(value: Any, legacy_only_teleport: Any = False) -> str:
    """Normalize BetterGI's enum and its retired boolean spelling."""

    if value is None:
        return "OnlyTeleport" if as_bool(legacy_only_teleport, False) else "AnyWaypoint"
    if isinstance(value, bool):
        return "OnlyTeleport" if value else "AnyWaypoint"
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None:
        return {0: "AnyWaypoint", 1: "OnlyTeleport", 2: "Never"}.get(
            numeric, "AnyWaypoint"
        )
    normalized = str(value).strip().casefold().replace("_", "")
    aliases = {
        "anywaypoint": "AnyWaypoint",
        "any": "AnyWaypoint",
        "任何路径点": "AnyWaypoint",
        "onlyteleport": "OnlyTeleport",
        "teleport": "OnlyTeleport",
        "只在传送点": "OnlyTeleport",
        "never": "Never",
        "none": "Never",
        "不回复": "Never",
    }
    return aliases.get(normalized, "AnyWaypoint")


_MISSING = object()


def _optional_name(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


@dataclass(frozen=True)
class AutoEatConfig:
    """Portable BetterGI ``PathingConfig.AutoEatConfig`` values.

    The desktop dispatcher resolves ``foodEffectType`` against these fields
    before creating ``AutoEatTask``.  Keeping the nested object separate from
    the route executor also lets old ScriptGroup JSON pass through unchanged.
    """

    default_atk_boosting_dish_name: str | None = "炸萝卜丸子"
    default_adventurers_dish_name: str | None = None
    default_def_boosting_dish_name: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | "AutoEatConfig" | None = None,
    ) -> "AutoEatConfig":
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            return cls()

        def read_name(*names: str, default: str | None = None) -> str | None:
            value = _value(raw, *names, default=_MISSING)
            if value is _MISSING:
                return default
            return _optional_name(value)

        return cls(
            default_atk_boosting_dish_name=read_name(
                "defaultAtkBoostingDishName",
                default=cls.default_atk_boosting_dish_name,
            ),
            default_adventurers_dish_name=read_name(
                "defaultAdventurersDishName",
                default=cls.default_adventurers_dish_name,
            ),
            default_def_boosting_dish_name=read_name(
                "defaultDefBoostingDishName",
                default=cls.default_def_boosting_dish_name,
            ),
        )


@dataclass(frozen=True)
class PathingPartyConfig:
    """Portable subset of BetterGI's pathing party configuration.

    Fields that affect Windows-only resurrection/food UI are retained for
    JSON compatibility where they are useful to callers, while the executor
    currently consumes the movement and trigger fields below.
    """

    enabled: bool = True
    auto_pick_enabled: bool = True
    party_name: str = ""
    is_visit_statue_before_switch_party: bool = False
    main_avatar_index: str = ""
    guardian_avatar_index: str = ""
    guardian_elemental_skill_second_interval: str = ""
    guardian_elemental_skill_long_press: bool = False
    js_script_use_enabled: bool = True
    solo_task_use_fight_enabled: bool = True
    skip_during: str = ""
    auto_skip_enabled: bool = True
    auto_run_enabled: bool = True
    auto_eat_enabled: bool = False
    auto_eat_config: AutoEatConfig = field(default_factory=AutoEatConfig)
    auto_fight_enabled: bool = True
    # BetterGI supplies a nested AutoFightConfig to pathing action handlers.
    # Keep its mapping intact so newer combat/pickup fields can flow through
    # without making this movement config depend on the whole combat model.
    auto_fight_config: dict[str, Any] = field(default_factory=dict)
    recover_timing: str = "AnyWaypoint"
    use_gadget_interval_ms: int = 0
    distance: int = 45
    approach_stop_distance: int = 25
    hurry_on_avatar: str = ""
    hurry_on_frame_interval: int = 100
    travel_mode: str = "精准靠近"
    switch_to_walk_enabled: bool = False
    mwk_jump_fly_enabled: bool = True
    mwk_jump_fly_distance: int = 75
    mwk_jump_fly_interval_seconds: float = 1.0
    mwk_disable_sprint_enabled: bool = False
    # Number of jump-fly actions after mounting that should receive a short
    # sprint input first.  BetterGI uses this for C6 Mavuika routes; zero
    # preserves the ordinary jump-fly cadence.
    mwk_jump_fly_sprint_count: int = 0

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | "PathingPartyConfig" | None = None,
    ) -> "PathingPartyConfig":
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            return cls()

        # Accept both a direct PathingConfig object and a whole ScriptGroup
        # mapping.  The latter is useful for callers that pass group.config
        # directly instead of extracting the nested object first.
        nested = _value(raw, "pathingConfig", default=None)
        if isinstance(nested, Mapping):
            raw = nested

        auto_eat_raw = _value(raw, "autoEatConfig", default=None)
        if not isinstance(auto_eat_raw, Mapping) and not isinstance(
            auto_eat_raw, AutoEatConfig
        ):
            auto_eat_raw = {}
        auto_fight_raw = _value(raw, "autoFightConfig", default=None)
        if not isinstance(auto_fight_raw, Mapping):
            auto_fight_raw = {}

        distance = max(1, _int(_value(raw, "distance", default=45), 45))
        approach = max(
            0,
            min(distance, _int(
                _value(raw, "approachStopDistance", default=25), 25
            )),
        )
        jump_distance = max(
            distance + 1,
            _int(_value(raw, "mwkJumpFlyDistance", default=75), 75),
        )
        frame_interval = max(
            1,
            min(150, _int(
                _value(raw, "hurryOnFrameInterval", default=100), 100
            )),
        )
        jump_interval = max(
            0.1,
            _float(_value(raw, "mwkJumpFlyIntervalSeconds", default=1), 1),
        )
        jump_sprint_count = max(
            0,
            _int(_value(raw, "mwkJumpFlySprintCount", default=0), 0),
        )
        hurry = str(_value(raw, "hurryOnAvatar", default="") or "").strip()
        # Unknown picker values should not silently select a character.  Keep
        # an empty value for execution, while preserving valid new characters
        # such as 法尔伽 and 夜兰.
        if hurry not in HURRY_AVATARS:
            hurry = ""
        travel_mode = str(
            _value(raw, "travelMode", default="精准靠近") or "精准靠近"
        ).strip() or "精准靠近"

        return cls(
            enabled=as_bool(_value(raw, "enabled", default=True), True),
            auto_pick_enabled=as_bool(
                _value(raw, "autoPickEnabled", default=True), True
            ),
            party_name=str(_value(raw, "partyName", default="") or "").strip(),
            is_visit_statue_before_switch_party=as_bool(
                _value(
                    raw,
                    "isVisitStatueBeforeSwitchParty",
                    default=False,
                ),
                False,
            ),
            main_avatar_index=str(
                _value(raw, "mainAvatarIndex", default="") or ""
            ).strip(),
            guardian_avatar_index=str(
                _value(raw, "guardianAvatarIndex", default="") or ""
            ).strip(),
            guardian_elemental_skill_second_interval=str(
                _value(raw, "guardianElementalSkillSecondInterval", default="") or ""
            ).strip(),
            guardian_elemental_skill_long_press=as_bool(
                _value(raw, "guardianElementalSkillLongPress", default=False), False
            ),
            js_script_use_enabled=as_bool(
                _value(raw, "jsScriptUseEnabled", default=True), True
            ),
            solo_task_use_fight_enabled=as_bool(
                _value(raw, "soloTaskUseFightEnabled", default=True), True
            ),
            skip_during=str(_value(raw, "skipDuring", default="") or "").strip(),
            auto_skip_enabled=as_bool(
                _value(raw, "autoSkipEnabled", default=True), True
            ),
            auto_run_enabled=as_bool(
                _value(raw, "autoRunEnabled", default=True), True
            ),
            auto_eat_enabled=as_bool(
                _value(raw, "autoEatEnabled", default=False), False
            ),
            auto_eat_config=AutoEatConfig.from_mapping(auto_eat_raw),
            auto_fight_enabled=as_bool(
                _value(raw, "autoFightEnabled", default=True), True
            ),
            auto_fight_config=dict(auto_fight_raw),
            recover_timing=_recover_timing(
                _value(raw, "recoverTiming", default=None),
                _value(raw, "onlyInTeleportRecover", default=False),
            ),
            use_gadget_interval_ms=max(
                0,
                _int(_value(raw, "useGadgetIntervalMs", default=0), 0),
            ),
            distance=distance,
            approach_stop_distance=approach,
            hurry_on_avatar=hurry,
            hurry_on_frame_interval=frame_interval,
            travel_mode=travel_mode,
            switch_to_walk_enabled=as_bool(
                _value(raw, "switchToWalkEnabled", default=False), False
            ),
            # Older BetterGI configurations used MwkFlyEnabled.  Prefer the
            # new field but accept the old spelling for route compatibility.
            mwk_jump_fly_enabled=as_bool(
                _value(
                    raw,
                    "mwkJumpFlyEnabled",
                    "mwkFlyEnabled",
                    default=True,
                ),
                True,
            ),
            mwk_jump_fly_distance=jump_distance,
            mwk_jump_fly_interval_seconds=jump_interval,
            mwk_disable_sprint_enabled=as_bool(
                _value(raw, "mwkDisableSprintEnabled", default=False), False
            ),
            mwk_jump_fly_sprint_count=jump_sprint_count,
        )

    @property
    def hurry_enabled(self) -> bool:
        return bool(self.hurry_on_avatar)

    @property
    def effective_hurry_frame_interval_ms(self) -> int:
        # Match the range used by the upstream executor.  The mobile path
        # loop does not poll faster merely because this value is lower; it is
        # an action scheduling floor and therefore cannot add screenshots.
        return max(5, min(150, int(self.hurry_on_frame_interval)))

    def _party_by_slot(self, party_slots: Mapping[str, int] | None) -> dict[int, str]:
        result: dict[int, str] = {}
        for raw_name, raw_slot in (party_slots or {}).items():
            slot = _slot_number(raw_slot)
            name = str(raw_name or "").strip()
            if slot is not None and name and slot not in result:
                result[slot] = name
        return result

    def resolve_hurry_avatar(
        self,
        party_slots: Mapping[str, int] | None,
    ) -> str | None:
        """Resolve the selected hurry character from a known party mapping."""

        by_slot = self._party_by_slot(party_slots)
        requested = self.hurry_on_avatar
        if not requested:
            return None
        if requested != "自动":
            return requested if requested in (party_slots or {}) else None

        main_slot = _slot_number(self.main_avatar_index)
        if main_slot is not None and by_slot.get(main_slot) in AUTO_HURRY_PRIORITY:
            return by_slot[main_slot]
        return next(
            (name for name in AUTO_HURRY_PRIORITY if name in (party_slots or {})),
            None,
        )

    def resolve_walk_avatar(
        self,
        party_slots: Mapping[str, int] | None,
        hurry_avatar: str | None,
    ) -> str | None:
        """Choose the first safe non-hurry party member for dismounting."""

        by_slot = self._party_by_slot(party_slots)
        main_slot = _slot_number(self.main_avatar_index)
        blocked = {
            value for value in {
                hurry_avatar,
                "玛薇卡", "希诺宁", "瓦雷莎", "茜特菈莉",
                "伊法", "恰斯卡", "玛拉妮", "基尼奇",
            }
            if value
        }
        if main_slot is not None:
            candidate = by_slot.get(main_slot)
            if candidate and candidate not in blocked:
                return candidate
        for slot in sorted(by_slot):
            candidate = by_slot[slot]
            if candidate not in blocked:
                return candidate
        # If every party member is a hurry/blacklisted character, preserve
        # the upstream fallback and use any member other than the hurry role.
        return next(
            (name for slot, name in sorted(by_slot.items()) if name != hurry_avatar),
            None,
        )


__all__ = [
    "AUTO_HURRY_PRIORITY",
    "AutoEatConfig",
    "HURRY_AVATARS",
    "PathingPartyConfig",
    "RECOVER_TIMINGS",
]
