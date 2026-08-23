"""BetterGI SkillCd trigger for the iOS runtime.

The trigger consumes frames already owned by :class:`TriggerLoop`.  Input
edges come from :class:`InputSimulator`, so switching characters or using E
never starts a competing DeviceHub capture request.  Cooldowns are stored as
monotonic deadlines; slow Wi-Fi frames therefore do not make timers drift.
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..vision.game_ui import is_main_ui
from ..vision.ocr import get_ocr

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARTY_PATH = ROOT / "config" / "party.json"
AVATAR_DATA_PATH = ROOT / "assets" / "data" / "combat_avatar.json"
READY_THRESHOLD_S = 0.8
LEAVE_DEBOUNCE_S = 0.8
RECENT_SKILL_S = 1.1


def _value(value: Any, key: str, default: Any = None) -> Any:
    """Read BetterGI camel/Pascal/snake-case configuration fields."""
    wanted = key.replace("_", "").casefold()
    if isinstance(value, Mapping):
        for candidate, result in value.items():
            if str(candidate).replace("_", "").casefold() == wanted:
                return result
        return default
    for candidate in dir(value):
        if candidate.replace("_", "").casefold() == wanted:
            result = getattr(value, candidate)
            return default if result is None else result
    return default


def _clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        return min(high, max(low, float(value)))
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=1)
def _avatar_cooldowns() -> dict[str, float]:
    try:
        records = json.loads(AVATAR_DATA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    result: dict[str, float] = {}
    for record in records if isinstance(records, list) else []:
        try:
            cooldown = float(record.get("skillCD", 0))
        except (AttributeError, TypeError, ValueError):
            continue
        if cooldown <= 0:
            continue
        names = [record.get("name"), *(record.get("alias") or [])]
        for name in names:
            if name:
                result[str(name)] = cooldown
    return result


def _party_names(ctx: Any, explicit: Mapping[str, int] | None = None) -> list[str]:
    slots = explicit or getattr(ctx, "party_slots", None)
    if not slots:
        try:
            slots = json.loads(DEFAULT_PARTY_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            slots = {}
    ordered: dict[int, str] = {}
    for name, raw_slot in (slots.items() if isinstance(slots, Mapping) else []):
        try:
            slot = int(raw_slot)
        except (TypeError, ValueError):
            continue
        if 1 <= slot <= 4 and str(name).strip():
            ordered[slot] = str(name).strip()
    return [ordered.get(slot, "") for slot in range(1, 5)]


def _parse_custom_rules(values: Any) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    if isinstance(values, Mapping):
        iterable = [
            {"roleName": name, "cdValueText": cooldown}
            for name, cooldown in values.items()
        ]
    else:
        iterable = values if isinstance(values, (list, tuple)) else []
    for item in iterable:
        name = str(_value(item, "roleName", "") or "").strip()
        if not name or name in result:
            continue
        raw = _value(item, "cdValueText", _value(item, "cdValue", None))
        if raw is None or str(raw).strip() == "":
            result[name] = None
            continue
        try:
            result[name] = max(0.0, float(raw))
        except (TypeError, ValueError):
            result[name] = None
    return result


class SkillCdState:
    """Thread-safe latest SkillCd overlay snapshot."""

    def __init__(self, config: dict[str, Any], clock: Callable[[], float]):
        self._lock = threading.Lock()
        self._clock = clock
        self._sequence = 0
        self._updated_at = clock()
        self._data: dict[str, Any] = {
            "active": True,
            "scene": "waiting",
            "visible": False,
            "activeSlot": 1,
            "team": [],
            "config": copy.deepcopy(config),
        }

    def update(self, **values: Any) -> None:
        with self._lock:
            changed = any(self._data.get(key) != value for key, value in values.items())
            self._data.update(copy.deepcopy(values))
            self._updated_at = self._clock()
            if changed:
                self._sequence += 1

    def deactivate(self) -> None:
        self.update(active=False, scene="disabled", visible=False)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = copy.deepcopy(self._data)
            now = self._clock()
            hide_when_zero = bool(result.get("config", {}).get("hideWhenZero", False))
            for avatar in result.get("team", []):
                deadline = float(avatar.pop("deadline", 0.0) or 0.0)
                remaining = max(0.0, deadline - now)
                avatar["remaining"] = round(remaining, 1)
                avatar["ready"] = remaining <= READY_THRESHOLD_S
                avatar["visible"] = bool(avatar.get("name")) and (
                    not hide_when_zero or remaining > 0
                )
            result["sequence"] = self._sequence
            result["ageMs"] = round(max(0.0, now - self._updated_at) * 1000)
            return result


class SkillCdTrigger:
    """Track four elemental-skill cooldowns with BetterGI semantics."""

    name = "SkillCd"

    def __init__(
        self,
        ctx: Any,
        *,
        party_slots: Mapping[str, int] | None = None,
        custom_cd_list: Any = None,
        trigger_on_skill_use: bool = False,
        hide_when_zero: bool = False,
        p_x: float = 1520.0,
        p_y: float = 245.0,
        gap: float = 91.2,
        scale: float = 1.0,
        background_normal_color: str = "#FFFFFFFF",
        text_normal_color: str = "#DA4A23FF",
        background_ready_color: str = "#FFFFFFFF",
        text_ready_color: str = "#5DCC17FF",
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
        main_ui_detector: Callable[[Any, np.ndarray], bool] = is_main_ui,
        ocr: Any = None,
    ):
        self.ctx = ctx
        self.enabled = True
        self.log = log
        self._clock = clock
        self._main_ui_detector = main_ui_detector
        self._ocr = ocr
        self._lock = threading.RLock()
        self._deadlines = [0.0] * 4
        self._team = _party_names(ctx, party_slots)
        self._custom_rules = _parse_custom_rules(custom_cd_list)
        self._last_frame: np.ndarray | None = None
        self._last_frame_at = 0.0
        self._last_e_at = float("-inf")
        self._raw_in_context = False
        self._was_visible_context = False
        self._leave_at: float | None = None
        self._active_slot = int(getattr(ctx.input, "_active_slot", 1))
        self.trigger_on_skill_use = bool(trigger_on_skill_use)
        self.hide_when_zero = bool(hide_when_zero)
        self.config = {
            "pX": _clamp(p_x, 0.0, 1920.0, 1520.0),
            "pY": _clamp(p_y, 0.0, 1080.0, 245.0),
            "gap": _clamp(gap, 0.0, 200.0, 91.2),
            "scale": _clamp(scale, 0.0, 10.0, 1.0),
            "backgroundNormalColor": str(background_normal_color or "#FFFFFFFF"),
            "textNormalColor": str(text_normal_color or "#DA4A23FF"),
            "backgroundReadyColor": str(background_ready_color or "#FFFFFFFF"),
            "textReadyColor": str(text_ready_color or "#5DCC17FF"),
            "hideWhenZero": self.hide_when_zero,
            "triggerOnSkillUse": self.trigger_on_skill_use,
        }
        self.state = SkillCdState(self.config, clock)
        subscribe = getattr(ctx.input, "subscribe", None)
        self._unsubscribe = subscribe(self._on_input) if callable(subscribe) else lambda: None
        self._publish("waiting", False, self._clock())

    def on_frame(self, region: Any) -> None:
        if not self.enabled:
            return
        now = self._clock()
        frame = region.bgr
        raw_in_context = bool(self._main_ui_detector(self.ctx, frame))
        with self._lock:
            self._raw_in_context = raw_in_context
            if raw_in_context:
                self._leave_at = None
                visible_context = True
            else:
                if self._was_visible_context and self._leave_at is None:
                    self._leave_at = now
                visible_context = (
                    self._leave_at is not None
                    and now - self._leave_at < LEAVE_DEBOUNCE_S
                )
            self._was_visible_context = visible_context
            self._active_slot = int(getattr(self.ctx.input, "_active_slot", self._active_slot))
            self._last_frame = frame.copy()
            captured_at = float(getattr(self.ctx, "_last_frame_at", 0.0) or 0.0)
            self._last_frame_at = captured_at if 0 < captured_at <= now else now
            scene = "gameplay" if visible_context else "other"
            self._publish(scene, visible_context and self._full_team(), now)

    def _on_input(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        now = float(event.get("timestamp", self._clock()))
        event_type = str(event.get("type", ""))
        key = str(event.get("key", "")).upper()
        with self._lock:
            if event_type in ("key_press", "key_down") and key == "E":
                self._last_e_at = now
                if self.trigger_on_skill_use and self._raw_in_context:
                    self._record_slot(self._active_slot, now, allow_fallback=True)
            elif event_type == "party_switch" and self._raw_in_context:
                from_slot = int(event.get("from_slot", self._active_slot))
                to_slot = int(event.get("to_slot", self._active_slot))
                self._record_slot(
                    from_slot,
                    now,
                    allow_fallback=now - self._last_e_at < RECENT_SKILL_S,
                )
                if 1 <= to_slot <= 4:
                    self._active_slot = to_slot
            else:
                return
            self._publish("gameplay", self._full_team(), now)

    def _record_slot(self, slot: int, now: float, *, allow_fallback: bool) -> None:
        if not 1 <= slot <= 4:
            return
        cooldown = self._recognize_skill_cd(now)
        if cooldown is None and allow_fallback:
            cooldown = self._fallback_cd(slot)
        if cooldown is not None and cooldown > 0:
            self._deadlines[slot - 1] = now + cooldown

    def _recognize_skill_cd(self, now: float) -> float | None:
        frame = self._last_frame
        if frame is None:
            return None
        height, width = frame.shape[:2]
        try:
            nx, ny = self.ctx.layout.buttons["skill"]
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        center_x, center_y = float(nx) * width, float(ny) * height
        size = max(36, round(height * 0.075))
        left = max(0, round(center_x - size / 2))
        top = max(0, round(center_y - size / 2))
        right = min(width, left + size)
        bottom = min(height, top + size)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        white = cv2.inRange(crop, (230, 230, 230), (255, 255, 255))
        enlarged = cv2.resize(white, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        source = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)
        try:
            items = (self._ocr or get_ocr()).recognize(source)
        except Exception as error:
            self.log(f"[SkillCD] OCR 失败（使用兜底）：{error}")
            return None
        text = " ".join(str(getattr(item, "text", "")) for item in items)
        match = re.search(r"\d+(?:\.\d+)?", text)
        if match is None:
            return None
        try:
            value = float(match.group(0)) - max(0.0, now - self._last_frame_at)
        except ValueError:
            return None
        return value if 0 < value < 60 else None

    def _fallback_cd(self, slot: int) -> float | None:
        name = self._team[slot - 1] if 1 <= slot <= len(self._team) else ""
        if not name:
            return None
        default = _avatar_cooldowns().get(name)
        if name not in self._custom_rules:
            return default
        custom = self._custom_rules[name]
        return default if custom is None else custom

    def _full_team(self) -> bool:
        return len(self._team) == 4 and all(self._team)

    def _publish(self, scene: str, visible: bool, now: float) -> None:
        team = []
        for index, name in enumerate(self._team):
            team.append({
                "slot": index + 1,
                "name": name,
                "deadline": self._deadlines[index],
            })
        self.state.update(
            active=True,
            scene=scene,
            visible=bool(visible),
            activeSlot=self._active_slot,
            team=team,
            config=self.config,
        )

    def close(self) -> None:
        if not self.enabled:
            return
        self.enabled = False
        self._unsubscribe()
        self.state.deactivate()
