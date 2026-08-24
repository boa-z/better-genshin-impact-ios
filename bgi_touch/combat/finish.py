"""Battle-finish detection adapted from BetterGI's party-screen probe."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .hud import enemies_nearby
from ..vision.game_ui import is_main_ui


def _sample_ref(ctx, frame, ref_x: float, ref_y: float):
    x, y = ctx.transform.to_device(ref_x, ref_y)
    height, width = frame.shape[:2]
    ix = max(0, min(width - 1, round(x)))
    iy = max(0, min(height - 1, round(y)))
    crop = frame[max(0, iy - 1):min(height, iy + 2),
                 max(0, ix - 1):min(width, ix + 2)]
    b, g, r = crop.reshape(-1, 3).mean(axis=0)
    return float(r), float(g), float(b)


def is_party_setup_open(ctx, frame) -> bool:
    """Match BetterGI's party-loading yellow bar and adjacent white tile."""
    if not hasattr(frame, "shape") or getattr(frame, "ndim", 0) != 3:
        return False
    yellow = _sample_ref(ctx, frame, 790, 50)
    white = _sample_ref(ctx, frame, 768, 50)
    is_yellow = yellow[0] >= 200 and yellow[1] >= 200 and yellow[2] <= 100
    is_white = all(value >= 240 for value in white)
    return is_yellow and is_white


def _value(raw: Any, key: str, default=None):
    if isinstance(raw, Mapping):
        wanted = key.replace("_", "").casefold()
        for candidate, value in raw.items():
            if str(candidate).replace("_", "").casefold() == wanted:
                return value
    try:
        value = getattr(raw, key)
    except (AttributeError, TypeError):
        wanted = key.replace("_", "").casefold()
        for candidate in dir(raw) if raw is not None else ():
            if candidate.replace("_", "").casefold() == wanted:
                return getattr(raw, candidate)
        return default
    return default if value is None else value


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fast_params(value: Any) -> tuple[float, frozenset[str]]:
    interval = 5.0
    names = set()
    for part in str(value or "").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            interval = float(part)
        except ValueError:
            names.add(part)
    return interval, frozenset(names)


def _check_delays(value: Any) -> tuple[int, dict[str, int]]:
    default_ms = 400
    names: dict[str, int] = {}
    for part in str(value or "").split(";"):
        fields = [item.strip() for item in part.split(",") if item.strip()]
        try:
            if len(fields) == 1:
                default_ms = max(0, round(float(fields[0]) * 1000))
            elif len(fields) == 2:
                names[fields[0]] = max(0, round(float(fields[1]) * 1000))
        except ValueError:
            continue
    return default_ms, names


@dataclass(frozen=True)
class FightFinishConfig:
    fast_check_enabled: bool = False
    check_interval_s: float = 5.0
    check_names: frozenset[str] = frozenset()
    check_after_switch_avatar: bool = False
    delay_ms: int = 400
    character_delay_ms: dict[str, int] = field(default_factory=dict)
    detect_delay_ms: int = 400
    skip_when_enemy_visible: bool = False
    block_after_start_s: float = 0.0
    paimon_check_enabled: bool = True
    paimon_check_delay_ms: int = 75
    max_enemy_skip_checks: int = 5

    @classmethod
    def from_mapping(cls, raw: Any) -> "FightFinishConfig":
        interval, names = _fast_params(_value(raw, "fastCheckParams", ""))
        delay_ms, character_delays = _check_delays(
            _value(raw, "checkEndDelay", "0.4;钟离,1.4;")
        )
        paimon_delay = _float(_value(raw, "paimonEndCheckDelay", 0.075), 0.075)
        return cls(
            fast_check_enabled=_as_bool(_value(raw, "fastCheckEnabled", False)),
            check_interval_s=interval,
            check_names=names,
            check_after_switch_avatar=_as_bool(
                _value(raw, "checkAfterSwitchAvatar", False)
            ),
            delay_ms=delay_ms,
            character_delay_ms=character_delays,
            detect_delay_ms=max(0, round(
                _float(_value(raw, "beforeDetectDelay", 0.4), 0.4) * 1000
            )),
            skip_when_enemy_visible=_as_bool(
                _value(raw, "skipFightEndCheckWhenEnemyVisible", False)
            ),
            block_after_start_s=max(0.0, min(10.0, _float(
                _value(raw, "blockCheckBeforeBattleSeconds", 0), 0,
            ))),
            paimon_check_enabled=_as_bool(
                _value(raw, "paimonEndCheckEnabled", True), True,
            ),
            paimon_check_delay_ms=round(max(0.05, min(0.4, paimon_delay)) * 1000),
        )


class FightFinishDetector:
    """Use an enemy-bar fast path and party-screen opening as confirmation."""

    def __init__(
        self,
        ctx,
        config: FightFinishConfig | None = None,
        *,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.ctx = ctx
        self.config = config or FightFinishConfig()
        self.log = log
        self.clock = clock
        self.started_at = clock()
        self.last_check_at = self.started_at
        self.enemy_skip_count = 0

    def start_battle(self) -> None:
        self.started_at = self.clock()
        self.last_check_at = self.started_at
        self.enemy_skip_count = 0

    def should_fast_check(self, previous_character: str | None = None) -> bool:
        if not self.config.fast_check_enabled:
            return False
        elapsed = self.clock() - self.last_check_at
        return (
            self.config.check_interval_s > 0 and elapsed > self.config.check_interval_s
        ) or bool(previous_character and previous_character in self.config.check_names)

    def _capture_after(self, version: int | None, timeout_ms: int):
        capture_after = getattr(self.ctx, "capture_bgr_after_frame", None)
        if callable(capture_after) and version is not None:
            try:
                return capture_after(version, timeout_ms=timeout_ms)
            except Exception:
                pass
        return self.ctx.capture_bgr()

    def check(self, previous_character: str | None = None, *,
              after_switch: bool = False,
              cancelled: Callable[[], bool] | None = None) -> bool:
        now = self.clock()
        if self.config.block_after_start_s > 0 and (
            now - self.started_at < self.config.block_after_start_s
        ):
            self.last_check_at = now
            return False
        self.last_check_at = now
        if cancelled and cancelled():
            return False

        if self.config.skip_when_enemy_visible:
            if self.enemy_skip_count < self.config.max_enemy_skip_checks:
                if enemies_nearby(self.ctx):
                    self.enemy_skip_count += 1
                    self.log(
                        "[AutoFight] 敌人可见，跳过战斗结束复核"
                        f"（{self.enemy_skip_count}/{self.config.max_enemy_skip_checks}）"
                    )
                    return False
            self.enemy_skip_count = 0

        delay_ms = self.config.character_delay_ms.get(
            previous_character or "", self.config.delay_ms,
        )
        if after_switch:
            delay_ms = min(delay_ms, 50)
        if delay_ms:
            self.ctx.sleep(delay_ms)
        if cancelled and cancelled():
            return False

        device = getattr(self.ctx, "device", None)
        version = getattr(device, "last_frame_version", None)
        self.ctx.input.key_press("L")
        self.ctx.sleep(self.config.paimon_check_delay_ms)
        frame = self._capture_after(version, max(500, self.config.detect_delay_ms + 400))

        if self.config.paimon_check_enabled and is_main_ui(self.ctx, frame):
            self.log("[AutoFight] 队伍界面未打开，战斗仍在继续")
            return False

        remaining = max(
            0, self.config.detect_delay_ms - self.config.paimon_check_delay_ms,
        )
        if remaining:
            self.ctx.sleep(remaining)
            frame = self.ctx.capture_bgr()
        opened = is_party_setup_open(self.ctx, frame)
        if opened:
            self.ctx.input.key_press("X")
            self.ctx.sleep(300)
            self.log("[AutoFight] 队伍界面可打开，确认战斗结束")
            return True
        if not is_main_ui(self.ctx, frame):
            # L may have opened a transition frame, or a burst animation may
            # have hidden Paimon. Clean up the attempted panel without claiming
            # the battle ended unless the party-screen signature is present.
            self.ctx.input.key_press("X")
            self.ctx.sleep(200)
            self.log("[AutoFight] 队伍界面探测不确定，继续战斗")
        return False
