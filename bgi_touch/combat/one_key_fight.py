"""BetterGI 一键战斗宏的 iOS 触控实现。

宏文件格式与 ``User/avatar_macro.json`` 兼容；每个角色可提供 1-5 套 DSL，
并用 ``round(1,3-5),...|...`` 限定当前片段的执行轮次。
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .dsl import CombatCommand, CombatExecutor, _split_commands


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MACRO_PATH = ROOT / "config" / "avatar_macro_default.json"
USER_MACRO_PATH = ROOT / "config" / "avatar_macro.json"

HOLD_ON_MODE = "按住时重复(新)"
HOLD_FINISH_MODE = "按住时重复(旧)"
TICK_MODE = "触发"
MODES = frozenset({HOLD_ON_MODE, HOLD_FINISH_MODE, TICK_MODE})


@dataclass(frozen=True)
class AvatarMacro:
    name: str
    scripts: tuple[str, str, str, str, str]
    priority: int = 0

    def script(self, global_priority: int) -> str:
        priority = self.priority if 1 <= self.priority <= 5 else global_priority
        if not 1 <= priority <= 5:
            priority = 1
        return self.scripts[priority - 1]


def _key(mapping: Mapping[str, Any], wanted: str, default: Any = None) -> Any:
    normalized = wanted.replace("_", "").casefold()
    for key, value in mapping.items():
        if str(key).replace("_", "").casefold() == normalized:
            return value
    return default


def _split_top_level(text: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("一键宏指令括号不匹配")
        if char == delimiter and depth == 0:
            value = "".join(current).strip()
            if value:
                parts.append(value)
            current = []
        else:
            current.append(char)
    if depth != 0:
        raise ValueError("一键宏指令括号不匹配")
    value = "".join(current).strip()
    if value:
        parts.append(value)
    return parts


def _rounds(params: list[str]) -> tuple[int, ...]:
    if not params:
        raise ValueError("round 必须指定执行轮次，例如 round(1,3-5)")
    values: set[int] = set()
    for raw in params:
        value = raw.strip()
        if "-" in value:
            pieces = [part.strip() for part in value.split("-")]
            if len(pieces) != 2:
                raise ValueError(f"round 范围无效：{raw}")
            start, end = (int(part) for part in pieces)
            if start <= 0 or start > end:
                raise ValueError(f"round 范围无效：{raw}")
            values.update(range(start, end + 1))
        else:
            number = int(value)
            if number <= 0:
                raise ValueError(f"round 轮次必须大于 0：{raw}")
            values.add(number)
    return tuple(sorted(values))


def _parse_command(raw: str) -> CombatCommand:
    match = re.fullmatch(r"([\w一-鿿]+)\s*(?:\((.*)\))?", raw.strip())
    if match is None:
        raise ValueError(f"无法解析一键宏指令：{raw}")
    params = [part.strip() for part in _split_commands(match.group(2) or "")]
    return CombatCommand(match.group(1).casefold(), params)


def parse_one_key_script(text: str) -> list[CombatCommand]:
    """Parse BetterGI avatar macro DSL, including per-clause round filters."""
    normalized = str(text).replace("（", "(").replace("）", ")").replace("，", ",")
    commands: list[CombatCommand] = []
    for raw_line in re.split(r"[\r\n;]+", normalized):
        line = re.sub(r"(//|#).*$", "", raw_line).strip()
        if not line:
            continue
        for clause in _split_top_level(line, "|"):
            parsed = [_parse_command(value) for value in _split_commands(clause)]
            activating: tuple[int, ...] = ()
            if parsed and parsed[0].action == "round":
                activating = _rounds(parsed.pop(0).params)
            for command in parsed:
                command.activating_rounds = activating
                commands.append(command)
    return commands


def load_avatar_macros(path: str | Path, global_priority: int = 1) -> dict[str, list[CombatCommand]]:
    source = Path(path).expanduser()
    raw = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("avatar_macro.json 顶层必须是数组")
    result: dict[str, list[CombatCommand]] = {}
    for record in raw:
        if not isinstance(record, Mapping):
            continue
        name = str(_key(record, "name", "")).strip()
        if not name:
            continue
        try:
            priority = int(_key(record, "macroPriority", 0) or 0)
        except (TypeError, ValueError):
            priority = 0
        macro = AvatarMacro(
            name=name,
            scripts=tuple(
                str(_key(record, f"scriptContent{index}", "") or "")
                for index in range(1, 6)
            ),  # type: ignore[arg-type]
            priority=priority,
        )
        script = macro.script(global_priority)
        if script.strip():
            result[name] = parse_one_key_script(script)
    return result


class OneKeyFightTask:
    """Thread-safe hotkey state machine mirroring BetterGI OneKeyFightTask."""

    def __init__(
        self,
        ctx: Any,
        *,
        party_slots: Mapping[str, int] | None = None,
        macro_path: str | Path | None = None,
        default_macro_path: str | Path = DEFAULT_MACRO_PATH,
        mode: str = HOLD_ON_MODE,
        priority: int = 1,
        enabled: bool = True,
        log: Callable[[str], None] = print,
    ):
        if mode not in MODES:
            raise ValueError(f"不支持的一键宏模式：{mode}")
        self.ctx = ctx
        self.party_slots = {
            str(name): int(slot) for name, slot in (party_slots or {}).items()
            if 1 <= int(slot) <= 4
        }
        self.macro_path = Path(macro_path).expanduser() if macro_path else USER_MACRO_PATH
        self.default_macro_path = Path(default_macro_path).expanduser()
        self.mode = mode
        self.priority = int(priority)
        self.enabled = bool(enabled)
        self.log = log
        self.executor = CombatExecutor.for_context(ctx, party_slots=self.party_slots, log=log)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._force_stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._hotkey_down = False
        self._macros: dict[str, list[CombatCommand]] = {}
        self._signature: tuple[int, int] | None = None
        self._pressed_keys: set[str] = set()
        self._pressed_buttons: set[str] = set()

    @property
    def running(self) -> bool:
        with self._lock:
            return bool(self._worker and self._worker.is_alive())

    def _ensure_macro_file(self) -> None:
        if self.macro_path.exists():
            return
        if not self.default_macro_path.is_file():
            raise FileNotFoundError(f"未找到一键宏默认配置：{self.default_macro_path}")
        self.macro_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.default_macro_path, self.macro_path)

    def reload_if_needed(self, *, force: bool = False) -> bool:
        self._ensure_macro_file()
        stat = self.macro_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if not force and signature == self._signature:
                return False
        macros = load_avatar_macros(self.macro_path, self.priority)
        with self._lock:
            self._macros = macros
            self._signature = signature
        self.log(f"[OneKeyFight] 加载宏配置完成（{len(macros)} 个角色）")
        return True

    def _active_character(self) -> str | None:
        try:
            active_slot = int(getattr(self.ctx.input, "_active_slot"))
        except (AttributeError, TypeError, ValueError):
            return None
        return next((
            name for name, slot in self.party_slots.items() if slot == active_slot
        ), None)

    def key_down(self) -> bool:
        with self._lock:
            if self._hotkey_down or not self.enabled:
                return False
            self._hotkey_down = True
        try:
            self.reload_if_needed()
        except Exception:
            with self._lock:
                self._hotkey_down = False
            raise
        with self._lock:
            if self._worker and self._worker.is_alive():
                if self.mode == TICK_MODE:
                    self._stop.set()
                    return True
                return False
            self._stop.clear()
            self._force_stop.clear()
            self._worker = threading.Thread(
                target=self._fight_loop, daemon=True, name="one-key-fight",
            )
            self._worker.start()
        return True

    def key_up(self) -> bool:
        with self._lock:
            was_down = self._hotkey_down
            self._hotkey_down = False
            mode = self.mode
        if self.enabled and mode in {HOLD_ON_MODE, HOLD_FINISH_MODE}:
            self._stop.set()
            if mode == HOLD_ON_MODE:
                self.release_pressed_macro_inputs()
        return was_down

    def stop(self, *, wait: bool = True, timeout: float = 2.0) -> None:
        with self._lock:
            self._hotkey_down = False
            worker = self._worker
        self._stop.set()
        self._force_stop.set()
        self.release_pressed_macro_inputs()
        if wait and worker and worker is not threading.current_thread():
            worker.join(timeout=max(0.0, timeout))

    def wait(self, timeout: float | None = None) -> bool:
        with self._lock:
            worker = self._worker
        if worker:
            worker.join(timeout)
        return not self.running

    def _fight_loop(self) -> None:
        character = self._active_character()
        with self._lock:
            commands = list(self._macros.get(character or "", ()))
            mode = self.mode
        try:
            if not character:
                self.log("[OneKeyFight] 无法确定当前角色，请配置 config/party.json")
                return
            if not commands:
                self.log(f"[OneKeyFight] {character} 的宏配置为空（优先级 {self.priority}）")
                return
            round_number = 1
            self.log(f"[OneKeyFight] → {character} 开始宏")
            while self.enabled and not self._stop.is_set():
                with self._lock:
                    hotkey_down = self._hotkey_down
                if mode != TICK_MODE and not hotkey_down:
                    break
                self.log(f"[OneKeyFight] → {character} 第 {round_number} 轮")
                for command in commands:
                    if self._force_stop.is_set():
                        break
                    if command.activating_rounds and round_number not in command.activating_rounds:
                        continue
                    if mode == HOLD_ON_MODE:
                        with self._lock:
                            hotkey_down = self._hotkey_down
                        if self._stop.is_set() or not hotkey_down:
                            break
                    self.executor.exec(command)
                    self._track_command(command)
                round_number += 1
        finally:
            if mode == HOLD_ON_MODE:
                self.release_pressed_macro_inputs()
            with self._lock:
                if self._worker is threading.current_thread():
                    self._worker = None
            if character:
                self.log(f"[OneKeyFight] → {character} 停止宏")

    def _track_command(self, command: CombatCommand) -> None:
        action = command.action
        value = command.params[0] if command.params else "left"
        with self._lock:
            if action == "keydown":
                self._pressed_keys.add(value)
            elif action == "keyup":
                self._pressed_keys.discard(value)
            elif action == "mousedown":
                self._pressed_buttons.add(value.casefold())
            elif action == "mouseup":
                self._pressed_buttons.discard(value.casefold())

    def release_pressed_macro_inputs(self) -> None:
        with self._lock:
            keys = tuple(self._pressed_keys)
            buttons = tuple(self._pressed_buttons)
            self._pressed_keys.clear()
            self._pressed_buttons.clear()
        for key in keys:
            self.ctx.input.key_up(key)
        for button in buttons:
            if button == "left":
                self.ctx.input.attack_up()
            elif button == "right":
                self.ctx.input.button_up("sprint")

    # BetterGI host naming aliases.
    KeyDown = key_down
    KeyUp = key_up
    Stop = stop


__all__ = [
    "AvatarMacro", "HOLD_FINISH_MODE", "HOLD_ON_MODE", "OneKeyFightTask",
    "TICK_MODE", "load_avatar_macros", "parse_one_key_script",
]
