"""BetterGI 键鼠宏（macroEvents JSON）→ 触控时间线的转换与回放。

原始事件类型：0 KeyDown / 1 KeyUp / 2 鼠标绝对移动 / 3 鼠标相对移动（相机）
/ 4 MouseDown / 5 MouseUp / 6 滚轮。

转换规则：
- 键盘事件 → key_down/key_up（由输入层翻译为摇杆/按钮触控）
- 连续的相对移动按 ≤120ms 窗口合并为一次相机滑动
- 绝对移动 + 左键按下/抬起 → 在该坐标 tap（按 info 记录分辨率归一到 1080p）
- 无预先移动的左键点击 → 普攻按钮
- 滚轮 → 合并相邻滚轮增量并转换为菜单安全区竖向滑动
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

VK_TABLE = {32: "SPACE", 16: "SHIFT", 160: "LSHIFT", 161: "RSHIFT", 17: "CTRL",
            162: "LCTRL", 163: "RCTRL", 18: "ALT", 13: "ENTER", 27: "ESCAPE", 9: "TAB",
            112: "F1", 113: "F2", 114: "F3", 115: "F4", 116: "F5", 117: "F6",
            118: "F7", 119: "F8"}


def vk_to_key(code: int) -> str | None:
    if 65 <= code <= 90 or 48 <= code <= 57:
        return chr(code)
    return VK_TABLE.get(code)


@dataclass
class TouchEvent:
    t: float  # ms since start
    kind: str  # keyDown / keyUp / cameraBy / verticalScroll / tapRef / attack* / sprint* / sight*
    key: str | None = None
    x: float = 0
    y: float = 0
    amount: float = 0


def convert_keymouse(macro: dict) -> tuple[list[TouchEvent], list[str]]:
    """BetterGI 宏 JSON → 触控事件时间线。返回 (事件, 警告)。"""
    info = macro.get("info") or {}
    rec_w = float(info.get("width") or 1920)
    rec_h = float(info.get("height") or 1080)
    sx, sy = 1920 / rec_w, 1080 / rec_h

    events: list[TouchEvent] = []
    warnings: list[str] = []
    cam: list[float] | None = None  # [t0, dx, dy, t_last]
    cursor: tuple[float, float] | None = None
    cursor_fresh = False
    wheel: list[float] | None = None  # [t0, raw_delta, t_last]

    def flush_cam() -> None:
        nonlocal cam
        if cam and (abs(cam[1]) > 1 or abs(cam[2]) > 1):
            events.append(TouchEvent(t=cam[0], kind="cameraBy", x=cam[1] * sx, y=cam[2] * sy))
        cam = None

    def flush_wheel() -> None:
        nonlocal wheel
        if wheel and abs(wheel[1]) > 1e-3:
            # Windows wheel delta 120 equals one BetterGI scroll click.
            events.append(TouchEvent(
                t=wheel[0], kind="verticalScroll", amount=wheel[1] / 120.0,
            ))
        wheel = None

    for ev in macro.get("macroEvents", []):
        etype = ev.get("type")
        t = float(ev.get("time", 0))
        if etype == 3:  # 相对移动：合并窗口
            flush_wheel()
            if cam and t - cam[3] <= 120:
                cam[1] += ev.get("mouseX", 0)
                cam[2] += ev.get("mouseY", 0)
                cam[3] = t
            else:
                flush_cam()
                cam = [t, float(ev.get("mouseX", 0)), float(ev.get("mouseY", 0)), t]
            continue
        flush_cam()
        if etype == 6:
            delta = float(ev.get("mouseY", 0) or 0)
            if abs(delta) < 1e-3:
                flush_wheel()
            elif wheel and t - wheel[2] <= 120:
                wheel[1] += delta
                wheel[2] = t
            else:
                flush_wheel()
                wheel = [t, delta, t]
            continue
        flush_wheel()
        if etype in (0, 1):
            key = vk_to_key(int(ev.get("keyCode", 0)))
            if key is None:
                continue
            events.append(TouchEvent(t=t, kind="keyDown" if etype == 0 else "keyUp", key=key))
        elif etype == 2:
            cursor = (float(ev.get("mouseX", 0)) * sx, float(ev.get("mouseY", 0)) * sy)
            cursor_fresh = True
        elif etype in (4, 5):
            button = str(ev.get("mouseButton", "Left")).casefold()
            if button == "right":
                events.append(TouchEvent(
                    t=t, kind="sprintDown" if etype == 4 else "sprintUp",
                ))
            elif button == "middle":
                events.append(TouchEvent(
                    t=t, kind="sightDown" if etype == 4 else "sightUp",
                ))
            elif button == "left" and etype == 4:
                if cursor_fresh and cursor:
                    events.append(TouchEvent(t=t, kind="tapRef", x=cursor[0], y=cursor[1]))
                    cursor_fresh = False
                else:
                    events.append(TouchEvent(t=t, kind="attackDown"))
            elif button == "left":
                if events and events[-1].kind == "attackDown":
                    # down/up 配对 → 由回放端按时长决定普攻或蓄力
                    events.append(TouchEvent(t=t, kind="attackUp"))
    flush_cam()
    flush_wheel()
    return events, warnings


def load_keymouse(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class MacroPlayer:
    def __init__(self, input_sim, sleep: Callable[[float], None] | None = None,
                 log: Callable[[str], None] = print):
        self.input = input_sim
        self.sleep = sleep or (lambda ms: time.sleep(ms / 1000))
        self.log = log

    def play(self, macro: dict) -> None:
        events, warnings = convert_keymouse(macro)
        for w in warnings:
            self.log(f"[macro] {w}")
        start = time.monotonic()
        attack_down_at: float | None = None
        for ev in events:
            delay = ev.t / 1000 - (time.monotonic() - start)
            if delay > 0:
                self.sleep(delay * 1000)
            if ev.kind == "keyDown":
                self.input.key_down(ev.key)
            elif ev.kind == "keyUp":
                self.input.key_up(ev.key)
            elif ev.kind == "cameraBy":
                self.input.move_camera_by(ev.x, ev.y)
            elif ev.kind == "verticalScroll":
                self.input.vertical_scroll(ev.amount)
            elif ev.kind == "sprintDown":
                self.input.button_down("sprint")
            elif ev.kind == "sprintUp":
                self.input.button_up("sprint")
            elif ev.kind == "sightDown":
                self.input.button_down("elementalSight")
            elif ev.kind == "sightUp":
                self.input.button_up("elementalSight")
            elif ev.kind == "tapRef":
                self.input.click_ref(ev.x, ev.y)
            elif ev.kind == "attackDown":
                attack_down_at = ev.t
            elif ev.kind == "attackUp" and attack_down_at is not None:
                held = ev.t - attack_down_at
                if held >= 350:
                    self.input.charged_attack(int(held))
                else:
                    self.input.attack()
                attack_down_at = None
        self.input.release_all()
