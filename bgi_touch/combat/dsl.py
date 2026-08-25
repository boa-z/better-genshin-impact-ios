"""BetterGI 战斗策略 DSL 解析与执行。

用于 repo/combat/*.txt、pathing 路点 action=="combat_script" 的 action_params、
以及 dispatcher.runCombatScript()。文法（对应原版 CombatScriptParser.cs）：

    // 注释
    角色名 动作1, 动作2(参数), ...
    动作1, 动作2            // 无角色名 = 用当前角色执行

动作：e/skill[(hold)], q/burst, attack(秒), charge(秒), dash(秒), jump,
w/a/s/d(秒), walk(方向,秒), wait(秒), aim, keydown/keyup/keypress(键),
click(left|middle|right), moveby(x,y), ready, check。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

ACTION_ALIASES = {
    # BetterGI's Method aliases. Keep the canonical values short because
    # community scripts and JSON strategy names use them in conditions.
    "e": "e", "skill": "skill",
    "q": "q", "burst": "burst",
    "attack": "attack", "普攻": "attack", "普通攻击": "attack",
    "charge": "charge", "重击": "charge",
    "dash": "dash", "冲刺": "dash",
    "jump": "jump", "j": "jump", "跳跃": "jump",
    "w": "w", "a": "a", "s": "s", "d": "d",
    "walk": "walk", "行走": "walk",
    "wait": "wait", "after": "wait", "等待": "wait",
    "ready": "ready", "完成": "ready",
    "check": "check", "检测": "check",
    "aim": "aim", "r": "aim", "瞄准": "aim",
    "fly": "fly",
    "keydown": "keydown", "keyup": "keyup", "keypress": "keypress",
    "click": "click", "mousedown": "mousedown", "mouseup": "mouseup",
    "moveby": "moveby",
    "scroll": "scroll", "verticalscroll": "scroll",
    "round": "round",
}
# Keep aliases public as well as canonical names; callers use this set when
# validating converted macro files before execution.
KNOWN_ACTIONS = frozenset(ACTION_ALIASES)

_CMD_RE = re.compile(r"^([\w一-鿿]+)\s*(?:\(([^)]*)\))?$")


def canonical_action(value: object) -> str:
    """Normalize BetterGI Method aliases to the mobile executor spelling."""

    text = str(value or "").strip()
    return ACTION_ALIASES.get(text.casefold(), text.casefold())


@dataclass
class CombatCommand:
    action: str
    params: list[str] = field(default_factory=list)
    activating_rounds: tuple[int, ...] = field(default_factory=tuple)


@dataclass
class CombatLine:
    character: Optional[str]
    commands: list[CombatCommand]


def _split_commands(body: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch in ",，、;" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def parse_combat_script(text: str) -> list[CombatLine]:
    lines: list[CombatLine] = []
    for raw in text.splitlines():
        line = re.sub(r"(//|#).*$", "", raw).strip()
        if not line:
            continue
        character: Optional[str] = None
        body = line
        if " " in line:
            head, rest = line.split(" ", 1)
            head_action = canonical_action(
                re.sub(r"\(.*", "", head).rstrip(",，、;")
            )
            if head_action not in KNOWN_ACTIONS:
                character, body = head.strip(), rest
        commands = []
        for part in _split_commands(body):
            m = _CMD_RE.match(part)
            if m:
                params = [p.strip() for p in re.split(r"[,，]", m.group(2))] if m.group(2) else []
                commands.append(CombatCommand(canonical_action(m.group(1)), params))
        if commands or character:
            lines.append(CombatLine(character, commands))
    return lines


class CombatExecutor:
    """在 InputSimulator 上执行战斗 DSL。

    party_slots：角色名 → 队伍槽位(1-4)。原版靠 OCR 识别队伍；此处优先读
    config/party.json，未配置的角色跳过切人并告警。
    """

    def __init__(self, input_sim, sleep: Callable[[float], None] | None = None,
                 party_slots: dict[str, int] | None = None,
                 log: Callable[[str], None] = print,
                 check_combat_end: Callable[[], bool] | None = None,
                 team_switcher=None, skill_ready: Callable[[], bool] | None = None,
                 hud_ready: Callable[[], bool] | None = None):
        self.input = input_sim
        self.sleep = sleep or (lambda ms: time.sleep(ms / 1000))
        self.party_slots = party_slots or {}
        self.log = log
        self.check_combat_end = check_combat_end
        self.team_switcher = team_switcher  # combat.hud.TeamSwitcher：按名 OCR 切人
        self.skill_ready = skill_ready      # 旧宿主的 ready 回调兼容
        self.hud_ready = hud_ready          # ready 指令的战斗 HUD 就绪检测

    @classmethod
    def for_context(cls, ctx, party_slots=None, log=print, sleep=None):
        """带识别增强的构造：按名切人 + HUD/技能识别 + 战斗结束检测。"""
        from .hud import (
            TeamSwitcher,
            enemies_nearby,
            is_party_hud_ready,
            is_skill_ready,
        )
        return cls(ctx.input, sleep=sleep or ctx.sleep, party_slots=party_slots, log=log,
                   team_switcher=TeamSwitcher(ctx, log),
                   skill_ready=lambda: is_skill_ready(ctx),
                   hud_ready=lambda: is_party_hud_ready(ctx),
                   check_combat_end=lambda: not enemies_nearby(ctx))

    def run(self, script: str | list[CombatLine], loop_until_end: bool = False) -> None:
        lines = parse_combat_script(script) if isinstance(script, str) else script
        while True:
            for line in lines:
                if line.character:
                    self.switch_to(line.character)
                for cmd in line.commands:
                    self.exec(cmd)
                    # BetterGI executes ``check`` first and probes the battle
                    # state afterwards.  This matters for scripts that finish
                    # an attack/skill and then immediately request a probe.
                    if (
                        canonical_action(cmd.action) == "check"
                        and self.check_combat_end
                        and self.check_combat_end()
                    ):
                        self.input.release_all()
                        return
            if not loop_until_end:
                break
            if self.check_combat_end and self.check_combat_end():
                break
        self.input.release_all()

    def switch_to(self, character: str) -> None:
        slot = self.party_slots.get(character)
        if slot:
            self.input.key_press(str(slot))
            self.sleep(600)
            return
        if self.team_switcher is not None and self.team_switcher.switch_by_name(character):
            return
        self.log(f"[combat] 无法切换到“{character}”（OCR 未命中且未配置槽位），跳过")

    @staticmethod
    def _sec(params: list[str], idx: int = 0, default: float = 0.2) -> float:
        try:
            return float(params[idx])
        except (IndexError, ValueError):
            return default

    def exec(self, cmd: CombatCommand) -> None:
        a, p = canonical_action(cmd.action), cmd.params
        if a in ("e", "skill"):
            options = {str(value).strip().casefold() for value in p}
            if "fast" in options and self.skill_ready is not None:
                if not self.skill_ready():
                    return
            if "wait" in options and self.skill_ready is not None:
                deadline = time.monotonic() + 8.0
                while time.monotonic() < deadline and not self.skill_ready():
                    self.sleep(200)
            hold = "hold" in options
            self.input.key_press("E", hold_ms=900 if hold else 80)
        elif a in ("q", "burst"):
            self.input.key_press("Q")
            self.sleep(2000)  # 大招运镜
        elif a == "attack":
            until = time.monotonic() + self._sec(p, 0, 0.2)
            while time.monotonic() < until:
                self.input.attack()
                self.sleep(180)
        elif a == "charge":
            self.input.charged_attack(int(self._sec(p, 0, 1.0) * 1000))
        elif a == "dash":
            self.input.key_press("LSHIFT", hold_ms=int(min(self._sec(p, 0, 0.2) * 1000, 900)))
        elif a == "jump":
            self.input.key_press("SPACE")
        elif a in ("w", "a", "s", "d"):
            self._walk(a, self._sec(p, 0, 0.2) * 1000)
        elif a == "walk":
            self._walk((p[0] if p else "w").lower(), self._sec(p, 1, 0.2) * 1000)
        elif a == "wait":
            self.sleep(self._sec(p, 0, 1.0) * 1000)
        elif a == "keydown":
            self._key_down(p[0] if p else "")
        elif a == "keyup":
            self._key_up(p[0] if p else "")
        elif a == "keypress":
            self._key_press(p[0] if p else "")
        elif a == "click":
            button = p[0].casefold() if p else "left"
            if button == "left":
                self.input.attack()
            elif button == "right":
                self.input.key_press("LSHIFT")
            elif button == "middle":
                self.input.tap_button("elementalSight")
        elif a == "mousedown":
            button = p[0].casefold() if p else "left"
            if button == "left":
                self.input.attack_down()
            elif button == "right":
                self.input.button_down("sprint")
            elif button == "middle":
                self.input.button_down("elementalSight")
        elif a == "mouseup":
            button = p[0].casefold() if p else "left"
            if button == "left":
                self.input.attack_up()
            elif button == "right":
                self.input.button_up("sprint")
            elif button == "middle":
                self.input.button_up("elementalSight")
        elif a == "moveby":
            self.input.move_camera_by(self._sec(p, 0, 0), self._sec(p, 1, 0))
        elif a == "aim":
            self.input.key_press("R")
        elif a == "ready":
            ready_check = self.hud_ready or self.skill_ready
            if ready_check is None:
                self.sleep(500)
            else:
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline and not ready_check():
                    self.sleep(300)
        elif a == "scroll":
            self.input.vertical_scroll(self._sec(p, 0, 0))
        elif a in ("fly", "check", "round"):
            pass
        else:
            self.log(f"[combat] 未知动作 {a}，已跳过")

    def _walk(self, direction: str, ms: float) -> None:
        key = direction.upper()
        self.input.key_down(key)
        self.sleep(ms)
        self.input.key_up(key)

    @staticmethod
    def _mouse_button(value: object) -> str | None:
        normalized = str(value or "").strip().casefold().replace("_", "")
        return {
            "vklbutton": "left", "lbutton": "left", "leftbutton": "left",
            "mouseleft": "left", "leftmouse": "left",
            "vkrbutton": "right", "rbutton": "right", "rightbutton": "right",
            "mouseright": "right", "rightmouse": "right",
            "vkmbutton": "middle", "mbutton": "middle", "middlebutton": "middle",
            "mousemiddle": "middle", "middlemouse": "middle",
        }.get(normalized)

    def _key_down(self, value: object) -> None:
        button = self._mouse_button(value)
        if button == "left":
            self.input.attack_down()
        elif button == "right":
            self.input.button_down("sprint")
        elif button == "middle":
            self.input.button_down("elementalSight")
        else:
            self.input.key_down(str(value or ""))

    def _key_up(self, value: object) -> None:
        button = self._mouse_button(value)
        if button == "left":
            self.input.attack_up()
        elif button == "right":
            self.input.button_up("sprint")
        elif button == "middle":
            self.input.button_up("elementalSight")
        else:
            self.input.key_up(str(value or ""))

    def _key_press(self, value: object) -> None:
        button = self._mouse_button(value)
        if button == "left":
            self.input.attack()
        elif button == "right":
            self.input.key_press("LSHIFT")
        elif button == "middle":
            self.input.tap_button("elementalSight")
        else:
            self.input.key_press(str(value or ""))
