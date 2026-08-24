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

KNOWN_ACTIONS = {
    "e", "skill", "q", "burst", "attack", "charge", "dash", "jump",
    "w", "a", "s", "d", "walk", "wait", "aim", "fly",
    "keydown", "keyup", "keypress", "click", "mousedown", "mouseup",
    "moveby", "scroll", "ready", "check",
}

_CMD_RE = re.compile(r"^([\w一-鿿]+)\s*(?:\(([^)]*)\))?$")


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
            head_action = re.sub(r"\(.*", "", head).rstrip(",，、;").lower()
            if head_action not in KNOWN_ACTIONS:
                character, body = head.strip(), rest
        commands = []
        for part in _split_commands(body):
            m = _CMD_RE.match(part)
            if m:
                params = [p.strip() for p in re.split(r"[,，]", m.group(2))] if m.group(2) else []
                commands.append(CombatCommand(m.group(1).lower(), params))
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
                 team_switcher=None, skill_ready: Callable[[], bool] | None = None):
        self.input = input_sim
        self.sleep = sleep or (lambda ms: time.sleep(ms / 1000))
        self.party_slots = party_slots or {}
        self.log = log
        self.check_combat_end = check_combat_end
        self.team_switcher = team_switcher  # combat.hud.TeamSwitcher：按名 OCR 切人
        self.skill_ready = skill_ready      # ready 指令的技能就绪检测

    @classmethod
    def for_context(cls, ctx, party_slots=None, log=print, sleep=None):
        """带识别增强的构造：按名切人 + 技能就绪 + 战斗结束检测。"""
        from .hud import TeamSwitcher, enemies_nearby, is_skill_ready
        return cls(ctx.input, sleep=sleep or ctx.sleep, party_slots=party_slots, log=log,
                   team_switcher=TeamSwitcher(ctx, log),
                   skill_ready=lambda: is_skill_ready(ctx),
                   check_combat_end=lambda: not enemies_nearby(ctx))

    def run(self, script: str | list[CombatLine], loop_until_end: bool = False) -> None:
        lines = parse_combat_script(script) if isinstance(script, str) else script
        while True:
            for line in lines:
                if line.character:
                    self.switch_to(line.character)
                for cmd in line.commands:
                    if cmd.action == "check" and self.check_combat_end and self.check_combat_end():
                        self.input.release_all()
                        return
                    self.exec(cmd)
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
        a, p = cmd.action, cmd.params
        if a in ("e", "skill"):
            hold = p and p[0].lower() == "hold"
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
            self.input.key_down(p[0] if p else "")
        elif a == "keyup":
            self.input.key_up(p[0] if p else "")
        elif a == "keypress":
            self.input.key_press(p[0] if p else "")
        elif a == "click":
            button = p[0].casefold() if p else "left"
            if button == "left":
                self.input.attack()
            elif button == "right":
                self.input.key_press("LSHIFT")
            # click(middle) 在 PC 是重置视角，触控端无对应操作
        elif a == "mousedown":
            button = p[0].casefold() if p else "left"
            if button == "left":
                self.input.attack_down()
            elif button == "right":
                self.input.button_down("sprint")
        elif a == "mouseup":
            button = p[0].casefold() if p else "left"
            if button == "left":
                self.input.attack_up()
            elif button == "right":
                self.input.button_up("sprint")
        elif a == "moveby":
            self.input.move_camera_by(self._sec(p, 0, 0), self._sec(p, 1, 0))
        elif a == "aim":
            self.input.key_press("R")
        elif a == "ready":
            if self.skill_ready is None:
                self.sleep(500)
            else:
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline and not self.skill_ready():
                    self.sleep(300)
        elif a == "scroll":
            self.input.vertical_scroll(self._sec(p, 0, 0))
        elif a in ("fly", "check"):
            pass
        else:
            self.log(f"[combat] 未知动作 {a}，已跳过")

    def _walk(self, direction: str, ms: float) -> None:
        key = direction.upper()
        self.input.key_down(key)
        self.sleep(ms)
        self.input.key_up(key)
