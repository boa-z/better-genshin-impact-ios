"""BetterGI AutoGeniusInvokation strategy parser and touch executor.

The parser intentionally follows the original Chinese strategy format.  The
executor uses OCR when card/button text is available and configurable fallback
points for mobile builds that render the controls without text.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Mapping

from ..engine.context import GameContext
from ..engine.recognition import RecognitionObject


@dataclass(frozen=True)
class TcgCharacter:
    index: int
    name: str


@dataclass(frozen=True)
class TcgCommand:
    character: str
    skill: int
    dice_delta: int = 0


def parse_tcg_strategy(text: str) -> tuple[dict[int, TcgCharacter], list[TcgCommand]]:
    stage = ""
    characters: dict[int, TcgCharacter] = {}
    commands: list[TcgCommand] = []
    for line_no, raw in enumerate(str(text).splitlines(), start=1):
        line = raw.strip()
        if not line or line == "---" or line.startswith("//"):
            continue
        if line.endswith(":") or line in ("角色定义:", "策略定义:"):
            stage = line
            continue
        if stage == "角色定义:":
            match = re.match(r"角色\s*(\d+)\s*=\s*([^|{]+)", line)
            if not match:
                raise ValueError(f"七圣召唤策略第 {line_no} 行角色定义无效")
            index = int(match.group(1))
            if index not in (1, 2, 3):
                raise ValueError(f"七圣召唤角色序号必须为 1-3：第 {line_no} 行")
            characters[index] = TcgCharacter(index, match.group(2).strip())
            continue
        if stage == "策略定义:":
            parts = line.split()
            if len(parts) < 3 or parts[1] != "使用":
                raise ValueError(f"七圣召唤策略第 {line_no} 行行动格式无效")
            skill = re.search(r"(\d+)", parts[2])
            if skill is None or not 1 <= int(skill.group(1)) <= 5:
                raise ValueError(f"七圣召唤技能编号无效：第 {line_no} 行")
            delta = 0
            if len(parts) >= 4:
                delta_match = re.search(r"(\d+)", parts[3])
                if delta_match and parts[3].startswith("骰子增加"):
                    delta = int(delta_match.group(1))
                elif delta_match and parts[3].startswith("骰子减少"):
                    delta = -int(delta_match.group(1))
                else:
                    raise ValueError(f"七圣召唤骰子参数无效：第 {line_no} 行")
            commands.append(TcgCommand(parts[0], int(skill.group(1)), delta))
            continue
        raise ValueError(f"七圣召唤策略第 {line_no} 行位于未知段落：{stage}")
    if set(characters) != {1, 2, 3}:
        raise ValueError("七圣召唤策略必须定义角色 1、2、3")
    known = {character.name for character in characters.values()}
    if any(command.character not in known for command in commands):
        raise ValueError("七圣召唤策略引用了未定义角色")
    if not commands:
        raise ValueError("七圣召唤策略没有行动命令")
    return characters, commands


class AutoGeniusInvokationTask:
    def __init__(
        self,
        ctx: GameContext,
        strategy: str,
        *,
        character_points: Mapping[int, tuple[float, float]] | None = None,
        skill_points: Mapping[int, tuple[float, float]] | None = None,
        max_commands: int | None = None,
        timeout_s: float = 900.0,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.characters, self.commands = parse_tcg_strategy(strategy)
        self.character_points = character_points or {
            1: (560, 890), 2: (960, 890), 3: (1360, 890)
        }
        self.skill_points = skill_points or {
            1: (1450, 910), 2: (1570, 910), 3: (1690, 910),
            4: (1570, 790), 5: (1690, 790),
        }
        self.max_commands = max_commands or len(self.commands)
        self.timeout_s = max(1.0, float(timeout_s))
        self.log = log

    def _tap_ocr(self, text: str, roi=(0, 0, 1920, 1080), timeout_s: float = 1.2) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for hit in self.ctx.capture_region().find_multi(
                RecognitionObject.ocr(*roi), limit=50
            ):
                if text.replace(" ", "") in hit.text.replace(" ", ""):
                    hit.click()
                    return True
            self.ctx.sleep(180)
        return False

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        deadline = time.monotonic() + self.timeout_s
        completed = 0
        self.log(f"[AutoGeniusInvokation] 执行 {len(self.commands)} 条策略")
        for command in self.commands[: self.max_commands]:
            if time.monotonic() >= deadline or (cancelled and cancelled()):
                return False
            character = next(
                value for value in self.characters.values()
                if value.name == command.character
            )
            selected = self._tap_ocr(command.character, roi=(0, 560, 1100, 430))
            if not selected:
                point = self.character_points[character.index]
                self.ctx.input.click_ref(*point)
            skill_text = f"技能{command.skill}"
            used = self._tap_ocr(skill_text, roi=(1100, 600, 820, 420))
            if not used:
                point = self.skill_points.get(command.skill)
                if point is None:
                    raise ValueError(f"未配置七圣召唤技能 {command.skill} 的触控点")
                self.ctx.input.click_ref(*point)
            self._tap_ocr("确认", roi=(1100, 600, 820, 420), timeout_s=0.8)
            completed += 1
            self.log(
                f"[AutoGeniusInvokation] {command.character} 使用技能{command.skill}"
            )
            self.ctx.sleep(500)
        self._tap_ocr("结束回合", roi=(1100, 600, 820, 420), timeout_s=0.8)
        return completed == min(self.max_commands, len(self.commands))
