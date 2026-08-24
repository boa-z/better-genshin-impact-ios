"""AutoFight SoloTask：循环执行战斗策略直到战斗结束/超时。

对应原版 GameTask/AutoFight。战斗结束复用 BetterGI 的敌血条快速跳过、开战阻断、
快速检查和队伍界面可打开确认，移动端通过 DeviceHub profile 的 L/X 映射执行。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from ..combat.dsl import CombatExecutor, parse_combat_script
from ..combat.finish import FightFinishConfig, FightFinishDetector
from ..combat.hud import enemies_nearby
from ..engine.context import GameContext


class AutoFightTask:
    def __init__(self, ctx: GameContext, combat_strategy_path: str | None = None,
                 timeout_s: float = 120, party_slots: dict[str, int] | None = None,
                 log: Callable[[str], None] = print,
                 fight_finish_detect_enabled: bool = True,
                 finish_detect_config: FightFinishConfig | None = None):
        self.ctx = ctx
        self.log = log
        self.timeout_s = timeout_s
        self.strategy = self._load_strategy(combat_strategy_path)
        self.lines = parse_combat_script(self.strategy)
        self.executor = CombatExecutor.for_context(ctx, party_slots=party_slots, log=log)
        self.finish_detect_enabled = bool(fight_finish_detect_enabled)
        self.finish_detector = FightFinishDetector(
            ctx, finish_detect_config, log=log,
        )

    def _load_strategy(self, path: str | None) -> str:
        if path and Path(path).exists():
            return Path(path).read_text(encoding="utf-8")
        self.log("[AutoFight] 未提供战斗策略，使用通用普攻循环")
        return "attack(1.5), dash, attack(1.5)"

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        deadline = time.monotonic() + self.timeout_s
        clear_streak = 0
        self.log(f"[AutoFight] 开始（超时 {self.timeout_s:.0f}s）")
        self.finish_detector.start_battle()
        current_character = None
        try:
            while time.monotonic() < deadline:
                if cancelled and cancelled():
                    self.log("[AutoFight] 已取消")
                    return False
                for line in self.lines:
                    previous_character = current_character
                    if (
                        self.finish_detect_enabled
                        and not self.finish_detector.config.check_after_switch_avatar
                        and self.finish_detector.should_fast_check(previous_character)
                        and self.finish_detector.check(
                            previous_character, cancelled=cancelled,
                        )
                    ):
                        self.log("[AutoFight] 快速检查确认战斗结束")
                        return True
                    if line.character:
                        self.executor.switch_to(line.character)
                        current_character = line.character
                    if (
                        self.finish_detect_enabled
                        and self.finish_detector.config.check_after_switch_avatar
                        and self.finish_detector.should_fast_check(previous_character)
                        and self.finish_detector.check(
                            previous_character, after_switch=True, cancelled=cancelled,
                        )
                    ):
                        self.log("[AutoFight] 切人后确认战斗结束")
                        return True
                    for command in line.commands:
                        if cancelled and cancelled():
                            self.log("[AutoFight] 已取消")
                            return False
                        if command.action == "check" and self.finish_detect_enabled:
                            if self.finish_detector.check(
                                current_character, cancelled=cancelled,
                            ):
                                self.log("[AutoFight] check 确认战斗结束")
                                return True
                        else:
                            self.executor.exec(command)

                if self.finish_detect_enabled:
                    if self.finish_detector.check(
                        current_character, cancelled=cancelled,
                    ):
                        self.log("[AutoFight] 战斗结束")
                        return True
                elif enemies_nearby(self.ctx):
                    clear_streak = 0
                else:
                    clear_streak += 1
                    if clear_streak >= 2:
                        self.log("[AutoFight] 战斗结束")
                        return True
            self.log("[AutoFight] 超时退出")
            return False
        finally:
            self.ctx.input.release_all()
