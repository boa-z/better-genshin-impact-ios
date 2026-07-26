"""AutoFight SoloTask：循环执行战斗策略直到战斗结束/超时。

对应原版 GameTask/AutoFight（原版另有 YOLO 掉落检测等，此处用敌血条启发式
作为结束判定，连续 N 次未见敌人即认为战斗结束）。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from ..combat.dsl import CombatExecutor
from ..combat.hud import enemies_nearby
from ..engine.context import GameContext


class AutoFightTask:
    def __init__(self, ctx: GameContext, combat_strategy_path: str | None = None,
                 timeout_s: float = 120, party_slots: dict[str, int] | None = None,
                 log: Callable[[str], None] = print):
        self.ctx = ctx
        self.log = log
        self.timeout_s = timeout_s
        self.strategy = self._load_strategy(combat_strategy_path)
        self.executor = CombatExecutor.for_context(ctx, party_slots=party_slots, log=log)

    def _load_strategy(self, path: str | None) -> str:
        if path and Path(path).exists():
            return Path(path).read_text(encoding="utf-8")
        self.log("[AutoFight] 未提供战斗策略，使用通用普攻循环")
        return "attack(1.5), dash, attack(1.5)"

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        deadline = time.monotonic() + self.timeout_s
        clear_streak = 0
        self.log(f"[AutoFight] 开始（超时 {self.timeout_s:.0f}s）")
        while time.monotonic() < deadline:
            if cancelled and cancelled():
                self.log("[AutoFight] 已取消")
                return False
            self.executor.run(self.strategy)
            if enemies_nearby(self.ctx):
                clear_streak = 0
            else:
                clear_streak += 1
                if clear_streak >= 2:  # 连续两轮无敌人 → 战斗结束
                    self.log("[AutoFight] 战斗结束")
                    return True
        self.log("[AutoFight] 超时退出")
        return False
