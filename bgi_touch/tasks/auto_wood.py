"""AutoWood SoloTask：自动伐木（对应原版 GameTask/AutoWood）。

一轮 = 普攻砍树 + 使用小道具「王树瑞佑」收集木材；每轮之间可重登重置
木材上限（原版逻辑）。wood_round_num=0 表示只砍不重登。
"""

from __future__ import annotations

import time
from typing import Callable

from ..engine.context import GameContext
from ..engine.genshin_api import GenshinApi


class AutoWoodTask:
    def __init__(self, ctx: GameContext, rounds: int = 1, per_round_attacks: int = 8,
                 relogin_between: bool = False, log: Callable[[str], None] = print):
        self.ctx = ctx
        self.rounds = max(1, int(rounds))
        self.per_round_attacks = per_round_attacks
        self.relogin_between = relogin_between
        self.log = log
        self.genshin = GenshinApi(ctx, log)

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        for rd in range(1, self.rounds + 1):
            self.log(f"[AutoWood] 第 {rd}/{self.rounds} 轮")
            for _ in range(self.per_round_attacks):
                if cancelled and cancelled():
                    return False
                self.ctx.input.attack()
                self.ctx.sleep(600)
            self.ctx.input.key_press("Z")  # 王树瑞佑
            self.ctx.sleep(3000)
            if self.relogin_between and rd < self.rounds:
                self.log("[AutoWood] 重登重置…")
                self.genshin.relogin()
        return True
