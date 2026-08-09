"""Reusable encounter loop for BetterGI boss and ley-line tasks.

The Windows tasks differ mainly in how they locate the encounter. Once the
player is at the encounter point, both use the same portable sequence:
pathing, combat, interaction, and optional resin selection. Keeping that
sequence here lets JS ``SoloTask`` names share cancellation and cleanup.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from ..engine.context import GameContext
from ..engine.recognition import RecognitionObject
from ..pathing.executor import PathingExecutor
from ..pathing.model import PathingTask
from .auto_fight import AutoFightTask


class AutoEncounterTask:
    def __init__(
        self,
        ctx: GameContext,
        *,
        name: str,
        route_path: str | Path | None = None,
        rounds: int = 1,
        combat_strategy_path: str | None = None,
        timeout_s: float = 240,
        party_slots: dict[str, int] | None = None,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.name = name
        self.route_path = Path(route_path).expanduser() if route_path else None
        self.rounds = max(1, int(rounds))
        self.log = log
        self.fight = AutoFightTask(
            ctx,
            combat_strategy_path=combat_strategy_path,
            timeout_s=max(30, float(timeout_s)),
            party_slots=party_slots,
            log=log,
        )

    def _route(self) -> PathingTask:
        if self.route_path is None:
            raise FileNotFoundError(
                f"{self.name} 未配置 routePath/pathingFile；请提供 BetterGI 地图追踪路线"
            )
        if not self.route_path.is_file():
            raise FileNotFoundError(f"{self.name} 路线不存在：{self.route_path}")
        return PathingTask.load(self.route_path)

    def _claim_reward(self) -> None:
        self.ctx.input.key_press("F")
        self.ctx.sleep(1600)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            hits = self.ctx.capture_region().find_multi(
                RecognitionObject.ocr(900, 350, 900, 650), limit=30
            )
            for hit in hits:
                text = hit.text.replace(" ", "")
                if any(term in text for term in (
                    "使用浓缩树脂", "使用原粹树脂", "浓缩树脂", "原粹树脂",
                    "领取奖励", "收取奖励", "Claim Reward",
                )):
                    hit.click()
                    self.ctx.sleep(1400)
                    return
            self.ctx.sleep(350)

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        for round_no in range(1, self.rounds + 1):
            if cancelled and cancelled():
                return False
            self.log(f"[{self.name}] 第 {round_no}/{self.rounds} 轮")
            route = self._route()
            if not PathingExecutor(
                self.ctx,
                party_slots=getattr(self.ctx, "party_slots", None),
                log=self.log,
            ).run(route):
                return False
            if cancelled and cancelled():
                return False
            if not self.fight.run(cancelled=cancelled):
                return False
            self._claim_reward()
        self.log(f"[{self.name}] 完成")
        return True


class AutoBossTask(AutoEncounterTask):
    def __init__(self, ctx: GameContext, *, boss_name: str = "",
                 route_path: str | Path | None = None, **kwargs):
        super().__init__(ctx, name="AutoBoss", route_path=route_path, **kwargs)
        self.boss_name = boss_name


class AutoLeyLineTask(AutoEncounterTask):
    def __init__(self, ctx: GameContext, *, route_path: str | Path | None = None,
                 **kwargs):
        super().__init__(ctx, name="AutoLeyLine", route_path=route_path, **kwargs)
