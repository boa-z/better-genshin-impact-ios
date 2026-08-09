"""Touch-compatible AutoStygianOnslaught orchestration.

The event's Windows state machine is mostly navigation and reward handling;
the iOS port reuses the already migrated pathing/combat/reward loop.  A route
must be supplied because event map coordinates vary by account and version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..engine.context import GameContext
from .auto_encounter import AutoEncounterTask


class AutoStygianOnslaughtTask:
    def __init__(
        self,
        ctx: GameContext,
        *,
        route_path: str | Path | None = None,
        boss_num: int = 1,
        combat_strategy_path: str | None = None,
        rounds: int | None = None,
        timeout_s: float = 360.0,
        party_slots: dict[str, int] | None = None,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.route_path = route_path
        self.rounds = max(1, int(rounds if rounds is not None else boss_num))
        self.log = log
        self.encounter = AutoEncounterTask(
            ctx,
            name="AutoStygianOnslaught",
            route_path=route_path,
            rounds=self.rounds,
            combat_strategy_path=combat_strategy_path,
            timeout_s=timeout_s,
            party_slots=party_slots,
            log=log,
        )

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        if self.route_path is None:
            raise FileNotFoundError(
                "AutoStygianOnslaught 未配置 routePath/pathingFile；"
                "请提供活动入口路线"
            )
        self.log(f"[AutoStygianOnslaught] 开始 {self.rounds} 轮")
        result = self.encounter.run(cancelled=cancelled)
        self.log(f"[AutoStygianOnslaught] {'完成' if result else '未完成'}")
        return result
