"""BetterGI-compatible SoloTask dispatcher shared by JS, WebUI and CLI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..combat.dsl import CombatExecutor
from ..engine.context import GameContext


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    try:
        value = getattr(obj, key)
    except (AttributeError, TypeError):
        return default
    return default if value is None else value


def _requested(token: Any) -> bool:
    if token is None:
        return False
    try:
        method = getattr(token, "isCancellationRequested", None)
        if callable(method) and method():
            return True
    except Exception:
        pass
    try:
        if bool(getattr(token, "cancelled", False)):
            return True
    except Exception:
        pass
    return False


class TaskDispatcher:
    """Run migrated BetterGI tasks with a single cancellation contract."""

    IMPLEMENTED = frozenset({
        "AutoFight", "AutoWood", "AutoDomain", "AutoCook", "AutoFishing", "AutoOpenChest",
    })

    def __init__(
        self,
        ctx: GameContext,
        party_slots: dict[str, int] | None = None,
        log: Callable[[str], None] = print,
        cancelled: Callable[[], bool] | None = None,
    ):
        self.ctx = ctx
        self.party_slots = party_slots or {}
        self.log = log
        self.cancelled = cancelled or (lambda: False)

    def _is_cancelled(self, token: Any = None) -> bool:
        return bool(self.cancelled()) or _requested(token)

    def _callback(self, token: Any = None) -> Callable[[], bool]:
        return lambda: self._is_cancelled(token)

    def run_task(self, task: Any, ct: Any = None) -> Any:
        name = str(_value(task, "name", task))
        cfg = _value(task, "config", {}) or {}
        if name == "AutoFight":
            return self.run_auto_fight_task(cfg, ct)
        if name == "AutoWood":
            from .auto_wood import AutoWoodTask
            return AutoWoodTask(
                self.ctx,
                rounds=int(_value(cfg, "woodRoundNum", _value(cfg, "rounds", 1)) or 1),
                per_round_attacks=int(_value(cfg, "perRoundAttacks", 8) or 8),
                relogin_between=bool(_value(cfg, "reloginBetween", False)),
                log=self.log,
            ).run(cancelled=self._callback(ct))
        if name == "AutoDomain":
            return self.run_auto_domain_task(cfg, ct)
        if name == "AutoCook":
            return self.run_auto_cook_task(cfg, ct)
        if name in ("AutoFishing", "AutoFish"):
            return self.run_auto_fishing_task(cfg, ct)
        if name in ("AutoOpenChest", "OpenChest"):
            return self.run_auto_open_chest_task(cfg, ct)
        raise NotImplementedError(
            f"SoloTask {name} 尚未移植；当前已支持 {', '.join(sorted(self.IMPLEMENTED))}"
        )

    def run_auto_fight_task(self, param: Any = None, ct: Any = None) -> bool:
        from .auto_fight import AutoFightTask
        strategy = _value(param, "combatStrategyPath", None)
        timeout = _value(param, "timeout", None)
        timeout_s = float(timeout) / 1000 if timeout else 120
        return AutoFightTask(
            self.ctx,
            combat_strategy_path=strategy,
            timeout_s=timeout_s,
            party_slots=self.party_slots,
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_auto_domain_task(self, param: Any = None, ct: Any = None) -> bool:
        from .auto_domain import AutoDomainTask
        return AutoDomainTask(
            self.ctx,
            rounds=int(_value(param, "domainRoundNum", _value(param, "rounds", 1)) or 1),
            combat_strategy_path=_value(param, "combatStrategyPath", None),
            use_condensed_resin=bool(_value(param, "useCondensedResin", True)),
            party_slots=self.party_slots,
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_auto_cook_task(self, param: Any = None, ct: Any = None) -> bool:
        from .auto_cook import AutoCookTask
        return AutoCookTask(
            self.ctx,
            check_interval_ms=int(_value(param, "checkIntervalMs", 400) or 400),
            stop_on_recover=bool(_value(param, "stopTaskWhenRecoverButtonDetected", True)),
            idle_timeout_s=float(_value(param, "idleTimeoutSeconds", 15) or 15),
            timeout_s=float(_value(param, "timeoutSeconds", 900) or 900),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_auto_fishing_task(self, param: Any = None, ct: Any = None) -> bool:
        from .auto_fishing import AutoFishingTask
        return AutoFishingTask(
            self.ctx,
            target_catches=int(_value(param, "targetCatches", _value(param, "fishCount", 1)) or 1),
            timeout_s=float(_value(param, "timeoutSeconds", 120) or 120),
            idle_timeout_s=float(_value(param, "idleTimeoutSeconds", 20) or 20),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_auto_open_chest_task(self, param: Any = None, ct: Any = None) -> bool:
        from .auto_open_chest import AutoOpenChestTask
        return AutoOpenChestTask(
            self.ctx,
            timeout_s=float(_value(param, "timeoutSeconds", 60) or 60),
            idle_timeout_s=float(_value(param, "idleTimeoutSeconds", 4) or 4),
            log=self.log,
        ).run(cancelled=self._callback(ct))

    def run_combat_script(self, script: str, avatar: str | None = None) -> Any:
        return CombatExecutor.for_context(
            self.ctx, party_slots=self.party_slots, log=self.log
        ).run(str(script))

    def add_timer(self, timer: Any) -> None:
        name = str(_value(timer, "name", timer))
        self.ctx.triggers.clear()
        if name == "AutoPick":
            config = _value(timer, "config", {}) or {}
            force_interaction = _value(
                config,
                "forceInteraction",
                _value(config, "force_interaction", False),
            )
            self.ctx.enable_trigger(name, force_interaction=bool(force_interaction))
        else:
            self.ctx.enable_trigger(name)

    def add_trigger(self, trigger: Any) -> None:
        self.ctx.enable_trigger(str(_value(trigger, "name", trigger)))

    def clear_all_triggers(self) -> None:
        self.ctx.triggers.clear()
