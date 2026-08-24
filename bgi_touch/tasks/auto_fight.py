"""AutoFight SoloTask：循环执行战斗策略直到战斗结束/超时。

对应原版 GameTask/AutoFight。战斗结束复用 BetterGI 的敌血条快速跳过、开战阻断、
快速检查和队伍界面可打开确认，移动端通过 DeviceHub profile 的 L/X 映射执行。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ..combat.dsl import CombatCommand, CombatExecutor, CombatLine, parse_combat_script
from ..combat.finish import FightFinishConfig, FightFinishDetector
from ..combat.hud import enemies_nearby, is_skill_ready
from ..combat.json_strategy import (
    ConditionEvaluator,
    JsonAction,
    JsonCombatStrategy,
    expand_priorities,
    load_json_strategy,
)
from ..engine.context import GameContext


ROOT = Path(__file__).resolve().parents[2]
AVATAR_DATA_PATH = ROOT / "assets" / "data" / "combat_avatar.json"


class AutoFightTask:
    def __init__(self, ctx: GameContext, combat_strategy_path: str | None = None,
                 timeout_s: float = 120, party_slots: dict[str, int] | None = None,
                 log: Callable[[str], None] = print,
                 fight_finish_detect_enabled: bool = True,
                 finish_detect_config: FightFinishConfig | None = None):
        self.ctx = ctx
        self.log = log
        self.timeout_s = timeout_s
        self.party_slots = dict(party_slots or {})
        self.strategy_path = Path(combat_strategy_path) if combat_strategy_path else None
        self.json_strategy: JsonCombatStrategy | None = None
        self.strategy = ""
        self.lines: list[CombatLine] = []
        if self.strategy_path and self.strategy_path.suffix.casefold() == ".json":
            self.json_strategy = load_json_strategy(self.strategy_path)
        else:
            self.strategy = self._load_strategy(combat_strategy_path)
            self.lines = parse_combat_script(self.strategy)
        self.executor = CombatExecutor.for_context(ctx, party_slots=party_slots, log=log)
        self.finish_detect_enabled = bool(fight_finish_detect_enabled)
        self.finish_detector = FightFinishDetector(
            ctx, finish_detect_config, log=log,
        )
        self._skill_deadlines: dict[str, float] = {}
        self._skill_cooldowns = self._load_skill_cooldowns()

    def _load_strategy(self, path: str | None) -> str:
        if path and Path(path).exists():
            return Path(path).read_text(encoding="utf-8")
        self.log("[AutoFight] 未提供战斗策略，使用通用普攻循环")
        return "attack(1.5), dash, attack(1.5)"

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        if self.json_strategy is not None:
            return self._run_json(cancelled)
        return self._run_txt(cancelled)

    def _run_txt(self, cancelled: Callable[[], bool] | None = None) -> bool:
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

    @staticmethod
    def _load_skill_cooldowns() -> dict[str, tuple[float, float]]:
        """Load tap/hold E cooldowns for deterministic e-cd() tracking."""
        try:
            records = json.loads(AVATAR_DATA_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        result: dict[str, tuple[float, float]] = {}
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, dict):
                continue
            try:
                tap = max(0.0, float(record.get("skillCD", 0) or 0))
                hold = max(0.0, float(record.get("skillHoldCD", tap) or tap))
            except (TypeError, ValueError):
                continue
            names = [record.get("name"), *(record.get("alias") or [])]
            for name in names:
                if name:
                    result[str(name).casefold()] = (tap, hold)
        return result

    def _active_character(self) -> str | None:
        try:
            active_slot = int(getattr(self.ctx.input, "_active_slot"))
        except (AttributeError, TypeError, ValueError):
            return None
        return next((
            name for name, slot in self.party_slots.items()
            if int(slot) == active_slot
        ), None)

    def _skill_cd_from_trigger(self, character: str | None) -> float | None:
        """Reuse SkillCd's shared-frame state when that trigger is enabled."""
        loop = getattr(self.ctx, "_trigger_loop", None)
        trigger = loop.get("SkillCd") if loop is not None and hasattr(loop, "get") else None
        state = getattr(trigger, "state", None)
        if state is None or not hasattr(state, "snapshot"):
            return None
        try:
            snapshot = state.snapshot()
        except Exception:
            return None
        for avatar in snapshot.get("team", []):
            if character and str(avatar.get("name", "")).casefold() == character.casefold():
                try:
                    return max(0.0, float(avatar.get("remaining", 0) or 0))
                except (TypeError, ValueError):
                    return None
        return None

    def _e_cd(self, character: str | None, _frame: Any = None) -> float:
        shared = self._skill_cd_from_trigger(character)
        if shared is not None:
            return shared
        if not character:
            return 0.0
        return max(0.0, self._skill_deadlines.get(character.casefold(), 0.0) - time.monotonic())

    def _record_skill(self, character: str | None, command: CombatCommand) -> None:
        if not character or command.action not in {"e", "skill"}:
            return
        cooldowns = self._skill_cooldowns.get(character.casefold())
        if cooldowns is None:
            return
        hold = bool(command.params and command.params[0].casefold() == "hold")
        cooldown = cooldowns[1 if hold else 0]
        self._skill_deadlines[character.casefold()] = time.monotonic() + cooldown

    def _slot_for(self, character: str | None) -> int | None:
        if not character:
            return None
        try:
            slot = int(self.party_slots.get(character, 0))
        except (TypeError, ValueError):
            return None
        return slot if 1 <= slot <= 4 else None

    @staticmethod
    def _bright_button(frame: np.ndarray, nx: float, ny: float) -> bool:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return False
        height, width = frame.shape[:2]
        radius = max(10, round(height * 0.035))
        x, y = round(nx * width), round(ny * height)
        crop = frame[max(0, y - radius):min(height, y + radius),
                     max(0, x - radius):min(width, x + radius)]
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        bright_color = (hsv[..., 2] > 155) & (hsv[..., 1] > 45)
        return float(bright_color.mean()) > 0.16

    def _q_ready(self, character: str | None, frame: Any) -> bool:
        """Detect the active burst button or an inactive party-row burst badge."""
        if not isinstance(frame, np.ndarray):
            return False
        target_slot = self._slot_for(character)
        try:
            active_slot = int(getattr(self.ctx.input, "_active_slot"))
        except (AttributeError, TypeError, ValueError):
            active_slot = None
        if target_slot is None or target_slot == active_slot:
            nx, ny = self.ctx.layout.buttons.get("burst", (0.649, 0.926))
            return self._bright_button(frame, nx, ny)
        others = [slot for slot in (1, 2, 3, 4) if slot != active_slot]
        if target_slot not in others:
            return False
        row = others.index(target_slot) + 1
        _, ny = self.ctx.layout.buttons.get(f"partyRow{row}", (0.96, 0.2))
        # Charged bursts are rendered as a bright elemental badge immediately
        # left of the inactive avatar portrait on mobile.
        return self._bright_button(frame, 0.905, ny)

    @staticmethod
    def _low_hp(frame: Any) -> bool:
        """Current HP turns red at the low-health threshold on the mobile HUD."""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return False
        height, width = frame.shape[:2]
        band = frame[int(height * 0.90):int(height * 0.98),
                     int(width * 0.30):int(width * 0.62)]
        if band.size == 0:
            return False
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        red = ((hsv[..., 0] < 8) | (hsv[..., 0] > 172)) & (
            (hsv[..., 1] > 130) & (hsv[..., 2] > 100)
        )
        return bool((red.sum(axis=1) > band.shape[1] * 0.08).any())

    def _execute_json_commands(
        self,
        commands: list[CombatCommand],
        character: str | None,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        """Execute commands and return True when a check confirms battle end."""
        for command in commands:
            if cancelled and cancelled():
                return False
            if command.action == "check" and self.finish_detect_enabled:
                if self.finish_detector.check(character, cancelled=cancelled):
                    return True
                continue
            self.executor.exec(command)
            self._record_skill(character, command)
        return False

    def _execute_json_action(
        self,
        action: JsonAction,
        current_character: str | None,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[str | None, bool]:
        character = action.character or current_character
        commands = [
            command
            for line in parse_combat_script(action.action)
            for command in line.commands
        ]
        self.log(f"[AutoFight] {action.name or action.action}")
        try:
            ended = self._execute_json_commands(commands, character, cancelled)
        except Exception as error:
            self.log(f"[AutoFight] {action.name or action.action} 执行失败：{error}")
            ended = False
        finally:
            self.ctx.input.release_all()
        return character, ended

    def _ensure_cast(
        self,
        action: JsonAction,
        current_character: str | None,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        """Repeat an EnsureCast action while the active E button remains ready."""
        for _ in range(5):
            if cancelled and cancelled():
                return False
            try:
                frame = self.ctx.capture_bgr()
            except Exception as error:
                self.log(f"[AutoFight] EnsureCast 截图失败，停止重试：{error}")
                return False
            if not is_skill_ready(self.ctx, frame):
                return False
            self.log(f"[AutoFight] {action.name} 未检测到技能冷却，重新执行")
            self.ctx.input.release_all()
            self.ctx.input.attack()
            self.ctx.sleep(200)
            current_character, ended = self._execute_json_action(
                action, current_character, cancelled,
            )
            if ended:
                return True
            self.ctx.sleep(30)
        return False

    def _run_pre_actions(
        self,
        strategy: JsonCombatStrategy,
        current_character: str | None,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[str | None, bool]:
        if not strategy.info.pre_actions:
            return current_character, False
        self.log("[AutoFight] JSON 策略：执行战斗前动作")
        for source in strategy.info.pre_actions:
            if cancelled and cancelled():
                return current_character, False
            lines = parse_combat_script(source)
            for line in lines:
                character = line.character or current_character
                if line.character:
                    self.executor.switch_to(line.character)
                ended = self._execute_json_commands(line.commands, character, cancelled)
                current_character = character
                if ended:
                    return current_character, True
            self.log(f"[AutoFight] 战斗前动作：{source}")
            self.ctx.sleep(300)
        return current_character, False

    def _run_json(self, cancelled: Callable[[], bool] | None = None) -> bool:
        strategy = self.json_strategy
        assert strategy is not None
        entries = expand_priorities(strategy, self.party_slots)
        if not entries:
            self.log("[AutoFight] JSON 策略没有当前队伍可用的动作")
            return False
        self.log(
            f"[AutoFight] JSON 策略“{strategy.info.name}”："
            f"{len(strategy.actions)} 个动作，展开为 {len(entries)} 个优先级条目"
        )
        evaluator = ConditionEvaluator(
            action_names=(action.name for action in strategy.actions),
            party_names=self.party_slots,
            active_character=self._active_character,
            q_ready=self._q_ready,
            e_cd=self._e_cd,
            low_hp=self._low_hp,
            last_check=lambda: max(
                0.0, time.monotonic() - self.finish_detector.last_check_at,
            ),
            log=self.log,
        )
        current_character = self._active_character()
        deadline = time.monotonic() + self.timeout_s
        self.finish_detector.start_battle()
        try:
            current_character, ended = self._run_pre_actions(
                strategy, current_character, cancelled,
            )
            if ended:
                return True
            while time.monotonic() < deadline:
                if cancelled and cancelled():
                    self.log("[AutoFight] 已取消")
                    return False
                frame = self.ctx.capture_bgr()
                evaluator.set_frame(frame)
                executed = False
                previous_character = current_character
                for entry in entries:
                    action = entry.action
                    if not evaluator.evaluate(
                        entry.expression, action.index, action.character, action.name,
                    ):
                        continue
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
                    if action.character:
                        self.executor.switch_to(action.character)
                        current_character = action.character
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
                    current_character, ended = self._execute_json_action(
                        action, current_character, cancelled,
                    )
                    if ended:
                        self.log("[AutoFight] check 确认战斗结束")
                        return True
                    if action.ensure_cast and self._ensure_cast(
                        action, current_character, cancelled,
                    ):
                        return True
                    evaluator.update_last_exec_time(action.index, action.name)
                    executed = True
                    break
                if not executed:
                    self.ctx.sleep(200)
            self.log("[AutoFight] 超时退出")
            return False
        finally:
            self.ctx.input.release_all()
