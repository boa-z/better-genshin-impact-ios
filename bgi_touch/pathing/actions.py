"""地图追踪路点动作。

BetterGI 的 Windows 动作处理器依赖桌面窗口、键盘扫描和角色识别。这里
把能稳定映射到 DeviceHub 触控输入的动作集中起来，避免 PathingExecutor
里堆积大量与路线控制无关的分支。
"""

from __future__ import annotations

import json
import re
from typing import Callable

import numpy as np

from ..combat.dsl import CombatExecutor
from ..engine.context import GameContext
from ..tasks.dispatcher import TaskDispatcher
from .model import Waypoint


class PathingActionRunner:
    """Execute the portable subset of BetterGI's waypoint action handlers."""

    SUPPORTED = frozenset({
        "combat_script", "fight", "normal_attack", "elemental_skill", "stop_flying", "mining",
        "linnea_mining", "nahida_collect", "hydro_collect", "electro_collect",
        "anemo_collect", "pyro_collect", "pick_around", "pick_up_collect",
        "fishing", "use_gadget", "up_down_grab_leaf", "log_output", "force_tp",
        "exit_and_relogin", "set_time", "wonderland_cycle",
    })

    def __init__(
        self,
        ctx: GameContext,
        *,
        party_slots: dict[str, int] | None = None,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.log = log
        self.combat = CombatExecutor.for_context(ctx, party_slots=party_slots, log=log)

    @staticmethod
    def _number(value: str, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def run(self, waypoint: Waypoint) -> bool:
        action = waypoint.action.strip().lower()
        if not action:
            return True
        handler = getattr(self, f"_action_{action}", None)
        if handler is None:
            self.log(f"[pathing] 动作 {action} 暂不支持，已停止在该路点前")
            return False
        handler(waypoint)
        return True

    def _action_combat_script(self, waypoint: Waypoint) -> None:
        if not waypoint.action_params.strip():
            self.log("[pathing] combat_script 缺少 action_params")
            return
        self.combat.run(waypoint.action_params)

    def _action_fight(self, waypoint: Waypoint) -> None:
        self.log("[pathing] 执行通用战斗")
        self.combat.run("attack(8)")

    def _action_normal_attack(self, waypoint: Waypoint) -> None:
        seconds = max(0.2, self._number(waypoint.action_params, 1.2))
        self.combat.run(f"attack({seconds})")

    def _action_elemental_skill(self, waypoint: Waypoint) -> None:
        self.ctx.input.key_press("E")
        self.ctx.sleep(1000)

    def _action_stop_flying(self, waypoint: Waypoint) -> None:
        # The original handler watches motion status. Two jump inputs are the
        # portable fallback; the attack input speeds up the final descent.
        wait_ms = max(0, int(self._number(waypoint.action_params, 0)))
        if wait_ms:
            self.ctx.input.key_press("SPACE")
            self.ctx.sleep(wait_ms)
            self.ctx.input.key_press("SPACE")
        self.ctx.input.attack()
        self.ctx.sleep(350)

    def _action_mining(self, waypoint: Waypoint) -> None:
        for _ in range(6):
            self.ctx.input.attack()
            self.ctx.sleep(280)

    def _action_linnea_mining(self, waypoint: Waypoint) -> None:
        self.ctx.input.key_press("E", hold_ms=900)
        self.ctx.sleep(700)
        self._action_mining(waypoint)

    def _action_nahida_collect(self, waypoint: Waypoint) -> None:
        self.ctx.input.key_press("E", hold_ms=900)
        self.ctx.sleep(900)

    def _action_hydro_collect(self, waypoint: Waypoint) -> None:
        self._action_elemental_collect(waypoint, "hydro")

    def _action_electro_collect(self, waypoint: Waypoint) -> None:
        self._action_elemental_collect(waypoint, "electro")

    def _action_anemo_collect(self, waypoint: Waypoint) -> None:
        self._action_elemental_collect(waypoint, "anemo")

    def _action_pyro_collect(self, waypoint: Waypoint) -> None:
        self._action_elemental_collect(waypoint, "pyro")

    def _action_elemental_collect(self, waypoint: Waypoint, element: str) -> None:
        self.log(f"[pathing] {element} 元素采集：使用当前角色元素战技")
        self.ctx.input.key_press("E", hold_ms=650)
        self.ctx.sleep(650)

    def _action_pick_around(self, waypoint: Waypoint) -> None:
        for _ in range(6):
            self.ctx.input.key_press("F")
            self.ctx.sleep(350)

    def _action_pick_up_collect(self, waypoint: Waypoint) -> None:
        if waypoint.action_params.strip():
            self.combat.run(waypoint.action_params)
        else:
            self._action_pick_around(waypoint)

    def _action_fishing(self, waypoint: Waypoint) -> None:
        config: dict = {}
        raw = waypoint.action_params.strip()
        if raw.startswith("{"):
            try:
                config = json.loads(raw)
            except json.JSONDecodeError:
                self.log("[pathing] fishing action_params 不是有效 JSON，使用默认参数")
        elif raw:
            config["targetCatches"] = max(1, int(self._number(raw, 1)))
        TaskDispatcher(self.ctx, log=self.log).run_auto_fishing_task(config)

    def _action_use_gadget(self, waypoint: Waypoint) -> None:
        self.ctx.input.key_press("Z")
        if "not_wait" not in waypoint.action_params.lower():
            self.ctx.sleep(max(300, int(self._number(waypoint.action_params, 300))))

    def _action_up_down_grab_leaf(self, waypoint: Waypoint) -> None:
        direction = -1.0 if waypoint.action_params.strip().lower() == "down" else 1.0
        for cycle in range(40):
            frame = self.ctx.capture_bgr()
            if self._leaf_prompt_visible(frame):
                self.log("[pathing] 检测到四叶印，执行交互")
                self.ctx.input.key_press("F")
                self.ctx.sleep(250)
                self.ctx.input.key_press("SPACE")
                return
            self.ctx.input.move_camera_by(0, direction * (1000 if cycle % 10 == 0 else 350))
            self.ctx.sleep(120)
        self.ctx.input.move_camera_by(0, -direction * 600)
        self.log("[pathing] 未检测到四叶印，已恢复视角")

    def _leaf_prompt_visible(self, bgr: np.ndarray) -> bool:
        points = ((1500, 1000), (1508, 1041), (1500, 987), (1500, 1010))
        shifted = (0, 120, -104)
        for offset in shifted:
            hits = 0
            for ref_x, ref_y in points:
                x, y = self.ctx.transform.to_device(ref_x + offset, ref_y)
                ix, iy = int(round(x)), int(round(y))
                if 0 <= iy < bgr.shape[0] and 0 <= ix < bgr.shape[1]:
                    pixel = bgr[iy, ix]
                    if bool(np.all(pixel >= 245)):
                        hits += 1
            if hits >= 3:
                return True
        return False

    def _action_log_output(self, waypoint: Waypoint) -> None:
        self.log(f"[pathing] {waypoint.action_params}")

    def _action_force_tp(self, waypoint: Waypoint) -> None:
        # PathingExecutor handles this action before movement so it can reuse
        # the waypoint coordinate as the teleport target.
        self.log("[pathing] force_tp 已由传送流程处理")

    def _action_exit_and_relogin(self, waypoint: Waypoint) -> None:
        from ..engine.genshin_api import GenshinApi

        self.log("[pathing] 退出并重新登录原神")
        GenshinApi(self.ctx, log=self.log).relogin()

    def _action_set_time(self, waypoint: Waypoint) -> None:
        from ..engine.genshin_api import GenshinApi

        raw = waypoint.action_params.strip()
        matched = re.fullmatch(
            r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<skip>true|false))?",
            raw,
            flags=re.IGNORECASE,
        )
        if matched is None:
            raise ValueError(
                "set_time action_params 必须为 H:mm、HH:mm 或 HH:mm:true/false"
            )
        hour = int(matched.group("hour"))
        minute = int(matched.group("minute"))
        if not 0 <= hour <= 24 or not 0 <= minute <= 59:
            raise ValueError("set_time 时间必须为 hour 0-24、minute 0-59")
        skip_value = matched.group("skip")
        # BetterGI's SetTimeHandler defaults to skipping the clock animation.
        skip_animation = skip_value is None or skip_value.lower() == "true"
        self.log(
            f"[pathing] 设置游戏时间 {hour:02d}:{minute:02d}"
            f"（跳过动画：{'是' if skip_animation else '否'}）"
        )
        if not GenshinApi(self.ctx, log=self.log).setTime(
            hour, minute, skip_animation
        ):
            raise RuntimeError("set_time 失败：未能打开或确认游戏时间界面")

    def _action_wonderland_cycle(self, waypoint: Waypoint) -> None:
        from ..engine.genshin_api import GenshinApi

        self.log("[pathing] 进入并退出千星奇域")
        if not GenshinApi(self.ctx, log=self.log).wonderlandCycle():
            raise RuntimeError("wonderland_cycle 失败：未能完成千星奇域进入退出流程")
