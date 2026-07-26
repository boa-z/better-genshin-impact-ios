"""pathing 执行框架。

原版 PathExecutor 的核心依赖是小地图定位（SIFT/模板匹配到大地图，地图特征
数据以 NuGet 包 BetterGI.Assets.Map 分发，不在脚本仓库内）。本移植版把定位
抽象为 Positioner 协议：
- 提供 Positioner 时，执行器可完整走点（转向 + 前进 + 到点触发 action）；
- 未提供时，teleport/寻路会抛出明确的 NotImplementedError，但文件解析、
  校验与 action 执行（战斗 DSL 等）可用。

相机朝向检测（小地图视野扇形）已实现简化版，供转向闭环使用。
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional, Protocol

import cv2
import numpy as np

from ..combat.dsl import CombatExecutor
from ..engine.context import GameContext
from .model import PathingTask, Waypoint


class Positioner(Protocol):
    def get_position(self, bgr: np.ndarray) -> Optional[tuple[float, float]]:
        """当前帧 → 世界地图坐标；无法定位返回 None。"""
        ...


def camera_orientation_deg(ctx: GameContext, bgr: np.ndarray) -> Optional[float]:
    """从小地图视野扇形估计相机朝向（度，地图北为 0，顺时针）。

    简化版：取小地图圆环区域，找亮度显著高于环境的扇形并取角平分线。
    小地图位置来自布局 profile 的 minimap 配置。
    """
    mm = getattr(ctx.layout, "buttons", {}).get("minimapCenter")
    if mm is None:
        return None
    cx, cy = mm[0] * ctx.transform.device_width, mm[1] * ctx.transform.device_height
    r = 0.06 * ctx.transform.device_width
    x0, y0 = int(cx - r), int(cy - r)
    size = int(2 * r)
    if x0 < 0 or y0 < 0 or y0 + size > bgr.shape[0] or x0 + size > bgr.shape[1]:
        return None
    crop = cv2.cvtColor(bgr[y0:y0 + size, x0:x0 + size], cv2.COLOR_BGR2GRAY).astype(np.float32)
    angles = np.linspace(0, 2 * math.pi, 360, endpoint=False)
    ring_r = size * 0.42
    xs = (size / 2 + ring_r * np.cos(angles)).astype(int).clip(0, size - 1)
    ys = (size / 2 + ring_r * np.sin(angles)).astype(int).clip(0, size - 1)
    profile = crop[ys, xs]
    profile = profile - profile.mean()
    # 视野扇形约 90°：与宽度 90 的窗口做圆周相关，峰值即扇形中心
    kernel = np.zeros(360, np.float32)
    kernel[:90] = 1
    kernel -= kernel.mean()
    corr = np.real(np.fft.ifft(np.fft.fft(profile) * np.conj(np.fft.fft(kernel))))
    center_idx = (int(np.argmax(corr)) + 45) % 360
    if corr.max() < profile.std() * 30:  # 峰值不显著则认为检测失败
        return None
    # 图像坐标角 → 地图方位角（北=0，顺时针）
    return (center_idx + 90) % 360


class PathingExecutor:
    def __init__(self, ctx: GameContext, positioner: Positioner | None = None,
                 party_slots: dict[str, int] | None = None,
                 log: Callable[[str], None] = print, map_name: str = "Teyvat"):
        self.ctx = ctx
        if positioner is None:
            try:
                from .positioner import MinimapPositioner
                positioner = MinimapPositioner(ctx, map_name)
                log(f"[pathing] 已加载 {map_name} 地图定位（SIFT）")
            except FileNotFoundError as e:
                log(f"[pathing] 无地图定位：{e}")
        self.positioner = positioner
        self.log = log
        self.combat = CombatExecutor(ctx.input, sleep=ctx.sleep, party_slots=party_slots, log=log)

    def run(self, task: PathingTask) -> None:
        self.log(f"[pathing] {task.name}: {len(task.positions)} 个路点 @ {task.map_name}")
        for wp in task.positions:
            if wp.type == "teleport":
                self._teleport(wp)
            elif wp.type in ("path", "target"):
                self._move_to(wp)
            elif wp.type == "orientation":
                pass  # 朝向点：由 _move_to 的转向闭环覆盖
            if wp.action:
                self._do_action(wp)
        self.ctx.input.release_all()

    # ---- movement ----

    _tp_task = None

    def _teleport(self, wp: Waypoint) -> None:
        if self._tp_task is None:
            from .tp import TpTask
            self._tp_task = TpTask(self.ctx, log=self.log)
        self._tp_task.tp(wp.x, wp.y)
        if self.positioner is not None:
            self.positioner.reset()  # 传送后位置突变，清除局部搜索缓存

    def _move_to(self, wp: Waypoint, timeout_s: float = 60, arrive_dist: float = 2.0) -> None:
        if self.positioner is None:
            raise NotImplementedError(
                "寻路需要小地图定位（Positioner）。当前未配置地图特征资产，"
                "参见 docs/ROADMAP.md。可先使用战斗/宏/JS 脚本能力。"
            )
        deadline = time.monotonic() + timeout_s
        moving = False
        try:
            while time.monotonic() < deadline:
                frame = self.ctx.capture_bgr()
                pos = self.positioner.get_position(frame)
                if pos is None:
                    self.log("[pathing] 定位失败，短暂停止重试")
                    if moving:
                        self.ctx.input.key_up("W")
                        moving = False
                    self.ctx.sleep(500)
                    continue
                dx, dy = wp.x - pos[0], wp.y - pos[1]
                dist = math.hypot(dx, dy)
                if dist <= arrive_dist:
                    break
                target_deg = (math.degrees(math.atan2(dx, -dy))) % 360
                cam = camera_orientation_deg(self.ctx, frame)
                if cam is not None:
                    delta = (target_deg - cam + 540) % 360 - 180
                    if abs(delta) > 8:
                        self.ctx.input.move_camera_by(delta * 8, 0)  # 8 px/°，需标定
                if not moving:
                    self.ctx.input.key_down("W")
                    moving = True
                if wp.move_mode == "dash":
                    self.ctx.input.key_press("LSHIFT")
                elif wp.move_mode in ("fly", "jump"):
                    self.ctx.input.key_press("SPACE")
                self.ctx.sleep(300)
        finally:
            if moving:
                self.ctx.input.key_up("W")

    # ---- actions ----

    def _do_action(self, wp: Waypoint) -> None:
        a = wp.action
        if a == "combat_script":
            self.combat.run(wp.action_params)
        elif a == "fight":
            self.log("[pathing] fight：使用通用攻击（未配置战斗策略）")
            self.combat.run("attack(8)")
        elif a == "stop_flying":
            self.ctx.input.key_press("SPACE")  # 落地：再按跳跃收翼
            try:
                self.ctx.sleep(float(wp.action_params or 500))
            except ValueError:
                self.ctx.sleep(500)
        elif a in ("mining", "pick_around", "pick_up_collect"):
            for _ in range(6):
                self.ctx.input.key_press("F")
                self.ctx.sleep(400)
        elif a in ("nahida_collect", "hydro_collect", "electro_collect", "anemo_collect", "pyro_collect"):
            self.ctx.input.key_press("E")
            self.ctx.sleep(1000)
        elif a == "use_gadget":
            self.ctx.input.key_press("Z")
        elif a == "log_output":
            self.log(f"[pathing] {wp.action_params}")
        elif a:
            self.log(f"[pathing] 动作 {a} 暂未实现，已跳过")
