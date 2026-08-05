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

from ..engine.context import GameContext
from .actions import PathingActionRunner
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
        self.positioner = positioner
        self._map_name = map_name
        self.log = log
        self._tp_task = None
        self._cam_gain = 5.5
        self._cam_sign = 1.0
        self._last_mode_action_at = 0.0
        self.actions = PathingActionRunner(ctx, party_slots=party_slots, log=log)

    def _ensure_positioner(self, map_name: str) -> None:
        if self.positioner is not None:
            return
        try:
            from .positioner import MinimapPositioner
            self.positioner = MinimapPositioner(self.ctx, map_name)
            self._map_name = map_name
            self.log(f"[pathing] 已加载 {map_name} 地图定位（SIFT）")
        except FileNotFoundError as e:
            self.log(f"[pathing] 无地图定位：{e}")

    def _enable_realtime_triggers(self, task: PathingTask) -> tuple[list[str], list[object] | None]:
        enabled: list[str] = []
        if not hasattr(self.ctx, "enable_trigger"):
            return enabled, None
        loop = getattr(self.ctx, "_trigger_loop", None)
        previous = list(loop.triggers) if loop is not None else None
        for name, active in task.realtime_triggers.items():
            if not active:
                continue
            if name not in {"AutoPick", "AutoSkip"}:
                self.log(f"[pathing] 实时触发器 {name} 暂不支持")
                continue
            self.ctx.enable_trigger(name)
            enabled.append(name)
        return enabled, previous

    def _clear_realtime_triggers(
        self,
        state: tuple[list[str], list[object] | None],
    ) -> None:
        enabled, previous = state
        if not enabled:
            return
        loop = getattr(self.ctx, "_trigger_loop", None)
        if loop is None:
            return
        if previous is None:
            loop.clear()
        else:
            loop.triggers = previous
            if previous:
                loop.start()

    def run(self, task: PathingTask) -> bool:
        task.validate()
        self.log(f"[pathing] {task.name}: {len(task.positions)} 个路点 @ {task.map_name}")
        if task.map_match_method.lower() not in {"sift"}:
            self.log(f"[pathing] {task.map_match_method} 暂未移植，回退 SIFT")
        self._ensure_positioner(task.map_name)
        trigger_state = self._enable_realtime_triggers(task)
        retry_count = self._retry_count(task)
        try:
            for index, wp in enumerate(task.positions, start=1):
                self.log(f"[pathing] 路点 {index}/{len(task.positions)} id={wp.id}")

                # BetterGI handles four-leaf seals before normal movement. The
                # waypoint itself is a camera target, not a walking target.
                if wp.action == "up_down_grab_leaf":
                    self._run_with_retry(lambda: self._face_to(wp), wp, retry_count)
                    self._do_action(wp)
                    continue

                if wp.action == "log_output":
                    self._do_action(wp)

                if wp.type == "teleport" or wp.action == "force_tp":
                    self._run_with_retry(lambda: self._teleport(wp), wp, retry_count)
                elif wp.type == "orientation":
                    self._run_with_retry(lambda: self._face_to(wp), wp, retry_count)
                elif wp.type in ("path", "target"):
                    self._run_with_retry(
                        lambda: self._move_to(wp, arrive_dist=2.0 if wp.type == "target" else 4.0),
                        wp,
                        retry_count,
                    )
                else:
                    self.log(f"[pathing] 未知路点类型 {wp.type}，跳过移动")

                if wp.action and wp.action not in {"force_tp", "log_output"}:
                    self._do_action(wp)
            return True
        finally:
            self.ctx.input.release_all()
            self._clear_realtime_triggers(trigger_state)

    @staticmethod
    def _retry_count(task: PathingTask) -> int:
        raw = task.config.get("retry_times", task.config.get("retryTimes", 1))
        try:
            return max(0, min(5, int(raw)))
        except (TypeError, ValueError):
            return 1

    def _run_with_retry(self, operation, waypoint: Waypoint, retries: int) -> None:
        for attempt in range(retries + 1):
            try:
                operation()
                return
            except TimeoutError as e:
                if attempt >= retries:
                    raise
                self.log(
                    f"[pathing] 路点 {waypoint.id} 失败，重试 {attempt + 1}/{retries}: {e}"
                )
                if self.positioner is not None:
                    reset = getattr(self.positioner, "reset", None)
                    if callable(reset):
                        reset()
                self.ctx.input.release_all()
                self.ctx.sleep(800)

    # ---- movement ----

    def _teleport(self, wp: Waypoint) -> None:
        if self._tp_task is None:
            from .tp import TpTask
            self._tp_task = TpTask(self.ctx, log=self.log)
        self._tp_task.tp(wp.x, wp.y)
        if self.positioner is not None:
            # 传送落点≈目标锚点：直接设为局部搜索先验（白天/城内全局匹配不稳）
            if hasattr(self.positioner, "set_prior"):
                self.positioner.set_prior(wp.x, wp.y)
            else:
                reset = getattr(self.positioner, "reset", None)
                if callable(reset):
                    reset()

    # 相机反馈增益（px/°）与符号：位移反馈自校准（见 _move_to）
    @staticmethod
    def _bearing(dx: float, dy: float) -> float:
        """世界坐标位移 → 罗盘方位角（北=0，顺时针）。

        原神世界坐标轴向：+x=地图西，+y=地图北（img = origin − 2·world）。
        """
        return math.degrees(math.atan2(-dx, dy)) % 360

    def _get_position(self, frame: np.ndarray) -> Optional[tuple[float, float]]:
        if self.positioner is None:
            return None
        stable = getattr(self.positioner, "get_position_stable", None)
        if callable(stable):
            return stable(frame)
        return self.positioner.get_position(frame)

    def _face_to(self, wp: Waypoint) -> None:
        if self.positioner is None:
            self.log("[pathing] 方位点缺少地图定位，跳过朝向")
            return
        frame = self.ctx.capture_bgr()
        position = self._get_position(frame)
        if position is None:
            self.log("[pathing] 方位点定位失败，跳过朝向")
            return
        desired = self._bearing(wp.x - position[0], wp.y - position[1])
        current = camera_orientation_deg(self.ctx, frame)
        if current is None:
            self.log("[pathing] 方位点无法识别当前视角")
            return
        delta = (desired - current + 540) % 360 - 180
        if abs(delta) > 10:
            self.ctx.input.move_camera_by(self._cam_sign * delta * self._cam_gain, 0)
            self.ctx.sleep(500)

    def _move_to(self, wp: Waypoint, timeout_s: float = 120, arrive_dist: float = 3.0) -> None:
        if self.positioner is None:
            raise NotImplementedError(
                "寻路需要小地图定位（Positioner）。当前未配置地图特征资产，"
                "参见 docs/ROADMAP.md。可先使用战斗/宏/JS 脚本能力。"
            )
        deadline = time.monotonic() + timeout_s
        moving = False
        last_fix: tuple[float, float, float] | None = None  # (x, y, t)
        prev_err: float | None = None
        recent_positions: list[tuple[float, float]] = []
        lost_fixes = 0
        reached = False
        try:
            while time.monotonic() < deadline:
                try:
                    frame = self.ctx.capture_bgr()
                except Exception as e:  # 设备偶发超时，重试
                    self.log(f"[pathing] 截图失败重试: {e}")
                    self.ctx.sleep(1000)
                    continue
                pos = self._get_position(frame)
                now = time.monotonic()
                if pos is None:
                    lost_fixes += 1
                    if moving:
                        self.ctx.input.key_up("W")
                        moving = False
                    if lost_fixes >= 8:
                        self.log("[pathing] 连续无法定位，重置地图匹配缓存")
                        reset = getattr(self.positioner, "reset", None)
                        if callable(reset):
                            reset()
                        lost_fixes = 0
                    self.ctx.sleep(600)
                    continue
                lost_fixes = 0
                dx, dy = wp.x - pos[0], wp.y - pos[1]
                dist = math.hypot(dx, dy)
                if dist <= arrive_dist:
                    self.log(f"[pathing] 到达路点 ({wp.x:.0f},{wp.y:.0f})")
                    reached = True
                    break

                recent_positions.append(pos)
                if len(recent_positions) > 8:
                    recent_positions.pop(0)
                if moving and len(recent_positions) == 8:
                    moved = math.hypot(
                        recent_positions[-1][0] - recent_positions[0][0],
                        recent_positions[-1][1] - recent_positions[0][1],
                    )
                    if moved < 3.0:
                        self.log("[pathing] 疑似卡死，停止移动并尝试脱困")
                        self.ctx.input.key_up("W")
                        moving = False
                        self.ctx.input.move_camera_by(self._cam_sign * 220, 0)
                        if wp.move_mode != "climb":
                            self.ctx.input.key_press("SPACE")
                        self.ctx.sleep(700)
                        recent_positions.clear()

                desired = self._bearing(dx, dy)

                # 位移反馈：用实际移动向量估计当前航向并修正相机
                if last_fix is not None and moving:
                    mdx, mdy = pos[0] - last_fix[0], pos[1] - last_fix[1]
                    if math.hypot(mdx, mdy) >= 1.5:  # 位移足够大才可信
                        heading = self._bearing(mdx, mdy)
                        err = (desired - heading + 540) % 360 - 180
                        self.log(f"[pathing] 距离{dist:.0f} 期望{desired:.0f}° 实际{heading:.0f}° "
                                 f"误差{err:+.0f}° 符号{self._cam_sign:+.0f}")
                        if prev_err is not None and abs(err) > abs(prev_err) + 25:
                            self._cam_sign = -self._cam_sign  # 越修越偏 → 反号
                            self.log(f"[pathing] 相机增益反号 → {self._cam_sign:+.0f}")
                        prev_err = err
                        if abs(err) > 10:
                            # 摇杆按住时合成手势内的相机拖动无效（实测），
                            # 必须停步后独立滑动转相机，再继续前进
                            self.ctx.input.key_up("W")
                            moving = False
                            # 等手势泵当前原子手势(≤1.4s)走完，否则滑动会被
                            # 当成第二根手指而失效
                            self.ctx.sleep(1600)
                            turn = max(-90.0, min(90.0, err))  # 单次限幅防过冲
                            self.ctx.input.move_camera_by(
                                self._cam_sign * turn * self._cam_gain, 0)
                            self.ctx.sleep(500)
                        last_fix = (pos[0], pos[1], now)
                elif last_fix is None:
                    # 起步引导：用小地图扇形粗对准一次（失败就直接开走靠反馈纠偏）
                    cam = camera_orientation_deg(self.ctx, frame)
                    if cam is not None:
                        delta = (desired - cam + 540) % 360 - 180
                        if abs(delta) > 12:
                            self.ctx.input.move_camera_by(
                                self._cam_sign * max(-90.0, min(90.0, delta)) * self._cam_gain, 0)
                            self.ctx.sleep(500)
                    last_fix = (pos[0], pos[1], now)

                if not moving:
                    self.ctx.input.key_down("W")
                    moving = True
                if now - self._last_mode_action_at >= 1.0:
                    if wp.move_mode in ("run", "dash"):
                        self.ctx.input.key_press("LSHIFT")
                    elif wp.move_mode in ("fly", "jump"):
                        self.ctx.input.key_press("SPACE")
                    self._last_mode_action_at = now
                if last_fix is not None and now - last_fix[2] < 0.9:
                    self.ctx.sleep(400)  # 攒足位移再做下一次航向估计
        finally:
            if moving:
                self.ctx.input.key_up("W")
        if not reached:
            raise TimeoutError(f"[pathing] 路点 ({wp.x:.0f},{wp.y:.0f}) 执行超时")

    # ---- actions ----

    def _do_action(self, wp: Waypoint) -> None:
        self.actions.run(wp)
