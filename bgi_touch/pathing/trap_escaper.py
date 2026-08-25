"""地图追踪卡死脱困。

BetterGI 的桌面实现把脱困动作放在 ``TrapEscaper`` 中，而不是直接散落在
路径循环里。本模块保留同一组行为约定，同时把输入和定位器作为依赖注入，
这样可以在没有设备的情况下验证按键释放、绕障和重试边界。
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from typing import Any, Optional, Protocol

import numpy as np

from .actions import detect_motion_status
from .camera import crop_minimap_for_orientation, orientation_with_confidence


class TrapPositioner(Protocol):
    def get_position(self, bgr: np.ndarray) -> Optional[tuple[float, float]]:
        ...


def _camera_orientation(ctx: Any, frame: np.ndarray) -> float | None:
    """Best-effort local copy of the pathing camera estimator.

    Importing ``executor`` here would create a module cycle.  Keeping this
    small adapter local also makes the escaper usable by other pathing tasks.
    """

    crop = crop_minimap_for_orientation(ctx, frame)
    if crop is None:
        return None
    angle, _confidence = orientation_with_confidence(crop)
    return angle


class StuckDetector:
    """Match BetterGI's eight-sample, low-displacement trap detector.

    The desktop code samples approximately once per second and compares the
    first and last item in an eight-position window using Manhattan distance.
    A detected window is consumed before the next recovery attempt so one
    stationary window cannot trigger multiple escapes.
    """

    def __init__(self, window_size: int = 8, movement_threshold: float = 3.0):
        if window_size < 2:
            raise ValueError("卡死检测窗口至少需要两个位置样本")
        if movement_threshold < 0:
            raise ValueError("卡死检测位移阈值不能为负数")
        self.window_size = int(window_size)
        self.movement_threshold = float(movement_threshold)
        self._positions: list[tuple[float, float]] = []
        self.trap_count = 0

    @property
    def positions(self) -> tuple[tuple[float, float], ...]:
        return tuple(self._positions)

    def reset_window(self) -> None:
        self._positions.clear()

    def add(self, position: tuple[float, float]) -> bool:
        """Add a position and return whether this sample closes a trap window."""

        current = (float(position[0]), float(position[1]))
        self._positions.append(current)
        if len(self._positions) > self.window_size:
            self._positions.pop(0)
        if len(self._positions) < self.window_size:
            return False

        start = self._positions[0]
        end = self._positions[-1]
        moved = abs(end[0] - start[0]) + abs(end[1] - start[1])
        if moved >= self.movement_threshold:
            return False

        self.trap_count += 1
        self.reset_window()
        return True


class TrapEscaper:
    """Perform the short recovery manoeuvres used by BetterGI pathing."""

    MAX_MOVE_SECONDS = 25.0
    ACTION_IDLE_SECONDS = 5.0
    RANDOM_RESET_SECONDS = 1.5

    def __init__(
        self,
        ctx: Any,
        positioner: TrapPositioner,
        *,
        log: Callable[[str], None] = print,
        cam_sign: float = 1.0,
        cam_gain: float = 5.5,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ):
        self.ctx = ctx
        self.positioner = positioner
        self.log = log
        self.cam_sign = float(cam_sign)
        self.cam_gain = float(cam_gain)
        self.clock = clock
        self.rng = rng or random.Random()
        self._last_action_index = 0
        self._last_action_at = self.clock()
        self._random_angle = 0

    def _increase_random_angle(self) -> None:
        self._random_angle += self.rng.randrange(30, 45)

    def _reduce_random_angle(self) -> None:
        self._random_angle += self.rng.randrange(-45, -29)

    @staticmethod
    def _bearing(dx: float, dy: float) -> float:
        return math.degrees(math.atan2(-dx, dy)) % 360

    @staticmethod
    def _delta(target: float, current: float) -> float:
        return (target - current + 540) % 360 - 180

    def _attack(self) -> None:
        attack = getattr(self.ctx.input, "attack", None)
        if callable(attack):
            attack()
        else:
            # Minimal hosts may only expose the semantic key API.
            self.ctx.input.key_press("J")

    def _move_sideways(self, key: str, delay_ms: int, *, drop: bool = False) -> None:
        self.ctx.input.key_down(key)
        try:
            self.ctx.sleep(300)
            self.ctx.input.key_press("SPACE")
            self.ctx.sleep(delay_ms)
        finally:
            self.ctx.input.key_up(key)
        if drop:
            self.ctx.input.key_press("SPACE")

    def rotate_and_move(self) -> None:
        """Break out of the current collision pocket.

        This is the bounded backward/left/right sequence from the upstream
        ``TrapEscaper.RotateAndMove``.  All held keys are released before and
        after the sequence so a failed device gesture cannot keep moving.
        """

        self._increase_random_angle()
        self.ctx.input.key_up("W")
        self.ctx.input.key_press("SPACE")
        self.ctx.sleep(75)
        self._attack()
        self.ctx.sleep(500)

        now = self.clock()
        if now - self._last_action_at >= 10.0:
            self._last_action_index = 0
        else:
            self._last_action_index += 1
        difference = self._last_action_index * 1000

        try:
            mode = self._last_action_index % 3
            if mode == 0:
                self.ctx.input.key_down("S")
                try:
                    self.ctx.sleep(500)
                    self.ctx.input.key_press("SPACE")
                    self.ctx.sleep(1000 + difference)
                finally:
                    self.ctx.input.key_up("S")
            elif mode == 1:
                self._move_sideways("A", 700 + difference)
            else:
                self._move_sideways("D", 700 + difference, drop=True)
        finally:
            self.ctx.input.key_up("W")
        self._last_action_at = self.clock()

    def _position(self, frame: np.ndarray) -> tuple[float, float] | None:
        stable = getattr(self.positioner, "get_position_stable", None)
        if callable(stable):
            return stable(frame)
        return self.positioner.get_position(frame)

    def _turn_towards(self, frame: np.ndarray, position: tuple[float, float],
                      target: tuple[float, float]) -> None:
        desired = self._bearing(target[0] - position[0], target[1] - position[1])
        try:
            current = _camera_orientation(self.ctx, frame)
        except Exception as error:
            self.log(f"[pathing] 脱困视角识别失败，使用随机绕障角度：{error}")
            current = None
        if current is None:
            delta = float(self._random_angle)
        else:
            delta = self._delta(desired, current) + self._random_angle
        delta = max(-90.0, min(90.0, delta))
        if abs(delta) <= 8.0:
            return
        self.ctx.input.key_up("W")
        # DeviceHub's touch pump serializes the current joystick gesture.  A
        # short settle period prevents the camera swipe becoming a second
        # finger in that gesture.
        self.ctx.sleep(1600)
        self.ctx.input.move_camera_by(self.cam_sign * delta * self.cam_gain, 0)
        self.ctx.sleep(500)
        self.ctx.input.key_down("W")

    def move_to(self, target: tuple[float, float], move_mode: str = "walk") -> None:
        """Walk toward a target for at most 25 s while correcting obstacles."""

        started = self.clock()
        last_action_at = started
        moving = False
        try:
            self.ctx.input.key_down("W")
            moving = True
            while True:
                now = self.clock()
                if now - last_action_at > self.ACTION_IDLE_SECONDS:
                    break
                if now - started > self.MAX_MOVE_SECONDS:
                    self.log("[pathing] 卡死脱困超时")
                    break
                try:
                    frame = self.ctx.capture_bgr()
                except Exception as error:
                    self.log(f"[pathing] 脱困截图失败：{error}")
                    self.ctx.sleep(250)
                    continue

                position = self._position(frame)
                if position is None:
                    self.ctx.sleep(250)
                    continue
                if math.hypot(target[0] - position[0], target[1] - position[1]) <= 4.0:
                    break

                self._turn_towards(frame, position, target)
                if move_mode != "climb":
                    try:
                        status = detect_motion_status(
                            frame, getattr(self.ctx, "transform", None)
                        )
                    except Exception as error:
                        self.log(f"[pathing] 脱困运动状态识别失败：{error}")
                        status = None
                    if status == "climb":
                        self.ctx.input.key_up("W")
                        moving = False
                        self.ctx.input.key_press("SPACE")
                        self.ctx.sleep(75)
                        self.ctx.input.key_down("S")
                        try:
                            self.ctx.sleep(700)
                        finally:
                            self.ctx.input.key_up("S")
                        self._increase_random_angle()
                        last_action_at = self.clock()
                        self.ctx.input.key_down("W")
                        moving = True
                        continue

                if self._random_angle and self.clock() - self._last_action_at > self.RANDOM_RESET_SECONDS:
                    self._random_angle = 0
                if not moving:
                    self.ctx.input.key_down("W")
                    moving = True
                self.ctx.sleep(100)
        finally:
            # Keep cleanup idempotent for hosts that track key state.
            self.ctx.input.key_up("W")
