"""Stateful BetterGI mouse-pointer semantics over iOS touch gestures."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from ..vision.coordinate import REF_HEIGHT, REF_WIDTH


class TouchPointer:
    """Keep a virtual PC cursor and collapse held movement into one swipe.

    iOS has no persistent pointer contact that can be moved through separate
    MCP calls. BetterGI UI scripts, however, commonly issue ``moveMouseTo``,
    ``leftButtonDown``, many moves, and ``leftButtonUp``. Buffering that path
    until release preserves the resulting drag without extra screenshots or
    dozens of tiny DeviceHub requests.
    """

    INTENT_TTL_S = 5.0

    def __init__(self, input_simulator, clock: Callable[[], float] = time.monotonic):
        self.input = input_simulator
        self.clock = clock
        self.width = REF_WIDTH
        self.height = REF_HEIGHT
        self.dpi = 1.0
        self.cursor: tuple[float, float] | None = None
        self._intent_at = 0.0
        self._drag_start: tuple[float, float] | None = None
        self._drag_started_at = 0.0
        self._attack_held = False

    def set_metrics(self, width: float, height: float, dpi: float = 1.0) -> None:
        width, height, dpi = int(width), int(height), float(dpi)
        if width <= 0 or height <= 0 or width * 9 != height * 16:
            raise ValueError("游戏分辨率必须是16:9的正整数分辨率")
        if not math.isfinite(dpi) or dpi <= 0:
            raise ValueError("DPI 缩放必须为正数")
        self.width, self.height, self.dpi = width, height, dpi

    def get_metrics(self) -> list[float]:
        return [self.width, self.height, self.dpi]

    def _to_ref(self, x: float, y: float) -> tuple[float, float]:
        x, y = float(x), float(y)
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            raise ValueError("鼠标坐标超出游戏窗口范围")
        return x * REF_WIDTH / self.width, y * REF_HEIGHT / self.height

    def _delta_to_ref(self, dx: float, dy: float) -> tuple[float, float]:
        return (
            float(dx) * REF_WIDTH / self.width,
            float(dy) * REF_HEIGHT / self.height,
        )

    def _arm(self) -> None:
        self._intent_at = self.clock()

    def clear_intent(self) -> None:
        if self._drag_start is None:
            self._intent_at = 0.0

    def _has_pointer_intent(self) -> bool:
        return (
            self.cursor is not None
            and self.clock() - self._intent_at <= self.INTENT_TTL_S
        )

    @property
    def dragging(self) -> bool:
        return self._drag_start is not None

    def move_to(self, x: float, y: float) -> None:
        self.move_to_ref(*self._to_ref(x, y))

    def move_to_ref(self, x: float, y: float) -> None:
        self.cursor = (
            min(REF_WIDTH, max(0.0, float(x))),
            min(REF_HEIGHT, max(0.0, float(y))),
        )
        self._arm()

    def move_by(self, dx: float, dy: float) -> None:
        ref_dx, ref_dy = self._delta_to_ref(dx, dy)
        if self.dragging and self.cursor is not None:
            self.cursor = (
                min(REF_WIDTH, max(0.0, self.cursor[0] + ref_dx)),
                min(REF_HEIGHT, max(0.0, self.cursor[1] + ref_dy)),
            )
            self._arm()
            return
        self.clear_intent()
        self.input.move_camera_by(ref_dx, ref_dy)

    def click_at(self, x: float, y: float) -> None:
        self.move_to(x, y)
        assert self.cursor is not None
        self.input.click_ref(*self.cursor)

    def click_at_ref(self, x: float, y: float) -> None:
        self.move_to_ref(x, y)
        assert self.cursor is not None
        self.input.click_ref(*self.cursor)

    def left_click(self) -> None:
        if self._has_pointer_intent():
            assert self.cursor is not None
            self.input.click_ref(*self.cursor)
            self._arm()
            return
        self.input.attack()

    def left_down(self) -> None:
        if self._has_pointer_intent():
            assert self.cursor is not None
            self._drag_start = self.cursor
            self._drag_started_at = self.clock()
            self._arm()
            return
        self._attack_held = True
        self.input.attack_down()

    def left_up(self) -> None:
        if self.dragging:
            assert self._drag_start is not None and self.cursor is not None
            elapsed_ms = int((self.clock() - self._drag_started_at) * 1000)
            self.input.drag_ref(
                *self._drag_start, *self.cursor,
                duration_ms=max(100, elapsed_ms),
            )
            self._drag_start = None
            self._drag_started_at = 0.0
            self._arm()
            return
        if self._attack_held:
            self._attack_held = False
            self.input.attack_up()

    def release_all(self) -> None:
        if self._attack_held:
            self.input.attack_up()
        self._attack_held = False
        self._drag_start = None
        self._drag_started_at = 0.0
        self._intent_at = 0.0
