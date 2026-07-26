"""键鼠 → 触控输入模拟。

devicehub-mask 的触控手势是原子的且最长 5 秒，无法无限期按住。因此：
- 按住状态（WASD/冲刺/蓄力）由后台"手势泵"线程维持：只要有按住的键，
  就连续下发 multi_touch 手势——一个触点承载合成摇杆方向，每个按住的
  按钮各占一个固定触点，待发的视角增量再占一个拖动触点。W+Shift 这类
  组合因此落在同一个 HID 手势内，游戏视为同时输入。
- 点按类按键直接 tap 对应按钮坐标。
- 相机转动（原版 moveMouseBy）映射为屏幕右侧视角区域的滑动。
"""

from __future__ import annotations

import math
import threading
import time

from ..device.client import DeviceClient
from ..vision.coordinate import ScreenTransform
from .layout import ControlLayout, normalize_key

MOVE_CYCLE_MS = 1400
JOYSTICK_OVERSHOOT = 2.2  # 终点超出摇杆半径，让大部分时长处于满偏移（跑步阈值以上）


class InputSimulator:
    def __init__(self, device: DeviceClient, layout: ControlLayout, transform: ScreenTransform):
        self.device = device
        self.layout = layout
        self.t = transform
        self._held: dict[str, dict] = {}
        self._held_lock = threading.Lock()
        self._pending_camera: list[float] | None = None
        self._pump: threading.Thread | None = None

    # ---- helpers ----

    @property
    def _wh(self) -> dict:
        return {"image_width": self.t.device_width, "image_height": self.t.device_height}

    def _button_pos(self, name: str) -> tuple[float, float]:
        nx, ny = self.layout.buttons[name]
        return nx * self.t.device_width, ny * self.t.device_height

    def _joystick_contact(self, bindings: list[dict]) -> dict | None:
        vx = vy = 0.0
        for b in bindings:
            if b.get("type") != "joystick":
                continue
            rad = math.radians(b["angleDeg"])
            vx += math.cos(rad)
            vy += math.sin(rad)
        mag = math.hypot(vx, vy)
        cx = self.layout.joystick_center[0] * self.t.device_width
        cy = self.layout.joystick_center[1] * self.t.device_height
        if mag < 1e-6:
            return None
        r = self.layout.joystick_radius_n * self.t.device_width * JOYSTICK_OVERSHOOT
        return {"x1": cx, "y1": cy, "x2": cx + vx / mag * r, "y2": cy + vy / mag * r}

    def _camera_contact(self, dx: float, dy: float) -> dict:
        nx, ny, nw, nh = self.layout.camera_region
        cx = (nx + nw / 2) * self.t.device_width
        cy = (ny + nh / 2) * self.t.device_height
        ddx, ddy = dx * self.t.scale, dy * self.t.scale
        return {"x1": cx - ddx / 2, "y1": cy - ddy / 2, "x2": cx + ddx / 2, "y2": cy + ddy / 2}

    # ---- key API（BetterGI 语义）----

    def key_press(self, key: str, hold_ms: int = 80) -> None:
        b = self.layout.binding(key)
        if b is None:
            return  # 未映射按键在触控端为空操作
        if b["type"] == "button":
            x, y = self._button_pos(b["button"])
            self.device.tap(x, y, hold_ms=hold_ms, **self._wh)
        else:
            contact = self._joystick_contact([b])
            if contact:
                self.device.multi_touch([contact], duration_ms=max(150, hold_ms), **self._wh)

    def key_down(self, key: str) -> None:
        b = self.layout.binding(key)
        if b is None:
            return
        with self._held_lock:
            self._held[normalize_key(key)] = b
        self._ensure_pump()

    def key_up(self, key: str) -> None:
        with self._held_lock:
            self._held.pop(normalize_key(key), None)

    def release_all(self) -> None:
        with self._held_lock:
            self._held.clear()
            self._pending_camera = None

    # ---- mouse API（BetterGI 语义）----

    def click_ref(self, ref_x: float, ref_y: float) -> None:
        x, y = self.t.to_device(ref_x, ref_y)
        self.device.tap(x, y, **self._wh)

    def attack(self, hold_ms: int = 80) -> None:
        x, y = self._button_pos("attack")
        self.device.tap(x, y, hold_ms=hold_ms, **self._wh)

    def charged_attack(self, hold_ms: int = 800) -> None:
        x, y = self._button_pos("attack")
        self.device.multi_touch([{"x1": x, "y1": y, "x2": x, "y2": y}], duration_ms=hold_ms, **self._wh)

    def move_camera_by(self, dx: float, dy: float) -> None:
        """相机转动。泵激活时并入下一个手势周期，否则立即滑动。"""
        with self._held_lock:
            if self._held:
                p = self._pending_camera or [0.0, 0.0]
                self._pending_camera = [p[0] + dx, p[1] + dy]
                return
        c = self._camera_contact(dx, dy)
        dist = math.hypot(c["x2"] - c["x1"], c["y2"] - c["y1"])
        self.device.swipe(c["x1"], c["y1"], c["x2"], c["y2"],
                          duration_ms=int(min(1000, max(120, dist * 0.8))), **self._wh)

    def tap_button(self, name: str, hold_ms: int = 80) -> None:
        x, y = self._button_pos(name)
        self.device.tap(x, y, hold_ms=hold_ms, **self._wh)

    # ---- 手势泵 ----

    def _ensure_pump(self) -> None:
        if self._pump and self._pump.is_alive():
            return
        self._pump = threading.Thread(target=self._pump_loop, daemon=True, name="input-pump")
        self._pump.start()

    def _pump_loop(self) -> None:
        while True:
            with self._held_lock:
                if not self._held:
                    return
                bindings = list(self._held.values())
                camera = self._pending_camera
                self._pending_camera = None
            contacts: list[dict] = []
            joy = self._joystick_contact(bindings)
            if joy:
                contacts.append(joy)
            for b in bindings:
                if b.get("type") == "button":
                    x, y = self._button_pos(b["button"])
                    contacts.append({"x1": x, "y1": y, "x2": x, "y2": y})
            if camera:
                contacts.append(self._camera_contact(*camera))
            if not contacts:
                return
            try:
                self.device.multi_touch(contacts[:5], duration_ms=MOVE_CYCLE_MS, **self._wh)
            except Exception:
                time.sleep(0.2)  # 设备暂时不可用时避免忙转
