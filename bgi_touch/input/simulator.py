"""键鼠 → 触控输入模拟。

可用 DeviceHub v2 profile 时，按住状态（WASD/冲刺/蓄力）由持久化 game
session 以 60Hz 在设备侧输出，并由本地线程按决策周期续租；旧服务器或 profile
不可用时回退到手势泵。手势泵中：
- 一个触点承载合成摇杆方向，每个按住的按钮各占一个固定触点，待发的视角增量
  再占一个拖动触点。W+Shift 这类组合因此落在同一个 HID 手势内。
- 点按类按键直接 tap 对应按钮坐标。
- 相机转动（原版 moveMouseBy）映射为屏幕右侧视角区域的滑动。
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Any

from ..device.client import DeviceClient
from ..vision.coordinate import ScreenTransform
from .layout import ControlLayout, normalize_key

MOVE_CYCLE_MS = 1400
JOYSTICK_OVERSHOOT = 2.2  # 终点超出摇杆半径，让大部分时长处于满偏移（跑步阈值以上）
# A native Wi-Fi screenshot on iPhone 13 Pro Max can hold the serialized MCP
# channel for 5-6 seconds. The lease must outlive that call so movement is not
# released while the next frame is being recognized. DeviceHub still provides
# an automatic safety release if the process disappears.
PROFILE_LEASE_MS = 15000
PROFILE_REFRESH_INTERVAL_S = 1.0


class InputSimulator:
    def __init__(self, device: DeviceClient, layout: ControlLayout, transform: ScreenTransform):
        self.device = device
        self.layout = layout
        self.t = transform
        self._held: dict[str, dict] = {}
        # Some native profile buttons (notably attack) intentionally have no
        # BetterGI keyboard binding. Keep their raw profile code alongside the
        # synthetic held-button entry so the lease heartbeat does not drop it.
        self._held_profile_overrides: dict[str, str] = {}
        self._held_lock = threading.Lock()
        self._pending_camera: list[float] | None = None
        self._pump: threading.Thread | None = None
        self._profile_session_id: str | None = None
        self._profile_failed = False
        self._profile_lock = threading.RLock()
        self._profile_heartbeat: threading.Thread | None = None
        # 移动端队伍列表只显示 3 个非当前角色的行，切人需按当前活跃槽位换算行号
        self._active_slot = 1
        self._event_lock = threading.Lock()
        self._event_listeners: dict[int, Callable[[dict[str, Any]], None]] = {}
        self._next_listener_id = 1

    # ---- helpers ----

    @property
    def _wh(self) -> dict:
        return {"image_width": self.t.device_width, "image_height": self.t.device_height}

    def _button_pos(self, name: str) -> tuple[float, float]:
        nx, ny = self.layout.buttons[name]
        return nx * self.t.device_width, ny * self.t.device_height

    def set_transform(self, transform: ScreenTransform) -> None:
        """截图分辨率变化（例如从状态缩略图切到原生帧）时热更新坐标。"""
        self.t = transform

    def _ensure_profile_session(self) -> bool:
        profile = self.layout.devicehub_profile
        if profile is None or self._profile_failed:
            return False
        with self._profile_lock:
            if self._profile_session_id is not None:
                device_session = getattr(
                    self.device, "game_session_id", self._profile_session_id
                )
                if device_session == self._profile_session_id:
                    return True
                # A direct DeviceHub HID action invalidates the exclusive game
                # session. DeviceClient clears its cache before sending it.
                self._profile_session_id = None
            try:
                self._profile_session_id = self.device.start_game_session(
                    profile.name,
                    lease_ms=PROFILE_LEASE_MS,
                    require_resolution_match=False,
                )
                self._profile_heartbeat = threading.Thread(
                    target=self._profile_heartbeat_loop,
                    args=(self._profile_session_id,),
                    daemon=True,
                    name="devicehub-game-lease",
                )
                self._profile_heartbeat.start()
                return True
            except Exception as e:
                self._profile_failed = True
                print(f"[input] DeviceHub game session 不可用，回退手势泵：{e}")
                return False

    def _profile_heartbeat_loop(self, session_id: str) -> None:
        """Refresh the game lease at the MCP decision rate while the session is live."""
        while True:
            time.sleep(PROFILE_REFRESH_INTERVAL_S)
            failed = False
            with self._profile_lock:
                if self._profile_session_id != session_id or self._profile_failed:
                    return
                device_session = getattr(self.device, "game_session_id", session_id)
                if device_session != session_id:
                    self._profile_session_id = None
                    with self._held_lock:
                        has_held_input = bool(self._held)
                    if has_held_input:
                        self._ensure_pump()
                    return
                try:
                    self.device.set_game_input(
                        session_id,
                        self._held_profile_keys(),
                        lease_ms=PROFILE_LEASE_MS,
                    )
                except Exception as e:
                    recoverable = self._drop_profile_session(session_id, e)
                    failed = True
                    suffix = "；下次输入自动重建" if recoverable else ""
                    print(f"[input] DeviceHub game session 租约刷新失败，回退触控{suffix}：{e}")
            if failed:
                self._ensure_pump()
                return

    def _profile_raw_keys(self, canonical_keys: list[str]) -> list[str]:
        keys: list[str] = []
        for key in canonical_keys:
            raw = self._held_profile_overrides.get(key) or self.layout.profile_key(key)
            if raw is not None and raw not in keys:
                keys.append(raw)
        return keys

    def _held_profile_keys(self) -> list[str]:
        with self._held_lock:
            return self._profile_raw_keys(list(self._held))

    def _drop_profile_session(self, session_id: str, error: Exception) -> bool:
        """Drop a failed session and report whether it may be rebuilt later."""
        expired = "session not found" in str(error).casefold()
        self._profile_failed = not expired
        self._profile_session_id = None
        try:
            # DeviceClient clears its cached ID in finally even when the remote
            # session has already expired.
            self.device.stop_game_session(session_id)
        except Exception:
            pass
        return expired

    def _sync_profile_keys(self, keys: list[str] | None = None) -> bool:
        if not self._ensure_profile_session():
            return False
        with self._profile_lock:
            session_id = self._profile_session_id
            if session_id is None:
                return False
            try:
                self.device.set_game_input(
                    session_id,
                    keys if keys is not None else self._held_profile_keys(),
                    lease_ms=PROFILE_LEASE_MS,
                )
                return True
            except Exception as e:
                print(f"[input] DeviceHub game session 输入失败，回退手势泵：{e}")
                self._drop_profile_session(session_id, e)
                return False

    def _profile_press_raw(self, raw_key: str, hold_ms: int = 80) -> bool:
        if not self._ensure_profile_session():
            return False
        with self._profile_lock:
            session_id = self._profile_session_id
            if session_id is None:
                return False
            try:
                held = self._held_profile_keys()
                if raw_key not in held:
                    held.append(raw_key)
                self.device.set_game_input(session_id, held, lease_ms=PROFILE_LEASE_MS)
                time.sleep(max(25, hold_ms) / 1000)
                self.device.set_game_input(session_id, self._held_profile_keys(),
                                           lease_ms=PROFILE_LEASE_MS)
                return True
            except Exception as e:
                print(f"[input] DeviceHub profile 按键失败，回退触控：{e}")
                self._drop_profile_session(session_id, e)
                return False

    def _profile_press(self, key: str, hold_ms: int = 80) -> bool:
        raw = self.layout.profile_key(key)
        return self._profile_press_raw(raw, hold_ms) if raw is not None else False

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

    def _after_direct_input(self) -> None:
        """Reconcile a DeviceHub session invalidated by direct touch input."""
        if not hasattr(self.device, "game_session_id"):
            return
        with self._profile_lock:
            device_session = self.device.game_session_id
            if self._profile_session_id == device_session:
                return
            self._profile_session_id = device_session
        with self._held_lock:
            has_held_input = bool(self._held)
        if has_held_input:
            self._sync_profile_keys()

    def _direct_input(self, action) -> None:
        try:
            action()
        finally:
            self._after_direct_input()

    def subscribe(self, listener: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """Subscribe to completed BetterGI input edges.

        Listeners are deliberately local to the simulator: they observe the
        same semantic actions scripts send to DeviceHub without polling the
        keyboard or requesting another screenshot.  A broken listener must
        never interrupt game input.
        """
        if not callable(listener):
            raise TypeError("input listener 必须可调用")
        with self._event_lock:
            listener_id = self._next_listener_id
            self._next_listener_id += 1
            self._event_listeners[listener_id] = listener

        def unsubscribe() -> None:
            with self._event_lock:
                self._event_listeners.pop(listener_id, None)

        return unsubscribe

    def _emit(self, event_type: str, **values: Any) -> None:
        lock = getattr(self, "_event_lock", None)
        if lock is None:
            return
        event = {"type": event_type, "timestamp": time.monotonic(), **values}
        with lock:
            listeners = list(self._event_listeners.values())
        for listener in listeners:
            try:
                listener(dict(event))
            except Exception as error:
                print(f"[input] 事件监听器失败（已忽略）：{error}")

    # ---- key API（BetterGI 语义）----

    def switch_party_slot(self, slot: int) -> None:
        """切换到队伍槽位 1-4（PC 数字键语义）。"""
        if slot == self._active_slot:
            return
        from_slot = self._active_slot
        others = [s for s in (1, 2, 3, 4) if s != from_slot]
        if slot not in others:
            return
        if self._profile_press(str(slot)):
            self._active_slot = slot
            self._emit("party_switch", from_slot=from_slot, to_slot=slot)
            return
        row = others.index(slot) + 1
        x, y = self._button_pos(f"partyRow{row}")
        self._direct_input(lambda: self.device.tap(x, y, **self._wh))
        self._active_slot = slot
        self._emit("party_switch", from_slot=from_slot, to_slot=slot)

    def key_press(self, key: str, hold_ms: int = 80) -> None:
        canonical = normalize_key(key)
        b = self.layout.binding(key)
        if b is None:
            if self._profile_press(key, hold_ms):
                self._emit("key_press", key=canonical)
                return
            return  # 未映射按键在触控端为空操作
        if b["type"] == "party":
            self.switch_party_slot(int(b["slot"]))
        elif self._profile_press(key, hold_ms):
            self._emit("key_press", key=canonical)
            return
        elif b["type"] == "button":
            x, y = self._button_pos(b["button"])
            self._direct_input(
                lambda: self.device.tap(x, y, hold_ms=hold_ms, **self._wh)
            )
            self._emit("key_press", key=canonical)
        else:
            contact = self._joystick_contact([b])
            if contact:
                self._direct_input(
                    lambda: self.device.multi_touch(
                        [contact], duration_ms=max(150, hold_ms), **self._wh
                    )
                )
                self._emit("key_press", key=canonical)

    def key_down(self, key: str) -> None:
        canonical = normalize_key(key)
        b = self.layout.binding(key)
        if b is None:
            return
        if b["type"] == "party":
            self.switch_party_slot(int(b["slot"]))
            return
        with self._held_lock:
            was_held = canonical in self._held
            self._held[canonical] = b
        if not self._sync_profile_keys():
            self._ensure_pump()
        if not was_held:
            self._emit("key_down", key=canonical)

    def key_up(self, key: str) -> None:
        with self._held_lock:
            self._held.pop(normalize_key(key), None)
        if self._profile_session_id is not None:
            self._sync_profile_keys()

    def release_all(self) -> None:
        with self._held_lock:
            self._held.clear()
            self._held_profile_overrides.clear()
            self._pending_camera = None
        session_id = self._profile_session_id
        if session_id is not None:
            with self._profile_lock:
                try:
                    self.device.set_game_input(session_id, [], lease_ms=PROFILE_LEASE_MS)
                except Exception:
                    pass
                try:
                    self.device.stop_game_session(session_id)
                except Exception:
                    pass
                self._profile_session_id = None

    # ---- mouse API（BetterGI 语义）----

    def click_ref(self, ref_x: float, ref_y: float) -> None:
        x, y = self.t.to_device(ref_x, ref_y)
        self._direct_input(lambda: self.device.tap(x, y, **self._wh))

    def attack(self, hold_ms: int = 80) -> None:
        if self._profile_press("X", hold_ms):
            return
        x, y = self._button_pos("attack")
        self._direct_input(
            lambda: self.device.tap(x, y, hold_ms=hold_ms, **self._wh)
        )

    def button_down(self, name: str) -> None:
        """Hold a semantic HUD button until :meth:`button_up` is called."""
        key = f"__button__:{name}"
        with self._held_lock:
            self._held[key] = {"type": "button", "button": name}
            raw = self.layout.profile_key_for_button(name)
            if raw is not None:
                self._held_profile_overrides[key] = raw
        if not self._sync_profile_keys():
            self._ensure_pump()

    def button_up(self, name: str) -> None:
        key = f"__button__:{name}"
        with self._held_lock:
            self._held.pop(key, None)
            self._held_profile_overrides.pop(key, None)
        if self._profile_session_id is not None:
            self._sync_profile_keys()

    def attack_down(self) -> None:
        self.button_down("attack")

    def attack_up(self) -> None:
        self.button_up("attack")

    def charged_attack(self, hold_ms: int = 800) -> None:
        if self._profile_press("X", hold_ms):
            return
        x, y = self._button_pos("attack")
        self._direct_input(
            lambda: self.device.multi_touch(
                [{"x1": x, "y1": y, "x2": x, "y2": y}],
                duration_ms=hold_ms,
                **self._wh,
            )
        )

    def move_camera_by(self, dx: float, dy: float) -> None:
        """相机转动。泵激活时并入下一个手势周期，否则立即滑动。

        大幅转角拆成多段，保证每段起止点都在安全视角区内——区外起点会被
        队伍头像等 UI 吃掉手势（实测教训：向左滑起点落在头像列上即失效）。
        """
        with self._held_lock:
            if self._held and self._profile_session_id is None:
                p = self._pending_camera or [0.0, 0.0]
                self._pending_camera = [p[0] + dx, p[1] + dy]
                return
        nx, ny, nw, nh = self.layout.camera_region
        max_dx = 0.9 * nw * self.t.device_width / self.t.scale   # ref 像素语义
        max_dy = 0.9 * nh * self.t.device_height / self.t.scale
        def swipe_all() -> None:
            nonlocal dx, dy
            while abs(dx) > 1 or abs(dy) > 1:
                sx = max(-max_dx, min(max_dx, dx))
                sy = max(-max_dy, min(max_dy, dy))
                c = self._camera_contact(sx, sy)
                dist = math.hypot(c["x2"] - c["x1"], c["y2"] - c["y1"])
                self.device.swipe(
                    c["x1"], c["y1"], c["x2"], c["y2"],
                    duration_ms=int(min(900, max(150, dist * 0.8))),
                    **self._wh,
                )
                dx -= sx
                dy -= sy
                if abs(dx) > 1 or abs(dy) > 1:
                    time.sleep(0.35)

        self._direct_input(swipe_all)

    def tap_button(self, name: str, hold_ms: int = 80) -> None:
        raw = self.layout.profile_key_for_button(name)
        if raw is not None and self._profile_press_raw(raw, hold_ms):
            return
        x, y = self._button_pos(name)
        self._direct_input(
            lambda: self.device.tap(x, y, hold_ms=hold_ms, **self._wh)
        )

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
