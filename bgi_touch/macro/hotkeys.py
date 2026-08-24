"""BetterGI 通用热键宏的 iOS 输入适配。"""

from __future__ import annotations

import threading
from typing import Any, Callable


class RepeatingKeyMacro:
    """Turn a held WebUI/KeyMouseHook key into periodic touch key presses."""

    def __init__(
        self,
        input_simulator: Any,
        *,
        thresholds_ms: dict[str, int] | None = None,
        intervals_ms: dict[str, int] | None = None,
        log: Callable[[str], None] = print,
    ):
        self.input = input_simulator
        self.thresholds_ms = {
            "F": 200, "SPACE": 300, **(thresholds_ms or {}),
        }
        self.intervals_ms = {
            "F": 100, "SPACE": 100, **(intervals_ms or {}),
        }
        self.log = log
        self._lock = threading.RLock()
        self._workers: dict[str, tuple[threading.Event, threading.Thread]] = {}

    @staticmethod
    def _key(value: str) -> str:
        key = str(value).strip().upper()
        return "SPACE" if key in {"SPACEBAR", " "} else key

    def key_down(self, key: str) -> bool:
        canonical = self._key(key)
        if canonical not in self.thresholds_ms:
            return False
        with self._lock:
            current = self._workers.get(canonical)
            if current and current[1].is_alive():
                return False
            stop = threading.Event()
            worker = threading.Thread(
                target=self._repeat,
                args=(canonical, stop),
                daemon=True,
                name=f"repeat-{canonical.casefold()}",
            )
            self._workers[canonical] = (stop, worker)
            # WebUI key events are control input and do not otherwise reach
            # the game, so preserve the initial physical-key press as a tap.
            self.input.key_press(canonical)
            worker.start()
        return True

    def _repeat(self, key: str, stop: threading.Event) -> None:
        threshold = max(0, int(self.thresholds_ms[key])) / 1000
        interval = max(1, int(self.intervals_ms.get(key, 100))) / 1000
        if stop.wait(threshold):
            return
        self.log(f"[Macro] {key} 长按连发开始")
        while not stop.is_set():
            self.input.key_press(key)
            if stop.wait(interval):
                break
        self.log(f"[Macro] {key} 长按连发停止")

    def key_up(self, key: str) -> bool:
        canonical = self._key(key)
        with self._lock:
            current = self._workers.pop(canonical, None)
        if current is None:
            return False
        current[0].set()
        return True

    def stop(self, *, wait: bool = True, timeout: float = 1.0) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for event, _worker in workers:
            event.set()
        if wait:
            for _event, worker in workers:
                if worker is not threading.current_thread():
                    worker.join(timeout=max(0.0, timeout))

    KeyDown = key_down
    KeyUp = key_up
    Stop = stop


class HotkeyMacroHost:
    """Host object for BetterGI's small synchronous hotkey macros."""

    def __init__(
        self,
        ctx: Any,
        *,
        runaround_mouse_x_interval: float = 500,
        runaround_interval_ms: int = 10,
        enhance_wait_delay_ms: int = 0,
        f_fire_interval_ms: int = 100,
        space_fire_interval_ms: int = 100,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.runaround_mouse_x_interval = float(runaround_mouse_x_interval) or 1.0
        self.runaround_interval_ms = max(0, int(runaround_interval_ms))
        self.enhance_wait_delay_ms = max(0, int(enhance_wait_delay_ms))
        self.repeater = RepeatingKeyMacro(
            ctx.input,
            intervals_ms={
                "F": max(1, int(f_fire_interval_ms)),
                "SPACE": max(1, int(space_fire_interval_ms)),
            },
            log=log,
        )

    def turn_around(self) -> None:
        self.ctx.input.move_camera_by(self.runaround_mouse_x_interval, 0)
        if self.runaround_interval_ms:
            self.ctx.sleep(self.runaround_interval_ms)

    def quick_enhance_artifact(self) -> None:
        """Run the four clicks from BetterGI QuickEnhanceArtifactMacro."""
        self.ctx.input.click_ref(1760, 770)   # 快捷放入
        self.ctx.sleep(100)
        self.ctx.input.click_ref(1760, 1020)  # 强化
        self.ctx.sleep(100 + self.enhance_wait_delay_ms)
        self.ctx.input.click_ref(150, 150)    # 详情菜单
        self.ctx.sleep(100)
        self.ctx.input.click_ref(150, 220)    # 强化菜单
        self.ctx.sleep(100)

    def key_down(self, key: str) -> bool:
        return self.repeater.key_down(key)

    def key_up(self, key: str) -> bool:
        return self.repeater.key_up(key)

    def stop(self) -> None:
        self.repeater.stop()

    turnAround = turn_around
    TurnAround = turn_around
    quickEnhanceArtifact = quick_enhance_artifact
    QuickEnhanceArtifact = quick_enhance_artifact
    KeyDown = key_down
    KeyUp = key_up
    Stop = stop


__all__ = ["HotkeyMacroHost", "RepeatingKeyMacro"]
