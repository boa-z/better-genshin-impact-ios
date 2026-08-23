"""BetterGI ``KeyMouseHook`` compatibility for the WebUI control surface.

Windows BetterGI listens to global desktop keyboard and mouse events.  On iOS
there is no game-window mouse, so the local WebUI preview is the equivalent
interactive surface.  FastAPI threads only enqueue normalized events here;
JavaScript callbacks are drained by the script thread at ``sleep`` checkpoints
to keep PythonMonkey access single-threaded.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable


_KEY_CODES = {
    "AltLeft": "LMenu", "AltRight": "RMenu",
    "ControlLeft": "LControlKey", "ControlRight": "RControlKey",
    "ShiftLeft": "LShiftKey", "ShiftRight": "RShiftKey",
    "MetaLeft": "LWin", "MetaRight": "RWin",
    "ArrowUp": "Up", "ArrowDown": "Down",
    "ArrowLeft": "Left", "ArrowRight": "Right",
    "Backquote": "Oem3", "Minus": "OemMinus", "Equal": "Oemplus",
    "BracketLeft": "OemOpenBrackets", "BracketRight": "Oem6",
    "Backslash": "Oem5", "Semicolon": "Oem1", "Quote": "Oem7",
    "Comma": "Oemcomma", "Period": "OemPeriod", "Slash": "OemQuestion",
}

_MOUSE_BUTTONS = {0: "Left", 1: "Middle", 2: "Right", 3: "XButton1", 4: "XButton2"}


def _key_code(event: dict[str, Any]) -> str:
    code = str(event.get("code") or "")
    if code in _KEY_CODES:
        return _KEY_CODES[code]
    if code.startswith("Key") and len(code) == 4:
        return code[-1].upper()
    if code.startswith("Digit") and len(code) == 6:
        return f"D{code[-1]}"
    if code.startswith("Numpad") and code[6:].isdigit():
        return f"NumPad{code[6:]}"
    if code:
        return code
    key = str(event.get("key") or "")
    return key.upper() if len(key) == 1 else key


def _key_data(event: dict[str, Any]) -> str:
    values = [_key_code(event)]
    for field, name in (("shiftKey", "Shift"), ("ctrlKey", "Control"),
                        ("altKey", "Alt")):
        if event.get(field) and name not in values:
            values.append(name)
    return ", ".join(value for value in values if value)


class KeyMouseHookManager:
    """Per-runtime hook registry and bounded cross-thread event queue."""

    def __init__(self, log: Callable[[str], None] = print):
        self.log = log
        self._lock = threading.RLock()
        self._hooks: list[KeyMouseHookHost] = []
        self._events: deque[dict[str, Any]] = deque(maxlen=2048)
        self._draining = False

    def create(self) -> "KeyMouseHookHost":
        hook = KeyMouseHookHost(self)
        with self._lock:
            self._hooks.append(hook)
        return hook

    def unregister(self, hook: "KeyMouseHookHost") -> None:
        with self._lock:
            if hook in self._hooks:
                self._hooks.remove(hook)

    def has_hooks(self) -> bool:
        with self._lock:
            return any(not hook.disposed for hook in self._hooks)

    def enqueue(self, event: dict[str, Any]) -> bool:
        kind = str(event.get("type") or "")
        if kind not in {
            "keyDown", "keyUp", "mouseDown", "mouseUp", "mouseMove", "mouseWheel",
        }:
            raise ValueError(f"不支持的 KeyMouseHook 事件: {kind}")
        with self._lock:
            if not any(not hook.disposed for hook in self._hooks):
                return False
            value = dict(event)
            value["timestamp"] = time.monotonic()
            self._events.append(value)
            return True

    def drain(self) -> int:
        """Invoke callbacks on the caller (JavaScript runtime) thread."""
        with self._lock:
            if self._draining:
                return 0
            self._draining = True
            events = list(self._events)
            self._events.clear()
        try:
            count = 0
            for event in events:
                with self._lock:
                    hooks = list(self._hooks)
                for hook in hooks:
                    count += hook.dispatch(event)
            return count
        finally:
            with self._lock:
                self._draining = False

    def close_all(self) -> None:
        with self._lock:
            hooks = list(self._hooks)
            self._events.clear()
        for hook in hooks:
            hook.dispose()


class KeyMouseHookHost:
    def __init__(self, manager: KeyMouseHookManager):
        self.manager = manager
        self.disposed = False
        self._callbacks: dict[str, list[tuple[Callable[..., Any], int]]] = {
            name: [] for name in (
                "keyDownCode", "keyDownData", "keyUpCode", "keyUpData",
                "mouseDown", "mouseUp", "mouseMove", "mouseWheel",
            )
        }
        self._last_mouse_move: dict[int, float] = {}

    def _add(self, name: str, callback: Callable[..., Any], interval: int = 0) -> None:
        if self.disposed:
            raise RuntimeError("KeyMouseHook 已释放")
        if not callable(callback):
            raise TypeError("KeyMouseHook 回调必须是函数")
        self._callbacks[name].append((callback, max(0, int(interval))))

    def OnKeyDown(self, callback, useCodeOnly=True):
        self._add("keyDownCode" if bool(useCodeOnly) else "keyDownData", callback)

    def OnKeyUp(self, callback, useCodeOnly=True):
        self._add("keyUpCode" if bool(useCodeOnly) else "keyUpData", callback)

    def OnMouseDown(self, callback): self._add("mouseDown", callback)
    def OnMouseUp(self, callback): self._add("mouseUp", callback)
    def OnMouseMove(self, callback, interval=200):
        self._add("mouseMove", callback, int(interval))
    def OnMouseWheel(self, callback): self._add("mouseWheel", callback)

    def _invoke(self, name: str, args: tuple[Any, ...], event: dict[str, Any]) -> int:
        invoked = 0
        for callback, interval in list(self._callbacks[name]):
            if name == "mouseMove" and interval:
                callback_id = id(callback)
                last = self._last_mouse_move.get(callback_id, float("-inf"))
                if (float(event["timestamp"]) - last) * 1000 < interval:
                    continue
                self._last_mouse_move[callback_id] = float(event["timestamp"])
            try:
                callback(*args)
                invoked += 1
            except Exception as error:  # Match upstream: callback failure releases this hook.
                self.manager.log(f"[KeyMouseHook] 回调失败，已释放监听: {error}")
                self.dispose()
                break
        return invoked

    def dispatch(self, event: dict[str, Any]) -> int:
        if self.disposed:
            return 0
        kind = str(event["type"])
        if kind in {"keyDown", "keyUp"}:
            prefix = "keyDown" if kind == "keyDown" else "keyUp"
            return (
                self._invoke(prefix + "Code", (_key_code(event),), event)
                + self._invoke(prefix + "Data", (_key_data(event),), event)
            )
        x, y = int(round(float(event.get("x", -1)))), int(round(float(event.get("y", -1))))
        if kind in {"mouseDown", "mouseUp"}:
            button = event.get("button", 0)
            try:
                button_name = _MOUSE_BUTTONS.get(int(button), str(button))
            except (TypeError, ValueError):
                button_name = str(button)
            return self._invoke(kind, (button_name, x, y), event)
        if kind == "mouseMove":
            return self._invoke(kind, (x, y), event)
        return self._invoke(kind, (int(round(float(event.get("delta", 0)))), x, y), event)

    def RemoveAllListeners(self):
        for callbacks in self._callbacks.values():
            callbacks.clear()
        self._last_mouse_move.clear()

    def Dispose(self): self.dispose()

    def dispose(self):
        if self.disposed:
            return
        self.disposed = True
        self.RemoveAllListeners()
        self.manager.unregister(self)
