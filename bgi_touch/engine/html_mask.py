"""BetterGI HTML mask host backed by the local WebUI console."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def _data(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _message(url: str, data: Any, request_id: str | None = None) -> dict:
    result = {"url": str(url), "data": _data(data)}
    if request_id:
        result["requestId"] = request_id
    return result


@dataclass
class _PendingRequest:
    result: Any = None
    done: bool = False


@dataclass
class HtmlMaskWindow:
    id: str
    root: Path
    entry: str
    click_through: bool = False
    outgoing: deque[dict] = field(default_factory=deque)
    incoming: deque[dict] = field(default_factory=deque)
    pending: dict[str, _PendingRequest] = field(default_factory=dict)


class HtmlMaskManager:
    def __init__(self):
        self._condition = threading.Condition(threading.RLock())
        self._windows: dict[str, HtmlMaskWindow] = {}
        self._sequence = 0

    def _changed(self) -> None:
        self._sequence += 1
        self._condition.notify_all()

    def show(self, root: Path, url: str, window_id: str | None = None) -> str:
        if str(url).lower().startswith(("http://", "https://")):
            raise ValueError("iOS WebUI htmlMask 当前仅允许脚本目录内的本地 HTML")
        root = root.resolve()
        target = (root / str(url)).resolve()
        if not target.is_relative_to(root):
            raise PermissionError(f"HTML 遮罩路径越出脚本目录: {url}")
        if not target.is_file():
            raise FileNotFoundError(f"HTML 遮罩文件不存在: {url}")
        identifier = str(window_id or f"html-mask-{uuid.uuid4().hex}")
        with self._condition:
            if identifier in self._windows:
                self.close(identifier)
            self._windows[identifier] = HtmlMaskWindow(
                id=identifier,
                root=root,
                entry=str(target.relative_to(root)),
            )
            self._changed()
        return identifier

    def close(self, window_id: str) -> bool:
        with self._condition:
            window = self._windows.pop(str(window_id), None)
            if window is None:
                return False
            for pending in window.pending.values():
                pending.done = True
            self._changed()
            return True

    def close_many(self, window_ids: list[str]) -> None:
        for window_id in list(window_ids):
            self.close(window_id)

    def exists(self, window_id: str) -> bool:
        with self._condition:
            return str(window_id) in self._windows

    def window_ids(self) -> list[str]:
        with self._condition:
            return list(self._windows)

    def set_click_through(self, window_id: str, enabled: bool) -> None:
        with self._condition:
            window = self._require(window_id)
            if window.click_through != bool(enabled):
                window.click_through = bool(enabled)
                self._changed()

    def get_click_through(self, window_id: str) -> bool:
        with self._condition:
            return self._require(window_id).click_through

    def toggle_click_through(self, window_id: str) -> None:
        self.set_click_through(window_id, not self.get_click_through(window_id))

    def send(self, window_id: str, url: str, data: Any,
             request_id: str | None = None) -> None:
        with self._condition:
            self._require(window_id).outgoing.append(
                _message(url, data, request_id)
            )
            self._changed()

    def post_from_html(self, window_id: str, url: str, data: Any,
                       request_id: str | None = None) -> bool:
        with self._condition:
            window = self._require(window_id)
            if request_id and request_id in window.pending:
                pending = window.pending[request_id]
                pending.result = _data(data)
                pending.done = True
                self._condition.notify_all()
                return True
            window.incoming.append(_message(url, data, request_id))
            self._condition.notify_all()
            return True

    def request(self, window_id: str, url: str, data: Any,
                timeout_ms: int = 0,
                cancelled: Callable[[], bool] = lambda: False) -> str | None:
        window_id = str(window_id)
        request_id = uuid.uuid4().hex
        pending = _PendingRequest()
        with self._condition:
            window = self._require(window_id)
            window.pending[request_id] = pending
            window.outgoing.append(_message(url, data, request_id))
            self._changed()
            deadline = (
                time.monotonic() + int(timeout_ms) / 1000
                if int(timeout_ms) > 0 else None
            )
            try:
                while not pending.done and window_id in self._windows:
                    if cancelled():
                        return None
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        return None
                    self._condition.wait(
                        min(0.1, remaining) if remaining is not None else 0.1
                    )
                if window_id not in self._windows or not pending.done:
                    return None
                return json.dumps(pending.result, ensure_ascii=False, separators=(",", ":"))
            finally:
                window.pending.pop(request_id, None)

    def receive(self, window_id: str, timeout_ms: int = 0,
                cancelled: Callable[[], bool] = lambda: False) -> str | None:
        window_id = str(window_id)
        with self._condition:
            window = self._windows.get(str(window_id))
            if window is None:
                return None
            deadline = (
                time.monotonic() + int(timeout_ms) / 1000
                if int(timeout_ms) > 0 else None
            )
            while not window.incoming and window_id in self._windows:
                if cancelled():
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(
                    min(0.1, remaining) if remaining is not None else 0.1
                )
            if not window.incoming:
                return None
            return json.dumps(window.incoming.popleft(), ensure_ascii=False,
                              separators=(",", ":"))

    def poll(self, window_id: str) -> str | None:
        with self._condition:
            window = self._windows.get(str(window_id))
            if window is None or not window.incoming:
                return None
            return json.dumps(window.incoming.popleft(), ensure_ascii=False,
                              separators=(",", ":"))

    def poll_all(self, window_id: str) -> str:
        with self._condition:
            window = self._windows.get(str(window_id))
            values = list(window.incoming) if window else []
            if window:
                window.incoming.clear()
            return json.dumps(values, ensure_ascii=False, separators=(",", ":"))

    def events(self, after: int = 0, timeout_s: float = 20) -> dict:
        with self._condition:
            if int(after) == self._sequence:
                self._condition.wait_for(
                    lambda: int(after) != self._sequence,
                    timeout=max(0.0, float(timeout_s)),
                )
            windows = []
            for window in self._windows.values():
                messages = list(window.outgoing)
                window.outgoing.clear()
                windows.append({
                    "id": window.id,
                    "entry": window.entry,
                    "clickThrough": window.click_through,
                    "messages": messages,
                })
            return {"seq": self._sequence, "windows": windows}

    def resolve_asset(self, window_id: str, asset_path: str) -> Path:
        with self._condition:
            window = self._require(window_id)
            target = (window.root / str(asset_path)).resolve()
            if not target.is_relative_to(window.root):
                raise PermissionError("HTML 遮罩资源路径越界")
            if not target.is_file():
                raise FileNotFoundError(asset_path)
            return target

    def _require(self, window_id: str) -> HtmlMaskWindow:
        window = self._windows.get(str(window_id))
        if window is None:
            raise RuntimeError(f"HTML 遮罩窗口不存在或已关闭: {window_id}")
        return window


html_mask_manager = HtmlMaskManager()


class HtmlMaskHost:
    def __init__(self, root: str | Path,
                 manager: HtmlMaskManager = html_mask_manager,
                 cancelled: Callable[[], bool] = lambda: False):
        self.root = Path(root).resolve()
        self.manager = manager
        self.cancelled = cancelled
        self.opened: list[str] = []

    def show(self, url, window_id=None):
        identifier = self.manager.show(self.root, str(url), window_id)
        self.opened.append(identifier)
        return identifier

    def close(self, window_id):
        identifier = str(window_id)
        if identifier in self.opened:
            self.opened.remove(identifier)
        return self.manager.close(identifier)

    def closeAll(self):
        self.manager.close_many(self.opened)
        self.opened.clear()

    def getWindowIds(self): return self.manager.window_ids()
    def exists(self, window_id): return self.manager.exists(str(window_id))
    def setClickThrough(self, window_id, enabled):
        self.manager.set_click_through(str(window_id), bool(enabled))
    def getClickThrough(self, window_id):
        return self.manager.get_click_through(str(window_id))
    def toggleClickThrough(self, window_id):
        self.manager.toggle_click_through(str(window_id))
    def send(self, window_id, url, json_data="{}"):
        self.manager.send(str(window_id), str(url), json_data)
    def respond(self, window_id, request_id, json_data="{}"):
        self.manager.send(str(window_id), "/__response__", json_data, str(request_id))
    def request(self, window_id, url, json_data="{}", timeout_ms=0):
        return self.manager.request(
            str(window_id), str(url), json_data, int(timeout_ms), self.cancelled
        )
    def receive(self, window_id, timeout_ms=0):
        return self.manager.receive(
            str(window_id), int(timeout_ms), self.cancelled
        )
    def poll(self, window_id): return self.manager.poll(str(window_id))
    def pollAll(self, window_id): return self.manager.poll_all(str(window_id))
