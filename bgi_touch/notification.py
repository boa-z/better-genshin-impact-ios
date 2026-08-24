"""BetterGI-compatible notifications with a non-blocking Gotify backend."""

from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "notification.json"


def _first(raw: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in raw:
            return raw[name]
    return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _event_codes(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        values = ()
    return frozenset(str(item).strip().casefold() for item in values if str(item).strip())


@dataclass(frozen=True)
class NotificationConfig:
    js_notification_enabled: bool = False
    subscribed_events: frozenset[str] = frozenset()
    gotify_enabled: bool = False
    gotify_url: str = ""
    gotify_app_token: str = ""
    gotify_priority: int = 3
    timeout_s: float = 10.0
    queue_size: int = 128

    @classmethod
    def load(cls, path: str | Path | None = None) -> "NotificationConfig":
        configured = path or os.environ.get("BGI_NOTIFICATION_CONFIG")
        source = Path(configured).expanduser() if configured else DEFAULT_CONFIG
        source = source.resolve()
        if not source.is_file():
            return cls(
                gotify_url=os.environ.get("BGI_GOTIFY_URL", ""),
                gotify_app_token=os.environ.get("BGI_GOTIFY_TOKEN", ""),
            )
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"通知配置 JSON 无效：{source}: {error}") from error
        if not isinstance(raw, Mapping):
            raise ValueError(f"通知配置根节点必须是对象：{source}")

        nested = _first(raw, "gotify", "Gotify", default={})
        if not isinstance(nested, Mapping):
            nested = {}
        url = os.environ.get("BGI_GOTIFY_URL") or str(_first(
            nested, "url", default=_first(raw, "gotifyUrl", "gotify_url", default="")
        ) or "")
        token = os.environ.get("BGI_GOTIFY_TOKEN") or str(_first(
            nested, "appToken", "app_token", "token",
            default=_first(raw, "gotifyAppToken", "gotify_app_token", default=""),
        ) or "")
        enabled = _as_bool(_first(
            nested, "enabled",
            default=_first(raw, "gotifyNotificationEnabled", "gotify_enabled", default=False),
        ))
        try:
            priority = max(0, int(_first(
                nested, "priority", "notifyLevel",
                default=_first(raw, "gotifyNotifyLevel", "gotify_priority", default=3),
            )))
        except (TypeError, ValueError):
            priority = 3
        try:
            timeout = max(1.0, min(60.0, float(_first(
                nested, "timeoutSeconds", "timeout_seconds",
                default=_first(raw, "timeoutSeconds", "timeout_seconds", default=10),
            ))))
        except (TypeError, ValueError):
            timeout = 10.0
        try:
            queue_size = max(1, min(4096, int(_first(
                raw, "queueSize", "queue_size", default=128,
            ))))
        except (TypeError, ValueError):
            queue_size = 128
        return cls(
            js_notification_enabled=_as_bool(_first(
                raw, "jsNotificationEnabled", "js_notification_enabled", default=False,
            )),
            subscribed_events=_event_codes(_first(
                raw, "notificationEventSubscribe", "subscribedEvents", "subscribed_events",
                default="",
            )),
            gotify_enabled=enabled,
            gotify_url=url.strip(),
            gotify_app_token=token.strip(),
            gotify_priority=priority,
            timeout_s=timeout,
            queue_size=queue_size,
        )


@dataclass(frozen=True)
class Notification:
    event: str
    message: str
    result: str = "Success"
    timestamp: datetime | None = None


class NotificationError(RuntimeError):
    pass


class GotifyNotifier:
    def __init__(
        self,
        url: str,
        app_token: str,
        priority: int = 3,
        timeout_s: float = 10.0,
        *,
        opener: Callable[..., Any] = urlopen,
    ):
        self.url = str(url).rstrip("/")
        self.app_token = str(app_token).strip()
        self.priority = max(0, int(priority))
        self.timeout_s = max(1.0, float(timeout_s))
        self._opener = opener

    def _endpoint(self) -> str:
        parsed = urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise NotificationError("Gotify 服务地址无效")
        if not self.app_token:
            raise NotificationError("Gotify App Token 为空")
        return f"{self.url}/message"

    def send(self, notification: Notification) -> None:
        timestamp = notification.timestamp or datetime.now()
        message = f"时间: {timestamp:%Y-%m-%d %H:%M:%S}"
        if notification.message.strip():
            message += f"\n\n消息: {notification.message}"
        payload = json.dumps({
            "title": "BetterGI·更好的原神",
            "message": message,
            "priority": self.priority,
        }, ensure_ascii=False).encode("utf-8")
        request = Request(
            self._endpoint(),
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "X-Gotify-Key": self.app_token,
            },
        )
        try:
            with self._opener(request, timeout=self.timeout_s) as response:
                status = getattr(response, "status", 200)
                if not 200 <= int(status) < 300:
                    raise NotificationError(f"Gotify 调用失败，状态码: {status}")
        except HTTPError as error:
            raise NotificationError(f"Gotify 调用失败，状态码: {error.code}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise NotificationError(f"Gotify 通知发送失败：{error}") from error


class NotificationService:
    """Dispatch notifications away from automation and screenshot threads."""

    _STOP = object()

    def __init__(
        self,
        config: NotificationConfig,
        *,
        notifier: GotifyNotifier | None = None,
        log: Callable[[str], None] = print,
    ):
        self.config = config
        self.log = log
        self.notifier = notifier or (
            GotifyNotifier(
                config.gotify_url,
                config.gotify_app_token,
                config.gotify_priority,
                config.timeout_s,
            ) if config.gotify_enabled else None
        )
        self._queue: queue.Queue[Notification | object] = queue.Queue(config.queue_size)
        self._thread: threading.Thread | None = None
        self._closed = False

    @classmethod
    def load(cls, path: str | Path | None = None,
             *, log: Callable[[str], None] = print) -> "NotificationService":
        return cls(NotificationConfig.load(path), log=log)

    def _allowed(self, event: str) -> bool:
        subscriptions = self.config.subscribed_events
        return not subscriptions or str(event).strip().casefold() in subscriptions

    def notify(self, event: str, message: str, *, result: str = "Success",
               from_js: bool = False) -> bool:
        if self._closed or self.notifier is None:
            return False
        if from_js and not self.config.js_notification_enabled:
            return False
        if not self._allowed(event):
            return False
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._worker,
                daemon=True,
                name="bgi-notification",
            )
            self._thread.start()
        try:
            self._queue.put_nowait(Notification(str(event), str(message), str(result)))
        except queue.Full:
            self.log("[通知] 推送队列已满，已丢弃本条消息")
            return False
        return True

    def notify_now(self, event: str, message: str, *, result: str = "Success",
                   from_js: bool = False) -> bool:
        """Send synchronously for configuration tests and short-lived tools."""
        if self._closed or self.notifier is None:
            return False
        if from_js and not self.config.js_notification_enabled:
            return False
        if not self._allowed(event):
            return False
        self.notifier.send(Notification(str(event), str(message), str(result)))
        return True

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                assert isinstance(item, Notification)
                try:
                    assert self.notifier is not None
                    self.notifier.send(item)
                except Exception as error:  # Notification failures must not abort tasks.
                    self.log(f"[通知] Gotify 发送失败：{error}")
            finally:
                self._queue.task_done()

    def close(self, timeout_s: float | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        thread = self._thread
        if thread is None:
            return
        try:
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            # The worker will free one slot shortly; use a bounded wait so a
            # broken notification endpoint cannot stall task cleanup forever.
            try:
                self._queue.put(self._STOP, timeout=min(1.0, self.config.timeout_s))
            except queue.Full:
                return
        thread.join(timeout=timeout_s if timeout_s is not None else self.config.timeout_s + 1)
