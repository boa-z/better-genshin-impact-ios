import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_notification_config_accepts_bettergi_flat_fields(tmp_path):
    from bgi_touch.notification import NotificationConfig

    path = tmp_path / "notification.json"
    path.write_text(json.dumps({
        "jsNotificationEnabled": True,
        "notificationEventSubscribe": "JsNotification, TaskError,jsnotification",
        "gotifyNotificationEnabled": True,
        "gotifyUrl": "https://gotify.example/",
        "gotifyAppToken": "secret",
        "gotifyNotifyLevel": 7,
    }), encoding="utf-8")

    config = NotificationConfig.load(path)

    assert config.js_notification_enabled is True
    assert config.subscribed_events == frozenset({"jsnotification", "taskerror"})
    assert config.gotify_enabled is True
    assert config.gotify_url == "https://gotify.example/"
    assert config.gotify_app_token == "secret"
    assert config.gotify_priority == 7


def test_gotify_notifier_posts_bettergi_message_and_auth_header():
    from bgi_touch.notification import GotifyNotifier, Notification

    calls = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def open_request(request, *, timeout):
        calls.append((request, timeout))
        return Response()

    notifier = GotifyNotifier(
        "https://gotify.example/", "app-token", priority=5,
        timeout_s=4, opener=open_request,
    )
    notifier.send(Notification(
        "TaskError", "任务失败", "Fail", datetime(2026, 8, 24, 12, 34, 56),
    ))

    request, timeout = calls[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "https://gotify.example/message"
    assert request.get_header("X-gotify-key") == "app-token"
    assert timeout == 4
    assert payload == {
        "title": "BetterGI·更好的原神",
        "message": "时间: 2026-08-24 12:34:56\n\n消息: 任务失败",
        "priority": 5,
    }


def test_notification_service_filters_events_and_keeps_js_non_blocking():
    from bgi_touch.notification import NotificationConfig, NotificationService

    sent = []

    class Notifier:
        def send(self, item):
            sent.append(item)

    config = NotificationConfig(
        js_notification_enabled=True,
        subscribed_events=frozenset({"jsnotification"}),
        gotify_enabled=True,
    )
    service = NotificationService(config, notifier=Notifier())

    assert not service.notify("TaskError", "filtered")
    assert service.notify("JsNotification", "脚本完成", from_js=True)
    service.close()

    assert [(item.event, item.message, item.result) for item in sent] == [
        ("JsNotification", "脚本完成", "Success"),
    ]


def test_js_notifications_require_explicit_permission():
    from bgi_touch.notification import NotificationConfig, NotificationService

    class Notifier:
        def send(self, _item):
            raise AssertionError("disabled JS notifications must not send")

    service = NotificationService(
        NotificationConfig(js_notification_enabled=False, gotify_enabled=True),
        notifier=Notifier(),
    )
    assert not service.notify("JsNotification", "hidden", from_js=True)
    service.close()


def test_js_notification_host_routes_send_and_error_to_service(tmp_path, monkeypatch):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.notification import NotificationService
    from bgi_touch.vision.coordinate import ScreenTransform

    notification_service = SimpleNamespace(notify=Mock(), close=Mock())
    monkeypatch.setattr(
        NotificationService,
        "load",
        lambda *_args, **_kwargs: notification_service,
    )
    (tmp_path / "main.js").write_text(
        'notification.send("完成"); notification.error("失败"); return "ok";',
        encoding="utf-8",
    )
    input_simulator = SimpleNamespace(
        key_down=Mock(), key_up=Mock(), key_press=Mock(), click_ref=Mock(),
        move_camera_by=Mock(), attack=Mock(), attack_down=Mock(), attack_up=Mock(),
        button_down=Mock(), button_up=Mock(), release_all=Mock(),
    )
    context = SimpleNamespace(
        input=input_simulator,
        device=SimpleNamespace(paste_text=Mock()),
        transform=ScreenTransform(1920, 1080),
        sleep=lambda _ms: None,
    )

    assert JsScriptRuntime(context, tmp_path, log=lambda _message: None).run() == "ok"
    assert notification_service.notify.call_args_list[0].args == (
        "JsNotification", "完成",
    )
    assert notification_service.notify.call_args_list[0].kwargs == {
        "result": "Success", "from_js": True,
    }
    assert notification_service.notify.call_args_list[1].args == (
        "JsNotification", "失败",
    )
    assert notification_service.notify.call_args_list[1].kwargs == {
        "result": "Fail", "from_js": True,
    }
    notification_service.close.assert_called_once_with()
