import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock


def test_key_mouse_hook_dispatches_bettergi_names_and_lifecycle():
    from bgi_touch.engine.keymouse_hook import KeyMouseHookManager

    manager = KeyMouseHookManager(log=lambda _message: None)
    hook = manager.create()
    events = []
    hook.OnKeyDown(lambda value: events.append(("down", value)))
    hook.OnKeyUp(lambda value: events.append(("data", value)), False)
    hook.OnMouseDown(lambda button, x, y: events.append((button, x, y)))
    hook.OnMouseWheel(lambda delta, x, y: events.append(("wheel", delta, x, y)))

    assert manager.enqueue({"type": "keyDown", "code": "AltLeft"})
    assert manager.enqueue({
        "type": "keyUp", "code": "KeyR", "shiftKey": True,
    })
    assert manager.enqueue({
        "type": "mouseDown", "button": 2, "x": 123.4, "y": 456.6,
    })
    assert manager.enqueue({
        "type": "mouseWheel", "delta": 120, "x": 10, "y": 20,
    })
    assert manager.drain() == 4
    assert events == [
        ("down", "LMenu"), ("data", "R, Shift"),
        ("Right", 123, 457), ("wheel", 120, 10, 20),
    ]

    hook.RemoveAllListeners()
    assert manager.enqueue({"type": "keyDown", "code": "KeyN"})
    assert manager.drain() == 0
    hook.Dispose()
    assert not manager.enqueue({"type": "keyDown", "code": "KeyN"})


def test_key_mouse_hook_manager_can_disable_webui_event_monitoring():
    from bgi_touch.engine.keymouse_hook import KeyMouseHookManager

    manager = KeyMouseHookManager(log=lambda _message: None, enabled=False)
    hook = manager.create()
    events = []
    hook.OnKeyDown(lambda value: events.append(value))

    assert not manager.has_hooks()
    assert not manager.enqueue({"type": "keyDown", "code": "KeyN"})
    assert manager.drain() == 0
    assert events == []


def test_js_runtime_key_mouse_hook_uses_script_thread_checkpoints(tmp_path):
    import pytest

    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    (tmp_path / "main.js").write_text(
        """
const hook = new KeyMouseHook();
const events = [];
hook.OnKeyDown(key => events.push(['down', key]));
hook.OnKeyUp(key => events.push(['up', key]));
hook.OnMouseDown((button, x, y) => events.push([button, x, y]));
await sleep(250);
hook.dispose();
return JSON.stringify(events);
""",
        encoding="utf-8",
    )
    input_simulator = SimpleNamespace(
        key_down=Mock(), key_up=Mock(), key_press=Mock(), click_ref=Mock(),
        move_camera_by=Mock(), attack=Mock(), attack_down=Mock(), attack_up=Mock(),
        button_down=Mock(), button_up=Mock(), release_all=Mock(),
    )
    ctx = SimpleNamespace(
        input=input_simulator,
        device=SimpleNamespace(paste_text=Mock()),
        transform=ScreenTransform(2778, 1284),
        sleep=lambda _ms: None,
    )
    runtime = JsScriptRuntime(ctx, tmp_path)

    def produce_events():
        deadline = time.monotonic() + 1
        while not runtime.has_key_mouse_hooks() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert runtime.enqueue_key_mouse_event({"type": "keyDown", "code": "KeyN"})
        assert runtime.enqueue_key_mouse_event({"type": "keyUp", "code": "KeyN"})
        assert runtime.enqueue_key_mouse_event({
            "type": "mouseDown", "button": 2, "nx": 0.5, "ny": 0.5,
        })

    producer = threading.Thread(target=produce_events)
    producer.start()
    result = json.loads(runtime.run())
    producer.join(timeout=1)

    assert result == [["down", "N"], ["up", "N"], ["Right", 960, 540]]
    assert not runtime.has_key_mouse_hooks()


def test_key_mouse_hook_web_endpoint_never_connects_device(monkeypatch):
    from bgi_touch.webui import server

    fake_runner = SimpleNamespace(enqueue_key_mouse_event=Mock(return_value=True))
    monkeypatch.setattr(server, "runner", fake_runner)
    connect = Mock(side_effect=AssertionError("hook events must not connect device"))
    monkeypatch.setattr(server, "get_ctx", connect)

    event = {"type": "keyDown", "code": "Backquote"}
    assert server.api_key_mouse_hook_event(event) == {"ok": True, "accepted": True}
    fake_runner.enqueue_key_mouse_event.assert_called_once_with(event)
    connect.assert_not_called()


def test_task_runner_reports_hook_activity_without_touching_context():
    from bgi_touch.webui.server import TaskRunner

    runner = TaskRunner()
    assert not runner.key_mouse_hook_active()

    runtime = SimpleNamespace(has_key_mouse_hooks=Mock(return_value=True))
    runner._js_runtime = runtime
    assert runner.key_mouse_hook_active()
    runtime.has_key_mouse_hooks.assert_called_once_with()


def test_webui_does_not_post_hook_events_until_a_script_registers_one():
    from bgi_touch.webui.server import STATIC_DIR, _mask_bridge

    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "let keyMouseHookEnabled = false" in page
    assert "if (!keyMouseHookEnabled) return" in page
    assert "s.keyMouseHookActive === true" in page
    assert "bgi-key-mouse-hook" in _mask_bridge("test-mask")


def test_webui_forwards_preview_and_mask_keyboard_to_key_mouse_hook():
    from bgi_touch.webui.server import STATIC_DIR, _mask_bridge

    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    bridge = _mask_bridge("test-mask")
    assert "/api/key-mouse-hook/event" in page
    assert "mouseDown" in page and "mouseWheel" in page
    assert "bgi-key-mouse-hook" in page
    assert "bgi-key-mouse-hook" in bridge
    assert "keydown" in bridge and "keyup" in bridge
