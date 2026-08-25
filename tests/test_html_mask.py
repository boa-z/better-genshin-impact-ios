import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock


def test_html_mask_host_window_messages_and_sandbox(tmp_path):
    from bgi_touch.engine.html_mask import HtmlMaskHost, HtmlMaskManager

    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "mask.html").write_text(
        "<html><head></head><body>mask</body></html>", encoding="utf-8"
    )
    manager = HtmlMaskManager()
    host = HtmlMaskHost(tmp_path, manager=manager)

    window_id = host.show("assets/mask.html", "test-mask")
    assert window_id == "test-mask"
    assert host.exists(window_id)
    host.setClickThrough(window_id, True)
    assert host.getClickThrough(window_id)
    host.send(window_id, "/status", '{"count":2}')

    events = manager.events(after=0, timeout_s=0)
    assert events["windows"] == [{
        "id": "test-mask",
        "entry": "assets/mask.html",
        "clickThrough": True,
        "messages": [{"url": "/status", "data": {"count": 2}}],
    }]
    manager.post_from_html(window_id, "/ready", {"ok": True})
    assert json.loads(host.poll(window_id)) == {
        "url": "/ready", "data": {"ok": True}
    }
    assert host.pollAll(window_id) == "[]"

    try:
        host.show("../outside.html")
    except PermissionError:
        pass
    else:
        raise AssertionError("htmlMask must reject paths outside the script root")

    assert host.close(window_id)
    assert not host.exists(window_id)


def test_html_mask_request_response_matching(tmp_path):
    from bgi_touch.engine.html_mask import HtmlMaskHost, HtmlMaskManager

    (tmp_path / "mask.html").write_text("<html></html>", encoding="utf-8")
    manager = HtmlMaskManager()
    host = HtmlMaskHost(tmp_path, manager=manager)
    window_id = host.show("mask.html", "request-mask")
    initial = manager.events(after=-1, timeout_s=0)["seq"]
    result = []

    thread = threading.Thread(
        target=lambda: result.append(
            host.request(window_id, "/calculate", '{"value":3}', 1000)
        )
    )
    thread.start()
    events = manager.events(after=initial, timeout_s=1)
    request = events["windows"][0]["messages"][0]
    assert request["url"] == "/calculate"
    assert request["data"] == {"value": 3}
    manager.post_from_html(
        window_id, "/__response__", {"answer": 6}, request["requestId"]
    )
    thread.join(timeout=2)

    assert result == ['{"answer":6}']
    host.closeAll()


def test_html_mask_same_custom_id_isolated_between_script_roots(tmp_path):
    from bgi_touch.engine.html_mask import HtmlMaskManager

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "mask.html").write_text("<html></html>", encoding="utf-8")
    (second_root / "mask.html").write_text("<html></html>", encoding="utf-8")

    manager = HtmlMaskManager()
    first = manager.show(first_root, "mask.html", "shared-id")
    second = manager.show(second_root, "mask.html", "shared-id")

    assert first == "shared-id"
    assert second != first
    assert manager.exists(first)
    assert manager.exists(second)
    assert manager.storage_namespace(first) != manager.storage_namespace(second)
    assert manager.resolve_asset(first, "mask.html") == (first_root / "mask.html").resolve()
    assert manager.resolve_asset(second, "mask.html") == (second_root / "mask.html").resolve()

    manager.close_many([first, second])


def test_webui_injects_html_mask_bridge_and_serves_assets(tmp_path):
    from bgi_touch.engine.html_mask import html_mask_manager
    from bgi_touch.webui.server import api_html_mask_file

    window_id = "test-web-mask"
    (tmp_path / "index.html").write_text(
        "<HTML><HEAD></HEAD><BODY>hello</BODY></HTML>", encoding="utf-8"
    )
    (tmp_path / "asset.txt").write_text("asset", encoding="utf-8")
    html_mask_manager.show(tmp_path, "index.html", window_id)
    try:
        response = api_html_mask_file(window_id, "index.html")
        body = response.body.decode("utf-8")
        assert "window.htmlMask" in body
        assert "bgi-html-mask" in body
        assert "scopedStorage" in body
        assert "storageNamespace" in body
        assert "const existingMask = window.htmlMask" in body
        assert "bgi-html-mask-ready" in body
        assert body.lower().index("window.htmlmask") < body.lower().index("<body")
        assert f'/api/html-masks/{window_id}/files/' in body

        asset = api_html_mask_file(window_id, "asset.txt")
        assert asset.path == (tmp_path / "asset.txt").resolve()
    finally:
        html_mask_manager.close(window_id)


def test_webui_mask_page_tracks_iframe_ready_and_source():
    from bgi_touch.webui.server import STATIC_DIR

    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert "function markMaskReady" in page
    assert "bgi-html-mask-ready" in page
    assert "event.source !== state.iframe.contentWindow" in page


def test_js_runtime_exposes_html_mask_lifecycle(tmp_path):
    import pytest

    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    (tmp_path / "mask.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "main.js").write_text(
        """
const id = htmlMask.Show("mask.html", "runtime-mask");
const before = htmlMask.Exists(id);
htmlMask.SetClickThrough(id, true);
const clickThrough = htmlMask.GetClickThrough(id);
const closed = htmlMask.Close(id);
return JSON.stringify({id, before, clickThrough, closed, after: htmlMask.Exists(id)});
""",
        encoding="utf-8",
    )
    input_simulator = SimpleNamespace(
        key_down=Mock(), key_up=Mock(), key_press=Mock(), click_ref=Mock(),
        move_camera_by=Mock(), attack=Mock(), attack_down=Mock(), attack_up=Mock(),
        button_down=Mock(), button_up=Mock(),
    )
    ctx = SimpleNamespace(
        input=input_simulator,
        device=SimpleNamespace(paste_text=Mock()),
        transform=ScreenTransform(1920, 1080),
        sleep=lambda _ms: None,
    )

    result = json.loads(JsScriptRuntime(ctx, tmp_path).run())
    assert result == {
        "id": "runtime-mask", "before": True, "clickThrough": True,
        "closed": True, "after": False,
    }


def test_webui_read_only_page_polling_does_not_connect_device(monkeypatch):
    from unittest.mock import Mock

    from bgi_touch.webui import server

    monkeypatch.setattr(server, "_ctx", None)
    connect = Mock(side_effect=AssertionError("read-only polling must not connect"))
    monkeypatch.setattr(server, "get_ctx", connect)

    status = server.api_status()
    screenshot = server.api_screenshot()
    triggers = server.api_triggers()

    assert status["connected"] is False
    assert screenshot.status_code == 409
    assert triggers == {"active": [], "connected": False}
    connect.assert_not_called()
