import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


def test_map_asset_layer_parser_supports_underground_layers():
    from bgi_touch.pathing.map_locator import map_layer_from_path

    assert map_layer_from_path("Teyvat_0_256_SIFT.kp.bin") == 0
    assert map_layer_from_path("SeaOfBygoneEras_-2_1024_SIFT.mat.png") == -2


def test_map_image_size_reads_png_header_without_decoding(tmp_path, monkeypatch):
    from bgi_touch.triggers import map_mask

    image = tmp_path / "map.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" +
        (5632).to_bytes(4, "big") + (3840).to_bytes(4, "big") + b"\x00" * 8
    )
    map_mask.map_image_size.cache_clear()
    monkeypatch.setattr(
        map_mask.cv2, "imread",
        Mock(side_effect=AssertionError("PNG metadata must not decode the full image")),
    )

    assert map_mask.map_image_size(str(image)) == (5632, 3840)


def test_map_mask_tracks_minimap_from_existing_frame_without_capture():
    from bgi_touch.triggers.map_mask import MapMaskTrigger

    positioner = SimpleNamespace(
        locator=SimpleNamespace(last_layer=0),
        get_position_stable=Mock(return_value=(100.0, -50.0)),
    )
    big = SimpleNamespace(locate_view=Mock())
    ctx = SimpleNamespace(capture_bgr=Mock(side_effect=AssertionError("must not capture")))
    trigger = MapMaskTrigger(
        ctx,
        positioner=positioner,
        big_locator=big,
        main_ui_detector=lambda _ctx, _frame: True,
        log=lambda _message: None,
    )

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    trigger.on_frame(SimpleNamespace(bgr=frame))
    assert trigger.wait_idle()

    state = trigger.state.snapshot()
    assert state["scene"] == "gameplay"
    assert state["positionValid"] is True
    assert state["position"] == {
        "worldX": 100.0,
        "worldY": -50.0,
        "imageX": 4071.0,
        "imageY": 2060.5,
    }
    assert state["viewport"] == {
        "x": 4026.0,
        "y": 2015.5,
        "width": 90.0,
        "height": 90.0,
        "layer": 0,
    }
    ctx.capture_bgr.assert_not_called()
    big.locate_view.assert_not_called()


def test_map_mask_tracks_big_map_viewport_and_layer():
    from bgi_touch.triggers.map_mask import MapMaskTrigger

    class BigMap:
        last_layer = -1

        @staticmethod
        def locate_view(_frame):
            return 2000.0, 1000.0, 2.0

        @staticmethod
        def feature_to_world(x, y):
            return 48.0, -24.0

    trigger = MapMaskTrigger(
        SimpleNamespace(),
        map_name="旧日之海",
        positioner=SimpleNamespace(),
        big_locator=BigMap(),
        main_ui_detector=lambda _ctx, _frame: False,
        big_map_detector=lambda _ctx, _frame: True,
        log=lambda _message: None,
    )
    trigger.on_frame(SimpleNamespace(bgr=np.zeros((1000, 1800, 3), dtype=np.uint8)))
    assert trigger.wait_idle()

    state = trigger.state.snapshot()
    assert state["mapName"] == "SeaOfBygoneEras"
    assert state["scene"] == "bigMap"
    assert state["inBigMapUi"] is True
    assert state["layer"] == -1
    assert state["position"] == {
        "worldX": 48.0,
        "worldY": -24.0,
        "imageX": 6096.0,
        "imageY": 3096.0,
    }
    assert state["viewport"] == {
        "x": 1550.0,
        "y": 750.0,
        "width": 900.0,
        "height": 500.0,
        "layer": -1,
    }


def test_map_mask_skips_sift_outside_main_and_big_map_ui():
    from bgi_touch.triggers.map_mask import MapMaskTrigger

    big = SimpleNamespace(locate_view=Mock())
    trigger = MapMaskTrigger(
        SimpleNamespace(),
        positioner=SimpleNamespace(),
        big_locator=big,
        main_ui_detector=lambda _ctx, _frame: False,
        big_map_detector=lambda _ctx, _frame: False,
        log=lambda _message: None,
    )
    trigger.on_frame(SimpleNamespace(bgr=np.zeros((4, 8, 3), dtype=np.uint8)))
    assert trigger.wait_idle()

    assert trigger.state.snapshot()["scene"] == "other"
    big.locate_view.assert_not_called()


def test_map_mask_worker_drops_intermediate_pending_frame():
    from bgi_touch.triggers.map_mask import MapMaskTrigger

    entered = threading.Event()
    release = threading.Event()
    seen = []

    class Positioner:
        locator = SimpleNamespace(last_layer=0)

        def get_position_stable(self, frame):
            value = int(frame[0, 0, 0])
            seen.append(value)
            if value == 1:
                entered.set()
                assert release.wait(2)
            return float(value), 0.0

    trigger = MapMaskTrigger(
        SimpleNamespace(),
        positioner=Positioner(),
        big_locator=SimpleNamespace(),
        main_ui_detector=lambda _ctx, _frame: True,
        log=lambda _message: None,
    )
    for value in (1, 2, 3):
        frame = np.full((2, 2, 3), value, dtype=np.uint8)
        trigger.on_frame(SimpleNamespace(bgr=frame))
        if value == 1:
            assert entered.wait(1)
    release.set()
    assert trigger.wait_idle()

    assert seen == [1, 3]
    assert trigger.state.snapshot()["position"]["worldX"] == 3.0


def test_replacing_map_mask_trigger_closes_previous_worker():
    from bgi_touch.triggers.loop import TriggerLoop

    old = SimpleNamespace(name="MapMask", enabled=True, close=Mock())
    new = SimpleNamespace(name="MapMask", enabled=True)
    loop = TriggerLoop(SimpleNamespace(), log=lambda _message: None)
    loop.add(old)
    loop.add(new)

    old.close.assert_called_once_with()
    assert loop.get("MapMask") is new


def test_trigger_loop_restarts_new_map_mask_after_slow_capture_stops():
    from bgi_touch.triggers.loop import TriggerLoop

    capture_started = threading.Event()
    release_capture = threading.Event()
    new_frame_seen = threading.Event()

    class Context:
        def capture_region(self):
            capture_started.set()
            release_capture.wait(2)
            return object()

    old = SimpleNamespace(name="MapMask", enabled=True, close=Mock(), on_frame=Mock())
    new = SimpleNamespace(
        name="MapMask", enabled=True,
        on_frame=lambda _region: new_frame_seen.set(),
    )
    loop = TriggerLoop(Context(), interval_s=0.01, log=lambda _message: None)
    loop.add(old)
    loop.start()
    assert capture_started.wait(1)

    loop.remove("MapMask")
    loop.stop()
    loop.add(new)
    loop.start()
    release_capture.set()

    assert new_frame_seen.wait(1)
    loop.remove("MapMask")
    loop.stop()


def test_map_mask_api_polling_does_not_connect_or_capture(monkeypatch):
    from bgi_touch.webui import server

    monkeypatch.setattr(server, "_ctx", None)
    connect = Mock(side_effect=AssertionError("read-only map polling must not connect"))
    monkeypatch.setattr(server, "get_ctx", connect)

    state = server.api_map_mask()

    assert state["active"] is False
    assert state["scene"] == "disabled"
    connect.assert_not_called()


def test_map_mask_snapshot_cannot_mutate_shared_position():
    from bgi_touch.triggers.map_mask import MapMaskState

    state = MapMaskState("Teyvat")
    state.update(position={"worldX": 1.0}, viewport={"x": 2.0})
    snapshot = state.snapshot()
    snapshot["position"]["worldX"] = 99.0
    snapshot["viewport"]["x"] = 99.0

    assert state.snapshot()["position"]["worldX"] == 1.0
    assert state.snapshot()["viewport"]["x"] == 2.0


def test_map_mask_timer_maps_bettergi_configuration():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    ctx = SimpleNamespace(
        triggers=SimpleNamespace(clear=Mock()),
        enable_trigger=Mock(),
    )
    TaskDispatcher(ctx).add_timer({
        "name": "地图遮罩",
        "config": {"mapName": "渊下宫", "miniMapMaskEnabled": False},
    })

    ctx.triggers.clear.assert_called_once_with()
    ctx.enable_trigger.assert_called_once_with(
        "MapMask", map_name="渊下宫", mini_map_enabled=False
    )


def test_webui_contains_cached_map_tracking_panel():
    page = (
        Path(__file__).parents[1] / "bgi_touch" / "webui" / "static" / "index.html"
    ).read_text(encoding="utf-8")

    assert 'id="mapCanvas"' in page
    assert "fetch('/api/map-mask'" in page
    assert "地图面板不会主动请求设备截图" in page
