import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_js_runtime_core_host_types_and_compat_version(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import BETTERGI_COMPAT_VERSION, JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    (tmp_path / "main.js").write_text(
        """
const recognition = new RecognitionObject();
recognition.Name = 'custom';
const point = new Point2f(3, 4);
const emptyMat = new Mat();
const region = new Region(10, 20, 30, 40);
const derived = region.Derive(5, 6, 7, 8);
GameCaptureRegion.gameRegion1080PPosClick(297, 437);
GameCaptureRegion.GameRegionClick((size, scale) => [size.Width / 2, size.Height / 2]);
recognition.Dispose();
emptyMat.Dispose();
region.Dispose();
return JSON.stringify({
  version: getVersion(), name: recognition.name,
  distance: point.DistanceTo(new Point2f(0, 0)),
  wasEmpty: emptyMat.Empty(), x: derived.X, y: derived.Y,
  material: GridScreenName.Materials,
  iconMode: ItemIconRecognitionMode.GridIcon
});
""",
        encoding="utf-8",
    )
    device = SimpleNamespace(paste_text=Mock(), tap=Mock())
    input_simulator = SimpleNamespace(
        key_down=Mock(), key_up=Mock(), key_press=Mock(), click_ref=Mock(),
        move_camera_by=Mock(), attack=Mock(), attack_down=Mock(), attack_up=Mock(),
        button_down=Mock(), button_up=Mock(), release_all=Mock(),
    )
    ctx = SimpleNamespace(
        input=input_simulator,
        device=device,
        transform=ScreenTransform(2778, 1284),
        sleep=lambda _ms: None,
    )

    result = json.loads(JsScriptRuntime(ctx, tmp_path).run())

    assert result.pop("x") == pytest.approx(15)
    assert result.pop("y") == pytest.approx(26)
    assert result == {
        "version": BETTERGI_COMPAT_VERSION,
        "name": "custom",
        "distance": 5,
        "wasEmpty": True,
        "material": "Materials",
        "iconMode": "GridIcon",
    }
    input_simulator.click_ref.assert_called_once_with(297.0, 437.0)
    device.tap.assert_called_once_with(
        1389.0, 642.0, image_width=2778, image_height=1284,
    )


def test_region_click_to_uses_reference_coordinates():
    from bgi_touch.engine.recognition import Region
    from bgi_touch.vision.coordinate import ScreenTransform

    device = SimpleNamespace(tap=Mock())
    ctx = SimpleNamespace(
        device=device,
        transform=ScreenTransform(1920, 1080),
    )
    region = Region(ctx, 100, 200, 80, 60)

    region.ClickTo(10, 20, 20, 10)

    device.tap.assert_called_once_with(
        120.0, 225.0, image_width=1920, image_height=1080,
    )


def test_ultrawide_transform_round_trips_all_reference_anchors():
    from bgi_touch.vision.coordinate import ScreenTransform

    transform = ScreenTransform(2778, 1284)
    for point in ((10, 20), (500, 300), (700, 540), (960, 540),
                  (1220, 700), (1500, 900), (1910, 1060)):
        device_point = transform.to_device(*point)
        reference_point = transform.to_ref(*device_point)
        assert reference_point == pytest.approx(point)
