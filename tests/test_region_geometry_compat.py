import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


def _context(frame=None):
    from bgi_touch.vision.coordinate import ScreenTransform

    frame = frame if frame is not None else np.zeros((1080, 1920, 3), np.uint8)
    return SimpleNamespace(
        transform=ScreenTransform(1920, 1080),
        device=SimpleNamespace(tap=Mock(), paste_text=Mock()),
        input=SimpleNamespace(
            key_down=Mock(), key_up=Mock(), key_press=Mock(), click_ref=Mock(),
            move_camera_by=Mock(), attack=Mock(), attack_down=Mock(),
            attack_up=Mock(), button_down=Mock(), button_up=Mock(),
            tap_button=Mock(), drag_ref=Mock(), release_all=Mock(),
        ),
        sleep=lambda _ms: None,
        capture_bgr=lambda: frame.copy(),
    )


def test_js_rect_overloads_and_region_coordinate_conversion(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    (tmp_path / "main.js").write_text(
        """
const rect = new OpenCvSharp.OpenCvSharp.Rect(100, 200, 300, 120);
const ro = RecognitionObject.Ocr(rect);
const image = captureGameRegion();
const crop = image.DeriveCrop(rect);
const child = crop.Derive(new OpenCvSharp.Rect(10, 20, 30, 40));
const point = child.ConvertPositionToGameCaptureRegion(5, 6);
const box = child.ConvertPositionToGameCaptureRegion(1, 2, 7, 8);
const own = child.ConvertSelfPositionToGameCaptureRegion();
return JSON.stringify({
  roi: [ro.RegionOfInterest.X, ro.RegionOfInterest.Y,
        ro.RegionOfInterest.Width, ro.RegionOfInterest.Height],
  crop: [crop.X, crop.Y, crop.Width, crop.Height],
  child: [child.X, child.Y, child.Width, child.Height],
  point: [point.Item1, point.Item2],
  box: [box.X, box.Y, box.Width, box.Height],
  own: [own.X, own.Y, own.Width, own.Height]
});
""",
        encoding="utf-8",
    )

    result = json.loads(JsScriptRuntime(
        _context(), tmp_path, log=lambda _message: None,
    ).run())

    assert result == {
        "roi": [100, 200, 300, 120],
        "crop": [100, 200, 300, 120],
        "child": [110, 220, 30, 40],
        "point": [115, 226],
        "box": [111, 222, 7, 8],
        "own": [110, 220, 30, 40],
    }


def test_region_move_to_uses_absolute_virtual_pointer_coordinates():
    from bgi_touch.engine.recognition import Region
    from bgi_touch.input.pointer import TouchPointer

    ctx = _context()
    pointer = TouchPointer(ctx.input)
    ctx._script_pointer = pointer
    region = Region(ctx, 100, 200, 300, 120)

    region.move_to(10, 20, 30, 40)
    pointer.left_click()

    ctx.input.click_ref.assert_called_once_with(125, 240)
