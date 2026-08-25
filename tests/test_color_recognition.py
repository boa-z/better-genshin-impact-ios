import json
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest


def _context(frame):
    from bgi_touch.vision.coordinate import ScreenTransform

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


def test_empty_recognition_object_matches_nothing():
    from bgi_touch.engine.recognition import ImageRegion, RecognitionObject

    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    image = ImageRegion(_context(frame), frame)
    ro = RecognitionObject()
    failed = []

    assert ro.RecognitionType == "None"
    assert image.Find(ro, fail_action=lambda: failed.append(True)).IsEmpty()
    assert image.FindMulti(ro, fail_action=lambda: failed.append(True)) == []
    assert failed == [True, True]


def test_color_match_uses_conversion_bounds_and_match_count():
    from bgi_touch.engine.recognition import ImageRegion, RecognitionObject

    frame = np.zeros((100, 140, 3), dtype=np.uint8)
    target_bgr = np.array([[[20, 160, 230]]], dtype=np.uint8)
    frame[30:50, 40:80] = target_bgr
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0, 0]

    ro = RecognitionObject()
    ro.RecognitionType = "ColorMatch"
    ro.RegionOfInterest = (40, 30, 40, 20)
    ro.ColorConversionCode = "BGR2HSV"
    ro.LowerColor = tuple(int(value) for value in target_hsv)
    ro.UpperColor = tuple(int(value) for value in target_hsv)
    ro.MatchCount = 400

    found = ImageRegion(_context(frame), frame).find(ro)

    assert found.IsExist()
    assert (found.x, found.y, found.width, found.height) == (40, 30, 40, 20)
    assert found.matchScore == 800
    multi = ImageRegion(_context(frame), frame).find_multi(ro)
    assert len(multi) == 1
    assert (multi[0].x, multi[0].y, multi[0].width, multi[0].height) == (40, 30, 40, 20)

    ro.MatchCount = 801
    assert ImageRegion(_context(frame), frame).find(ro).IsEmpty()


def test_color_range_and_ocr_passes_binary_mask_to_ocr(monkeypatch):
    from bgi_touch.engine import recognition

    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    target_bgr = np.array([[[30, 180, 220]]], dtype=np.uint8)
    frame[15:25, 25:65] = target_bgr
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0, 0]
    seen = []

    class Ocr:
        def recognize(self, image):
            seen.append(image.copy())
            return [SimpleNamespace(
                text="目标", x=3, y=4, width=20, height=8, confidence=0.95,
            )]

    monkeypatch.setattr(recognition, "get_ocr", lambda: Ocr())
    ro = recognition.RecognitionObject()
    ro.RecognitionType = "ColorRangeAndOcr"
    ro.RegionOfInterest = (20, 10, 50, 20)
    ro.ColorConversionCode = cv2.COLOR_BGR2HSV
    ro.LowerColor = target_hsv.tolist()
    ro.UpperColor = target_hsv.tolist()

    image = recognition.ImageRegion(_context(frame), frame)
    found = image.find(ro)
    multi = image.find_multi(ro, limit=1)

    assert found.IsExist()
    assert found.Text == "目标"
    assert (found.x, found.y, found.width, found.height) == (20, 10, 50, 20)
    assert len(multi) == 1
    assert len(seen) == 2
    assert seen[0].shape == (20, 50)
    assert seen[0][5, 5] == 255
    assert seen[0][0, 0] == 0


def test_recognition_json_loads_color_fields(tmp_path):
    from bgi_touch.engine.recognition_assets import load_recognition_object

    path = tmp_path / "Recognition.json"
    path.write_text(json.dumps({
        "objects": {
            "Marker": {
                "type": "ColorMatch",
                "roi": "rect(10, 20, 30, 40)",
                "colorCode": "BGR2HSV",
                "lowerColor": [90, 80, 80],
                "upperColor": [130, 255, 255],
                "matchCount": 12,
            },
        },
    }), encoding="utf-8")

    ro = load_recognition_object(path, "Marker")

    assert ro.RecognitionType == "ColorMatch"
    assert ro.ColorConversionCode == cv2.COLOR_BGR2HSV
    assert tuple(ro.LowerColor) == (90, 80, 80, 0)
    assert tuple(ro.UpperColor) == (130, 255, 255, 0)
    assert ro.MatchCount == 12


def test_js_exposes_scalar_and_color_recognition(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[100:120, 200:240] = (20, 160, 230)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"main": "main.js"}), encoding="utf-8",
    )
    (tmp_path / "main.js").write_text(
        """
const ro = new RecognitionObject();
ro.RecognitionType = "ColorMatch";
ro.RegionOfInterest = new OpenCvSharp.Rect(200, 100, 40, 20);
ro.ColorConversionCode = "BGR2RGB";
ro.LowerColor = new OpenCvSharp.Scalar(230, 160, 20);
ro.UpperColor = new OpenCvSharp.Scalar(230, 160, 20);
ro.MatchCount = 800;
const found = captureGameRegion().Find(ro);
return JSON.stringify({
  found: found.IsExist(),
  lower: [ro.LowerColor.Val0, ro.LowerColor.Val1, ro.LowerColor.Val2],
  matchCount: ro.MatchCount
});
""",
        encoding="utf-8",
    )

    result = json.loads(JsScriptRuntime(
        _context(frame), tmp_path, log=lambda _message: None,
    ).run())

    assert result == {
        "found": True,
        "lower": [230, 160, 20],
        "matchCount": 800,
    }
