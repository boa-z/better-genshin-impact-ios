import json
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest


def _context(frame):
    from bgi_touch.vision.coordinate import ScreenTransform

    return SimpleNamespace(
        transform=ScreenTransform(frame.shape[1], frame.shape[0]),
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


def _reference_ro(ro_type="Ocr"):
    from bgi_touch.engine.recognition import RecognitionObject, SearchOptions

    ro = RecognitionObject()
    ro.recognition_type = ro_type
    ro.ReferenceImageSize = (100, 100)
    ro.ReferenceBoundingBox = (10, 20, 10, 10)
    ro.SearchOptions = SearchOptions(
        AnchorMode="TopLeft",
        ReferenceSearchBox=(10, 20, 20, 20),
        ExpandPercent=[0, 0, 0, 0],
    )
    return ro


@pytest.mark.parametrize(
    ("anchor", "width", "height", "expected"),
    [
        ("TopLeft", 300, 200, (20, 40, 40, 40)),
        ("TopRight", 300, 200, (120, 40, 40, 40)),
        ("BottomLeft", 200, 300, (20, 140, 40, 40)),
        ("BottomRight", 300, 200, (120, 40, 40, 40)),
        ("Center", 300, 200, (70, 40, 40, 40)),
    ],
)
def test_reference_search_anchor_transforms_reference_box(
    anchor, width, height, expected,
):
    from bgi_touch.engine.recognition import ImageRegion

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    image = ImageRegion(_context(frame), frame)
    ro = _reference_ro()
    ro.SearchOptions.AnchorMode = anchor

    resolved = image._resolve_search_region(ro)

    assert resolved == (expected, (20, 20))


def test_reference_search_auto_uses_responsive_anchor_zones():
    from bgi_touch.engine.recognition import ImageRegion

    cases = [
        ((100, 100, 20, 20), (1400, 1000), (100, 100, 20, 20)),
        ((490, 100, 20, 20), (1400, 1000), (690, 100, 20, 20)),
        ((880, 100, 20, 20), (1400, 1000), (1280, 100, 20, 20)),
        ((100, 490, 20, 20), (1000, 1400), (100, 690, 20, 20)),
        ((100, 880, 20, 20), (1000, 1400), (100, 1280, 20, 20)),
    ]
    for bbox, (width, height), expected in cases:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        image = ImageRegion(_context(frame), frame)
        ro = _reference_ro()
        ro.ReferenceImageSize = (1000, 1000)
        ro.ReferenceBoundingBox = bbox
        ro.SearchOptions.AnchorMode = "Auto"
        ro.SearchOptions.ReferenceSearchBox = None

        resolved = image._resolve_search_region(ro)

        assert resolved[0] == expected


def test_reference_template_match_scales_template_and_keeps_wide_screen_anchor():
    from bgi_touch.engine.recognition import ImageRegion, Mat, RecognitionObject, SearchOptions

    rng = np.random.default_rng(20260824)
    template = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    frame = np.zeros((1284, 2778, 3), dtype=np.uint8)
    scale = min(frame.shape[1] / 1920, frame.shape[0] / 1080)
    offset_x = frame.shape[1] - 1920 * scale
    x = round(offset_x + 1680 * scale)
    y = round(32 * scale)
    w = round(32 * scale)
    h = round(32 * scale)
    target = cv2.resize(template, (w, h), interpolation=cv2.INTER_LINEAR)
    frame[y:y + h, x:x + w] = target

    ro = RecognitionObject.template_match(Mat(template))
    ro.Use3Channels = True
    ro.Threshold = 0.99
    ro.ReferenceImageSize = (1920, 1080)
    ro.ReferenceBoundingBox = (1680, 32, 32, 32)
    ro.SearchOptions = SearchOptions(AnchorMode="TopRight", ExpandPercent=[0])

    found = ImageRegion(_context(frame), frame).find(ro)

    assert found.is_exist()
    assert (found.dx, found.dy, found.dw, found.dh) == (x, y, w, h)


def test_reference_search_percent_expand_uses_current_image_edges():
    from bgi_touch.engine.recognition import ImageRegion

    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    image = ImageRegion(_context(frame), frame)
    ro = _reference_ro()
    ro.ReferenceImageSize = (100, 100)
    ro.ReferenceBoundingBox = (20, 20, 10, 10)
    ro.SearchOptions.ReferenceSearchBox = (20, 20, 20, 20)
    ro.SearchOptions.ExpandPercent = [0.1, 0.1, 0.2, 0.05]

    resolved = image._resolve_search_region(ro)

    assert resolved[0] == (10, 20, 130, 70)


def test_reference_search_too_small_for_template_returns_miss_without_cv_error():
    from bgi_touch.engine.recognition import ImageRegion, Mat, RecognitionObject, SearchOptions

    template = np.full((30, 30, 3), 255, dtype=np.uint8)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    ro = RecognitionObject.template_match(Mat(template))
    ro.ReferenceImageSize = (100, 100)
    ro.ReferenceBoundingBox = (20, 20, 30, 30)
    ro.SearchOptions = SearchOptions(
        AnchorMode="TopLeft",
        ReferenceSearchBox=(20, 20, 10, 10),
        ExpandPercent=[0],
    )

    found = ImageRegion(_context(frame), frame).find(ro)

    assert found.is_empty()


def test_reference_search_is_rejected_after_derive_crop():
    from bgi_touch.engine.recognition import ImageRegion

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    root = ImageRegion(_context(frame), frame)
    cropped = root.derive_crop(0, 0, 100, 100)
    ro = _reference_ro()

    assert cropped.find(ro).is_empty()


def test_clone_deep_copies_reference_search_options():
    from bgi_touch.engine.recognition import RecognitionObject, SearchOptions

    ro = _reference_ro()
    cloned = ro.clone()
    cloned.SearchOptions.AnchorMode = "Center"
    cloned.SearchOptions.ExpandSize = (5, 6)

    assert ro.SearchOptions is not cloned.SearchOptions
    assert ro.SearchOptions.AnchorMode == "TopLeft"
    assert ro.SearchOptions.ExpandSize is None


def test_js_runtime_exposes_reference_search_types_and_fields(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    (tmp_path / "main.js").write_text(
        """
const ro = new RecognitionObject();
ro.ReferenceImageSize = new OpenCvSharp.Size(1920, 1080);
ro.ReferenceBoundingBox = new OpenCvSharp.Rect(100, 80, 40, 30);
const search = new SearchOptions();
search.AnchorMode = SearchAnchorMode.BottomRight;
search.ReferenceSearchBox = new OpenCvSharp.Rect(90, 70, 80, 60);
search.ExpandSize = new OpenCvSharp.Size(12, 13);
search.ExpandPercent = new SearchExpandRatio(0.1, 0.2, 0.3, 0.4);
ro.SearchOptions = search;
const clone = ro.Clone();
clone.SearchOptions.AnchorMode = SearchAnchorMode.Center;
return JSON.stringify({
  size: [ro.ReferenceImageSize.Width, ro.ReferenceImageSize.Height],
  bbox: [ro.ReferenceBoundingBox.X, ro.ReferenceBoundingBox.Y,
         ro.ReferenceBoundingBox.Width, ro.ReferenceBoundingBox.Height],
  anchor: ro.SearchOptions.AnchorMode,
  box: [ro.SearchOptions.ReferenceSearchBox.X, ro.SearchOptions.ReferenceSearchBox.Y],
  expand: [ro.SearchOptions.ExpandSize.Width, ro.SearchOptions.ExpandSize.Height],
  percent: [ro.SearchOptions.ExpandPercent.Left, ro.SearchOptions.ExpandPercent.Top,
            ro.SearchOptions.ExpandPercent.Right, ro.SearchOptions.ExpandPercent.Bottom],
  cloneAnchor: clone.SearchOptions.AnchorMode,
  originalAnchor: ro.SearchOptions.AnchorMode
});
""",
        encoding="utf-8",
    )

    result = json.loads(JsScriptRuntime(
        _context(frame), tmp_path, log=lambda _message: None,
    ).run())

    assert result == {
        "size": [1920, 1080],
        "bbox": [100, 80, 40, 30],
        "anchor": "BottomRight",
        "box": [90, 70],
        "expand": [12, 13],
        "percent": [0.1, 0.2, 0.3, 0.4],
        "cloneAnchor": "Center",
        "originalAnchor": "BottomRight",
    }
