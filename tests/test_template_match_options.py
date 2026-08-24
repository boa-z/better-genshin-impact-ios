from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest


def _context():
    from bgi_touch.vision.coordinate import ScreenTransform

    return SimpleNamespace(
        transform=ScreenTransform(1920, 1080),
        device=SimpleNamespace(tap=Mock()),
        sleep=lambda _ms: None,
    )


def _runtime_context(frame):
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


def test_use_three_channels_distinguishes_equal_grayscale_templates():
    from bgi_touch.engine.recognition import ImageRegion, Mat, RecognitionObject

    rng = np.random.default_rng(20260824)
    template = rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    grayscale_decoy = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    screen = np.zeros((80, 120, 3), np.uint8)
    screen[10:22, 10:22] = grayscale_decoy
    screen[10:22, 60:72] = template
    image = ImageRegion(_context(), screen)

    grayscale = RecognitionObject.template_match(Mat(template))
    grayscale.threshold = 0.99
    color = grayscale.clone()
    color.use3Channels = True

    assert image.find(grayscale).x == 10
    assert image.find(color).x == 60
    assert color.Use3Channels is True
    assert grayscale.Use3Channels is False


def test_template_match_mask_ignores_default_green_background():
    from bgi_touch.engine.recognition import ImageRegion, Mat, RecognitionObject

    rng = np.random.default_rng(7)
    template = np.full((16, 16, 3), (0, 255, 0), np.uint8)
    feature = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    template[4:12, 4:12] = feature
    target = np.full_like(template, (255, 0, 0))
    target[4:12, 4:12] = feature
    screen = np.zeros((60, 100, 3), np.uint8)
    screen[20:36, 50:66] = target

    ro = RecognitionObject.template_match(Mat(template), True)
    ro.Use3Channels = True
    ro.TemplateMatchMode = cv2.TM_CCORR_NORMED
    ro.Threshold = 0.99
    found = ImageRegion(_context(), screen).find(ro)

    assert found.is_exist()
    assert (found.x, found.y) == (50, 20)
    assert ro.UseMask is True
    assert ro.MaskMat is not None


def test_max_match_count_limits_find_multi_results():
    from bgi_touch.engine.recognition import ImageRegion, Mat, RecognitionObject

    rng = np.random.default_rng(11)
    template = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    screen = np.zeros((50, 120, 3), np.uint8)
    for x in (10, 40, 70):
        screen[10:18, x:x + 8] = template
    ro = RecognitionObject.template_match(Mat(template))
    ro.Threshold = 0.99
    ro.MaxMatchCount = 2

    found = ImageRegion(_context(), screen).find_multi(ro, limit=10)

    assert len(found) == 2


def test_js_template_match_mask_overload_and_options(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    rng = np.random.default_rng(19)
    template = np.full((16, 16, 3), (0, 255, 0), np.uint8)
    feature = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    template[4:12, 4:12] = feature
    target = np.full_like(template, (255, 0, 0))
    target[4:12, 4:12] = feature
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[120:136, 300:316] = target
    cv2.imwrite(str(tmp_path / "template.png"), template)
    (tmp_path / "main.js").write_text(
        """
const ro = RecognitionObject.TemplateMatch(
  file.ReadImageMatSync("template.png"), true
);
ro.Use3Channels = true;
ro.TemplateMatchMode = 3;
ro.MaxMatchCount = 1;
ro.Threshold = 0.99;
const found = captureGameRegion().Find(ro);
return JSON.stringify({
  x: found.X, y: found.Y, useMask: ro.UseMask,
  use3Channels: ro.Use3Channels, hasMask: ro.MaskMat !== null
});
""",
        encoding="utf-8",
    )

    result = __import__("json").loads(JsScriptRuntime(
        _runtime_context(frame), tmp_path, log=lambda _message: None,
    ).run())

    assert result == {
        "x": 300,
        "y": 120,
        "useMask": True,
        "use3Channels": True,
        "hasMask": True,
    }
