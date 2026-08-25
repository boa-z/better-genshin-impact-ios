import json
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest


def _context():
    from bgi_touch.vision.coordinate import ScreenTransform

    return SimpleNamespace(
        transform=ScreenTransform(1920, 1080),
        device=SimpleNamespace(tap=Mock(), paste_text=Mock()),
        input=SimpleNamespace(
            key_down=Mock(), key_up=Mock(), key_press=Mock(),
            click_ref=Mock(), move_camera_by=Mock(), attack=Mock(),
            attack_down=Mock(), attack_up=Mock(), button_down=Mock(),
            button_up=Mock(), release_all=Mock(),
        ),
        sleep=lambda _ms: None,
        capture_bgr=lambda: np.zeros((1080, 1920, 3), dtype=np.uint8),
    )


def test_mat_factories_and_common_opencv_operations():
    from bgi_touch.engine.recognition import Mat, Size

    zeros = Mat.Zeros(2, 3, 16)  # CV_8UC3
    assert (zeros.Rows, zeros.Cols, zeros.Channels(), zeros.Type()) == (2, 3, 3, 16)
    assert np.all(zeros.bgr == 0)

    ones = Mat.Ones(Size(3, 2), 0)
    eye = Mat.Eye(3, 0)
    assert np.array_equal(ones.bgr, np.ones((2, 3), dtype=np.uint8))
    assert np.array_equal(eye.bgr, np.eye(3, dtype=np.uint8))

    source = Mat.FromArray([[0, 10], [20, 30]])
    assert source.T().Get(0, 1) == 20
    assert source.Diag().Get(0) == 0
    assert source.GreaterThan(9).Get(1, 0) == 255
    assert source.Threshold(15, 255, cv2.THRESH_BINARY).Get(1, 0) == 255
    assert source.CountNonZero() == 3
    assert source.Sum().Val0 == 60

    bgr = Mat(np.array([[[1, 2, 3]]], dtype=np.uint8))
    hsv = bgr.CvtColor(cv2.COLOR_BGR2HSV)
    assert (hsv.Rows, hsv.Cols, hsv.Channels()) == (1, 1, 3)


def test_mat_pixel_data_and_image_decode_factories():
    from bgi_touch.engine.recognition import Mat

    pixels = Mat.FromPixelData(2, 1, 16, bytes([1, 2, 3, 4, 5, 6]))
    assert pixels.Get(0, 1)["Item2"] == 6

    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    decoded = Mat.ImDecode(encoded.tobytes())
    assert np.array_equal(decoded.bgr, image)

    rgba = Mat.FromImageData({
        "width": 1,
        "height": 1,
        "data": [30, 20, 10, 255],
    })
    assert rgba.Get(0, 0)["Item0"] == 10
    assert rgba.Get(0, 0)["Item2"] == 30


def test_mat_static_factories_are_available_to_javascript(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    (tmp_path / "main.js").write_text(
        """
const zeros = Mat.Zeros(2, 3, 16);
const source = Mat.FromArray([[1, 2], [3, 4]]);
const transposed = source.T();
const diagonal = Mat.Diag(source);
return JSON.stringify({
  shape: [zeros.Rows, zeros.Cols, zeros.Channels(), zeros.Type()],
  value: source.Get(1, 1),
  transposed: transposed.Get(0, 1),
  diagonal: [diagonal.Rows, diagonal.Cols, diagonal.Get(1)]
});
""",
        encoding="utf-8",
    )

    result = json.loads(JsScriptRuntime(_context(), tmp_path).run())
    assert result == {
        "shape": [2, 3, 3, 16],
        "value": 4,
        "transposed": 3,
        "diagonal": [2, 1, 4],
    }
