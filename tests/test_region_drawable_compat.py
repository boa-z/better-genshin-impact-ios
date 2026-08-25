import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


def _context(frame):
    from bgi_touch.vision.coordinate import ScreenTransform

    return SimpleNamespace(
        transform=ScreenTransform(frame.shape[1], frame.shape[0]),
        capture_bgr=Mock(return_value=frame.copy()),
        device=SimpleNamespace(tap=Mock()),
        input=SimpleNamespace(),
        sleep=lambda _ms: None,
    )


def test_region_drawables_are_geometry_only_and_do_not_capture():
    from bgi_touch.engine.recognition import GameCaptureRegion, RecognitionObject, Region

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ctx = _context(frame)
    region = Region(ctx, 100, 200, 80, 60)
    pen = object()

    own = region.SelfToRectDrawable("own", pen)
    rect = region.ToRectDrawable(1, 2, 3, 4, "child", pen)
    line = region.ToLineDrawable(1, 2, 30, 40, "path", pen)

    assert (own.Kind, own.Name, own.X, own.Y, own.Width, own.Height) == (
        "rect", "own", 100, 200, 80, 60,
    )
    assert (rect.X, rect.Y, rect.Width, rect.Height, rect.Pen) == (
        1, 2, 3, 4, pen,
    )
    assert (line.Kind, line.X1, line.Y1, line.X2, line.Y2) == (
        "line", 1, 2, 30, 40,
    )

    capture = GameCaptureRegion(ctx, frame, 300, 400)
    converted = capture.ConvertToRectDrawable(5, 6, 7, 8, pen, "capture")
    assert (converted.Name, converted.X, converted.Y) == ("capture", 305, 406)
    assert ctx.capture_bgr.call_count == 0

    ro = RecognitionObject()
    ro.DrawOnWindow = True
    ro.DrawOnWindowPen = pen
    assert ro.DrawOnWindow is True
    assert ro.DrawOnWindowPen is pen
    assert ro.Clone().DrawOnWindowPen is pen


def test_game_capture_region_drawable_methods_are_available_to_javascript(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    (tmp_path / "main.js").write_text(
        """
const mat = Mat.FromArray([[1, 2], [3, 4]]);
const capture = new GameCaptureRegion(mat, 100, 200);
const pen = new Pen(Color.Coral, 2);
const rect = capture.ConvertToRectDrawable(5, 6, 7, 8, pen, 'capture');
const line = capture.ConvertToLineDrawable(1, 2, 3, 4, pen, 'line');
const ro = new RecognitionObject();
ro.DrawOnWindowPen = pen;
return JSON.stringify({
  capture: [capture.X, capture.Y, capture.Width, capture.Height],
  rect: [rect.Kind, rect.Name, rect.X, rect.Y, rect.Width, rect.Height],
  line: [line.Kind, line.Name, line.X1, line.Y1, line.X2, line.Y2],
  pen: [ro.DrawOnWindowPen.Color.Name, ro.DrawOnWindowPen.Width]
});
""",
        encoding="utf-8",
    )

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = json.loads(JsScriptRuntime(_context(frame), tmp_path).run())
    assert result == {
        "capture": [100, 200, 2, 2],
        "rect": ["rect", "capture", 105, 206, 7, 8],
        "line": ["line", "line", 101, 202, 103, 204],
        "pen": ["Coral", 2],
    }
