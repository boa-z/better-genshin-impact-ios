import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


class _Ocr:
    def __init__(self, batches):
        self.batches = list(batches)

    def recognize(self, _image):
        return self.batches.pop(0)


def _item(text, x=2, y=3, width=20, height=10, confidence=0.9):
    return SimpleNamespace(
        text=text, x=x, y=y, width=width, height=height,
        confidence=confidence,
    )


def _context(frame=None):
    from bgi_touch.vision.coordinate import ScreenTransform

    frame = frame if frame is not None else np.zeros((1080, 1920, 3), np.uint8)
    return SimpleNamespace(
        input=SimpleNamespace(
            key_down=Mock(), key_up=Mock(), key_press=Mock(), click_ref=Mock(),
            move_camera_by=Mock(), attack=Mock(), attack_down=Mock(),
            attack_up=Mock(), button_down=Mock(), button_up=Mock(),
            tap_button=Mock(), drag_ref=Mock(), release_all=Mock(),
        ),
        device=SimpleNamespace(paste_text=Mock(), tap=Mock()),
        transform=ScreenTransform(1920, 1080),
        sleep=lambda _ms: None,
        capture_bgr=lambda: frame.copy(),
    )


def test_find_ocr_returns_combined_roi_text_and_matches_across_lines(monkeypatch):
    from bgi_touch.engine import recognition

    ctx = _context()
    image = recognition.ImageRegion(ctx, ctx.capture_bgr())
    ocr = _Ocr([
        [_item("领取 "), _item("奖励", x=30)],
        [_item("一键 "), _item("领取", x=30)],
    ])
    monkeypatch.setattr(recognition, "get_ocr", lambda: ocr)

    plain = image.find(recognition.RecognitionObject.ocr(100, 200, 300, 120))
    matched = image.find(
        recognition.RecognitionObject.ocr_match(
            100, 200, 300, 120, "一键领取",
        )
    )

    assert (plain.x, plain.y, plain.width, plain.height) == (100, 200, 300, 120)
    assert plain.text == "领取奖励"
    assert matched.is_exist()
    assert matched.text == "一键领取"


def test_js_find_and_find_multi_invoke_bettergi_callbacks(tmp_path, monkeypatch):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine import recognition
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    ocr = _Ocr([
        [_item("跨行 "), _item("成功", x=30)],
        [_item("甲"), _item("乙", x=30)],
        [],
    ])
    monkeypatch.setattr(recognition, "get_ocr", lambda: ocr)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"main": "main.js"}), encoding="utf-8"
    )
    (tmp_path / "main.js").write_text(
        """
(async function () {
  const image = captureGameRegion();
  let foundText = "";
  let multiCount = 0;
  let failed = 0;
  const found = image.Find(
    RecognitionObject.OcrMatch(0, 0, 1920, 1080, "跨行成功"),
    region => { foundText = region.Text; },
    () => { failed++; }
  );
  const multi = image.FindMulti(
    RecognitionObject.Ocr(0, 0, 1920, 1080),
    regions => { multiCount = regions.length; },
    () => { failed++; }
  );
  const empty = image.Find(
    RecognitionObject.Ocr(0, 0, 1920, 1080),
    () => {},
    () => { failed++; }
  );
  return JSON.stringify({
    foundText, found: found.IsExist(), multiCount,
    returnedCount: multi.length, empty: empty.IsEmpty(), failed
  });
})();
""",
        encoding="utf-8",
    )

    result = json.loads(JsScriptRuntime(
        _context(), tmp_path, log=lambda _message: None,
    ).run())

    assert result == {
        "foundText": "跨行成功",
        "found": True,
        "multiCount": 2,
        "returnedCount": 2,
        "empty": True,
        "failed": 1,
    }
