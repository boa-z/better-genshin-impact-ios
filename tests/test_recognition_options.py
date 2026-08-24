import json
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np


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


def test_binary_template_match_thresholds_source_before_matching():
    from bgi_touch.engine.recognition import ImageRegion, Mat, RecognitionObject

    template = np.zeros((12, 12, 3), dtype=np.uint8)
    template[2:10, 3:9] = 255
    frame = np.full((80, 120, 3), 180, dtype=np.uint8)
    target = np.full_like(template, 20)
    target[2:10, 3:9] = 220
    frame[30:42, 50:62] = target

    ro = RecognitionObject.template_match(Mat(template))
    ro.Threshold = 0.99
    ro.UseBinaryMatch = True
    ro.BinaryThreshold = 128

    found = ImageRegion(_context(frame), frame).find(ro)

    assert found.IsExist()
    assert (found.x, found.y) == (50, 30)


def test_ocr_replacements_apply_to_find_and_find_multi(monkeypatch):
    from bgi_touch.engine import recognition

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    class Ocr:
        def recognize(self, _image):
            return [SimpleNamespace(
                text="原神误", x=5, y=6, width=30, height=12, confidence=0.9,
            )]

    monkeypatch.setattr(recognition, "get_ocr", lambda: Ocr())
    image = recognition.ImageRegion(_context(frame), frame)

    plain = recognition.RecognitionObject.ocr(100, 200, 300, 120)
    plain.ReplaceDictionary = {"原神": ["原神误"]}
    found = image.find(plain)

    matched = recognition.RecognitionObject.ocr_match(
        100, 200, 300, 120, "原神",
    )
    matched.ReplaceDictionary = {"原神": ["原神误"]}
    multi = image.find_multi(matched, limit=1)

    assert found.IsExist()
    assert found.Text == "原神"
    assert len(multi) == 1
    assert multi[0].Text == "原神"


def test_recognition_json_loads_binary_and_replace_options(tmp_path):
    from bgi_touch.engine.recognition_assets import load_recognition_object

    path = tmp_path / "Recognition.json"
    path.write_text(json.dumps({
        "objects": {
            "Target": {
                "type": "TemplateMatch",
                "template": "target.png",
                "useBinaryMatch": True,
                "binaryThreshold": 200,
            },
            "Text": {
                "type": "Ocr",
                "replace": {"原神": ["原误", "原申"]},
                "ocrEngine": "Paddle",
            },
        },
    }), encoding="utf-8")

    template = np.zeros((4, 4, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "target.png"), template)

    binary = load_recognition_object(path, "Target")
    text = load_recognition_object(path, "Text")

    assert binary.UseBinaryMatch is True
    assert binary.BinaryThreshold == 200
    assert text.OcrEngine == "Paddle"
    assert text.ReplaceDictionary == {"原神": ["原误", "原申"]}
