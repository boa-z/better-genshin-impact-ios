import json

import cv2
import numpy as np
import pytest


def test_load_recognition_json_resolves_aliases_expressions_and_search(tmp_path):
    from bgi_touch.engine.recognition_assets import load_recognition_object

    config = {
        "version": 1,
        "vars": {"top": "ch / 10", "boxWidth": "cw / 4"},
        "regions": {
            "target": "rect(cw - boxWidth, top, boxWidth, ch / 2)",
        },
        "objects": {
            "Target": {
                "type": "Ocr",
                "roi": "@target",
                "reference": {
                    "size": [1920, 1080],
                    "bbox": "rect(1680, 32, 48, 48)",
                },
                "search": {
                    "anchor": "TopRight",
                    "box": "rect(1540, 0, 380, 160)",
                    "expand": [12, 13],
                    "expandPercent": [0.02, 0.01, 0.03, 0.01],
                },
                "text": "目标",
            },
        },
    }
    path = tmp_path / "Recognition.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    ro = load_recognition_object(path, "Target", capture_width=1920, capture_height=1080)

    # An explicit roi keeps its BetterGI precedence over reference search.
    assert ro.RegionOfInterest == {
        "x": 1440, "y": 108, "width": 480, "height": 540,
        "X": 1440, "Y": 108, "Width": 480, "Height": 540,
    }
    assert (ro.ReferenceImageSize.Width, ro.ReferenceImageSize.Height) == (1920, 1080)
    assert ro.ReferenceBoundingBox["X"] == 1680
    assert ro.SearchOptions.AnchorMode == "TopRight"
    assert ro.SearchOptions.ReferenceSearchBox["Width"] == 380
    # expandPercent intentionally wins over pixel expand.
    assert tuple(ro.SearchOptions.ExpandPercent) == (0.02, 0.01, 0.03, 0.01)
    assert ro.Text == "目标"


def test_load_recognition_json_template_alias_and_match_mode(tmp_path):
    from bgi_touch.engine.recognition import ImageRegion
    from bgi_touch.engine.recognition_assets import load_recognition_object
    from bgi_touch.vision.coordinate import ScreenTransform
    from types import SimpleNamespace
    from unittest.mock import Mock

    template = np.zeros((12, 10, 3), dtype=np.uint8)
    template[2:10, 3:8] = (20, 170, 240)
    cv2.imwrite(str(tmp_path / "target.png"), template)
    config = {
        "templates": {"target": "target.png"},
        "objects": {
            "Target": {
                "type": "TemplateMatch",
                "template": "@target",
                "templateMatchMode": "CCorrNormed",
                "use3Channels": True,
                "threshold": 0.99,
                "reference": {
                    "size": [1920, 1080],
                    "bbox": "rect(300, 200, 10, 12)",
                },
                "search": {"expandPercent": [0]},
            },
        },
    }
    path = tmp_path / "Recognition.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[200:212, 300:310] = template
    ctx = SimpleNamespace(
        transform=ScreenTransform(1920, 1080),
        device=SimpleNamespace(tap=Mock()),
        sleep=lambda _ms: None,
    )

    ro = load_recognition_object(path, "Target")
    found = ImageRegion(ctx, frame).find(ro)

    assert ro.TemplateMatchMode == cv2.TM_CCORR_NORMED
    assert found.is_exist()
    assert (found.dx, found.dy) == (300, 200)


@pytest.mark.parametrize("values", [[], [0.1, 0.2, 0.3], [0.1] * 5, [-0.1]])
def test_load_recognition_json_rejects_invalid_expand_percent(tmp_path, values):
    from bgi_touch.engine.recognition_assets import RecognitionJsonError, load_recognition_object

    path = tmp_path / "Recognition.json"
    path.write_text(json.dumps({
        "objects": {
            "Target": {
                "type": "Ocr",
                "reference": {"size": [100, 100], "bbox": "rect(1, 1, 2, 2)"},
                "search": {"expandPercent": values},
            },
        },
    }), encoding="utf-8")

    with pytest.raises(RecognitionJsonError):
        load_recognition_object(path, "Target")


def test_recognition_asset_store_caches_and_clears_by_task(tmp_path):
    from bgi_touch.engine.recognition_assets import RecognitionAssetStore

    task = tmp_path / "Demo" / "Assets"
    task.mkdir(parents=True)
    (task / "Recognition.json").write_text(json.dumps({
        "objects": {"Target": {"type": "Ocr", "roi": "rect(0, 0, cw, ch)"}},
    }), encoding="utf-8")
    store = RecognitionAssetStore(tmp_path)

    first = store.get("Demo", "Target", 1280, 720)
    second = store.get("Demo", "Target", 1280, 720)
    assert first is second
    store.clear_task("Demo")
    assert store.get("Demo", "Target", 1280, 720) is not first
