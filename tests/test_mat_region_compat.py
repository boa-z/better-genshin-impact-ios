import json
from types import SimpleNamespace

import numpy as np
import pytest


def _context(frame):
    from bgi_touch.vision.coordinate import ScreenTransform

    return SimpleNamespace(
        transform=ScreenTransform(frame.shape[1], frame.shape[0]),
        capture_bgr=lambda: frame.copy(),
        input=SimpleNamespace(),
        device=SimpleNamespace(),
        sleep=lambda _ms: None,
    )


def test_mat_pixel_overloads_mutation_and_cached_region_views():
    from bgi_touch.engine.recognition import ImageRegion, Mat, Region

    frame = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=np.uint8,
    )
    mat = Mat(frame.copy())
    assert mat.Get(object(), 1, 0) == {
        "Item0": 7, "Item1": 8, "Item2": 9,
        "item0": 7, "item1": 8, "item2": 9,
    }
    assert mat.Get(1, 0)["Item2"] == 9
    assert mat.Get(2)["Item0"] == 7
    assert mat.At(0, 1)["item1"] == 5
    mat.Set(0, 1, {"Item0": 20, "Item1": 21, "Item2": 22})
    assert mat.Get(0, 1)["Item0"] == 20
    mat.Set(2, [30, 31, 32])
    assert mat.Get(1, 0)["Item2"] == 32
    assert mat.Clone().Get(0, 1)["Item0"] == 20
    assert mat.Add(1).Get(0, 0)["Item0"] == 2
    assert mat.OnesComplement().Get(0, 0)["Item0"] == 254

    ctx = _context(frame)
    image = ImageRegion(ctx, frame)
    assert image.SrcMat is image.SrcMat
    assert image.CacheGreyMat is image.CacheGreyMat
    assert image.CacheGreyMat.channels() == 1
    assert image.CacheImage is image.SrcMat
    result = Region(ctx, 1, 0, 1, 1, prev=image)
    cropped = result.ToImageRegion()
    assert cropped.SrcMat.Get(0, 0)["Item0"] == 4
    assert cropped.SrcMat.Get(0, 0)["Item2"] == 6
    assert cropped.Prev is None


def test_mat_and_image_region_properties_are_script_compatible(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    frame = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8,
    )
    (tmp_path / "main.js").write_text(
        """
const image = captureGameRegion();
const mat = image.SrcMat;
const typed = mat.Get(OpenCvSharp.OpenCvSharp.Vec3b, 1, 0);
const direct = mat.Get(1, 0);
const grey = image.CacheGreyMat;
return JSON.stringify({
  typed: [typed.Item0, typed.Item1, typed.Item2],
  direct: [direct.Item0, direct.Item1, direct.Item2],
  grey: [grey.Rows, grey.Cols, grey.Channels(), grey.Get(1, 0)],
  cached: image.SrcMat.Rows === image.SrcMat.Rows && image.CacheGreyMat.Rows === image.CacheGreyMat.Rows,
  empty: mat.Empty()
});
""",
        encoding="utf-8",
    )
    result = json.loads(JsScriptRuntime(
        _context(frame), tmp_path, log=lambda _msg: None,
    ).run())
    assert result == {
        "typed": [7, 8, 9],
        "direct": [7, 8, 9],
        "grey": [2, 2, 1, 8],
        "cached": True,
        "empty": False,
    }
