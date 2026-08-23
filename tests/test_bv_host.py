import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import cv2
import numpy as np
import pytest


def test_js_bv_page_template_locator_and_rect(tmp_path):
    pytest.importorskip("pythonmonkey")

    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    rng = np.random.default_rng(7)
    template = rng.integers(0, 255, (10, 12, 3), dtype=np.uint8)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[200:210, 300:312] = template
    cv2.imwrite(str(tmp_path / "template.png"), template)
    (tmp_path / "main.js").write_text(
        """
const Rect = OpenCvSharp.OpenCvSharp.Rect;
const roi = new Rect(250, 150, 200, 150);
const page = new BvPage();
page.DefaultTimeout = 20;
page.DefaultRetryInterval = 5;
const ro = RecognitionObject.TemplateMatch(file.ReadImageMatSync("template.png"));
ro.RegionOfInterest = [roi.X, roi.Y, roi.Width, roi.Height];
const locator = page.Locator(ro).WithTimeout(20).WithRetryInterval(5);
const exists = await locator.IsExist();
const all = locator.FindAll();
await locator.Click();
page.Click(25, 35);
return JSON.stringify({ exists, count: all.Count, first: [all[0].X, all[0].Y] });
""",
        encoding="utf-8",
    )
    input_simulator = SimpleNamespace(
        click_ref=Mock(), key_down=Mock(), key_up=Mock(), key_press=Mock(),
        move_camera_by=Mock(), attack=Mock(), attack_down=Mock(), attack_up=Mock(),
        button_down=Mock(), button_up=Mock(),
    )
    device = SimpleNamespace(paste_text=Mock(), tap=Mock())
    ctx = SimpleNamespace(
        input=input_simulator,
        device=device,
        transform=ScreenTransform(1920, 1080),
        capture_bgr=lambda: frame.copy(),
        sleep=lambda _ms: None,
    )

    result = json.loads(JsScriptRuntime(ctx, tmp_path).run())

    assert result == {"exists": True, "count": 1, "first": [300, 200]}
    device.tap.assert_called_once()
    input_simulator.click_ref.assert_called_once_with(25.0, 35.0)


def test_bv_locator_retries_without_extra_capture_per_attempt():
    from bgi_touch.engine.bv import BvLocator
    from bgi_touch.engine.recognition import Mat, RecognitionObject
    from bgi_touch.vision.coordinate import ScreenTransform

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    template = np.zeros((8, 8, 3), dtype=np.uint8)
    template[1:7, 2:6] = (20, 180, 250)
    template[3:5, :] = (255, 30, 10)
    captures = Mock(side_effect=[frame.copy(), frame.copy()])
    ctx = SimpleNamespace(
        capture_bgr=captures,
        transform=ScreenTransform(1920, 1080),
        sleep=Mock(),
        device=SimpleNamespace(tap=Mock()),
    )
    locator = BvLocator(ctx, RecognitionObject.template_match(Mat(template)))
    locator.with_timeout(10).with_retry_interval(5)

    with pytest.raises(TimeoutError):
        locator.wait_for()

    assert captures.call_count == 2
    ctx.sleep.assert_called_once_with(5)


def test_js_bv_image_desktop_region_and_vec3b_host_types(tmp_path):
    pytest.importorskip("pythonmonkey")

    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    rng = np.random.default_rng(11)
    template = rng.integers(0, 255, (9, 13, 3), dtype=np.uint8)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[310:319, 410:423] = template
    asset_dir = tmp_path / "assets" / "Demo"
    asset_dir.mkdir(parents=True)
    cv2.imwrite(str(asset_dir / "pixel.png"), template)
    (tmp_path / "main.js").write_text(
        """
const Rect = OpenCvSharp.OpenCvSharp.Rect;
const image = new BvImage('Demo:pixel.png', new Rect(350, 250, 200, 150), 0.9);
const page = new BvPage();
const found = page.Locator(image).FindAll();
const mat = file.ReadImageMatSync('assets/Demo/pixel.png');
const pixel = mat.Get(OpenCvSharp.OpenCvSharp.Vec3b, 0, 0);
const desktop = new DesktopRegion();
desktop.DesktopRegionClick(10, 20, 30, 40);
DesktopRegion.DesktopRegionClick(100, 200, 20, 10);
const capture = desktop.Derive(mat, 25, 35);
const pen = new Pen(Color.Coral, 2.5);
const customColor = Color.FromArgb(10, 20, 30);
return JSON.stringify({
  count: found.Count,
  point: [found[0].X, found[0].Y],
  pixel: [pixel.Item0, pixel.Item1, pixel.Item2],
  capture: [capture.X, capture.Y, capture.Width, capture.Height],
  assetName: image.RecognitionObject.Name,
  pen: [pen.Color.Name, pen.Width],
  customColor: [customColor.A, customColor.R, customColor.G, customColor.B]
});
""",
        encoding="utf-8",
    )
    input_simulator = SimpleNamespace(
        click_ref=Mock(), key_down=Mock(), key_up=Mock(), key_press=Mock(),
        move_camera_by=Mock(), attack=Mock(), attack_down=Mock(), attack_up=Mock(),
        button_down=Mock(), button_up=Mock(), release_all=Mock(),
    )
    device = SimpleNamespace(paste_text=Mock(), tap=Mock())
    ctx = SimpleNamespace(
        input=input_simulator,
        device=device,
        transform=ScreenTransform(1920, 1080),
        capture_bgr=lambda: frame.copy(),
        sleep=lambda _ms: None,
    )

    result = json.loads(JsScriptRuntime(ctx, tmp_path).run())

    assert result["count"] == 1
    assert result["point"] == pytest.approx([410, 310])
    assert result["pixel"] == template[0, 0].tolist()
    assert result["capture"] == pytest.approx([25, 35, 13, 9])
    assert result["assetName"] == "Demo:pixel.png"
    assert result["pen"] == ["Coral", 2.5]
    assert result["customColor"] == [255, 10, 20, 30]
    assert input_simulator.click_ref.call_args_list == [call(110.0, 205.0)]
    device.tap.assert_called_once_with(
        25.0, 40.0, image_width=1920, image_height=1080,
    )
