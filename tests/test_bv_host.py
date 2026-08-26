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


def test_bv_locator_consumes_fresh_trigger_frame_without_new_capture():
    from bgi_touch.engine.bv import BvLocator
    from bgi_touch.engine.recognition import Mat, RecognitionObject
    from bgi_touch.vision.coordinate import ScreenTransform

    template = np.zeros((8, 8, 3), dtype=np.uint8)
    template[1:7, 2:6] = (20, 180, 250)
    template[3:5, :] = (255, 30, 10)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[200:208, 300:308] = template
    cached = Mock(return_value=(frame.copy(), 0.2))
    capture = Mock(side_effect=AssertionError("新鲜触发器帧不应再次截图"))
    ctx = SimpleNamespace(
        _trigger_loop=SimpleNamespace(active=True),
        cached_frame=cached,
        capture_bgr=capture,
        transform=ScreenTransform(1920, 1080),
    )

    locator = BvLocator(ctx, RecognitionObject.template_match(Mat(template)))
    found = locator.find_all()

    assert len(found) == 1
    assert (found[0].x, found[0].y) == (300, 200)
    cached.assert_called_once_with()
    capture.assert_not_called()


def test_bv_page_falls_back_to_capture_when_trigger_frame_is_stale():
    from bgi_touch.engine.bv import BvPage
    from bgi_touch.vision.coordinate import ScreenTransform

    cached = np.zeros((1080, 1920, 3), dtype=np.uint8)
    direct = np.full_like(cached, 17)
    cached_frame = Mock(return_value=(cached, 2.0))
    capture = Mock(return_value=direct)
    ctx = SimpleNamespace(
        _trigger_loop=SimpleNamespace(active=True),
        cached_frame=cached_frame,
        capture_bgr=capture,
        transform=ScreenTransform(1920, 1080),
        input=SimpleNamespace(),
    )

    screen = BvPage(ctx).screenshot()

    assert screen.bgr is direct
    cached_frame.assert_called_once_with()
    capture.assert_called_once_with()


def test_bv_rect_cut_helpers_and_function_roi_match_bettergi_geometry():
    from bgi_touch.engine.bv import BvLocator, Rect
    from bgi_touch.engine.recognition import RecognitionObject

    rect = Rect(100, 50, 800, 400)
    assert (rect.CutLeft(0.25).X, rect.CutLeft(0.25).Y,
            rect.CutLeft(0.25).Width, rect.CutLeft(0.25).Height) == (
                100, 50, 200, 400,
            )
    right_bottom = rect.CutRightBottom(0.25, 0.5)
    assert (right_bottom.X, right_bottom.Y,
            right_bottom.Width, right_bottom.Height) == (
                700, 250, 200, 200,
            )

    locator = BvLocator(SimpleNamespace(), RecognitionObject())
    locator.WithRoi(lambda capture: capture.CutRight(0.25))
    roi = locator.RecognitionObject.RegionOfInterest
    assert (roi["X"], roi["Y"], roi["Width"], roi["Height"]) == (
        1440.0, 0.0, 480.0, 1080.0,
    )


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
const found = page.GetByImage(image).FindAll();
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


def test_js_bv_page_exposes_roi_callback_and_chainable_keyboard_mouse(tmp_path):
    pytest.importorskip("pythonmonkey")

    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    (tmp_path / "main.js").write_text(
        """
const page = new BvPage();
page.Keyboard.KeyDown('VK_W').KeyUp('VK_W');
page.Keyboard.ModifiedKeyStroke(['VK_LCTRL'], ['VK_V']).TextEntry('hello').Sleep(0);
page.Mouse.MoveMouseTo(10, 20).LeftButtonDown().MoveMouseBy(20, 0).LeftButtonUp();
page.Mouse.VerticalScroll(2);
const roi = page.GetByText('target').WithRoi(r => r.CutRightBottom(0.25, 0.5))
  .RecognitionObject.RegionOfInterest;
return JSON.stringify({roi: [roi.X, roi.Y, roi.Width, roi.Height]});
""",
        encoding="utf-8",
    )
    input_simulator = SimpleNamespace(
        _active_slot=1,
        key_down=Mock(), key_up=Mock(), key_press=Mock(),
        move_camera_by=Mock(), drag_ref=Mock(), vertical_scroll=Mock(),
        attack=Mock(), attack_down=Mock(), attack_up=Mock(),
        button_down=Mock(), button_up=Mock(), tap_button=Mock(),
        release_all=Mock(),
    )
    device = SimpleNamespace(paste_text=Mock(), tap=Mock())
    ctx = SimpleNamespace(
        input=input_simulator,
        device=device,
        transform=ScreenTransform(1920, 1080),
        capture_bgr=lambda: np.zeros((1080, 1920, 3), dtype=np.uint8),
        sleep=Mock(),
    )

    result = json.loads(JsScriptRuntime(ctx, tmp_path).run())

    assert result["roi"] == [1440, 540, 480, 540]
    assert input_simulator.key_down.call_args_list == [call("VK_W"), call("VK_LCTRL")]
    assert input_simulator.key_up.call_args_list == [call("VK_W"), call("VK_LCTRL")]
    assert input_simulator.key_press.call_args_list == [call("VK_V")]
    assert input_simulator.drag_ref.call_args[0] == (10.0, 20.0, 30.0, 20.0)
    input_simulator.vertical_scroll.assert_called_once_with(2.0)
    device.paste_text.assert_called_once_with("hello")
