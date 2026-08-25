import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest


def _context():
    from bgi_touch.vision.coordinate import ScreenTransform

    input_simulator = SimpleNamespace(
        key_down=Mock(), key_up=Mock(), key_press=Mock(), click_ref=Mock(),
        move_camera_by=Mock(), attack=Mock(), attack_down=Mock(), attack_up=Mock(),
        button_down=Mock(), button_up=Mock(), tap_button=Mock(), drag_ref=Mock(),
        release_all=Mock(),
    )
    return SimpleNamespace(
        input=input_simulator,
        device=SimpleNamespace(paste_text=Mock(), tap=Mock()),
        transform=ScreenTransform(1920, 1080),
        sleep=lambda _ms: None,
    )


def test_js_runtime_loads_es_modules_default_named_library_and_image(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    (tmp_path / "utils").mkdir()
    (tmp_path / "lib").mkdir()
    (tmp_path / "assets").mkdir()
    cv2.imwrite(str(tmp_path / "assets" / "icon.png"),
                np.zeros((3, 5, 3), dtype=np.uint8))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "main": "main.js", "library": [".", "lib"],
    }), encoding="utf-8")
    (tmp_path / "utils" / "math.js").write_text(
        """
export const base = 4;
export function add(a, b) { return a + b; }
function renamedSource() { return 'renamed'; }
export { renamedSource as renamed };
export default { label: 'module-default' };
""",
        encoding="utf-8",
    )
    (tmp_path / "lib" / "shared.js").write_text(
        "export const sharedValue = 9;", encoding="utf-8",
    )
    (tmp_path / "main.js").write_text(
        """
import defaults, {
  base,
  add,
  renamed
} from './utils/math.js';
import { sharedValue } from 'shared';
import icon from 'assets/icon.png';
(async function () {
  await sleep(1);
  return JSON.stringify({
    value: add(base, sharedValue), label: defaults.label,
    renamed: renamed(), width: icon.Width, height: icon.Height
  });
})();
""",
        encoding="utf-8",
    )

    result = json.loads(JsScriptRuntime(_context(), tmp_path).run())

    assert result == {
        "value": 13, "label": "module-default", "renamed": "renamed",
        "width": 5, "height": 3,
    }


def test_js_get_avatars_uses_cached_frame_before_device_capture(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    (tmp_path / "main.js").write_text(
        "return JSON.stringify(getAvatars());", encoding="utf-8",
    )
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ctx = _context()
    ctx.cached_frame = Mock(return_value=(frame, 0.05))
    ctx.capture_bgr = Mock(side_effect=AssertionError("不应创建第二个截图请求"))
    ctx.party_slots = {"钟离": 1}

    with patch(
        "bgi_touch.engine.party_hud.recognize_party_slots",
        return_value={"钟离": 1, "夜兰": 2, "纳西妲": 3, "芙宁娜": 4},
    ) as recognize:
        result = json.loads(JsScriptRuntime(
            ctx, tmp_path, party_slots={"钟离": 1},
        ).run())

    assert result == ["钟离", "夜兰", "纳西妲", "芙宁娜"]
    ctx.cached_frame.assert_called_once_with()
    ctx.capture_bgr.assert_not_called()
    assert recognize.call_count == 1
    region = recognize.call_args.args[1]
    assert region.bgr is frame


def test_js_runtime_replays_virtual_cursor_drag_and_mouse_virtual_keys(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    (tmp_path / "manifest.json").write_text(
        json.dumps({"main": "main.js"}), encoding="utf-8",
    )
    (tmp_path / "main.js").write_text(
        """
(async function () {
  setGameMetrics(3840, 2160, 2);
  moveMouseTo(800, 1500);
  leftButtonDown();
  moveMouseBy(0, -100);
  moveMouseTo(800, 1200);
  leftButtonUp();
  moveMouseTo(1000, 1000);
  keyPress('VK_LBUTTON');
  keyPress('W');
  keyPress('MBUTTON');
  return JSON.stringify(getGameMetrics());
})();
""",
        encoding="utf-8",
    )
    ctx = _context()

    result = json.loads(JsScriptRuntime(ctx, tmp_path).run())

    assert result == [3840, 2160, 2]
    drag = ctx.input.drag_ref.call_args
    assert drag.args[:4] == (400, 750, 400, 600)
    ctx.input.click_ref.assert_called_once_with(500, 500)
    ctx.input.key_press.assert_called_once_with("W")
    ctx.input.tap_button.assert_called_once_with("elementalSight")


def test_js_region_static_move_apis_share_virtual_pointer(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    (tmp_path / "manifest.json").write_text(
        json.dumps({"main": "main.js"}), encoding="utf-8",
    )
    (tmp_path / "main.js").write_text(
        """
(async function () {
  DesktopRegion.DesktopRegionMove(100, 200, 20, 40);
  leftButtonDown();
  DesktopRegion.DesktopRegionMoveBy(0, 100);
  leftButtonUp();
  GameCaptureRegion.GameRegion1080PPosMove(300, 400);
  keyPress('LBUTTON');
})();
""",
        encoding="utf-8",
    )
    ctx = _context()

    JsScriptRuntime(ctx, tmp_path).run()

    drag = ctx.input.drag_ref.call_args
    assert drag.args[:4] == (110, 220, 110, 320)
    ctx.input.click_ref.assert_called_once_with(300, 400)


def test_js_module_loader_rejects_path_outside_sandbox(tmp_path):
    pytest.importorskip("pythonmonkey")
    import pythonmonkey as pm

    from bgi_touch.engine.js_modules import JsModuleLoader

    package = tmp_path / "package"
    package.mkdir()
    outside = tmp_path / "outside.js"
    outside.write_text("export const secret = 1;", encoding="utf-8")
    loader = JsModuleLoader(pm, package, {}, lambda value: value)

    with pytest.raises(FileNotFoundError, match="无法解析 JS 模块"):
        loader.require("../outside.js", package / "main.js")


def test_converter_vendors_shared_bettergi_modules_and_runtime_loads_them(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.converter.convert import convert_js_package
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    root = tmp_path / "community"
    package = root / "repo" / "js" / "Demo"
    shared = root / "packages" / "utils"
    package.mkdir(parents=True)
    shared.mkdir(parents=True)
    (package / "manifest.json").write_text(json.dumps({
        "name": "Demo", "main": "main.js",
    }), encoding="utf-8")
    (package / "main.js").write_text(
        """
import { sharedValue } from '../../../packages/utils/tool';
(async function () { return String(sharedValue); })();
""",
        encoding="utf-8",
    )
    (shared / "tool.js").write_text(
        "export const sharedValue = 42;", encoding="utf-8",
    )

    destination, info = convert_js_package(package, tmp_path / "converted")

    assert info["vendored_imports"] == 1
    assert (destination / ".bgi-touch-vendor" / "packages" /
            "utils" / "tool.js").is_file()
    main = (destination / "main.js").read_text(encoding="utf-8")
    assert ".bgi-touch-vendor/packages/utils/tool.js" in main
    assert JsScriptRuntime(_context(), destination).run() == "42"
