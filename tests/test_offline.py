"""无设备回归测试：解析器、坐标系、宏转换、（有资产时）地图定位。

运行：.venv/bin/python -m pytest tests/ -q
"""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
ASSETS = Path(__file__).parents[1] / "assets" / "map"


# ---- 坐标系 ----

def test_screen_transform_anchors():
    from bgi_touch.vision.coordinate import ScreenTransform
    t = ScreenTransform(2816, 1296)
    assert t.scale == pytest.approx(1.2)
    # 左锚点：贴左边元素
    x, y = t.to_device(62, 19)
    assert x == pytest.approx(62 * 1.2)
    # 右锚点：贴右边元素随设备加宽外移
    x, _ = t.to_device(1870, 50)
    assert x == pytest.approx(2816 - (1920 - 1870) * 1.2)
    # 中锚点往返
    rx, ry = t.to_ref(*t.to_device(960, 540))
    assert (rx, ry) == (pytest.approx(960), pytest.approx(540))


def test_teyvat_world_image_roundtrip():
    from bgi_touch.pathing.map_locator import MapConfig
    c = MapConfig()
    ix, iy = c.world_to_image(0, 0)
    assert (ix, iy) == (32768, 16384)
    wx, wy = c.image_to_world(*c.world_to_image(-910.5, 2249.9))
    assert wx == pytest.approx(-910.5)
    assert wy == pytest.approx(2249.9)


# ---- 战斗 DSL ----

def test_combat_parser():
    from bgi_touch.combat.dsl import parse_combat_script
    lines = parse_combat_script(
        "// c\n钟离 s(0.2), e(hold), wait(0.2), click(middle)\nwait(1),keypress(VK_SPACE)\n")
    assert lines[0].character == "钟离"
    assert [c.action for c in lines[0].commands] == ["s", "e", "wait", "click"]
    assert lines[0].commands[1].params == ["hold"]
    assert lines[1].character is None
    assert lines[1].commands[1].params == ["VK_SPACE"]


def test_combat_executor_maps_keys():
    from unittest.mock import MagicMock
    from bgi_touch.combat.dsl import CombatExecutor
    inp = MagicMock()
    ex = CombatExecutor(inp, sleep=lambda ms: None)
    ex.run("e, q, jump, dash, w(0.01)")
    inp.key_press.assert_any_call("E", hold_ms=80)
    inp.key_press.assert_any_call("Q")
    inp.key_press.assert_any_call("SPACE")
    inp.key_down.assert_called_with("W")
    inp.key_up.assert_called_with("W")


# ---- 键鼠宏转换 ----

def test_keymouse_conversion():
    from bgi_touch.macro.keymouse import convert_keymouse
    macro = {
        "info": {"width": 1920, "height": 1080},
        "macroEvents": [
            {"type": 0, "keyCode": 87, "time": 0},
            {"type": 3, "mouseX": 50, "mouseY": 0, "time": 10},
            {"type": 3, "mouseX": 50, "mouseY": 0, "time": 60},
            {"type": 1, "keyCode": 87, "time": 500},
            {"type": 2, "mouseX": 960, "mouseY": 540, "time": 600},
            {"type": 4, "mouseButton": "Left", "time": 620},
            {"type": 5, "mouseButton": "Left", "time": 700},
        ],
    }
    events, warnings = convert_keymouse(macro)
    kinds = [e.kind for e in events]
    assert kinds == ["keyDown", "cameraBy", "keyUp", "tapRef"]
    cam = events[1]
    assert cam.x == pytest.approx(100)  # 两次相对移动合并
    assert not warnings


# ---- 键位规范化 ----

def test_key_normalization():
    from bgi_touch.input.layout import normalize_key
    assert normalize_key("VK_W") == "W"
    assert normalize_key("LeftShift") == "LSHIFT"
    assert normalize_key(" ") == "SPACE"
    assert normalize_key("esc") == "ESCAPE"


# ---- 地图定位（需要资产与夹具，缺则跳过）----

@pytest.mark.skipif(not (ASSETS / "Teyvat" / "Teyvat_0_2048_SIFT.kp.bin").exists(),
                    reason="地图资产未下载")
def test_minimap_localization_mondstadt():
    import cv2
    from bgi_touch.pathing.map_locator import MapLocator
    mm = cv2.imread(str(FIXTURES / "minimap-mondstadt.png"), cv2.IMREAD_GRAYSCALE)
    assert mm is not None
    loc = MapLocator()
    p = loc.locate_pixel(mm)
    assert p is not None
    wx, wy = loc.config.image_to_world(*p)
    # 录制时角色在蒙德城（约 -910, 2250）
    assert wx == pytest.approx(-910, abs=15)
    assert wy == pytest.approx(2250, abs=15)
