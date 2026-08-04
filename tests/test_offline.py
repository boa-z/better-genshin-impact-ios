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
    assert normalize_key("KeyE") == "E"
    assert normalize_key("Digit4") == "4"


def test_devicehub_profile_maps_native_codes_to_bettergi_keys():
    from bgi_touch.input.layout import ControlLayout, DeviceHubProfile

    profile = {
        "version": 2,
        "name": "test-profile",
        "mappings": [
            {"type": "Press", "bind": ["Space"], "position": {"x": 0.88, "y": 0.66}},
            {"type": "Press", "bind": ["ShiftLeft"], "position": {"x": 0.88, "y": 0.86}},
            {"type": "Press", "bind": ["KeyE"], "position": {"x": 0.72, "y": 0.89}},
            {"type": "Press", "bind": ["KeyR"], "position": {"x": 0.697, "y": 0.746}},
            {"type": "Press", "bind": ["KeyX"], "position": {"x": 0.795, "y": 0.778}},
            {
                "type": "DirectionPad",
                "bind": {"up": ["KeyW"], "down": ["KeyS"],
                         "left": ["KeyA"], "right": ["KeyD"]},
                "position": {"x": 0.18, "y": 0.75},
            },
        ],
    }
    layout = ControlLayout.load(devicehub_profile=profile)

    assert isinstance(layout.devicehub_profile, DeviceHubProfile)
    assert layout.profile_key("W") == "KeyW"
    assert layout.profile_key("E") == "KeyE"
    # The supplied profile deliberately places Space on sprint and ShiftLeft on jump.
    assert layout.profile_key("SPACE") == "ShiftLeft"
    assert layout.profile_key("LSHIFT") == "Space"
    assert layout.profile_key_for_button("aim") == "KeyR"
    assert layout.profile_key_for_button("attack") == "KeyX"

    class FakeDevice:
        def __init__(self):
            self.inputs = []
            self.stopped = []

        def start_game_session(self, profile_name, **kwargs):
            assert profile_name == "test-profile"
            return "session-1"

        def set_game_input(self, session_id, keys, **kwargs):
            self.inputs.append((session_id, list(keys)))

        def stop_game_session(self, session_id):
            self.stopped.append(session_id)

    from bgi_touch.input.simulator import InputSimulator
    from bgi_touch.vision.coordinate import ScreenTransform
    device = FakeDevice()
    simulator = InputSimulator(device, layout, ScreenTransform(1920, 1080))
    simulator.key_down("W")
    simulator.key_down("LSHIFT")
    simulator.key_up("W")
    simulator.key_press("E", hold_ms=25)
    simulator.key_press("X", hold_ms=25)
    simulator.attack_down()
    simulator.attack_up()
    assert device.inputs[:3] == [
        ("session-1", ["KeyW"]),
        ("session-1", ["KeyW", "Space"]),
        ("session-1", ["Space"]),
    ]
    assert ("session-1", ["Space", "KeyE"]) in device.inputs
    assert ("session-1", ["Space", "KeyX"]) in device.inputs
    assert device.inputs[-2:] == [
        ("session-1", ["Space", "KeyX"]),
        ("session-1", ["Space"]),
    ]
    simulator.release_all()
    assert device.stopped == ["session-1"]


def test_native_ui_layout_overlay_changes_space_semantics():
    from bgi_touch.input.layout import ControlLayout

    layout = ControlLayout.load(Path(__file__).parents[1] / "config" / "controls" / "genshin-native-ui.json")
    assert layout.binding("SPACE")["button"] == "jump"
    assert layout.binding("SPACE")["profileCode"] == "Space"


def test_auto_cook_color_peak_detector():
    import cv2
    import numpy as np

    from bgi_touch.tasks.auto_cook import count_target_color, update_peak

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # TargetCookColor is RGB (255, 192, 64), while the fixture is BGR.
    frame[700:720, 700:900] = (64, 192, 255)
    assert count_target_color(frame, rect=(600, 660, 730, 190)) == 4000
    candidate = stable = None
    built = None
    for value in (1000, 1005, 995):
        candidate, stable, built = update_peak(value, candidate, stable or 0)
    assert built == 1005


def test_auto_fishing_bar_detector_and_controller():
    import numpy as np

    from bgi_touch.tasks.auto_fishing import fish_bar_action, get_fish_bar_rects

    frame = np.zeros((240, 800, 3), dtype=np.uint8)
    yellow = (192, 255, 255)  # RGB (255, 255, 192), HSV_FULL hue≈60°
    frame[100:120, 120:140] = yellow
    frame[100:120, 180:580] = yellow
    rects = get_fish_bar_rects(frame)
    assert len(rects) == 2
    assert fish_bar_action(rects) == "hold"


def test_task_dispatcher_declares_migrated_core_tasks():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    assert TaskDispatcher.IMPLEMENTED >= {
        "AutoFight", "AutoWood", "AutoDomain", "AutoCook", "AutoFishing", "AutoOpenChest"
    }
    with pytest.raises(NotImplementedError, match="尚未移植"):
        TaskDispatcher(object()).run_task({"name": "AutoBoss", "config": {}})


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
