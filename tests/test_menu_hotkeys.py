import json
from pathlib import Path
from unittest.mock import Mock


ROOT = Path(__file__).parents[1]


def _profile():
    return {
        "version": 2,
        "name": "menu-profile",
        "mappings": [
            {"type": "Press", "bind": ["KeyB"], "position": {"x": 0.8296, "y": 0.0657}},
            {"type": "Press", "bind": ["KeyP"], "position": {"x": 0.7846, "y": 0.0655}},
            {"type": "Press", "bind": ["KeyO"], "position": {"x": 0.7417, "y": 0.0637}},
        ],
    }


def test_default_layout_maps_stable_bettergi_menu_hotkeys():
    from bgi_touch.input.layout import ControlLayout, DEFAULT_LAYOUT

    layout = ControlLayout.load(DEFAULT_LAYOUT, devicehub_profile=_profile())

    assert layout.binding("F1") == {
        "type": "button", "button": "adventurerHandbook", "profileCode": "KeyB",
    }
    assert layout.binding("VK_F3") == {
        "type": "button", "button": "wishMenu", "profileCode": "KeyP",
    }
    assert layout.profile_key("F1") == "KeyB"
    assert layout.profile_key("F3") == "KeyP"
    assert layout.profile_key("F5") == "KeyO"


def test_menu_hotkeys_fall_back_to_calibrated_touch_without_profile():
    from bgi_touch.input.layout import ControlLayout, DEFAULT_LAYOUT
    from bgi_touch.input.simulator import InputSimulator
    from bgi_touch.vision.coordinate import ScreenTransform

    layout = ControlLayout.load(DEFAULT_LAYOUT)
    device = Mock()
    simulator = InputSimulator(device, layout, ScreenTransform(2816, 1296))

    simulator.key_press("F1")
    simulator.key_press("VK_F3")

    assert device.tap.call_count == 2
    first = device.tap.call_args_list[0]
    second = device.tap.call_args_list[1]
    assert first.args == (
        layout.buttons["adventurerHandbook"][0] * 2816,
        layout.buttons["adventurerHandbook"][1] * 1296,
    )
    assert second.args == (
        layout.buttons["wishMenu"][0] * 2816,
        layout.buttons["wishMenu"][1] * 1296,
    )


def test_keymouse_converter_retains_all_bettergi_function_key_codes():
    from bgi_touch.macro.keymouse import convert_keymouse

    macro = {
        "macroEvents": [
            {"type": 0, "time": index * 10, "keyCode": code}
            for index, code in enumerate(range(112, 120))
        ],
    }

    events, warnings = convert_keymouse(macro)

    assert not warnings
    assert [event.key for event in events] == [f"F{index}" for index in range(1, 9)]
    assert all(event.kind == "keyDown" for event in events)


def test_layout_json_keeps_profile_coordinates_in_sync():
    raw = json.loads(
        (ROOT / "config" / "controls" / "genshin-default.json").read_text(encoding="utf-8")
    )

    assert raw["buttons"]["adventurerHandbook"] == {
        "nx": 0.8296257062146892,
        "ny": 0.06572637686719256,
    }
    assert raw["buttons"]["wishMenu"] == {
        "nx": 0.7845852092161016,
        "ny": 0.06554655093869134,
    }
