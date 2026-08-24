from pathlib import Path
from unittest.mock import Mock


ROOT = Path(__file__).parents[1]


def _devicehub_profile():
    return {
        "version": 2,
        "name": "arbitrary-profile-labels",
        "mappings": [
            {"type": "Press", "bind": ["KeyB"], "position": {"x": 0.8296, "y": 0.0657}},
            {"type": "Press", "bind": ["KeyC"], "position": {"x": 0.8725, "y": 0.0671}},
            {"type": "Press", "bind": ["KeyX"], "position": {"x": 0.7961, "y": 0.7778}},
            {"type": "Press", "bind": ["ControlLeft"], "position": {"x": 0.9581, "y": 0.9506}},
            {"type": "Press", "bind": ["Enter"], "position": {"x": 0.0723, "y": 0.9211}},
            {"type": "Press", "bind": ["KeyY"], "position": {"x": 0.2283, "y": 0.0563}},
            {"type": "Press", "bind": ["KeyU"], "position": {"x": 0.6563, "y": 0.0605}},
            {"type": "Press", "bind": ["KeyG"], "position": {"x": 0.6338, "y": 0.4144}},
            {"type": "Press", "bind": ["KeyT"], "position": {"x": 0.7574, "y": 0.6742}},
        ],
    }


def test_default_pc_keys_override_arbitrary_profile_labels():
    from bgi_touch.input.layout import ControlLayout, DEFAULT_LAYOUT

    layout = ControlLayout.load(DEFAULT_LAYOUT, devicehub_profile=_devicehub_profile())

    # The source profile labels B/C/X do not carry PC semantics. The layout
    # intentionally redirects them to the profile touch points that do.
    assert layout.profile_key("B") == "KeyC"
    assert layout.profile_key("X") == "ControlLeft"
    assert layout.profile_key("C") is None
    assert layout.profile_key_for_button("characterMenu") is None
    assert layout.binding("B")["button"] == "inventory"
    assert layout.binding("C")["button"] == "characterMenu"
    assert layout.binding("T")["button"] == "contextInteraction"

    for key, raw in {
        "ENTER": "Enter",
        "Y": "KeyY",
        "U": "KeyU",
        "G": "KeyG",
        "T": "KeyT",
    }.items():
        assert layout.profile_key(key) == raw


def test_character_menu_uses_direct_touch_when_profile_has_no_semantic_key():
    from bgi_touch.input.layout import ControlLayout, DEFAULT_LAYOUT
    from bgi_touch.input.simulator import InputSimulator
    from bgi_touch.vision.coordinate import ScreenTransform

    layout = ControlLayout.load(DEFAULT_LAYOUT, devicehub_profile=_devicehub_profile())
    device = Mock()
    device.start_game_session.return_value = "session"
    simulator = InputSimulator(device, layout, ScreenTransform(2816, 1296))

    simulator.key_press("VK_C")

    device.start_game_session.assert_not_called()
    device.tap.assert_called_once_with(
        layout.buttons["characterMenu"][0] * 2816,
        layout.buttons["characterMenu"][1] * 1296,
        hold_ms=80,
        image_width=2816,
        image_height=1296,
    )


def test_inventory_and_drop_use_semantic_profile_codes():
    from bgi_touch.input.layout import ControlLayout, DEFAULT_LAYOUT
    from bgi_touch.input.simulator import InputSimulator
    from bgi_touch.vision.coordinate import ScreenTransform

    layout = ControlLayout.load(DEFAULT_LAYOUT, devicehub_profile=_devicehub_profile())
    device = Mock()
    device.start_game_session.return_value = "session"
    simulator = InputSimulator(device, layout, ScreenTransform(2816, 1296))

    simulator.key_press("B", hold_ms=25)
    simulator.key_press("X", hold_ms=25)

    assert [call.args[1] for call in device.set_game_input.call_args_list] == [
        ["KeyC"], [], ["ControlLeft"], [],
    ]
    device.tap.assert_not_called()
