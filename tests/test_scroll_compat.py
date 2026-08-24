from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_input_vertical_scroll_splits_large_bursts_in_safe_region():
    from bgi_touch.input.layout import ControlLayout, DEFAULT_LAYOUT
    from bgi_touch.input.simulator import InputSimulator
    from bgi_touch.vision.coordinate import ScreenTransform

    device = Mock()
    simulator = InputSimulator(
        device, ControlLayout.load(DEFAULT_LAYOUT), ScreenTransform(1920, 1080),
    )
    events = []
    simulator.subscribe(events.append)
    simulator.vertical_scroll(-4)

    assert device.swipe.call_count == 2
    for call_value in device.swipe.call_args_list:
        x1, y1, x2, y2 = call_value.args[:4]
        assert x1 == pytest.approx(1920 * 0.62)
        assert x2 == x1
        assert y2 < y1  # wheel-down => finger swipes upward
        assert 0 < y2 < y1 < 1080
    assert events[-1]["type"] == "vertical_scroll"
    assert events[-1]["amount"] == -4


def test_keymouse_wheel_events_are_combined_into_scroll_clicks():
    from bgi_touch.macro.keymouse import convert_keymouse

    events, warnings = convert_keymouse({
        "macroEvents": [
            {"type": 6, "mouseY": -120, "time": 0},
            {"type": 6, "mouseY": -120, "time": 30},
            {"type": 6, "mouseY": 0, "time": 40},
            {"type": 6, "mouseY": 300, "time": 500},
        ],
    })
    assert warnings == []
    assert [(event.kind, event.amount, event.t) for event in events] == [
        ("verticalScroll", -2, 0),
        ("verticalScroll", 2.5, 500),
    ]


def test_macro_player_replays_wheel_as_touch_scroll():
    from bgi_touch.macro.keymouse import MacroPlayer

    input_sim = SimpleNamespace(vertical_scroll=Mock(), release_all=Mock())
    MacroPlayer(input_sim, sleep=lambda _ms: None).play({
        "macroEvents": [{"type": 6, "mouseY": -300, "time": 0}],
    })
    input_sim.vertical_scroll.assert_called_once_with(-2.5)
    input_sim.release_all.assert_called_once_with()


def test_combat_dsl_and_avatar_scroll_share_input_implementation():
    from bgi_touch.combat.dsl import CombatExecutor, parse_combat_script
    from bgi_touch.engine.combat_host import Avatar

    input_sim = SimpleNamespace(vertical_scroll=Mock())
    executor = CombatExecutor(input_sim, sleep=lambda _ms: None)
    executor.exec(parse_combat_script("scroll(-3)")[0].commands[0])
    input_sim.vertical_scroll.assert_called_once_with(-3)

    input_sim.vertical_scroll.reset_mock()
    avatar = Avatar(SimpleNamespace(ctx=SimpleNamespace(input=input_sim)), "可莉", 1)
    avatar.scroll(2)
    input_sim.vertical_scroll.assert_called_once_with(2.0)


def test_js_vertical_scroll_calls_touch_input(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    avatar_macro = tmp_path / "avatar.json"
    avatar_macro.write_text("[]", encoding="utf-8")
    (tmp_path / "main.js").write_text(
        "verticalScroll(-2); return true;", encoding="utf-8",
    )
    input_sim = SimpleNamespace(
        _active_slot=1, vertical_scroll=Mock(), key_up=Mock(), attack_up=Mock(),
        button_up=Mock(), release_all=Mock(),
    )
    ctx = SimpleNamespace(
        input=input_sim,
        device=SimpleNamespace(paste_text=Mock()),
        transform=ScreenTransform(1920, 1080),
        sleep=lambda _ms: None,
    )
    runtime = JsScriptRuntime(
        ctx, tmp_path, settings={"avatarMacroPath": str(avatar_macro)},
        log=lambda _message: None,
    )
    assert runtime.run() is True
    input_sim.vertical_scroll.assert_called_once_with(-2.0)


def test_converter_reports_vertical_scroll_as_supported():
    from bgi_touch.converter.convert import PARTIAL, SUPPORTED

    assert "verticalScroll" in SUPPORTED
    assert "verticalScroll" not in PARTIAL
