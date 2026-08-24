from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest


def test_combat_right_mouse_is_sprint_while_aim_stays_r_key():
    from bgi_touch.combat.dsl import CombatExecutor, parse_combat_script

    input_sim = SimpleNamespace(
        key_press=Mock(), button_down=Mock(), button_up=Mock(),
    )
    executor = CombatExecutor(input_sim, sleep=lambda _ms: None)
    commands = parse_combat_script(
        "click(right),mousedown(right),mouseup(right),aim"
    )[0].commands
    for command in commands:
        executor.exec(command)

    assert input_sim.key_press.call_args_list == [call("LSHIFT"), call("R")]
    input_sim.button_down.assert_called_once_with("sprint")
    input_sim.button_up.assert_called_once_with("sprint")


def test_keymouse_right_button_replays_sprint_hold():
    from bgi_touch.macro.keymouse import MacroPlayer, convert_keymouse

    macro = {"macroEvents": [
        {"type": 4, "mouseButton": "Right", "time": 10},
        {"type": 5, "mouseButton": "Right", "time": 210},
    ]}
    events, warnings = convert_keymouse(macro)
    assert warnings == []
    assert [(event.kind, event.t) for event in events] == [
        ("sprintDown", 10), ("sprintUp", 210),
    ]

    input_sim = SimpleNamespace(
        button_down=Mock(), button_up=Mock(), release_all=Mock(),
    )
    MacroPlayer(input_sim, sleep=lambda _ms: None).play(macro)
    input_sim.button_down.assert_called_once_with("sprint")
    input_sim.button_up.assert_called_once_with("sprint")
    input_sim.release_all.assert_called_once_with()


def test_js_right_button_globals_use_sprint(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    avatar_macro = tmp_path / "avatar.json"
    avatar_macro.write_text("[]", encoding="utf-8")
    (tmp_path / "main.js").write_text(
        "rightButtonDown(); rightButtonClick(); rightButtonUp(); return true;",
        encoding="utf-8",
    )
    input_sim = SimpleNamespace(
        _active_slot=1, button_down=Mock(), button_up=Mock(), key_press=Mock(),
        key_up=Mock(), attack_up=Mock(), release_all=Mock(),
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
    input_sim.button_down.assert_called_once_with("sprint")
    input_sim.key_press.assert_called_once_with("LSHIFT")
    input_sim.button_up.assert_called_once_with("sprint")


def test_converter_reports_right_button_as_supported():
    from bgi_touch.converter.convert import PARTIAL, SUPPORTED

    assert "rightButton" in SUPPORTED
    assert "rightButton" not in PARTIAL
