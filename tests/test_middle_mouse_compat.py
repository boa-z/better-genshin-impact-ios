from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest


def test_combat_middle_mouse_maps_to_elemental_sight():
    from bgi_touch.combat.dsl import CombatExecutor, parse_combat_script

    input_sim = SimpleNamespace(
        tap_button=Mock(), button_down=Mock(), button_up=Mock(),
    )
    executor = CombatExecutor(input_sim, sleep=lambda _ms: None)
    commands = parse_combat_script(
        "click(middle),mousedown(middle),mouseup(middle)"
    )[0].commands
    for command in commands:
        executor.exec(command)

    input_sim.tap_button.assert_called_once_with("elementalSight")
    input_sim.button_down.assert_called_once_with("elementalSight")
    input_sim.button_up.assert_called_once_with("elementalSight")


def test_keymouse_middle_button_replays_elemental_sight_hold():
    from bgi_touch.macro.keymouse import MacroPlayer, convert_keymouse

    macro = {"macroEvents": [
        {"type": 4, "mouseButton": "Middle", "time": 10},
        {"type": 5, "mouseButton": "Middle", "time": 110},
    ]}
    events, warnings = convert_keymouse(macro)
    assert warnings == []
    assert [(event.kind, event.t) for event in events] == [
        ("sightDown", 10), ("sightUp", 110),
    ]

    input_sim = SimpleNamespace(
        button_down=Mock(), button_up=Mock(), release_all=Mock(),
    )
    MacroPlayer(input_sim, sleep=lambda _ms: None).play(macro)
    input_sim.button_down.assert_called_once_with("elementalSight")
    input_sim.button_up.assert_called_once_with("elementalSight")


def test_avatar_middle_mouse_uses_elemental_sight_button():
    from bgi_touch.engine.combat_host import Avatar

    input_sim = SimpleNamespace(
        tap_button=Mock(), button_down=Mock(), button_up=Mock(),
    )
    avatar = Avatar(SimpleNamespace(ctx=SimpleNamespace(input=input_sim)), "玛薇卡", 1)
    avatar.click("middle")
    avatar.mouse_down("middle")
    avatar.mouse_up("middle")
    input_sim.tap_button.assert_called_once_with("elementalSight")
    input_sim.button_down.assert_called_once_with("elementalSight")
    input_sim.button_up.assert_called_once_with("elementalSight")


def test_js_middle_button_globals_use_elemental_sight(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    avatar_macro = tmp_path / "avatar.json"
    avatar_macro.write_text("[]", encoding="utf-8")
    (tmp_path / "main.js").write_text(
        "middleButtonClick(); middleButtonDown(); middleButtonUp(); return true;",
        encoding="utf-8",
    )
    input_sim = SimpleNamespace(
        _active_slot=1, tap_button=Mock(), button_down=Mock(), button_up=Mock(),
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
    input_sim.tap_button.assert_called_once_with("elementalSight")
    input_sim.button_down.assert_called_once_with("elementalSight")
    input_sim.button_up.assert_called_once_with("elementalSight")


def test_converter_reports_middle_button_as_supported():
    from bgi_touch.converter.convert import PARTIAL, SUPPORTED

    assert "middleButton" in SUPPORTED
    assert "middleButton" not in PARTIAL
