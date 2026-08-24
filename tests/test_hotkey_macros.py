import time
from types import SimpleNamespace
from unittest.mock import Mock, call


def test_turn_around_matches_bettergi_mouse_delta_and_delay():
    from bgi_touch.macro.hotkeys import HotkeyMacroHost

    ctx = SimpleNamespace(
        input=SimpleNamespace(move_camera_by=Mock()), sleep=Mock(),
    )
    host = HotkeyMacroHost(
        ctx, runaround_mouse_x_interval=640, runaround_interval_ms=12,
    )
    host.turn_around()
    ctx.input.move_camera_by.assert_called_once_with(640.0, 0)
    ctx.sleep.assert_called_once_with(12)


def test_quick_enhance_artifact_preserves_upstream_click_sequence():
    from bgi_touch.macro.hotkeys import HotkeyMacroHost

    ctx = SimpleNamespace(
        input=SimpleNamespace(click_ref=Mock()), sleep=Mock(),
    )
    host = HotkeyMacroHost(ctx, enhance_wait_delay_ms=250)
    host.quick_enhance_artifact()
    assert ctx.input.click_ref.call_args_list == [
        call(1760, 770), call(1760, 1020), call(150, 150), call(150, 220),
    ]
    assert ctx.sleep.call_args_list == [call(100), call(350), call(100), call(100)]


def test_short_hold_sends_only_initial_key_press():
    from bgi_touch.macro.hotkeys import RepeatingKeyMacro

    input_sim = SimpleNamespace(key_press=Mock())
    macro = RepeatingKeyMacro(
        input_sim, thresholds_ms={"F": 50}, intervals_ms={"F": 5},
        log=lambda _message: None,
    )
    assert macro.key_down("f")
    assert not macro.key_down("F")
    time.sleep(0.01)
    assert macro.key_up("F")
    macro.stop()
    input_sim.key_press.assert_called_once_with("F")


def test_long_hold_repeats_and_key_up_stops_worker():
    from bgi_touch.macro.hotkeys import RepeatingKeyMacro

    input_sim = SimpleNamespace(key_press=Mock())
    macro = RepeatingKeyMacro(
        input_sim, thresholds_ms={"SPACE": 5}, intervals_ms={"SPACE": 5},
        log=lambda _message: None,
    )
    assert macro.key_down("Spacebar")
    deadline = time.monotonic() + 0.3
    while input_sim.key_press.call_count < 3 and time.monotonic() < deadline:
        time.sleep(0.002)
    assert input_sim.key_press.call_count >= 3
    macro.key_up("SPACE")
    count = input_sim.key_press.call_count
    time.sleep(0.02)
    assert input_sim.key_press.call_count == count


def test_js_runtime_exposes_hotkey_macro_host(tmp_path):
    import pytest

    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    macro = tmp_path / "avatar.json"
    macro.write_text("[]", encoding="utf-8")
    (tmp_path / "main.js").write_text(
        "hotkeyMacros.TurnAround(); return true;", encoding="utf-8",
    )
    input_sim = SimpleNamespace(
        _active_slot=1, move_camera_by=Mock(), key_press=Mock(),
        key_up=Mock(), attack_up=Mock(), button_up=Mock(), release_all=Mock(),
    )
    ctx = SimpleNamespace(
        input=input_sim,
        device=SimpleNamespace(paste_text=Mock()),
        transform=ScreenTransform(1920, 1080),
        sleep=Mock(),
    )
    runtime = JsScriptRuntime(
        ctx, tmp_path, settings={"avatarMacroPath": str(macro)},
        log=lambda _message: None,
    )
    assert runtime.run() is True
    input_sim.move_camera_by.assert_called_once_with(500.0, 0)
