from unittest.mock import Mock

import pytest


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


def test_pointer_drag_collapses_incremental_moves_into_one_touch_swipe():
    from bgi_touch.input.pointer import TouchPointer

    input_sim = Mock()
    clock = Clock()
    pointer = TouchPointer(input_sim, clock=clock)

    pointer.move_to(400, 750)
    pointer.left_down()
    for _ in range(5):
        pointer.move_by(0, -10)
    clock.now += 0.7
    pointer.left_up()

    input_sim.drag_ref.assert_called_once_with(
        400, 750, 400, 700, duration_ms=699,
    )
    input_sim.attack_down.assert_not_called()
    input_sim.move_camera_by.assert_not_called()


def test_pointer_scales_4k_script_metrics_to_1080p_reference_space():
    from bgi_touch.input.pointer import TouchPointer

    input_sim = Mock()
    pointer = TouchPointer(input_sim)
    pointer.set_metrics(3840, 2160, 2)

    pointer.click_at(1920, 1080)
    pointer.move_to(1000, 1000)
    pointer.left_down()
    pointer.move_to(3000, 1800)
    pointer.left_up()

    input_sim.click_ref.assert_called_once_with(960, 540)
    drag = input_sim.drag_ref.call_args
    assert drag.args[:4] == (500, 500, 1500, 900)
    assert pointer.get_metrics() == [3840, 2160, 2.0]


def test_pointer_without_recent_ui_intent_keeps_gameplay_attack_and_camera():
    from bgi_touch.input.pointer import TouchPointer

    input_sim = Mock()
    clock = Clock()
    pointer = TouchPointer(input_sim, clock=clock)

    pointer.left_click()
    pointer.left_down()
    pointer.left_up()
    pointer.move_by(30, -20)

    input_sim.attack.assert_called_once_with()
    input_sim.attack_down.assert_called_once_with()
    input_sim.attack_up.assert_called_once_with()
    input_sim.move_camera_by.assert_called_once_with(30, -20)


def test_keyboard_activity_clears_stale_pointer_intent():
    from bgi_touch.input.pointer import TouchPointer

    input_sim = Mock()
    pointer = TouchPointer(input_sim)
    pointer.move_to(960, 540)
    pointer.clear_intent()
    pointer.left_click()

    input_sim.attack.assert_called_once_with()
    input_sim.click_ref.assert_not_called()


def test_pointer_validates_bettergi_metrics_contract():
    from bgi_touch.input.pointer import TouchPointer

    pointer = TouchPointer(Mock())
    with pytest.raises(ValueError, match="16:9"):
        pointer.set_metrics(2816, 1296, 1)
    with pytest.raises(ValueError, match="DPI"):
        pointer.set_metrics(1920, 1080, 0)


def test_windows_control_key_aliases_match_devicehub_codes():
    from bgi_touch.input.layout import normalize_key

    assert normalize_key("LCONTROL") == "LCTRL"
    assert normalize_key("VK_LCONTROL") == "LCTRL"
    assert normalize_key("RCONTROL") == "RCTRL"


def test_region_move_arms_runtime_virtual_pointer():
    from bgi_touch.engine.recognition import Region
    from bgi_touch.vision.coordinate import ScreenTransform

    pointer = Mock()
    ctx = Mock(transform=ScreenTransform(1920, 1080))
    ctx._script_pointer = pointer
    region = Region(ctx, 100, 200, 80, 40)

    region.move()

    pointer.move_to.assert_called_once_with(140, 220)
