from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from bgi_touch.pathing.actions import PathingActionRunner
from bgi_touch.pathing.model import Waypoint


def _waypoint(action: str, params: str = "") -> Waypoint:
    return Waypoint(
        id=1,
        x=0,
        y=0,
        type="target",
        move_mode="walk",
        action=action,
        action_params=params,
    )


def _runner() -> PathingActionRunner:
    ctx = MagicMock()
    ctx.sleep = MagicMock()
    return PathingActionRunner(ctx, log=MagicMock())


def test_circular_motion_calculator_matches_bettergi_geometry():
    from bgi_touch.pathing.actions import CircularMotionCalculator

    calculator = CircularMotionCalculator(1.1)
    edge, radius, angle = calculator.get_circle_info(0)

    assert edge == pytest.approx(600 / 1.1)
    assert radius == pytest.approx(311.8097606, rel=1e-7)
    assert angle == pytest.approx(1.272495441, rel=1e-7)


def test_pick_around_replays_original_arc_with_profile_middle_button():
    input_simulator = SimpleNamespace(
        key_down=Mock(), key_up=Mock(), key_press=Mock(),
        tap_button=Mock(),
    )
    ctx = SimpleNamespace(input=input_simulator, sleep=Mock())
    runner = PathingActionRunner(ctx, log=Mock())

    assert runner.run(_waypoint("pick_around", "1"))

    assert input_simulator.key_press.call_args_list[:2] == [call("S"), call("A")]
    assert input_simulator.key_down.call_args_list == [
        call("W"), call("W"), call("A"),
    ]
    assert input_simulator.key_up.call_args_list == [
        call("W"), call("W"), call("A"),
    ]
    assert input_simulator.tap_button.call_args_list[0].args == ("elementalSight",)
    assert input_simulator.tap_button.call_count == 9


def test_elemental_collect_selects_matching_party_member_and_action():
    ctx = SimpleNamespace(
        input=SimpleNamespace(key_press=Mock(), attack=Mock()),
        sleep=Mock(), party_slots={"芭芭拉": 2},
    )
    runner = PathingActionRunner(ctx, log=Mock())
    runner.combat = Mock()

    with patch(
        "bgi_touch.engine.party_hud.canonical_avatar_name",
        side_effect=lambda value: str(value),
    ):
        assert runner.run(_waypoint("hydro_collect"))

    runner.combat.switch_to.assert_called_once_with("芭芭拉")
    runner.combat.exec.assert_called_once()
    assert runner.combat.exec.call_args.args[0].action == "attack"


def test_elemental_collect_without_matching_character_is_safe():
    ctx = SimpleNamespace(
        input=SimpleNamespace(key_press=Mock(), attack=Mock()),
        sleep=Mock(), party_slots={"钟离": 1},
    )
    log = Mock()
    runner = PathingActionRunner(ctx, log=log)

    assert runner.run(_waypoint("hydro_collect"))
    assert "没有可用的hydro元素采集角色" in log.call_args.args[0]


def test_pick_up_collect_resolves_character_alias_and_builtin_macro():
    ctx = SimpleNamespace(
        input=SimpleNamespace(key_press=Mock()),
        sleep=Mock(), party_slots={"万叶": 4},
    )
    runner = PathingActionRunner(ctx, log=Mock())
    runner.combat = Mock()

    with patch(
        "bgi_touch.engine.party_hud.canonical_avatar_name",
        return_value="枫原万叶",
    ):
        assert runner.run(_waypoint("pick_up_collect", "万叶-短E"))

    script = runner.combat.run.call_args.args[0]
    assert script.startswith("万叶 attack(0.08),keydown(E)")
    assert "keyup(E),attack(0.5)" in script


def test_fight_action_dispatches_to_auto_fight_instead_of_blind_attacks():
    ctx = SimpleNamespace(
        input=SimpleNamespace(key_press=Mock()),
        sleep=Mock(), party_slots={"钟离": 1},
    )
    with patch("bgi_touch.pathing.actions.TaskDispatcher") as dispatcher_type:
        dispatcher_type.return_value.run_auto_fight_task.return_value = True
        assert PathingActionRunner(ctx, log=Mock()).run(
            _waypoint("fight", '{"combatStrategyPath":"route.txt"}')
        )

    dispatcher_type.return_value.run_auto_fight_task.assert_called_once_with(
        {"combatStrategyPath": "route.txt"}
    )


def test_mining_uses_the_first_matching_character_macro():
    ctx = SimpleNamespace(
        input=SimpleNamespace(key_press=Mock()),
        sleep=Mock(), party_slots={"钟离": 1},
    )
    runner = PathingActionRunner(ctx, log=Mock())
    runner.combat = Mock()

    with patch(
        "bgi_touch.engine.party_hud.canonical_avatar_name",
        side_effect=lambda value: str(value),
    ):
        assert runner.run(_waypoint("mining"))

    script = runner.combat.run.call_args.args[0]
    assert script.startswith("钟离 e(hold,wait)")


def test_stop_flying_always_releases_glider_before_attack():
    input_simulator = SimpleNamespace(
        key_press=Mock(), attack=Mock(),
    )
    ctx = SimpleNamespace(input=input_simulator, sleep=Mock())

    assert PathingActionRunner(ctx, log=Mock()).run(
        _waypoint("stop_flying", "0")
    )

    assert input_simulator.key_press.call_args_list == [call("SPACE"), call("SPACE")]
    input_simulator.attack.assert_called_once_with()


def test_stop_flying_waits_for_motion_detector_to_leave_flight():
    import numpy as np

    input_simulator = SimpleNamespace(
        key_press=Mock(), attack=Mock(),
    )
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    ctx = SimpleNamespace(
        input=input_simulator,
        sleep=Mock(),
        capture_bgr=Mock(side_effect=frames),
    )
    detector = Mock(side_effect=["fly", "normal"])
    runner = PathingActionRunner(ctx, log=Mock(), motion_detector=detector)

    assert runner.run(_waypoint("stop_flying", "100"))

    assert input_simulator.key_press.call_args_list == [
        call("SPACE"), call("SPACE"),
    ]
    input_simulator.attack.assert_called_once_with()
    assert ctx.sleep.call_args_list == [call(100), call(300), call(300)]
    assert ctx.capture_bgr.call_count == 2
    assert detector.call_count == 2


def test_read_gadget_cooldown_uses_white_hsv_crop_and_ocr():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    from bgi_touch.pathing.actions import read_gadget_cooldown
    from bgi_touch.vision.coordinate import ScreenTransform

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[814:838, 1790:1850] = (255, 255, 255)
    provider = SimpleNamespace(
        recognize=Mock(return_value=[SimpleNamespace(text="7,5")]),
    )

    with patch("bgi_touch.pathing.actions.get_ocr", return_value=provider):
        assert read_gadget_cooldown(frame, ScreenTransform(1920, 1080)) == pytest.approx(7.5)

    crop = provider.recognize.call_args.args[0]
    assert crop.shape == (24, 60)
    assert int(crop.max()) == 255


def test_use_gadget_waits_for_cooldown_in_seconds_and_settles():
    import numpy as np

    input_simulator = SimpleNamespace(key_press=Mock())
    ctx = SimpleNamespace(
        input=input_simulator,
        sleep=Mock(),
        capture_bgr=Mock(return_value=np.zeros((4, 4, 3), dtype=np.uint8)),
        transform=SimpleNamespace(),
    )
    runner = PathingActionRunner(ctx, log=Mock())

    with patch(
        "bgi_touch.pathing.actions.read_gadget_cooldown", return_value=7.5
    ):
        assert runner.run(_waypoint("use_gadget", "5"))

    assert input_simulator.key_press.call_args_list == [call("Z"), call("Z")]
    assert ctx.sleep.call_args_list == [call(5100), call(300)]


def test_use_gadget_not_wait_does_not_capture_a_second_frame():
    input_simulator = SimpleNamespace(key_press=Mock())
    ctx = SimpleNamespace(input=input_simulator, sleep=Mock(), capture_bgr=Mock())

    assert PathingActionRunner(ctx, log=Mock()).run(
        _waypoint("use_gadget", "not_wait")
    )

    assert input_simulator.key_press.call_args_list == [call("Z"), call("Z")]
    ctx.capture_bgr.assert_not_called()
    ctx.sleep.assert_called_once_with(300)


def test_nahida_collect_replays_full_scan_and_restores_view():
    input_simulator = SimpleNamespace(
        key_down=Mock(), key_up=Mock(), move_camera_by=Mock(), tap_button=Mock(),
    )
    ctx = SimpleNamespace(input=input_simulator, sleep=Mock())
    runner = PathingActionRunner(ctx, log=Mock())
    runner.combat = Mock()

    assert runner.run(_waypoint("nahida_collect"))

    runner.combat.switch_to.assert_called_once_with("纳西妲")
    assert input_simulator.key_down.call_args_list == [call("E")]
    assert input_simulator.key_up.call_args_list == [call("E")]
    assert input_simulator.move_camera_by.call_count == 76
    input_simulator.tap_button.assert_called_once_with("elementalSight")


def test_grab_leaf_requires_two_consecutive_detections():
    input_simulator = SimpleNamespace(
        key_press=Mock(), move_camera_by=Mock(), tap_button=Mock(),
    )
    ctx = SimpleNamespace(
        input=input_simulator, sleep=Mock(), capture_bgr=Mock(),
    )
    runner = PathingActionRunner(ctx, log=Mock())
    runner._leaf_prompt_visible = Mock(side_effect=[True, False, True, True])

    assert runner.run(_waypoint("up_down_grab_leaf", "up"))

    assert input_simulator.key_press.call_args_list == [
        call("F"), call("SPACE"),
    ]
    input_simulator.tap_button.assert_called_once_with("elementalSight")
    assert runner._leaf_prompt_visible.call_count == 4


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ("7:00", (7, 0, True)),
        ("07:05", (7, 5, True)),
        ("19:30:false", (19, 30, False)),
        ("24:00:TRUE", (24, 0, True)),
    ],
)
def test_set_time_action_maps_bettergi_parameters(params, expected):
    api = MagicMock()
    api.setTime.return_value = True

    with patch("bgi_touch.engine.genshin_api.GenshinApi", return_value=api):
        assert _runner().run(_waypoint("set_time", params))

    api.setTime.assert_called_once_with(*expected)


@pytest.mark.parametrize(
    "params",
    ["", "7", "07:0", "07:60", "25:00", "07:00:yes", "07:00:true:extra"],
)
def test_set_time_action_rejects_invalid_parameters(params):
    with pytest.raises(ValueError, match="set_time"):
        _runner().run(_waypoint("set_time", params))


def test_set_time_action_reports_ui_failure():
    api = MagicMock()
    api.setTime.return_value = False

    with patch("bgi_touch.engine.genshin_api.GenshinApi", return_value=api):
        with pytest.raises(RuntimeError, match="set_time 失败"):
            _runner().run(_waypoint("set_time", "07:00"))


def test_wonderland_cycle_action_calls_genshin_api():
    api = MagicMock()
    api.wonderlandCycle.return_value = True

    with patch("bgi_touch.engine.genshin_api.GenshinApi", return_value=api):
        assert _runner().run(_waypoint("wonderland_cycle"))

    api.wonderlandCycle.assert_called_once_with()


def test_wonderland_cycle_action_reports_ui_failure():
    api = MagicMock()
    api.wonderlandCycle.return_value = False

    with patch("bgi_touch.engine.genshin_api.GenshinApi", return_value=api):
        with pytest.raises(RuntimeError, match="wonderland_cycle 失败"):
            _runner().run(_waypoint("wonderland_cycle"))
