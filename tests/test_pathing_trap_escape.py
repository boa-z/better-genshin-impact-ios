from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np


def test_stuck_detector_consumes_eight_sample_windows_and_counts_traps():
    from bgi_touch.pathing.trap_escaper import StuckDetector

    detector = StuckDetector()

    assert [detector.add((100, 200)) for _ in range(7)] == [False] * 7
    assert detector.add((101, 201)) is True
    assert detector.trap_count == 1
    assert detector.positions == ()

    for _ in range(7):
        assert detector.add((100, 200)) is False
    assert detector.add((100, 200)) is True
    assert detector.trap_count == 2


def test_stuck_detector_uses_manhattan_threshold():
    from bgi_touch.pathing.trap_escaper import StuckDetector

    detector = StuckDetector()
    for index in range(7):
        detector.add((0, 0))
    assert detector.add((1.5, 1.5)) is False
    assert detector.trap_count == 0


def test_trap_escaper_rotate_and_move_releases_keys_and_uses_attack():
    from bgi_touch.pathing.trap_escaper import TrapEscaper

    input_sim = SimpleNamespace(
        key_down=Mock(), key_up=Mock(), key_press=Mock(), attack=Mock(),
    )
    ctx = SimpleNamespace(input=input_sim, sleep=Mock())
    positioner = SimpleNamespace()
    escaper = TrapEscaper(
        ctx, positioner, clock=Mock(side_effect=[0.0, 0.0, 1.0]),
    )

    escaper.rotate_and_move()

    input_sim.key_up.assert_any_call("W")
    input_sim.attack.assert_called_once_with()
    input_sim.key_down.assert_any_call("A")
    input_sim.key_up.assert_any_call("A")
    assert input_sim.key_press.call_args_list[:1] == [call("SPACE")]


def test_trap_escaper_moves_to_target_and_stops_at_target(monkeypatch):
    from bgi_touch.pathing import trap_escaper as module
    from bgi_touch.pathing.trap_escaper import TrapEscaper

    input_sim = SimpleNamespace(
        key_down=Mock(), key_up=Mock(), key_press=Mock(),
        move_camera_by=Mock(), attack=Mock(),
    )
    frames = [
        np.zeros((20, 20, 3), dtype=np.uint8),
        np.zeros((20, 20, 3), dtype=np.uint8),
    ]
    ctx = SimpleNamespace(
        input=input_sim,
        sleep=Mock(),
        capture_bgr=Mock(side_effect=frames),
        transform=None,
    )
    positions = iter([(0.0, 0.0), (10.0, 0.0)])
    positioner = SimpleNamespace(get_position=lambda _frame: next(positions))
    clock = Mock(return_value=0.0)
    escaper = TrapEscaper(ctx, positioner, clock=clock)

    with patch.object(module, "_camera_orientation", return_value=0.0):
        escaper.move_to((10.0, 0.0))

    input_sim.key_down.assert_any_call("W")
    input_sim.key_up.assert_called_with("W")
    input_sim.move_camera_by.assert_called_once()
