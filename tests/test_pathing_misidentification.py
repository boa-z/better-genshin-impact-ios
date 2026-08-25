from unittest.mock import Mock


def _executor(logs=None):
    from bgi_touch.pathing.executor import PathingExecutor

    executor = PathingExecutor.__new__(PathingExecutor)
    executor.log = logs.append if logs is not None else Mock()
    return executor


def _waypoint(types, mode="previousDetectedPoint"):
    from bgi_touch.pathing.model import Waypoint

    return Waypoint.parse({
        "id": 1,
        "x": 10,
        "y": 20,
        "type": "path",
        "moveMode": "walk",
        "pointExtParams": {
            "misidentification": {
                "type": types,
                "handlingMode": mode,
            },
        },
    })


def test_previous_detected_point_handles_missing_minimap_fix():
    logs = []
    executor = _executor(logs)
    wp = _waypoint(["unrecognized"])

    position, marker = executor._resolve_misidentified_position(
        wp,
        None,
        (7.0, 8.0),
        None,
        last_map_recognition_at=float("-inf"),
        now=10.0,
    )

    assert position == (7.0, 8.0)
    assert marker == float("-inf")
    assert logs == ["[pathing] 未识别到具体路径，取上次点位"]


def test_previous_detected_point_handles_path_too_far_without_steering_from_bad_fix():
    executor = _executor()
    wp = _waypoint(["pathTooFar"])

    position, _ = executor._resolve_misidentified_position(
        wp,
        (900.0, 900.0),
        (7.0, 8.0),
        1200.0,
        last_map_recognition_at=float("-inf"),
        now=10.0,
    )

    assert position == (7.0, 8.0)


def test_map_recognition_fallback_updates_cooldown_marker():
    executor = _executor()
    executor._position_from_big_map = Mock(return_value=(12.0, 24.0))
    wp = _waypoint(["unrecognized"], mode="mapRecognition")

    position, marker = executor._resolve_misidentified_position(
        wp,
        None,
        (7.0, 8.0),
        None,
        last_map_recognition_at=float("-inf"),
        now=10.0,
    )

    assert position == (12.0, 24.0)
    assert marker == 10.0
    executor._position_from_big_map.assert_called_once_with()


def test_unknown_misidentification_mode_keeps_previous_position():
    executor = _executor()
    wp = _waypoint(["unrecognized"], mode="scheduledArrival")

    position, _ = executor._resolve_misidentified_position(
        wp,
        None,
        (7.0, 8.0),
        None,
        last_map_recognition_at=float("-inf"),
        now=10.0,
    )

    assert position == (7.0, 8.0)
