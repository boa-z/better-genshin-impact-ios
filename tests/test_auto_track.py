from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np


def test_extract_mission_distance_handles_ocr_variants():
    from bgi_touch.tasks.auto_track import extract_mission_distance

    assert extract_mission_distance(["前往目标 128m", "高度 42米"]) == 42
    assert extract_mission_distance(["没有距离", "12 min"]) is None


def test_blue_track_marker_detector_prefers_compact_cyan_marker():
    from bgi_touch.tasks.auto_track import find_blue_track_marker

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cyan = cv2.cvtColor(np.uint8([[[110, 180, 240]]]), cv2.COLOR_HSV2BGR)[0, 0]
    frame[380:410, 1080:1110] = cyan
    frame[600:604, 500:700] = cyan  # thin scenery/UI fragment

    marker = find_blue_track_marker(frame)
    assert marker is not None
    assert 1090 <= marker.x <= 1100
    assert 390 <= marker.y <= 400


class _Hit:
    def __init__(self, text):
        self.text = text


class _Region:
    def __init__(self, texts=()):
        self.bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.texts = texts

    def find_multi(self, _recognition, limit=20):
        return [_Hit(text) for text in self.texts][:limit]


def _ctx(regions):
    return SimpleNamespace(
        capture_region=Mock(side_effect=regions),
        sleep=Mock(),
        input=SimpleNamespace(
            key_press=Mock(), click_ref=Mock(), move_camera_by=Mock(),
            key_down=Mock(), key_up=Mock(), release_all=Mock(),
        ),
    )


def test_auto_track_starts_navigation_and_stops_at_arrival():
    from bgi_touch.tasks.auto_track import AutoTrackTask

    ctx = _ctx([_Region(["10m"]), _Region(["2m"])])
    with patch("bgi_touch.tasks.auto_track.is_main_ui", return_value=True):
        assert AutoTrackTask(ctx).run()

    ctx.input.key_press.assert_called_once_with("V")
    ctx.input.key_down.assert_not_called()
    ctx.input.release_all.assert_called_once_with()


def test_auto_track_steers_and_holds_forward_when_marker_is_aligned():
    from bgi_touch.tasks.auto_track import AutoTrackTask, TrackMarker

    ctx = _ctx([_Region(), _Region(), _Region(["3m"])])
    marker = TrackMarker(1000, 410, 28, 28, 10)
    with patch("bgi_touch.tasks.auto_track.is_main_ui", return_value=True), \
            patch("bgi_touch.tasks.auto_track.find_blue_track_marker", return_value=marker):
        assert AutoTrackTask(ctx).run()

    ctx.input.key_down.assert_called_once_with("W")
    ctx.input.key_up.assert_called_once_with("W")
    ctx.input.move_camera_by.assert_called()


def test_auto_track_uses_far_quest_teleport_before_navigation():
    from bgi_touch.tasks.auto_track import AutoTrackTask

    ctx = _ctx([_Region(["200m"]), _Region(["1m"])])
    with patch("bgi_touch.tasks.auto_track.is_main_ui", return_value=True), \
            patch.object(AutoTrackTask, "_teleport_near_quest", return_value=True) as teleport:
        assert AutoTrackTask(ctx).run()

    teleport.assert_called_once()
    ctx.input.key_press.assert_called_once_with("V")


def test_dispatcher_maps_auto_track_parameters():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    ctx = Mock()
    with patch("bgi_touch.tasks.auto_track.AutoTrackTask") as task:
        task.return_value.run.return_value = True
        assert TaskDispatcher(ctx).run_task({
            "name": "AutoTrack",
            "config": {
                "timeoutSeconds": 45,
                "farDistance": 180,
                "arrivalDistance": 5,
                "teleportWhenFar": False,
            },
        })

    assert task.call_args.kwargs["timeout_s"] == 45
    assert task.call_args.kwargs["far_distance_m"] == 180
    assert task.call_args.kwargs["arrival_distance_m"] == 5
    assert task.call_args.kwargs["teleport_when_far"] is False


def test_default_layout_has_quest_menu_and_navigation_fallbacks():
    from bgi_touch.input.layout import ControlLayout, DEFAULT_LAYOUT

    layout = ControlLayout.load(DEFAULT_LAYOUT)
    assert layout.binding("J")["button"] == "questMenu"
    assert layout.binding("V")["button"] == "questNavigation"


def test_auto_track_pauses_and_restores_realtime_triggers():
    from bgi_touch.tasks.auto_track import AutoTrackTask

    ctx = _ctx([_Region(["2m"]), _Region(["2m"])])
    state = object()
    loop = SimpleNamespace(
        active=True,
        pause=Mock(return_value=state),
        resume=Mock(),
    )
    ctx._trigger_loop = loop
    with patch("bgi_touch.tasks.auto_track.is_main_ui", return_value=True):
        assert AutoTrackTask(ctx).run()

    loop.pause.assert_called_once_with()
    loop.resume.assert_called_once_with(state)
