from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest


def _task(monkeypatch):
    import bgi_touch.pathing.tp as module

    monkeypatch.setattr(module, "BigMapLocator", lambda _name: SimpleNamespace())
    return module.TpTask(SimpleNamespace())


@pytest.mark.parametrize(
    ("center_y", "expected"),
    ((468.0, 1.0), (540.0, 3.5), (612.0, 6.0)),
)
def test_map_scale_button_position_maps_to_upstream_zoom_level(monkeypatch, center_y, expected):
    task = _task(monkeypatch)
    hit = SimpleNamespace(
        y=center_y - 12.0,
        height=24.0,
        is_exist=lambda: True,
    )
    region = SimpleNamespace(find=Mock(return_value=hit))

    assert task._measure_big_map_zoom_level(region) == pytest.approx(expected)


def test_map_scale_button_is_measured_in_native_iphone_coordinates(monkeypatch):
    from bgi_touch.engine.recognition import ImageRegion
    from bgi_touch.pathing.tp import MAP_SCALE_BUTTON
    from bgi_touch.vision.coordinate import ScreenTransform

    task = _task(monkeypatch)
    transform = ScreenTransform(2816, 1296)
    ctx = SimpleNamespace(transform=transform)
    task.ctx = ctx

    template = MAP_SCALE_BUTTON.template.bgr
    scaled = cv2.resize(template, (round(25 * transform.scale), round(24 * transform.scale)))
    frame = np.zeros((transform.device_height, transform.device_width, 3), dtype=np.uint8)
    x, y = transform.to_device(30, 540 - 12)
    x, y = round(x), round(y)
    frame[y:y + scaled.shape[0], x:x + scaled.shape[1]] = scaled

    assert task._measure_big_map_zoom_level(ImageRegion(ctx, frame)) == pytest.approx(
        3.5, abs=0.05,
    )


def test_get_zoom_level_prefers_cached_frame_without_new_screenshot(monkeypatch):
    task = _task(monkeypatch)
    cached_frame = object()
    task.ctx.cached_frame = Mock(return_value=(cached_frame, 0.01))
    task.ctx.capture_region = Mock(side_effect=AssertionError("must use cached frame"))
    task._measure_zoom_from_frame = Mock(side_effect=[None, 2.25])

    assert task.get_big_map_zoom_level() == pytest.approx(2.25)
    task.ctx.capture_region.assert_not_called()


def test_set_zoom_level_calibrates_pinch_delta_from_feedback(monkeypatch):
    from bgi_touch.pathing.tp import MAP_ZOOM_INITIAL_GESTURE_DELTA, TpTask
    from bgi_touch.vision.coordinate import ScreenTransform

    ctx = SimpleNamespace(
        transform=ScreenTransform(1920, 1080),
        device=SimpleNamespace(multi_touch=Mock()),
        sleep=Mock(),
    )
    task = TpTask.__new__(TpTask)
    task.ctx = ctx
    task.log = Mock()
    task.open_map = Mock(return_value=True)
    task._read_map_zoom_level = Mock(return_value=1.0)
    task._frame_cursor = Mock(return_value=None)
    task._capture_after_zoom_gesture = Mock(return_value=object())
    task._measure_zoom_from_frame = Mock(side_effect=[1.5, 2.0])
    task._zoom_level = None
    task._zoom_gesture_delta = MAP_ZOOM_INITIAL_GESTURE_DELTA
    task._zoom_span_sign = -1.0

    task._measure_zoom_from_frame = Mock(side_effect=[1.7])

    result = task._set_big_map_zoom_level(1.7)

    assert result == pytest.approx(1.7)
    assert ctx.device.multi_touch.call_count == 1
    assert task._zoom_gesture_delta > MAP_ZOOM_INITIAL_GESTURE_DELTA
    assert task._capture_after_zoom_gesture.call_count == 1


def test_get_zoom_level_rejects_non_map_frame(monkeypatch):
    task = _task(monkeypatch)
    task._read_map_zoom_level = Mock(return_value=None)

    with pytest.raises(RuntimeError, match="不能使用GetBigMapZoomLevel"):
        task.get_big_map_zoom_level()
