from types import SimpleNamespace

import cv2
import numpy as np


def _synthetic_minimap(angle: float) -> np.ndarray:
    """Create a minimap with two radial edges 90 degrees apart."""

    size = 216
    image = np.full((size, size), 80, dtype=np.uint8)
    center = np.array([size / 2, size / 2])
    for radius in range(20, 78):
        for offset in range(91):
            theta = np.deg2rad(angle + offset)
            point = np.rint(center + radius * np.array([
                np.cos(theta), np.sin(theta),
            ])).astype(int)
            if 0 <= point[0] < size and 0 <= point[1] < size:
                image[point[1], point[0]] = 230
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_gia_orientation_returns_a_confident_direction_for_sector_edges():
    from bgi_touch.pathing.camera import gia_orientation_with_confidence

    angle, confidence = gia_orientation_with_confidence(_synthetic_minimap(35))

    assert angle is not None
    assert confidence >= 0.20
    # The polar detector quantizes to one-degree bins; the edge convention
    # contributes a fixed 45-degree sector offset.
    assert abs(angle - ((35 + 45) % 360)) <= 2


def test_orientation_rejects_a_uniform_low_signal_minimap():
    from bgi_touch.pathing.camera import orientation_with_confidence

    image = np.full((216, 216, 3), 100, dtype=np.uint8)
    angle, confidence = orientation_with_confidence(image)

    assert angle is None
    assert confidence < 0.20


def test_camera_orientation_crops_and_normalizes_ios_frame():
    from bgi_touch.pathing.camera import (
        ORIENTATION_MINIMAP_SIZE,
        crop_minimap_for_orientation,
    )

    frame = np.zeros((1284, 2778, 3), dtype=np.uint8)
    ctx = SimpleNamespace(
        layout=SimpleNamespace(buttons={"minimapCenter": (0.08, 0.15)})
    )

    crop = crop_minimap_for_orientation(ctx, frame)

    assert crop is not None
    assert crop.shape == (ORIENTATION_MINIMAP_SIZE, ORIENTATION_MINIMAP_SIZE, 3)


def test_camera_orientation_returns_none_for_missing_minimap():
    from bgi_touch.pathing.executor import camera_orientation_deg

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ctx = SimpleNamespace(
        layout=SimpleNamespace(buttons={}),
        transform=SimpleNamespace(device_width=1920, device_height=1080),
    )

    assert camera_orientation_deg(ctx, frame) is None
