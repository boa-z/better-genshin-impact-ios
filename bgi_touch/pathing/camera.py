"""Camera orientation estimation from the Genshin minimap.

The desktop BetterGI implementation unwraps the minimap into polar space and
uses the two edges of the player-view sector to estimate its direction. This
module keeps that algorithm independent from :mod:`pathing.executor`, so it
can be tested with recorded or synthetic minimaps and can safely fall back to
the older ring-intensity estimator on low-signal frames.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from .minimap import (
    MINIMAP_CAPTURE_RADIUS_N,
    prepare_minimap_source,
)


ORIENTATION_MINIMAP_SIZE = 212
ORIENTATION_PROCESS_SIZE = 156
# The iPhone 13 Pro Max layout uses a minimap radius of roughly 4.2% of the
# landscape capture width.  Normalize that native crop to BetterGI's 212px
# asset size before applying its central 156px processing crop.
MINIMAP_RADIUS_N = MINIMAP_CAPTURE_RADIUS_N
ORIENTATION_ANGLE_BINS = 360
ORIENTATION_PEAK_WIDTH = ORIENTATION_ANGLE_BINS // 4
ORIENTATION_CONFIDENCE_THRESHOLD = 0.20


def _find_peaks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < 3:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(
        (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
    ) + 1


def _shift(values: np.ndarray, amount: int) -> np.ndarray:
    """Match BetterGI's circular ``RightShift``/``LeftShift`` semantics."""

    return np.roll(values, int(amount))


def _confidence(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    peak = float(values.max())
    if peak <= 0:
        return 0.0
    # A unique direction produces a peak well above the broad background. A
    # percentile baseline is less sensitive than comparing only the top two
    # bins, because the convolution intentionally creates a short plateau.
    baseline = float(np.percentile(values, 75))
    return max(0.0, min(1.0, (peak - baseline) / peak))


def gia_orientation_with_confidence(
    minimap: np.ndarray,
) -> tuple[float | None, float]:
    """Return ``(angle, confidence)`` using BetterGI's polar edge detector.

    The result uses the same convention as BetterGI: 0 degrees is map north
    and angles increase clockwise. The input may be BGR or grayscale.
    """

    if not isinstance(minimap, np.ndarray) or minimap.size == 0:
        return None, 0.0
    if minimap.ndim == 3:
        if minimap.shape[2] == 4:
            gray = cv2.cvtColor(minimap, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
    elif minimap.ndim == 2:
        gray = minimap
    else:
        return None, 0.0

    if gray.shape[0] < 32 or gray.shape[1] < 32:
        return None, 0.0
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    height, width = gray.shape[:2]
    polar = cv2.warpPolar(
        gray,
        (ORIENTATION_ANGLE_BINS, ORIENTATION_ANGLE_BINS),
        (width / 2.0, height / 2.0),
        float(min(width, height) / 2.0),
        cv2.INTER_LINEAR | cv2.WARP_POLAR_LINEAR,
    )
    # The upstream detector ignores the center and outer edge of the map. In
    # the polar image the first coordinate is the angular column.
    polar_roi = polar[:, 10:80]
    if polar_roi.size == 0:
        return None, 0.0
    polar_roi = cv2.rotate(polar_roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
    scharr = cv2.Scharr(polar_roi, cv2.CV_32F, 1, 0).reshape(-1)

    left = np.zeros(ORIENTATION_ANGLE_BINS, dtype=np.int64)
    right = np.zeros(ORIENTATION_ANGLE_BINS, dtype=np.int64)
    np.add.at(left, _find_peaks(scharr) % ORIENTATION_ANGLE_BINS, 1)
    np.add.at(right, _find_peaks(-scharr) % ORIENTATION_ANGLE_BINS, 1)

    left_only = np.maximum(left - right, 0)
    right_only = np.maximum(right - left, 0)
    combined = np.zeros(ORIENTATION_ANGLE_BINS, dtype=np.int64)
    for offset in range(-2, 3):
        weight = 3 - abs(offset)
        combined += (left_only * _shift(
            right_only, -ORIENTATION_PEAK_WIDTH + offset
        ) * weight) // 3

    result = np.zeros(ORIENTATION_ANGLE_BINS, dtype=np.int64)
    for offset in range(-2, 3):
        weight = 3 - abs(offset)
        result += (_shift(combined, offset) * weight) // 3
    if not np.any(result):
        return None, 0.0

    index = int(np.argmax(result))
    angle = float((index + 45) % 360)
    return angle, _confidence(result)


@lru_cache(maxsize=1)
def _orientation_remap_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the remap and alpha tables from BetterGI's calculator."""

    radius = np.linspace(19.0, 78.0, 60, dtype=np.float32)
    theta = np.linspace(0.0, 360.0, 360, endpoint=False, dtype=np.float32)
    radius_grid = np.repeat(radius[None, :], 360, axis=0)
    theta_grid = np.deg2rad(np.repeat(theta[:, None], 60, axis=1))
    remap_x = radius_grid * np.cos(theta_grid) + ORIENTATION_PROCESS_SIZE / 2
    remap_y = radius_grid * np.sin(theta_grid) + ORIENTATION_PROCESS_SIZE / 2
    alpha_params = np.array(
        [18.632, 20.157, 24.093, 34.617, 38.566, 41.94, 47.654,
         51.087, 58.561, 63.925, 67.759, 71.77, 75.214],
        dtype=np.float32,
    )
    alpha1 = 229.0 + np.searchsorted(alpha_params, radius, side="left")
    alpha2 = 137.0 + 1.43 * radius
    return (
        remap_x.astype(np.float32),
        remap_y.astype(np.float32),
        np.repeat(alpha1[None, :], 360, axis=0).astype(np.float32),
        np.repeat(alpha2[None, :], 360, axis=0).astype(np.float32),
    )


def _apply_alpha_mask(image: np.ndarray, alpha: np.ndarray, background: float) -> np.ndarray:
    return (image.astype(np.float32) - background) * (255.0 / alpha) + background


def bettergi_orientation_with_confidence(
    minimap: np.ndarray,
) -> tuple[float | None, float]:
    """Port the current BetterGI ``CameraOrientationCalculator``."""

    prepared = prepare_minimap_source(minimap)
    if prepared is None:
        return None, 0.0
    source, mask = prepared
    remap_x, remap_y, alpha1, alpha2 = _orientation_remap_data()
    remapped = cv2.remap(source, remap_x, remap_y, cv2.INTER_LINEAR)
    remapped_mask = cv2.remap(mask, remap_x, remap_y, cv2.INTER_NEAREST)
    hls = cv2.cvtColor(remapped, cv2.COLOR_BGR2HLS_FULL)
    hue = hls[:, :, 0].astype(np.float32)
    original_gray = cv2.cvtColor(remapped, cv2.COLOR_BGR2GRAY).astype(np.float32)

    normalized_a = _apply_alpha_mask(original_gray, alpha1, 0.0)
    fa = np.where(remapped_mask != 0, original_gray, normalized_a)
    transformed_b = _apply_alpha_mask(original_gray, alpha2, 255.0)
    normalized_b = _apply_alpha_mask(transformed_b, alpha1, 0.0)
    fb = np.where(remapped_mask != 0, transformed_b, normalized_b)

    hist_a, _, _ = np.histogram2d(
        hue.reshape(-1), fa.reshape(-1), bins=(256, 256),
        range=((0, 256), (0, 256)),
    )
    hist_b, _, _ = np.histogram2d(
        hue.reshape(-1), fb.reshape(-1), bins=(256, 256),
        range=((0, 256), (0, 256)),
    )
    scores = np.zeros_like(hue, dtype=np.float32)
    valid_hue = (hue >= 0) & (hue < 256)
    # The C# implementation skips pixels with fa<0 or fb>=256 before its
    # two out-of-range branches; preserve that precedence explicitly.
    branchable = valid_hue & (fa >= 0) & (fb < 256)
    scores[branchable & (fa >= 256)] = 255.0
    scores[branchable & (fb < 0) & (fa < 256)] = -255.0
    valid = branchable & (fa < 256) & (fb >= 0)
    if np.any(valid):
        h_index = np.clip(hue[valid].astype(np.int32), 0, 255)
        a_index = np.clip(fa[valid].astype(np.int32), 0, 255)
        b_index = np.clip(fb[valid].astype(np.int32), 0, 255)
        ha = hist_a[h_index, a_index]
        hb = hist_b[h_index, b_index]
        scores[valid] = np.where(
            ha > hb, 0.0, np.where(np.isclose(ha, hb), 100.0, 255.0)
        )

    result = scores.sum(axis=1, dtype=np.float32)
    if not np.any(result):
        return None, 0.0
    resized = cv2.resize(
        result.reshape(1, -1), (ORIENTATION_ANGLE_BINS * 2, 1),
        interpolation=cv2.INTER_CUBIC,
    ).reshape(-1)
    peak_width = ORIENTATION_ANGLE_BINS // 4 * 2 + 1
    shifted = np.roll(resized, peak_width)
    peak_region_sum = float(shifted[:peak_width].sum())
    integral = np.concatenate(([0.0], np.cumsum(resized - shifted)))
    max_index = int(np.argmax(integral))
    max_value = float(integral[max_index])
    degree = (max_index - 1) / 2.0 - 45.0
    if degree < 0:
        degree += 360.0
    confidence = (max_value + peak_region_sum) / (peak_width * 60 * 255)
    return float(degree % 360), max(0.0, float(confidence))


def _sector_orientation(minimap: np.ndarray) -> float | None:
    """Legacy brightness-sector fallback used when polar edges are weak."""

    if not isinstance(minimap, np.ndarray) or minimap.size == 0:
        return None
    gray = (
        cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
        if minimap.ndim == 3 else minimap
    )
    if gray.ndim != 2 or gray.shape[0] < 16 or gray.shape[1] < 16:
        return None
    size = min(gray.shape[:2])
    center = size / 2.0
    angles = np.linspace(0, 2 * math.pi, 360, endpoint=False)
    ring_radius = size * 0.42
    xs = np.rint(center + ring_radius * np.cos(angles)).astype(int)
    ys = np.rint(center + ring_radius * np.sin(angles)).astype(int)
    xs = np.clip(xs, 0, gray.shape[1] - 1)
    ys = np.clip(ys, 0, gray.shape[0] - 1)
    profile = gray[ys, xs].astype(np.float32)
    profile -= profile.mean()
    spread = float(profile.std())
    if spread <= 1e-3:
        return None
    kernel = np.zeros(360, np.float32)
    kernel[:90] = 1.0
    kernel -= kernel.mean()
    correlation = np.real(np.fft.ifft(
        np.fft.fft(profile) * np.conj(np.fft.fft(kernel))
    ))
    peak = float(correlation.max())
    if peak < spread * 30:
        return None
    center_index = (int(np.argmax(correlation)) + 45) % 360
    return float((center_index + 90) % 360)


def orientation_with_confidence(minimap: np.ndarray) -> tuple[float | None, float]:
    """Use the upstream detector and retain the compatible fallback."""

    angle, confidence = bettergi_orientation_with_confidence(minimap)
    if angle is not None and confidence >= ORIENTATION_CONFIDENCE_THRESHOLD:
        return angle, confidence
    legacy_angle, legacy_confidence = gia_orientation_with_confidence(minimap)
    if legacy_angle is not None and legacy_confidence >= ORIENTATION_CONFIDENCE_THRESHOLD:
        return legacy_angle, legacy_confidence
    fallback = _sector_orientation(minimap)
    if fallback is not None:
        # The fallback has no calibrated probability; distinguish it from a
        # high-confidence polar result without making it unusable to callers.
        return fallback, max(0.0, confidence)
    # A low-confidence polar result is not useful for a steering command. Do
    # not expose it merely because the numerical pipeline had a flat maximum.
    return None, max(confidence, legacy_confidence)


def crop_minimap_for_orientation(ctx: Any, frame: np.ndarray) -> np.ndarray | None:
    """Crop and normalize the minimap from a full iOS screenshot."""

    if not isinstance(frame, np.ndarray) or frame.ndim < 2:
        return None
    buttons = getattr(getattr(ctx, "layout", None), "buttons", {})
    center = buttons.get("minimapCenter") if hasattr(buttons, "get") else None
    if center is None or len(center) < 2:
        return None
    height, width = frame.shape[:2]
    side = int(round(2 * MINIMAP_RADIUS_N * width))
    radius = side / 2.0
    cx, cy = float(center[0]) * width, float(center[1]) * height
    x0, y0 = int(round(cx - radius)), int(round(cy - radius))
    if side <= 0 or x0 < 0 or y0 < 0 or x0 + side > width or y0 + side > height:
        return None
    crop = frame[y0:y0 + side, x0:x0 + side]
    if crop.shape[:2] != (ORIENTATION_MINIMAP_SIZE, ORIENTATION_MINIMAP_SIZE):
        crop = cv2.resize(
            crop,
            (ORIENTATION_MINIMAP_SIZE, ORIENTATION_MINIMAP_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
    return crop
