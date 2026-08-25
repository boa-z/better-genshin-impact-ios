"""BetterGI minimap preprocessing for orientation and map matching.

The desktop implementation removes HUD icons, compensates for the minimap's
radial alpha fade, and builds a mask before matching the view against map
assets.  This module keeps that image-only part independent from the device
and pathing layers so it can be exercised with recorded iOS frames.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np


MINIMAP_CAPTURE_RADIUS_N = 0.042
MINIMAP_NORMALIZED_SIZE = 212
MINIMAP_PROCESS_SIZE = 156


def prepare_minimap_source(
    minimap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return BetterGI's central 156px source and icon mask.

    ``MaskCalculator.Process1`` receives a normalized 212px color minimap and
    takes the central 156px without another resize.  The mask contains 255
    for likely monochrome HUD/icon pixels and 0 elsewhere.
    """

    if (
        not isinstance(minimap, np.ndarray)
        or minimap.ndim != 3
        or minimap.shape[2] < 3
    ):
        return None
    height, width = minimap.shape[:2]
    if min(height, width) < MINIMAP_PROCESS_SIZE:
        return None

    x0 = (width - MINIMAP_PROCESS_SIZE) // 2
    y0 = (height - MINIMAP_PROCESS_SIZE) // 2
    source = np.ascontiguousarray(
        minimap[
            y0:y0 + MINIMAP_PROCESS_SIZE,
            x0:x0 + MINIMAP_PROCESS_SIZE,
            :3,
        ]
    )

    # Port of MaskCalculator.CreateIconMask.  Only monochrome mid-tone
    # pixels participate in the icon branch; treating every grayscale map
    # pixel as an icon removes the actual terrain signal.
    cmax = source.max(axis=2).astype(np.float32)
    cmin = source.min(axis=2).astype(np.float32)
    icon_pixels = (cmax == cmin) & (cmax >= 50) & (cmax <= 127)
    diff = cmax - cmin
    np.minimum((255.0 - cmax) / 6.0, diff, out=diff)
    diff += 10.0
    normalized = np.divide(
        np.where(icon_pixels, 255.0, cmax) * 10.0,
        diff,
        out=np.zeros_like(diff),
        where=diff != 0,
    )
    mask = np.where(normalized > 200.0, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return source, mask


@lru_cache(maxsize=1)
def _radial_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the radius/angle and alpha LUTs used by MaskCalculator."""

    values = np.linspace(
        -MINIMAP_PROCESS_SIZE / 2,
        MINIMAP_PROCESS_SIZE / 2,
        MINIMAP_PROCESS_SIZE,
        endpoint=False,
        dtype=np.float32,
    )
    x_grid, y_grid = np.meshgrid(values, values)
    radius_float = np.hypot(x_grid, y_grid)
    angle_float = np.degrees(np.arctan2(y_grid, x_grid)) % 360.0
    # MaskCalculator stores radius as uint8 and divides the angle by two
    # before storing it as uint8, giving a compact 0..180 angle LUT.
    radius = np.clip(np.rint(radius_float), 0, 255).astype(np.uint8)
    angle = np.clip(np.rint(angle_float / 2.0), 0, 255).astype(np.uint8)

    alpha_params = np.array(
        [18.632, 20.157, 24.093, 34.617, 38.566, 41.94, 47.654,
         51.087, 58.561, 63.925, 67.759, 71.77, 75.214],
        dtype=np.float32,
    )
    alpha1 = 229.0 + np.searchsorted(alpha_params, radius, side="left")
    alpha2 = np.minimum(255.0, 137.0 + 1.43 * radius)
    return (
        radius,
        angle,
        alpha1.astype(np.float32),
        alpha2.astype(np.float32),
    )


def _apply_alpha_mask(
    image: np.ndarray,
    alpha: np.ndarray,
    background: float,
) -> np.ndarray:
    alpha_view = (
        alpha[..., None]
        if image.ndim == 3 and alpha.ndim == 2
        else alpha
    )
    return (image.astype(np.float32) - background) * (255.0 / alpha_view) + background


def _background_mask(image: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    """Port MaskCalculator.CreateBgMask and keep only the circular map area."""

    radius, angle, _, _ = _radial_tables()
    result = cv2.bitwise_not(base_mask)
    background = cv2.inRange(
        image,
        np.array((165, 165, 55), dtype=np.uint8),
        np.array((180, 180, 75), dtype=np.uint8),
    )
    background = cv2.morphologyEx(
        background,
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
    )
    if cv2.countNonZero(background):
        min_distance = np.full(256, 255, dtype=np.uint8)
        selected = background == 255
        np.minimum.at(min_distance, angle[selected], radius[selected])
        distance_lut = min_distance[angle]
        inner_background = np.where(radius < distance_lut, 255, 0).astype(np.uint8)
        bright = cv2.inRange(
            image,
            np.array((100, 100, 100), dtype=np.uint8),
            np.array((255, 255, 255), dtype=np.uint8),
        )
        result = cv2.bitwise_or(result, bright)
        result = cv2.bitwise_or(result, inner_background)

    circle = np.zeros_like(result)
    cv2.circle(
        circle,
        (MINIMAP_PROCESS_SIZE // 2, MINIMAP_PROCESS_SIZE // 2),
        MINIMAP_PROCESS_SIZE // 2,
        255,
        -1,
    )
    return cv2.bitwise_and(circle, result)


def preprocess_minimap_for_matching(
    minimap: np.ndarray,
    angle: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(gray_query, valid_mask)`` for map feature matching.

    ``angle`` is the camera orientation in BetterGI's map convention.  When
    it is unavailable, the radial compensation and icon mask are still useful
    and the sector blanking step is skipped instead of inventing a turn.
    """

    prepared = prepare_minimap_source(minimap)
    if prepared is None:
        return None
    source, icon_mask = prepared
    _, _, alpha1, alpha2 = _radial_tables()

    sector_mask = np.repeat(alpha2[..., None], 3, axis=2).astype(np.uint8)
    if angle is not None:
        cv2.ellipse(
            sector_mask,
            (MINIMAP_PROCESS_SIZE // 2, MINIMAP_PROCESS_SIZE // 2),
            (MINIMAP_PROCESS_SIZE, MINIMAP_PROCESS_SIZE),
            0,
            float(angle) + 45.5,
            float(angle) + 314.5,
            (255, 255, 255),
            -1,
        )

    output = source.astype(np.float32)
    output = _apply_alpha_mask(output, sector_mask.astype(np.float32), 255.0)
    output = _apply_alpha_mask(output, alpha1, 0.0)
    output = np.clip(np.rint(output), 0, 255).astype(np.uint8)
    valid_mask = _background_mask(output, icon_mask)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    return gray, valid_mask
