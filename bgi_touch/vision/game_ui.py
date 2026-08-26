"""Shared BetterGI game-UI state recognition.

The minimap remains visible behind the translucent Paimon menu on iOS, so a
circle-only heuristic cannot distinguish gameplay from that menu. BetterGI
uses the Paimon HUD icon for the same purpose; keep that contract in one place
for tasks and realtime triggers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..engine.recognition import ImageRegion, Mat, RecognitionObject


TEMPLATES = Path(__file__).resolve().parents[2] / "assets" / "templates" / "stygian"
TELEPORT_TEMPLATES = Path(__file__).resolve().parents[2] / "assets" / "templates" / "teleport"
QUICK_TELEPORT_TEMPLATES = Path(__file__).resolve().parents[2] / "assets" / "templates" / "quick_teleport"

PAIMON_HUD = RecognitionObject.template_match(
    Mat.from_file(str(TEMPLATES / "paimon_menu.png")),
    0,
    0,
    500,
    300,
)
# The mobile HUD applies its own safe-area scale rather than the PC 1080p
# scale. Real iPhone 13 Pro Max frames score around 0.53 while the Paimon menu
# and door screen stay below 0.40.
PAIMON_HUD.threshold = 0.50

MAP_CLOSE = RecognitionObject.template_match(
    Mat.from_file(str(TELEPORT_TEMPLATES / "MapCloseButton.png")),
    1600,
    0,
    320,
    140,
)
MAP_CLOSE.threshold = 0.65


def _map_marker(path: Path, roi, threshold: float):
    if not path.is_file():
        return None
    try:
        marker = RecognitionObject.template_match(Mat.from_file(str(path)), *roi)
        marker.threshold = threshold
        return marker
    except (OSError, ValueError, TypeError):
        return None


MAP_SCALE_BUTTON = _map_marker(
    QUICK_TELEPORT_TEMPLATES / "MapScaleButton.png",
    (30, 440, 40, 200),
    0.68,
)


# DeviceHub/iPhone frames can resample the small close button below the
# template threshold.  These two independent upper-right controls are only
# present on the same big-map overlay and provide a bounded fallback without
# scanning the whole frame with OCR.
MAP_FALLBACK_MARKERS = tuple(
    marker for marker in (
        _map_marker(
            QUICK_TELEPORT_TEMPLATES / "MapChoose.png",
            (1440, 0, 300, 120),
            0.68,
        ),
        _map_marker(
            QUICK_TELEPORT_TEMPLATES / "MapSettingsButton.png",
            (1550, 0, 370, 180),
            0.68,
        ),
    ) if marker is not None
)


def is_main_ui(ctx, bgr: np.ndarray) -> bool:
    """Return whether the normal gameplay HUD is visible."""

    region = ImageRegion(ctx, bgr)
    return region.find(PAIMON_HUD).is_exist() and not is_big_map_ui(ctx, bgr, region=region)


def is_big_map_ui(ctx, bgr: np.ndarray, *, region: ImageRegion | None = None) -> bool:
    """Return whether a safe-area-aware big-map marker is visible."""

    region = region or ImageRegion(ctx, bgr)
    if region.find(MAP_CLOSE).is_exist():
        return True
    if MAP_SCALE_BUTTON is not None and region.find(MAP_SCALE_BUTTON).is_exist():
        return True
    return any(region.find(marker).is_exist() for marker in MAP_FALLBACK_MARKERS)
