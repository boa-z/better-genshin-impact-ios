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


def is_main_ui(ctx, bgr: np.ndarray) -> bool:
    """Return whether the normal gameplay HUD is visible."""

    return ImageRegion(ctx, bgr).find(PAIMON_HUD).is_exist()


def is_big_map_ui(ctx, bgr: np.ndarray) -> bool:
    """Return whether the safe-area-aware big-map close button is visible."""

    return ImageRegion(ctx, bgr).find(MAP_CLOSE).is_exist()
