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


def is_main_ui(ctx, bgr: np.ndarray) -> bool:
    """Return whether the normal gameplay HUD is visible."""

    return ImageRegion(ctx, bgr).find(PAIMON_HUD).is_exist()
