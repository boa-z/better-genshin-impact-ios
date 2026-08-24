"""BetterGI ``tp.json`` teleport-point index.

The desktop implementation resolves ``genshin.tp(x, y)`` to the closest
known teleport point before opening the map.  Keeping that lookup separate
from touch/OCR code makes the coordinate policy deterministic and lets the
mobile implementation keep the same ``force`` contract.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TeleportPoint:
    map_name: str
    point_id: str
    point_type: str
    name: str
    country: str | None
    areas: tuple[str, ...]
    x: float
    y: float
    transfer_x: float
    transfer_y: float

    @classmethod
    def from_mapping(cls, map_name: str, raw: Any) -> "TeleportPoint | None":
        if not isinstance(raw, dict):
            return None
        position = raw.get("position")
        transfer = raw.get("tranPosition", raw.get("transferPosition"))
        if not isinstance(position, (list, tuple)) or len(position) < 3:
            return None
        if not isinstance(transfer, (list, tuple)) or len(transfer) < 3:
            transfer = position
        try:
            # BetterGI's x/y script coordinates are the world position's
            # horizontal (index 2) and vertical (index 0) components.
            x = float(position[2])
            y = float(position[0])
            transfer_x = float(transfer[2])
            transfer_y = float(transfer[0])
        except (TypeError, ValueError):
            return None
        values = (x, y, transfer_x, transfer_y)
        if not all(math.isfinite(value) for value in values):
            return None
        areas = raw.get("areas")
        if isinstance(areas, str):
            areas = (areas,)
        elif isinstance(areas, (list, tuple)):
            areas = tuple(str(value).strip() for value in areas if str(value).strip())
        else:
            areas = ()
        return cls(
            map_name=str(map_name),
            point_id=str(raw.get("id", "")),
            point_type=str(raw.get("type", "") or ""),
            name=str(raw.get("name", "") or ""),
            country=(
                str(raw["country"]).strip()
                if raw.get("country") is not None and str(raw["country"]).strip()
                else None
            ),
            areas=areas,
            x=x,
            y=y,
            transfer_x=transfer_x,
            transfer_y=transfer_y,
        )


def _candidate_paths() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[2]
    original_root = root.parent / "better-genshin-impact"
    candidates: list[Path] = []
    configured = os.environ.get("BGI_TP_POINTS_PATH", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend((
        root / "assets" / "data" / "tp.json",
        original_root / "BetterGenshinImpact" / "GameTask" /
        "AutoTrackPath" / "Assets" / "tp.json",
    ))
    return tuple(dict.fromkeys(path.resolve() for path in candidates))


class TeleportPointStore:
    """Lazy, read-only index of the upstream teleport-point asset."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser().resolve() if path is not None else None
        self._scenes: dict[str, tuple[TeleportPoint, ...]] | None = None

    @property
    def scenes(self) -> dict[str, tuple[TeleportPoint, ...]]:
        if self._scenes is None:
            self._scenes = self._load()
        return self._scenes

    def _load(self) -> dict[str, tuple[TeleportPoint, ...]]:
        source = self.path
        if source is None:
            source = next((candidate for candidate in _candidate_paths() if candidate.is_file()), None)
        if source is None:
            return {}
        try:
            raw = json.loads(source.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return {}
        data = raw.get("data") if isinstance(raw, dict) else None
        if not isinstance(data, list):
            return {}
        scenes: dict[str, tuple[TeleportPoint, ...]] = {}
        for scene in data:
            if not isinstance(scene, dict):
                continue
            map_name = str(scene.get("mapName", "")).strip()
            points = scene.get("points")
            if not map_name or not isinstance(points, list):
                continue
            parsed = tuple(
                point for raw_point in points
                if (point := TeleportPoint.from_mapping(map_name, raw_point)) is not None
            )
            if parsed:
                scenes[map_name] = parsed
        return scenes

    def nearest(
        self,
        map_name: str,
        x: float,
        y: float,
        count: int = 1,
    ) -> tuple[TeleportPoint, ...]:
        if count < 1:
            raise ValueError("count 必须大于等于 1")
        try:
            target_x, target_y = float(x), float(y)
        except (TypeError, ValueError) as error:
            raise ValueError("传送点坐标必须是数字") from error
        if not math.isfinite(target_x) or not math.isfinite(target_y):
            raise ValueError("传送点坐标必须是有限数字")
        points = self.scenes.get(str(map_name), ())
        return tuple(sorted(
            points,
            key=lambda point: math.hypot(point.x - target_x, point.y - target_y),
        )[:count])

    def nearest_point(self, map_name: str, x: float, y: float) -> TeleportPoint | None:
        return next(iter(self.nearest(map_name, x, y, 1)), None)


@lru_cache(maxsize=1)
def default_teleport_point_store() -> TeleportPointStore:
    return TeleportPointStore()
