"""BetterGI map-mask tracking state for the iOS/WebUI runtime.

The DeviceHub capture stream remains owned by :class:`TriggerLoop`.  This
trigger only consumes frames already captured by that loop and publishes a
small immutable snapshot for WebUI polling.  SIFT work runs on one background
worker which keeps only the newest pending frame, matching BetterGI's
``MapMaskTrigger`` back-pressure semantics.
"""

from __future__ import annotations

import copy
import struct
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..pathing.map_locator import (
    ASSETS,
    MAP_DEFINITIONS,
    MapConfig,
    get_map_definition,
    resolve_map_name,
)
from ..pathing.positioner import MinimapPositioner
from ..pathing.tp import BigMapLocator
from ..vision.game_ui import is_big_map_ui, is_main_ui

MINIMAP_VIEW_WORLD_SIZE = 360.0


def map_image_path(map_name: str, layer: int = 0) -> Path | None:
    """Resolve a display map asset without accepting arbitrary paths."""

    definition = get_map_definition(map_name)
    base = ASSETS / definition.name
    stem = f"{definition.name}_{int(layer)}_{definition.big_map_scale}"
    for suffix in (".png", ".webp", ".jpg", ".jpeg"):
        candidate = base / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def available_map_layers(map_name: str) -> list[int]:
    definition = get_map_definition(map_name)
    base = ASSETS / definition.name
    layers: set[int] = set()
    for path in base.glob(f"{definition.name}_*_{definition.big_map_scale}.*"):
        if "_SIFT." in path.name:
            continue
        middle = path.name.removeprefix(f"{definition.name}_").split("_", 1)[0]
        try:
            layer = int(middle)
        except ValueError:
            continue
        if map_image_path(definition.name, layer) is not None:
            layers.add(layer)
    return sorted(layers, key=lambda value: (value != 0, value))


@lru_cache(maxsize=32)
def map_image_size(path: str) -> tuple[int, int]:
    with open(path, "rb") as stream:
        header = stream.read(32)
    if header.startswith(b"\x89PNG\r\n\x1a\n") and header[12:16] == b"IHDR":
        return tuple(int(value) for value in struct.unpack(">II", header[16:24]))
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        chunk = header[12:16]
        if chunk == b"VP8X" and len(header) >= 30:
            width = 1 + int.from_bytes(header[24:27], "little")
            height = 1 + int.from_bytes(header[27:30], "little")
            return width, height
        if chunk == b"VP8 " and len(header) >= 30 and header[23:26] == b"\x9d\x01\x2a":
            width = int.from_bytes(header[26:28], "little") & 0x3FFF
            height = int.from_bytes(header[28:30], "little") & 0x3FFF
            return width, height
        if chunk == b"VP8L" and len(header) >= 25 and header[20] == 0x2F:
            b1, b2, b3, b4 = header[21:25]
            width = 1 + b1 + ((b2 & 0x3F) << 8)
            height = 1 + (b2 >> 6) + (b3 << 2) + ((b4 & 0x0F) << 10)
            return width, height
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"无法读取地图图像: {path}")
    return int(image.shape[1]), int(image.shape[0])


def map_catalog() -> list[dict]:
    catalog = []
    for definition in MAP_DEFINITIONS.values():
        layers = []
        for layer in available_map_layers(definition.name):
            path = map_image_path(definition.name, layer)
            if path is None:
                continue
            width, height = map_image_size(str(path))
            layers.append({
                "id": layer,
                "width": width,
                "height": height,
                "url": f"/api/map-mask/maps/{definition.name}/layers/{layer}/image",
            })
        catalog.append({
            "name": definition.name,
            "displayName": definition.aliases[0] if definition.aliases else definition.name,
            "aliases": list(definition.aliases),
            "available": bool(layers),
            "layers": layers,
        })
    return catalog


class MapMaskState:
    """Thread-safe latest-only map tracking snapshot."""

    def __init__(self, map_name: str):
        self._lock = threading.Lock()
        self._sequence = 0
        self._updated_at = time.monotonic()
        self._data = {
            "active": True,
            "mapName": resolve_map_name(map_name),
            "layer": 0,
            "scene": "waiting",
            "inBigMapUi": False,
            "positionValid": False,
            "position": None,
            "viewport": None,
            "error": None,
        }

    def update(self, **values) -> None:
        with self._lock:
            changed = any(self._data.get(key) != value for key, value in values.items())
            self._data.update(values)
            self._updated_at = time.monotonic()
            if changed:
                self._sequence += 1

    def deactivate(self) -> None:
        self.update(active=False, scene="disabled", inBigMapUi=False,
                    positionValid=False, viewport=None)

    def snapshot(self) -> dict:
        with self._lock:
            result = copy.deepcopy(self._data)
            result["sequence"] = self._sequence
            result["ageMs"] = round(max(0.0, time.monotonic() - self._updated_at) * 1000)
            return result


@dataclass(frozen=True)
class _WorkItem:
    frame: np.ndarray


class MapMaskTrigger:
    """Locate the player or current big-map viewport from TriggerLoop frames."""

    name = "MapMask"

    def __init__(
        self,
        ctx,
        *,
        map_name: str = "Teyvat",
        mini_map_enabled: bool = True,
        log: Callable[[str], None] = print,
        positioner=None,
        big_locator=None,
        main_ui_detector: Callable[[object, np.ndarray], bool] = is_main_ui,
        big_map_detector: Callable[[object, np.ndarray], bool] = is_big_map_ui,
    ):
        self.ctx = ctx
        self.map_name = resolve_map_name(map_name)
        self.mini_map_enabled = bool(mini_map_enabled)
        self.log = log
        self.enabled = True
        self.state = MapMaskState(self.map_name)
        self.positioner = positioner or MinimapPositioner(ctx, self.map_name)
        self.big_locator = big_locator or BigMapLocator(self.map_name)
        self._main_ui_detector = main_ui_detector
        self._big_map_detector = big_map_detector
        self._pending: _WorkItem | None = None
        self._worker_running = False
        self._work_lock = threading.Lock()
        self._idle = threading.Condition(self._work_lock)

    def on_frame(self, region) -> None:
        if not self.enabled:
            return
        item = _WorkItem(region.bgr.copy())
        with self._work_lock:
            self._pending = item
            if self._worker_running:
                return
            self._worker_running = True
        threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="map-mask-locator",
        ).start()

    def _worker_loop(self) -> None:
        while True:
            with self._work_lock:
                item = self._pending
                self._pending = None
                if item is None or not self.enabled:
                    self._worker_running = False
                    self._idle.notify_all()
                    return
            try:
                self._process(item.frame)
            except Exception as error:  # one bad frame must not stop the trigger
                if self.enabled:
                    self.state.update(error=str(error))
                    self.log(f"[MapMask] 地图定位失败: {error}")

    def _display_position(self, world: tuple[float, float]) -> dict:
        definition = get_map_definition(self.map_name)
        native_x, native_y = MapConfig.for_map(self.map_name).world_to_image(*world)
        ratio = definition.big_map_scale / definition.feature_scale
        return {
            "worldX": round(float(world[0]), 3),
            "worldY": round(float(world[1]), 3),
            "imageX": round(float(native_x * ratio), 3),
            "imageY": round(float(native_y * ratio), 3),
        }

    def _mini_viewport(self, position: dict, layer: int) -> dict:
        definition = get_map_definition(self.map_name)
        pixels_per_world = definition.coordinate_scale * (
            definition.big_map_scale / definition.feature_scale
        )
        size = MINIMAP_VIEW_WORLD_SIZE * pixels_per_world
        return {
            "x": round(position["imageX"] - size / 2, 3),
            "y": round(position["imageY"] - size / 2, 3),
            "width": round(size, 3),
            "height": round(size, 3),
            "layer": int(layer),
        }

    def _process(self, frame: np.ndarray) -> None:
        if self._main_ui_detector(self.ctx, frame):
            if not self.mini_map_enabled:
                self.state.update(scene="gameplay", inBigMapUi=False,
                                  positionValid=False, viewport=None, error=None)
                return
            world = self.positioner.get_position_stable(frame)
            if not self.enabled:
                return
            layer = getattr(getattr(self.positioner, "locator", None), "last_layer", 0)
            layer = 0 if layer is None else int(layer)
            if world is None:
                self.state.update(scene="gameplay", inBigMapUi=False,
                                  positionValid=False, viewport=None, error=None)
                return
            position = self._display_position(world)
            self.state.update(
                scene="gameplay",
                inBigMapUi=False,
                layer=layer,
                positionValid=True,
                position=position,
                viewport=self._mini_viewport(position, layer),
                error=None,
            )
            return

        if not self._big_map_detector(self.ctx, frame):
            self.state.update(scene="other", inBigMapUi=False,
                              positionValid=False, viewport=None, error=None)
            return

        match = self.big_locator.locate_view(frame)
        if not self.enabled:
            return
        if match is None:
            self.state.update(scene="other", inBigMapUi=False,
                              positionValid=False, viewport=None, error=None)
            return
        center_x, center_y, pixels_per_feature = match
        layer = getattr(self.big_locator, "last_layer", 0)
        layer = 0 if layer is None else int(layer)
        world = self.big_locator.feature_to_world(center_x, center_y)
        # A global SIFT query is deliberately conservative and can reject a
        # city minimap covered by icons.  A successfully located big-map view
        # gives the same coarse prior BetterGI carries between frames, allowing
        # the next gameplay frame to use the much less ambiguous local subset.
        set_prior = getattr(self.positioner, "set_prior", None)
        if callable(set_prior):
            set_prior(*world)
        position = self._display_position(world)
        self.state.update(
            scene="bigMap",
            inBigMapUi=True,
            layer=layer,
            positionValid=True,
            position=position,
            viewport={
                "x": round(center_x - frame.shape[1] / pixels_per_feature / 2, 3),
                "y": round(center_y - frame.shape[0] / pixels_per_feature / 2, 3),
                "width": round(frame.shape[1] / pixels_per_feature, 3),
                "height": round(frame.shape[0] / pixels_per_feature, 3),
                "layer": layer,
            },
            error=None,
        )

    def wait_idle(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._work_lock:
            while self._worker_running or self._pending is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
            return True

    def close(self) -> None:
        self.enabled = False
        with self._work_lock:
            self._pending = None
        self.state.deactivate()
