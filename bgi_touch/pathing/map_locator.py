"""BetterGI multi-map locator: minimap → map pixels → world coordinates.

提瓦特使用 2048 尺度特征；独立地图使用上游原生 1024 尺度特征。
有上次位置时优先做局部匹配，失败后遍历当前地图的主层与地下层做全局匹配。

世界坐标换算常数由 map_config 提供（从原版 TeyvatMap.cs 移植）。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .feature_store import SiftFeatureStore

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "map"

@dataclass(frozen=True)
class MapDefinition:
    name: str
    origin_x: float
    origin_y: float
    coordinate_scale: float
    feature_scale: int
    big_map_scale: int
    big_map_query_resize: float = 1.0
    aliases: tuple[str, ...] = ()

    @property
    def big_to_native_scale(self) -> float:
        return self.feature_scale / self.big_map_scale


MAP_DEFINITIONS = {
    definition.name: definition for definition in (
        MapDefinition(
            "Teyvat", 32768, 16384, 2.0, 2048, 256, 0.25,
            ("提瓦特", "提瓦特大陆"),
        ),
        MapDefinition("TheChasm", 2048, 2048, 1.0, 1024, 1024,
                      aliases=("层岩巨渊", "层岩巨渊地下矿区")),
        MapDefinition("Enkanomiya", 2048, 2048, 1.0, 1024, 1024,
                      aliases=("渊下宫",)),
        MapDefinition("SeaOfBygoneEras", 6144, 3072, 1.0, 1024, 1024,
                      aliases=("旧日之海",)),
        MapDefinition("AncientSacredMountain", 2048, 2048, 1.0, 1024, 1024,
                      aliases=("远古圣山",)),
        MapDefinition("TempleOfSpace", 2048, 2048, 1.0, 1024, 1024,
                      aliases=("空之神殿",)),
        MapDefinition("MoonCanon", 11264, 4096, 1.0, 1024, 1024,
                      aliases=("霜月",)),
    )
}


def resolve_map_name(value: str | None) -> str:
    raw = str(value or "Teyvat").strip()
    normalized = raw.replace(" ", "").replace("·", "").lower()
    for definition in MAP_DEFINITIONS.values():
        values = (definition.name, *definition.aliases)
        if any(normalized == item.replace(" ", "").replace("·", "").lower()
               for item in values):
            return definition.name
    raise ValueError(f"未知地图名称: {raw}")


def get_map_definition(value: str | None) -> MapDefinition:
    return MAP_DEFINITIONS[resolve_map_name(value)]


@dataclass
class MapConfig:
    """世界坐标 ↔ 地图原生特征尺度像素（移植自 SceneBaseMap.cs）。

    Teyvat：原点在图像 (32768, 16384)，块宽 2048 → scale = 2048/1024 = 2，
    双轴反向：img = origin − world × scale。
    """
    origin_x: float = 32768.0  # (GameMapLeftCols+1) × 2048
    origin_y: float = 16384.0  # (GameMapUpRows+1) × 2048
    scale: float = 2.0         # mapImageBlockWidth / 1024

    @classmethod
    def for_map(cls, map_name: str | None = None) -> "MapConfig":
        definition = get_map_definition(map_name)
        return cls(
            origin_x=definition.origin_x,
            origin_y=definition.origin_y,
            scale=definition.coordinate_scale,
        )

    def world_to_image(self, wx: float, wy: float) -> tuple[float, float]:
        return (self.origin_x - wx * self.scale, self.origin_y - wy * self.scale)

    def image_to_world(self, ix: float, iy: float) -> tuple[float, float]:
        return ((self.origin_x - ix) / self.scale, (self.origin_y - iy) / self.scale)


class MapLocator:
    def __init__(self, map_name: str = "Teyvat", config: MapConfig | None = None):
        self.definition = get_map_definition(map_name)
        self.map_name = self.definition.name
        base = ASSETS / self.map_name
        scale = self.definition.feature_scale
        keypoint_paths = sorted(
            base.glob(f"{self.map_name}_*_{scale}_SIFT.kp.bin"),
            key=lambda path: ("_0_" not in path.name, path.name),
        )
        if not keypoint_paths:
            raise FileNotFoundError(
                f"缺少地图特征资产 {base}。先运行 tools/fetch_map_assets.py")
        self.layers: list[SiftFeatureStore] = []
        for keypoints in keypoint_paths:
            descriptors = keypoints.with_name(
                keypoints.name.removesuffix(".kp.bin") + ".mat.png"
            )
            if descriptors.is_file():
                self.layers.append(SiftFeatureStore(keypoints, descriptors))
        if not self.layers:
            raise FileNotFoundError(f"地图描述子不完整: {base}")
        self.fine = self.layers[0]  # backward-compatible diagnostic handle
        self.coarse = self.fine
        self.config = config or MapConfig.for_map(self.map_name)
        self._sift = cv2.SIFT_create()
        self._lock = threading.Lock()
        self.prev: tuple[float, float] | None = None  # 当前地图原生特征尺度像素

    # ---- 小地图帧处理 ----

    def _extract(self, minimap_gray: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        h, w = minimap_gray.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (w // 2, h // 2), int(min(h, w) * 0.46), 255, -1)
        kps, desc = self._sift.detectAndCompute(minimap_gray, mask)
        pts = np.float32([k.pt for k in kps]) if kps else np.zeros((0, 2), np.float32)
        return desc, pts

    def locate_pixel(self, minimap_gray: np.ndarray) -> tuple[float, float] | None:
        """小地图灰度图 → 当前地图原生特征尺度像素坐标。线程安全。"""
        with self._lock:
            desc, pts = self._extract(minimap_gray)
            if desc is None:
                return None
            center = (minimap_gray.shape[1] / 2, minimap_gray.shape[0] / 2)

            # ① 有历史位置：在每层的一个原生地图块半径内局部匹配。
            if self.prev is not None:
                local_radius = float(self.definition.feature_scale)
                for layer in self.layers:
                    r = layer.locate(
                        desc, pts, center, prev=self.prev,
                        local_radius=local_radius,
                    )
                    if r is not None:
                        self.prev = (r.x, r.y)
                        return self.prev

            # ② 全局回退：逐层全量匹配（较慢，仅在丢失时使用）。
            for layer in self.layers:
                r = layer.locate(desc, pts, center)
                if r is not None:
                    self.prev = (r.x, r.y)
                    return self.prev
            return None

    def locate_world(self, minimap_gray: np.ndarray) -> tuple[float, float] | None:
        if self.config is None:
            raise RuntimeError("MapConfig 未设置（世界坐标换算常数）")
        p = self.locate_pixel(minimap_gray)
        return None if p is None else self.config.image_to_world(*p)

    def reset(self) -> None:
        self.prev = None
