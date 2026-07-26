"""提瓦特大地图定位器：小地图截图 → 地图像素坐标 → 原神世界坐标。

两级匹配（对应资产的两个尺度库）：
- 全局：256 尺度库（~3 万特征）粗定位，BF 可承受；
- 精化：坐标 ×8 到 2048 尺度库，在邻域子集内精匹配；
- 有上次位置时跳过全局，直接 2048 库局部匹配（原版同策略）。

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

SCALE_FINE = 2048
SCALE_COARSE = 256
FINE_PER_COARSE = SCALE_FINE / SCALE_COARSE  # 8


@dataclass
class MapConfig:
    """世界坐标 ↔ 2048 尺度图像像素（移植自 TeyvatMap.cs / SceneBaseMap.cs）。

    Teyvat：原点在图像 (32768, 16384)，块宽 2048 → scale = 2048/1024 = 2，
    双轴反向：img = origin − world × scale。
    """
    origin_x: float = 32768.0  # (GameMapLeftCols+1) × 2048
    origin_y: float = 16384.0  # (GameMapUpRows+1) × 2048
    scale: float = 2.0         # mapImageBlockWidth / 1024

    def world_to_image(self, wx: float, wy: float) -> tuple[float, float]:
        return (self.origin_x - wx * self.scale, self.origin_y - wy * self.scale)

    def image_to_world(self, ix: float, iy: float) -> tuple[float, float]:
        return ((self.origin_x - ix) / self.scale, (self.origin_y - iy) / self.scale)


class MapLocator:
    def __init__(self, map_name: str = "Teyvat", config: MapConfig | None = MapConfig()):
        base = ASSETS / map_name
        if not (base / f"{map_name}_0_{SCALE_FINE}_SIFT.kp.bin").exists():
            raise FileNotFoundError(
                f"缺少地图特征资产 {base}。先运行 tools/fetch_map_assets.py")
        self.fine = SiftFeatureStore(base / f"{map_name}_0_{SCALE_FINE}_SIFT.kp.bin",
                                     base / f"{map_name}_0_{SCALE_FINE}_SIFT.mat.png")
        self.coarse = SiftFeatureStore(base / f"{map_name}_0_{SCALE_COARSE}_SIFT.kp.bin",
                                       base / f"{map_name}_0_{SCALE_COARSE}_SIFT.mat.png")
        self.config = config
        self._sift = cv2.SIFT_create()
        self._lock = threading.Lock()
        self.prev: tuple[float, float] | None = None  # 2048 尺度像素

    # ---- 小地图帧处理 ----

    def _extract(self, minimap_gray: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        h, w = minimap_gray.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (w // 2, h // 2), int(min(h, w) * 0.46), 255, -1)
        kps, desc = self._sift.detectAndCompute(minimap_gray, mask)
        pts = np.float32([k.pt for k in kps]) if kps else np.zeros((0, 2), np.float32)
        return desc, pts

    def locate_pixel(self, minimap_gray: np.ndarray) -> tuple[float, float] | None:
        """小地图灰度图 → 2048 尺度地图像素坐标。线程安全。"""
        with self._lock:
            desc, pts = self._extract(minimap_gray)
            if desc is None:
                return None
            center = (minimap_gray.shape[1] / 2, minimap_gray.shape[0] / 2)

            # ① 有历史位置：2048 库局部匹配（原版为上次位置 3×3 邻域块，
            #    块宽 1024 → 等效半径 ~1536，取 2048 留余量）
            if self.prev is not None:
                r = self.fine.locate(desc, pts, center, prev=self.prev, local_radius=2048)
                if r is not None:
                    self.prev = (r.x, r.y)
                    return self.prev

            # ② 全局回退：2048 库分块全量匹配（较慢，仅在丢失时使用）
            r = self.fine.locate(desc, pts, center)
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
