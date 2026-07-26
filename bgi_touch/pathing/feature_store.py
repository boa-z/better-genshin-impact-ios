"""BetterGI 官方地图 SIFT 特征库加载与查询。

资产格式（BetterGI.Assets.Map NuGet 包，tools/fetch_map_assets.py 获取）：
- <Map>_<layer>_<blocksize>_SIFT.kp.bin —— 关键点数组，每个 28 字节：
  x, y, size, angle, response (float32×5) + octave, class_id (int32×2)，小端。
- <Map>_<layer>_<blocksize>_SIFT.mat.png —— N×128 uint8 灰度 PNG，即 SIFT 描述子矩阵。

查询策略与原版一致：优先在上次位置附近的局部子集内匹配，失败回退全局。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

KP_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("size", "<f4"), ("angle", "<f4"),
    ("response", "<f4"), ("octave", "<i4"), ("class_id", "<i4"),
])


@dataclass
class MatchResult:
    x: float          # 大地图图像像素坐标（特征所在尺度）
    y: float
    inliers: int
    scale: float      # minimap→map 的估计尺度（诊断用）


class SiftFeatureStore:
    def __init__(self, kp_path: str | Path, mat_path: str | Path):
        raw = np.fromfile(str(kp_path), dtype=KP_DTYPE)
        self.pts = np.stack([raw["x"], raw["y"]], axis=1).astype(np.float32)  # (N,2)
        desc = cv2.imread(str(mat_path), cv2.IMREAD_GRAYSCALE)
        if desc is None or desc.shape[0] != len(raw):
            raise ValueError(f"描述子与关键点数量不一致: {mat_path}")
        self.desc = np.ascontiguousarray(desc, dtype=np.uint8)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

    def __len__(self) -> int:
        return len(self.pts)

    # OpenCV BFMatcher 的单个训练集上限为 2^18 行，超过则分块匹配后合并 knn
    _CHUNK = 200_000

    def _knn2(self, query_desc: np.ndarray, idx: np.ndarray) -> list[tuple[float, float, int]]:
        """返回每个 query 的 (最近距离, 次近距离, 最近的全局训练索引)。"""
        q = query_desc.astype(np.float32)
        nq = len(q)
        d1 = np.full(nq, np.inf, np.float64)
        d2 = np.full(nq, np.inf, np.float64)
        i1 = np.full(nq, -1, np.int64)
        for chunk in np.array_split(idx, max(1, int(np.ceil(len(idx) / self._CHUNK)))):
            if len(chunk) < 2:
                continue
            knn = self._matcher.knnMatch(q, self.desc[chunk].astype(np.float32), k=2)
            for qi, pair in enumerate(knn):
                for m in pair:
                    d = m.distance
                    if d < d1[qi]:
                        d2[qi] = d1[qi]
                        d1[qi] = d
                        i1[qi] = chunk[m.trainIdx]
                    elif d < d2[qi]:
                        d2[qi] = d
        return [(d1[i], d2[i], int(i1[i])) for i in range(nq)]

    def _match_subset(self, query_desc: np.ndarray, query_pts: np.ndarray,
                      idx: np.ndarray, min_matches: int = 8) -> MatchResult | None:
        """query 特征 ↔ 库子集匹配，仿射估计后返回 minimap 中心对应的地图坐标。"""
        if len(idx) < min_matches:
            return None
        knn = self._knn2(query_desc, idx)
        good = [(qi, ti) for qi, (da, db, ti) in enumerate(knn)
                if ti >= 0 and da < 0.75 * db]
        if len(good) < min_matches:
            return None
        src = np.float32([query_pts[qi] for qi, _ in good]).reshape(-1, 1, 2)
        dst = np.float32([self.pts[ti] for _, ti in good]).reshape(-1, 1, 2)
        M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=5.0)
        if M is None or mask is None or int(mask.sum()) < min_matches:
            return None
        scale = float(np.hypot(M[0, 0], M[0, 1]))
        return M, int(mask.sum()), scale

    def locate(self, query_desc: np.ndarray, query_pts: np.ndarray,
               query_center: tuple[float, float],
               prev: tuple[float, float] | None = None,
               local_radius: float = 2048.0) -> MatchResult | None:
        """定位 query 图像中心点在大地图上的像素坐标。

        prev 给定时先在其半径 local_radius 内的特征子集中匹配（快且稳），
        失败或未给定时回退全局匹配。
        """
        if query_desc is None or len(query_desc) < 8:
            return None
        attempts = []
        if prev is not None:
            d2 = ((self.pts[:, 0] - prev[0]) ** 2 + (self.pts[:, 1] - prev[1]) ** 2)
            attempts.append(np.nonzero(d2 <= local_radius ** 2)[0])
        attempts.append(np.arange(len(self.pts)))
        for idx in attempts:
            r = self._match_subset(query_desc, query_pts, idx)
            if r is not None:
                M, inliers, scale = r
                cx, cy = query_center
                mapped = M @ np.array([cx, cy, 1.0], dtype=np.float64)
                return MatchResult(float(mapped[0]), float(mapped[1]), inliers, scale)
        return None
