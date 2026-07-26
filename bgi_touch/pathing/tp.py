"""大地图传送（genshin.tp 的移植）。

与原版 TpTask 的差异（移动端适配）：
- 原版用固定常数 2.361px/世界单位/缩放级换算拖动距离并用滚轮控制缩放；
  移动端改为**由 SIFT 匹配直接测出当前"屏幕像素/地图像素"比例**，
  把世界坐标差换算成滑动像素，迭代拖动直到目标进入容差，不需管理缩放级。
- 确认传送：原版按 F/点 GoTeleport 模板；移动端 OCR 找「传送」按钮点击，
  失败回退模板匹配传送锚点图标。
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..engine.context import GameContext
from ..engine.recognition import Mat, RecognitionObject
from ..vision.ocr import get_ocr
from .feature_store import SiftFeatureStore
from .map_locator import ASSETS, MapConfig

TEMPLATES = Path(__file__).resolve().parents[2] / "assets" / "templates" / "teleport"


class BigMapLocator:
    """全屏大地图截图 → 256 尺度地图坐标与比例。"""

    def __init__(self, map_name: str = "Teyvat"):
        base = ASSETS / map_name
        self.store = SiftFeatureStore(base / f"{map_name}_0_256_SIFT.kp.bin",
                                      base / f"{map_name}_0_256_SIFT.mat.png")
        self._sift = cv2.SIFT_create()

    def locate_view(self, bgr: np.ndarray) -> tuple[float, float, float] | None:
        """返回 (视野中心在 256 图的 x, y, 屏幕像素/256图像素 比例)；失败 None。

        按原版流程：灰度 + 1/4 缩放后匹配 256 库。
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        kps, desc = self._sift.detectAndCompute(small, None)
        if desc is None or len(kps) < 10:
            return None
        pts = np.float32([k.pt for k in kps])
        r = self.store.locate(desc, pts, (small.shape[1] / 2, small.shape[0] / 2))
        if r is None:
            return None
        # r.scale = 256图px / 1/4屏幕px → 屏幕px/256图px = 0.25*4/scale…直接算：
        # 1 个 256 图像素对应 (1/r.scale) 个 1/4 屏幕像素 = 4/r.scale 个屏幕像素
        if r.scale <= 1e-6:
            return None
        return r.x, r.y, 4.0 / r.scale


class TpTask:
    def __init__(self, ctx: GameContext, log: Callable[[str], None] = print,
                 map_name: str = "Teyvat"):
        self.ctx = ctx
        self.log = log
        self.big = BigMapLocator(map_name)
        self.config = MapConfig()

    # ---- 步骤 ----

    def open_map(self) -> bool:
        for _ in range(3):
            frame = self.ctx.capture_bgr()
            if self.big.locate_view(frame) is not None:
                return True
            self.ctx.input.tap_button("map")
            self.ctx.sleep(1800)
        return self.big.locate_view(self.ctx.capture_bgr()) is not None

    def _drag_map(self, dx: float, dy: float) -> None:
        """把地图内容平移 (dx, dy) 设备像素（正值=内容向右/下移动）。"""
        t = self.ctx.transform
        W, H = t.device_width, t.device_height
        max_step = 0.30 * W
        while abs(dx) > 1 or abs(dy) > 1:
            sx = max(-max_step, min(max_step, dx))
            sy = max(-max_step, min(max_step, dy))
            # 起点选屏幕中央偏移，避开左上返回键与右下 UI
            x0 = W * 0.5 - sx / 2
            y0 = H * 0.5 - sy / 2
            self.ctx.device.swipe(x0, y0, x0 + sx, y0 + sy, duration_ms=650,
                                  image_width=W, image_height=H)
            self.ctx.sleep(900)  # 等惯性衰减
            dx -= sx
            dy -= sy

    def _find_and_tap_confirm(self) -> bool:
        """在弹出卡片中找「传送」并点击。"""
        region = self.ctx.capture_region()
        hits = region.find_multi(RecognitionObject.ocr(960, 200, 960, 880), limit=20)
        for h in hits:
            if "传送" in h.text and "锚点" not in h.text and "追踪" not in h.text:
                h.click()
                return True
        return False

    def _tap_anchor_icon_near_center(self) -> bool:
        """回退：模板匹配屏幕中央附近的传送锚点/神像图标并点击最近者。"""
        region = self.ctx.capture_region()
        t = self.ctx.transform
        cx, cy = t.device_width / 2, t.device_height / 2
        best = None
        for name in ("TeleportWaypoint", "StatueOfTheSeven", "Domain"):
            tpl = Mat.from_file(str(TEMPLATES / f"{name}.png"))
            for h in region.find_multi(RecognitionObject.template_match(tpl), limit=5):
                d = math.hypot(h.dx + h.dw / 2 - cx, h.dy + h.dh / 2 - cy)
                if best is None or d < best[0]:
                    best = (d, h)
        if best and best[0] < 0.25 * t.device_width:
            best[1].click()
            return True
        return False

    def tp(self, wx: float, wy: float, timeout_s: float = 90) -> bool:
        """传送到世界坐标 (wx, wy) 附近的锚点。"""
        self.log(f"[tp] 目标世界坐标 ({wx:.1f}, {wy:.1f})")
        if not self.open_map():
            raise RuntimeError("无法打开大地图（SIFT 未匹配到大地图视野）")

        tx2048, ty2048 = self.config.world_to_image(wx, wy)
        tx, ty = tx2048 / 8, ty2048 / 8  # 256 尺度
        deadline = time.monotonic() + timeout_s
        t = self.ctx.transform
        tol = 0.05 * t.device_width

        for it in range(14):
            if time.monotonic() > deadline:
                break
            view = self.big.locate_view(self.ctx.capture_bgr())
            if view is None:
                self.log("[tp] 大地图视野匹配失败，重试")
                self.ctx.sleep(800)
                continue
            vx, vy, px_per_map = view
            dx_screen = (tx - vx) * px_per_map
            dy_screen = (ty - vy) * px_per_map
            dist = math.hypot(dx_screen, dy_screen)
            self.log(f"[tp] 迭代{it}: 视野中心 256图({vx:.0f},{vy:.0f}) 比例{px_per_map:.2f} 目标偏移 {dist:.0f}px")
            if dist <= tol:
                # 目标已接近中心：点它
                tap_x = t.device_width / 2 + dx_screen
                tap_y = t.device_height / 2 + dy_screen
                self.ctx.device.tap(tap_x, tap_y, image_width=t.device_width,
                                    image_height=t.device_height)
                self.ctx.sleep(1200)
                if self._find_and_tap_confirm():
                    break
                if self._tap_anchor_icon_near_center():
                    self.ctx.sleep(1200)
                    if self._find_and_tap_confirm():
                        break
                self.log("[tp] 未弹出传送确认，微调后重试")
                self.ctx.sleep(500)
                continue
            # 拖动地图：目标向中心移动 = 内容朝反方向平移
            self._drag_map(-dx_screen, -dy_screen)
        else:
            raise RuntimeError("传送失败：迭代耗尽仍未点到锚点")

        # 等待传送加载完成（回到主界面）
        self.log("[tp] 已确认传送，等待加载…")
        self.ctx.sleep(3000)
        for _ in range(20):
            frame = self.ctx.capture_bgr()
            if self.big.locate_view(frame) is None:  # 大地图已关闭
                self.ctx.sleep(2000)
                return True
            self.ctx.sleep(1000)
        return True
