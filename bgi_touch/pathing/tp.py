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
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..engine.context import GameContext
from ..engine.recognition import Mat, RecognitionObject
from ..vision.ocr import get_ocr
from .feature_store import SiftFeatureStore
from .map_locator import ASSETS, MapConfig, get_map_definition, resolve_map_name

TEMPLATES = Path(__file__).resolve().parents[2] / "assets" / "templates" / "teleport"


class BigMapLocator:
    """全屏大地图截图 → 当前地图大地图特征坐标与比例。"""

    def __init__(self, map_name: str = "Teyvat"):
        self.definition = get_map_definition(map_name)
        self.map_name = self.definition.name
        self.config = MapConfig.for_map(self.map_name)
        base = ASSETS / self.map_name
        scale = self.definition.big_map_scale
        stores = []
        for keypoints in sorted(
            base.glob(f"{self.map_name}_*_{scale}_SIFT.kp.bin"),
            key=lambda path: ("_0_" not in path.name, path.name),
        ):
            descriptors = keypoints.with_name(
                keypoints.name.removesuffix(".kp.bin") + ".mat.png"
            )
            if descriptors.is_file():
                stores.append(SiftFeatureStore(keypoints, descriptors))
        if not stores:
            raise FileNotFoundError(f"缺少大地图特征资产: {base}")
        self.stores = stores
        self.store = stores[0]
        self._sift = cv2.SIFT_create()

    def world_to_feature(self, wx: float, wy: float) -> tuple[float, float]:
        x, y = self.config.world_to_image(wx, wy)
        ratio = self.definition.big_map_scale / self.definition.feature_scale
        return x * ratio, y * ratio

    def feature_to_world(self, x: float, y: float) -> tuple[float, float]:
        ratio = self.definition.big_to_native_scale
        return self.config.image_to_world(x * ratio, y * ratio)

    def locate_view(self, bgr: np.ndarray) -> tuple[float, float, float] | None:
        """返回 (特征图视野中心 x, y, 屏幕像素/特征图像素比例)；失败 None。

        提瓦特按原版缩小到 1/4 后匹配 256 库，独立地图匹配原生 1024 库。
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        resize = self.definition.big_map_query_resize
        small = (
            cv2.resize(gray, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
            if resize != 1.0 else gray
        )
        kps, desc = self._sift.detectAndCompute(small, None)
        if desc is None or len(kps) < 10:
            return None
        pts = np.float32([k.pt for k in kps])
        center = (small.shape[1] / 2, small.shape[0] / 2)
        for store in self.stores:
            r = store.locate(desc, pts, center)
            if r is not None and r.scale > 1e-6:
                return r.x, r.y, 1.0 / (resize * r.scale)
        return None


class TpTask:
    def __init__(self, ctx: GameContext, log: Callable[[str], None] = print,
                 map_name: str = "Teyvat"):
        self.ctx = ctx
        self.log = log
        self.map_name = resolve_map_name(map_name)
        self.big = BigMapLocator(self.map_name)
        self.config = MapConfig.for_map(self.map_name)
        # Touch zoom has no reliable semantic level from DeviceHub. Keep the
        # BetterGI 1..6 scale in-process and use pinch gestures for changes.
        self._zoom_level = 3.0
        self._go_teleport = RecognitionObject.template_match(
            Mat.from_file(str(TEMPLATES / "GoTeleport.png")), 1440, 960, 100, 120
        )
        self._go_teleport.threshold = 0.7
        self._map_close = RecognitionObject.template_match(
            Mat.from_file(str(TEMPLATES / "MapCloseButton.png")), 0, 0, 180, 140
        )
        self._map_close.threshold = 0.65

    # ---- 步骤 ----

    @contextmanager
    def exclusive_triggers(self):
        """让地图手势独占设备输入，结束后恢复之前的实时触发器。"""
        loop = getattr(self.ctx, "_trigger_loop", None)
        if loop is None or not loop.active:
            yield
            return
        state = loop.pause()
        try:
            yield
        finally:
            loop.resume(state)

    def open_map(self) -> bool:
        for _ in range(3):
            frame = self.ctx.capture_bgr()
            if self.big.locate_view(frame) is not None:
                return True
            if self._is_map_ui():
                if self._switch_area():
                    return self._wait_for_target_map()
                return False
            self.ctx.input.tap_button("map")
            self.ctx.sleep(900)
            try:
                frame = self.ctx.capture_bgr_after_frame(
                    self.ctx.device.last_frame_version, timeout_ms=1800
                )
            except Exception:
                frame = self.ctx.capture_bgr()
            if self.big.locate_view(frame) is not None:
                return True
            if self._is_map_ui() and self._switch_area():
                return self._wait_for_target_map()
        return self.big.locate_view(self.ctx.capture_bgr()) is not None

    def _is_map_ui(self) -> bool:
        try:
            return self.ctx.capture_region().find(self._map_close).is_exist()
        except Exception:
            return False

    @staticmethod
    def _normalize_area_text(value: str) -> str:
        return (
            str(value).replace(" ", "").replace("\n", "")
            .replace("·", "")
            .replace('"', "").replace("“", "").replace("”", "")
            .replace("「", "").replace("」", "")
            .replace("渊下宮", "渊下宫").replace("蒙徳", "蒙德")
            .replace("娜塔", "纳塔")
        )

    def _switch_area(self, timeout_s: float = 2.5) -> bool:
        """Open the area list and select this locator's map by OCR."""
        definition = get_map_definition(self.map_name)
        wanted = [self._normalize_area_text(value) for value in (
            definition.name, *definition.aliases,
        )]
        self.ctx.input.click_ref(1760, 1020)
        self.ctx.sleep(250)
        deadline = time.monotonic() + timeout_s
        seen: list[str] = []
        while time.monotonic() < deadline:
            region = self.ctx.capture_region()
            hits = region.find_multi(
                RecognitionObject.ocr(1280, 0, 640, 1080), limit=30,
            )
            matches = []
            for hit in hits:
                text = self._normalize_area_text(hit.text)
                if text:
                    seen.append(text)
                if any(value in text or text in value for value in wanted):
                    matches.append(hit)
            if matches:
                target = max(matches, key=lambda hit: hit.y)
                self.log(f"[tp] 切换地图区域：{target.text.strip()}")
                target.click()
                self.ctx.sleep(700)
                return True
            self.ctx.sleep(150)
        candidates = " / ".join(dict.fromkeys(seen[:12])) or "无"
        self.log(f"[tp] 切换地图区域失败：{self.map_name}，OCR候选：{candidates}")
        return False

    def _wait_for_target_map(self, timeout_s: float = 4.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.big.locate_view(self.ctx.capture_bgr()) is not None:
                return True
            self.ctx.sleep(250)
        self.log(f"[tp] 已选择 {self.map_name}，但大地图特征匹配失败")
        return False

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
            before = self.ctx.device.last_frame_version
            self.ctx.device.swipe(x0, y0, x0 + sx, y0 + sy, duration_ms=650,
                                  image_width=W, image_height=H)
            if before is not None:
                try:
                    self.ctx.device.wait_for_frame(before, timeout_ms=1800)
                except Exception:
                    # Older headless builds do not expose frame cursors; the
                    # settle delay below remains the compatibility fallback.
                    pass
            self.ctx.sleep(700)  # 等惯性衰减
            dx -= sx
            dy -= sy

    def move_map_to(self, wx: float, wy: float, timeout_s: float = 90) -> bool:
        with self.exclusive_triggers():
            return self._move_map_to(wx, wy, timeout_s)

    def _move_map_to(self, wx: float, wy: float, timeout_s: float = 90) -> bool:
        """Center the visible map on a world coordinate without selecting it."""
        if not self.open_map():
            raise RuntimeError("无法打开大地图（SIFT 未匹配到大地图视野）")
        tx, ty = self.big.world_to_feature(wx, wy)
        deadline = time.monotonic() + timeout_s
        t = self.ctx.transform
        tol = 0.05 * t.device_width
        for it in range(14):
            if time.monotonic() > deadline:
                break
            # DeviceHub's legacy screenshot endpoint can legally return the
            # last decoded frame while its capture worker is busy. Waiting on
            # the frame cursor keeps map feedback tied to the swipe we just
            # issued and also avoids competing with the WebUI preview poller.
            try:
                frame = self.ctx.capture_bgr_after_frame(
                    self.ctx.device.last_frame_version, timeout_ms=2200
                )
            except Exception:
                frame = self.ctx.capture_bgr()
            view = self.big.locate_view(frame)
            if view is None:
                self.log("[tp] 大地图视野匹配失败，重试")
                self.ctx.sleep(800)
                continue
            vx, vy, px_per_map = view
            dx_screen = (tx - vx) * px_per_map
            dy_screen = (ty - vy) * px_per_map
            dist = math.hypot(dx_screen, dy_screen)
            self.log(f"[map] 迭代{it}: 目标偏移 {dist:.0f}px")
            if dist <= tol:
                return True
            self._drag_map(-dx_screen, -dy_screen)
        raise RuntimeError("大地图移动失败：迭代/超时耗尽")

    def get_big_map_zoom_level(self) -> float:
        return float(self._zoom_level)

    def set_big_map_zoom_level(self, level: float) -> float:
        with self.exclusive_triggers():
            return self._set_big_map_zoom_level(level)

    def _set_big_map_zoom_level(self, level: float) -> float:
        """Best-effort touch equivalent of BetterGI's 1.0..6.0 map zoom."""
        target = max(1.0, min(6.0, float(level)))
        if not self.open_map():
            raise RuntimeError("无法打开大地图，不能调整缩放")
        delta = target - self._zoom_level
        steps = min(8, max(0, int(round(abs(delta) * 2))))
        if steps:
            W, H = self.ctx.transform.device_width, self.ctx.transform.device_height
            cx, cy = W / 2, H / 2
            old_span = min(W, H) * 0.14
            # BetterGI's level increases as the map zooms out. Pinch inward
            # for a larger level and outward for a smaller one.
            sign = -1 if delta > 0 else 1
            for _ in range(steps):
                span = old_span + sign * min(W, H) * 0.035
                self.ctx.device.multi_touch([
                    {"x1": cx - old_span, "y1": cy, "x2": cx - span, "y2": cy},
                    {"x1": cx + old_span, "y1": cy, "x2": cx + span, "y2": cy},
                ], duration_ms=350, image_width=W, image_height=H)
                old_span = span
                self.ctx.sleep(350)
        self._zoom_level = target
        return self._zoom_level

    def click_map_point(self, wx: float, wy: float, timeout_s: float = 90) -> bool:
        """Center a world coordinate and click the nearest map point once."""
        with self.exclusive_triggers():
            if not self.open_map():
                raise RuntimeError("无法打开大地图（SIFT 未匹配到大地图视野）")
            self._move_map_to(wx, wy, timeout_s)
            frame = self.ctx.capture_bgr()
            view = self.big.locate_view(frame)
            if view is None:
                raise RuntimeError("点击地图点前视野匹配失败")
            t = self.ctx.transform
            tx, ty = self.big.world_to_feature(wx, wy)
            tap_x = t.device_width / 2 + (tx - view[0]) * view[2]
            tap_y = t.device_height / 2 + (ty - view[1]) * view[2]
            selected = self._tap_anchor_icon_near(
                tap_x, tap_y, max(0.10 * t.device_width, 0.12 * t.device_width)
            )
            if not selected:
                self.ctx.device.tap(
                    tap_x,
                    tap_y,
                    image_width=t.device_width,
                    image_height=t.device_height,
                )
            self.ctx.sleep(1000)
            return True

    def move_independent_map_to(self, wx: float, wy: float, map_name: str,
                                timeout_s: float = 90) -> bool:
        """Move a named map when its local feature assets are available."""
        if resolve_map_name(map_name) != self.map_name:
            return TpTask(self.ctx, self.log, map_name).move_map_to(
                wx, wy, timeout_s,
            )
        return self.move_map_to(wx, wy, timeout_s)

    def tp_to_statue(self, timeout_s: float = 30) -> bool:
        with self.exclusive_triggers():
            return self._tp_to_statue(timeout_s)

    def _tp_to_statue(self, timeout_s: float = 30) -> bool:
        """Teleport through a visible Statue of the Seven icon."""
        if not self.open_map():
            raise RuntimeError("无法打开大地图（SIFT 未匹配到大地图视野）")
        tpl = Mat.from_file(str(TEMPLATES / "StatueOfTheSeven.png"))
        region = self.ctx.capture_region()
        hits = region.find_multi(RecognitionObject.template_match(tpl), limit=20)
        if not hits:
            raise RuntimeError("当前大地图视野未找到七天神像图标")
        cx, cy = self.ctx.transform.device_width / 2, self.ctx.transform.device_height / 2
        target = min(hits, key=lambda h: math.hypot(h.dx + h.dw / 2 - cx, h.dy + h.dh / 2 - cy))
        target.click()
        self.ctx.sleep(1000)
        if not self._find_and_tap_confirm():
            raise RuntimeError("七天神像已选中，但未找到传送确认")
        self.ctx.sleep(max(1000, int(timeout_s * 1000 / 10)))
        return True

    # 候选列表里可点的传送目标类型（点位重叠时弹出）
    _ANCHOR_ENTRIES = ("传送锚点", "七天神像", "秘境", "浪船锚点", "壶中洞天")

    def _find_and_tap_confirm(
        self,
        timeout_s: float = 3.6,
        initial_delay_ms: int = 200,
    ) -> bool:
        """Wait for the panel, choose at most one candidate, then confirm."""
        if initial_delay_ms > 0:
            self.ctx.sleep(initial_delay_ms)
        deadline = time.monotonic() + max(0.2, timeout_s)
        selected_entry = False
        while time.monotonic() < deadline:
            region = self.ctx.capture_region()
            # The confirmation control is a stable icon and is faster/more
            # reliable than OCR on the small iOS panel.
            button = region.find(self._go_teleport)
            if button.is_exist():
                self.log("[tp] 点击确认传送按钮")
                button.click()
                return True
            hits = region.find_multi(RecognitionObject.ocr(900, 100, 1020, 980), limit=25)
            # 最终确认按钮：短文本「传送」
            for h in hits:
                t = h.text.strip().replace(" ", "")
                if t == "传送" or (t.endswith("传送") and len(t) <= 4):
                    self.log(f"[tp] 点击确认「{h.text.strip()}」")
                    h.click()
                    return True
            # 候选列表条目
            entry = next((
                h for h in hits
                if 1 < len(h.text.strip()) < 10
                and any(k in h.text for k in self._ANCHOR_ENTRIES)
            ), None)
            if entry is not None and not selected_entry:
                self.log(f"[tp] 点击候选列表：「{entry.text.strip()}」")
                entry.click()
                selected_entry = True
                self.ctx.sleep(600)
                continue
            self.ctx.sleep(250)
        return False

    def _anchor_icons_near(self, x: float, y: float, max_distance: float):
        """Return nearby teleport icons ordered by distance from the raw point."""
        region = self.ctx.capture_region()
        candidates = []
        for name in ("TeleportWaypoint", "StatueOfTheSeven", "Domain"):
            tpl = Mat.from_file(str(TEMPLATES / f"{name}.png"))
            for h in region.find_multi(RecognitionObject.template_match(tpl), limit=5):
                d = math.hypot(h.dx + h.dw / 2 - x, h.dy + h.dh / 2 - y)
                if d <= max_distance:
                    candidates.append((d, h))
        candidates.sort(key=lambda item: item[0])
        return candidates

    def _tap_anchor_icon_near(self, x: float, y: float, max_distance: float) -> bool:
        """模板匹配目标附近的传送锚点/神像图标并点击最近者。"""
        candidates = self._anchor_icons_near(x, y, max_distance)
        if candidates:
            candidates[0][1].click()
            return True
        return False

    def _tap_anchor_icon_near_center(self) -> bool:
        """回退：模板匹配屏幕中央附近的传送锚点/神像图标并点击最近者。"""
        t = self.ctx.transform
        return self._tap_anchor_icon_near(
            t.device_width / 2,
            t.device_height / 2,
            0.25 * t.device_width,
        )

    def tp(self, wx: float, wy: float, timeout_s: float = 90) -> bool:
        with self.exclusive_triggers():
            return self._tp(wx, wy, timeout_s)

    def _tp(self, wx: float, wy: float, timeout_s: float = 90) -> bool:
        """传送到世界坐标 (wx, wy) 附近的锚点。"""
        self.log(f"[tp] 目标世界坐标 ({wx:.1f}, {wy:.1f})")
        if not self.open_map():
            raise RuntimeError("无法打开大地图（SIFT 未匹配到大地图视野）")

        tx, ty = self.big.world_to_feature(wx, wy)
        deadline = time.monotonic() + timeout_s
        t = self.ctx.transform
        tol = 0.05 * t.device_width

        last_view: tuple[float, float] | None = None
        stagnant_iterations = 0
        for it in range(20):
            if time.monotonic() > deadline:
                break
            try:
                frame = self.ctx.capture_bgr_after_frame(
                    self.ctx.device.last_frame_version, timeout_ms=2200
                )
            except Exception:
                frame = self.ctx.capture_bgr()
            view = self.big.locate_view(frame)
            if view is None:
                self.log("[tp] 大地图视野匹配失败，重试")
                self.ctx.sleep(800)
                continue
            vx, vy, px_per_map = view
            dx_screen = (tx - vx) * px_per_map
            dy_screen = (ty - vy) * px_per_map
            dist = math.hypot(dx_screen, dy_screen)
            self.log(f"[tp] 迭代{it}: 视野中心 256图({vx:.0f},{vy:.0f}) 比例{px_per_map:.2f} 目标偏移 {dist:.0f}px")
            if last_view is not None and math.hypot(vx - last_view[0], vy - last_view[1]) < 1.0:
                stagnant_iterations += 1
            else:
                stagnant_iterations = 0
            last_view = (vx, vy)
            if dist <= tol:
                tap_x = t.device_width / 2 + dx_screen
                tap_y = t.device_height / 2 + dy_screen
                if not self._select_target_and_confirm(tap_x, tap_y, tol):
                    raise RuntimeError(
                        "传送失败：点击传送点后未出现交互面板，可能是传送点未激活"
                    )
                self.log("[tp] 已确认传送，等待加载…")
                self._wait_for_teleport_completion()
                return True
            if stagnant_iterations >= 2:
                raise RuntimeError("大地图移动失败：连续拖动后地图视野未发生变化")
            # 拖动地图：目标向中心移动 = 内容朝反方向平移
            self._drag_map(-dx_screen, -dy_screen)
        raise RuntimeError("传送失败：未能把目标移动到可点击区域（迭代/超时耗尽）")

    def _select_target_and_confirm(self, tap_x: float, tap_y: float, tol: float) -> bool:
        """Click one resolved point and allow one precomputed fallback only."""
        width = self.ctx.transform.device_width
        candidates = self._anchor_icons_near(
            tap_x,
            tap_y,
            max(0.10 * width, 1.8 * tol),
        )
        fallback = None
        if candidates:
            nearest_distance, nearest = candidates[0]
            # Overlapping icons cannot be safely distinguished by template
            # correction. Click the route's raw point first and preserve the
            # already computed nearest icon as a single fallback.
            ambiguous = len(candidates) > 1 and (
                candidates[1][0] - nearest_distance < 0.035 * width
            )
            if ambiguous and nearest_distance > 6:
                self.log("[tp] 邻近图标间距不足，先点击原始目标点")
                self.ctx.device.tap(
                    tap_x,
                    tap_y,
                    image_width=width,
                    image_height=self.ctx.transform.device_height,
                )
                fallback = nearest
            else:
                self.log(f"[tp] 点击校正后的锚点（偏差 {nearest_distance:.0f}px）")
                nearest.click()
        else:
            self.log("[tp] 未匹配到邻近锚点图标，点击原始目标点")
            self.ctx.device.tap(
                tap_x,
                tap_y,
                image_width=width,
                image_height=self.ctx.transform.device_height,
            )

        if self._find_and_tap_confirm():
            return True
        if fallback is None:
            return False

        self.log("[tp] 原始点未弹出面板，回退一次校正锚点")
        fallback.click()
        return self._find_and_tap_confirm()

    def _wait_for_teleport_completion(self, timeout_s: float = 60) -> None:
        """Require a loading state followed by a stable gameplay minimap."""
        from ..engine.genshin_api import GenshinApi

        deadline = time.monotonic() + timeout_s
        started_at = time.monotonic()
        observed_loading = False
        stable_main_ui = 0
        api = GenshinApi(self.ctx, log=self.log)
        while time.monotonic() < deadline:
            frame = self.ctx.capture_bgr()
            is_main_ui = api._is_main_ui(frame)
            if is_main_ui:
                if observed_loading and time.monotonic() - started_at >= 1.0:
                    stable_main_ui += 1
                    if stable_main_ui >= 3:
                        self.log("[tp] 传送完成")
                        return
            else:
                stable_main_ui = 0
                if not observed_loading and self.big.locate_view(frame) is None:
                    observed_loading = True
            self.ctx.sleep(500)
        self.log("[tp] 传送加载等待超时，继续执行后续任务")
