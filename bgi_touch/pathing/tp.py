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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..engine.context import GameContext
from ..engine.recognition import ImageRegion, Mat, RecognitionObject
from ..vision.game_ui import is_big_map_ui, is_main_ui
from .feature_store import SiftFeatureStore
from .map_locator import (
    ASSETS,
    MapConfig,
    get_map_definition,
    map_layer_from_path,
    nearest_teyvat_country,
    resolve_map_name,
    resolve_country_name,
)
from .teleport_points import TeleportPoint, default_teleport_point_store

TEMPLATES = Path(__file__).resolve().parents[2] / "assets" / "templates" / "teleport"
MAP_ICON_TEMPLATES = TEMPLATES.parent / "quick_teleport"
MIN_VIEW_PX_PER_FEATURE = 0.25
MAX_VIEW_PX_PER_FEATURE = 20.0
MAP_MOVE_MAX_ITERATIONS = 24
TP_MOVE_MAX_ITERATIONS = 28
MAP_MOVE_STAGNANT_LIMIT = 2
# A map locator can report a small, plausible center drift while the target
# itself is not getting any closer. Treat that as a separate failure mode
# from an exactly duplicated frame so a bad gesture/profile cannot keep
# issuing swipes until the iteration budget is exhausted.
MAP_MOVE_DISTANCE_STAGNANT_LIMIT = 3
MAP_MOVE_PROGRESS_EPSILON = 1.5
# SIFT coordinates are in the 256/1024 feature space, while the target
# distance in the diagnostic log is in device pixels.  Use a small relative
# margin as well so an ordinary rounding wobble does not trigger a direction
# change on a very distant target.
MAP_MOVE_DISTANCE_EPSILON = 4.0
MAP_MOVE_DIRECTION_COS_EPSILON = 0.15
MAP_MOVE_FRAME_TIMEOUT_MS = 1800
MAP_MOVE_SETTLE_MS = 700
MAP_MOVE_STALE_RETRY_DELAY_MS = 250
TELEPORT_PANEL_TIMEOUT_S = 4.0
TELEPORT_PANEL_INITIAL_DELAY_MS = 200
TELEPORT_PANEL_SELECTION_SETTLE_MS = 600
# A mobile overlap row can consume the first tap while its panel is still
# animating. Match BetterGI's bounded retry contract instead of repeatedly
# clicking the same row until the whole panel timeout is exhausted.
TELEPORT_PANEL_CANDIDATE_CLICK_RETRIES = 2
TELEPORT_COMPLETION_MINIMUM_S = 1.0
TELEPORT_COMPLETION_STABLE_CHECKS = 2
ABSOLUTE_ICON_MAX_CORRECTION_REF = 60.0
ABSOLUTE_ICON_INLIER_RADIUS_REF = 14.0
ABSOLUTE_ICON_OFFSET_BUCKET_REF = 4.0
ABSOLUTE_ICON_BASE_UNCERTAINTY_REF = 16.0
ABSOLUTE_ICON_NEIGHBOR_ERROR_RATIO = 0.25
ABSOLUTE_ICON_MAX_HYPOTHESES = 64


@dataclass(frozen=True)
class _ExpectedMapIcon:
    x: float
    y: float
    icon_types: frozenset[str]


@dataclass
class _ObservedMapIcon:
    x: float
    y: float
    icon_types: set[str]


@dataclass(frozen=True)
class _MapIconAlignment:
    offset_x: float
    offset_y: float
    pair_count: int
    mean_error: float


@dataclass
class _TeleportPanelCandidate:
    """One row from the overlap list shown after selecting a map icon."""

    index: int
    icon_types: set[str]
    text: str
    icon_hit: object | None
    text_hit: object
    row_y: float

    def click(self) -> None:
        """Click a forgiving row-sized hit area when a native Region is available.

        OCR bounds are often only the glyphs on iOS. Tapping that tiny box can
        miss the actual overlap-list row while the desktop implementation taps
        a wider rectangle beside the icon. Keep lightweight test/fallback
        hosts compatible by using the old Region.click() path when the native
        coordinates are unavailable.
        """
        text_hit = self.text_hit
        icon_hit = self.icon_hit
        context = getattr(text_hit, "ctx", None) or getattr(icon_hit, "ctx", None)
        if context is None or text_hit is None:
            fallback = text_hit or icon_hit
            if fallback is not None:
                fallback.click()
            return
        try:
            transform = context.transform
            scale = max(0.5, float(getattr(transform, "scale", 1.0)))
            if icon_hit is not None:
                left = float(icon_hit.dx) + float(icon_hit.dw)
                top = float(icon_hit.dy) - 8.0 * scale
                height = float(icon_hit.dh) + 16.0 * scale
            else:
                left = float(text_hit.dx) - 8.0 * scale
                top = float(text_hit.dy) - 8.0 * scale
                height = float(text_hit.dh) + 16.0 * scale
            # BetterGI's row click rectangle is at least 220 reference pixels
            # wide. A fixed width also handles short OCR strings reliably.
            width = 220.0 * scale
            device_width = float(getattr(transform, "device_width", left + width))
            device_height = float(getattr(transform, "device_height", top + height))
            left = max(0.0, min(left, device_width - width))
            top = max(0.0, min(top, device_height - height))
            context.device.tap(
                left + width / 2.0,
                top + height / 2.0,
                image_width=transform.device_width,
                image_height=transform.device_height,
            )
        except (AttributeError, TypeError, ValueError, RuntimeError):
            fallback = text_hit or icon_hit
            if fallback is not None:
                fallback.click()


MAP_ICON_FILES: dict[str, tuple[str, ...]] = {
    "TeleportWaypoint": ("TeleportWaypoint.png",),
    "StatueOfTheSeven": ("StatueOfTheSeven.png",),
    "Domain": ("Domain.png", "Domain2.png"),
    "ObsidianTotemPole": ("ObsidianTotemPole.png",),
    "PortableWaypoint": ("PortableWaypoint.png",),
    "Mansion": ("Mansion.png",),
    "SubSpaceWaypoint": ("SubSpaceWaypoint.png",),
    "NodKraiMeetingPoint": ("NodKraiMeetingPoint.png",),
    "TabletOfTona": ("TabletOfTona.png",),
    "MarkTransPointMoonTower": ("MarkTransPointMoonTower.png",),
}


class TeleportPanelNotOpenedError(RuntimeError):
    """The selected map point did not open an interaction panel."""


class BigMapLocator:
    """全屏大地图截图 → 当前地图大地图特征坐标与比例。"""

    def __init__(self, map_name: str = "Teyvat"):
        self.definition = get_map_definition(map_name)
        self.map_name = self.definition.name
        self.config = MapConfig.for_map(self.map_name)
        base = ASSETS / self.map_name
        scale = self.definition.big_map_scale
        stores = []
        layer_ids = []
        for keypoints in sorted(
            base.glob(f"{self.map_name}_*_{scale}_SIFT.kp.bin"),
            key=lambda path: ("_0_" not in path.name, path.name),
        ):
            descriptors = keypoints.with_name(
                keypoints.name.removesuffix(".kp.bin") + ".mat.png"
            )
            if descriptors.is_file():
                stores.append(SiftFeatureStore(keypoints, descriptors))
                layer_ids.append(map_layer_from_path(keypoints))
        if not stores:
            raise FileNotFoundError(f"缺少大地图特征资产: {base}")
        self.stores = stores
        self.layer_ids = layer_ids
        self.store = stores[0]
        self.last_layer: int | None = None
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
        for index, store in enumerate(self.stores):
            r = store.locate(desc, pts, center)
            if r is not None and r.scale > 1e-6:
                px_per_feature = 1.0 / (resize * r.scale)
                # Degenerate affine fits can satisfy the inlier threshold on
                # unrelated maps while collapsing almost every query point to
                # one train point. Real map zoom levels stay within this very
                # generous screen-pixels / feature-pixel range.
                if MIN_VIEW_PX_PER_FEATURE <= px_per_feature <= MAX_VIEW_PX_PER_FEATURE:
                    self.last_layer = self.layer_ids[index]
                    return r.x, r.y, px_per_feature
        return None


class TpTask:
    def __init__(self, ctx: GameContext, log: Callable[[str], None] = print,
                 map_name: str = "Teyvat"):
        self.ctx = ctx
        self.log = log
        self.map_name = resolve_map_name(map_name)
        self.big = BigMapLocator(self.map_name)
        self.config = MapConfig.for_map(self.map_name)
        self._teleport_points = default_teleport_point_store()
        # Touch zoom has no reliable semantic level from DeviceHub. Keep the
        # BetterGI 1..6 scale in-process and use pinch gestures for changes.
        self._zoom_level = 3.0
        # Once an independent map has been selected, the game keeps that
        # selection after closing/reopening the map.  Remember it locally so a
        # transient SIFT miss does not reopen the area selector and disturb a
        # gesture that is already in progress.
        self._area_ready = False
        self._selected_area: str | None = None
        self._absolute_icon_templates: dict[str, tuple[RecognitionObject, ...]] | None = None
        self._panel_icon_templates: dict[str, tuple[RecognitionObject, ...]] | None = None
        self._go_teleport = RecognitionObject.template_match(
            Mat.from_file(str(TEMPLATES / "GoTeleport.png")), 1440, 960, 100, 120
        )
        self._go_teleport.threshold = 0.7
        self._map_close = RecognitionObject.template_match(
            # iPhone safe areas move the button farther inward than the PC
            # ``cw - 107*s`` ROI. Keep the top-right search right-anchored but
            # wide enough for 16:9 and 19.5:9 layouts.
            Mat.from_file(str(TEMPLATES / "MapCloseButton.png")), 1600, 0, 320, 140
        )
        self._map_close.threshold = 0.65

    # ---- 步骤 ----

    @contextmanager
    def exclusive_triggers(self):
        """让地图手势独占设备输入，结束后恢复之前的实时触发器。"""
        loop = getattr(self.ctx, "_trigger_loop", None)
        if loop is None:
            yield
            return
        exclusive = getattr(loop, "exclusive", None)
        if callable(exclusive):
            with exclusive():
                yield
            return
        # ``active`` only describes a currently running producer. A trigger
        # list can still be configured while its thread is between stop/start
        # (or before its first start). In that window another caller may start
        # the producer while a map gesture is in progress, allowing AutoPick
        # or AutoSkip to consume a map frame and inject competing input.
        # Pause any configured loop; TriggerLoop preserves its previous
        # inactive state when it is resumed.
        configured_triggers = getattr(loop, "triggers", None)
        if not loop.active and not configured_triggers:
            yield
            return
        state = loop.pause()
        try:
            yield
        finally:
            loop.resume(state)

    def _recover_device_channel(self, reason: str) -> bool:
        """Rebuild a stale Wi-Fi HID channel once without losing map state."""
        self.log(f"[tp] {reason}，重建设备输入通道后重试")
        try:
            self.ctx.device.reconnect_device()
            self.ctx.sleep(2000)
            refresh = getattr(self.ctx, "refresh_orientation", None)
            if callable(refresh):
                refresh()
            return True
        except Exception as error:
            self.log(f"[tp] 设备输入通道重建失败：{error}")
            return False

    def _target_area(
        self,
        wx: float | None = None,
        wy: float | None = None,
        area_name: str | None = None,
    ) -> str | None:
        """Resolve the map selector entry needed for a target coordinate."""
        if self.map_name == "Teyvat":
            if area_name:
                return resolve_country_name(area_name)
            if wx is not None and wy is not None:
                return nearest_teyvat_country(wx, wy)
            return None
        return self.map_name

    def open_map(
        self,
        *,
        wx: float | None = None,
        wy: float | None = None,
        area_name: str | None = None,
    ) -> bool:
        """Open the requested map and, when needed, its area selector entry.

        ``area_name`` is a country for Teyvat and a map name for an independent
        map.  With no target arguments this preserves the old lightweight
        behavior used by zoom/statue helpers: recognize the currently visible
        map without opening the area selector.
        """
        target_area = self._target_area(wx, wy, area_name)
        recovered = False
        for attempt in range(3):
            frame = self.ctx.capture_bgr()
            if self.big.locate_view(frame) is not None:
                if target_area is None or self._selected_area == target_area:
                    self._area_ready = True
                    return True
                if self._switch_area(target_area):
                    return self._wait_for_target_map()
                return False
            if self._is_map_ui():
                if target_area is not None and self._switch_area(target_area):
                    return self._wait_for_target_map()
                if target_area is None and self._area_ready and self._wait_for_target_map(timeout_s=1.0):
                    return True
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
                if target_area is None or self._selected_area == target_area:
                    self._area_ready = True
                    return True
                if self._switch_area(target_area):
                    return self._wait_for_target_map()
                return False
            if self._is_map_ui() and target_area is not None and self._switch_area(target_area):
                return self._wait_for_target_map()
            if attempt == 0 and not recovered:
                recovered = self._recover_device_channel("地图按键未生效")
        view = self.big.locate_view(self.ctx.capture_bgr())
        return view is not None and (target_area is None or self._selected_area == target_area)

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

    def _switch_area(
        self,
        area_name: str | None = None,
        timeout_s: float = 2.5,
    ) -> bool:
        """Open the area list and select a country/map entry by OCR."""
        definition = get_map_definition(self.map_name)
        if self.map_name == "Teyvat":
            selected_area = resolve_country_name(area_name)
            if selected_area is None:
                self.log("[tp] 切换地图区域失败：未指定提瓦特国家")
                return False
            wanted_values = (selected_area,)
        else:
            selected_area = self.map_name
            wanted_values = (definition.name, *definition.aliases)
        wanted = [self._normalize_area_text(value) for value in wanted_values]
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
                self._selected_area = selected_area
                self._area_ready = False
                return True
            self.ctx.sleep(150)
        candidates = " / ".join(dict.fromkeys(seen[:12])) or "无"
        self.log(f"[tp] 切换地图区域失败：{selected_area}，OCR候选：{candidates}")
        return False

    def _wait_for_target_map(self, timeout_s: float = 4.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.big.locate_view(self.ctx.capture_bgr()) is not None:
                self._area_ready = True
                return True
            self.ctx.sleep(250)
        self.log(f"[tp] 已选择 {self.map_name}，但大地图特征匹配失败")
        return False

    def _drag_map(self, dx: float, dy: float) -> np.ndarray | None:
        """Move map content and return the newest frame produced by the gesture."""
        t = self.ctx.transform
        W, H = t.device_width, t.device_height
        max_step = 0.30 * W
        feedback = None
        while abs(dx) > 1 or abs(dy) > 1:
            sx = max(-max_step, min(max_step, dx))
            sy = max(-max_step, min(max_step, dy))
            # 起点选屏幕中央偏移，避开左上返回键与右下 UI
            x0 = W * 0.5 - sx / 2
            y0 = H * 0.5 - sy / 2
            before = self.ctx.device.last_frame_version
            self.ctx.device.swipe(x0, y0, x0 + sx, y0 + sy, duration_ms=650,
                                  image_width=W, image_height=H)
            self.ctx.sleep(MAP_MOVE_SETTLE_MS)  # 等惯性衰减
            if before is not None:
                try:
                    feedback = self.ctx.capture_bgr_after_frame(
                        before, timeout_ms=MAP_MOVE_FRAME_TIMEOUT_MS,
                    )
                except Exception:
                    # Older headless builds do not expose frame cursors; the
                    # next loop capture remains the compatibility fallback.
                    feedback = None
            if feedback is None:
                # Old devicehub-mask builds do not expose a frame cursor on
                # action responses.  Still consume one post-gesture screenshot
                # instead of letting the next iteration repeatedly inspect a
                # pre-swipe frame.
                try:
                    feedback = self.ctx.capture_bgr()
                except Exception:
                    feedback = None
            dx -= sx
            dy -= sy
        return feedback

    def _capture_fresh_map_frame(self) -> np.ndarray | None:
        """Wait for another observation when a map frame looks unchanged.

        A plain screenshot can legally be the last decoded video frame while
        the iOS stream is catching up.  If the DeviceHub frame cursor exists,
        wait after that cursor first; old servers fall back to their legacy
        screenshot endpoint.  This path is only used after a duplicate map
        position, so it does not add a competing screenshot producer during
        normal gestures.
        """
        capture_after = getattr(self.ctx, "capture_bgr_after_frame", None)
        version = getattr(getattr(self.ctx, "device", None), "last_frame_version", None)
        if version is not None and callable(capture_after):
            try:
                return capture_after(version, timeout_ms=MAP_MOVE_FRAME_TIMEOUT_MS)
            except Exception:
                pass
        try:
            return self.ctx.capture_bgr()
        except Exception:
            return None

    def _locate_view_with_stale_frame_guard(
        self,
        frame: np.ndarray,
        previous: tuple[float, float] | None,
    ) -> tuple[float, float, float] | None:
        """Locate a map frame and refresh once when the result is unchanged.

        A slow observation stream can return the frame from immediately before
        a swipe even though the gesture itself succeeded.  Treating that frame
        as a real no-op causes the old implementation to repeat the same drag
        and eventually fail with a constant target offset.  The extra capture
        is only made for a duplicate result, so normal map movement keeps the
        one-frame-per-gesture behavior used by the screenshot consumers.
        """
        self._last_located_frame = frame
        view = self.big.locate_view(frame)
        if view is None or previous is None:
            return view
        if math.hypot(view[0] - previous[0], view[1] - previous[1]) >= MAP_MOVE_PROGRESS_EPSILON:
            return view
        refreshed = self._capture_fresh_map_frame()
        try:
            refreshed_view = self.big.locate_view(refreshed)
        except Exception:
            return view
        if refreshed_view is not None:
            self._last_located_frame = refreshed
            return refreshed_view
        return view

    def _move_map_view_to(
        self,
        wx: float,
        wy: float,
        timeout_s: float,
        *,
        area_name: str | None = None,
        log_prefix: str,
        max_iterations: int,
        error_message: str,
    ) -> tuple[float, float, float]:
        """Move a target into the safe center and return its final map view.

        ``pan_sign`` starts with the iOS convention (the finger follows the
        map).  If a device profile reports the opposite swipe convention, the
        first measurable movement automatically flips the sign once.  This is
        useful for older DeviceHub profiles and is harmless for the current
        fixed 16:9 profile.
        """
        if not self.open_map(wx=wx, wy=wy, area_name=area_name):
            raise RuntimeError("无法打开大地图（SIFT 未匹配到大地图视野）")
        # The successful locator frame is also the safest frame for the final
        # map click. Clear it before the loop so a failed/recovered attempt
        # cannot leak an observation from a previous teleport.
        self._last_located_frame = None
        tx, ty = self.big.world_to_feature(wx, wy)
        deadline = time.monotonic() + timeout_s
        t = self.ctx.transform
        tol = 0.05 * t.device_width
        last_view: tuple[float, float] | None = None
        last_distance: float | None = None
        last_expected_direction: tuple[float, float] | None = None
        stagnant_iterations = 0
        distance_stagnant_iterations = 0
        recovered = False
        pan_sign = -1.0
        feedback_frame = None
        for it in range(max_iterations):
            if time.monotonic() > deadline:
                break
            # Consume the frame obtained after the previous drag. Asking for
            # another frame here can time out on a slow stream and return a
            # stale pre-gesture screenshot.
            frame = feedback_frame if feedback_frame is not None else self.ctx.capture_bgr()
            feedback_frame = None
            view = self._locate_view_with_stale_frame_guard(frame, last_view)
            if view is None:
                self.log("[tp] 大地图视野匹配失败，重试")
                self.ctx.sleep(300)
                continue
            vx, vy, px_per_map = view
            dx_screen = (tx - vx) * px_per_map
            dy_screen = (ty - vy) * px_per_map
            dist = math.hypot(dx_screen, dy_screen)
            self.log(
                f"{log_prefix}{it}: 视野中心 特征图({vx:.0f},{vy:.0f}) "
                f"比例{px_per_map:.2f} 目标偏移 {dist:.0f}px"
            )

            distance_margin = max(
                MAP_MOVE_DISTANCE_EPSILON,
                0.02 * max(
                    last_distance if last_distance is not None else dist,
                    dist,
                ),
            )
            distance_not_reduced = (
                last_distance is not None
                and dist >= last_distance - distance_margin
            )
            if last_distance is None:
                distance_stagnant_iterations = 0
            elif distance_not_reduced:
                distance_stagnant_iterations += 1
            else:
                distance_stagnant_iterations = 0

            if last_view is not None and last_expected_direction is not None:
                actual_dx = vx - last_view[0]
                actual_dy = vy - last_view[1]
                actual_distance = math.hypot(actual_dx, actual_dy)
                expected_distance = math.hypot(*last_expected_direction)
                # Only infer the sign when the frame actually moved.  A
                # duplicate frame is handled by the stagnant/recovery path.
                # Besides the vector direction, compare the actual target
                # distance.  Some low-feature frames produce a plausible
                # center displacement whose projection is close to zero; in
                # that case repeatedly issuing the same swipe can walk the
                # view away from the target without tripping the old strict
                # ``dot < 0`` check.
                if (
                    actual_distance >= MAP_MOVE_PROGRESS_EPSILON
                    and expected_distance >= MAP_MOVE_PROGRESS_EPSILON
                ):
                    projection = (
                        actual_dx * last_expected_direction[0]
                        + actual_dy * last_expected_direction[1]
                    ) / (actual_distance * expected_distance)
                    direction_is_wrong = projection < -MAP_MOVE_DIRECTION_COS_EPSILON
                    direction_is_ambiguous = (
                        distance_not_reduced
                        and projection <= MAP_MOVE_DIRECTION_COS_EPSILON
                    )
                else:
                    projection = 0.0
                    direction_is_wrong = False
                    direction_is_ambiguous = False
                if direction_is_wrong or direction_is_ambiguous:
                    pan_sign *= -1.0
                    reason = "拖动方向相反" if direction_is_wrong else "目标距离未缩小"
                    self.log(
                        f"[tp] 地图拖动反馈异常（{reason}，投影{projection:.2f}，"
                        "切换触控方向）"
                    )
                    last_expected_direction = None

            if dist <= tol:
                return view

            if last_view is not None and math.hypot(vx - last_view[0], vy - last_view[1]) < MAP_MOVE_PROGRESS_EPSILON:
                stagnant_iterations += 1
            else:
                stagnant_iterations = 0
            last_view = (vx, vy)
            last_distance = dist
            recovery_reason = None
            if stagnant_iterations >= MAP_MOVE_STAGNANT_LIMIT:
                recovery_reason = "连续拖动后地图视野未变化"
            elif distance_stagnant_iterations >= MAP_MOVE_DISTANCE_STAGNANT_LIMIT:
                recovery_reason = "目标距离连续未缩小"
            if recovery_reason is not None:
                if recovered or not self._recover_device_channel(recovery_reason):
                    self.log(f"[tp] {recovery_reason}，恢复后仍未收敛")
                    raise RuntimeError(error_message)
                recovered = True
                stagnant_iterations = 0
                distance_stagnant_iterations = 0
                last_view = None
                last_distance = None
                last_expected_direction = None
                feedback_frame = None
                # Let the rebuilt channel publish a fresh map frame before
                # sending another gesture; otherwise the first retry can
                # still be based on the stale pre-reconnect frame.
                self.ctx.sleep(250)
                continue

            if stagnant_iterations:
                # Do not issue the same swipe again while the observation
                # stream is still returning the pre-gesture frame.  Waiting
                # for one more frame gives a successful iOS gesture time to
                # surface, and reaches the reconnect path above quickly when
                # the input channel really did not move the map.
                self.log("[tp] 拖动后地图视野未更新，等待新帧")
                self.ctx.sleep(MAP_MOVE_STALE_RETRY_DELAY_MS)
                feedback_frame = None
                continue

            # 目标向中心移动 = 地图内容朝反方向平移。保存期望方向，下一
            # 帧可以验证 profile 的 swipe 坐标语义是否相反。
            last_expected_direction = (tx - vx, ty - vy)
            feedback_frame = self._drag_map(
                pan_sign * dx_screen,
                pan_sign * dy_screen,
            )
            if feedback_frame is None:
                self.log("[tp] 未取得拖动后的新截图，等待下一帧")
        raise RuntimeError(error_message)

    def move_map_to(
        self,
        wx: float,
        wy: float,
        timeout_s: float = 90,
        force_country: str | None = None,
    ) -> bool:
        with self.exclusive_triggers():
            return self._move_map_to(wx, wy, timeout_s, force_country=force_country)

    def _move_map_to(
        self,
        wx: float,
        wy: float,
        timeout_s: float = 90,
        *,
        force_country: str | None = None,
    ) -> bool:
        """Center the visible map on a world coordinate without selecting it."""
        self._move_map_view_to(
            wx,
            wy,
            timeout_s,
            area_name=force_country,
            log_prefix="[map] 迭代",
            max_iterations=MAP_MOVE_MAX_ITERATIONS,
            error_message="大地图移动失败：迭代/超时耗尽",
        )
        return True

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

    def click_map_point(
        self,
        wx: float,
        wy: float,
        timeout_s: float = 90,
        force_country: str | None = None,
    ) -> bool:
        """Center a world coordinate and click the nearest map point once."""
        with self.exclusive_triggers():
            if not self.open_map(wx=wx, wy=wy, area_name=force_country):
                raise RuntimeError("无法打开大地图（SIFT 未匹配到大地图视野）")
            self._move_map_to(wx, wy, timeout_s, force_country=force_country)
            frame = getattr(self, "_last_located_frame", None)
            if frame is None:
                frame = self.ctx.capture_bgr()
            view = self.big.locate_view(frame)
            if view is None:
                raise RuntimeError("点击地图点前视野匹配失败")
            map_region = ImageRegion(self.ctx, frame)
            t = self.ctx.transform
            tx, ty = self.big.world_to_feature(wx, wy)
            tap_x = t.device_width / 2 + (tx - view[0]) * view[2]
            tap_y = t.device_height / 2 + (ty - view[1]) * view[2]
            selected = self._tap_anchor_icon_near(
                tap_x,
                tap_y,
                max(0.10 * t.device_width, 0.12 * t.device_width),
                region=map_region,
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
                                timeout_s: float = 90,
                                force_country: str | None = None) -> bool:
        """Move a named map when its local feature assets are available."""
        if resolve_map_name(map_name) != self.map_name:
            return TpTask(self.ctx, self.log, map_name).move_map_to(
                wx, wy, timeout_s, force_country=force_country,
            )
        return self.move_map_to(wx, wy, timeout_s, force_country=force_country)

    def tp_to_statue(self, timeout_s: float = 30) -> bool:
        with self.exclusive_triggers():
            return self._tp_to_statue(timeout_s)

    def _tp_to_statue(self, timeout_s: float = 30) -> bool:
        """Teleport through a Statue of the Seven, including off-screen fallback.

        The fast path uses a visible icon from the current map frame.  When
        the current map center is far from every statue, use the shared
        ``tp.json`` index and move the map to the nearest known statue instead
        of assuming that an off-screen icon exists.
        """
        if not self.open_map():
            raise RuntimeError("无法打开大地图（SIFT 未匹配到大地图视野）")
        tpl = Mat.from_file(str(TEMPLATES / "StatueOfTheSeven.png"))
        frame = self.ctx.capture_bgr()
        region = ImageRegion(self.ctx, frame)
        hits = region.find_multi(RecognitionObject.template_match(tpl), limit=20)
        if not hits:
            view = self.big.locate_view(frame)
            if view is not None:
                center_x, center_y = self.big.feature_to_world(view[0], view[1])
                statue = next(iter(self._teleport_points.nearest_of_type(
                    self.map_name,
                    center_x,
                    center_y,
                    {"Goddess", "StatueOfTheSeven"},
                )), None)
                if statue is not None:
                    self.log(
                        "[tp] 当前视野无神像图标，按地图中心选择最近神像："
                        f"{statue.name or statue.country or '七天神像'}"
                    )
                    return self._tp(
                        statue.x,
                        statue.y,
                        timeout_s=timeout_s,
                        force=False,
                    )
            raise RuntimeError("当前大地图视野未找到七天神像图标，且传送点索引无可用回退")
        cx, cy = self.ctx.transform.device_width / 2, self.ctx.transform.device_height / 2
        target = min(hits, key=lambda h: math.hypot(h.dx + h.dw / 2 - cx, h.dy + h.dh / 2 - cy))
        target.click()
        self.ctx.sleep(1000)
        if not self._find_and_tap_confirm():
            raise TeleportPanelNotOpenedError("七天神像已选中，但未找到传送确认")
        self._wait_for_teleport_completion(timeout_s=timeout_s)
        return True

    # 候选列表里可点的传送目标类型（点位重叠时弹出）
    _ANCHOR_ENTRIES = ("传送锚点", "七天神像", "秘境", "浪船锚点", "壶中洞天")
    # Keep the same strict upper bound as BetterGI's candidate OCR filter
    # (raw text length must stay below ten characters).
    _ANCHOR_ENTRY_MAX_TEXT_LENGTH = 9
    _MAP_CHOOSE_ICON_ROI = (1270, 100, 50, 880)
    _PANEL_TEXT_ROI = (1320, 100, 560, 880)
    _PANEL_ROW_MERGE_DISTANCE = 18.0

    @staticmethod
    def _map_icon_types_for_point(point_type: str) -> frozenset[str]:
        """Map ``tp.json`` point kinds to the quick-teleport icon kinds."""
        value = str(point_type or "")
        if value in {"TeleportWaypoint", "MarkTransPointMoonTower"}:
            return frozenset({"TeleportWaypoint"})
        if value in {"Goddess", "StatueOfTheSeven"}:
            return frozenset({"StatueOfTheSeven"})
        if "Domain" in value:
            return frozenset({"Domain"})
        if value in MAP_ICON_FILES:
            return frozenset({value})
        if value == "NatlanObsidianTotemPole":
            return frozenset({"ObsidianTotemPole"})
        return frozenset()

    def _load_panel_icon_templates(
        self,
    ) -> dict[str, tuple[RecognitionObject, ...]]:
        """Load list-row icon recognizers restricted to the upstream list ROI."""
        cached = getattr(self, "_panel_icon_templates", None)
        if cached is not None:
            return cached
        loaded: dict[str, tuple[RecognitionObject, ...]] = {}
        for icon_type, filenames in MAP_ICON_FILES.items():
            recognizers: list[RecognitionObject] = []
            for filename in filenames:
                path = MAP_ICON_TEMPLATES / filename
                if not path.is_file():
                    continue
                try:
                    template = Mat.from_file(str(path))
                    if template.empty():
                        continue
                    recognition = RecognitionObject.template_match(template)
                    recognition.roi = self._MAP_CHOOSE_ICON_ROI
                    recognition.threshold = 0.70 if icon_type == "TeleportWaypoint" else 0.65
                    recognizers.append(recognition)
                except (OSError, ValueError, RuntimeError) as error:
                    self.log(f"[tp] 传送候选图标模板加载失败 {filename}：{error}")
            if recognizers:
                loaded[icon_type] = tuple(recognizers)
        self._panel_icon_templates = loaded
        return loaded

    @staticmethod
    def _normalize_candidate_text(value: str) -> str:
        text = TpTask._normalize_panel_text(value)
        return "".join(
            char for char in text
            if not char.isspace() and char not in {"-", "－", "·", "."}
        )

    @staticmethod
    def _candidate_name_overlap_score(candidate_text: str, target_text: str) -> int:
        candidate = TpTask._normalize_candidate_text(candidate_text)
        target = TpTask._normalize_candidate_text(target_text)
        if not candidate or not target:
            return 0
        if candidate == target:
            return 10_000
        if candidate in target or target in candidate:
            return 5_000 + min(len(candidate), len(target))
        return sum(char in target for char in candidate)

    @classmethod
    def _candidate_type_matches(
        cls,
        candidate: _TeleportPanelCandidate,
        target_point: TeleportPoint | None,
    ) -> bool:
        if target_point is None:
            return False
        expected = cls._map_icon_types_for_point(target_point.point_type)
        return bool(expected.intersection(candidate.icon_types))

    @classmethod
    def _choose_panel_candidate(
        cls,
        candidates: list[_TeleportPanelCandidate],
        target_point: TeleportPoint | None = None,
    ) -> _TeleportPanelCandidate | None:
        """Choose by target name/type, retaining row order as the safe fallback.

        ``tp.json`` often calls every waypoint ``传送锚点`` while the game list
        displays a local place name.  In that case the icon type is still useful
        evidence, and the first row matches BetterGI's historical fallback.
        """
        if not candidates:
            return None
        ordered = sorted(candidates, key=lambda item: (item.row_y, item.index))
        if target_point is None:
            return ordered[0]

        labels = [target_point.name, *target_point.areas]
        if target_point.country:
            labels.append(target_point.country)
        scored: list[tuple[tuple[int, int, int], _TeleportPanelCandidate]] = []
        for candidate in ordered:
            type_score = 1 if cls._candidate_type_matches(candidate, target_point) else 0
            name_score = max(
                (cls._candidate_name_overlap_score(candidate.text, label)
                 for label in labels if label),
                default=0,
            )
            # A known type must win over a name-only OCR hit from an unrelated
            # map label. ``-index`` keeps the deterministic top-row fallback.
            scored.append(((type_score, name_score, -candidate.index), candidate))
        return max(scored, key=lambda item: item[0])[1]

    def _panel_text_hit_for_icon(self, region: ImageRegion, icon_hit):
        """OCR the single row beside an icon without taking another screenshot."""
        try:
            icon_x = float(icon_hit.x)
            icon_y = float(icon_hit.y)
            icon_width = max(1.0, float(icon_hit.width))
            icon_height = max(1.0, float(icon_hit.height))
        except (AttributeError, TypeError, ValueError):
            return None
        text_x = icon_x + icon_width
        text_y = max(0.0, icon_y - 10.0)
        text_width = min(420.0, max(0.0, 1910.0 - text_x))
        text_height = max(32.0, icon_height + 20.0)
        if text_width <= 0:
            return None
        hits = region.find_multi(
            RecognitionObject.ocr(text_x, text_y, text_width, text_height),
            limit=4,
        )
        if not hits:
            return None
        center_y = icon_y + icon_height / 2
        row_hits = [
            hit for hit in hits
            if abs((float(hit.y) + float(hit.height) / 2) - center_y)
            <= max(18.0, icon_height * 1.5)
        ]
        return max(
            row_hits or hits,
            key=lambda hit: (len(self._normalize_candidate_text(getattr(hit, "text", ""))),
                             -abs(float(hit.y) - icon_y)),
        )

    def _find_panel_candidates(
        self,
        region: ImageRegion,
    ) -> list[_TeleportPanelCandidate]:
        """Recognize candidate rows from the same panel frame.

        The icon ROI prevents ordinary map labels from becoming clickable rows;
        OCR is only used for the text immediately adjacent to a recognized row
        icon.  This is important on iOS, where the overlap list text is usually
        a place name rather than the generic ``传送锚点`` label.
        """
        candidates: list[_TeleportPanelCandidate] = []
        for icon_type, recognizers in self._load_panel_icon_templates().items():
            for recognition in recognizers:
                try:
                    icon_hits = region.find_multi(recognition, limit=12)
                except (OSError, ValueError, RuntimeError) as error:
                    self.log(f"[tp] 传送候选图标识别失败 {icon_type}：{error}")
                    continue
                for icon_hit in icon_hits:
                    text_hit = self._panel_text_hit_for_icon(region, icon_hit)
                    if text_hit is None:
                        continue
                    text = str(getattr(text_hit, "text", "")).strip()
                    normalized = self._normalize_candidate_text(text)
                    # BetterGI rejects OCR rows whose raw text is ten or more
                    # characters. Long matches are commonly a map label or a
                    # combined ``传送锚点·地点`` string rather than one row in
                    # the overlap list.
                    if len(text) > self._ANCHOR_ENTRY_MAX_TEXT_LENGTH or len(normalized) <= 1:
                        continue
                    row_y = float(getattr(icon_hit, "y", 0.0))
                    existing = next(
                        (
                            item for item in candidates
                            if abs(item.row_y - row_y) <= self._PANEL_ROW_MERGE_DISTANCE
                        ),
                        None,
                    )
                    if existing is not None:
                        existing.icon_types.add(icon_type)
                        if len(normalized) > len(self._normalize_candidate_text(existing.text)):
                            existing.text = text
                            existing.text_hit = text_hit
                        continue
                    candidates.append(_TeleportPanelCandidate(
                        index=len(candidates) + 1,
                        icon_types={icon_type},
                        text=text,
                        icon_hit=icon_hit,
                        text_hit=text_hit,
                        row_y=row_y,
                    ))
        return sorted(candidates, key=lambda item: (item.row_y, item.index))

    def _find_target_text_candidate(
        self,
        region: ImageRegion,
        target_point: TeleportPoint | None,
    ) -> _TeleportPanelCandidate | None:
        """OCR-only fallback for a named target when its row icon is missed."""
        if target_point is None:
            return None
        labels = [target_point.name, *target_point.areas]
        labels = [self._normalize_candidate_text(label) for label in labels if label]
        labels = [label for label in labels if len(label) > 1]
        if not labels:
            return None
        hits = region.find_multi(RecognitionObject.ocr(*self._PANEL_TEXT_ROI), limit=30)
        matched = [
            hit for hit in hits
            if len(str(getattr(hit, "text", "") or "").strip())
            <= self._ANCHOR_ENTRY_MAX_TEXT_LENGTH
            and any(
                self._candidate_name_overlap_score(getattr(hit, "text", ""), label)
                >= 5_000
                for label in labels
            )
        ]
        if len(matched) != 1:
            return None
        hit = matched[0]
        return _TeleportPanelCandidate(
            index=1,
            icon_types=set(),
            text=str(getattr(hit, "text", "")),
            icon_hit=None,
            text_hit=hit,
            row_y=float(getattr(hit, "y", 0.0)),
        )

    def _find_and_tap_confirm(
        self,
        timeout_s: float = TELEPORT_PANEL_TIMEOUT_S,
        initial_delay_ms: int = TELEPORT_PANEL_INITIAL_DELAY_MS,
        target_point: TeleportPoint | None = None,
    ) -> bool:
        """Wait for the panel, choose at most one candidate, then confirm."""
        if initial_delay_ms > 0:
            self.ctx.sleep(initial_delay_ms)
        deadline = time.monotonic() + max(0.2, timeout_s)
        selected_entry = False
        candidate_click_attempts = 0
        selected_at = 0.0
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
                text = self._normalize_panel_text(getattr(h, "text", ""))
                if (
                    (text == "传送" or (text.endswith("传送") and len(text) <= 4))
                    and self._is_confirm_hit(h)
                ):
                    self.log(f"[tp] 点击确认「{h.text.strip()}」")
                    h.click()
                    return True
            # 候选列表条目。优先按列表图标识别，避免把地图地点名当作
            # 普通 OCR 文本丢弃；无图标时再对已知目标名称做唯一匹配回退。
            candidates = self._find_panel_candidates(region)
            if not candidates:
                target_candidate = self._find_target_text_candidate(region, target_point)
                if target_candidate is not None:
                    candidates = [target_candidate]
            candidate = self._choose_panel_candidate(candidates, target_point)
            if candidate is None:
                # 兼容旧版面板只返回「传送锚点」文本的识别器。
                entry = next((
                    h for h in hits
                    if self._is_anchor_entry_text(getattr(h, "text", ""))
                ), None)
                candidate = (
                    _TeleportPanelCandidate(
                        index=1,
                        icon_types=set(),
                        text=str(getattr(entry, "text", "")),
                        icon_hit=None,
                        text_hit=entry,
                        row_y=float(getattr(entry, "y", 0.0)),
                    )
                    if entry is not None else None
                )
            if candidate is not None and not selected_entry:
                self.log(f"[tp] 点击候选列表：「{candidate.text.strip()}」")
                candidate.click()
                selected_entry = True
                candidate_click_attempts = 1
                selected_at = time.monotonic()
                self.ctx.sleep(TELEPORT_PANEL_SELECTION_SETTLE_MS)
                continue
            # A candidate row remaining after the settle delay means the tap
            # did not select it.  Returning here lets the caller use its one
            # precomputed icon fallback instead of clicking the same stale row
            # repeatedly for the whole panel timeout.
            if selected_entry and candidate is not None and time.monotonic() - selected_at >= 0.8:
                if candidate_click_attempts < TELEPORT_PANEL_CANDIDATE_CLICK_RETRIES:
                    candidate_click_attempts += 1
                    self.log(
                        f"[tp] 传送候选列表仍在，重试点选 "
                        f"{candidate_click_attempts}/{TELEPORT_PANEL_CANDIDATE_CLICK_RETRIES}"
                    )
                    candidate.click()
                    selected_at = time.monotonic()
                    self.ctx.sleep(TELEPORT_PANEL_SELECTION_SETTLE_MS)
                    continue
                self.log("[tp] 传送候选列表仍在，判定本次点选未生效")
                return False
            self.ctx.sleep(250)
        return False

    @staticmethod
    def _normalize_panel_text(value: str) -> str:
        return (
            str(value or "")
            .replace(" ", "")
            .replace("\n", "")
            .replace("傳送錨點", "传送锚点")
            .replace("傳送", "传送")
            .replace("錨點", "锚点")
            .replace("锚点>", "锚点")
            .replace(">", "")
            .replace("＞", "")
            .strip()
        )

    def _load_absolute_icon_templates(
        self,
    ) -> dict[str, tuple[RecognitionObject, ...]]:
        cached = getattr(self, "_absolute_icon_templates", None)
        if cached is not None:
            return cached
        loaded: dict[str, tuple[RecognitionObject, ...]] = {}
        for icon_type, filenames in MAP_ICON_FILES.items():
            objects: list[RecognitionObject] = []
            for filename in filenames:
                path = MAP_ICON_TEMPLATES / filename
                if not path.is_file():
                    continue
                try:
                    template = Mat.from_file(str(path))
                    if template.empty():
                        continue
                    recognition = RecognitionObject.template_match(template)
                    recognition.threshold = 0.65
                    objects.append(recognition)
                except (OSError, ValueError, RuntimeError) as error:
                    self.log(f"[tp] 地图图标模板加载失败 {filename}：{error}")
            if objects:
                loaded[icon_type] = tuple(objects)
        self._absolute_icon_templates = loaded
        return loaded

    def _map_icon_scale(self) -> float:
        transform = self.ctx.transform
        try:
            value = float(transform.scale)
        except (AttributeError, TypeError, ValueError):
            value = 1.0
        return max(0.5, value)

    def _is_map_icon_search_area(self, x: float, y: float) -> bool:
        width = float(self.ctx.transform.device_width)
        height = float(self.ctx.transform.device_height)
        margin = max(35.0, 0.035 * min(width, height))
        if x < margin or y < margin or x > width - margin or y > height - margin:
            return False
        # The top-left return/menu controls overlap the map and frequently
        # produce a false template match.  Keep the same exclusion as the
        # upstream absolute-icon aligner.
        return not (x < 0.20 * width and y < 0.35 * height)

    def _expected_visible_map_icons(
        self,
        view: tuple[float, float, float],
    ) -> list[_ExpectedMapIcon]:
        points = getattr(self._teleport_points, "scenes", {}).get(self.map_name, ())
        if not points:
            return []
        vx, vy, px_per_map = view
        width = float(self.ctx.transform.device_width)
        height = float(self.ctx.transform.device_height)
        result: list[_ExpectedMapIcon] = []
        for point in points:
            icon_types = self._map_icon_types_for_point(point.point_type)
            if not icon_types:
                continue
            try:
                fx, fy = self.big.world_to_feature(point.x, point.y)
                screen_x = width / 2 + (fx - vx) * px_per_map
                screen_y = height / 2 + (fy - vy) * px_per_map
            except (AttributeError, TypeError, ValueError):
                continue
            if self._is_map_icon_search_area(screen_x, screen_y):
                result.append(_ExpectedMapIcon(screen_x, screen_y, icon_types))
        return result

    def _observed_visible_map_icons(
        self,
        region: ImageRegion,
        allowed_types: set[str],
    ) -> list[_ObservedMapIcon]:
        templates = self._load_absolute_icon_templates()
        observed: list[_ObservedMapIcon] = []
        limit = 160
        for icon_type in sorted(allowed_types):
            for recognition in templates.get(icon_type, ()):
                try:
                    hits = region.find_multi(recognition, limit=limit)
                except (OSError, ValueError, RuntimeError) as error:
                    self.log(f"[tp] 地图图标识别失败 {icon_type}：{error}")
                    continue
                for hit in hits:
                    try:
                        x_value = getattr(hit, "dx", None)
                        y_value = getattr(hit, "dy", None)
                        width_value = getattr(hit, "dw", None)
                        height_value = getattr(hit, "dh", None)
                        x = float(x_value if x_value is not None else getattr(hit, "x"))
                        y = float(y_value if y_value is not None else getattr(hit, "y"))
                        width = float(
                            width_value if width_value is not None
                            else getattr(hit, "width")
                        )
                        height = float(
                            height_value if height_value is not None
                            else getattr(hit, "height")
                        )
                    except (AttributeError, TypeError, ValueError):
                        continue
                    center_x = x + width / 2
                    center_y = y + height / 2
                    if not self._is_map_icon_search_area(center_x, center_y):
                        continue
                    same = next(
                        (
                            item for item in observed
                            if math.hypot(item.x - center_x, item.y - center_y)
                            <= 18 * self._map_icon_scale()
                        ),
                        None,
                    )
                    if same is None:
                        observed.append(_ObservedMapIcon(
                            center_x, center_y, {icon_type},
                        ))
                    else:
                        same.icon_types.add(icon_type)
        return observed

    @staticmethod
    def _assign_map_icon_pairs(
        expected: list[_ExpectedMapIcon],
        observed: list[_ObservedMapIcon],
        offset_x: float,
        offset_y: float,
        max_error: float,
    ) -> list[tuple[_ExpectedMapIcon, _ObservedMapIcon, float]]:
        candidates: list[tuple[float, int, int]] = []
        for expected_index, expected_icon in enumerate(expected):
            for observed_index, observed_icon in enumerate(observed):
                if not expected_icon.icon_types.intersection(observed_icon.icon_types):
                    continue
                error = math.hypot(
                    expected_icon.x + offset_x - observed_icon.x,
                    expected_icon.y + offset_y - observed_icon.y,
                )
                if error <= max_error:
                    candidates.append((error, expected_index, observed_index))
        candidates.sort(key=lambda item: item[0])
        used_expected: set[int] = set()
        used_observed: set[int] = set()
        pairs: list[tuple[_ExpectedMapIcon, _ObservedMapIcon, float]] = []
        for error, expected_index, observed_index in candidates:
            if expected_index in used_expected or observed_index in used_observed:
                continue
            used_expected.add(expected_index)
            used_observed.add(observed_index)
            pairs.append((expected[expected_index], observed[observed_index], error))
        return pairs

    @staticmethod
    def _estimate_map_icon_alignment(
        expected: list[_ExpectedMapIcon],
        observed: list[_ObservedMapIcon],
        scale: float = 1.0,
    ) -> _MapIconAlignment:
        """Estimate the small absolute screen translation between map layers."""
        inlier_radius = ABSOLUTE_ICON_INLIER_RADIUS_REF * scale
        max_correction = ABSOLUTE_ICON_MAX_CORRECTION_REF * scale
        bucket_size = max(1.0, ABSOLUTE_ICON_OFFSET_BUCKET_REF * scale)

        def score(offset_x: float, offset_y: float):
            pairs = TpTask._assign_map_icon_pairs(
                expected, observed, offset_x, offset_y, inlier_radius,
            )
            mean_error = (
                sum(pair[2] for pair in pairs) / len(pairs)
                if pairs else float("inf")
            )
            return pairs, mean_error

        hypotheses: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for expected_icon in expected:
            for observed_icon in observed:
                if not expected_icon.icon_types.intersection(observed_icon.icon_types):
                    continue
                offset_x = observed_icon.x - expected_icon.x
                offset_y = observed_icon.y - expected_icon.y
                if abs(offset_x) > max_correction or abs(offset_y) > max_correction:
                    continue
                key = (
                    round(offset_x / bucket_size),
                    round(offset_y / bucket_size),
                )
                hypotheses.setdefault(key, []).append((offset_x, offset_y))

        transforms = [(0.0, 0.0)]
        transforms.extend(
            (
                sum(value[0] for value in values) / len(values),
                sum(value[1] for value in values) / len(values),
            )
            for values in sorted(
                hypotheses.values(),
                key=lambda values: (
                    -len(values),
                    math.hypot(
                        sum(value[0] for value in values) / len(values),
                        sum(value[1] for value in values) / len(values),
                    ),
                ),
            )[:ABSOLUTE_ICON_MAX_HYPOTHESES]
        )

        best_offset = (0.0, 0.0)
        best_pairs, best_error = score(*best_offset)
        for offset in transforms[1:]:
            pairs, mean_error = score(*offset)
            if len(pairs) > len(best_pairs) or (
                len(pairs) == len(best_pairs)
                and (mean_error < best_error - 0.25
                     or (abs(mean_error - best_error) <= 0.25
                         and math.hypot(*offset) < math.hypot(*best_offset)))
            ):
                best_offset, best_pairs, best_error = offset, pairs, mean_error

        correction = math.hypot(*best_offset)
        if (
            not best_pairs
            or (len(best_pairs) == 1 and correction > 8 * scale)
            or (len(best_pairs) == 2 and correction > 10 * scale)
        ):
            identity_pairs, identity_error = score(0.0, 0.0)
            return _MapIconAlignment(
                0.0,
                0.0,
                len(identity_pairs),
                0.0 if not identity_pairs else identity_error,
            )

        if best_pairs:
            xs = sorted(pair[1].x - pair[0].x for pair in best_pairs)
            ys = sorted(pair[1].y - pair[0].y for pair in best_pairs)
            middle = len(xs) // 2
            refined = (
                xs[middle] if len(xs) % 2 else (xs[middle - 1] + xs[middle]) / 2,
                ys[middle] if len(ys) % 2 else (ys[middle - 1] + ys[middle]) / 2,
            )
            refined_pairs, refined_error = score(*refined)
            if len(refined_pairs) > len(best_pairs) or (
                len(refined_pairs) == len(best_pairs)
                and refined_error < best_error - 0.25
            ):
                best_offset, best_pairs, best_error = refined, refined_pairs, refined_error

        return _MapIconAlignment(
            best_offset[0], best_offset[1], len(best_pairs),
            0.0 if not best_pairs else best_error,
        )

    def _absolute_map_click_point(
        self,
        region: ImageRegion,
        view: tuple[float, float, float],
        raw_x: float,
        raw_y: float,
    ) -> tuple[float, float] | None:
        expected = self._expected_visible_map_icons(view)
        if not expected:
            return None
        allowed_types = set().union(*(item.icon_types for item in expected))
        observed = self._observed_visible_map_icons(region, allowed_types)
        alignment = self._estimate_map_icon_alignment(
            expected, observed, self._map_icon_scale(),
        )
        correction = math.hypot(alignment.offset_x, alignment.offset_y)
        if alignment.pair_count == 0 or correction < 1.0:
            return None

        nearby = sorted(
            math.hypot(item.x - raw_x, item.y - raw_y)
            for item in expected
            if math.hypot(item.x - raw_x, item.y - raw_y) > 8 * self._map_icon_scale()
        )
        scale = self._map_icon_scale()
        max_error = (
            max(8 * scale, nearby[0] * ABSOLUTE_ICON_NEIGHBOR_ERROR_RATIO)
            if nearby else float("inf")
        )
        if alignment.pair_count == 1:
            uncertainty = max(10 * scale, alignment.mean_error + 6 * scale)
        elif alignment.pair_count == 2:
            uncertainty = max(7 * scale, alignment.mean_error + 4 * scale)
        else:
            uncertainty = max(4 * scale, alignment.mean_error + 3 * scale)
        if uncertainty > max_error or correction > max_error:
            self.log(
                f"[tp] 地图图标校正不安全：校正 {correction:.1f}px，"
                f"误差 {uncertainty:.1f}px，上限 {max_error:.1f}px"
            )
            return None

        corrected = (raw_x + alignment.offset_x, raw_y + alignment.offset_y)
        if not self._is_clickable_map_point(*corrected):
            self.log("[tp] 地图图标校正后目标位于不可点击区域，保留原始点")
            return None
        self.log(
            f"[tp] 地图图标绝对校正：({raw_x:.1f},{raw_y:.1f}) -> "
            f"({corrected[0]:.1f},{corrected[1]:.1f})，"
            f"匹配 {alignment.pair_count} 个"
        )
        return corrected

    @classmethod
    def _is_anchor_entry_text(cls, value: str) -> bool:
        """Accept only short, recognizable rows from the overlap list."""
        text = cls._normalize_panel_text(value)
        return (
            1 < len(text) <= cls._ANCHOR_ENTRY_MAX_TEXT_LENGTH
            and any(key in text for key in cls._ANCHOR_ENTRIES)
        )

    def _is_confirm_hit(self, hit) -> bool:
        """Reject map labels that happen to contain the word 传送."""
        width = float(getattr(self.ctx.transform, "device_width", 1920))
        height = float(getattr(self.ctx.transform, "device_height", 1080))
        try:
            x = float(getattr(hit, "dx"))
            y = float(getattr(hit, "dy"))
            w = float(getattr(hit, "dw"))
            h = float(getattr(hit, "dh"))
        except (AttributeError, TypeError, ValueError):
            # Keep compatibility with lightweight recognition fakes and old
            # headless responses that only expose text/click().
            return True
        center_x = x + w / 2
        center_y = y + h / 2
        return center_x >= width * 0.58 and center_y >= height * 0.55

    def _anchor_icons_near(
        self,
        x: float,
        y: float,
        max_distance: float,
        *,
        region: ImageRegion | None = None,
    ):
        """Return nearby teleport icons ordered by distance from the raw point."""
        # The caller normally already owns the post-map frame used to compute
        # ``x/y``. Reusing it avoids a second DeviceHub screenshot request,
        # which is important on the slow iPhone stream: a new request can
        # describe the map before the preceding gesture has been rendered.
        region = region if region is not None else self.ctx.capture_region()
        width = float(self.ctx.transform.device_width)
        height = float(self.ctx.transform.device_height)
        candidates = []
        for name in ("TeleportWaypoint", "StatueOfTheSeven", "Domain"):
            tpl = Mat.from_file(str(TEMPLATES / f"{name}.png"))
            for h in region.find_multi(RecognitionObject.template_match(tpl), limit=5):
                center_x = h.dx + h.dw / 2
                center_y = h.dy + h.dh / 2
                margin = max(35.0, 0.035 * min(width, height))
                if (
                    center_x < margin or center_y < margin
                    or center_x > width - margin or center_y > height - margin
                    or (center_x < 0.20 * width and center_y < 0.35 * height)
                ):
                    continue
                d = math.hypot(center_x - x, center_y - y)
                if d <= max_distance:
                    candidates.append((d, h))
        candidates.sort(key=lambda item: item[0])
        return candidates

    def _tap_anchor_icon_near(
        self,
        x: float,
        y: float,
        max_distance: float,
        *,
        region: ImageRegion | None = None,
    ) -> bool:
        """模板匹配目标附近的传送锚点/神像图标并点击最近者。"""
        if region is None:
            candidates = self._anchor_icons_near(x, y, max_distance)
        else:
            candidates = self._anchor_icons_near(
                x, y, max_distance, region=region,
            )
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

    def tp(
        self,
        wx: float,
        wy: float,
        timeout_s: float = 90,
        *,
        force: bool = False,
    ) -> bool:
        with self.exclusive_triggers():
            try:
                return self._tp(wx, wy, timeout_s, force=force)
            except RuntimeError:
                # Keep direct PathingExecutor callers safe as well as the
                # genshin API retry wrapper: a failed point panel must not
                # remain over the next map attempt.
                self._dismiss_teleport_panel()
                raise

    def _tp(
        self,
        wx: float,
        wy: float,
        timeout_s: float = 90,
        *,
        force: bool = False,
    ) -> bool:
        """传送到世界坐标 (wx, wy) 附近的锚点。"""
        target_x, target_y, target_country, target_point = self._resolve_tp_target(
            wx, wy, force=force,
        )
        self.log(f"[tp] 目标世界坐标 ({target_x:.1f}, {target_y:.1f})")
        if target_point is not None:
            label = target_point.name or target_point.point_type or "传送点"
            distance = math.hypot(float(wx) - target_point.x, float(wy) - target_point.y)
            self.log(
                f"[tp] force=false 吸附最近传送点：{label}"
                f"（距离 {distance:.1f}）"
            )
        if force:
            self.log("[tp] 使用 force 坐标，不吸附到最近传送点")
        tx, ty = self.big.world_to_feature(target_x, target_y)
        t = self.ctx.transform
        view = self._move_map_view_to(
            target_x,
            target_y,
            timeout_s,
            area_name=target_country,
            log_prefix="[tp] 迭代",
            max_iterations=TP_MOVE_MAX_ITERATIONS,
            error_message="传送失败：未能把目标移动到可点击区域（迭代/超时耗尽）",
        )
        vx, vy, px_per_map = view
        dx_screen = (tx - vx) * px_per_map
        dy_screen = (ty - vy) * px_per_map
        tol = 0.05 * t.device_width
        tap_x = t.device_width / 2 + dx_screen
        tap_y = t.device_height / 2 + dy_screen
        if not self._is_clickable_map_point(tap_x, tap_y):
            raise RuntimeError("传送失败：目标点仍位于大地图不可点击区域")
        frame = getattr(self, "_last_located_frame", None)
        if frame is None:
            # Compatibility fallback for callers that provide a custom map
            # mover without going through _move_map_view_to.
            frame = self.ctx.capture_bgr()
        map_region = ImageRegion(self.ctx, frame)
        corrected_point = self._absolute_map_click_point(
            map_region, view, tap_x, tap_y,
        )
        if not self._select_target_and_confirm(
            tap_x,
            tap_y,
            tol,
            target_point=target_point,
            corrected_point=corrected_point,
            map_region=map_region,
        ):
            raise TeleportPanelNotOpenedError(
                "传送失败：点击传送点后未出现交互面板，可能是传送点未激活"
            )
        self.log("[tp] 已确认传送，等待加载…")
        self._wait_for_teleport_completion()
        return True

    def _resolve_tp_target(
        self,
        wx: float,
        wy: float,
        *,
        force: bool,
    ) -> tuple[float, float, str | None, TeleportPoint | None]:
        """Apply BetterGI's nearest-point rule unless ``force`` is enabled."""
        request_x, request_y = float(wx), float(wy)
        if force:
            return request_x, request_y, None, None
        point = self._teleport_points.nearest_point(
            self.map_name, request_x, request_y,
        )
        if point is None:
            # Keep routes usable with an older checkout that has no tp.json.
            self.log("[tp] 未找到传送点索引，force=false 回退原始坐标")
            return request_x, request_y, None, None
        return point.x, point.y, point.country, point

    def _is_clickable_map_point(self, x: float, y: float) -> bool:
        width = float(self.ctx.transform.device_width)
        height = float(self.ctx.transform.device_height)
        margin = max(35.0, 0.035 * min(width, height))
        if x < margin or y < margin or x > width - margin or y > height - margin:
            return False
        return not (x < 0.20 * width and y < 0.35 * height)

    def _dismiss_teleport_panel(self) -> None:
        """Close a possibly stale map selection before a retry."""
        try:
            input_controller = getattr(self.ctx, "input", None)
            key_press = getattr(input_controller, "key_press", None)
            if callable(key_press):
                key_press("ESCAPE")
            else:
                self.ctx.device.press_key("ESCAPE")
            self.ctx.sleep(350)
        except Exception as error:
            self.log(f"[tp] 关闭传送面板失败（忽略）：{error}")

    def dismiss_after_failure(self) -> None:
        """Public retry cleanup used by the genshin API wrapper."""
        self._dismiss_teleport_panel()

    def _confirm_selected_target(self, target_point: TeleportPoint | None) -> bool:
        """Keep the no-target call shape compatible with lightweight test hosts."""
        if target_point is None:
            return self._find_and_tap_confirm()
        return self._find_and_tap_confirm(target_point=target_point)

    def _select_target_and_confirm(
        self,
        tap_x: float,
        tap_y: float,
        tol: float,
        *,
        target_point: TeleportPoint | None = None,
        corrected_point: tuple[float, float] | None = None,
        map_region: ImageRegion | None = None,
    ) -> bool:
        """Click one resolved point and allow one precomputed fallback only."""
        width = self.ctx.transform.device_width
        if corrected_point is not None:
            corrected_x, corrected_y = corrected_point
            self.log("[tp] 先点击地图图标绝对校正坐标")
            self.ctx.device.tap(
                corrected_x,
                corrected_y,
                image_width=width,
                image_height=self.ctx.transform.device_height,
            )
            if self._confirm_selected_target(target_point):
                return True
            self.log("[tp] 绝对校正坐标未弹出面板，回退原始目标点")

        if map_region is None:
            candidates = self._anchor_icons_near(
                tap_x,
                tap_y,
                max(0.10 * width, 1.8 * tol),
            )
        else:
            candidates = self._anchor_icons_near(
                tap_x,
                tap_y,
                max(0.10 * width, 1.8 * tol),
                region=map_region,
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

        if self._confirm_selected_target(target_point):
            return True
        if fallback is None:
            return False

        self.log("[tp] 原始点未弹出面板，回退一次校正锚点")
        fallback.click()
        return self._confirm_selected_target(target_point)

    def _wait_for_teleport_completion(self, timeout_s: float = 60) -> None:
        """Require a loading state followed by a stable gameplay minimap."""
        deadline = time.monotonic() + timeout_s
        started_at = time.monotonic()
        observed_loading = False
        stable_main_ui = 0
        while time.monotonic() < deadline:
            frame = self.ctx.capture_bgr()
            in_main_ui = is_main_ui(self.ctx, frame)
            if in_main_ui:
                if observed_loading and time.monotonic() - started_at >= TELEPORT_COMPLETION_MINIMUM_S:
                    stable_main_ui += 1
                    if stable_main_ui >= TELEPORT_COMPLETION_STABLE_CHECKS:
                        self.log("[tp] 传送完成")
                        return
            else:
                stable_main_ui = 0
                if not observed_loading and not is_big_map_ui(self.ctx, frame):
                    observed_loading = True
            self.ctx.sleep(350)
        self.log("[tp] 传送加载等待超时，继续执行后续任务")
