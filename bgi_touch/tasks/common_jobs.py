"""Portable equivalents of BetterGI's small, reusable Common Job tasks.

The desktop implementations of these jobs are used by several independent
task families.  Keeping their iOS versions in one module gives those callers
the same cancellation, timeout, and input-cleanup behaviour without creating a
new screenshot producer for every feature.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..engine.context import GameContext
from ..engine.recognition import ImageRegion, Mat, RecognitionObject


PROJECT_ROOT = Path(__file__).resolve().parents[2]
F_TEMPLATE_PATH = PROJECT_ROOT / "assets" / "templates" / "autopick" / "F.png"


def _cancelled(cancelled: Callable[[], bool] | None) -> bool:
    try:
        return bool(cancelled and cancelled())
    except Exception:
        # A cancellation callback must never leave movement held if a host
        # object disappears during script shutdown.
        return True


def _compact(value: object) -> str:
    return "".join(str(value or "").split()).casefold()


def _pause_realtime_triggers(ctx: GameContext):
    """Pause the caller-owned screenshot producer while a job consumes frames."""
    try:
        loop = getattr(ctx, "triggers", None)
        pause = getattr(loop, "pause", None)
        if callable(pause):
            return loop, pause()
    except Exception:
        # A minimal script host may expose no trigger loop.  The job can still
        # run with its own capture path in that environment.
        pass
    return None, None


def _resume_realtime_triggers(loop, state) -> None:
    if loop is None or state is None:
        return
    try:
        loop.resume(state)
    except Exception:
        pass


class InteractionPromptDetector:
    """Detect the PC-style F prompt and the mobile interaction text.

    DeviceHub's keyboard profile can expose the desktop prompt in some game
    layouts, while the normal iOS HUD exposes an OCR interaction list instead.
    The template is therefore the cheap first probe and OCR is a compatibility
    fallback.  Both probes operate on a caller-owned frame.
    """

    PROMPT_ROI = (1060, 280, 500, 500)
    TEXTS = frozenset({
        "f", "交互", "激活", "接触", "开启", "开始", "调查", "领取",
        "进入", "挑战", "启动", "拾取", "采集",
    })

    def __init__(
        self,
        ctx: GameContext,
        *,
        roi: tuple[float, float, float, float] | None = None,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.roi = roi or self.PROMPT_ROI
        self.log = log
        self._template: RecognitionObject | None = None
        self._template_failed = False

    def _template_object(self) -> RecognitionObject | None:
        if self._template is not None or self._template_failed:
            return self._template
        try:
            self._template = RecognitionObject.template_match(
                Mat.from_file(str(F_TEMPLATE_PATH)), *self.roi,
            )
            # The 32px upstream asset is intentionally allowed a little more
            # tolerance after safe-area scaling and JPEG/video compression.
            self._template.threshold = 0.58
        except (OSError, ValueError, TypeError) as error:
            self._template_failed = True
            self.log(f"[common-job] F 键模板不可用，改用 OCR：{error}")
        return self._template

    def _template_visible(self, region: ImageRegion) -> bool:
        template = self._template_object()
        if template is None:
            return False
        try:
            return region.find(template).is_exist()
        except Exception as error:
            self.log(f"[common-job] F 键模板识别失败，改用 OCR：{error}")
            return False

    def _text_visible(self, region: ImageRegion, *, activation_only: bool = False) -> bool:
        try:
            hits = region.find_multi(
                RecognitionObject.ocr(*self.roi), limit=12,
            )
        except Exception as error:
            self.log(f"[common-job] 交互 OCR 失败：{error}")
            return False
        for hit in hits:
            text = _compact(getattr(hit, "text", ""))
            if not text:
                continue
            if activation_only:
                if any(word in text for word in ("激活", "接触", "地脉之花")):
                    return True
            elif text in self.TEXTS or any(word in text for word in self.TEXTS if len(word) > 1):
                return True
        return False

    def visible(self, region: ImageRegion, *, activation_only: bool = False) -> bool:
        if self._template_visible(region):
            return True
        return self._text_visible(region, activation_only=activation_only)


class WalkToFTask:
    """Walk forward until an interaction prompt appears.

    This mirrors ``Common/Job/WalkToFTask`` while using the iOS profile's
    mapped W/Shift/F actions.  ``run_to_f`` is intentionally named after the
    upstream parameter; callers that use the older ``!WalkToF`` setting can
    pass the inverted value in the dispatcher.
    """

    def __init__(
        self,
        ctx: GameContext,
        *,
        need_press: bool = True,
        run_to_f: bool = False,
        timeout_s: float = 30.0,
        poll_interval_ms: int = 100,
        log: Callable[[str], None] = print,
        detector: InteractionPromptDetector | Callable[[ImageRegion], bool] | None = None,
    ):
        self.ctx = ctx
        self.need_press = bool(need_press)
        self.run_to_f = bool(run_to_f)
        self.timeout_s = max(0.1, float(timeout_s))
        self.poll_interval_ms = max(30, int(poll_interval_ms))
        self.log = log
        self.detector = detector or InteractionPromptDetector(ctx, log=log)

    def _visible(self, region: ImageRegion) -> bool:
        detector = self.detector
        if callable(detector) and not isinstance(detector, InteractionPromptDetector):
            return bool(detector(region))
        return detector.visible(region)  # type: ignore[union-attr]

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        deadline = time.monotonic() + self.timeout_s
        trigger_loop, trigger_state = _pause_realtime_triggers(self.ctx)
        try:
            self.ctx.input.key_down("W")
            if self.run_to_f:
                self.ctx.input.key_down("LSHIFT")
            while time.monotonic() < deadline:
                if _cancelled(cancelled):
                    return False
                try:
                    region = self.ctx.capture_region()
                except Exception as error:
                    self.log(f"[WalkToF] 截图失败：{error}")
                    self.ctx.sleep(self.poll_interval_ms)
                    continue
                if self._visible(region):
                    if self.need_press:
                        self.ctx.input.key_press("F")
                    self.log("[WalkToF] 检测到交互键")
                    return True
                self.ctx.sleep(self.poll_interval_ms)
            self.log("[WalkToF] 前往目标[F]超时")
            return False
        finally:
            self.ctx.input.key_up("W")
            if self.run_to_f:
                self.ctx.input.key_up("LSHIFT")
            _resume_realtime_triggers(trigger_loop, trigger_state)

    # C#-style spelling is useful to converted scripts and keeps this class
    # interchangeable with the upstream job in small Python adapters.
    Start = run


@dataclass(frozen=True)
class PickItem:
    """A pickable rectangle in reference (1920x1080) coordinates."""

    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def bottom(self) -> float:
        return self.y + self.height


def _read_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        wanted = key.replace("_", "").casefold()
        for candidate, item in value.items():
            if str(candidate).replace("_", "").casefold() == wanted:
                return item
        return default
    try:
        result = getattr(value, key)
    except (AttributeError, TypeError):
        return default
    return default if result is None else result


def _as_pick_item(raw: Any, transform: Any) -> PickItem | None:
    """Normalize detector output from mappings, YOLO detections, or Regions."""
    def member(name: str, fallback: Any = None):
        if isinstance(raw, Mapping):
            return raw.get(name, raw.get(name.capitalize(), fallback))
        return getattr(raw, name, getattr(raw, name.capitalize(), fallback))

    try:
        x = float(member("x"))
        y = float(member("y"))
        width = float(member("width", member("w")))
        height = float(member("height", member("h")))
        confidence = float(member("confidence", member("score", 1.0)))
    except (TypeError, ValueError):
        return None
    # YoloPredictor returns device-pixel rectangles; convert them to the
    # reference canvas used by the original scan geometry.  Detector adapters
    # may mark an item as reference-space to avoid this conversion.
    if bool(member("device_space", member("deviceSpace", False))):
        try:
            left, top = transform.to_ref(x, y)
            right, bottom = transform.to_ref(x + width, y + height)
            x, y, width, height = left, top, right - left, bottom - top
        except (AttributeError, TypeError, ValueError):
            pass
    if width <= 0 or height <= 0:
        return None
    return PickItem(x, y, width, height, confidence)


class ScanPickTask:
    """Scan nearby drops without competing with the realtime frame loop.

    On iOS the normal path delegates item selection to the shared AutoPick
    trigger and only emits the same camera/drop sweep as BetterGI.  A supplied
    detector (or an optional ``bgi_world`` model) enables the more precise
    move-towards-item path when that asset is installed.
    """

    def __init__(
        self,
        ctx: GameContext,
        *,
        seconds: float = 15.0,
        camera_step: float = 180.0,
        sweep_interval_ms: int = 700,
        forward_after_turns: int = 6,
        detector: Callable[[np.ndarray], Any] | None = None,
        use_world_model: bool = False,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.ctx = ctx
        self.seconds = max(0.0, min(120.0, float(seconds)))
        self.camera_step = float(camera_step)
        self.sweep_interval_ms = max(100, int(sweep_interval_ms))
        self.forward_after_turns = max(0, int(forward_after_turns))
        self.detector = detector
        self.use_world_model = bool(use_world_model)
        self.log = log
        self.clock = clock
        self._predictor = None
        self._predictor_attempted = False

    def _load_predictor(self):
        if self.detector is not None or self._predictor_attempted or not self.use_world_model:
            return self._predictor
        self._predictor_attempted = True
        try:
            from ..vision.yolo import YoloPredictor

            self._predictor = YoloPredictor("bgi_world")
        except (FileNotFoundError, ImportError, OSError, RuntimeError) as error:
            self.log(f"[ScanPick] bgi_world 模型不可用，使用 AutoPick 扫描：{error}")
        return self._predictor

    def _detect(self, frame: np.ndarray) -> list[PickItem]:
        detector = self.detector
        predictor_output = False
        if detector is None:
            predictor = self._load_predictor()
            if predictor is None:
                return []
            predictor_output = True
            detector = lambda image: predictor.predict(image, conf_threshold=0.4)
        try:
            raw_items = detector(frame)
        except TypeError:
            # A few adapters expose ``predict`` rather than a callable.
            raw_items = detector.predict(frame, conf_threshold=0.4)  # type: ignore[attr-defined]
        if isinstance(raw_items, Mapping):
            raw_items = [
                item for name, values in raw_items.items()
                if str(name).casefold() in {"drops", "drop", "ore", "items"}
                for item in (values if isinstance(values, (list, tuple)) else [values])
            ]
        if raw_items is None:
            return []
        transform = getattr(self.ctx, "transform", None)
        result = []
        for raw in raw_items:
            if predictor_output:
                # YoloPredictor coordinates are native frame pixels, while
                # injected test/script detectors conventionally use reference
                # coordinates.  Keep the distinction explicit at this edge.
                raw = {
                    "x": getattr(raw, "x", None),
                    "y": getattr(raw, "y", None),
                    "width": getattr(raw, "width", None),
                    "height": getattr(raw, "height", None),
                    "confidence": getattr(raw, "confidence", 1.0),
                    "device_space": True,
                }
            item = _as_pick_item(raw, transform)
            if item is not None:
                result.append(item)
        return result

    def _move_towards(self, item: PickItem) -> None:
        """Keep horizontal and vertical movement mutually exclusive."""
        if item.center_x < 880:
            self.ctx.input.key_up("D")
            self.ctx.input.key_down("A")
        elif item.center_x > 1040:
            self.ctx.input.key_up("A")
            self.ctx.input.key_down("D")
        else:
            self.ctx.input.key_up("A")
            self.ctx.input.key_up("D")

        if item.bottom < 770:
            self.ctx.input.key_up("S")
            self.ctx.input.key_down("W")
        elif item.bottom > 900:
            self.ctx.input.key_up("W")
            self.ctx.input.key_down("S")
        else:
            self.ctx.input.key_up("W")
            self.ctx.input.key_up("S")

    def _tap_elemental_sight(self) -> None:
        tap_button = getattr(self.ctx.input, "tap_button", None)
        if callable(tap_button):
            tap_button("elementalSight")
        else:
            self.ctx.input.key_press("X")

    def _reset_camera(self) -> None:
        # The upstream Drop action recentres the camera.  On the mobile
        # profile the same semantic is exposed by the elemental-sight button;
        # keep the vertical correction as a separate gesture.
        self.ctx.input.key_press("X")
        self._tap_elemental_sight()
        self.ctx.sleep(500)
        self.ctx.input.move_camera_by(0, 500)
        self.ctx.sleep(100)

    def _prepare_autopick(self):
        loop = getattr(self.ctx, "triggers", None)
        if loop is None:
            return None, False
        try:
            previous = list(loop.triggers)
            if loop.get("AutoPick") is None:
                self.ctx.enable_trigger("AutoPick")
                return previous, True
            return previous, False
        except Exception as error:
            self.log(f"[ScanPick] AutoPick 触发器未能启用：{error}")
            return None, False

    def _restore_autopick(self, previous, owned: bool) -> None:
        if not owned or previous is None:
            return
        loop = getattr(self.ctx, "triggers", None)
        if loop is None:
            return
        try:
            loop.replace(previous)
            if not previous:
                loop.stop()
        except Exception as error:
            self.log(f"[ScanPick] 恢复 AutoPick 触发器失败：{error}")

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        if self.seconds <= 0:
            return True
        previous, owned = self._prepare_autopick()
        predictor_available = (
            self.detector is not None or self._load_predictor() is not None
        )
        deadline = self.clock() + self.seconds
        max_iterations = max(1, int(math.ceil(self.seconds * 1000 / self.sweep_interval_ms)) + 2)
        iterations = 0
        self.log(f"[ScanPick] 开始扫描周边掉落（{self.seconds:.0f}s）")
        try:
            self._reset_camera()
            while self.clock() < deadline and iterations < max_iterations:
                iterations += 1
                if _cancelled(cancelled):
                    return False
                items: list[PickItem] = []
                if predictor_available:
                    try:
                        frame = self.ctx.capture_bgr()
                        if isinstance(frame, np.ndarray):
                            items = self._detect(frame)
                    except Exception as error:
                        self.log(f"[ScanPick] 掉落识别失败，继续扫圈：{error}")
                if items:
                    target = min(
                        items,
                        key=lambda item: (item.center_x - 960) ** 2
                        + 14 * (item.bottom - 888.88) ** 2,
                    )
                    self._move_towards(target)
                    self.ctx.sleep(200)
                    self.ctx.input.key_press("F")
                else:
                    self.ctx.input.move_camera_by(self.camera_step, 0)
                    if iterations > self.forward_after_turns:
                        self.ctx.input.key_down("W")
                        try:
                            self.ctx.sleep(100)
                        finally:
                            self.ctx.input.key_up("W")
                    # If AutoPick could not be installed, retain a direct
                    # interaction fallback for the mapped mobile button.
                    if previous is None:
                        self.ctx.input.key_press("F")
                self.ctx.sleep(self.sweep_interval_ms)
            self.log("[ScanPick] 扫描结束")
            return not _cancelled(cancelled)
        finally:
            self.ctx.input.key_up("W")
            self.ctx.input.key_up("A")
            self.ctx.input.key_up("S")
            self.ctx.input.key_up("D")
            # Do not tear down a caller-owned profile session when another
            # realtime trigger was already active.  In the standalone path
            # release_all also closes the short-lived game session cleanly.
            if previous is None:
                self.ctx.input.release_all()
            self._restore_autopick(previous, owned)

    Start = run


class LowerHeadThenWalkToTask:
    """Lower the camera and walk toward a tracked template until interaction."""

    def __init__(
        self,
        ctx: GameContext,
        target_mat_name: str,
        *,
        timeout_s: float = 30.0,
        threshold: float = 0.6,
        log: Callable[[str], None] = print,
        track_detector: Callable[[ImageRegion], Any] | None = None,
        interaction_detector: InteractionPromptDetector | None = None,
    ):
        self.ctx = ctx
        self.target_mat_name = str(target_mat_name or "").strip()
        self.timeout_s = max(0.1, float(timeout_s))
        self.threshold = float(threshold)
        self.log = log
        self.track_detector = track_detector
        self.interaction_detector = interaction_detector or InteractionPromptDetector(
            ctx, log=log,
        )
        self._track_object: RecognitionObject | None = None
        self._track_failed = False

    def _target_path(self) -> Path | None:
        raw = Path(self.target_mat_name).expanduser()
        candidates = [raw] if raw.is_absolute() else []
        candidates.extend([
            PROJECT_ROOT / "assets" / "templates" / self.target_mat_name,
            PROJECT_ROOT / "assets" / "templates" / "stygian" / raw.name,
            PROJECT_ROOT / "assets" / "templates" / "common" / raw.name,
        ])
        return next((candidate for candidate in candidates if candidate.is_file()), None)

    def _target_object(self) -> RecognitionObject | None:
        if self._track_object is not None or self._track_failed:
            return self._track_object
        path = self._target_path()
        if path is None:
            self._track_failed = True
            self.log(f"[LowerHeadThenWalkTo] 未找到追踪模板：{self.target_mat_name}")
            return None
        try:
            self._track_object = RecognitionObject.template_match(
                Mat.from_file(str(path)), 300, 0, 1320, 1080,
            )
            self._track_object.threshold = self.threshold
        except (OSError, ValueError, TypeError) as error:
            self._track_failed = True
            self.log(f"[LowerHeadThenWalkTo] 追踪模板加载失败：{error}")
        return self._track_object

    def _find_target(self, region: ImageRegion):
        if self.track_detector is not None:
            return self.track_detector(region)
        target = self._target_object()
        return None if target is None else region.find(target)

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        trigger_loop, trigger_state = _pause_realtime_triggers(self.ctx)
        try:
            first = self.ctx.capture_region()
            initial = self._find_target(first)
            if initial is None or not getattr(initial, "is_exist", lambda: False)():
                self.log("[LowerHeadThenWalkTo] 未找到追踪点，停止任务")
                return False

            deadline = time.monotonic() + self.timeout_s
            previous_move = 0
            while time.monotonic() < deadline:
                if _cancelled(cancelled):
                    return False
                region = self.ctx.capture_region()
                hit = self._find_target(region)
                if hit is not None and getattr(hit, "is_exist", lambda: False)():
                    center_x = float(hit.x + hit.width / 2)
                    center_y = float(hit.y + hit.height / 2)
                    height = float(getattr(self.ctx.transform, "device_height", 1080))
                    height_ref = height / max(0.001, float(getattr(self.ctx.transform, "scale", 1.0)))
                    if center_y > height_ref / 2:
                        self.ctx.input.key_up("W")
                        self.ctx.input.move_camera_by(-50, 0)
                        self.ctx.input.move_camera_by(0, 800)
                        self.ctx.sleep(100)
                        continue

                    move_x = int((center_x - 960) / 8)
                    if abs(move_x) < 10 and move_x:
                        move_x = 10 if move_x > 0 else -10
                    if move_x:
                        self.ctx.input.move_camera_by(move_x, 0)
                    if move_x == 0 or previous_move * move_x < 0:
                        self.ctx.input.key_down("W")
                    else:
                        self.ctx.input.key_up("W")
                    previous_move = move_x

                    if self.interaction_detector.visible(region, activation_only=True):
                        self.ctx.input.key_up("W")
                        self.log("[LowerHeadThenWalkTo] 检测到目标交互提示")
                        return True
                self.ctx.input.move_camera_by(0, 800)
                self.ctx.sleep(100)
            self.log("[LowerHeadThenWalkTo] 追踪超时")
            return False
        finally:
            self.ctx.input.key_up("W")
            self.ctx.input.release_all()
            _resume_realtime_triggers(trigger_loop, trigger_state)

    Start = run
