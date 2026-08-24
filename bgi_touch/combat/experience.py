"""Combat experience-icon detection for elite-drop pickup decisions.

BetterGI uses the small ``experience_57/58/60`` icons as a cheap signal that
an elite enemy has died.  The desktop implementation runs a second capture
loop; the iOS runtime deliberately keeps this detector frame-fed so it shares
the AutoFight/DeviceHub capture path and cannot compete with the screenshot
producer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ..engine.recognition import ImageRegion, Mat, RecognitionObject


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_ROOT = PROJECT_ROOT / "assets" / "templates" / "autofight"
EXPERIENCE_TEMPLATE_NAMES = ("experience_57.png", "experience_58.png", "experience_60.png")


@dataclass(frozen=True)
class ExperienceDetectorConfig:
    """Visual thresholds expressed in BetterGI's 1920x1080 reference space."""

    enabled: bool = True
    interval_s: float = 0.1
    template_threshold: float = 0.82
    pixel_offset_x: float = -147.0
    color_min_bgr: tuple[int, int, int] = (200, 200, 150)
    color_max_bgr: tuple[int, int, int] = (255, 255, 200)
    sample_radius: int = 1

    @classmethod
    def from_mapping(cls, raw: object) -> "ExperienceDetectorConfig":
        """Read optional detector overrides without making config mandatory."""

        if not isinstance(raw, dict):
            return cls()

        def number(name: str, default: float, minimum: float | None = None) -> float:
            try:
                value = float(raw.get(name, default))
            except (TypeError, ValueError):
                return default
            if not np.isfinite(value):
                return default
            return max(minimum, value) if minimum is not None else value

        def integer(name: str, default: int, minimum: int = 0) -> int:
            try:
                return max(minimum, int(raw.get(name, default)))
            except (TypeError, ValueError):
                return default

        def color(name: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
            value = raw.get(name, default)
            if not isinstance(value, (list, tuple)) or len(value) < 3:
                return default
            try:
                channels = tuple(max(0, min(255, int(channel))) for channel in value[:3])
            except (TypeError, ValueError):
                return default
            return channels  # type: ignore[return-value]

        return cls(
            enabled=bool(raw.get("enabled", True)),
            interval_s=number("intervalSeconds", 0.1, 0.0),
            template_threshold=number("templateThreshold", 0.82, 0.0),
            pixel_offset_x=number("pixelOffsetX", -147.0),
            color_min_bgr=color("colorMinBgr", (200, 200, 150)),
            color_max_bgr=color("colorMaxBgr", (255, 255, 200)),
            sample_radius=integer("sampleRadius", 1, 0),
        )


def _load_templates(
    template_root: str | Path = DEFAULT_TEMPLATE_ROOT,
    config: ExperienceDetectorConfig | None = None,
) -> list[RecognitionObject]:
    """Load the fixed-size upstream templates, ignoring unavailable assets."""

    root = Path(template_root).expanduser()
    threshold = (config or ExperienceDetectorConfig()).template_threshold
    result: list[RecognitionObject] = []
    for name in EXPERIENCE_TEMPLATE_NAMES:
        path = root / name
        if not path.is_file():
            continue
        try:
            recognition = RecognitionObject.template_match(Mat.from_file(str(path)))
        except (OSError, ValueError):
            continue
        recognition.name = Path(name).stem
        recognition.threshold = threshold
        result.append(recognition)
    return result


def validate_experience_pixel(
    ctx,
    frame: np.ndarray,
    match_x: float,
    match_y: float,
    config: ExperienceDetectorConfig | None = None,
) -> bool:
    """Validate the characteristic pale-yellow pixel left of a template hit.

    ``match_x``/``match_y`` are device-pixel coordinates (``Region.dx/dy``),
    while the offset is a BetterGI reference-space distance.  Converting the
    offset through the current screen scale keeps the check valid on the wide
    iPhone canvas and on ordinary 16:9 captures.
    """

    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3:
        return False
    config = config or ExperienceDetectorConfig()
    transform = getattr(ctx, "transform", None)
    scale = float(getattr(transform, "scale", 1.0) or 1.0)
    check_x = round(float(match_x) + config.pixel_offset_x * scale)
    check_y = round(float(match_y))
    radius = max(0, int(config.sample_radius))
    height, width = frame.shape[:2]
    x0, x1 = max(0, check_x - radius), min(width, check_x + radius + 1)
    y0, y1 = max(0, check_y - radius), min(height, check_y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return False
    pixels = frame[y0:y1, x0:x1, :3].reshape(-1, 3).astype(np.int16)
    lower = np.asarray(config.color_min_bgr, dtype=np.int16)
    upper = np.asarray(config.color_max_bgr, dtype=np.int16)
    return bool(np.all((pixels >= lower) & (pixels <= upper), axis=1).any())


class ExperienceDetector:
    """Frame-fed, single-shot detector used during one AutoFight run."""

    def __init__(
        self,
        ctx,
        *,
        config: ExperienceDetectorConfig | None = None,
        templates: Iterable[RecognitionObject] | None = None,
        template_root: str | Path = DEFAULT_TEMPLATE_ROOT,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.ctx = ctx
        self.config = config or ExperienceDetectorConfig()
        self.log = log
        self.clock = clock
        self.templates = list(templates) if templates is not None else _load_templates(
            template_root, self.config,
        )
        for template in self.templates:
            template.threshold = self.config.template_threshold
        self._detected = False
        self._active = False
        self._last_scan_at = float("-inf")

    @property
    def has_detected_experience(self) -> bool:
        return self._detected

    # Match the upstream property name for task adapters and JS-facing code.
    HasDetectedExperience = property(lambda self: self.has_detected_experience)

    @property
    def available(self) -> bool:
        return bool(self.config.enabled and self.templates)

    def start(self) -> None:
        self._detected = False
        self._last_scan_at = float("-inf")
        self._active = bool(self.config.enabled and self.templates)
        if self.config.enabled and not self.templates:
            self.log("[AutoFight] 经验值检测无可用模板，跳过精英掉落判断")

    def stop(self) -> bool:
        self._active = False
        return self._detected

    def observe(self, frame: np.ndarray | None) -> bool:
        """Inspect one already-captured frame and return the latched result."""

        if not self._active or self._detected or not isinstance(frame, np.ndarray):
            return self._detected
        now = self.clock()
        if now - self._last_scan_at < max(0.0, self.config.interval_s):
            return False
        self._last_scan_at = now
        try:
            region = ImageRegion(self.ctx, frame)
            for template in self.templates:
                hit = region.find(template)
                if not hit.is_exist():
                    continue
                if not validate_experience_pixel(
                    self.ctx, frame, hit.dx, hit.dy, self.config,
                ):
                    continue
                self._detected = True
                label = template.name or "experience"
                self.log(f"[AutoFight] 经验值检测命中 {label}，启用战后拾取")
                break
        except Exception as error:
            # Recognition should never abort combat. The next frame may still
            # produce a valid hit after a transient orientation/asset failure.
            self.log(f"[AutoFight] 经验值检测失败，继续战斗：{error}")
        return self._detected

