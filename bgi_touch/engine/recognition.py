"""BetterGI 识别模型的移植：Mat / RecognitionObject / Region / ImageRegion。

- 脚本可见坐标一律为 1920x1080 参考空间；内部保留设备像素矩形供 click() 使用。
- 模板资产按 1080p 基准制作，匹配前按设备比例缩放。
- 全部同步，与原版 ClearScript 宿主语义一致。
"""

from __future__ import annotations

import copy
import math
import re
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from ..vision.coordinate import REF_WIDTH, ScreenTransform
from ..vision.ocr import get_ocr

if TYPE_CHECKING:
    from .context import GameContext


def _rect_tuple(value) -> tuple[float, float, float, float]:
    """Read OpenCvSharp.Rect, JS objects, mappings, or four-value arrays."""
    unwrapped = getattr(value, "__wrapped__", None)
    if unwrapped is not None:
        value = unwrapped
    if isinstance(value, dict):
        folded = {str(key).casefold(): item for key, item in value.items()}
        parts = (
            folded.get("x", 0), folded.get("y", 0),
            folded.get("width", 0), folded.get("height", 0),
        )
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        parts = value
    else:
        def member(lower: str, upper: str):
            for name in (lower, upper):
                try:
                    return value[name]
                except (KeyError, TypeError, AttributeError):
                    pass
                try:
                    return getattr(value, name)
                except (AttributeError, TypeError):
                    pass
            return 0

        parts = tuple(member(lower, upper) for lower, upper in (
            ("x", "X"), ("y", "Y"),
            ("width", "Width"), ("height", "Height"),
        ))
    return tuple(float(part) for part in parts)


def _rect_result(x: float, y: float, width: float, height: float) -> dict:
    return {
        "x": x, "y": y, "width": width, "height": height,
        "X": x, "Y": y, "Width": width, "Height": height,
    }


def _size_tuple(value) -> tuple[float, float]:
    """Read OpenCvSharp.Size, JS objects, mappings, or two-value arrays."""
    unwrapped = getattr(value, "__wrapped__", None)
    if unwrapped is not None:
        value = unwrapped
    if isinstance(value, dict):
        folded = {str(key).casefold(): item for key, item in value.items()}
        parts = (folded.get("width", 0), folded.get("height", 0))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        parts = (value[0], value[1])
    else:
        def member(lower: str, upper: str):
            for name in (lower, upper):
                try:
                    return value[name]
                except (KeyError, TypeError, AttributeError):
                    pass
                try:
                    return getattr(value, name)
                except (AttributeError, TypeError):
                    pass
            return 0

        parts = (member("width", "Width"), member("height", "Height"))
    return tuple(float(part) for part in parts)


class Size:
    """Small OpenCvSharp.Size-compatible value object for host properties."""

    def __init__(self, width: float = 0, height: float = 0):
        self.width = float(width)
        self.height = float(height)

    Width = property(
        lambda self: self.width,
        lambda self, value: setattr(self, "width", float(value)),
    )
    Height = property(
        lambda self: self.height,
        lambda self, value: setattr(self, "height", float(value)),
    )

    def __iter__(self):
        yield self.width
        yield self.height

    def __repr__(self) -> str:
        return f"Size({self.width:g}, {self.height:g})"


def _scalar_tuple(value, *, default=(0.0, 0.0, 0.0, 0.0)) -> tuple[float, ...]:
    """Read an OpenCvSharp.Scalar-like value as up to four components."""
    unwrapped = getattr(value, "__wrapped__", None)
    if unwrapped is not None:
        value = unwrapped
    if value is None:
        return tuple(float(item) for item in default)
    if isinstance(value, dict):
        folded = {str(key).casefold(): item for key, item in value.items()}
        values = []
        for index in range(4):
            values.append(folded.get(
                f"val{index}", folded.get(f"item{index}", default[index]),
            ))
    elif isinstance(value, (list, tuple, np.ndarray)):
        values = list(value)[:4]
    else:
        values = []
        for index in range(4):
            found = False
            for name in (
                f"val{index}", f"Val{index}",
                f"item{index}", f"Item{index}",
            ):
                try:
                    values.append(value[name])
                    found = True
                    break
                except (KeyError, TypeError, AttributeError):
                    pass
                try:
                    values.append(getattr(value, name))
                    found = True
                    break
                except (AttributeError, TypeError):
                    pass
            if not found:
                values.append(default[index])
    if len(values) < 4:
        values.extend(default[len(values):4])
    try:
        return tuple(float(item) for item in values[:4])
    except (TypeError, ValueError) as error:
        raise TypeError("Scalar 的四个分量必须是数字") from error


class Scalar:
    """Small OpenCvSharp.Scalar-compatible value object for host properties."""

    def __init__(self, val0: float = 0, val1: float = 0,
                 val2: float = 0, val3: float = 0):
        self.val0 = float(val0)
        self.val1 = float(val1)
        self.val2 = float(val2)
        self.val3 = float(val3)

    Val0 = property(
        lambda self: self.val0,
        lambda self, value: setattr(self, "val0", float(value)),
    )
    Val1 = property(
        lambda self: self.val1,
        lambda self, value: setattr(self, "val1", float(value)),
    )
    Val2 = property(
        lambda self: self.val2,
        lambda self, value: setattr(self, "val2", float(value)),
    )
    Val3 = property(
        lambda self: self.val3,
        lambda self, value: setattr(self, "val3", float(value)),
    )
    Item0 = Val0
    Item1 = Val1
    Item2 = Val2
    Item3 = Val3

    def __iter__(self):
        yield from (self.val0, self.val1, self.val2, self.val3)

    def __getitem__(self, index: int) -> float:
        return tuple(self)[index]

    def __repr__(self) -> str:
        return (
            f"Scalar({self.val0:g}, {self.val1:g}, "
            f"{self.val2:g}, {self.val3:g})"
        )


def _color_conversion_code(value) -> int:
    """Normalize OpenCV color-conversion enum names and numeric values."""
    raw = getattr(value, "value", value)
    raw = getattr(raw, "__wrapped__", raw)
    if isinstance(raw, str):
        normalized = raw.strip()
        if not normalized:
            return int(cv2.COLOR_BGR2RGB)
        name = normalized if normalized.startswith("COLOR_") else f"COLOR_{normalized}"
        code = getattr(cv2, name, None)
        if code is None:
            # OpenCV's Python constants are upper-case while enum names in
            # BetterGI JSON are conventionally PascalCase.
            folded = name.casefold()
            code = next(
                (
                    candidate for candidate in dir(cv2)
                    if candidate.casefold() == folded
                ), None,
            )
            if code is None:
                raise ValueError(f"不支持的颜色转换方式: {value}")
            return int(getattr(cv2, code))
        return int(code)
    try:
        return int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"颜色转换方式必须是 OpenCV 枚举名或数字: {value}") from error


class SearchAnchorMode:
    """BetterGI ``SearchAnchorMode`` enum values exposed to Python and JS."""

    Auto = "Auto"
    TopLeft = "TopLeft"
    TopRight = "TopRight"
    BottomLeft = "BottomLeft"
    BottomRight = "BottomRight"
    Center = "Center"

    @classmethod
    def normalize(cls, value) -> str:
        raw = getattr(value, "value", value)
        text = str(raw)
        for name in ("Auto", "TopLeft", "TopRight", "BottomLeft", "BottomRight", "Center"):
            if text.casefold() == name.casefold():
                return name
        return cls.Auto


class SearchExpandRatio:
    """Four-sided percentage expansion using XAML Thickness ordering."""

    def __init__(self, *values, left=None, top=None, right=None, bottom=None):
        if any(value is not None for value in (left, top, right, bottom)):
            if any(value is None for value in (left, top, right, bottom)):
                raise ValueError("SearchExpandRatio 的四边参数不能缺少")
            values = (left, top, right, bottom)
        elif len(values) == 1 and isinstance(values[0], (list, tuple)):
            values = tuple(values[0])
        if len(values) == 0:
            values = (0, 0, 0, 0)
        elif len(values) == 1:
            values = (values[0],) * 4
        elif len(values) == 2:
            values = (values[0], values[1], values[0], values[1])
        elif len(values) != 4:
            raise ValueError("SearchExpandRatio 需要 1、2 或 4 个数字")
        self.left, self.top, self.right, self.bottom = (
            float(value) for value in values
        )

    Left = property(
        lambda self: self.left,
        lambda self, value: setattr(self, "left", float(value)),
    )
    Top = property(
        lambda self: self.top,
        lambda self, value: setattr(self, "top", float(value)),
    )
    Right = property(
        lambda self: self.right,
        lambda self, value: setattr(self, "right", float(value)),
    )
    Bottom = property(
        lambda self: self.bottom,
        lambda self, value: setattr(self, "bottom", float(value)),
    )

    @property
    def is_valid(self) -> bool:
        return all(
            math.isfinite(value) and value >= 0
            for value in (self.left, self.top, self.right, self.bottom)
        )

    IsValid = property(lambda self: self.is_valid)

    def __iter__(self):
        yield self.left
        yield self.top
        yield self.right
        yield self.bottom

    def __repr__(self) -> str:
        return (
            "SearchExpandRatio("
            f"{self.left:g}, {self.top:g}, {self.right:g}, {self.bottom:g})"
        )


def _expand_ratio(value) -> SearchExpandRatio | None:
    if value is None:
        return None
    if isinstance(value, SearchExpandRatio):
        return value
    unwrapped = getattr(value, "__wrapped__", None)
    if unwrapped is not None:
        value = unwrapped
    if isinstance(value, dict):
        folded = {str(key).casefold(): item for key, item in value.items()}
        values = [
            folded.get("left", 0), folded.get("top", 0),
            folded.get("right", 0), folded.get("bottom", 0),
        ]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = []
        for name in ("left", "top", "right", "bottom"):
            try:
                values.append(value[name])
            except (KeyError, TypeError, AttributeError):
                values.append(getattr(value, name, getattr(value, name.capitalize(), 0)))
    try:
        return SearchExpandRatio(*values)
    except (TypeError, ValueError):
        # Keep an invalid object visible to the resolver so it can safely
        # produce a miss instead of silently broadening the search region.
        return SearchExpandRatio(float("nan"), 0, 0, 0)


class SearchOptions:
    """Reference-canvas search options from BetterGI's recognition model."""

    def __init__(self, anchor_mode: str = SearchAnchorMode.Auto,
                 reference_search_box=None, expand_size=None, expand_percent=None,
                 **kwargs):
        # Python callers and ClearScript callers commonly use the public
        # PascalCase property names in object initializers.
        anchor_mode = kwargs.pop("AnchorMode", kwargs.pop("anchorMode", anchor_mode))
        reference_search_box = kwargs.pop(
            "ReferenceSearchBox", kwargs.pop("referenceSearchBox", reference_search_box),
        )
        expand_size = kwargs.pop("ExpandSize", kwargs.pop("expandSize", expand_size))
        expand_percent = kwargs.pop(
            "ExpandPercent", kwargs.pop("expandPercent", expand_percent),
        )
        if kwargs:
            raise TypeError(f"SearchOptions 不支持字段: {', '.join(kwargs)}")
        self.anchor_mode = SearchAnchorMode.normalize(anchor_mode)
        self.reference_search_box = (
            None if reference_search_box is None else _rect_tuple(reference_search_box)
        )
        self.expand_size = None if expand_size is None else Size(*_size_tuple(expand_size))
        self.expand_percent = _expand_ratio(expand_percent)

    AnchorMode = property(
        lambda self: self.anchor_mode,
        lambda self, value: setattr(self, "anchor_mode", SearchAnchorMode.normalize(value)),
    )
    ReferenceSearchBox = property(
        lambda self: None if self.reference_search_box is None else _rect_result(*self.reference_search_box),
        lambda self, value: setattr(
            self, "reference_search_box",
            None if value is None else _rect_tuple(value),
        ),
    )
    ExpandSize = property(
        lambda self: self.expand_size,
        lambda self, value: setattr(
            self, "expand_size", None if value is None else Size(*_size_tuple(value)),
        ),
    )
    ExpandPercent = property(
        lambda self: self.expand_percent,
        lambda self, value: setattr(self, "expand_percent", _expand_ratio(value)),
    )

    anchorMode = AnchorMode
    referenceSearchBox = ReferenceSearchBox
    expandSize = ExpandSize
    expandPercent = ExpandPercent

    def clone(self) -> "SearchOptions":
        ratio = _expand_ratio(self.expand_percent)
        return SearchOptions(
            anchor_mode=self.anchor_mode,
            reference_search_box=(
                None if self.reference_search_box is None
                else _rect_tuple(self.reference_search_box)
            ),
            expand_size=(
                None if self.expand_size is None
                else Size(*_size_tuple(self.expand_size))
            ),
            expand_percent=ratio,
        )

    Clone = clone


class Mat:
    """OpenCvSharp Mat 的替身：BGR ndarray + 惰性灰度缓存。"""

    def __init__(self, bgr: np.ndarray | None = None):
        self.bgr = (
            bgr if bgr is not None
            else np.empty((0, 0, 3), dtype=np.uint8)
        )
        self._gray: np.ndarray | None = None

    @classmethod
    def from_file(cls, path: str) -> "Mat":
        data = cv2.imread(path, cv2.IMREAD_COLOR)
        if data is None:
            raise FileNotFoundError(f"无法读取图像: {path}")
        return cls(data)

    @property
    def rows(self) -> int:
        return self.bgr.shape[0]

    @property
    def cols(self) -> int:
        return self.bgr.shape[1]

    width = property(lambda self: self.cols)
    height = property(lambda self: self.rows)

    def gray(self) -> np.ndarray:
        if self._gray is None:
            self._gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        return self._gray

    def empty(self) -> bool:
        return self.bgr.size == 0

    def channels(self) -> int:
        if self.bgr.ndim < 3:
            return 1
        return int(self.bgr.shape[2])

    def get(self, _pixel_type, row: int, column: int):
        """Return one pixel using OpenCvSharp's ``Mat.Get<T>`` shape.

        Community scripts pass ``OpenCvSharp.Vec3b`` as the first argument
        and consume the returned value through ``Item0``/``Item1``/``Item2``.
        The type token is intentionally only a compatibility marker: the
        actual channel count comes from the matrix.
        """
        y, x = int(row), int(column)
        if y < 0 or x < 0 or y >= self.rows or x >= self.cols:
            raise IndexError(f"Mat.Get 坐标越界: row={y}, column={x}")
        pixel = self.bgr[y, x]
        if np.isscalar(pixel):
            values = [int(pixel)]
        else:
            values = [int(value) for value in np.asarray(pixel).reshape(-1)]
        return {
            **{f"Item{index}": value for index, value in enumerate(values)},
            **{f"item{index}": value for index, value in enumerate(values)},
        }

    def dispose(self) -> None:
        self.bgr = np.empty((0, 0, 3), dtype=np.uint8)
        self._gray = None

    Dispose = dispose
    Empty = empty
    Channels = channels
    Get = get


class Point2f:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance_to(self, other: "Point2f") -> float:
        return float(np.hypot(self.x - other.x, self.y - other.y))

    distanceTo = distance_to


class RecognitionObject:
    def __init__(self):
        self.recognition_type = "TemplateMatch"
        self.template: Mat | None = None
        self.roi: tuple[float, float, float, float] | None = None  # ref 空间 x,y,w,h
        self.reference_image_size: Size | None = None
        self.reference_bounding_box: tuple[float, float, float, float] | None = None
        self.search_options: SearchOptions | None = None
        self.threshold = 0.8
        self.name = ""
        self.text = ""
        self.use_3_channels = False
        self.template_match_mode = cv2.TM_CCOEFF_NORMED
        self.use_mask = False
        self.mask_color = (0, 255, 0)  # BGR
        self.mask_mat: Mat | None = None
        self.draw_on_window = False
        self.max_match_count = -1
        self.color_conversion_code = int(cv2.COLOR_BGR2RGB)
        self.lower_color = Scalar()
        self.upper_color = Scalar()
        self.match_count = 1
        self.one_contain_match_text: list[str] = []
        self.all_contain_match_text: list[str] = []
        self.regex_match_text: list[str] = []

    @classmethod
    def template_match(cls, mat: Mat, x: float | None = None, y: float | None = None,
                       w: float | None = None, h: float | None = None) -> "RecognitionObject":
        ro = cls()
        ro.recognition_type = "TemplateMatch"
        ro.template = mat
        if isinstance(x, bool):
            ro.use_mask = x
            if y is not None:
                ro.mask_color = cls._parse_mask_color(y)
        elif None not in (x, y, w, h):
            ro.roi = (x, y, w, h)
        return ro.init_template()

    @staticmethod
    def _parse_mask_color(value) -> tuple[int, int, int]:
        value = getattr(value, "__wrapped__", value)
        if isinstance(value, dict):
            folded = {str(key).casefold(): item for key, item in value.items()}
            return (
                int(folded.get("b", folded.get("blue", 0))),
                int(folded.get("g", folded.get("green", 255))),
                int(folded.get("r", folded.get("red", 0))),
            )
        channels = []
        for name in ("B", "G", "R"):
            channel = getattr(value, name, getattr(value, name.lower(), None))
            if channel is None:
                channels = []
                break
            channels.append(int(channel))
        if channels:
            return tuple(channels)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            # BetterGI accepts System.Drawing.Color; array callers generally
            # provide RGB, so convert it to OpenCV's BGR order.
            return int(value[2]), int(value[1]), int(value[0])
        return (0, 255, 0)

    @classmethod
    def ocr(cls, *args) -> "RecognitionObject":
        if len(args) == 1:
            x, y, w, h = _rect_tuple(args[0])
        elif len(args) == 4:
            x, y, w, h = args
        else:
            raise TypeError("RecognitionObject.Ocr 需要 Rect 或 x, y, w, h")
        ro = cls()
        ro.recognition_type = "Ocr"
        ro.roi = (x, y, w, h)
        return ro

    @classmethod
    def ocr_match(cls, x: float, y: float, w: float, h: float, *texts: str) -> "RecognitionObject":
        ro = cls.ocr(x, y, w, h)
        ro.recognition_type = "OcrMatch"
        ro.one_contain_match_text = list(texts)
        return ro

    @classmethod
    def ocr_this(cls) -> "RecognitionObject":
        ro = cls()
        ro.recognition_type = "Ocr"
        return ro

    # ClearScript exposes Pascal/camel properties while the Python model uses
    # snake_case internally. These aliases also survive when a result leaves
    # the outer case-insensitive JS Proxy through another host method.
    recognitionType = property(
        lambda self: self.recognition_type,
        lambda self, value: setattr(self, "recognition_type", str(value)),
    )
    regionOfInterest = property(
        lambda self: _rect_result(*self.roi) if self.roi is not None else None,
        lambda self, value: setattr(
            self, "roi", None if value is None else _rect_tuple(value),
        ),
    )
    referenceImageSize = property(
        lambda self: self.reference_image_size,
        lambda self, value: setattr(
            self, "reference_image_size",
            None if value is None else Size(*_size_tuple(value)),
        ),
    )
    referenceBoundingBox = property(
        lambda self: (
            None if self.reference_bounding_box is None
            else _rect_result(*self.reference_bounding_box)
        ),
        lambda self, value: setattr(
            self, "reference_bounding_box",
            None if value is None else _rect_tuple(value),
        ),
    )
    searchOptions = property(
        lambda self: self.search_options,
        lambda self, value: setattr(
            self, "search_options", getattr(value, "__wrapped__", value),
        ),
    )
    templateImageMat = property(
        lambda self: self.template,
        lambda self, value: setattr(self, "template", value),
    )
    templateImageGreyMat = property(
        lambda self: Mat(self.template.gray()) if self.template is not None else None,
    )
    use3Channels = property(
        lambda self: self.use_3_channels,
        lambda self, value: setattr(self, "use_3_channels", bool(value)),
    )
    templateMatchMode = property(
        lambda self: self.template_match_mode,
        lambda self, value: setattr(self, "template_match_mode", int(value)),
    )
    useMask = property(
        lambda self: self.use_mask,
        lambda self, value: setattr(self, "use_mask", bool(value)),
    )
    maskColor = property(
        lambda self: self.mask_color,
        lambda self, value: setattr(self, "mask_color", self._parse_mask_color(value)),
    )
    maskMat = property(
        lambda self: self.mask_mat,
        lambda self, value: setattr(self, "mask_mat", getattr(value, "__wrapped__", value)),
    )
    drawOnWindow = property(
        lambda self: self.draw_on_window,
        lambda self, value: setattr(self, "draw_on_window", bool(value)),
    )
    maxMatchCount = property(
        lambda self: self.max_match_count,
        lambda self, value: setattr(self, "max_match_count", int(value)),
    )
    colorConversionCode = property(
        lambda self: self.color_conversion_code,
        lambda self, value: setattr(
            self, "color_conversion_code", _color_conversion_code(value),
        ),
    )
    lowerColor = property(
        lambda self: self.lower_color,
        lambda self, value: setattr(
            self, "lower_color", Scalar(*_scalar_tuple(value)),
        ),
    )
    upperColor = property(
        lambda self: self.upper_color,
        lambda self, value: setattr(
            self, "upper_color", Scalar(*_scalar_tuple(value)),
        ),
    )
    matchCount = property(
        lambda self: self.match_count,
        lambda self, value: setattr(self, "match_count", int(value)),
    )
    oneContainMatchText = property(
        lambda self: self.one_contain_match_text,
        lambda self, value: setattr(self, "one_contain_match_text", list(value)),
    )
    allContainMatchText = property(
        lambda self: self.all_contain_match_text,
        lambda self, value: setattr(self, "all_contain_match_text", list(value)),
    )
    regexMatchText = property(
        lambda self: self.regex_match_text,
        lambda self, value: setattr(self, "regex_match_text", list(value)),
    )
    RecognitionType = recognitionType
    RegionOfInterest = regionOfInterest
    ReferenceImageSize = referenceImageSize
    ReferenceBoundingBox = referenceBoundingBox
    SearchOptions = searchOptions
    TemplateImageMat = templateImageMat
    TemplateImageGreyMat = templateImageGreyMat
    Name = property(
        lambda self: self.name,
        lambda self, value: setattr(self, "name", str(value)),
    )
    Text = property(
        lambda self: self.text,
        lambda self, value: setattr(self, "text", str(value)),
    )
    Threshold = property(
        lambda self: self.threshold,
        lambda self, value: setattr(self, "threshold", float(value)),
    )
    Use3Channels = use3Channels
    TemplateMatchMode = templateMatchMode
    UseMask = useMask
    MaskColor = maskColor
    MaskMat = maskMat
    DrawOnWindow = drawOnWindow
    MaxMatchCount = maxMatchCount
    ColorConversionCode = colorConversionCode
    LowerColor = lowerColor
    UpperColor = upperColor
    MatchCount = matchCount
    OneContainMatchText = oneContainMatchText
    AllContainMatchText = allContainMatchText
    RegexMatchText = regexMatchText

    def init_template(self) -> "RecognitionObject":
        if self.use_mask and self.template is not None and self.mask_mat is None:
            color = np.array(self.mask_color, dtype=np.uint8)
            ignored = cv2.inRange(self.template.bgr, color, color)
            self.mask_mat = Mat(cv2.bitwise_not(ignored))
        return self

    initTemplate = init_template
    InitTemplate = init_template

    def clone(self) -> "RecognitionObject":
        # BetterGI shares Mat/list references when cloning RecognitionObject;
        # copy.copy preserves that contract while isolating scalar options.
        cloned = copy.copy(self)
        if self.search_options is not None:
            cloned.search_options = self.search_options.clone()
        return cloned

    Clone = clone

    def dispose(self) -> None:
        """OpenCV resources are Python-owned; retain BetterGI lifecycle API."""

    Dispose = dispose


class Region:
    """已识别（或派生）的矩形。对脚本暴露 ref 坐标，内部持有设备像素矩形。"""

    def __init__(self, ctx: "GameContext", dx: float, dy: float, dw: float, dh: float,
                 text: str = "", score: float = 0.0, empty: bool = False):
        self.ctx = ctx
        self.dx, self.dy, self.dw, self.dh = dx, dy, dw, dh
        self.text = text
        self.score = score
        self._empty = empty

    @property
    def x(self) -> float:
        return self.ctx.transform.to_ref(self.dx, self.dy)[0]

    @x.setter
    def x(self, value: float) -> None:
        self.dx = self.ctx.transform.to_device(float(value), self.y)[0]

    @property
    def y(self) -> float:
        return self.ctx.transform.to_ref(self.dx, self.dy)[1]

    @y.setter
    def y(self, value: float) -> None:
        self.dy = self.ctx.transform.to_device(self.x, float(value))[1]

    @property
    def width(self) -> float:
        return self.dw / self.ctx.transform.scale

    @width.setter
    def width(self, value: float) -> None:
        self.dw = self.ctx.transform.scale_len(float(value))

    @property
    def height(self) -> float:
        return self.dh / self.ctx.transform.scale

    @height.setter
    def height(self, value: float) -> None:
        self.dh = self.ctx.transform.scale_len(float(value))

    def is_empty(self) -> bool:
        return self._empty

    def is_exist(self) -> bool:
        return not self._empty

    def click(self) -> "Region":
        if self._empty:
            raise RuntimeError("对空 Region 调用 click()")
        self.ctx.device.tap(self.dx + self.dw / 2, self.dy + self.dh / 2,
                            image_width=self.ctx.transform.device_width,
                            image_height=self.ctx.transform.device_height)
        return self

    def double_click(self) -> "Region":
        self.click()
        self.ctx.sleep(120)
        self.click()
        return self

    def click_to(self, x: float, y: float, w: float = 0, h: float = 0) -> None:
        ref_x = self.x + float(x) + float(w) / 2
        ref_y = self.y + float(y) + float(h) / 2
        dx, dy = self.ctx.transform.to_device(ref_x, ref_y)
        self.ctx.device.tap(dx, dy,
                            image_width=self.ctx.transform.device_width,
                            image_height=self.ctx.transform.device_height)

    def move(self) -> None:
        """Move the active script's virtual pointer to this region."""
        if self._empty:
            raise RuntimeError("对空 Region 调用 move()")
        pointer = getattr(self.ctx, "_script_pointer", None)
        if pointer is not None:
            pointer.move_to(self.x + self.width / 2, self.y + self.height / 2)

    def move_to(self, x: float, y: float, w: float = 0, h: float = 0) -> None:
        pointer = getattr(self.ctx, "_script_pointer", None)
        if pointer is not None:
            pointer.move_to_ref(
                self.x + float(x) + float(w) / 2,
                self.y + float(y) + float(h) / 2,
            )

    def background_click(self) -> None:
        self.click()

    def derive(self, *args) -> "Region":
        if len(args) == 1:
            x, y, w, h = _rect_tuple(args[0])
        elif len(args) == 2:
            x, y = args
            w = h = 0
        elif len(args) == 4:
            x, y, w, h = args
        else:
            raise TypeError("Region.Derive 需要 Rect、x/y 或 x/y/w/h")
        s = self.ctx.transform.scale
        return Region(self.ctx, self.dx + x * s, self.dy + y * s, w * s, h * s, empty=self._empty)

    def convert_position_to_game_capture_region(
        self, x: float, y: float, w: float | None = None, h: float | None = None,
    ):
        absolute_x = self.x + float(x)
        absolute_y = self.y + float(y)
        if w is None and h is None:
            return {
                "item1": absolute_x, "item2": absolute_y,
                "Item1": absolute_x, "Item2": absolute_y,
            }
        return _rect_result(
            absolute_x, absolute_y, float(w or 0), float(h or 0),
        )

    def convert_self_position_to_game_capture_region(self):
        return self.to_rect()

    def convert_position_to_desktop_region(self, x: float, y: float):
        return self.convert_position_to_game_capture_region(x, y)

    def to_rect(self):
        return _rect_result(self.x, self.y, self.width, self.height)

    def draw_self(self, name: str = "rect", pen=None) -> None:
        """Drawing is optional on the touch console; preserve script flow."""

    def draw_rect(self, *args) -> None:
        """Compatibility no-op until a script requests persistent debug drawing."""

    def draw_line(self, *args) -> None:
        """Compatibility no-op until a script requests persistent debug drawing."""

    def dispose(self) -> None:
        """Region arrays are garbage-collected; expose IDisposable semantics."""

    @classmethod
    def empty_region(cls, ctx: "GameContext") -> "Region":
        return cls(ctx, 0, 0, 0, 0, empty=True)

    isEmpty = is_empty
    IsEmpty = is_empty
    isExist = is_exist
    IsExist = is_exist
    Click = click
    clickTo = click_to
    ClickTo = click_to
    doubleClick = double_click
    DoubleClick = double_click
    Move = move
    moveTo = move_to
    MoveTo = move_to
    backgroundClick = background_click
    BackgroundClick = background_click
    Derive = derive
    convertSelfPositionToGameCaptureRegion = convert_self_position_to_game_capture_region
    ConvertSelfPositionToGameCaptureRegion = convert_self_position_to_game_capture_region
    convertPositionToGameCaptureRegion = convert_position_to_game_capture_region
    ConvertPositionToGameCaptureRegion = convert_position_to_game_capture_region
    convertPositionToDesktopRegion = convert_position_to_desktop_region
    ConvertPositionToDesktopRegion = convert_position_to_desktop_region
    toRect = to_rect
    ToRect = to_rect
    drawSelf = draw_self
    DrawSelf = draw_self
    drawRect = draw_rect
    DrawRect = draw_rect
    drawLine = draw_line
    DrawLine = draw_line
    Dispose = dispose
    X = property(lambda self: self.x, lambda self, value: setattr(self, "x", value))
    Y = property(lambda self: self.y, lambda self, value: setattr(self, "y", value))
    Width = property(
        lambda self: self.width,
        lambda self, value: setattr(self, "width", value),
    )
    Height = property(
        lambda self: self.height,
        lambda self, value: setattr(self, "height", value),
    )
    Text = property(lambda self: self.text, lambda self, value: setattr(self, "text", str(value)))
    top = property(lambda self: self.y, lambda self, value: setattr(self, "y", value))
    Top = top
    bottom = property(lambda self: self.y + self.height)
    Bottom = bottom
    left = property(lambda self: self.x, lambda self, value: setattr(self, "x", value))
    Left = left
    right = property(lambda self: self.x + self.width)
    Right = right
    matchScore = property(
        lambda self: self.score,
        lambda self, value: setattr(self, "score", float(value)),
    )


class DesktopRegion(Region):
    """Touch equivalent of BetterGI's unscaled desktop coordinate region."""

    def __init__(self, ctx: "GameContext", width: float | None = None,
                 height: float | None = None):
        # Script-visible desktop coordinates use the same 1920x1080 reference
        # space as every other BetterGI host region. Device pixels stay internal.
        ref_width = float(width) if width is not None else 1920.0
        ref_height = float(height) if height is not None else 1080.0
        super().__init__(
            ctx, 0, 0,
            ctx.transform.scale_len(ref_width),
            ctx.transform.scale_len(ref_height),
        )

    def desktop_region_click(self, x: float, y: float, width: float = 0,
                             height: float = 0) -> None:
        self.click_to(x, y, width, height)

    def desktop_region_move(self, x: float, y: float, width: float = 0,
                            height: float = 0) -> None:
        pointer = getattr(self.ctx, "_script_pointer", None)
        if pointer is not None:
            pointer.move_to(
                float(x) + float(width) / 2,
                float(y) + float(height) / 2,
            )

    def derive_capture(self, mat: Mat, x: float, y: float) -> "ImageRegion":
        value = getattr(mat, "__wrapped__", mat)
        if not isinstance(value, Mat):
            raise TypeError("DesktopRegion.Derive 需要 Mat")
        dx, dy = self.ctx.transform.to_device(float(x), float(y))
        return ImageRegion(self.ctx, value.bgr, dx, dy)

    def derive(self, *args):
        if args and isinstance(getattr(args[0], "__wrapped__", args[0]), Mat):
            if len(args) != 3:
                raise TypeError("DesktopRegion.Derive(Mat, x, y) 需要三个参数")
            return self.derive_capture(args[0], args[1], args[2])
        return super().derive(*args)

    desktopRegionClick = desktop_region_click
    DesktopRegionClick = desktop_region_click
    desktopRegionMove = desktop_region_move
    DesktopRegionMove = desktop_region_move
    Derive = derive


class ImageRegion(Region):
    def __init__(self, ctx: "GameContext", bgr: np.ndarray, dx: float = 0, dy: float = 0,
                 *, reference_search_allowed: bool = True):
        super().__init__(ctx, dx, dy, bgr.shape[1], bgr.shape[0])
        self.bgr = bgr
        self._reference_search_allowed = bool(reference_search_allowed)

    @property
    def src_mat(self) -> Mat:
        return Mat(self.bgr)

    def _roi_to_device(self, roi) -> tuple[int, int, int, int]:
        h_img, w_img = self.bgr.shape[:2]
        if not roi or roi == (0, 0, 0, 0):
            return 0, 0, w_img, h_img
        t = self.ctx.transform
        x, y = t.to_device(roi[0], roi[1])
        w = t.scale_len(roi[2])
        if roi[0] <= 1 and roi[2] >= REF_WIDTH - 2:  # 全宽 ROI 视为整个屏幕宽
            w = w_img - x
        h = t.scale_len(roi[3])
        x = int(max(0, min(w_img - 1, x)))
        y = int(max(0, min(h_img - 1, y)))
        return x, y, int(min(w, w_img - x)), int(min(h, h_img - y))

    @staticmethod
    def _has_reference_search(ro: RecognitionObject) -> bool:
        roi = ro.roi
        has_explicit_roi = roi is not None and tuple(roi) != (0, 0, 0, 0)
        return (
            not has_explicit_roi
            and ro.reference_image_size is not None
            and ro.reference_bounding_box is not None
        )

    @classmethod
    def _has_partial_reference_search(cls, ro: RecognitionObject) -> bool:
        roi = ro.roi
        has_explicit_roi = roi is not None and tuple(roi) != (0, 0, 0, 0)
        return (
            not has_explicit_roi
            and (
                ro.reference_image_size is not None
                or ro.reference_bounding_box is not None
                or ro.search_options is not None
            )
            and not cls._has_reference_search(ro)
        )

    @staticmethod
    def _round(value: float) -> int:
        # Python's round uses the same ties-to-even rule as Math.Round(double)
        # used by BetterGI's reference-search helper.
        return int(round(value))

    @classmethod
    def _transform_reference_rect(
        cls, rect: tuple[float, float, float, float],
        scale: float, offset_x: float, offset_y: float,
    ) -> tuple[int, int, int, int]:
        x, y, width, height = rect
        left = cls._round(offset_x + x * scale)
        top = cls._round(offset_y + y * scale)
        right = cls._round(offset_x + (x + width) * scale)
        bottom = cls._round(offset_y + (y + height) * scale)
        return left, top, max(1, right - left), max(1, bottom - top)

    @classmethod
    def _resolve_reference_search(
        cls, image: "ImageRegion", ro: RecognitionObject,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int]] | None:
        """Resolve BetterGI reference-canvas search to source-image pixels."""
        if cls._has_partial_reference_search(ro):
            return None
        if not cls._has_reference_search(ro):
            return image._roi_to_device(ro.roi), None
        if not image._reference_search_allowed:
            return None

        ref_w, ref_h = _size_tuple(ro.reference_image_size)
        bbox = _rect_tuple(ro.reference_bounding_box)
        options = ro.search_options or SearchOptions()
        ratio = _expand_ratio(options.expand_percent)
        search_box = (
            None if options.reference_search_box is None
            else _rect_tuple(options.reference_search_box)
        )
        expand_size = (
            None if options.expand_size is None
            else _size_tuple(options.expand_size)
        )
        if (
            ref_w <= 0 or ref_h <= 0
            or bbox[2] <= 0 or bbox[3] <= 0
            or (
                search_box is not None
                and (
                    search_box[2] <= 0 or search_box[3] <= 0
                )
            )
            or (ratio is not None and not ratio.is_valid)
        ):
            return None

        image_h, image_w = image.bgr.shape[:2]
        scale = min(image_w / ref_w, image_h / ref_h)
        if scale <= 0:
            return None

        anchor = SearchAnchorMode.normalize(options.anchor_mode)
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        if anchor == SearchAnchorMode.TopLeft:
            horizontal, vertical = "left", "top"
        elif anchor == SearchAnchorMode.TopRight:
            horizontal, vertical = "right", "top"
        elif anchor == SearchAnchorMode.BottomLeft:
            horizontal, vertical = "left", "bottom"
        elif anchor == SearchAnchorMode.BottomRight:
            horizontal, vertical = "right", "bottom"
        elif anchor == SearchAnchorMode.Center:
            horizontal, vertical = "center", "center"
        else:
            horizontal = (
                "left" if center_x < ref_w * 0.4
                else "right" if center_x > ref_w * 0.6
                else "center"
            )
            vertical = (
                "top" if center_y < ref_h * 0.4
                else "bottom" if center_y > ref_h * 0.6
                else "center"
            )

        scaled_w, scaled_h = ref_w * scale, ref_h * scale
        offset_x = (
            image_w - scaled_w if horizontal == "right"
            else (image_w - scaled_w) / 2 if horizontal == "center"
            else 0
        )
        offset_y = (
            image_h - scaled_h if vertical == "bottom"
            else (image_h - scaled_h) / 2 if vertical == "center"
            else 0
        )

        template_size = (
            max(1, cls._round(bbox[2] * scale)),
            max(1, cls._round(bbox[3] * scale)),
        )
        base = search_box or bbox
        left, top, width, height = cls._transform_reference_rect(
            base, scale, offset_x, offset_y,
        )
        if ratio is not None:
            expand_left = image_w * ratio.left
            expand_top = image_h * ratio.top
            expand_right = image_w * ratio.right
            expand_bottom = image_h * ratio.bottom
        else:
            width, height = expand_size or (10, 10)
            expand_left = expand_right = width
            expand_top = expand_bottom = height

        right = left + width
        bottom = top + height
        left = cls._round(max(0, min(image_w, left - expand_left)))
        top = cls._round(max(0, min(image_h, top - expand_top)))
        right = cls._round(max(0, min(image_w, right + expand_right)))
        bottom = cls._round(max(0, min(image_h, bottom + expand_bottom)))
        effective = (left, top, max(0, right - left), max(0, bottom - top))
        if (
            effective[2] <= 0 or effective[3] <= 0
            or (
                ro.recognition_type == "TemplateMatch"
                and (
                    effective[2] < template_size[0]
                    or effective[3] < template_size[1]
                )
            )
        ):
            return None
        return effective, template_size

    def _resolve_search_region(
        self, ro: RecognitionObject,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int] | None] | None:
        resolved = self._resolve_reference_search(self, ro)
        if resolved is None:
            return None
        roi, template_size = resolved
        return roi, template_size

    @staticmethod
    def _compact_ocr_text(items) -> str:
        return "".join(
            re.sub(r"\s+", "", str(item.text))
            for item in items
        )

    @staticmethod
    def _ocr_match_text(ro: RecognitionObject, text: str) -> bool:
        return (
            all(value in text for value in ro.all_contain_match_text)
            and (
                not ro.one_contain_match_text
                or any(value in text for value in ro.one_contain_match_text)
            )
            and all(re.search(pattern, text) for pattern in ro.regex_match_text)
        )

    @staticmethod
    def _color_bounds(ro: RecognitionObject, channels: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Adapt OpenCvSharp's four-component Scalar to the converted image."""
        if channels <= 0:
            raise ValueError("颜色识别得到的图像通道数无效")
        lower = _scalar_tuple(ro.lower_color)
        upper = _scalar_tuple(ro.upper_color)
        if channels > len(lower):
            lower += (0.0,) * (channels - len(lower))
            upper += (0.0,) * (channels - len(upper))
        return lower[:channels], upper[:channels]

    @classmethod
    def _color_mask(cls, crop: np.ndarray, ro: RecognitionObject) -> np.ndarray:
        """Convert a BGR crop and retain pixels inside the configured range."""
        source = crop
        code = int(ro.color_conversion_code)
        # BetterGI receives BGR captures. BGRA2BGR is its sentinel for
        # "already BGR" in the color-range path, so do not reject it because
        # the portable capture has no alpha channel.
        if code != int(getattr(cv2, "COLOR_BGRA2BGR", 1)):
            try:
                source = cv2.cvtColor(source, code)
            except cv2.error as error:
                raise ValueError(
                    f"颜色识别无法执行 OpenCV 转换 {code}: {error}"
                ) from error
        channels = 1 if source.ndim == 2 else int(source.shape[2])
        lower, upper = cls._color_bounds(ro, channels)
        try:
            return cv2.inRange(source, lower, upper)
        except cv2.error as error:
            raise ValueError(
                f"颜色识别上下界与图像通道不匹配: {lower}..{upper}"
            ) from error

    def _ocr_source(self, crop: np.ndarray, ro: RecognitionObject) -> np.ndarray:
        if ro.recognition_type == "ColorRangeAndOcr":
            return self._color_mask(crop, ro)
        return crop

    def find(self, ro: RecognitionObject, success_action=None,
             fail_action=None) -> Region:
        ro = getattr(ro, "__wrapped__", ro)
        if not isinstance(ro, RecognitionObject):
            raise TypeError("find/findMulti 需要 RecognitionObject")

        if ro.recognition_type in ("Ocr", "OcrMatch", "ColorRangeAndOcr"):
            resolved = self._resolve_search_region(ro)
            if resolved is None:
                if callable(fail_action):
                    fail_action()
                return Region.empty_region(self.ctx)
            (cx, cy, cw, ch), _template_size = resolved
            if cw <= 0 or ch <= 0:
                if callable(fail_action):
                    fail_action()
                return Region.empty_region(self.ctx)
            crop = self.bgr[cy:cy + ch, cx:cx + cw]
            items = get_ocr().recognize(self._ocr_source(crop, ro))
            text = self._compact_ocr_text(items)
            matched = bool(text)
            if ro.recognition_type == "OcrMatch":
                if not (ro.one_contain_match_text or ro.all_contain_match_text
                        or ro.regex_match_text):
                    raise ValueError("OcrMatch 的匹配文本不能全为空")
                matched = matched and self._ocr_match_text(ro, text)
            if matched:
                if (
                    ro.roi and ro.roi != (0, 0, 0, 0)
                ) or self._has_reference_search(ro):
                    result = Region(
                        self.ctx, self.dx + cx, self.dy + cy, cw, ch, text=text,
                    )
                else:
                    self.text = text
                    result = self
                if callable(success_action):
                    success_action(result)
                return result
            if callable(fail_action):
                fail_action()
            return Region.empty_region(self.ctx)

        results = self.find_multi(ro, limit=1)
        result = results[0] if results else Region.empty_region(self.ctx)
        if results:
            if callable(success_action):
                success_action(result)
        elif callable(fail_action):
            fail_action()
        return result

    def find_multi(self, ro: RecognitionObject, success_action=None,
                   fail_action=None, *, limit: int = 10) -> list[Region]:
        ro = getattr(ro, "__wrapped__", ro)
        if not isinstance(ro, RecognitionObject):
            raise TypeError("find/findMulti 需要 RecognitionObject")
        resolved = self._resolve_search_region(ro)
        if resolved is None:
            if callable(fail_action):
                fail_action()
            return []
        (cx, cy, cw, ch), reference_template_size = resolved
        if cw <= 0 or ch <= 0:
            if callable(fail_action):
                fail_action()
            return []
        crop = self.bgr[cy:cy + ch, cx:cx + cw]
        t = self.ctx.transform

        if ro.recognition_type == "TemplateMatch":
            template = getattr(ro.template, "__wrapped__", ro.template)
            if template is None:
                raise ValueError("TemplateMatch 缺少模板图像")
            if not isinstance(template, Mat):
                raise TypeError("TemplateMatch 模板必须为 Mat")
            ro.init_template()
            tpl = template.bgr if ro.use_3_channels else template.gray()
            th, tw = tpl.shape[:2]
            if reference_template_size is not None:
                target_width, target_height = reference_template_size
            else:
                target_width = max(1, round(tw * t.scale))
                target_height = max(1, round(th * t.scale))
            tpl = cv2.resize(
                tpl, (target_width, target_height),
                interpolation=cv2.INTER_AREA if target_width < tw or target_height < th
                else cv2.INTER_LINEAR,
            )
            source = crop if ro.use_3_channels else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if source.shape[0] < tpl.shape[0] or source.shape[1] < tpl.shape[1]:
                results = []
                if callable(fail_action):
                    fail_action()
                return results
            mask = None
            mask_mat = getattr(ro.mask_mat, "__wrapped__", ro.mask_mat)
            if ro.use_mask and isinstance(mask_mat, Mat) and not mask_mat.empty():
                mask = mask_mat.bgr
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                mask = cv2.resize(
                    mask, (tpl.shape[1], tpl.shape[0]), interpolation=cv2.INTER_NEAREST,
                )
            method = int(ro.template_match_mode)
            res = cv2.matchTemplate(source, tpl, method, mask=mask)
            if method in (cv2.TM_SQDIFF, cv2.TM_CCORR, cv2.TM_CCOEFF):
                res = cv2.normalize(res, None, 0, 1, cv2.NORM_MINMAX)
            work = 1.0 - res if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED) else res
            work = np.nan_to_num(work, nan=-1.0, posinf=-1.0, neginf=-1.0)
            out: list[Region] = []
            match_limit = max(1, int(limit))
            if ro.max_match_count > 0:
                match_limit = min(match_limit, ro.max_match_count)
            for _ in range(match_limit):
                _, max_val, _, max_loc = cv2.minMaxLoc(work)
                if max_val < ro.threshold:
                    break
                mx, my = max_loc
                out.append(Region(self.ctx, self.dx + cx + mx, self.dy + cy + my,
                                  tpl.shape[1], tpl.shape[0], score=float(max_val)))
                x0 = max(0, mx - tpl.shape[1] // 2)
                y0 = max(0, my - tpl.shape[0] // 2)
                work[y0:my + tpl.shape[0] // 2 + 1, x0:mx + tpl.shape[1] // 2 + 1] = -1.0
            if out:
                if callable(success_action):
                    success_action(out)
            elif callable(fail_action):
                fail_action()
            return out

        if ro.recognition_type == "ColorMatch":
            if limit <= 0:
                if callable(fail_action):
                    fail_action()
                return []
            mask = self._color_mask(crop, ro)
            match_count = int(cv2.countNonZero(mask))
            if match_count < int(ro.match_count):
                if callable(fail_action):
                    fail_action()
                return []
            result = Region(
                self.ctx, self.dx + cx, self.dy + cy, cw, ch,
                score=float(match_count),
            )
            results = [result]
            if callable(success_action):
                success_action(results)
            return results

        if ro.recognition_type in ("Ocr", "OcrMatch", "ColorRangeAndOcr"):
            items = get_ocr().recognize(self._ocr_source(crop, ro))
            if (ro.recognition_type == "OcrMatch" or ro.one_contain_match_text
                    or ro.all_contain_match_text or ro.regex_match_text):
                def keep(it) -> bool:
                    if ro.one_contain_match_text and not any(s in it.text for s in ro.one_contain_match_text):
                        return False
                    if ro.all_contain_match_text and not all(s in it.text for s in ro.all_contain_match_text):
                        return False
                    if ro.regex_match_text and not any(re.search(rx, it.text) for rx in ro.regex_match_text):
                        return False
                    return True
                items = [it for it in items if keep(it)]
            results = [
                Region(self.ctx, self.dx + cx + it.x, self.dy + cy + it.y,
                       it.width, it.height, text=it.text, score=it.confidence)
                for it in items[:limit]
            ]
            if results:
                if callable(success_action):
                    success_action(results)
            elif callable(fail_action):
                fail_action()
            return results

        raise NotImplementedError(f"识别类型 {ro.recognition_type} 暂未支持")

    def derive_crop(self, *args) -> "ImageRegion":
        if len(args) == 1:
            x, y, w, h = _rect_tuple(args[0])
        elif len(args) == 4:
            x, y, w, h = args
        else:
            raise TypeError("ImageRegion.DeriveCrop 需要 Rect 或 x, y, w, h")
        s = self.ctx.transform.scale
        x0, y0 = int(round(float(x) * s)), int(round(float(y) * s))
        x1 = max(0, min(self.bgr.shape[1], x0 + int(round(float(w) * s))))
        y1 = max(0, min(self.bgr.shape[0], y0 + int(round(float(h) * s))))
        x0 = max(0, min(self.bgr.shape[1], x0))
        y0 = max(0, min(self.bgr.shape[0], y0))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"DeriveCrop 裁剪区域无效: ({x}, {y}, {w}, {h})")
        crop = self.bgr[y0:y1, x0:x1]
        return ImageRegion(
            self.ctx, crop, self.dx + x0, self.dy + y0,
            reference_search_allowed=False,
        )

    def derive_to_1080p(self) -> "ImageRegion":
        return self  # 本移植版坐标已通过 ScreenTransform 归一化

    def ocr_text(self) -> str:
        return " ".join(it.text for it in get_ocr().recognize(self.bgr))

    srcMat = property(lambda self: self.src_mat)
    SrcMat = property(lambda self: self.src_mat)
    Find = find
    findMulti = find_multi
    FindMulti = find_multi
    deriveCrop = derive_crop
    DeriveCrop = derive_crop
    deriveTo1080P = derive_to_1080p
    DeriveTo1080P = derive_to_1080p
    ocrText = ocr_text
    OcrText = ocr_text
