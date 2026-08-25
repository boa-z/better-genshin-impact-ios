"""BetterGI 识别模型的移植：Mat / RecognitionObject / Region / ImageRegion。

- 脚本可见坐标一律为 1920x1080 参考空间；内部保留设备像素矩形供 click() 使用。
- 模板资产按 1080p 基准制作，匹配前按设备比例缩放。
- 全部同步，与原版 ClearScript 宿主语义一致。
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
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


def _replacement_dictionary(value) -> dict[str, list[str]]:
    """Normalize BetterGI's OCR replacement dictionary for script callers."""
    value = getattr(value, "__wrapped__", value)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("ReplaceDictionary 必须是对象或字典")
    result: dict[str, list[str]] = {}
    for target, replacements in value.items():
        replacements = getattr(replacements, "__wrapped__", replacements)
        if isinstance(replacements, str):
            replacements = [replacements]
        elif not isinstance(replacements, (list, tuple)):
            raise TypeError("ReplaceDictionary 的值必须是字符串数组")
        result[str(target)] = [str(item) for item in replacements]
    return result


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


def _native_array_value(value):
    """Remove PythonMonkey's array protocol shim before NumPy conversion."""
    unwrapped = getattr(value, "__wrapped__", None)
    if unwrapped is not None:
        value = unwrapped
    if isinstance(value, np.ndarray) or isinstance(value, (bytes, bytearray, memoryview)):
        return value
    if isinstance(value, (list, tuple)):
        return [_native_array_value(item) for item in value]
    return value


class Mat:
    """OpenCvSharp Mat 的替身：BGR ndarray + 惰性灰度缓存。"""

    _CV_DEPTH_DTYPES = {
        0: np.uint8,
        1: np.int8,
        2: np.uint16,
        3: np.int16,
        4: np.int32,
        5: np.float32,
        6: np.float64,
    }

    def __init__(self, bgr: np.ndarray | None = None):
        self.bgr = np.asarray(
            bgr if bgr is not None else np.empty((0, 0, 3), dtype=np.uint8)
        )
        self._gray: np.ndarray | None = None
        self._disposed = False

    @classmethod
    def from_file(cls, path: str) -> "Mat":
        data = cv2.imread(path, cv2.IMREAD_COLOR)
        if data is None:
            raise FileNotFoundError(f"无法读取图像: {path}")
        return cls(data)

    @classmethod
    def _from_cv_type(cls, rows: int, cols: int, type_code: int,
                      fill=None) -> "Mat":
        """Create an array from OpenCV's packed depth/channel type code."""
        code = int(type_code)
        depth = code & 7
        channels = (code >> 3) + 1
        dtype = cls._CV_DEPTH_DTYPES.get(depth)
        if dtype is None:
            raise ValueError(f"不支持的 OpenCV Mat 深度: {depth}")
        shape = (int(rows), int(cols))
        if channels > 1:
            shape += (channels,)
        array = np.full(shape, 0 if fill is None else fill, dtype=dtype)
        return cls(array)

    @staticmethod
    def _size_rows_cols(size) -> tuple[int, int]:
        width, height = _size_tuple(size)
        return int(round(height)), int(round(width))

    @classmethod
    def from_array(cls, array) -> "Mat":
        value = _native_array_value(array)
        if isinstance(value, Mat):
            return value.clone()
        return cls(np.asarray(value).copy())

    @classmethod
    def from_pixel_data(cls, width: int, height: int, *args) -> "Mat":
        """Build a Mat from a packed pixel buffer.

        OpenCvSharp exposes several ``FromPixelData`` overloads.  JS scripts
        generally pass either ``(width, height, data)`` or
        ``(width, height, type, data[, step])``.  The portable host accepts
        bytes, typed-array-like lists, and already shaped nested arrays.
        """
        if len(args) == 1:
            type_code, data, step = None, args[0], None
        elif len(args) in (2, 3):
            type_code, data = args[:2]
            step = args[2] if len(args) == 3 else None
        else:
            raise TypeError(
                "Mat.FromPixelData 需要 width、height、data 或 type、data[, step]"
            )
        rows, cols = int(height), int(width)
        if rows < 0 or cols < 0:
            raise ValueError("Mat 尺寸不能为负数")
        value = _native_array_value(data)
        if isinstance(value, Mat):
            return value.clone()
        if isinstance(value, dict) and "data" in value:
            value = value["data"]
        array = np.asarray(value)
        if type_code is None:
            if array.ndim >= 2 and array.shape[:2] == (rows, cols):
                return cls(array.copy())
            raw = np.frombuffer(value, dtype=np.uint8) if isinstance(
                value, (bytes, bytearray, memoryview)
            ) else np.asarray(value).reshape(-1)
            pixels = rows * cols
            if pixels == 0:
                return cls(np.empty((rows, cols), dtype=raw.dtype))
            if raw.size % pixels:
                raise ValueError("像素数据长度与 Mat 尺寸不匹配")
            channels = raw.size // pixels
            if channels not in (1, 2, 3, 4):
                raise ValueError(f"无法从像素数据推断通道数: {channels}")
            shape = (rows, cols) if channels == 1 else (rows, cols, channels)
            return cls(raw.reshape(shape).copy())

        code = int(type_code)
        depth = code & 7
        channels = (code >> 3) + 1
        dtype = cls._CV_DEPTH_DTYPES.get(depth)
        if dtype is None:
            raise ValueError(f"不支持的 OpenCV Mat 深度: {depth}")
        bytes_per_pixel = channels * np.dtype(dtype).itemsize
        row_step = int(step) if step is not None else cols * bytes_per_pixel
        if row_step < cols * bytes_per_pixel:
            raise ValueError("Mat 行步长小于一行像素所需字节数")
        if isinstance(value, (bytes, bytearray, memoryview)):
            raw = np.frombuffer(value, dtype=np.uint8)
            needed = row_step * rows
            if raw.size < needed:
                raise ValueError("像素数据长度小于 Mat 行步长要求")
            rows_data = raw[:needed].reshape(rows, row_step)
            packed = rows_data[:, :cols * bytes_per_pixel].reshape(-1)
            typed = packed.view(dtype)
        else:
            typed = np.asarray(value, dtype=dtype).reshape(-1)
            needed_values = rows * cols * channels
            if typed.size < needed_values:
                raise ValueError("像素数据长度小于 Mat 尺寸要求")
            typed = typed[:needed_values]
        shape = (rows, cols) if channels == 1 else (rows, cols, channels)
        return cls(typed.reshape(shape).copy())

    @classmethod
    def im_decode(cls, data, flags=None) -> "Mat":
        value = _native_array_value(data)
        if isinstance(value, Mat):
            return value.clone()
        if isinstance(value, str):
            encoded = np.fromfile(value, dtype=np.uint8)
        elif isinstance(value, (bytes, bytearray, memoryview)):
            encoded = np.frombuffer(value, dtype=np.uint8)
        else:
            encoded = np.asarray(value, dtype=np.uint8).reshape(-1)
        mode = cv2.IMREAD_COLOR if flags is None else int(flags)
        image = cv2.imdecode(encoded, mode)
        if image is None:
            raise ValueError("Mat.ImDecode 无法解码图像数据")
        return cls(image)

    @classmethod
    def from_image_data(cls, image_data, flags=None) -> "Mat":
        value = _native_array_value(image_data)
        if isinstance(value, Mat):
            return value.clone()
        if isinstance(value, dict):
            width = value.get("width", value.get("Width"))
            height = value.get("height", value.get("Height"))
            data = value.get("data", value.get("Data"))
        else:
            width = getattr(value, "width", getattr(value, "Width", None))
            height = getattr(value, "height", getattr(value, "Height", None))
            data = getattr(value, "data", getattr(value, "Data", value))
        if width is None or height is None:
            return cls.from_array(data)
        raw = np.asarray(data, dtype=np.uint8)
        channels = raw.size // (int(width) * int(height)) if width and height else 0
        mat = cls.from_pixel_data(int(width), int(height), raw)
        if channels == 4 and (flags is None or int(flags) != cv2.IMREAD_UNCHANGED):
            mat = cls(cv2.cvtColor(mat.bgr, cv2.COLOR_RGBA2BGR))
        return mat

    @staticmethod
    def _factory_dimensions(*args) -> tuple[int, int, int]:
        if len(args) == 1:
            rows, columns = Mat._size_rows_cols(args[0])
            return rows, columns, 0
        if len(args) == 2:
            rows, columns = Mat._size_rows_cols(args[0]) if not isinstance(
                args[0], (int, float, np.integer, np.floating)
            ) else (int(args[0]), int(args[0]))
            return rows, columns, int(args[1])
        if len(args) == 3:
            return int(args[0]), int(args[1]), int(args[2])
        raise TypeError("Mat 工厂需要 Size/type 或 rows/cols/type")

    @staticmethod
    def zeros(*args) -> "Mat":
        rows, columns, type_value = Mat._factory_dimensions(*args)
        return Mat._from_cv_type(rows, columns, type_value)

    @staticmethod
    def ones(*args) -> "Mat":
        result = Mat.zeros(*args)
        result.bgr[...] = 1
        return result

    @staticmethod
    def eye(*args) -> "Mat":
        rows, columns, type_value = Mat._factory_dimensions(*args)
        result = Mat._from_cv_type(rows, columns, type_value)
        if result.channels() == 1:
            np.fill_diagonal(result.bgr, 1)
        else:
            for channel in range(result.channels()):
                np.fill_diagonal(result.bgr[:, :, channel], 1)
        return result

    @staticmethod
    def im_decode_bytes(data, flags=None) -> "Mat":
        return Mat.im_decode(data, flags)

    @property
    def rows(self) -> int:
        return int(self.bgr.shape[0]) if self.bgr.ndim else 0

    @property
    def cols(self) -> int:
        if self.bgr.ndim < 2:
            return 1 if self.bgr.ndim else 0
        return int(self.bgr.shape[1])

    @property
    def dims(self) -> int:
        return int(self.bgr.ndim)

    @property
    def flags(self) -> int:
        return int(self.bgr.flags.c_contiguous)

    @property
    def data(self):
        self.throw_if_disposed()
        return self.bgr.tobytes()

    @property
    def data_pointer(self):
        return int(self.bgr.__array_interface__["data"][0]) if self.bgr.size else 0

    @property
    def data_start(self):
        return self.data_pointer

    @property
    def data_end(self):
        return self.data_pointer + int(self.bgr.nbytes)

    @property
    def data_limit(self):
        return self.data_end

    @property
    def cv_ptr(self):
        return self.data_pointer

    @property
    def is_disposed(self) -> bool:
        return self._disposed

    @property
    def is_enabled_dispose(self) -> bool:
        return True

    width = property(lambda self: self.cols)
    height = property(lambda self: self.rows)
    Rows = property(lambda self: self.rows)
    Cols = property(lambda self: self.cols)
    Width = property(lambda self: self.width)
    Height = property(lambda self: self.height)

    def gray(self) -> np.ndarray:
        self.throw_if_disposed()
        if self._gray is None:
            if self.bgr.ndim == 2:
                self._gray = self.bgr
            elif self.bgr.ndim == 3 and self.bgr.shape[2] == 1:
                self._gray = self.bgr[:, :, 0]
            else:
                self._gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        return self._gray

    def empty(self) -> bool:
        return self.bgr.size == 0

    def channels(self) -> int:
        if self.bgr.ndim < 3:
            return 1
        return int(self.bgr.shape[2])

    def total(self, start_dim: int = 0, end_dim: int | None = None) -> int:
        self.throw_if_disposed()
        start = max(0, int(start_dim))
        end = self.bgr.ndim if end_dim is None else min(self.bgr.ndim, int(end_dim) + 1)
        return int(np.prod(self.bgr.shape[start:end], dtype=np.int64))

    def size(self, dim: int | None = None):
        self.throw_if_disposed()
        if dim is not None:
            return int(self.bgr.shape[int(dim)])
        return Size(self.cols, self.rows)

    def step(self, dim: int | None = None) -> int:
        self.throw_if_disposed()
        if dim is None:
            return int(self.bgr.strides[0]) if self.bgr.ndim else 0
        return int(self.bgr.strides[int(dim)])

    def step1(self, dim: int = 0) -> int:
        return int(self.step(dim) // max(1, self.bgr.dtype.itemsize))

    def elem_size1(self) -> int:
        return int(self.bgr.dtype.itemsize)

    def elem_size(self) -> int:
        return int(self.bgr.dtype.itemsize * self.channels())

    def depth(self) -> int:
        dtype = np.dtype(self.bgr.dtype)
        for depth, candidate in self._CV_DEPTH_DTYPES.items():
            if dtype == np.dtype(candidate):
                return depth
        return 0

    def type(self) -> int:
        return int(self.depth() + ((self.channels() - 1) << 3))

    def is_continuous(self) -> bool:
        return bool(self.bgr.flags.c_contiguous)

    def is_submatrix(self) -> bool:
        return not bool(self.bgr.flags.owndata)

    def throw_if_disposed(self) -> None:
        if self._disposed:
            raise RuntimeError("Mat 已释放")

    @staticmethod
    def _pixel_result(pixel):
        values = []
        for value in np.asarray(pixel).reshape(-1):
            values.append(
                int(value) if np.issubdtype(np.asarray(value).dtype, np.integer)
                else float(value)
            )
        return {
            **{f"Item{index}": value for index, value in enumerate(values)},
            **{f"item{index}": value for index, value in enumerate(values)},
        }

    def get(self, *args):
        """Return one pixel using OpenCvSharp's ``Mat.Get<T>`` shape.

        Community scripts pass ``OpenCvSharp.Vec3b`` as the first argument
        and consume the returned value through ``Item0``/``Item1``/``Item2``.
        The type token is intentionally only a compatibility marker.  The
        one- and two-index forms mirror OpenCvSharp's direct indexer access.
        """
        self.throw_if_disposed()
        if len(args) == 3 and not isinstance(args[0], (int, float)):
            _pixel_type, row, column = args
            location = (int(row), int(column))
        elif len(args) == 2:
            location = tuple(int(value) for value in args)
        elif len(args) == 1:
            index = int(args[0])
            location = (index,) if self.bgr.ndim < 2 else divmod(index, self.cols)
        elif len(args) == 3:
            location = tuple(int(value) for value in args)
        else:
            raise TypeError("Mat.Get 需要一个、两个或三个索引参数")
        pixel = self.bgr[location]
        if np.asarray(pixel).ndim == 0:
            value = np.asarray(pixel).item()
            return int(value) if isinstance(value, (int, np.integer)) else float(value)
        return self._pixel_result(pixel)

    def set(self, *args) -> None:
        """Set a pixel using one-, two- or three-index overloads."""
        self.throw_if_disposed()
        if len(args) == 2:
            index, value = args
            location = (int(index),) if self.bgr.ndim < 2 else divmod(int(index), self.cols)
        elif len(args) == 3:
            row, column, value = args
            location = (int(row), int(column))
        elif len(args) == 4:
            i0, i1, i2, value = args
            location = (int(i0), int(i1), int(i2))
        else:
            raise TypeError("Mat.Set 需要索引和一个值")
        if isinstance(value, Mapping):
            folded = {str(key).casefold(): item for key, item in value.items()}
            values = [folded.get(f"item{index}", folded.get(f"val{index}", 0))
                      for index in range(self.channels())]
        elif isinstance(value, (list, tuple, np.ndarray)):
            values = list(np.asarray(value).reshape(-1))
        else:
            values = [value]
        target = self.bgr[location]
        if np.asarray(target).ndim == 0:
            self.bgr[location] = values[0]
        else:
            channels = int(np.asarray(target).size)
            values = (values + [values[-1]])[:channels]
            self.bgr[location] = np.asarray(values, dtype=self.bgr.dtype)
        self._gray = None

    def at(self, *args):
        return self.get(*args)

    def item(self, *args):
        return self.get(*args)

    def get_array(self, _index=None):
        self.throw_if_disposed()
        return self.bgr.tolist()

    def get_rectangular_array(self, _index=None):
        return self.get_array(_index)

    def set_array(self, data) -> None:
        self.bgr = np.asarray(data).copy()
        self._gray = None
        self._disposed = False

    def set_rectangular_array(self, data) -> None:
        self.set_array(data)

    def clone(self, roi=None) -> "Mat":
        self.throw_if_disposed()
        if roi is None:
            return Mat(self.bgr.copy())
        x, y, width, height = _rect_tuple(roi)
        x, y, width, height = map(int, (x, y, width, height))
        return Mat(self.bgr[y:y + height, x:x + width].copy())

    def copy_to(self, dst: "Mat", mask=None) -> None:
        self.throw_if_disposed()
        dst = getattr(dst, "__wrapped__", dst)
        if not isinstance(dst, Mat):
            raise TypeError("Mat.CopyTo 需要 Mat 目标")
        mask = getattr(mask, "__wrapped__", mask)
        if isinstance(mask, Mat) and not mask.empty():
            selection = mask.bgr if mask.bgr.ndim == 2 else mask.bgr[:, :, 0]
            dst.bgr = np.where(selection[..., None] > 0, self.bgr, dst.bgr)
        else:
            dst.bgr = self.bgr.copy()
        dst._gray = None
        dst._disposed = False

    def set_to(self, value, mask=None) -> "Mat":
        self.throw_if_disposed()
        if isinstance(value, Mapping):
            folded = {str(key).casefold(): item for key, item in value.items()}
            values = [folded.get(f"val{index}", folded.get(f"item{index}", 0))
                      for index in range(max(1, self.channels()))]
        elif isinstance(value, (list, tuple, np.ndarray)):
            values = list(np.asarray(value).reshape(-1))
        else:
            values = [value]
        fill = np.asarray(values, dtype=self.bgr.dtype)
        if self.bgr.ndim == 3 and fill.size > 1:
            fill = fill[:self.bgr.shape[2]]
        else:
            fill = fill.flat[0]
        mask = getattr(mask, "__wrapped__", mask)
        if isinstance(mask, Mat) and not mask.empty():
            selection = mask.bgr if mask.bgr.ndim == 2 else mask.bgr[:, :, 0]
            self.bgr[selection > 0] = fill
        else:
            self.bgr[...] = fill
        self._gray = None
        return self

    def _binary(self, other, operation: str) -> "Mat":
        self.throw_if_disposed()
        value = getattr(other, "__wrapped__", other)
        operand = value.bgr if isinstance(value, Mat) else value
        operations = {
            "add": cv2.add, "subtract": cv2.subtract,
            "multiply": cv2.multiply, "divide": cv2.divide,
            "and": cv2.bitwise_and, "or": cv2.bitwise_or,
            "xor": cv2.bitwise_xor,
        }
        try:
            result = operations[operation](self.bgr, operand)
        except cv2.error as error:
            raise ValueError(f"Mat.{operation} 操作失败: {error}") from error
        return Mat(result)

    def _compare(self, other, operation: str) -> "Mat":
        self.throw_if_disposed()
        value = getattr(other, "__wrapped__", other)
        operand = value.bgr if isinstance(value, Mat) else value
        operations = {
            "lt": np.less, "le": np.less_equal,
            "ne": np.not_equal, "gt": np.greater,
            "ge": np.greater_equal,
        }
        result = operations[operation](self.bgr, operand)
        return Mat(np.asarray(result, dtype=np.uint8) * 255)

    def less_than(self, other): return self._compare(other, "lt")
    def less_than_or_equal(self, other): return self._compare(other, "le")
    def not_equals(self, other): return self._compare(other, "ne")
    def greater_than(self, other): return self._compare(other, "gt")
    def greater_than_or_equal(self, other): return self._compare(other, "ge")

    def col(self, x: int) -> "Mat":
        return Mat(self.bgr[:, int(x):int(x) + 1].copy())

    def row(self, y: int) -> "Mat":
        return Mat(self.bgr[int(y):int(y) + 1].copy())

    @staticmethod
    def _range_tuple(value) -> tuple[int, int]:
        value = getattr(value, "__wrapped__", value)
        if isinstance(value, (list, tuple)):
            return int(value[0]), int(value[1])
        start = getattr(value, "start", getattr(value, "Start", 0))
        end = getattr(value, "end", getattr(value, "End", 0))
        return int(start), int(end)

    def col_range(self, *args) -> "Mat":
        if len(args) == 1:
            start, end = self._range_tuple(args[0])
        elif len(args) == 2:
            start, end = map(int, args)
        else:
            raise TypeError("Mat.ColRange 需要 Range 或 start/end")
        return Mat(self.bgr[:, start:end].copy())

    def row_range(self, *args) -> "Mat":
        if len(args) == 1:
            start, end = self._range_tuple(args[0])
        elif len(args) == 2:
            start, end = map(int, args)
        else:
            raise TypeError("Mat.RowRange 需要 Range 或 start/end")
        return Mat(self.bgr[start:end].copy())

    def sub_mat(self, *args) -> "Mat":
        if len(args) == 1:
            x, y, width, height = _rect_tuple(args[0])
            return self.clone((x, y, width, height))
        if len(args) == 2:
            row_start, row_end = self._range_tuple(args[0])
            col_start, col_end = self._range_tuple(args[1])
            return Mat(self.bgr[row_start:row_end, col_start:col_end].copy())
        if len(args) == 4:
            row_start, row_end, col_start, col_end = map(int, args)
            return Mat(self.bgr[row_start:row_end, col_start:col_end].copy())
        raise TypeError("Mat.SubMat 需要 Rect、两个 Range 或四个索引")

    def diag(self_or_mat, diagonal: int = 0) -> "Mat":
        """Support both ``Mat.Diag(mat)`` and ``mat.Diag()`` forms."""
        value = getattr(self_or_mat, "__wrapped__", self_or_mat)
        if not isinstance(value, Mat):
            raise TypeError("Mat.Diag 需要 Mat")
        return Mat(np.diag(value.bgr, k=int(diagonal)))

    def convert_to(self, dst: "Mat", rtype: int, alpha: float = 1,
                   beta: float = 0) -> None:
        self.throw_if_disposed()
        dst = getattr(dst, "__wrapped__", dst)
        if not isinstance(dst, Mat):
            raise TypeError("Mat.ConvertTo 需要 Mat 目标")
        depth = int(rtype) & 7
        dtype = self._CV_DEPTH_DTYPES.get(depth)
        if dtype is None:
            raise ValueError(f"不支持的目标深度: {depth}")
        converted = self.bgr.astype(np.float64) * float(alpha) + float(beta)
        if np.issubdtype(np.dtype(dtype), np.integer):
            info = np.iinfo(dtype)
            converted = np.clip(np.rint(converted), info.min, info.max)
        dst.bgr = converted.astype(dtype)
        dst._gray = None
        dst._disposed = False

    def assign_to(self, dst: "Mat", type_code: int = -1) -> None:
        if int(type_code) < 0:
            self.copy_to(dst)
            return
        self.convert_to(dst, int(type_code))

    def reshape(self, cn: int, rows: int = 0) -> "Mat":
        self.throw_if_disposed()
        channels = int(cn) or self.channels()
        total_values = self.bgr.size
        target_rows = int(rows) or self.rows
        if target_rows <= 0 or total_values % (target_rows * channels):
            raise ValueError("Mat.Reshape 的尺寸与元素数量不匹配")
        target_cols = total_values // (target_rows * channels)
        shape = (target_rows, target_cols) if channels == 1 else (target_rows, target_cols, channels)
        return Mat(self.bgr.reshape(shape).copy())

    def t(self) -> "Mat":
        self.throw_if_disposed()
        if self.bgr.ndim == 2:
            return Mat(self.bgr.T.copy())
        return Mat(np.transpose(self.bgr, (1, 0, *range(2, self.bgr.ndim))).copy())

    def inv(self, method: int = cv2.DECOMP_LU) -> "Mat":
        return Mat(cv2.invert(self.bgr, int(method))[1])

    def mul(self, other, scale: float = 1) -> "Mat":
        value = getattr(other, "__wrapped__", other)
        operand = value.bgr if isinstance(value, Mat) else value
        return Mat(cv2.multiply(self.bgr, operand, scale=float(scale)))

    def cross(self, other: "Mat") -> "Mat":
        value = getattr(other, "__wrapped__", other)
        return Mat(np.cross(self.bgr, value.bgr if isinstance(value, Mat) else value))

    def dot(self, other: "Mat") -> float:
        value = getattr(other, "__wrapped__", other)
        operand = value.bgr if isinstance(value, Mat) else value
        return float(np.dot(self.bgr.reshape(-1), np.asarray(operand).reshape(-1)))

    def create(self, *args) -> None:
        if len(args) == 2:
            rows, cols = self._size_rows_cols(args[0])
            type_code = args[1]
        elif len(args) == 3:
            rows, cols, type_code = map(int, args)
        else:
            raise TypeError("Mat.Create 需要 Size/type 或 rows/cols/type")
        replacement = self._from_cv_type(rows, cols, int(type_code))
        self.bgr, self._gray, self._disposed = replacement.bgr, None, False

    def abs(self) -> "Mat":
        return Mat(np.abs(self.bgr))

    def sum(self) -> Scalar:
        values = np.asarray(self.bgr, dtype=np.float64)
        if values.ndim >= 3:
            channels = [float(values[:, :, index].sum()) for index in range(values.shape[2])]
        else:
            channels = [float(values.sum())]
        return Scalar(*(channels + [0.0] * (4 - len(channels))))

    def count_non_zero(self) -> int:
        source = self.bgr if self.bgr.ndim == 2 else cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        return int(cv2.countNonZero(source))

    def find_non_zero(self) -> "Mat":
        source = self.bgr if self.bgr.ndim == 2 else cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        points = cv2.findNonZero(source)
        return Mat(np.empty((0, 1, 2), dtype=np.int32) if points is None else points)

    def mean(self, mask=None) -> Scalar:
        value = getattr(mask, "__wrapped__", mask)
        mask_array = None if value is None else value.bgr if isinstance(value, Mat) else value
        channels = cv2.mean(self.bgr, mask=mask_array)
        return Scalar(*channels)

    def split(self) -> list["Mat"]:
        if self.bgr.ndim == 2:
            return [self.clone()]
        return [Mat(channel.copy()) for channel in cv2.split(self.bgr)]

    def extract_channel(self, coi: int) -> "Mat":
        return Mat(cv2.extractChannel(self.bgr, int(coi)))

    def insert_channel(self, channel: "Mat", coi: int) -> None:
        value = getattr(channel, "__wrapped__", channel)
        if not isinstance(value, Mat):
            raise TypeError("Mat.InsertChannel 需要 Mat 通道")
        cv2.insertChannel(value.bgr, self.bgr, int(coi))
        self._gray = None

    def flip(self, flip_code: int) -> "Mat":
        return Mat(cv2.flip(self.bgr, int(flip_code)))

    def repeat(self, ny: int, nx: int) -> "Mat":
        return Mat(cv2.repeat(self.bgr, int(ny), int(nx)))

    def in_range(self, lower, upper) -> "Mat":
        return Mat(cv2.inRange(self.bgr, _scalar_tuple(lower), _scalar_tuple(upper)))

    def sqrt(self) -> "Mat": return Mat(np.sqrt(self.bgr.astype(np.float64)))
    def pow(self, power: float) -> "Mat": return Mat(np.power(self.bgr, float(power)))
    def exp(self) -> "Mat": return Mat(np.exp(self.bgr.astype(np.float64)))
    def log(self) -> "Mat": return Mat(np.log(np.maximum(self.bgr, 1e-12)))

    def threshold(self, thresh: float, maxval: float, threshold_type: int) -> "Mat":
        source = self.gray()
        if source.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
            source = source.astype(np.float32)
        _, result = cv2.threshold(source, float(thresh), float(maxval), int(threshold_type))
        return Mat(result)

    def adaptive_threshold(self, max_value: float, adaptive_method: int,
                           threshold_type: int, block_size: int, c: float) -> "Mat":
        result = cv2.adaptiveThreshold(
            self.gray(), float(max_value), int(adaptive_method),
            int(threshold_type), int(block_size), float(c),
        )
        return Mat(result)

    def cvt_color(self, code: int, dst_cn: int = 0) -> "Mat":
        kwargs = {} if not dst_cn else {"dstCn": int(dst_cn)}
        return Mat(cv2.cvtColor(self.bgr, int(code), **kwargs))

    def match_template(self, template: "Mat", method: int, mask=None) -> "Mat":
        value = getattr(template, "__wrapped__", template)
        if not isinstance(value, Mat):
            raise TypeError("Mat.MatchTemplate 需要 Mat 模板")
        mask_value = getattr(mask, "__wrapped__", mask)
        mask_array = mask_value.bgr if isinstance(mask_value, Mat) else mask_value
        return Mat(cv2.matchTemplate(self.bgr, value.bgr, int(method), mask=mask_array))

    def add(self, other): return self._binary(other, "add")
    def subtract(self, other): return self._binary(other, "subtract")
    def multiply(self, other): return self._binary(other, "multiply")
    def divide(self, other): return self._binary(other, "divide")
    def bitwise_and(self, other): return self._binary(other, "and")
    def bitwise_or(self, other): return self._binary(other, "or")
    def xor(self, other): return self._binary(other, "xor")
    def plus(self): return self.clone()
    def negate(self): return Mat(cv2.multiply(self.bgr, -1))
    def ones_complement(self): return Mat(cv2.bitwise_not(self.bgr))

    def dispose(self) -> None:
        self.bgr = np.empty((0, 0, 3), dtype=np.uint8)
        self._gray = None
        self._disposed = True

    # OpenCvSharp exposes both camelCase and PascalCase names.  Keep the
    # aliases on the Python object as well as in the JS constructor so values
    # returned from another host method remain script-compatible.
    Dims = property(lambda self: self.dims)
    Flags = property(lambda self: self.flags)
    Data = property(lambda self: self.data)
    DataPointer = property(lambda self: self.data_pointer)
    DataStart = property(lambda self: self.data_start)
    DataEnd = property(lambda self: self.data_end)
    DataLimit = property(lambda self: self.data_limit)
    CvPtr = property(lambda self: self.cv_ptr)
    IsDisposed = property(lambda self: self.is_disposed)
    IsEnabledDispose = property(lambda self: self.is_enabled_dispose)
    Total = total
    Size = size
    Step = step
    Step1 = step1
    ElemSize = elem_size
    ElemSize1 = elem_size1
    Type = type
    Depth = depth
    IsContinuous = is_continuous
    IsSubmatrix = is_submatrix
    ThrowIfDisposed = throw_if_disposed
    Dispose = dispose
    Release = dispose
    Empty = empty
    Channels = channels
    Get = get
    At = at
    Set = set
    GetArray = get_array
    GetRectangularArray = get_rectangular_array
    SetArray = set_array
    SetRectangularArray = set_rectangular_array
    Item = item
    Clone = clone
    CopyTo = copy_to
    SetTo = set_to
    Add = add
    Subtract = subtract
    Multiply = multiply
    Divide = divide
    BitwiseAnd = bitwise_and
    BitwiseOr = bitwise_or
    Xor = xor
    Plus = plus
    Negate = negate
    OnesComplement = ones_complement
    LessThan = less_than
    LessThanOrEqual = less_than_or_equal
    NotEquals = not_equals
    GreaterThan = greater_than
    GreaterThanOrEqual = greater_than_or_equal
    Col = col
    ColRange = col_range
    Row = row
    RowRange = row_range
    SubMat = sub_mat
    Diag = diag
    ConvertTo = convert_to
    AssignTo = assign_to
    Reshape = reshape
    T = t
    Inv = inv
    Mul = mul
    Cross = cross
    Dot = dot
    Create = create
    Abs = abs
    Sum = sum
    CountNonZero = count_non_zero
    FindNonZero = find_non_zero
    Mean = mean
    Split = split
    ExtractChannel = extract_channel
    InsertChannel = insert_channel
    Flip = flip
    Repeat = repeat
    InRange = in_range
    Sqrt = sqrt
    Pow = pow
    Exp = exp
    Log = log
    Threshold = threshold
    AdaptiveThreshold = adaptive_threshold
    CvtColor = cvt_color
    MatchTemplate = match_template

    # Static OpenCvSharp factories.  They are assigned explicitly instead of
    # relying on the case-insensitive JS proxy for class properties.
    FromArray = from_array
    FromPixelData = from_pixel_data
    ImDecode = im_decode
    FromImageData = from_image_data
    Zeros = zeros
    Ones = ones
    Eye = eye


def _point_tuple(value) -> tuple[float, float]:
    """Read an OpenCvSharp point/vector-like value.

    BetterGI scripts pass ``Point2f`` values back through PythonMonkey, but
    ``FromPoint``/``FromVec2f`` are also commonly used with plain JS objects
    and OpenCvSharp tuple values.  Keeping this conversion in one place makes
    the arithmetic methods behave consistently for all three forms.
    """
    value = getattr(value, "__wrapped__", value)
    if isinstance(value, Mapping):
        folded = {str(key).casefold(): item for key, item in value.items()}
        x = folded.get("x", folded.get("item0", 0))
        y = folded.get("y", folded.get("item1", 0))
    elif isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        x, y = value[0], value[1]
    else:
        def member(*names):
            for name in names:
                try:
                    return value[name]
                except (KeyError, TypeError, AttributeError):
                    pass
                try:
                    return getattr(value, name)
                except (AttributeError, TypeError):
                    pass
            return 0

        x = member("x", "X", "item0", "Item0")
        y = member("y", "Y", "item1", "Item1")
    try:
        return float(x), float(y)
    except (TypeError, ValueError) as error:
        raise TypeError("点坐标必须是数字") from error


class Point2f:
    """OpenCvSharp.Point2f 的脚本兼容值对象。"""

    def __init__(self, x: float = 0, y: float = 0):
        self.x = float(x)
        self.y = float(y)

    X = property(
        lambda self: self.x,
        lambda self, value: setattr(self, "x", float(value)),
    )
    Y = property(
        lambda self: self.y,
        lambda self, value: setattr(self, "y", float(value)),
    )

    def to_point(self) -> dict[str, int]:
        """Return the integer OpenCvSharp.Point-shaped value."""
        return {
            "x": int(round(self.x)), "y": int(round(self.y)),
            "X": int(round(self.x)), "Y": int(round(self.y)),
        }

    def to_vec2f(self) -> dict[str, float]:
        """Return the OpenCvSharp.Vec2f-shaped value."""
        return {
            "item0": self.x, "item1": self.y,
            "Item0": self.x, "Item1": self.y,
        }

    def plus(self) -> "Point2f":
        return Point2f(self.x, self.y)

    def negate(self) -> "Point2f":
        return Point2f(-self.x, -self.y)

    def add(self, other) -> "Point2f":
        x, y = _point_tuple(other)
        return Point2f(self.x + x, self.y + y)

    def subtract(self, other) -> "Point2f":
        x, y = _point_tuple(other)
        return Point2f(self.x - x, self.y - y)

    def multiply(self, scalar: float) -> "Point2f":
        return Point2f(self.x * float(scalar), self.y * float(scalar))

    def distance_to(self, other) -> float:
        x, y = _point_tuple(other)
        return float(np.hypot(self.x - x, self.y - y))

    def dot_product(self, *args) -> float:
        if len(args) == 1:
            x, y = _point_tuple(args[0])
        elif len(args) == 2:
            x, y = _point_tuple(args[1])
        else:
            raise TypeError("Point2f.DotProduct 需要一个或两个点参数")
        if len(args) == 2:
            x0, y0 = _point_tuple(args[0])
            return float(x0 * x + y0 * y)
        return float(self.x * x + self.y * y)

    def cross_product(self, *args) -> float:
        if len(args) == 1:
            x, y = _point_tuple(args[0])
            return float(self.x * y - self.y * x)
        if len(args) == 2:
            x0, y0 = _point_tuple(args[0])
            x1, y1 = _point_tuple(args[1])
            return float(x0 * y1 - y0 * x1)
        raise TypeError("Point2f.CrossProduct 需要一个或两个点参数")

    def deconstruct(self, *_output) -> list[float]:
        # PythonMonkey cannot mutate C#-style out parameters.  Returning the
        # tuple as an array preserves the normal JS destructuring form.
        return [self.x, self.y]

    @staticmethod
    def from_point(point) -> "Point2f":
        return Point2f(*_point_tuple(point))

    @staticmethod
    def from_vec2f(vec) -> "Point2f":
        return Point2f(*_point_tuple(vec))

    @staticmethod
    def distance(p1, p2) -> float:
        x0, y0 = _point_tuple(p1)
        x1, y1 = _point_tuple(p2)
        return float(np.hypot(x0 - x1, y0 - y1))

    @staticmethod
    def dot_product_static(*args) -> float:
        if len(args) == 1:
            x, y = _point_tuple(args[0])
            return float(x * x + y * y)
        if len(args) == 2:
            x0, y0 = _point_tuple(args[0])
            x1, y1 = _point_tuple(args[1])
            return float(x0 * x1 + y0 * y1)
        raise TypeError("Point2f.DotProduct 需要一个或两个点参数")

    @staticmethod
    def cross_product_static(*args) -> float:
        if len(args) == 1:
            x, y = _point_tuple(args[0])
            return float(x * y)
        if len(args) == 2:
            x0, y0 = _point_tuple(args[0])
            x1, y1 = _point_tuple(args[1])
            return float(x0 * y1 - y0 * x1)
        raise TypeError("Point2f.CrossProduct 需要一个或两个点参数")

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self) -> str:
        return f"Point2f({self.x:g}, {self.y:g})"

    ToPoint = to_point
    ToVec2f = to_vec2f
    Plus = plus
    Negate = negate
    Add = add
    Subtract = subtract
    Multiply = multiply
    DistanceTo = distance_to
    DotProduct = dot_product
    CrossProduct = cross_product
    Deconstruct = deconstruct
    FromPoint = from_point
    FromVec2f = from_vec2f
    Distance = distance


class RecognitionObject:
    def __init__(self):
        # ``RecognitionTypes.None`` is the default value of BetterGI's C#
        # enum.  A script normally assigns the type immediately after using
        # ``new RecognitionObject()``, but preserving the default matters for
        # feature-detection code and for objects used only as draw settings.
        self.recognition_type = "None"
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
        self.draw_on_window_pen = None
        self.max_match_count = -1
        self.use_binary_match = False
        self.binary_threshold = 128
        self.color_conversion_code = int(cv2.COLOR_BGR2RGB)
        self.lower_color = Scalar()
        self.upper_color = Scalar()
        self.match_count = 1
        self.ocr_engine = "Paddle"
        self.replace_dictionary: dict[str, list[str]] = {}
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
    drawOnWindowPen = property(
        lambda self: self.draw_on_window_pen,
        lambda self, value: setattr(self, "draw_on_window_pen", value),
    )
    maxMatchCount = property(
        lambda self: self.max_match_count,
        lambda self, value: setattr(self, "max_match_count", int(value)),
    )
    useBinaryMatch = property(
        lambda self: self.use_binary_match,
        lambda self, value: setattr(self, "use_binary_match", bool(value)),
    )
    binaryThreshold = property(
        lambda self: self.binary_threshold,
        lambda self, value: setattr(self, "binary_threshold", int(value)),
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
    ocrEngine = property(
        lambda self: self.ocr_engine,
        lambda self, value: setattr(self, "ocr_engine", str(value)),
    )
    replaceDictionary = property(
        lambda self: self.replace_dictionary,
        lambda self, value: setattr(
            self, "replace_dictionary", _replacement_dictionary(value),
        ),
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
    DrawOnWindowPen = drawOnWindowPen
    MaxMatchCount = maxMatchCount
    UseBinaryMatch = useBinaryMatch
    BinaryThreshold = binaryThreshold
    ColorConversionCode = colorConversionCode
    LowerColor = lowerColor
    UpperColor = upperColor
    MatchCount = matchCount
    OcrEngine = ocrEngine
    ReplaceDictionary = replaceDictionary
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


class _Drawable:
    """Serializable drawing descriptor used by the touch runtime.

    BetterGI normally hands these objects to a desktop overlay.  DeviceHub
    has no window overlay, but scripts still construct and pass drawables
    while debugging recognition.  Keeping the geometry as a plain host value
    preserves that flow without asking for a new screenshot or mutating the
    trigger frame.
    """

    def __init__(self, kind: str, name: str, pen=None, **geometry):
        self.kind = str(kind)
        self.name = str(name)
        self.pen = pen
        for key, value in geometry.items():
            setattr(self, key, float(value))

    Kind = property(lambda self: self.kind)
    Name = property(lambda self: self.name)
    Pen = property(lambda self: self.pen)
    X = property(lambda self: getattr(self, "x", 0.0))
    Y = property(lambda self: getattr(self, "y", 0.0))
    Width = property(lambda self: getattr(self, "width", 0.0))
    Height = property(lambda self: getattr(self, "height", 0.0))
    X1 = property(lambda self: getattr(self, "x1", 0.0))
    Y1 = property(lambda self: getattr(self, "y1", 0.0))
    X2 = property(lambda self: getattr(self, "x2", 0.0))
    Y2 = property(lambda self: getattr(self, "y2", 0.0))

    def to_dict(self) -> dict:
        result = {
            "kind": self.kind,
            "Kind": self.kind,
            "name": self.name,
            "Name": self.name,
            "pen": self.pen,
            "Pen": self.pen,
        }
        for key, value in vars(self).items():
            if key not in {"kind", "name", "pen"}:
                result[key] = value
                result[key[:1].upper() + key[1:]] = value
        return result


class Region:
    """已识别（或派生）的矩形。对脚本暴露 ref 坐标，内部持有设备像素矩形。"""

    def __init__(self, ctx: "GameContext", dx: float, dy: float, dw: float, dh: float,
                 text: str = "", score: float = 0.0, empty: bool = False,
                 prev: "Region | None" = None):
        self.ctx = ctx
        self.dx, self.dy, self.dw, self.dh = dx, dy, dw, dh
        self.text = text
        self.score = score
        self._empty = empty
        self._prev = prev

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
        return Region(
            self.ctx, self.dx + x * s, self.dy + y * s, w * s, h * s,
            empty=self._empty, prev=self,
        )

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

    def to_image_region(self) -> "ImageRegion":
        """Materialize a result region without taking another screenshot."""
        if isinstance(self, ImageRegion):
            return self
        source = self._prev
        while source is not None and not isinstance(source, ImageRegion):
            source = source._prev
        if source is not None:
            return source.derive_crop(
                self.x - source.x, self.y - source.y,
                self.width, self.height,
            )
        return ImageRegion(self.ctx, self.ctx.capture_bgr()).derive_crop(
            self.x, self.y, self.width, self.height,
        )

    def self_to_rect_drawable(self, name: str = "rect", pen=None) -> _Drawable:
        return _Drawable(
            "rect", name, pen,
            x=self.x, y=self.y, width=self.width, height=self.height,
        )

    def to_rect_drawable(self, *args) -> _Drawable:
        if len(args) in (1, 2, 3):
            rect = _rect_tuple(args[0])
            name = args[1] if len(args) >= 2 else "rect"
            pen = args[2] if len(args) >= 3 else None
            x, y, width, height = rect
        elif len(args) in (4, 5, 6):
            x, y, width, height = map(float, args[:4])
            name = args[4] if len(args) >= 5 else "rect"
            pen = args[5] if len(args) >= 6 else None
        else:
            raise TypeError("Region.ToRectDrawable 需要 Rect 或 x/y/w/h")
        return _Drawable("rect", name, pen, x=x, y=y, width=width, height=height)

    def to_line_drawable(self, x1: float, y1: float, x2: float, y2: float,
                         name: str = "line", pen=None) -> _Drawable:
        return _Drawable(
            "line", name, pen,
            x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
        )

    def draw_self(self, name: str = "rect", pen=None) -> _Drawable:
        return self.self_to_rect_drawable(name, pen)

    def draw_rect(self, *args) -> _Drawable:
        return self.to_rect_drawable(*args)

    def draw_line(self, *args) -> _Drawable:
        return self.to_line_drawable(*args)

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
    toImageRegion = to_image_region
    ToImageRegion = to_image_region
    toRect = to_rect
    ToRect = to_rect
    drawSelf = draw_self
    DrawSelf = draw_self
    drawRect = draw_rect
    DrawRect = draw_rect
    drawLine = draw_line
    DrawLine = draw_line
    selfToRectDrawable = self_to_rect_drawable
    SelfToRectDrawable = self_to_rect_drawable
    toRectDrawable = to_rect_drawable
    ToRectDrawable = to_rect_drawable
    toLineDrawable = to_line_drawable
    ToLineDrawable = to_line_drawable
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
    prev = property(lambda self: self._prev)
    Prev = prev
    prevConverter = property(lambda self: None)
    PrevConverter = prevConverter
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
        return GameCaptureRegion(self.ctx, value.bgr, dx, dy)

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
        self._src_mat: Mat | None = None
        self._cache_grey_mat: Mat | None = None

    @property
    def src_mat(self) -> Mat:
        if self._src_mat is None:
            self._src_mat = Mat(self.bgr)
        return self._src_mat

    @property
    def cache_grey_mat(self) -> Mat:
        if self._cache_grey_mat is None:
            self._cache_grey_mat = Mat(self.src_mat.gray())
        return self._cache_grey_mat

    @property
    def cache_image(self) -> Mat:
        """Portable image backing used by BetterGI's detector API."""
        return self.src_mat

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
    def _apply_text_replacements(ro: RecognitionObject, text: str) -> str:
        for target, replacements in ro.replace_dictionary.items():
            for replacement in replacements:
                text = text.replace(replacement, target)
        return text

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

        # Match BetterGI's empty enum value: an unconfigured recognition
        # object is a non-match, not an accidental template lookup with no
        # image.  This also keeps optional script probes side-effect free.
        if ro.recognition_type == "None":
            if callable(fail_action):
                fail_action()
            return Region.empty_region(self.ctx)

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
            text = self._apply_text_replacements(
                ro, self._compact_ocr_text(items),
            )
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
                        prev=self,
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
        if ro.recognition_type == "None":
            if callable(fail_action):
                fail_action()
            return []
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
            if ro.use_binary_match and not ro.use_3_channels:
                _, source = cv2.threshold(
                    source, int(ro.binary_threshold), 255, cv2.THRESH_BINARY,
                )
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
                out.append(Region(
                    self.ctx, self.dx + cx + mx, self.dy + cy + my,
                    tpl.shape[1], tpl.shape[0], score=float(max_val), prev=self,
                ))
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
                score=float(match_count), prev=self,
            )
            results = [result]
            if callable(success_action):
                success_action(results)
            return results

        if ro.recognition_type in ("Ocr", "OcrMatch", "ColorRangeAndOcr"):
            items = get_ocr().recognize(self._ocr_source(crop, ro))
            if (ro.recognition_type == "OcrMatch" or ro.one_contain_match_text
                    or ro.all_contain_match_text or ro.regex_match_text):
                def normalized(it) -> str:
                    return self._apply_text_replacements(ro, str(it.text))

                def keep(it) -> bool:
                    text = normalized(it)
                    if ro.one_contain_match_text and not any(s in text for s in ro.one_contain_match_text):
                        return False
                    if ro.all_contain_match_text and not all(s in text for s in ro.all_contain_match_text):
                        return False
                    if ro.regex_match_text and not any(re.search(rx, text) for rx in ro.regex_match_text):
                        return False
                    return True
                items = [it for it in items if keep(it)]
            results = [
                Region(self.ctx, self.dx + cx + it.x, self.dy + cy + it.y,
                       it.width, it.height,
                       text=self._apply_text_replacements(ro, str(it.text)),
                       score=it.confidence, prev=self)
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
        return type(self)(
            self.ctx, crop, self.dx + x0, self.dy + y0,
            reference_search_allowed=False,
        )

    def derive_to_1080p(self) -> "ImageRegion":
        return self  # 本移植版坐标已通过 ScreenTransform 归一化

    def ocr_text(self) -> str:
        return " ".join(it.text for it in get_ocr().recognize(self.bgr))

    srcMat = property(lambda self: self.src_mat)
    SrcMat = property(lambda self: self.src_mat)
    cacheGreyMat = property(lambda self: self.cache_grey_mat)
    CacheGreyMat = property(lambda self: self.cache_grey_mat)
    cacheImage = property(lambda self: self.cache_image)
    CacheImage = property(lambda self: self.cache_image)
    Find = find
    findMulti = find_multi
    FindMulti = find_multi
    deriveCrop = derive_crop
    DeriveCrop = derive_crop
    deriveTo1080P = derive_to_1080p
    DeriveTo1080P = derive_to_1080p
    ocrText = ocr_text
    OcrText = ocr_text


class GameCaptureRegion(ImageRegion):
    """ImageRegion with BetterGI's game-coordinate drawable helpers."""

    def convert_to_rect_drawable(self, x: float, y: float, width: float,
                                 height: float, pen=None,
                                 name: str = "rect") -> _Drawable:
        return _Drawable(
            "rect", name, pen,
            x=self.x + float(x), y=self.y + float(y),
            width=float(width), height=float(height),
        )

    def convert_to_line_drawable(self, x1: float, y1: float, x2: float, y2: float,
                                 pen=None, name: str = "line") -> _Drawable:
        return _Drawable(
            "line", name, pen,
            x1=self.x + float(x1), y1=self.y + float(y1),
            x2=self.x + float(x2), y2=self.y + float(y2),
        )

    def derive_to_1080p(self) -> "ImageRegion":
        # All script-visible coordinates are already normalized to BetterGI's
        # 1920x1080 reference canvas by ScreenTransform.
        return self

    convertToRectDrawable = convert_to_rect_drawable
    ConvertToRectDrawable = convert_to_rect_drawable
    convertToLineDrawable = convert_to_line_drawable
    ConvertToLineDrawable = convert_to_line_drawable
    deriveTo1080P = derive_to_1080p
    DeriveTo1080P = derive_to_1080p
