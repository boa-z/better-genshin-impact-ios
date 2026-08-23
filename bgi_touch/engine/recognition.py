"""BetterGI 识别模型的移植：Mat / RecognitionObject / Region / ImageRegion。

- 脚本可见坐标一律为 1920x1080 参考空间；内部保留设备像素矩形供 click() 使用。
- 模板资产按 1080p 基准制作，匹配前按设备比例缩放。
- 全部同步，与原版 ClearScript 宿主语义一致。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from ..vision.coordinate import REF_WIDTH, ScreenTransform
from ..vision.ocr import get_ocr

if TYPE_CHECKING:
    from .context import GameContext


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

    def dispose(self) -> None:
        self.bgr = np.empty((0, 0, 3), dtype=np.uint8)
        self._gray = None

    Dispose = dispose


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
        self.threshold = 0.8
        self.name = ""
        self.one_contain_match_text: list[str] = []
        self.all_contain_match_text: list[str] = []
        self.regex_match_text: list[str] = []

    @classmethod
    def template_match(cls, mat: Mat, x: float | None = None, y: float | None = None,
                       w: float | None = None, h: float | None = None) -> "RecognitionObject":
        ro = cls()
        ro.recognition_type = "TemplateMatch"
        ro.template = mat
        if None not in (x, y, w, h):
            ro.roi = (x, y, w, h)
        return ro

    @classmethod
    def ocr(cls, x: float, y: float, w: float, h: float) -> "RecognitionObject":
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
        lambda self: self.roi,
        lambda self, value: setattr(self, "roi", value),
    )
    templateImageMat = property(
        lambda self: self.template,
        lambda self, value: setattr(self, "template", value),
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

    def init_template(self) -> "RecognitionObject":
        return self

    initTemplate = init_template

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
        """原版移动鼠标指针；触控端无指针，空操作。"""

    def background_click(self) -> None:
        self.click()

    def derive(self, x: float, y: float, w: float = 0, h: float = 0) -> "Region":
        s = self.ctx.transform.scale
        return Region(self.ctx, self.dx + x * s, self.dy + y * s, w * s, h * s, empty=self._empty)

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
    backgroundClick = background_click
    BackgroundClick = background_click
    Derive = derive
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


class ImageRegion(Region):
    def __init__(self, ctx: "GameContext", bgr: np.ndarray, dx: float = 0, dy: float = 0):
        super().__init__(ctx, dx, dy, bgr.shape[1], bgr.shape[0])
        self.bgr = bgr

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

    def find(self, ro: RecognitionObject) -> Region:
        ro = getattr(ro, "__wrapped__", ro)
        results = self.find_multi(ro, limit=1)
        return results[0] if results else Region.empty_region(self.ctx)

    def find_multi(self, ro: RecognitionObject, limit: int = 10) -> list[Region]:
        ro = getattr(ro, "__wrapped__", ro)
        if not isinstance(ro, RecognitionObject):
            raise TypeError("find/findMulti 需要 RecognitionObject")
        cx, cy, cw, ch = self._roi_to_device(ro.roi)
        crop = self.bgr[cy:cy + ch, cx:cx + cw]
        t = self.ctx.transform

        if ro.recognition_type == "TemplateMatch":
            template = getattr(ro.template, "__wrapped__", ro.template)
            if template is None:
                raise ValueError("TemplateMatch 缺少模板图像")
            if not isinstance(template, Mat):
                raise TypeError("TemplateMatch 模板必须为 Mat")
            tpl = template.gray()
            th, tw = tpl.shape[:2]
            tpl = cv2.resize(tpl, (max(1, round(tw * t.scale)), max(1, round(th * t.scale))),
                             interpolation=cv2.INTER_AREA if t.scale < 1 else cv2.INTER_LINEAR)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if gray.shape[0] < tpl.shape[0] or gray.shape[1] < tpl.shape[1]:
                return []
            res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
            out: list[Region] = []
            work = res.copy()
            for _ in range(max(1, limit)):
                _, max_val, _, max_loc = cv2.minMaxLoc(work)
                if max_val < ro.threshold:
                    break
                mx, my = max_loc
                out.append(Region(self.ctx, self.dx + cx + mx, self.dy + cy + my,
                                  tpl.shape[1], tpl.shape[0], score=float(max_val)))
                x0 = max(0, mx - tpl.shape[1] // 2)
                y0 = max(0, my - tpl.shape[0] // 2)
                work[y0:my + tpl.shape[0] // 2 + 1, x0:mx + tpl.shape[1] // 2 + 1] = -1.0
            return out

        if ro.recognition_type in ("Ocr", "OcrMatch"):
            items = get_ocr().recognize(crop)
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
            return [Region(self.ctx, self.dx + cx + it.x, self.dy + cy + it.y,
                           it.width, it.height, text=it.text, score=it.confidence)
                    for it in items[:limit]]

        raise NotImplementedError(f"识别类型 {ro.recognition_type} 暂未支持")

    def derive_crop(self, x: float, y: float, w: float, h: float) -> "ImageRegion":
        s = self.ctx.transform.scale
        x0, y0 = int(x * s), int(y * s)
        crop = self.bgr[y0:y0 + int(h * s), x0:x0 + int(w * s)]
        return ImageRegion(self.ctx, crop, self.dx + x0, self.dy + y0)

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
