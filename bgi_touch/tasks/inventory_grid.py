"""Inventory grid scanning and BetterGI-compatible item utilities.

BetterGI normally identifies inventory icons with an ONNX embedding model.
Those runtime model assets are not shipped in its source tree, so the iOS port
uses the same grid-card detector and reads the selected item's detail name via
OCR.  It is slower, but preserves the task result contract without depending
on Windows-only packaged assets.
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import cv2
import numpy as np

from ..engine.context import GameContext
from ..engine.genshin_api import GenshinApi
from ..engine.recognition import Mat, RecognitionObject
from ..vision.ocr import get_ocr
from .common_jobs import exclusive_realtime_triggers


ASSETS = Path(__file__).resolve().parents[2] / "assets" / "templates" / "inventory"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class InventoryCategory:
    name: str
    asset_prefix: str
    roi: tuple[int, int, int, int]
    columns: int = 8


_DEFAULT_ROI = (106, 110, 1171, 845)
_CATEGORIES = {
    "Weapons": InventoryCategory("Weapons", "weapon", _DEFAULT_ROI),
    "Artifacts": InventoryCategory("Artifacts", "artifact", (106, 162, 1171, 783)),
    "CharacterDevelopmentItems": InventoryCategory(
        "CharacterDevelopmentItems", "characterdevelopmentitem", _DEFAULT_ROI
    ),
    "Food": InventoryCategory("Food", "food", _DEFAULT_ROI),
    "Materials": InventoryCategory("Materials", "material", _DEFAULT_ROI),
    "Gadget": InventoryCategory("Gadget", "gadget", _DEFAULT_ROI),
    "Quest": InventoryCategory("Quest", "quest", _DEFAULT_ROI),
    "PreciousItems": InventoryCategory("PreciousItems", "preciousitem", _DEFAULT_ROI),
    "Furnishings": InventoryCategory("Furnishings", "furnishing", _DEFAULT_ROI),
}

_CATEGORY_ALIASES = {
    "weapon": "Weapons", "weapons": "Weapons", "武器": "Weapons",
    "artifact": "Artifacts", "artifacts": "Artifacts", "圣遗物": "Artifacts",
    "聖遺物": "Artifacts",
    "characterdevelopmentitem": "CharacterDevelopmentItems",
    "characterdevelopmentitems": "CharacterDevelopmentItems",
    "development": "CharacterDevelopmentItems", "养成道具": "CharacterDevelopmentItems",
    "養成道具": "CharacterDevelopmentItems",
    "food": "Food", "食物": "Food",
    "material": "Materials", "materials": "Materials", "材料": "Materials",
    "gadget": "Gadget", "gadgets": "Gadget", "小道具": "Gadget",
    "quest": "Quest", "任务": "Quest", "任務": "Quest",
    "preciousitem": "PreciousItems", "preciousitems": "PreciousItems",
    "贵重道具": "PreciousItems", "貴重道具": "PreciousItems",
    "furnishing": "Furnishings", "furnishings": "Furnishings",
    "摆设": "Furnishings", "擺設": "Furnishings",
}


def inventory_category(value: object) -> InventoryCategory:
    raw = str(value or "").strip()
    if raw.startswith("GridScreenName."):
        raw = raw.rsplit(".", 1)[-1]
    canonical = _CATEGORY_ALIASES.get(raw.casefold(), raw)
    try:
        return _CATEGORIES[canonical]
    except KeyError as error:
        raise ValueError(f"不支持的背包分类：{value}") from error


@dataclass(frozen=True)
class GridCell:
    x: int
    y: int
    width: int
    height: int
    row: int = 0
    column: int = 0

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y:self.y + self.height, self.x:self.x + self.width]


@dataclass(frozen=True)
class ItemCountResult:
    raw_text: str
    count: int
    component_count: int
    bounds: tuple[int, int, int, int] | None = None
    reason: str = ""
    normalized: np.ndarray | None = None


def _cluster_axis(values: list[int], tolerance: int) -> list[int]:
    if not values:
        return []
    ordered = sorted((value, index) for index, value in enumerate(values))
    output = [0] * len(values)
    group: list[tuple[int, int]] = []

    def flush() -> None:
        if not group:
            return
        group_values = sorted(value for value, _ in group)
        median = group_values[len(group_values) // 2]
        for _, index in group:
            output[index] = median
        group.clear()

    for entry in ordered:
        if group and entry[0] - group[-1][0] > tolerance:
            flush()
        group.append(entry)
    flush()
    return output


def detect_inventory_cells(grid_bgr: np.ndarray, columns: int = 8) -> list[GridCell]:
    """Port GridScreen's Canny/shape card detector and row/column ordering."""
    if grid_bgr.size == 0:
        return []
    gray = cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    expected_width = grid_bgr.shape[1] / max(1, columns)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if not height:
            continue
        # Ignore the top-right rarity/lock decoration before measuring the card.
        points = contour.reshape(-1, 2)
        kept = points[
            ~((points[:, 0] - x > width * 0.60) & (points[:, 1] - y < height * 0.28))
        ]
        if kept.size:
            x, y, width, height = cv2.boundingRect(kept.astype(np.int32))
        ratio = width / max(1, height)
        if width < expected_width * 0.62 or width > expected_width * 1.08:
            continue
        if not 0.72 <= ratio <= 0.91:
            continue
        boxes.append((x, y, width, height))
    if not boxes:
        return []

    corrected_x = _cluster_axis([box[0] for box in boxes], max(4, round(expected_width / 3)))
    typical_height = int(np.median([box[3] for box in boxes]))
    corrected_y = _cluster_axis([box[1] for box in boxes], max(4, typical_height // 3))
    deduped: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for index, box in enumerate(boxes):
        key = corrected_x[index], corrected_y[index]
        old = deduped.get(key)
        if old is None or box[2] * box[3] > old[2] * old[3]:
            deduped[key] = box
    ordered = sorted(deduped.values(), key=lambda box: (box[1], box[0]))
    row_values = sorted(set(_cluster_axis([box[1] for box in ordered], typical_height // 3)))
    cells: list[GridCell] = []
    for box in ordered:
        row = min(range(len(row_values)), key=lambda i: abs(row_values[i] - box[1]))
        same_row = sorted(candidate for candidate in ordered if abs(candidate[1] - box[1]) <= typical_height // 3)
        column = min(range(len(same_row)), key=lambda i: abs(same_row[i][0] - box[0]))
        cells.append(GridCell(*box, row=row, column=column))
    return cells


def detect_artifact_set_filter_cells(grid_bgr: np.ndarray, columns: int = 2) -> list[GridCell]:
    """Detect the wide two-column rows used by the artifact-set filter."""
    if grid_bgr.size == 0:
        return []
    gray = cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_width = grid_bgr.shape[1] / columns * 0.66
    boxes = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width >= min_width and height and abs(width / height - 8.63) < 0.4:
            boxes.append((x, y, width, height))
    boxes.sort(key=lambda box: (box[1], box[0]))
    if not boxes:
        return []
    typical_height = max(1, round(float(np.median([box[3] for box in boxes]))))
    row_axis = sorted(set(_cluster_axis([box[1] for box in boxes], max(4, typical_height // 2))))
    cells = []
    for box in boxes:
        row = min(range(len(row_axis)), key=lambda i: abs(row_axis[i] - box[1]))
        cells.append(GridCell(*box, row=row, column=0 if box[0] < grid_bgr.shape[1] / 2 else 1))
    return cells


def _artifact_set_filter_bounds(ctx: GameContext) -> tuple[int, int, int, int]:
    """Return the device-space ROI used by the upstream two-column filter."""
    scale = ctx.transform.scale
    return tuple(round(value * scale) for value in (40, 100, 1300, 852))


def crop_artifact_set_filter_icon(
    card_bgr: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Crop the 60px flower icon from one ArtifactSetFilter card.

    The filter card is a wide row rather than a regular inventory tile.  The
    icon is approximately 267 reference pixels to the left of the row center;
    keeping that geometry here makes GetGridIcons and the accuracy diagnostic
    use exactly the same ItemV2 input.
    """
    if card_bgr.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)
    try:
        scale = max(0.5, float(scale))
    except (TypeError, ValueError):
        scale = 1.0
    icon_size = max(1, round(60 * scale))
    center_x = card_bgr.shape[1] / 2
    center_y = card_bgr.shape[0] / 2
    left = round(center_x - 267 * scale - icon_size / 2)
    top = round(center_y - icon_size / 2)
    right = left + icon_size
    bottom = top + icon_size
    left = max(0, min(left, card_bgr.shape[1]))
    top = max(0, min(top, card_bgr.shape[0]))
    right = max(left, min(right, card_bgr.shape[1]))
    bottom = max(top, min(bottom, card_bgr.shape[0]))
    crop = card_bgr[top:bottom, left:right]
    if crop.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return cv2.resize(crop, (125, 125), interpolation=cv2.INTER_AREA)


def recognize_artifact_set_filter_name(
    ctx: GameContext,
    frame: np.ndarray,
) -> str:
    """Read the flower/set name from the ArtifactSetFilter detail panel."""
    transform = ctx.transform
    x0, y0 = transform.to_device(1371, 545, "right")
    x1, y1 = transform.to_device(1863, 945, "right")
    left, right = sorted((round(x0), round(x1)))
    top, bottom = sorted((round(y0), round(y1)))
    items = get_ocr().recognize(frame[max(0, top):bottom, max(0, left):right])
    items.sort(key=lambda item: (item.y, item.x))
    for index, item in enumerate(items):
        clean = str(item.text).replace(" ", "")
        if (
            "套装包含" in clean
            or "套裝包含" in clean
            or "Set Includes" in str(item.text)
        ) and index + 1 < len(items):
            return str(items[index + 1].text).strip().lstrip("✿❀ ")
    ignored = ("套装", "套裝", "筛选", "篩選", "Set", "件套")
    candidates = [
        str(item.text).strip().lstrip("✿❀ ")
        for item in items
        if str(item.text).strip()
    ]
    return next(
        (text for text in candidates if not any(word in text for word in ignored)),
        "",
    )


_FULLWIDTH_NUMBERS = str.maketrans("０１２３４５６７８９", "0123456789")


def parse_inventory_count(text: str) -> int | None:
    normalized = str(text or "").translate(_FULLWIDTH_NUMBERS).strip()
    normalized = normalized.replace(" ", "")
    return int(normalized) if normalized.isdigit() else None


def _ocr_text(image: np.ndarray) -> str:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    items = get_ocr().recognize(image)
    return "".join(item.text for item in sorted(items, key=lambda item: (item.y, item.x))).strip()


def recognize_inventory_count(cell_bgr: np.ndarray) -> ItemCountResult:
    """Port GridItemCountRecognizer's cropped-number preprocessing."""
    if cell_bgr.size == 0:
        return ItemCountResult("", -2, 0, reason="EMPTY")
    height, width = cell_bgr.shape[:2]
    count_area = cell_bgr[
        height * 128 // 153:height * 150 // 153,
        width * 5 // 125:width * 120 // 125,
    ]
    if count_area.size == 0:
        return ItemCountResult("", -2, 0, reason="EMPTY")
    resized = cv2.resize(count_area, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 0), (180, 140, 210))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components = []
    for contour in contours:
        x, y, item_width, item_height = cv2.boundingRect(contour)
        if (
            item_width > 1 and item_height >= 12 and item_width * item_height >= 20
            and y + item_height >= mask.shape[0] * 0.55
        ):
            components.append((x, y, item_width, item_height))
    components.sort()
    if not components:
        return ItemCountResult("", -2, 0, reason="EMPTY")
    left = min(box[0] for box in components)
    top = min(box[1] for box in components)
    right = max(box[0] + box[2] for box in components)
    bottom = max(box[1] + box[3] for box in components)
    foreground = cv2.bitwise_not(mask[top:bottom, left:right])
    normalized_height = 48
    scaled_width = max(1, round(foreground.shape[1] * normalized_height / foreground.shape[0]))
    scaled = cv2.resize(
        foreground, (scaled_width, normalized_height), interpolation=cv2.INTER_NEAREST
    )
    padding = normalized_height // 2
    normalized = np.full((60, scaled_width + padding * 2), 255, dtype=np.uint8)
    normalized[6:54, padding:padding + scaled_width] = scaled
    raw = _ocr_text(normalized)
    parsed = parse_inventory_count(raw)
    bounds = (left, top, right - left, bottom - top)
    aspect = (right - left) / max(1, bottom - top)
    if len(components) == 1 and aspect <= 0.45 and parsed != 1:
        return ItemCountResult(raw, 1, 1, bounds, "NARROW_ONE", normalized)
    if len(components) == 1 and parsed == 1 and aspect >= 0.65:
        return ItemCountResult(raw, 7, 1, bounds, "WIDE_SEVEN", normalized)
    if parsed is None:
        return ItemCountResult(raw, -2, len(components), bounds, "PARSE", normalized)
    return ItemCountResult(raw, parsed, len(components), bounds, normalized=normalized)


def count_stars(star_bgr: np.ndarray) -> int:
    if star_bgr.size == 0:
        return 0
    mask = cv2.inRange(star_bgr, (45, 199, 250), (55, 209, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len([contour for contour in contours if cv2.contourArea(contour) >= 2])


def normalize_grid_icon(card_bgr: np.ndarray) -> np.ndarray:
    """Return the 125×125 model crop used by BetterGI's ``GetGridIcon``.

    Inventory cards contain a 125×153 image: the bottom 28 pixels are the
    quantity strip.  The desktop implementation first normalizes the whole
    card to 125×153 and then keeps the upper square.  Keeping that ordering is
    important on iPhone layouts because the contour detector may return a
    slightly different card size after safe-area scaling.
    """
    if card_bgr.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)
    if card_bgr.shape[:2] == (125, 125):
        return card_bgr.copy()
    normalized = cv2.resize(card_bgr, (125, 153), interpolation=cv2.INTER_AREA)
    return normalized[:125, :125].copy()


class InventoryGridScanner:
    def __init__(
        self,
        ctx: GameContext,
        category: object,
        *,
        max_pages: int = 100,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.category = inventory_category(category)
        self.max_pages = max(1, min(100, int(max_pages)))
        self.log = log
        prefix = self.category.asset_prefix
        self._checked = Mat.from_file(str(ASSETS / f"bag_{prefix}_checked.png"))
        self._unchecked = Mat.from_file(str(ASSETS / f"bag_{prefix}_unchecked.png"))

    def _find_tab(self, checked: bool):
        template = self._checked if checked else self._unchecked
        ro = RecognitionObject.template_match(template, 0, 0, 1920, 120)
        ro.threshold = 0.76 if checked else 0.82
        return self.ctx.capture_region().find(ro)

    def open(self) -> bool:
        if not GenshinApi(self.ctx, log=self.log).returnMainUi():
            return False
        self.ctx.input.key_press("B")
        self.ctx.sleep(900)
        for _ in range(6):
            checked = self._find_tab(True)
            if checked.is_exist():
                return True
            unchecked = self._find_tab(False)
            if unchecked.is_exist():
                unchecked.click()
                self.ctx.sleep(600)
                return True
            self.ctx.input.key_press("B")
            self.ctx.sleep(500)
        self.log(f"[inventory] 无法打开背包分类 {self.category.name}")
        return False

    def close(self) -> bool:
        return GenshinApi(self.ctx, log=self.log).returnMainUi()

    def _bounds(self) -> tuple[int, int, int, int]:
        x, y, width, height = self.category.roi
        scale = self.ctx.transform.scale
        return round(x * scale), round(y * scale), round(width * scale), round(height * scale)

    def cells(self, frame: np.ndarray) -> list[GridCell]:
        x, y, width, height = self._bounds()
        local = detect_inventory_cells(frame[y:y + height, x:x + width], self.category.columns)
        return [
            GridCell(cell.x + x, cell.y + y, cell.width, cell.height, cell.row, cell.column)
            for cell in local
        ]

    def tap(self, cell: GridCell) -> None:
        x, y = cell.center
        transform = self.ctx.transform
        self.ctx.device.tap(
            x, y,
            image_width=transform.device_width,
            image_height=transform.device_height,
        )

    def _crop_ref(self, frame: np.ndarray, rect: tuple[int, int, int, int], anchor="right"):
        x, y, width, height = rect
        transform = self.ctx.transform
        x0, y0 = transform.to_device(x, y, anchor)
        x1, y1 = transform.to_device(x + width, y + height, anchor)
        left, right = sorted((round(x0), round(x1)))
        top, bottom = sorted((round(y0), round(y1)))
        return frame[max(0, top):bottom, max(0, left):right]

    def detail_name(self, frame: np.ndarray | None = None) -> str:
        frame = self.ctx.capture_bgr() if frame is None else frame
        text = _ocr_text(self._crop_ref(frame, (1310, 120, 490, 62)))
        return re.sub(r"[\r\n]+", "", text).strip()

    def detail_stars(self, frame: np.ndarray | None = None) -> int:
        frame = self.ctx.capture_bgr() if frame is None else frame
        return count_stars(self._crop_ref(frame, (1310, 350, 205, 48)))

    def _fingerprint(self, frame: np.ndarray) -> np.ndarray:
        x, y, width, height = self._bounds()
        gray = cv2.cvtColor(frame[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (48, 32), interpolation=cv2.INTER_AREA)

    def _scroll(self) -> None:
        transform = self.ctx.transform
        x1, y1 = transform.to_device(650, 880, "left")
        x2, y2 = transform.to_device(650, 180, "left")
        self.ctx.device.swipe(
            x1, y1, x2, y2,
            duration_ms=550,
            image_width=transform.device_width,
            image_height=transform.device_height,
        )
        self.ctx.sleep(650)

    def pages(self, cancelled: Callable[[], bool] | None = None) -> Iterator[tuple[int, np.ndarray, list[GridCell]]]:
        for page in range(1, self.max_pages + 1):
            if cancelled and cancelled():
                return
            frame = self.ctx.capture_bgr()
            cells = self.cells(frame)
            if not cells:
                self.log(f"[inventory] 第 {page} 页未检测到网格物品")
                return
            yield page, frame, cells
            if page >= self.max_pages:
                return
            before_frame = self.ctx.capture_bgr()
            before = self._fingerprint(before_frame)
            self._scroll()
            after_frame = self.ctx.capture_bgr()
            after = self._fingerprint(after_frame)
            if float(np.mean(cv2.absdiff(before, after))) < 1.5:
                return


class ArtifactSetFilterGridScanner:
    """Scanner for the manually opened ArtifactSetFilter two-column grid."""

    category = InventoryCategory(
        "ArtifactSetFilter", "artifactsetfilter", (40, 100, 1300, 852), 2,
    )

    def __init__(
        self,
        ctx: GameContext,
        *,
        max_pages: int = 100,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.max_pages = max(1, min(100, int(max_pages)))
        self.log = log

    def open(self) -> bool:
        """The upstream task intentionally requires this page to be open."""
        return True

    def close(self) -> bool:
        # Do not close the manually opened filter page.  This mirrors
        # BetterGI's GetGridIcons/accuracy diagnostic behavior and lets a
        # caller inspect the selected set after the scan.
        return True

    def _bounds(self) -> tuple[int, int, int, int]:
        return _artifact_set_filter_bounds(self.ctx)

    def cells(self, frame: np.ndarray) -> list[GridCell]:
        x, y, width, height = self._bounds()
        local = detect_artifact_set_filter_cells(
            frame[y:y + height, x:x + width], columns=2,
        )
        return [
            GridCell(
                cell.x + x, cell.y + y, cell.width, cell.height,
                cell.row, cell.column,
            )
            for cell in local
        ]

    def tap(self, cell: GridCell) -> None:
        transform = self.ctx.transform
        center_x, center_y = cell.center
        self.ctx.device.tap(
            center_x,
            center_y,
            image_width=transform.device_width,
            image_height=transform.device_height,
        )

    def detail_name(self, frame: np.ndarray | None = None) -> str:
        return recognize_artifact_set_filter_name(
            self.ctx,
            self.ctx.capture_bgr() if frame is None else frame,
        )

    def detail_stars(self, _frame: np.ndarray | None = None) -> int:
        # The filter detail card describes a set/flower and has no stable
        # rarity field.  -1 is the report's explicit "not available" value.
        return -1

    def _fingerprint(self, frame: np.ndarray) -> np.ndarray:
        x, y, width, height = self._bounds()
        gray = cv2.cvtColor(frame[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (48, 32), interpolation=cv2.INTER_AREA)

    def _scroll(self) -> None:
        transform = self.ctx.transform
        x1, y1 = transform.to_device(650, 850, "left")
        x2, y2 = transform.to_device(650, 190, "left")
        self.ctx.device.swipe(
            x1,
            y1,
            x2,
            y2,
            duration_ms=500,
            image_width=transform.device_width,
            image_height=transform.device_height,
        )
        self.ctx.sleep(600)

    def pages(
        self,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[int, np.ndarray, list[GridCell]]]:
        for page in range(1, self.max_pages + 1):
            if cancelled and cancelled():
                return
            frame = self.ctx.capture_bgr()
            cells = self.cells(frame)
            if not cells:
                self.log(f"[inventory] ArtifactSetFilter 第 {page} 页未检测到网格物品")
                return
            yield page, frame, cells
            if page >= self.max_pages:
                return
            before = self._fingerprint(frame)
            self._scroll()
            after = self._fingerprint(self.ctx.capture_bgr())
            if float(np.mean(cv2.absdiff(before, after))) < 1.5:
                return


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "_", str(value)).strip(" .")
    return cleaned or "未识别"


class GetGridIconsTask:
    def __init__(
        self,
        ctx: GameContext,
        category: object,
        *,
        star_as_suffix: bool = False,
        max_num_to_get: int | None = None,
        max_pages: int = 100,
        output_dir: str | Path | None = None,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        raw_category = str(category).split(".")[-1].strip()
        self.artifact_set_filter = raw_category.casefold() in (
            "artifactsetfilter", "圣遗物套装筛选", "聖遺物套裝篩選"
        )
        self.scanner = (
            None if self.artifact_set_filter
            else InventoryGridScanner(ctx, category, max_pages=max_pages, log=log)
        )
        self.category_name = "ArtifactSetFilter" if self.artifact_set_filter else self.scanner.category.name
        self.max_pages = max(1, min(100, int(max_pages)))
        self.star_as_suffix = bool(star_as_suffix)
        self.max_num_to_get = max(1, int(max_num_to_get)) if max_num_to_get else 2**31 - 1
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_dir = Path(output_dir).expanduser() if output_dir else (
            PROJECT_ROOT / "log" / "gridIcons" / f"{self.category_name}{stamp}"
        )
        self.log = log

    def _artifact_filter_bounds(self) -> tuple[int, int, int, int]:
        return _artifact_set_filter_bounds(self.ctx)

    def _artifact_filter_name(self, frame: np.ndarray) -> str:
        return recognize_artifact_set_filter_name(self.ctx, frame)

    def _run_artifact_set_filter(
        self, cancelled: Callable[[], bool] | None
    ) -> list[str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        names: set[str] = set()
        for page in range(1, self.max_pages + 1):
            if cancelled and cancelled():
                break
            frame = self.ctx.capture_bgr()
            x, y, width, height = self._artifact_filter_bounds()
            local = detect_artifact_set_filter_cells(frame[y:y + height, x:x + width])
            cells = [
                GridCell(cell.x + x, cell.y + y, cell.width, cell.height, cell.row, cell.column)
                for cell in local
            ]
            if not cells:
                break
            for cell in cells:
                if len(saved) >= self.max_num_to_get or (cancelled and cancelled()):
                    return saved
                center_x, center_y = cell.center
                transform = self.ctx.transform
                self.ctx.device.tap(
                    center_x, center_y,
                    image_width=transform.device_width,
                    image_height=transform.device_height,
                )
                self.ctx.sleep(280)
                name = self._artifact_filter_name(self.ctx.capture_bgr())
                if not name or name in names:
                    continue
                names.add(name)
                icon = crop_artifact_set_filter_icon(
                    cell.crop(frame),
                    transform.scale,
                )
                if icon.size == 0:
                    continue
                path = self.output_dir / f"{_safe_filename(name)}.png"
                if cv2.imwrite(str(path), icon):
                    saved.append(str(path))
            if page >= self.max_pages:
                break
            before = cv2.resize(
                cv2.cvtColor(frame[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY),
                (48, 32), interpolation=cv2.INTER_AREA,
            )
            transform = self.ctx.transform
            x1, y1 = transform.to_device(650, 850, "left")
            x2, y2 = transform.to_device(650, 190, "left")
            self.ctx.device.swipe(
                x1, y1, x2, y2, duration_ms=500,
                image_width=transform.device_width, image_height=transform.device_height,
            )
            self.ctx.sleep(600)
            after_frame = self.ctx.capture_bgr()
            after = cv2.resize(
                cv2.cvtColor(after_frame[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY),
                (48, 32), interpolation=cv2.INTER_AREA,
            )
            if float(np.mean(cv2.absdiff(before, after))) < 1.5:
                break
        return saved

    def run(self, cancelled: Callable[[], bool] | None = None) -> list[str]:
        with exclusive_realtime_triggers(self.ctx):
            return self._run_impl(cancelled)

    def _run_impl(self, cancelled: Callable[[], bool] | None = None) -> list[str]:
        if self.artifact_set_filter:
            self.log("[GetGridIcons] 圣遗物套装筛选需提前手动打开对应双列界面")
            return self._run_artifact_set_filter(cancelled)
        assert self.scanner is not None
        if not self.scanner.open():
            raise RuntimeError(f"无法打开背包分类 {self.scanner.category.name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        names: set[str] = set()
        try:
            for page, frame, cells in self.scanner.pages(cancelled):
                for cell in cells:
                    if len(saved) >= self.max_num_to_get or (cancelled and cancelled()):
                        return saved
                    self.scanner.tap(cell)
                    self.ctx.sleep(280)
                    detail = self.ctx.capture_bgr()
                    name = self.scanner.detail_name(detail)
                    if not name:
                        self.log(f"[GetGridIcons] 第 {page} 页 ({cell.row},{cell.column}) 名称识别失败")
                        continue
                    suffix = ""
                    if self.star_as_suffix:
                        suffix = "★" * self.scanner.detail_stars(detail)
                    filename = _safe_filename(name + suffix)
                    if filename in names:
                        continue
                    names.add(filename)
                    path = self.output_dir / f"{filename}.png"
                    if cv2.imwrite(str(path), cell.crop(frame)):
                        saved.append(str(path))
                        self.log(f"[GetGridIcons] 图片保存成功：{filename}")
            return saved
        finally:
            self.scanner.close()


class CountInventoryItemTask:
    def __init__(
        self,
        ctx: GameContext,
        category: object,
        *,
        item_name: str | None = None,
        item_names: Iterable[str] | None = None,
        icon_recognition_mode: object = "GridIcon",
        max_pages: int = 100,
        log: Callable[[str], None] = print,
    ):
        one = str(item_name).strip() if item_name is not None else ""
        many = [str(name).strip() for name in (item_names or [])]
        if one and many:
            raise ValueError("参数 itemName 和 itemNames 不能同时使用")
        if not one and not many:
            raise ValueError("参数 itemName 和 itemNames 不能同时为空")
        if any(not name for name in many):
            raise ValueError("参数 itemNames 不能包含空名称")
        self.ctx = ctx
        self.scanner = InventoryGridScanner(ctx, category, max_pages=max_pages, log=log)
        self.item_name = one or None
        self.item_names = tuple(dict.fromkeys(many))
        self.icon_recognition_mode = str(icon_recognition_mode)
        self.log = log

    def run(self, cancelled: Callable[[], bool] | None = None) -> int | dict[str, int]:
        with exclusive_realtime_triggers(self.ctx):
            return self._run_impl(cancelled)

    def _run_impl(self, cancelled: Callable[[], bool] | None = None) -> int | dict[str, int]:
        if not self.scanner.open():
            raise RuntimeError(f"无法打开背包分类 {self.scanner.category.name}")
        wanted = {self.item_name} if self.item_name else set(self.item_names)
        found: dict[str, int] = {}
        seen_names: set[str] = set()
        try:
            for _, frame, cells in self.scanner.pages(cancelled):
                for cell in cells:
                    if cancelled and cancelled():
                        return found if self.item_name is None else found.get(self.item_name, -1)
                    self.scanner.tap(cell)
                    self.ctx.sleep(260)
                    name = self.scanner.detail_name()
                    if not name or name in seen_names:
                        continue
                    seen_names.add(name)
                    if name not in wanted:
                        continue
                    count = recognize_inventory_count(cell.crop(frame)).count
                    found[name] = count
                    self.log(f"[CountInventoryItem] {name}: {count}")
                    if wanted <= found.keys():
                        break
                if wanted <= found.keys():
                    break
        finally:
            self.scanner.close()
        if self.item_name is not None:
            return found.get(self.item_name, -1)
        return found


class InventoryCountComparisonTask:
    """Save regular-vs-cropped quantity OCR diagnostics like BetterGI's tool."""

    def __init__(
        self,
        ctx: GameContext,
        target: object = "CurrentPage",
        *,
        max_pages: int = 100,
        output_dir: str | Path | None = None,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.target = str(target).split(".")[-1]
        self.current_page = self.target.casefold() in ("currentpage", "当前一页", "當前一頁")
        category = "CharacterDevelopmentItems" if self.current_page else self.target
        self.scanner = InventoryGridScanner(ctx, category, max_pages=max_pages, log=log)
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_dir = Path(output_dir).expanduser() if output_dir else (
            PROJECT_ROOT / "log" / "InventoryCountComparison" / stamp
        )
        self.log = log

    @staticmethod
    def _regular_count(cell_bgr: np.ndarray) -> tuple[str, int]:
        height, width = cell_bgr.shape[:2]
        crop = cell_bgr[height * 120 // 153:height, :width]
        raw = _ocr_text(crop)
        parsed = parse_inventory_count(raw)
        return raw, parsed if parsed is not None else -2

    def run(self, cancelled: Callable[[], bool] | None = None) -> str:
        with exclusive_realtime_triggers(self.ctx):
            return self._run_impl(cancelled)

    def _run_impl(self, cancelled: Callable[[], bool] | None = None) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        normalized_dir = self.output_dir / "normalized"
        normalized_dir.mkdir(exist_ok=True)
        csv_path = self.output_dir / "results.csv"
        opened = False
        if not self.current_page:
            opened = self.scanner.open()
            if not opened:
                raise RuntimeError(f"无法打开背包分类 {self.scanner.category.name}")
        try:
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "category", "page", "row", "column", "regular_text", "regular_count",
                    "cropped_text", "cropped_count", "components", "bounds", "reason",
                ])
                pages = self.scanner.pages(cancelled)
                for page, frame, cells in pages:
                    for cell in cells:
                        crop = cell.crop(frame)
                        regular_text, regular_count = self._regular_count(crop)
                        result = recognize_inventory_count(crop)
                        writer.writerow([
                            self.target, page, cell.row, cell.column, regular_text, regular_count,
                            result.raw_text, result.count, result.component_count,
                            result.bounds or "", result.reason,
                        ])
                        if result.normalized is not None and (
                            regular_count != result.count or result.count < 0 or result.reason
                        ):
                            cv2.imwrite(
                                str(normalized_dir / f"p{page}_r{cell.row}_c{cell.column}.png"),
                                result.normalized,
                            )
                    cv2.imwrite(str(self.output_dir / f"page_{page}.png"), frame)
                    if self.current_page:
                        break
            return str(csv_path)
        finally:
            if opened:
                self.scanner.close()


class GridIconsAccuracyTestTask:
    """Compare ItemV2 icon predictions with the selected-card detail OCR.

    BetterGI's Windows build uses a separate legacy ``gridIcon.onnx`` model
    for this developer task.  The iOS port ships the current ItemV2 model,
    which has the same 125×125 input contract and a cosine score.  This task
    therefore keeps the original workflow (open a grid, click every card,
    compare model output with name/star OCR) while returning a structured
    report that is useful from JS and the WebUI.
    """

    def __init__(
        self,
        ctx: GameContext,
        category: object,
        *,
        max_num_to_test: int | None = None,
        max_pages: int = 100,
        output_dir: str | Path | None = None,
        score_threshold: float = 0.75,
        log: Callable[[str], None] = print,
        recognizer: Any | None = None,
        scanner: Any | None = None,
    ):
        self.ctx = ctx
        self.log = log
        raw_category = str(category).split(".")[-1].strip()
        self.artifact_set_filter = raw_category.casefold() in (
            "artifactsetfilter", "圣遗物套装筛选", "聖遺物套裝篩選",
        )
        self.category = None if self.artifact_set_filter else inventory_category(category)
        self.category_name = (
            "ArtifactSetFilter" if self.artifact_set_filter else self.category.name
        )
        self.max_pages = max(1, min(100, int(max_pages)))
        self.max_num_to_test = (
            2**31 - 1 if max_num_to_test is None else max(1, int(max_num_to_test))
        )
        self.score_threshold = float(score_threshold)
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("scoreThreshold 必须在 0 到 1 之间")
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_dir = Path(output_dir).expanduser() if output_dir else (
            PROJECT_ROOT / "log" / "gridIconsAccuracy" / f"{self.category_name}{stamp}"
        )
        self.recognizer = recognizer
        self.scanner = scanner or (
            ArtifactSetFilterGridScanner(ctx, max_pages=self.max_pages, log=log)
            if self.artifact_set_filter
            else InventoryGridScanner(ctx, category, max_pages=self.max_pages, log=log)
        )

    def _recognizer(self) -> Any:
        if self.recognizer is not None:
            return self.recognizer
        from ..vision.item_recognizer import ItemIconRecognizer

        try:
            # ItemV2 stores the quality level for all inventory classes.  The
            # ordinary reward path keeps the upstream relic-only semantics;
            # this diagnostic explicitly asks for all levels so its star
            # comparison remains meaningful for food/material/weapon grids.
            self.recognizer = ItemIconRecognizer(include_quality_levels=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "GridIconsAccuracyTest 缺少 ItemV2 模型或原型表；请先运行 "
                "`.venv/bin/python tools/fetch_map_assets.py --models`"
            ) from error
        return self.recognizer

    @staticmethod
    def _compact_name(value: object) -> str:
        return re.sub(
            r"[\s\u3000·•.…,:：;；!?！？'\"“”‘’()（）\[\]【】<>《》]",
            "",
            str(value or ""),
        ).casefold()

    @classmethod
    def _names_match(cls, predicted: str, ocr_name: str) -> bool:
        left, right = cls._compact_name(predicted), cls._compact_name(ocr_name)
        return bool(left and right) and (left in right or right in left)

    @staticmethod
    def _json_score(value: object) -> float | None:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        return score if math.isfinite(score) else None

    def _normalize_icon(self, cell: GridCell, frame: np.ndarray) -> np.ndarray:
        if self.artifact_set_filter:
            return crop_artifact_set_filter_icon(
                cell.crop(frame),
                self.ctx.transform.scale,
            )
        return normalize_grid_icon(cell.crop(frame))

    def _record(
        self,
        page: int,
        cell: GridCell,
        match: Any,
        ocr_name: str,
        ocr_stars: int,
    ) -> dict[str, object]:
        raw_name = str(getattr(match, "name", "") or "")
        score = self._json_score(getattr(match, "score", None))
        accepted = bool(score is not None and score >= self.score_threshold and raw_name)
        predicted_name = raw_name if accepted else ""
        raw_stars = getattr(match, "quality_level", -1)
        try:
            predicted_stars = int(raw_stars)
        except (TypeError, ValueError):
            predicted_stars = -1
        star_match: bool | None = (
            predicted_stars == int(ocr_stars)
            if predicted_stars >= 0 and int(ocr_stars) >= 0
            else None
        )
        name_match = self._names_match(predicted_name, ocr_name)
        matched = bool(name_match and star_match is not False)
        return {
            "page": page,
            "row": cell.row,
            "column": cell.column,
            "predictedName": predicted_name or None,
            "rawPredictedName": raw_name or None,
            "predictedStars": predicted_stars if predicted_stars >= 0 else None,
            "score": score,
            "ocrName": str(ocr_name or ""),
            "ocrStars": int(ocr_stars),
            "nameMatch": name_match,
            "starMatch": star_match,
            "matched": matched,
        }

    def _write_report(self, report: dict[str, object]) -> str | None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "results.json"
            report["reportPath"] = str(path)
            path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return str(path)
        except OSError as error:
            self.log(f"[GridIconsAccuracyTest] 报告保存失败：{error}")
            return None

    def run(self, cancelled: Callable[[], bool] | None = None) -> dict[str, object]:
        with exclusive_realtime_triggers(self.ctx):
            return self._run_impl(cancelled)

    def _run_impl(self, cancelled: Callable[[], bool] | None = None) -> dict[str, object]:
        recognizer = self._recognizer()
        # ArtifactSetFilter is intentionally a manual-entry page in BetterGI;
        # its scanner.open() is a no-op, but keeping the branch documents that
        # no inventory navigation should be injected on the user's current
        # filter screen.
        opened = True if self.artifact_set_filter else self.scanner.open()
        if not opened:
            raise RuntimeError(f"无法打开背包分类 {self.category.name}")

        records: list[dict[str, object]] = []
        was_cancelled = False
        try:
            for page, frame, cells in self.scanner.pages(cancelled):
                for cell in cells:
                    if cancelled and cancelled():
                        was_cancelled = True
                        break
                    icon = self._normalize_icon(cell, frame)
                    if icon.size == 0:
                        self.log(
                            f"[GridIconsAccuracyTest] 第 {page} 页 ({cell.row},{cell.column}) 图标为空"
                        )
                        continue
                    match = recognizer.match(icon)
                    self.scanner.tap(cell)
                    self.ctx.sleep(280)
                    detail = self.ctx.capture_bgr()
                    ocr_name = self.scanner.detail_name(detail)
                    ocr_stars = self.scanner.detail_stars(detail)
                    record = self._record(page, cell, match, ocr_name, ocr_stars)
                    records.append(record)
                    state = "✔" if record["matched"] else "❌"
                    self.log(
                        f"[GridIconsAccuracyTest] {record['rawPredictedName'] or '未知'}|"
                        f"{record['predictedStars'] if record['predictedStars'] is not None else '?'}星 "
                        f"应为 {ocr_name or '未知'}|{ocr_stars}星，{state}，"
                        f"分数 {record['score'] if record['score'] is not None else '?'}"
                    )
                    if len(records) >= self.max_num_to_test:
                        break
                if was_cancelled or len(records) >= self.max_num_to_test:
                    break
        finally:
            self.scanner.close()

        total = len(records)
        correct = sum(bool(record["matched"]) for record in records)
        name_correct = sum(bool(record["nameMatch"]) for record in records)
        checked_stars = [record for record in records if record["starMatch"] is not None]
        star_correct = sum(bool(record["starMatch"]) for record in checked_stars)
        report: dict[str, object] = {
            "category": self.category_name,
            "model": "ItemV2",
            "scoreThreshold": self.score_threshold,
            "tested": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "nameCorrect": name_correct,
            "nameAccuracy": name_correct / total if total else 0.0,
            "starChecked": len(checked_stars),
            "starCorrect": star_correct,
            "starAccuracy": star_correct / len(checked_stars) if checked_stars else None,
            "cancelled": was_cancelled,
            "results": records,
        }
        report_path = self._write_report(report)
        if report_path is None:
            report.pop("reportPath", None)
        return report
