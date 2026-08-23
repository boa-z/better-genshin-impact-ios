"""Inventory grid scanning and BetterGI-compatible item utilities.

BetterGI normally identifies inventory icons with an ONNX embedding model.
Those runtime model assets are not shipped in its source tree, so the iOS port
uses the same grid-card detector and reads the selected item's detail name via
OCR.  It is slower, but preserves the task result contract without depending
on Windows-only packaged assets.
"""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator

import cv2
import numpy as np

from ..engine.context import GameContext
from ..engine.genshin_api import GenshinApi
from ..engine.recognition import Mat, RecognitionObject
from ..vision.ocr import get_ocr


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
        scale = self.ctx.transform.scale
        return tuple(round(value * scale) for value in (40, 100, 1300, 852))

    def _artifact_filter_name(self, frame: np.ndarray) -> str:
        transform = self.ctx.transform
        x0, y0 = transform.to_device(1371, 545, "right")
        x1, y1 = transform.to_device(1863, 945, "right")
        left, right = sorted((round(x0), round(x1)))
        top, bottom = sorted((round(y0), round(y1)))
        items = get_ocr().recognize(frame[top:bottom, left:right])
        items.sort(key=lambda item: (item.y, item.x))
        for index, item in enumerate(items):
            clean = item.text.replace(" ", "")
            if ("套装包含" in clean or "套裝包含" in clean or "Set Includes" in item.text) \
                    and index + 1 < len(items):
                return items[index + 1].text.strip().lstrip("✿❀ ")
        ignored = ("套装", "套裝", "筛选", "篩選", "Set", "件套")
        candidates = [item.text.strip().lstrip("✿❀ ") for item in items if item.text.strip()]
        return next((text for text in candidates if not any(word in text for word in ignored)), "")

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
                icon_size = max(1, round(60 * transform.scale))
                icon_x = round(cell.x + cell.width / 2 - 267 * transform.scale)
                icon_y = round(cell.y + cell.height / 2 - icon_size / 2)
                icon = frame[
                    max(0, icon_y):icon_y + icon_size,
                    max(0, icon_x):icon_x + icon_size,
                ]
                if icon.size == 0:
                    continue
                icon = cv2.resize(icon, (125, 125), interpolation=cv2.INTER_AREA)
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
