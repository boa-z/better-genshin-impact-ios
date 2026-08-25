"""BetterGI ``CraftMaterialTask`` port for the iOS touch host.

The original job is a small state machine around the crafting grid, not a
series of blind text clicks.  This implementation keeps the same important
guards on a mobile screen:

* resolve the material filter from ``assets/models/item.csv`` when the caller
  did not provide one;
* identify grid cards with the existing ItemV2 embedding model and confirm the
  selected material in the detail panel;
* set the quantity through a touch drag, then correct and verify the displayed
  integer;
* submit the craft, accept the consumption dialog, collect the reward result,
  and expose a structured result to JS/dispatcher callers.

The task owns one frame at a time and pauses the realtime trigger loop while
the crafting UI is changing, so AutoPick/AutoSkip cannot inject input into a
transition frame.
"""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from ..engine.context import GameContext
from ..engine.recognition import ImageRegion, RecognitionObject
from ..vision.item_recognizer import ItemIconRecognizer
from .common_jobs import exclusive_realtime_triggers
from .inventory_grid import GridCell, detect_inventory_cells
from .reward_result import RewardResultRecognizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ITEM_CSV = PROJECT_ROOT / "assets" / "models" / "item.csv"

# Values are BetterGI's 1920x1080 reference coordinates.  The crafting page
# has five columns and a narrower grid than the regular eight-column inventory.
CRAFTING_ROI = (45, 170, 705, 790)
CRAFTING_COLUMNS = 5
FILTER_BUTTON_ROI = (70, 970, 180, 75)
FILTER_OPTION_ROI = (35, 124, 300, 650)
DETAIL_NAME_ROI = (1139, 85, 520, 110)
CURRENT_QUANTITY_ROI = (1248, 615, 221, 58)
MAX_QUANTITY_ROI = (1500, 630, 150, 80)
MATERIAL_COUNTS_ROI = (900, 880, 650, 90)
CRAFT_BUTTON_ROI = (1540, 900, 380, 180)
CONFIRM_DIALOG_ROI = (850, 650, 650, 220)
RESULT_CONFIRM_ROI = (700, 820, 500, 180)

_CRAFTING_MARKERS = ("合成", "合成台", "Craft", "Crafting")
_FILTER_MARKERS = ("筛选", "篩選", "Filter")
_CRAFT_MARKERS = ("合成", "Craft")
_CONFIRM_MARKERS = ("确认", "確定", "确认合成", "確定合成", "Confirm")
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def _compact(value: Any) -> str:
    return (
        str(value or "")
        .translate(_FULLWIDTH_DIGITS)
        .replace(" ", "")
        .replace("\u3000", "")
        .replace("\n", "")
        .replace("\r", "")
        .strip()
    )


def _contains(value: Any, markers: Iterable[str]) -> bool:
    text = _compact(value).casefold()
    return any(_compact(marker).casefold() in text for marker in markers)


def _first_positive_int(value: Any) -> int:
    match = re.search(r"\d+", _compact(value))
    return int(match.group()) if match else 0


@lru_cache(maxsize=1)
def load_material_types(path: str | Path = ITEM_CSV) -> dict[str, str]:
    """Load the same material-type source used by BetterGI's ItemV2 task."""
    source = Path(path)
    if not source.is_file():
        return {}
    result: dict[str, str] = {}
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                name = str(row.get("item_name", "")).strip()
                material_type = str(row.get("material_type", "")).strip()
                if name and material_type:
                    result[name] = material_type
    except (OSError, csv.Error, UnicodeError):
        return {}
    return result


@dataclass
class CraftMaterialResult:
    success: bool = False
    material_name: str = ""
    target_quantity: int = 0
    actual_quantity: int = 0
    material_type: str = ""
    crafted: int = 0
    rewards: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    cancelled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": bool(self.success),
            "materialName": self.material_name,
            "targetQuantity": self.target_quantity,
            "actualQuantity": self.actual_quantity,
            "materialType": self.material_type,
            "crafted": self.crafted,
            "requested": self.target_quantity,
            "rewards": list(self.rewards),
            **({"error": self.error} if self.error else {}),
            **({"cancelled": True} if self.cancelled else {}),
        }


class CraftMaterialTask:
    """Craft one material in an already-open crafting screen."""

    def __init__(
        self,
        ctx: GameContext,
        material_name: str,
        quantity: int,
        material_type: str | None = None,
        *,
        max_pages: int = 30,
        icon_threshold: float = 0.75,
        timeout_s: float = 90.0,
        recognizer: Any | None = None,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.material_name = str(material_name or "").strip()
        try:
            self.target_quantity = int(quantity)
        except (TypeError, ValueError) as error:
            raise ValueError("quantity 必须为正整数") from error
        if not self.material_name:
            raise ValueError("material_name 不能为空")
        if self.target_quantity <= 0:
            raise ValueError("quantity 必须大于 0")
        self.material_type = str(material_type or "").strip() or None
        self.max_pages = max(1, min(100, int(max_pages)))
        self.icon_threshold = max(0.5, min(0.99, float(icon_threshold)))
        self.timeout_s = max(10.0, min(300.0, float(timeout_s)))
        self._recognizer = recognizer
        self.log = log
        self._full_ocr = RecognitionObject.ocr(0, 0, 1920, 1080)

    def _log(self, message: str) -> None:
        self.log(f"[CraftMaterial] {message}")

    @staticmethod
    def _cancelled(cancelled: Callable[[], bool] | None) -> bool:
        try:
            return bool(cancelled and cancelled())
        except Exception:
            return True

    def _resolve_material_type(self) -> str | None:
        if self.material_type:
            return self.material_type
        types = load_material_types()
        return types.get(self.material_name)

    def _recognizer_instance(self):
        if self._recognizer is None:
            self._recognizer = ItemIconRecognizer()
        return self._recognizer

    def _text_hits(
        self,
        frame: np.ndarray,
        roi: tuple[int, int, int, int] = (0, 0, 1920, 1080),
        *,
        limit: int = 60,
    ):
        region = ImageRegion(self.ctx, frame)
        return region, region.find_multi(RecognitionObject.ocr(*roi), limit=limit)

    @staticmethod
    def _hit_text(hits: Iterable[Any]) -> str:
        return "".join(
            str(getattr(hit, "text", "") or "")
            for hit in sorted(
                hits,
                key=lambda hit: (
                    float(getattr(hit, "y", 0)),
                    float(getattr(hit, "x", 0)),
                ),
            )
        )

    @classmethod
    def _find_hit(cls, hits: Iterable[Any], markers: Iterable[str]):
        return next(
            (hit for hit in hits if _contains(getattr(hit, "text", ""), markers)),
            None,
        )

    def _wait_for_crafting_ui(
        self,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        while time.monotonic() < deadline:
            if self._cancelled(cancelled):
                return False
            frame = self.ctx.capture_bgr()
            _region, hits = self._text_hits(frame)
            if (
                self._find_hit(hits, _CRAFTING_MARKERS) is not None
                and self._find_hit(hits, _FILTER_MARKERS) is not None
            ):
                return True
            self.ctx.sleep(350)
        return False

    def _select_material_type(
        self,
        material_type: str,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        # If the selected filter is already shown in the footer, no click is
        # necessary and we avoid opening a transition dialog.
        while time.monotonic() < deadline:
            if self._cancelled(cancelled):
                return False
            frame = self.ctx.capture_bgr()
            _region, footer_hits = self._text_hits(frame, FILTER_BUTTON_ROI)
            if self._find_hit(footer_hits, (material_type,)) is not None:
                return True
            filter_hit = self._find_hit(footer_hits, _FILTER_MARKERS)
            if filter_hit is not None:
                filter_hit.click()
                self.ctx.sleep(450)
                break
            self.ctx.sleep(300)
        else:
            return False

        while time.monotonic() < deadline:
            if self._cancelled(cancelled):
                return False
            frame = self.ctx.capture_bgr()
            _region, options = self._text_hits(frame, FILTER_OPTION_ROI)
            option = self._find_hit(options, (material_type,))
            if option is not None:
                option.click()
                self.ctx.sleep(500)
                return True
            self.ctx.sleep(300)
        return False

    def _grid_bounds(self) -> tuple[int, int, int, int]:
        transform = self.ctx.transform
        x, y, width, height = CRAFTING_ROI
        x0, y0 = transform.to_device(x, y, anchor="left")
        x1, y1 = transform.to_device(x + width, y + height, anchor="left")
        left, right = sorted((round(x0), round(x1)))
        top, bottom = sorted((round(y0), round(y1)))
        return left, top, right - left, bottom - top

    def _scan_cells(self, frame: np.ndarray) -> list[GridCell]:
        left, top, width, height = self._grid_bounds()
        right = min(frame.shape[1], left + width)
        bottom = min(frame.shape[0], top + height)
        left = max(0, left)
        top = max(0, top)
        if right <= left or bottom <= top:
            return []
        cells = detect_inventory_cells(
            frame[top:bottom, left:right], columns=CRAFTING_COLUMNS,
        )
        return [
            GridCell(
                cell.x + left, cell.y + top, cell.width, cell.height,
                cell.row, cell.column,
            )
            for cell in cells
        ]

    @staticmethod
    def _icon_crop(cell: GridCell, frame: np.ndarray) -> np.ndarray:
        crop = cell.crop(frame)
        if crop.size == 0:
            return crop
        # ItemV2 expects the 125x125 icon, while a crafting card also contains
        # the quantity strip at its bottom.  Prefer the square upper portion.
        side = min(crop.shape[0], crop.shape[1], 125)
        return crop[:side, :side]

    @staticmethod
    def _same_name(left: str, right: str) -> bool:
        return _compact(left).casefold() == _compact(right).casefold()

    def _tap_cell(self, cell: GridCell) -> None:
        transform = self.ctx.transform
        device = getattr(self.ctx, "device", None)
        if device is not None and callable(getattr(device, "tap", None)):
            device.tap(
                *cell.center,
                image_width=transform.device_width,
                image_height=transform.device_height,
            )
            return
        ref_x, ref_y = transform.to_ref(*cell.center, anchor="left")
        self.ctx.input.click_ref(ref_x, ref_y)

    def _detail_matches(self, frame: np.ndarray) -> bool:
        _region, hits = self._text_hits(frame, DETAIL_NAME_ROI, limit=20)
        return self._find_hit(hits, (self.material_name,)) is not None

    def _scroll_grid(self) -> None:
        transform = self.ctx.transform
        x1, y1 = transform.to_device(650, 880, anchor="left")
        x2, y2 = transform.to_device(650, 250, anchor="left")
        self.ctx.device.swipe(
            x1, y1, x2, y2,
            duration_ms=550,
            image_width=transform.device_width,
            image_height=transform.device_height,
        )
        self.ctx.sleep(650)

    @staticmethod
    def _fingerprint(frame: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = bounds
        crop = frame[max(0, y):y + height, max(0, x):x + width]
        if crop.size == 0:
            return np.zeros((16, 24), dtype=np.uint8)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (24, 16), interpolation=cv2.INTER_AREA)

    def _find_and_select_material(
        self,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        recognizer = self._recognizer_instance()
        previous: np.ndarray | None = None
        seen_pages: set[bytes] = set()
        bounds = self._grid_bounds()
        for page in range(self.max_pages):
            if self._cancelled(cancelled) or time.monotonic() >= deadline:
                return False
            frame = self.ctx.capture_bgr()
            fingerprint = self._fingerprint(frame, bounds)
            key = fingerprint.tobytes()
            if key in seen_pages:
                return False
            seen_pages.add(key)
            cells = self._scan_cells(frame)
            for cell in cells:
                if self._cancelled(cancelled):
                    return False
                try:
                    match = recognizer.match(self._icon_crop(cell, frame))
                except (OSError, RuntimeError, ValueError, TypeError) as error:
                    self._log(f"材料图标识别失败：{error}")
                    return False
                name = str(getattr(match, "name", "") or "")
                score = float(getattr(match, "score", 0.0) or 0.0)
                if score < self.icon_threshold or not self._same_name(name, self.material_name):
                    continue
                self._tap_cell(cell)
                self.ctx.sleep(500)
                selected_frame = self.ctx.capture_bgr()
                if self._detail_matches(selected_frame):
                    self._log(f"选中材料 {self.material_name}（置信度 {score:.2f}）")
                    return True
                self._log(f"材料卡片已匹配但详情未确认：{name}（{score:.2f}）")

            if page + 1 >= self.max_pages:
                break
            current = self._fingerprint(self.ctx.capture_bgr(), bounds)
            if previous is not None and float(np.mean(cv2.absdiff(previous, current))) < 1.5:
                break
            previous = current
            self._scroll_grid()
        return False

    def _read_text(self, roi: tuple[int, int, int, int]) -> str:
        frame = self.ctx.capture_bgr()
        _region, hits = self._text_hits(frame, roi, limit=30)
        return self._hit_text(hits)

    def _read_max_quantity(self) -> int:
        direct = _first_positive_int(self._read_text(MAX_QUANTITY_ROI))
        if direct > 0:
            return direct
        raw = _compact(self._read_text(MATERIAL_COUNTS_ROI))
        candidates = []
        for owned, required in re.findall(r"(\d+)\s*/\s*(\d+)", raw):
            owned_i, required_i = int(owned), int(required)
            if required_i > 0:
                candidates.append(owned_i // required_i)
        return min(candidates) if candidates else 0

    def _read_current_quantity(self) -> int:
        return _first_positive_int(self._read_text(CURRENT_QUANTITY_ROI))

    def _drag_slider(self, ratio: float) -> None:
        ratio = max(0.0, min(1.0, float(ratio)))
        start_x, end_x, y = 1173.0, 1521.0, 672.0
        drag = getattr(self.ctx.input, "drag_ref", None)
        if callable(drag):
            drag(start_x, y, start_x + (end_x - start_x) * ratio, y, duration_ms=350)
            return
        transform = self.ctx.transform
        x1, y1 = transform.to_device(start_x, y)
        x2, y2 = transform.to_device(start_x + (end_x - start_x) * ratio, y)
        self.ctx.device.swipe(
            x1, y1, x2, y2,
            duration_ms=350,
            image_width=transform.device_width,
            image_height=transform.device_height,
        )

    def _set_quantity(self, deadline: float, cancelled: Callable[[], bool] | None) -> int:
        max_quantity = self._read_max_quantity()
        if max_quantity <= 0:
            raise RuntimeError("未能识别最大可合成个数")
        if self.target_quantity > max_quantity:
            raise RuntimeError(
                f"材料不足以合成指定个数：目标 {self.target_quantity}，最多 {max_quantity}"
            )
        if max_quantity == 1:
            return 1

        self._drag_slider(0.0)
        self.ctx.sleep(250)
        ratio = (self.target_quantity - 1) / (max_quantity - 1)
        self._drag_slider(ratio)
        self.ctx.sleep(350)
        actual = self._read_current_quantity()
        if actual <= 0:
            raise RuntimeError("未能读取当前合成个数")

        # Slider precision differs slightly between device scales. Correct it
        # using the visible +/- buttons, then read it back once more.
        delta = self.target_quantity - actual
        if abs(delta) > 30:
            self._drag_slider(ratio)
            self.ctx.sleep(350)
            actual = self._read_current_quantity()
            delta = self.target_quantity - actual
        if abs(delta) > 30:
            raise RuntimeError(
                f"滑块调整失败：目标 {self.target_quantity}，当前 {actual}"
            )
        button_x = 1614 if delta > 0 else 1074
        for _ in range(abs(delta)):
            if self._cancelled(cancelled) or time.monotonic() >= deadline:
                raise RuntimeError("合成数量调整已取消或超时")
            self.ctx.input.click_ref(button_x, 672)
            self.ctx.sleep(60)
        self.ctx.sleep(200)
        verified = self._read_current_quantity()
        if verified != self.target_quantity:
            raise RuntimeError(
                f"最终合成个数与目标不一致：目标 {self.target_quantity}，当前 {verified}"
            )
        return verified

    def _wait_click_text(
        self,
        markers: Iterable[str],
        roi: tuple[int, int, int, int],
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        while time.monotonic() < deadline:
            if self._cancelled(cancelled):
                return False
            frame = self.ctx.capture_bgr()
            _region, hits = self._text_hits(frame, roi, limit=30)
            hit = self._find_hit(hits, tuple(markers))
            if hit is not None:
                hit.click()
                self.ctx.sleep(500)
                return True
            self.ctx.sleep(300)
        return False

    def _recognize_rewards(self, actual_quantity: int) -> list[dict[str, Any]]:
        try:
            summary = RewardResultRecognizer(self.ctx, log=self.log).recognize_multi_page(1)
            return [
                {"name": str(name), "quantity": int(quantity)}
                for name, quantity in summary.items()
                if int(quantity) > 0
            ]
        except (OSError, RuntimeError, ValueError, TypeError, ImportError) as error:
            self._log(f"产物识别失败，使用默认产物：{error}")
            return [{"name": self.material_name, "quantity": actual_quantity}]

    def _submit(
        self,
        actual_quantity: int,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> list[dict[str, Any]]:
        if not self._wait_click_text(_CRAFT_MARKERS, CRAFT_BUTTON_ROI, deadline, cancelled):
            raise RuntimeError("未找到合成按钮")
        if not self._wait_click_text(_CONFIRM_MARKERS, CONFIRM_DIALOG_ROI, deadline, cancelled):
            raise RuntimeError("未找到合成确认按钮")
        self.ctx.sleep(900)
        rewards = self._recognize_rewards(actual_quantity)
        if not self._wait_click_text(_CONFIRM_MARKERS, RESULT_CONFIRM_ROI, deadline, cancelled):
            raise RuntimeError("未找到产物弹窗确认按钮")
        return rewards

    def _run_locked(
        self,
        cancelled: Callable[[], bool] | None,
        deadline: float,
    ) -> CraftMaterialResult:
        result = CraftMaterialResult(
            material_name=self.material_name,
            target_quantity=self.target_quantity,
        )
        material_type = self._resolve_material_type()
        if not material_type:
            result.error = (
                f"未找到材料 {self.material_name} 的材料类型；请显式传入 materialType"
            )
            return result
        result.material_type = material_type
        if not self._wait_for_crafting_ui(deadline, cancelled):
            result.cancelled = self._cancelled(cancelled)
            result.error = "当前不在合成界面或等待超时"
            return result
        if not self._select_material_type(material_type, deadline, cancelled):
            result.error = f"未能选择材料筛选类型：{material_type}"
            return result
        if not self._find_and_select_material(deadline, cancelled):
            result.error = f"未找到材料：{self.material_name}"
            return result
        result.actual_quantity = self._set_quantity(deadline, cancelled)
        result.rewards = self._submit(result.actual_quantity, deadline, cancelled)
        result.crafted = result.actual_quantity
        result.success = True
        self._log(
            f"合成完成：{self.material_name} x{result.actual_quantity}，产物 {result.rewards}"
        )
        return result

    def run(self, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        try:
            with exclusive_realtime_triggers(self.ctx):
                result = self._run_locked(cancelled, time.monotonic() + self.timeout_s)
        except Exception as error:
            result = CraftMaterialResult(
                material_name=self.material_name,
                target_quantity=self.target_quantity,
                error=str(error),
                cancelled=self._cancelled(cancelled),
            )
            self._log(f"执行失败：{error}")
        return result.as_dict()


__all__ = [
    "CRAFTING_ROI",
    "CraftMaterialResult",
    "CraftMaterialTask",
    "load_material_types",
]
