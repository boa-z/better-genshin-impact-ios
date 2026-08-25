"""BetterGI-compatible automatic woodcutting task.

The desktop task does more than repeatedly press attack and ``Z``: it can
count the reward overlay, stop at a per-material daily limit, and refresh the
gadget cooldown by entering/leaving Wonderland.  This module keeps that
behaviour independent from the web host so both JavaScript scripts and the
native dispatcher use the same state machine.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ..engine.context import GameContext
from ..engine.genshin_api import GenshinApi
from ..engine.recognition import ImageRegion, Mat, RecognitionObject
from ..vision.ocr import get_ocr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GADGET_ROI = (1440, 540, 480, 540)
WOOD_REWARD_ROI = (100, 450, 300, 250)
GADGET_TEMPLATE_PATH = PROJECT_ROOT / "assets" / "templates" / "autowood" / "TheBoonOfTheElderTree.png"

# Names published by Genshin's wood inventory.  Keeping the allow-list avoids
# treating the OCR header ("获得") or a random HUD number as a material.
WOOD_NAMES = frozenset({
    "悬铃木", "白梣木", "炬木", "椴木", "香柏木", "刺葵木", "柽木",
    "辉木", "业果木", "证悟木", "枫木", "垂香木", "杉木", "竹节",
    "却砂木", "松木", "萃华木", "桦木", "孔雀木", "梦见木", "御伽木",
    "燃爆木", "桃椰子木", "灰灰楼林木", "白栗栎木",
})

_WOOD_QUANTITY_RE = re.compile(r"(?P<name>[^0-9\n×xX*]+?)[\s]*[×xX*][\s]*(?P<count>\d+)")
_WOOD_NAME_ORDER = tuple(sorted(WOOD_NAMES, key=len, reverse=True))


def _compact_text(value: Any) -> str:
    """Normalize OCR text without changing Chinese material names."""

    return re.sub(r"\s+", "", str(value or "")).replace("\u200b", "")


def _canonical_wood_name(value: str) -> str | None:
    name = _compact_text(value).strip(":：-—|·．.")
    if name in WOOD_NAMES:
        return name
    # OCR can attach the reward header to the first item.  Prefer a known
    # suffix rather than adding a false material to the running statistics.
    for candidate in _WOOD_NAME_ORDER:
        if name.endswith(candidate):
            return candidate
    return None


def parse_wood_statistics(text: str | Iterable[str] | None) -> dict[str, int]:
    """Parse ``获得\n竹节×30\n杉木×20`` OCR into material quantities.

    RapidOCR returns one item per line on some frames and one combined string
    on others, hence the parser accepts both a string and an iterable of OCR
    items/text fragments.  Both the multiplication sign and ASCII ``x`` are
    accepted because the latter is a common OCR substitution.
    """

    if text is None:
        return {}
    if isinstance(text, str):
        source = text
    else:
        source = "".join(
            str(getattr(item, "text", item)) for item in text
        )
    quantities: dict[str, int] = {}
    for match in _WOOD_QUANTITY_RE.finditer(source):
        name = _canonical_wood_name(match.group("name"))
        if name is None:
            continue
        quantity = int(match.group("count"))
        quantities[name] = quantities.get(name, 0) + quantity

    # A few OCR engines omit the separator around a short name when the
    # overlay is animated.  The known-name pass recovers those cases while
    # still rejecting unknown text.
    compact = _compact_text(source)
    for name in _WOOD_NAME_ORDER:
        for match in re.finditer(
            re.escape(name) + r"[×xX*](\d+)", compact,
        ):
            quantities[name] = max(
                quantities.get(name, 0), int(match.group(1)),
            )
    return quantities


def _normalized_limit(value: Any, *, default: int = 9999) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    # This is the exact WoodTaskParam convention in BetterGI: zero means
    # unlimited and 9999 is the sentinel used by the UI.
    return default if result <= 0 or result >= default else result


@dataclass
class WoodStatistics:
    """Cumulative per-material counter used by ``AutoWoodTask``."""

    daily_max_count: int = 9999
    empty_limit: int = 3
    totals: dict[str, int] = field(default_factory=dict)
    first_round_quantities: dict[str, int] = field(default_factory=dict)
    nothing_count: int = 0

    def __post_init__(self) -> None:
        self.daily_max_count = _normalized_limit(self.daily_max_count)
        self.empty_limit = max(1, int(self.empty_limit))

    @property
    def reached_max_count(self) -> bool:
        # Match the desktop implementation: the limit is reached only when
        # every material observed so far has reached it.
        return bool(self.totals) and min(self.totals.values()) >= self.daily_max_count

    @property
    def always_empty(self) -> bool:
        return self.nothing_count >= self.empty_limit

    def record(self, text: str | Iterable[str] | None) -> dict[str, int]:
        """Record one reward overlay and return the quantities added."""

        quantities = parse_wood_statistics(text)
        if not quantities:
            self.nothing_count += 1
            return {}

        self.nothing_count = 0
        if not self.first_round_quantities:
            self.first_round_quantities = dict(quantities)
        elif len(quantities) <= len(self.first_round_quantities):
            # The original task reuses the first complete OCR result when a
            # later animated frame contains only a subset of the entries.
            quantities = {
                name: self.first_round_quantities.get(name, count)
                for name, count in self.first_round_quantities.items()
                if name in quantities or count <= self.daily_max_count
            }

        for name, quantity in quantities.items():
            self.totals[name] = self.totals.get(name, 0) + int(quantity)
        return dict(quantities)


def _reference_crop(
    frame: np.ndarray,
    ctx: GameContext,
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop a 1920x1080 reference ROI from a native landscape frame."""

    rx, ry, rw, rh = roi
    transform = getattr(ctx, "transform", None)
    if transform is not None:
        try:
            x, y = transform.to_device(rx, ry)
            width = transform.scale_len(rw)
            height = transform.scale_len(rh)
        except (AttributeError, TypeError, ValueError):
            x = rx * frame.shape[0] / 1080.0
            y = ry * frame.shape[0] / 1080.0
            width = rw * frame.shape[0] / 1080.0
            height = rh * frame.shape[0] / 1080.0
    else:
        scale = frame.shape[0] / 1080.0
        x, y, width, height = rx * scale, ry * scale, rw * scale, rh * scale
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1 = min(frame.shape[1], int(round(x + width)))
    y1 = min(frame.shape[0], int(round(y + height)))
    if x1 <= x0 or y1 <= y0:
        return frame[0:0, 0:0]
    return frame[y0:y1, x0:x1]


class AutoWoodTask:
    """Automate woodcutting with the same controls as BetterGI's task."""

    def __init__(
        self,
        ctx: GameContext,
        rounds: int = 1,
        per_round_attacks: int = 8,
        relogin_between: bool | None = False,
        log: Callable[[str], None] = print,
        *,
        wood_daily_max_count: int = 9999,
        wood_count_ocr_enabled: bool = False,
        use_wonderland_refresh: bool = True,
        after_z_sleep_delay_ms: int = 0,
        empty_ocr_limit: int = 3,
        ocr_timeout_ms: int = 3500,
        ocr_poll_interval_ms: int = 300,
        ocr_final_round: bool = False,
        gadget_key: str = "Z",
        gadget_check_enabled: bool = True,
        gadget_check_strict: bool = True,
        gadget_wait_timeout_s: float = 3.0,
        refresh_fallback_to_relogin: bool = False,
        ocr_provider: Any = None,
        gadget_detector: Callable[[np.ndarray], bool | None] | None = None,
    ):
        self.ctx = ctx
        raw_rounds = int(rounds)
        self.rounds = 9999 if raw_rounds == 0 else max(1, raw_rounds)
        self.per_round_attacks = max(0, int(per_round_attacks))
        self.relogin_between = (
            None if relogin_between is None else bool(relogin_between)
        )
        self.wood_count_ocr_enabled = bool(wood_count_ocr_enabled)
        self.use_wonderland_refresh = bool(use_wonderland_refresh)
        self.after_z_sleep_delay_ms = max(0, int(after_z_sleep_delay_ms))
        self.ocr_timeout_ms = max(100, int(ocr_timeout_ms))
        self.ocr_poll_interval_ms = max(50, int(ocr_poll_interval_ms))
        self.ocr_final_round = bool(ocr_final_round)
        self.gadget_key = str(gadget_key or "Z").strip().upper()
        self.gadget_check_enabled = bool(gadget_check_enabled)
        self.gadget_check_strict = bool(gadget_check_strict)
        self.gadget_wait_timeout_s = max(0.2, float(gadget_wait_timeout_s))
        self.refresh_fallback_to_relogin = bool(refresh_fallback_to_relogin)
        self.ocr_provider = ocr_provider
        self.gadget_detector = gadget_detector
        self.log = log
        self.genshin = GenshinApi(ctx, log)
        self.statistics = WoodStatistics(
            daily_max_count=wood_daily_max_count,
            empty_limit=empty_ocr_limit,
        )
        self._gadget_template: Mat | None = None
        self._gadget_template_checked = False

    @property
    def wood_totals(self) -> dict[str, int]:
        """Return a copy for JS/task callers that want the final counters."""

        return dict(self.statistics.totals)

    @property
    def wood_daily_max_count(self) -> int:
        return self.statistics.daily_max_count

    @property
    def nothing_count(self) -> int:
        return self.statistics.nothing_count

    def _cancelled(self, cancelled: Callable[[], bool] | None) -> bool:
        return bool(cancelled and cancelled())

    def _capture(self, *, fresh: bool = False) -> np.ndarray | None:
        if not fresh:
            cached = getattr(self.ctx, "cached_frame", None)
            if callable(cached):
                try:
                    value = cached()
                    if isinstance(value, (tuple, list)):
                        frame = value[0] if value else None
                    else:
                        frame = value
                    if frame is not None:
                        return frame
                except Exception as error:
                    self.log(f"[AutoWood] 读取缓存截图失败：{error}")
        capture = getattr(self.ctx, "capture_bgr", None)
        if not callable(capture):
            return None
        try:
            return capture()
        except Exception as error:
            self.log(f"[AutoWood] 截图失败：{error}")
            return None

    def _ocr_text(self, frame: np.ndarray) -> str:
        crop = _reference_crop(frame, self.ctx, WOOD_REWARD_ROI)
        provider = self.ocr_provider or get_ocr()
        if callable(provider) and not callable(getattr(provider, "recognize", None)):
            result = provider(crop)
        else:
            result = provider.recognize(crop)
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return str(result.get("text", ""))
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict)
            else str(getattr(item, "text", item))
            for item in (result or ())
        )

    def _wait_wood_reward(self, cancelled: Callable[[], bool] | None) -> str:
        """Poll only the reward ROI, avoiding full-screen OCR requests."""

        attempts = max(1, (self.ocr_timeout_ms + self.ocr_poll_interval_ms - 1)
                       // self.ocr_poll_interval_ms)
        best = ""
        for _ in range(attempts):
            if self._cancelled(cancelled):
                return ""
            frame = self._capture(fresh=True)
            if frame is not None and getattr(frame, "size", 0):
                try:
                    text = self._ocr_text(frame)
                except Exception as error:
                    self.log(f"[AutoWood] 木材 OCR 失败：{error}")
                    text = ""
                if parse_wood_statistics(text):
                    if len(text) > len(best):
                        best = text
                    # The overlay is complete when at least one known item is
                    # visible.  A later, shorter animated frame is discarded.
                    if best and len(text) < len(best):
                        break
            self.ctx.sleep(self.ocr_poll_interval_ms)
        return best

    def _load_gadget_template(self) -> Mat | None:
        if self._gadget_template_checked:
            return self._gadget_template
        self._gadget_template_checked = True
        if not GADGET_TEMPLATE_PATH.is_file():
            return None
        try:
            self._gadget_template = Mat.from_file(str(GADGET_TEMPLATE_PATH))
        except Exception as error:
            self.log(f"[AutoWood] 小道具模板读取失败（继续 OCR 检查）：{error}")
        return self._gadget_template

    def _gadget_state(self, frame: np.ndarray | None) -> bool | None:
        """Return True/False when the gadget is known, None when undecidable."""

        if not self.gadget_check_enabled or frame is None:
            return None
        if self.gadget_detector is not None:
            try:
                value = self.gadget_detector(frame)
                return None if value is None else bool(value)
            except Exception as error:
                self.log(f"[AutoWood] 小道具检测器失败：{error}")
                return None
        # Empty synthetic frames are common in offline host tests and contain
        # no evidence either way.  Treat them as unknown rather than blocking
        # a task that is otherwise fully testable without a device.
        if not np.asarray(frame).size or float(np.std(frame)) < 0.5:
            return None
        template = self._load_gadget_template()
        if template is not None:
            try:
                region = ImageRegion(self.ctx, frame)
                ro = RecognitionObject.template_match(template, *GADGET_ROI)
                ro.threshold = 0.65
                if region.find(ro).is_exist():
                    return True
                return False
            except Exception as error:
                self.log(f"[AutoWood] 小道具模板检测失败（继续 OCR）：{error}")

        try:
            text = _compact_text(
                "".join(
                    str(getattr(item, "text", item))
                    for item in get_ocr().recognize(
                        _reference_crop(frame, self.ctx, GADGET_ROI)
                    )
                )
            )
        except Exception:
            text = ""
        if any(word in text for word in ("王树瑞佑", "王树瑞祐", "BoonOfTheElderTree")):
            return True
        if any(word in text for word in ("未装备", "未装配", "NotEquipped")):
            return False
        return None

    def _press_gadget(self, cancelled: Callable[[], bool] | None) -> bool:
        deadline = time.monotonic() + self.gadget_wait_timeout_s
        warned = False
        while True:
            if self._cancelled(cancelled):
                return False
            state = self._gadget_state(self._capture())
            if state is True or state is None or not self.gadget_check_strict:
                if state is False and not warned:
                    self.log("[AutoWood] 未确认「王树瑞佑」图标，按配置继续尝试使用")
                    warned = True
                self.ctx.input.key_press(self.gadget_key)
                self.ctx.sleep(300 + self.after_z_sleep_delay_ms)
                return True
            if time.monotonic() >= deadline:
                self.log("[AutoWood] 未检测到已装备的「王树瑞佑」，停止伐木")
                return False
            self.ctx.sleep(250)

    def _refresh_between_rounds(self, cancelled: Callable[[], bool] | None) -> bool:
        if self._cancelled(cancelled):
            return False
        # ``reloginBetween`` was used by the first iOS implementation and is
        # kept as an explicit compatibility override for old scripts.
        if self.relogin_between is True or not self.use_wonderland_refresh:
            self.log("[AutoWood] 退出并重新登录以刷新小道具冷却")
            try:
                self.genshin.relogin()
                return True
            except Exception as error:
                self.log(f"[AutoWood] 重登刷新失败：{error}")
                return False

        self.log("[AutoWood] 进入并退出千星奇域以刷新小道具冷却")
        try:
            result = self.genshin.wonderlandCycle()
            if result is not False:
                return True
        except Exception as error:
            self.log(f"[AutoWood] 千星奇域刷新失败：{error}")
        if not self.refresh_fallback_to_relogin:
            return False
        self.log("[AutoWood] 千星奇域刷新失败，回退退出重登")
        try:
            self.genshin.relogin()
            return True
        except Exception as error:
            self.log(f"[AutoWood] 重登回退失败：{error}")
            return False

    def _record_reward(self, text: str) -> None:
        added = self.statistics.record(text)
        if not added:
            self.log(
                f"[AutoWood] 未识别到木材奖励（连续 {self.statistics.nothing_count} 次）"
            )
            return
        for name, quantity in added.items():
            self.log(
                f"[AutoWood] 木材{name}累计获取：{self.statistics.totals[name]}"
                f"（本轮 +{quantity}）"
            )

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        for rd in range(1, self.rounds + 1):
            if self._cancelled(cancelled):
                return False
            if self.wood_count_ocr_enabled:
                if self.statistics.always_empty:
                    self.log(
                        f"[AutoWood] 连续 {self.statistics.nothing_count} 次未识别木材，停止伐木"
                    )
                    return True
                if self.statistics.reached_max_count:
                    self.log(
                        f"[AutoWood] 已达到木材数量上限：{self.wood_daily_max_count}"
                    )
                    return True

            self.log(f"[AutoWood] 第 {rd}/{self.rounds} 轮")
            for _ in range(self.per_round_attacks):
                if self._cancelled(cancelled):
                    return False
                self.ctx.input.attack()
                self.ctx.sleep(600)
            if not self._press_gadget(cancelled):
                return False

            is_last = rd == self.rounds
            # Keep the desktop task's final-round semantics by default: the
            # final Z is allowed to finish without another refresh. Callers
            # that need final counters can opt into ocr_final_round.
            if is_last and not self.ocr_final_round:
                return True

            if self.wood_count_ocr_enabled:
                reward_text = self._wait_wood_reward(cancelled)
                self._record_reward(reward_text)
                if self.statistics.always_empty:
                    self.log(
                        "[AutoWood] 连续 OCR 为空，判定附近没有可伐树木或已达每日上限"
                    )
                    return True
                if self.statistics.reached_max_count:
                    self.log(
                        f"[AutoWood] 木材已达到设置上限：{self.wood_daily_max_count}"
                    )
                    return True

            if not self._refresh_between_rounds(cancelled):
                return False
            self.ctx.sleep(500)
        return True
