"""BetterGI ``GoToCraftingBenchTask`` 的 iOS 触控实现。

合成台是一个公共 Job，不能简化成“走到终点后按一次 F”：原项目会在
交互失败时重复按键并后退一步，还要选择对话中的最后一个选项才会进入
合成页。本模块把这些步骤和浓缩树脂数量状态机放在同一个输入所有权范围
内，避免 AutoPick/AutoSkip 在过渡帧上抢先点按。

二进制模板优先使用 iOS 仓库已有的通用确认按钮；合成树脂专用模板则按
原项目 checkout 动态回退。这样开发机可以直接复用上游资产，同时不会把
本机绝对路径或桌面资源复制进提交。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..engine.context import GameContext
from ..engine.recognition import ImageRegion, Mat, RecognitionObject
from ..pathing.executor import PathingExecutor
from ..pathing.model import PathingTask
from .common_jobs import InteractionPromptDetector, exclusive_realtime_triggers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = PROJECT_ROOT.parent / "better-genshin-impact"
COMMON_ASSETS = PROJECT_ROOT / "assets" / "templates"

# 1920x1080 reference coordinates.  These regions are deliberately broad:
# iPhone safe-area layouts move the buttons horizontally while the vertical
# placement remains stable after ScreenTransform scaling.
TALK_OPTION_ROI = (1130, 260, 760, 760)
CRAFT_SCREEN_ROI = (600, 80, 1100, 900)
RESIN_ENTRY_ROI = (0, 120, 1200, 840)
RESIN_COUNT_ROI = (500, 120, 1350, 720)
CONFIRM_ROI = (480, 560, 960, 430)

CRAFTING_MARKERS = (
    "合成浓缩树脂", "合成", "Craft Condensed Resin", "Condensed Resin",
)
CONDENSED_RESIN_MARKERS = ("浓缩树脂", "濃縮樹脂", "Condensed Resin")
ORIGINAL_RESIN_MARKERS = ("原粹树脂", "原粹樹脂", "Original Resin")
CONFIRM_MARKERS = ("确认", "確定", "Confirm", "OK")
CRAFT_CONFIRM_MARKERS = ("合成", "制作", "製作", "Craft")
TALK_EXCLUDE_MARKERS = (
    "返回", "退出", "取消", "关闭", "關閉", "Confirm", "确认", "確定",
)
FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def _compact(value: Any) -> str:
    return (
        str(value or "")
        .translate(FULLWIDTH_DIGITS)
        .replace(" ", "")
        .replace("\u3000", "")
        .replace("\n", "")
        .replace("\r", "")
        .strip()
    )


def _folded(value: Any) -> str:
    return _compact(value).casefold()


def _contains(value: Any, markers: Iterable[str]) -> bool:
    text = _folded(value)
    return any(_folded(marker) in text for marker in markers)


def _is_cancelled(cancelled: Callable[[], bool] | None) -> bool:
    try:
        return bool(cancelled and cancelled())
    except Exception:
        # Cancellation cleanup is safer when a disappearing host token is
        # treated as cancelled rather than allowing held input to continue.
        return True


def _number_after_marker(text: str, markers: Iterable[str]) -> int | None:
    """Extract a resin count from one OCR line containing a named item."""

    compact = _compact(text)
    folded = compact.casefold()
    marker_end = None
    for marker in markers:
        wanted = _compact(marker)
        index = folded.find(wanted.casefold())
        if index >= 0:
            marker_end = max(marker_end or 0, index + len(wanted))
    if marker_end is None:
        return None
    suffix = compact[marker_end:]
    # OCR may turn a slash into ``|`` or ``I``.  The first side is the useful
    # current count for both ``160/200`` and ``2/5``.
    fraction = re.search(r"(\d+)\s*[/|丨Il]\s*(\d+)", suffix)
    if fraction:
        return int(fraction.group(1))
    values = [int(value) for value in re.findall(r"\d+", suffix)]
    return values[0] if values else None


def _next_count_line(lines: list[str], index: int, markers: Iterable[str]) -> str:
    """Join a nearby OCR line when the label and number were split apart."""

    for candidate in lines[index + 1:index + 3]:
        if _contains(candidate, markers):
            break
        if re.search(r"\d", _compact(candidate)):
            return candidate
    return ""


def parse_resin_inventory(lines: Iterable[str]) -> tuple[int | None, int | None]:
    """Parse ``(original_resin, condensed_resin)`` from OCR text lines.

    The upstream UI can return either one combined line (``原粹树脂 160/200``)
    or separate label/count boxes.  Keeping this parser pure makes both forms
    testable without a device or OCR runtime.
    """

    values = [_compact(line) for line in lines if _compact(line)]
    original: int | None = None
    condensed: int | None = None
    for index, line in enumerate(values):
        if original is None and _contains(line, ORIGINAL_RESIN_MARKERS):
            original = _number_after_marker(line, ORIGINAL_RESIN_MARKERS)
            if original is None:
                original = _number_after_marker(
                    _next_count_line(values, index, CONDENSED_RESIN_MARKERS),
                    (),
                )
        if condensed is None and _contains(line, CONDENSED_RESIN_MARKERS):
            condensed = _number_after_marker(line, CONDENSED_RESIN_MARKERS)
            if condensed is None:
                condensed = _number_after_marker(
                    _next_count_line(values, index, ORIGINAL_RESIN_MARKERS),
                    (),
                )

    # A split count line has no marker, so the helper above needs a small
    # fallback that does not require a marker at all.
    if original is None or condensed is None:
        for index, line in enumerate(values):
            if not re.search(r"\d", line):
                continue
            if original is None and index and _contains(values[index - 1], ORIGINAL_RESIN_MARKERS):
                original = _first_fraction_or_number(line, upper_bound=200)
            if condensed is None and index and _contains(values[index - 1], CONDENSED_RESIN_MARKERS):
                condensed = _first_fraction_or_number(line, upper_bound=5)

    if original is not None and original < 0:
        original = None
    if condensed is not None and not 0 <= condensed <= 5:
        condensed = None
    return original, condensed


def _first_fraction_or_number(text: str, *, upper_bound: int | None = None) -> int | None:
    compact = _compact(text)
    fraction = re.search(r"(\d+)\s*[/|丨Il]\s*(\d+)", compact)
    candidate = int(fraction.group(1)) if fraction else None
    if candidate is None:
        values = [int(value) for value in re.findall(r"\d+", compact)]
        candidate = values[0] if values else None
    if candidate is None:
        return None
    return candidate if upper_bound is None or candidate <= upper_bound else None


def calculate_condensed_resin_crafts(
    original_resin: int,
    condensed_resin: int,
    min_resin_to_keep: int = 0,
    *,
    resin_per_craft: int = 60,
    condensed_capacity: int = 5,
) -> int:
    """Calculate the safe number of condensed-resin crafts.

    This is the same policy used by the current desktop common Job: preserve
    the configured amount of Original Resin, never exceed the five-item
    condensed-resin inventory limit, and never return a negative quantity.
    """

    original = max(0, int(original_resin))
    condensed = max(0, min(int(condensed_resin), int(condensed_capacity)))
    keep = max(0, int(min_resin_to_keep))
    per_craft = max(1, int(resin_per_craft))
    capacity = max(0, int(condensed_capacity) - condensed)
    available = max(0, original - keep)
    return min(capacity, available // per_craft)


@dataclass(frozen=True)
class CraftingBenchConfig:
    country: str
    min_resin_to_keep: int = 0
    timeout_s: float = 180.0
    attempts: int = 2

    def __post_init__(self) -> None:
        if not str(self.country or "").strip():
            raise ValueError("前往合成台需要 country")
        object.__setattr__(self, "min_resin_to_keep", max(0, int(self.min_resin_to_keep)))
        object.__setattr__(self, "timeout_s", max(15.0, min(600.0, float(self.timeout_s))))
        object.__setattr__(self, "attempts", max(1, min(3, int(self.attempts))))


class CraftingBenchTask:
    """Navigate to a crafting bench and optionally craft condensed resin."""

    def __init__(
        self,
        ctx: GameContext,
        country: str,
        *,
        min_resin_to_keep: int = 0,
        timeout_s: float = 180.0,
        party_slots: Mapping[str, int] | None = None,
        route_resolver: Callable[[str, str], Path] | None = None,
        talk_detector: Callable[[Any], bool] | None = None,
        return_main_ui: Callable[[], bool] | None = None,
        log: Callable[[str], None] = print,
        interaction_detector: InteractionPromptDetector | None = None,
    ):
        self.ctx = ctx
        self.config = CraftingBenchConfig(
            str(country or "").strip(), min_resin_to_keep, timeout_s,
        )
        self.party_slots = dict(party_slots or {})
        self.route_resolver = route_resolver or self._default_route
        self.talk_detector = talk_detector
        self.return_main_ui = return_main_ui
        self.log = log
        self.interaction = interaction_detector or InteractionPromptDetector(ctx, log=log)
        self._templates: dict[str, RecognitionObject | None] = {}

    def _log(self, message: str) -> None:
        self.log(f"[CraftingBench] {message}")

    @staticmethod
    def _default_route(kind: str, country: str) -> Path:
        aliases = {
            "mondstadt": "蒙德", "liyue": "璃月", "inazuma": "稻妻",
            "sumeru": "须弥", "fontaine": "枫丹", "natlan": "纳塔",
            "nod-krai": "挪德卡莱", "nodkrai": "挪德卡莱",
        }
        key = aliases.get(country.strip().casefold(), country.strip())
        candidates = (
            PROJECT_ROOT / "assets" / "pathing" / "poi" / f"{kind}_{key}.json",
            UPSTREAM_ROOT / "BetterGenshinImpact" / "GameTask" / "Common" /
            "Element" / "Assets" / "Json" / f"{kind}_{key}.json",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"未找到 {country} 的 {kind} 路线")

    @staticmethod
    def _fontaine(country: str) -> bool:
        return str(country or "").strip().casefold() in {"枫丹", "fontaine"}

    @staticmethod
    def _group_ocr_lines(hits: Iterable[Any]) -> list[str]:
        """Keep OCR reading order while joining boxes on the same visual line."""

        ordered = sorted(
            (hit for hit in hits if _compact(getattr(hit, "text", ""))),
            key=lambda hit: (float(getattr(hit, "y", 0)), float(getattr(hit, "x", 0))),
        )
        lines: list[dict[str, Any]] = []
        for hit in ordered:
            y = float(getattr(hit, "y", 0))
            height = max(1.0, float(getattr(hit, "height", 0)))
            target = next(
                (line for line in reversed(lines)
                 if abs(y - line["y"]) <= max(18.0, height * 0.8)),
                None,
            )
            if target is None:
                lines.append({"y": y, "items": [hit]})
            else:
                target["items"].append(hit)
        return [
            "".join(_compact(getattr(hit, "text", "")) for hit in sorted(
                line["items"], key=lambda item: float(getattr(item, "x", 0))
            ))
            for line in lines
        ]

    @staticmethod
    def _find_marker(hits: Iterable[Any], markers: Iterable[str]):
        return next(
            (hit for hit in hits if _contains(getattr(hit, "text", ""), markers)),
            None,
        )

    def _ocr(self, frame: Any, roi: tuple[int, int, int, int]):
        region = ImageRegion(self.ctx, frame)
        hits = region.find_multi(RecognitionObject.ocr(*roi), limit=80)
        return region, hits

    def _is_talk(self, frame: Any) -> bool:
        if self.talk_detector is not None:
            try:
                return bool(self.talk_detector(frame))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return False
        # A lightweight host may not provide the private GenshinApi detector;
        # OCR still gives a useful fallback for test doubles and translations.
        try:
            _region, hits = self._ocr(frame, TALK_OPTION_ROI)
            return bool(hits)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
    def _template_path(self, name: str) -> Path | None:
        candidates: dict[str, tuple[Path, ...]] = {
            "white_confirm": (
                COMMON_ASSETS / "stygian" / "white_confirm.png",
                COMMON_ASSETS / "artifact_salvage" / "btn_white_confirm.png",
                UPSTREAM_ROOT / "BetterGenshinImpact" / "GameTask" / "Common" /
                "Element" / "Assets" / "1920x1080" / "btn_white_confirm.png",
            ),
            "black_confirm": (
                COMMON_ASSETS / "stygian" / "black_confirm.png",
                COMMON_ASSETS / "artifact_salvage" / "btn_black_confirm.png",
                UPSTREAM_ROOT / "BetterGenshinImpact" / "GameTask" / "Common" /
                "Element" / "Assets" / "1920x1080" / "btn_black_confirm.png",
            ),
            "condensed_resin": (
                COMMON_ASSETS / "crafting" / "craft_condensed_resin.png",
                UPSTREAM_ROOT / "BetterGenshinImpact" / "GameTask" / "Common" /
                "Element" / "Assets" / "1920x1080" / "craft_condensed_resin.png",
            ),
        }
        return next((path for path in candidates.get(name, ()) if path.is_file()), None)

    def _template(self, name: str, *, roi: tuple[int, int, int, int] | None = None):
        if name in self._templates:
            return self._templates[name]
        path = self._template_path(name)
        if path is None:
            self._templates[name] = None
            return None
        try:
            ro = RecognitionObject.template_match(Mat.from_file(str(path)), *(roi or ()))
            ro.threshold = 0.68 if name == "condensed_resin" else 0.72
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._log(f"模板 {name} 不可用：{error}")
            ro = None
        self._templates[name] = ro
        return ro

    def _click_ref(self, x: float, y: float) -> None:
        click = getattr(self.ctx.input, "click_ref", None)
        if callable(click):
            click(x, y)
            return
        transform = self.ctx.transform
        dx, dy = transform.to_device(x, y)
        self.ctx.device.tap(
            dx, dy,
            image_width=transform.device_width,
            image_height=transform.device_height,
        )

    def _route_once(self) -> bool:
        route_path = self.route_resolver("合成台", self.config.country)
        route = PathingTask.load(route_path)
        # No route action is allowed to press F on the final transition frame.
        # The public Job owns that interaction and keeps the full UI transition
        # inside exclusive_realtime_triggers below.
        route.realtime_triggers = {}
        party_config = {
            "enabled": True,
            "autoSkipEnabled": True,
            "autoRunEnabled": not self._fontaine(self.config.country),
        }
        executor = PathingExecutor(
            self.ctx,
            party_slots=self.party_slots,
            log=self.log,
            pathing_config=party_config,
        )
        return bool(executor.run(route))

    def _press_interaction_until_talk(
        self,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        # The upstream retry chain has an initial F, another F, then a short
        # backward step and a final F. Keep exactly that order.
        for attempt in range(3):
            if _is_cancelled(cancelled) or time.monotonic() >= deadline:
                return False
            frame = self.ctx.capture_bgr()
            if self._is_talk(frame):
                return True
            try:
                prompt_visible = self.interaction.visible(ImageRegion(self.ctx, frame))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                prompt_visible = False
            if attempt == 2:
                self.ctx.input.key_down("S")
                self.ctx.sleep(200)
                self.ctx.input.key_up("S")
            self.ctx.input.key_press("F")
            self._log(
                "检测到合成台交互" if prompt_visible else f"尝试与合成台交互（{attempt + 1}/3）"
            )
            self.ctx.sleep(1000)
        try:
            return self._is_talk(self.ctx.capture_bgr())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False

    def _is_crafting_screen(self, frame: Any) -> bool:
        region, hits = self._ocr(frame, CRAFT_SCREEN_ROI)
        if self._find_marker(hits, CONDENSED_RESIN_MARKERS) is not None:
            return True
        if self._find_marker(hits, ORIGINAL_RESIN_MARKERS) is not None:
            return True
        confirm = self._template("white_confirm", roi=CONFIRM_ROI)
        return bool(confirm is not None and region.find(confirm).is_exist())

    def _select_last_talk_option(
        self,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        while time.monotonic() < deadline:
            if _is_cancelled(cancelled):
                return False
            frame = self.ctx.capture_bgr()
            if self._is_crafting_screen(frame):
                return True
            region, hits = self._ocr(frame, TALK_OPTION_ROI)
            candidates = [
                hit for hit in hits
                if not _contains(getattr(hit, "text", ""), TALK_EXCLUDE_MARKERS)
                and _compact(getattr(hit, "text", ""))
            ]
            if candidates:
                # Dialogue options are vertically stacked on the right. The
                # last option is the bottom-most OCR hit; x breaks ties.
                option = max(
                    candidates,
                    key=lambda hit: (
                        float(getattr(hit, "y", 0)) + float(getattr(hit, "height", 0)),
                        float(getattr(hit, "x", 0)),
                    ),
                )
                option.click()
                self._log(f"选择合成台最后对话选项：{getattr(option, 'text', '')}")
                self.ctx.sleep(800)
            elif self._is_talk(frame):
                # Text-only dialogue transitions sometimes expose no OCR box;
                # advance once and let the next frame decide.
                self.ctx.input.key_press("SPACE")
                self.ctx.sleep(500)
            else:
                self.ctx.sleep(300)
        return False

    def _go_to_once(
        self,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        if not self._route_once():
            return False
        self.ctx.sleep(700)
        if not self._press_interaction_until_talk(deadline, cancelled):
            raise RuntimeError("未进入和合成台交互对话界面")
        if not self._select_last_talk_option(deadline, cancelled):
            raise RuntimeError("未进入合成界面或未能选择最后对话选项")
        self.ctx.sleep(800)
        return True

    def go_to_crafting_bench(
        self,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        with exclusive_realtime_triggers(self.ctx):
            for attempt in range(self.config.attempts):
                if _is_cancelled(cancelled):
                    return False
                try:
                    ok = self._go_to_once(
                        time.monotonic() + self.config.timeout_s,
                        cancelled,
                    )
                    if ok:
                        self._log("已进入合成界面")
                        return True
                except Exception as error:
                    self._log(f"前往合成台失败（{attempt + 1}/{self.config.attempts}）：{error}")
                    self.ctx.input.release_all()
                    if attempt + 1 < self.config.attempts:
                        self.ctx.sleep(1000)
            return False

    def _read_resin_counts(self, frame: Any) -> tuple[int | None, int | None]:
        _region, hits = self._ocr(frame, RESIN_COUNT_ROI)
        return parse_resin_inventory(self._group_ocr_lines(hits))

    def _click_condensed_resin(
        self,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        while time.monotonic() < deadline:
            if _is_cancelled(cancelled):
                return False
            frame = self.ctx.capture_bgr()
            region, hits = self._ocr(frame, RESIN_ENTRY_ROI)
            marker = self._find_marker(hits, CONDENSED_RESIN_MARKERS)
            if marker is not None:
                marker.click()
                self._log("打开浓缩树脂合成项")
                self.ctx.sleep(700)
                return True
            template = self._template("condensed_resin")
            if template is not None:
                hit = region.find(template)
                if hit.is_exist():
                    hit.click()
                    self._log("通过模板打开浓缩树脂合成项")
                    self.ctx.sleep(700)
                    return True
            self.ctx.sleep(350)
        return False

    def _wait_resin_counts(
        self,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[int, int] | None:
        while time.monotonic() < deadline:
            if _is_cancelled(cancelled):
                return None
            frame = self.ctx.capture_bgr()
            original, condensed = self._read_resin_counts(frame)
            if original is not None and condensed is not None:
                return original, condensed
            self.ctx.sleep(300)
        return None

    def _set_resin_quantity(self, quantity: int) -> None:
        # The upstream Bv helper resets the quantity to one with five safe
        # decrease clicks, then adds the requested number minus one. Both
        # controls are fixed reference-space positions in the common dialog.
        for _ in range(5):
            self._click_ref(1074, 672)
            self.ctx.sleep(120)
        for _ in range(max(0, quantity - 1)):
            self._click_ref(1614, 672)
            self.ctx.sleep(120)
        self._log(f"设置浓缩树脂合成次数：{quantity}")

    def _wait_click_confirm(
        self,
        markers: Iterable[str],
        template_name: str,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        while time.monotonic() < deadline:
            if _is_cancelled(cancelled):
                return False
            frame = self.ctx.capture_bgr()
            region, hits = self._ocr(frame, CONFIRM_ROI)
            template = self._template(template_name, roi=CONFIRM_ROI)
            hit = region.find(template) if template is not None else None
            if hit is not None and hit.is_exist():
                hit.click()
                self.ctx.sleep(500)
                return True
            marker = self._find_marker(hits, markers)
            if marker is not None:
                marker.click()
                self.ctx.sleep(500)
                return True
            self.ctx.sleep(300)
        return False

    def _leave_crafting_ui(self) -> bool:
        try:
            self.ctx.input.key_press("ESCAPE")
            self.ctx.sleep(1300)
        finally:
            if self.return_main_ui is not None:
                try:
                    return bool(self.return_main_ui())
                except Exception as error:
                    self._log(f"返回主界面失败：{error}")
                    return False
        return True

    def _craft_resin_once(
        self,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        if not self._click_condensed_resin(deadline, cancelled):
            raise RuntimeError("未找到浓缩树脂合成项")
        counts = self._wait_resin_counts(deadline, cancelled)
        if counts is None:
            raise RuntimeError("未能识别原粹树脂或浓缩树脂数量")
        original, condensed = counts
        crafts = calculate_condensed_resin_crafts(
            original,
            condensed,
            self.config.min_resin_to_keep,
        )
        self._log(
            f"原粹树脂 {original}，浓缩树脂 {condensed}，"
            f"保留 {self.config.min_resin_to_keep}，计划合成 {crafts} 次"
        )
        if crafts <= 0:
            self._log("无需合成浓缩树脂")
            return self._leave_crafting_ui()
        self._set_resin_quantity(crafts)
        if not self._wait_click_confirm(
            CRAFT_CONFIRM_MARKERS, "white_confirm", deadline, cancelled,
        ):
            raise RuntimeError("未找到合成白色确认按钮")
        if not self._wait_click_confirm(
            CONFIRM_MARKERS, "black_confirm", deadline, cancelled,
        ):
            raise RuntimeError("未找到合成黑色确认按钮")
        return self._leave_crafting_ui()

    def craft_resin(self, cancelled: Callable[[], bool] | None = None) -> bool:
        with exclusive_realtime_triggers(self.ctx):
            for attempt in range(self.config.attempts):
                if _is_cancelled(cancelled):
                    return False
                try:
                    if not self._go_to_once(
                        time.monotonic() + self.config.timeout_s,
                        cancelled,
                    ):
                        return False
                    return self._craft_resin_once(
                        time.monotonic() + self.config.timeout_s,
                        cancelled,
                    )
                except Exception as error:
                    self._log(f"合成浓缩树脂失败（{attempt + 1}/{self.config.attempts}）：{error}")
                    self.ctx.input.release_all()
                    try:
                        self.ctx.input.key_press("ESCAPE")
                    except Exception:
                        pass
                    if attempt + 1 < self.config.attempts:
                        self.ctx.sleep(1000)
            return False
