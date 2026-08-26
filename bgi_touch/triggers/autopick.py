"""自动拾取触发器（原版 AutoPick 的移动端适配）。

PC 版检测 "F" 键图标 + SVTR 文本识别；移动端拾取是屏幕右中部的物品列表按钮。
实现：OCR 交互按钮附近的条目文字，根据 BetterGI 的白/黑名单模式点按交互位。
"""

from __future__ import annotations

import time
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

from ..engine.context import GameContext
from ..engine.recognition import ImageRegion, Mat, RecognitionObject

# Synced from BetterGI.Assets.Other 1.0.24, used by current upstream BetterGI.
DEFAULT_LIST_ROOT = Path(__file__).resolve().parents[2] / "assets" / "config" / "pick"
DEFAULT_BLACKLIST_PATH = DEFAULT_LIST_ROOT / "default_pick_black_lists.json"
DEFAULT_WHITELIST_PATH = DEFAULT_LIST_ROOT / "default_pick_white_lists.json"
FALLBACK_BLACKLIST = frozenset({"对话", "进入", "传送", "离开", "阅读", "操作", "开启"})
USER_LIST_NAMES = {
    "blacklist": "pick_black_lists.txt",
    "fuzzy_blacklist": "pick_fuzzy_black_lists.txt",
    "blacklist_pick": "pick_white_lists.txt",
    "whitelist": "pick_whitelist_mode_pick_lists.txt",
    "whitelist_exclusions": "pick_whitelist_mode_do_not_pick_lists.txt",
}


def _load_default_list(path: Path) -> frozenset[str]:
    try:
        values = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return frozenset()
    return frozenset(_normalize_text(value) for value in values if _normalize_text(value))


def _load_text_list(path: Path) -> tuple[str, ...]:
    """Load one BetterGI ``User/pick_*.txt`` file.

    The desktop implementation treats each non-empty line as one literal
    entry.  Ignore comments and surrounding whitespace here because it makes
    the same files pleasant to maintain on a phone-hosted checkout while
    preserving the literal matching semantics for the actual entry.
    """
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        return ()
    result = []
    for line in lines:
        value = _normalize_text(line)
        if value and not value.startswith("#"):
            result.append(value)
    return tuple(result)


def _user_list_roots(ctx, explicit: str | Path | None = None) -> tuple[Path, ...]:
    """Return safe roots for BetterGI-compatible user pick lists.

    ``User`` is intentionally not hard-coded to the desktop source checkout:
    a converted script package can live elsewhere.  Callers may provide a
    root explicitly, ``GameContext`` may expose ``user_dir``/``user_root``,
    or ``BGI_USER_DIR`` can select a shared profile.  The repository-local
    ``User`` directory remains the final fallback.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for attribute in ("user_dir", "user_root"):
        value = getattr(ctx, attribute, None)
        if value:
            candidates.append(Path(value).expanduser())
    env_value = os.environ.get("BGI_USER_DIR")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    project_root = Path(__file__).resolve().parents[2]
    candidates.extend((project_root / "User", Path.cwd() / "User"))

    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return tuple(result)


def _user_list(ctx, name: str, explicit_root: str | Path | None = None) -> tuple[str, ...]:
    filename = USER_LIST_NAMES[name]
    for root in _user_list_roots(ctx, explicit_root):
        values = _load_text_list(root / filename)
        if values:
            return values
    return ()


@lru_cache(maxsize=1)
def _default_blacklist() -> frozenset[str]:
    return _load_default_list(DEFAULT_BLACKLIST_PATH) or FALLBACK_BLACKLIST


@lru_cache(maxsize=1)
def _default_whitelist() -> frozenset[str]:
    return _load_default_list(DEFAULT_WHITELIST_PATH)


def _normalize_text(value: object) -> str:
    return "".join(str(value or "").split()).strip()


def _process_ocr_text(value: object) -> str:
    """Normalize AutoPick OCR with BetterGI's edge-noise semantics.

    Mobile OCR often adds brackets or punctuation around an interaction
    label.  The desktop trigger converts square brackets to the game's corner
    quotes, trims non-Chinese noise at both edges, and repairs an unmatched
    quote before applying its lists.  Keep this separate from list-file
    normalization so configured names remain literal.
    """
    text = str(value or "")
    if not text:
        return ""
    text = (
        text.replace("【", "「")
        .replace("[", "「")
        .replace("】", "」")
        .replace("]", "」")
    )
    text = "".join(text.split())

    def is_edge_character(char: str, *, right: bool = False) -> bool:
        if char == "「":
            return not right
        if right and char in {"」", "！"}:
            return True
        # Match BetterGI's Chinese-character boundary rule while retaining
        # the corner quotes used to repair OCR output.
        return "\u4e00" <= char <= "\u9fff"

    start = 0
    end = len(text) - 1
    while start <= end and not is_edge_character(text[start]):
        start += 1
    while end >= start and not is_edge_character(text[end], right=True):
        end -= 1
    if start > end:
        return ""

    cleaned = text[start:end + 1]
    has_left = "「" in cleaned
    has_right = "」" in cleaned
    if has_left and not has_right:
        cleaned += "」"
    elif has_right and not has_left:
        cleaned = "「" + cleaned
    return cleaned


def _normalize_mode(value: object) -> str:
    mode = str(value or "Whitelist").strip().casefold()
    if mode in {"whitelist", "white", "白名单", "白名單"}:
        return "Whitelist"
    if mode in {"blacklist", "black", "黑名单", "黑名單"}:
        return "Blacklist"
    raise ValueError(f"不支持的 AutoPick 模式：{value}")


class AutoPickTrigger:
    name = "AutoPick"

    # BetterGI's AutoPickTrigger uses three fixed pixels in the interaction
    # list's scroll indicator.  Keep the samples in reference (1920x1080)
    # coordinates; the current frame is converted through ScreenTransform so
    # this also works on the iPhone's wider native screenshot.
    _SCROLL_ICON_SAMPLES = (
        (1062, 537, (44, 233, 255)),  # yellow indicator, BGR
        (1062, 524, (255, 255, 255)),
        (1062, 554, (255, 255, 255)),
    )
    _SCROLL_ICON_TOLERANCE = 24
    _SCROLL_ICON_NEIGHBOR_RADIUS = 1
    _SCROLL_COOLDOWN_S = 0.9

    def __init__(
        self,
        ctx: GameContext,
        blacklist: list[str] | None = None,
        whitelist: list[str] | None = None,
        text_list: list[str] | None = None,
        fuzzy_blacklist: list[str] | None = None,
        whitelist_exclusions: list[str] | None = None,
        blacklist_mode_pick_enabled: bool = False,
        whitelist_mode_do_not_pick_enabled: bool = True,
        mode: str = "Whitelist",
        log: Callable[[str], None] = print,
        force_interaction: bool = False,
        pick_key: str = "F",
        user_dir: str | Path | None = None,
        require_pick_prompt: bool = True,
    ):
        self.ctx = ctx
        self.enabled = True
        self.mode = _normalize_mode(mode)
        self.blacklist = set(_default_blacklist())
        self.blacklist.update(
            _normalize_text(value) for value in (blacklist or []) if _normalize_text(value)
        )
        self.blacklist.update(_user_list(ctx, "blacklist", user_dir))
        self.fuzzy_blacklist = tuple(
            _normalize_text(value) for value in (fuzzy_blacklist or []) if _normalize_text(value)
        )
        self.fuzzy_blacklist += tuple(
            value for value in _user_list(ctx, "fuzzy_blacklist", user_dir)
            if value
        )
        custom_whitelist = {
            _normalize_text(value) for value in (whitelist or []) if _normalize_text(value)
        }
        self.text_list = frozenset(
            _normalize_text(value) for value in (text_list or []) if _normalize_text(value)
        )
        selected_whitelist = (
            set(_default_whitelist())
            | set(_user_list(ctx, "whitelist", user_dir))
            | custom_whitelist
        )
        blacklist_pick_list = set(
            _user_list(ctx, "blacklist_pick", user_dir)
        ) | custom_whitelist
        exclusions = {
            _normalize_text(value) for value in (whitelist_exclusions or []) if _normalize_text(value)
        }
        exclusions.update(_user_list(ctx, "whitelist_exclusions", user_dir))
        if whitelist_mode_do_not_pick_enabled:
            selected_whitelist.difference_update(exclusions)
        self.whitelist = frozenset(selected_whitelist)
        self.blacklist_pick_list = frozenset(blacklist_pick_list)
        self.blacklist_mode_pick_enabled = bool(blacklist_mode_pick_enabled)
        self.log = log
        self.force_interaction = bool(force_interaction)
        self.pick_key = str(pick_key or "F").strip() or "F"
        self.require_pick_prompt = bool(require_pick_prompt)
        self._last_action_at = 0.0
        self._last_scroll_at = float("-inf")
        self._last_text = ""
        self._pick_prompt_ro = None
        self._chat_icon_ro = None
        self._settings_icon_ro = None
        self._l_key_ro = None
        # 交互列表出现在交互按钮右侧（ref 空间），OCR 该竖条区域
        self.roi = (1080, 380, 420, 320)

    def on_frame(self, region: ImageRegion) -> None:
        # A task may have paused the frame loop while this callback was already
        # holding a decoded frame.  Re-check the context gate before any scene
        # recognition so a map/menu label can never become an input edge during
        # the hand-off to teleport or another exclusive task.
        if bool(getattr(self.ctx, "input_exclusive", False)):
            return
        # 地图/菜单页也会在右侧出现地名。先确认左上角小地图仍在，
        # 避免把「优兰尼娅湖」等地图文字当成可拾取物点击。
        if not self._is_gameplay_frame(region):
            return
        prompt = self._find_pick_prompt(region)
        if prompt is None:
            # The interaction list can contain more entries than fit on the
            # screen.  In that state the game hides the interaction-key
            # prompt but leaves a small yellow/white scroll marker.  Reuse
            # this callback's frame, just like the desktop trigger, and do
            # not start another DeviceHub screenshot producer.
            if self._scroll_interaction_list(region):
                return
            if self.require_pick_prompt:
                return
        if prompt is not None and self._has_excluded_item_icon(region, prompt):
            return
        if self.force_interaction:
            self._press_interaction()
            return
        hits = region.find_multi(RecognitionObject.ocr(*self.roi), limit=5)
        # OCR engines do not promise reading order.  The mobile interaction
        # list is vertical, so process the top-most candidate first and keep
        # the input edge deterministic when several drops are visible.
        hits = sorted(hits, key=lambda hit: (float(hit.y), float(hit.x)))
        for h in hits:
            text = _process_ocr_text(h.text)
            if not self._should_pick(text):
                continue
            now = time.monotonic()
            if text == self._last_text and now - self._last_action_at < 1.2:
                continue
            # 命中可拾取物：复用 BetterGI 的拾取/交互键。DeviceHub profile
            # 会把该键解析为 KeyF（或用户配置的自定义键），避免直接 tap
            # OCR 文字区域而绕过 profile 会话。
            self.log(f"[AutoPick] 拾取: {text}")
            if not self._press_interaction():
                return
            self._last_text = text
            self._last_action_at = now
            return

    @staticmethod
    def _template(path: Path, *, roi=None, threshold: float = 0.72):
        if not path.is_file():
            return None
        try:
            ro = RecognitionObject.template_match(
                Mat.from_file(str(path)),
                *(roi or ()),
            )
            ro.threshold = threshold
            return ro
        except (OSError, ValueError, TypeError):
            return None

    def _find_pick_prompt(self, region):
        """Return the visible interaction-key prompt, if any.

        Upstream AutoPick never runs its OCR branch without seeing the F/L/G
        interaction marker first.  On a lightweight test double without a
        ``find`` method we retain the old permissive behavior; real
        ``ImageRegion`` instances always have the guard.
        """
        find = getattr(region, "find", None)
        if not callable(find):
            return True
        if self._pick_prompt_ro is None:
            self._pick_prompt_ro = self._template(
                Path(__file__).resolve().parents[2] / "assets" / "templates" / "autopick" / "F.png",
                roi=(1090, 330, 60, 420),
            )
        if self._pick_prompt_ro is None:
            return None
        hit = find(self._pick_prompt_ro)
        if hit is None:
            return None
        try:
            return hit if hit.is_exist() else None
        except AttributeError:
            return hit

    def _has_excluded_item_icon(self, region, prompt) -> bool:
        """Reject chat/settings/L prompts before OCR can click their labels."""
        find = getattr(region, "find", None)
        if not callable(find) or prompt is True:
            return False
        prompt_x = float(getattr(prompt, "x", 1090.0))
        prompt_y = float(getattr(prompt, "y", 330.0))
        prompt_h = float(getattr(prompt, "height", 32.0))
        try:
            icon_region = region.derive_crop(
                prompt_x + 42,
                max(0.0, prompt_y - 8),
                92,
                max(45.0, prompt_h + 16),
            )
        except (AttributeError, TypeError, ValueError):
            return False

        if self._chat_icon_ro is None:
            self._chat_icon_ro = self._template(
                Path(__file__).resolve().parents[2] / "assets" / "templates" / "autoskip" / "icon_option.png",
                threshold=0.78,
            )
        if self._settings_icon_ro is None:
            self._settings_icon_ro = self._template(
                Path(__file__).resolve().parents[2] / "assets" / "templates" / "autopick" / "icon_settings.png",
                threshold=0.78,
            )
        if self._l_key_ro is None:
            self._l_key_ro = self._template(
                Path(__file__).resolve().parents[2] / "assets" / "templates" / "autopick" / "L.png",
                threshold=0.78,
            )
        for ro in (self._chat_icon_ro, self._settings_icon_ro, self._l_key_ro):
            if ro is None:
                continue
            hit = icon_region.find(ro)
            try:
                if hit.is_exist():
                    return True
            except AttributeError:
                if hit:
                    return True
        return False

    def _should_pick(self, text: str) -> bool:
        """Apply BetterGI's exact whitelist and exact/fuzzy blacklist semantics."""
        text = _normalize_text(text)
        if len(text) <= 1 or self._always_excluded(text):
            return False
        # RealtimeTimer("AutoPick", { TextList: [...] }) is the upstream
        # script-level escape hatch for dialogue/action labels that are not
        # part of the persistent pick lists.  An explicit list is exclusive.
        if getattr(self, "text_list", ()):
            return text in self.text_list
        if self.mode == "Whitelist":
            return text in self.whitelist
        if self.blacklist_mode_pick_enabled and text in self.blacklist_pick_list:
            return True
        if text in self.blacklist:
            return False
        return not any(value in text for value in self.fuzzy_blacklist)

    @staticmethod
    def _always_excluded(text: str) -> bool:
        # Keep the dynamic interaction guards from upstream AutoPickTrigger.
        if "长时间" in text or "聚所" in text:
            return True
        if "霜月" in text and "坊" in text:
            return True
        if "叮铃" in text or "眶螂" in text:
            return True
        if "蛋卷" in text and "坊" in text:
            return True
        if any(value in text for value in ("西风成垒", "望崖营壁", "魔女的花园")):
            return True
        if "月谕圣牌" in text:
            return True
        return "我在" in text and any(
            value in text for value in ("声望", "回声", "悬木人", "流泉")
        )

    def _press_interaction(self) -> bool:
        if bool(getattr(self.ctx, "input_exclusive", False)):
            return False
        now = time.monotonic()
        if now - self._last_action_at < 0.8:
            return False
        self.ctx.input.key_press(getattr(self, "pick_key", "F"))
        self.log("[AutoPick] 直接交互")
        self._last_text = ""
        self._last_action_at = now
        return True

    def _scroll_interaction_list(self, region) -> bool:
        """Scroll a visible mobile interaction list when its prompt is hidden.

        BetterGI samples exact BGR pixels at the centre of the list marker.
        DeviceHub screenshots may resample or shift a marker by a pixel, so
        allow a small neighbourhood and bounded channel tolerance while
        requiring all three independent samples.  The cooldown is important
        on the mobile frame cadence: without it, one unchanged frame can emit
        repeated swipes before the game has rendered the next list page.
        """
        vertical_scroll = getattr(getattr(self.ctx, "input", None), "vertical_scroll", None)
        if not callable(vertical_scroll):
            return False
        if bool(getattr(self.ctx, "input_exclusive", False)):
            return False
        now = time.monotonic()
        if now - self._last_scroll_at < self._SCROLL_COOLDOWN_S:
            return False

        frame = getattr(region, "bgr", None)
        if frame is None:
            src_mat = getattr(region, "src_mat", None)
            frame = getattr(src_mat, "bgr", None)
        if frame is None:
            return False
        try:
            height, width = frame.shape[:2]
        except (AttributeError, TypeError, ValueError):
            return False
        if frame.ndim < 3 or frame.shape[2] < 3:
            return False

        transform = getattr(self.ctx, "transform", None)
        if transform is None or not callable(getattr(transform, "to_device", None)):
            return False

        radius = self._SCROLL_ICON_NEIGHBOR_RADIUS
        tolerance = self._SCROLL_ICON_TOLERANCE
        for ref_x, ref_y, expected in self._SCROLL_ICON_SAMPLES:
            try:
                device_x, device_y = transform.to_device(ref_x, ref_y)
                center_x = int(round(device_x))
                center_y = int(round(device_y))
            except (TypeError, ValueError):
                return False
            left = max(0, center_x - radius)
            right = min(width, center_x + radius + 1)
            top = max(0, center_y - radius)
            bottom = min(height, center_y + radius + 1)
            if left >= right or top >= bottom:
                return False
            window = frame[top:bottom, left:right, :3].astype("int16", copy=False)
            target = window - expected
            if not ((abs(target) <= tolerance).all(axis=2)).any():
                return False

        # Re-check ownership immediately before emitting the input edge; a
        # map/menu task can acquire exclusivity while recognition is running.
        if bool(getattr(self.ctx, "input_exclusive", False)):
            return False
        vertical_scroll(2)
        self._last_scroll_at = now
        self.log("[AutoPick] 滚动交互列表")
        return True

    def _is_gameplay_frame(self, region: ImageRegion) -> bool:
        """Require gameplay HUD and explicitly reject the big-map overlay.

        The mobile map is translucent: the Paimon HUD template can remain
        visible behind it, while the map's OCR labels occupy the same right
        side ROI as interaction entries.  Checking only ``is_main_ui`` can
        therefore turn labels such as ``优兰尼娅湖`` into pickup actions during
        ``genshin.tp``.  The map-close marker is the shared scene boundary
        used by AutoSkip, MapMask, and QuickTeleport, so keep AutoPick on the
        same guard before either OCR or forced interaction.
        """
        from ..vision.game_ui import is_main_ui

        # is_main_ui already performs the big-map rejection.  Calling
        # is_big_map_ui a second time doubled template work on every shared
        # trigger frame and made the WebUI/trigger loop compete for CPU while
        # the screenshot producer was active.
        return is_main_ui(self.ctx, region.bgr)
