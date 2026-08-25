"""自动拾取触发器（原版 AutoPick 的移动端适配）。

PC 版检测 "F" 键图标 + SVTR 文本识别；移动端拾取是屏幕右中部的物品列表按钮。
实现：OCR 交互按钮附近的条目文字，根据 BetterGI 的白/黑名单模式点按交互位。
"""

from __future__ import annotations

import time
import json
from functools import lru_cache
from pathlib import Path
from typing import Callable

from ..engine.context import GameContext
from ..engine.recognition import ImageRegion, RecognitionObject

# Synced from BetterGI.Assets.Other 1.0.24, used by current upstream BetterGI.
DEFAULT_LIST_ROOT = Path(__file__).resolve().parents[2] / "assets" / "config" / "pick"
DEFAULT_BLACKLIST_PATH = DEFAULT_LIST_ROOT / "default_pick_black_lists.json"
DEFAULT_WHITELIST_PATH = DEFAULT_LIST_ROOT / "default_pick_white_lists.json"
FALLBACK_BLACKLIST = frozenset({"对话", "进入", "传送", "离开", "阅读", "操作", "开启"})


def _load_default_list(path: Path) -> frozenset[str]:
    try:
        values = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return frozenset()
    return frozenset(_normalize_text(value) for value in values if _normalize_text(value))


@lru_cache(maxsize=1)
def _default_blacklist() -> frozenset[str]:
    return _load_default_list(DEFAULT_BLACKLIST_PATH) or FALLBACK_BLACKLIST


@lru_cache(maxsize=1)
def _default_whitelist() -> frozenset[str]:
    return _load_default_list(DEFAULT_WHITELIST_PATH)


def _normalize_text(value: object) -> str:
    return "".join(str(value or "").split()).strip()


def _normalize_mode(value: object) -> str:
    mode = str(value or "Whitelist").strip().casefold()
    if mode in {"whitelist", "white", "白名单", "白名單"}:
        return "Whitelist"
    if mode in {"blacklist", "black", "黑名单", "黑名單"}:
        return "Blacklist"
    raise ValueError(f"不支持的 AutoPick 模式：{value}")


class AutoPickTrigger:
    name = "AutoPick"

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
    ):
        self.ctx = ctx
        self.enabled = True
        self.mode = _normalize_mode(mode)
        self.blacklist = set(_default_blacklist())
        self.blacklist.update(
            _normalize_text(value) for value in (blacklist or []) if _normalize_text(value)
        )
        self.fuzzy_blacklist = tuple(
            _normalize_text(value) for value in (fuzzy_blacklist or []) if _normalize_text(value)
        )
        custom_whitelist = {
            _normalize_text(value) for value in (whitelist or []) if _normalize_text(value)
        }
        self.text_list = frozenset(
            _normalize_text(value) for value in (text_list or []) if _normalize_text(value)
        )
        selected_whitelist = set(_default_whitelist()) | custom_whitelist
        exclusions = {
            _normalize_text(value) for value in (whitelist_exclusions or []) if _normalize_text(value)
        }
        if whitelist_mode_do_not_pick_enabled:
            selected_whitelist.difference_update(exclusions)
        self.whitelist = frozenset(selected_whitelist)
        self.blacklist_pick_list = frozenset(custom_whitelist)
        self.blacklist_mode_pick_enabled = bool(blacklist_mode_pick_enabled)
        self.log = log
        self.force_interaction = bool(force_interaction)
        self.pick_key = str(pick_key or "F").strip() or "F"
        self._last_action_at = 0.0
        self._last_text = ""
        # 交互列表出现在交互按钮右侧（ref 空间），OCR 该竖条区域
        self.roi = (1080, 380, 420, 320)

    def on_frame(self, region: ImageRegion) -> None:
        # 地图/菜单页也会在右侧出现地名。先确认左上角小地图仍在，
        # 避免把「优兰尼娅湖」等地图文字当成可拾取物点击。
        if not self._is_gameplay_frame(region):
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
            text = _normalize_text(h.text)
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
        now = time.monotonic()
        if now - self._last_action_at < 0.8:
            return False
        self.ctx.input.key_press(getattr(self, "pick_key", "F"))
        self.log("[AutoPick] 直接交互")
        self._last_text = ""
        self._last_action_at = now
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
        from ..vision.game_ui import is_big_map_ui, is_main_ui

        return is_main_ui(self.ctx, region.bgr) and not is_big_map_ui(
            self.ctx, region.bgr
        )
