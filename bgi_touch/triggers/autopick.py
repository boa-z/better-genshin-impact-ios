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
        fuzzy_blacklist: list[str] | None = None,
        whitelist_exclusions: list[str] | None = None,
        blacklist_mode_pick_enabled: bool = False,
        whitelist_mode_do_not_pick_enabled: bool = True,
        mode: str = "Whitelist",
        log: Callable[[str], None] = print,
        force_interaction: bool = False,
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
        for h in hits:
            text = _normalize_text(h.text)
            if not self._should_pick(text):
                continue
            now = time.monotonic()
            if text == self._last_text and now - self._last_action_at < 1.2:
                continue
            # 命中可拾取物：直接点该条目（移动端点条目即拾取）
            self.log(f"[AutoPick] 拾取: {text}")
            h.click()
            self._last_text = text
            self._last_action_at = now
            return

    def _should_pick(self, text: str) -> bool:
        """Apply BetterGI's exact whitelist and exact/fuzzy blacklist semantics."""
        text = _normalize_text(text)
        if len(text) <= 1 or self._always_excluded(text):
            return False
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
        return "我在" in text and any(
            value in text for value in ("声望", "回声", "悬木人", "流泉")
        )

    def _press_interaction(self) -> None:
        now = time.monotonic()
        if now - self._last_action_at < 0.8:
            return
        self.ctx.input.key_press("F")
        self.log("[AutoPick] 直接交互")
        self._last_text = ""
        self._last_action_at = now

    def _is_gameplay_frame(self, region: ImageRegion) -> bool:
        """Require BetterGI's Paimon HUD marker, which menus hide."""
        from ..vision.game_ui import is_main_ui

        return is_main_ui(self.ctx, region.bgr)
