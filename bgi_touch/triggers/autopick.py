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

# 黑名单模式的移动端安全兜底（对话/进入类交互，交给 AutoSkip 或玩家）。
DEFAULT_BLACKLIST = ["对话", "进入", "传送", "离开", "调查", "阅读", "操作", "开启", "参加"]
DEFAULT_WHITELIST_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets" / "config" / "pick" / "default_pick_white_lists.json"
)


@lru_cache(maxsize=1)
def _default_whitelist() -> frozenset[str]:
    try:
        values = json.loads(DEFAULT_WHITELIST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return frozenset()
    return frozenset(_normalize_text(value) for value in values if _normalize_text(value))


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
        mode: str = "Whitelist",
        log: Callable[[str], None] = print,
        force_interaction: bool = False,
    ):
        self.ctx = ctx
        self.enabled = True
        self.mode = _normalize_mode(mode)
        self.blacklist = {
            _normalize_text(value) for value in (
                blacklist if blacklist is not None else DEFAULT_BLACKLIST
            ) if _normalize_text(value)
        }
        self.fuzzy_blacklist = tuple(
            _normalize_text(value) for value in (fuzzy_blacklist or []) if _normalize_text(value)
        )
        selected_whitelist = (
            _default_whitelist() if whitelist is None
            else {_normalize_text(value) for value in whitelist if _normalize_text(value)}
        )
        exclusions = {
            _normalize_text(value) for value in (whitelist_exclusions or []) if _normalize_text(value)
        }
        self.whitelist = frozenset(selected_whitelist) - exclusions
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
        """Use the minimap circle as a cheap main-gameplay guard."""
        import cv2

        mm = self.ctx.layout.buttons.get("minimapCenter")
        if mm is None:
            return False
        width, height = self.ctx.transform.device_width, self.ctx.transform.device_height
        cx, cy = int(mm[0] * width), int(mm[1] * height)
        radius = max(20, int(0.075 * width))
        x0, y0 = max(0, cx - radius), max(0, cy - radius)
        crop = region.bgr[y0:cy + radius, x0:cx + radius]
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=radius,
            param1=120,
            param2=40,
            minRadius=int(radius * 0.55),
            maxRadius=int(radius * 0.95),
        )
        return circles is not None
