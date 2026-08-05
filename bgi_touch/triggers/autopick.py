"""自动拾取触发器（原版 AutoPick 的移动端适配）。

PC 版检测 "F" 键图标 + SVTR 文本识别；移动端拾取是屏幕右中部的物品列表按钮。
实现：OCR 交互按钮附近的条目文字，命中可拾取词条（或非黑名单）时点按交互位。
"""

from __future__ import annotations

import time
from typing import Callable

from ..engine.context import GameContext
from ..engine.recognition import ImageRegion, RecognitionObject

# 出现即不点的词条（对话/进入类交互，交给 AutoSkip 或玩家）
DEFAULT_BLACKLIST = ["对话", "进入", "传送", "离开", "调查", "阅读", "操作", "开启", "参加"]


class AutoPickTrigger:
    name = "AutoPick"

    def __init__(self, ctx: GameContext, blacklist: list[str] | None = None,
                 log: Callable[[str], None] = print,
                 force_interaction: bool = False):
        self.ctx = ctx
        self.enabled = True
        self.blacklist = blacklist if blacklist is not None else DEFAULT_BLACKLIST
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
            text = h.text.strip()
            if len(text) < 2 or any(b in text for b in self.blacklist):
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
