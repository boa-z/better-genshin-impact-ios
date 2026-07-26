"""自动拾取触发器（原版 AutoPick 的移动端适配）。

PC 版检测 "F" 键图标 + SVTR 文本识别；移动端拾取是屏幕右中部的物品列表按钮。
实现：OCR 交互按钮附近的条目文字，命中可拾取词条（或非黑名单）时点按交互位。
"""

from __future__ import annotations

from typing import Callable

from ..engine.context import GameContext
from ..engine.recognition import ImageRegion, RecognitionObject

# 出现即不点的词条（对话/进入类交互，交给 AutoSkip 或玩家）
DEFAULT_BLACKLIST = ["对话", "进入", "传送", "离开", "调查", "阅读", "操作", "开启", "参加"]


class AutoPickTrigger:
    name = "AutoPick"

    def __init__(self, ctx: GameContext, blacklist: list[str] | None = None,
                 log: Callable[[str], None] = print):
        self.ctx = ctx
        self.enabled = True
        self.blacklist = blacklist if blacklist is not None else DEFAULT_BLACKLIST
        self.log = log
        # 交互列表出现在交互按钮右侧（ref 空间），OCR 该竖条区域
        self.roi = (1080, 380, 420, 320)

    def on_frame(self, region: ImageRegion) -> None:
        hits = region.find_multi(RecognitionObject.ocr(*self.roi), limit=5)
        for h in hits:
            text = h.text.strip()
            if len(text) < 2 or any(b in text for b in self.blacklist):
                continue
            # 命中可拾取物：直接点该条目（移动端点条目即拾取）
            self.log(f"[AutoPick] 拾取: {text}")
            h.click()
            return
