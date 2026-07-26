"""自动剧情触发器（原版 AutoSkip 的移动端适配）。

检测条件（防误触，先判定在对话中）：
- 对话选项图标 icon_option 模板命中 → 点选项（可偏好含指定文本/最上面一项）
- 或左上角自动播放指示（stop_auto 模板）命中 → 点屏幕中下部推进对话

模板来自原版 AutoSkip/Assets（1080p 基准，识别层自动缩放）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..engine.context import GameContext
from ..engine.recognition import ImageRegion, Mat, RecognitionObject

TEMPLATES = Path(__file__).resolve().parents[2] / "assets" / "templates" / "autoskip"


class AutoSkipTrigger:
    name = "AutoSkip"

    def __init__(self, ctx: GameContext, prefer_text: str | None = None,
                 log: Callable[[str], None] = print):
        self.ctx = ctx
        self.enabled = True
        self.prefer_text = prefer_text
        self.log = log
        # 选项图标出现在屏幕右侧偏下（ref 空间 ROI 收窄降误报）
        self.ro_option = RecognitionObject.template_match(
            Mat.from_file(str(TEMPLATES / "icon_option.png")), 1000, 280, 850, 700)
        self.ro_option.threshold = 0.75
        # 对话中的左上"自动播放"指示
        self.ro_auto = RecognitionObject.template_match(
            Mat.from_file(str(TEMPLATES / "stop_auto.png")), 0, 0, 400, 140)
        self.ro_auto.threshold = 0.75

    def on_frame(self, region: ImageRegion) -> None:
        options = region.find_multi(self.ro_option, limit=6)
        if options:
            chosen = options[0]
            if self.prefer_text:
                for o in options:
                    # 选项文字在图标右侧：OCR 该行
                    line = region.find(RecognitionObject.ocr(o.x + 30, o.y - 12, 800, 60))
                    if line.is_exist() and self.prefer_text in line.text:
                        chosen = o
                        break
            self.log(f"[AutoSkip] 点击对话选项 @({chosen.x:.0f},{chosen.y:.0f})")
            chosen.click()
            return
        if region.find(self.ro_auto).is_exist():
            # 对话进行中且无选项 → 点中下部推进
            self.ctx.input.click_ref(960, 820)
