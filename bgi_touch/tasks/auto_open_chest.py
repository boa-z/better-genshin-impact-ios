"""AutoOpenChest task migrated from BetterGI's template/movement loop."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from ..engine.context import GameContext
from ..engine.recognition import Mat, RecognitionObject

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "templates" / "autoopenchest"


class AutoOpenChestTask:
    def __init__(
        self,
        ctx: GameContext,
        timeout_s: float = 60,
        idle_timeout_s: float = 4,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.timeout_s = max(1.0, float(timeout_s))
        self.idle_timeout_s = max(1.0, float(idle_timeout_s))
        self.log = log
        self._templates: dict[str, Mat] = {}

    def _template(self, name: str) -> Mat:
        if name not in self._templates:
            self._templates[name] = Mat.from_file(str(ASSETS / f"{name}.png"))
        return self._templates[name]

    def _find(self, region, name: str, roi=None):
        ro = RecognitionObject.template_match(self._template(name), *(roi or (None,) * 4))
        ro.threshold = 0.70
        return region.find(ro)

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        deadline = time.monotonic() + self.timeout_s
        last_seen = time.monotonic()
        moving = False
        self.log("[AutoOpenChest] 开始寻找宝箱")
        try:
            while time.monotonic() < deadline:
                if cancelled and cancelled():
                    return False
                region = self.ctx.capture_region()
                prompt = self._find(region, "chest_F_icon", (1150, 450, 100, 300))
                flower = self._find(region, "flower_F_icon", (1150, 450, 100, 300))
                if not prompt.is_empty() or not flower.is_empty():
                    if moving:
                        self.ctx.input.key_up("W")
                        moving = False
                    self.ctx.input.key_press("F")
                    self.log("[AutoOpenChest] 已交互" + ("地脉花" if not flower.is_empty() else "宝箱"))
                    return True

                chest = self._find(region, "chest", (330, 130, 1250, 840))
                now = time.monotonic()
                if chest.is_empty():
                    if now - last_seen >= self.idle_timeout_s:
                        self.log("[AutoOpenChest] 未检测到宝箱图标，结束任务")
                        return False
                    self.ctx.sleep(250)
                    continue

                last_seen = now
                # The chest icon's ref coordinates are enough to steer the
                # camera. Keep the movement key held only while a target exists.
                if not moving:
                    self.ctx.input.key_down("W")
                    moving = True
                cx = chest.x + chest.width / 2
                cy = chest.y + chest.height / 2
                if cy > 600:
                    self.ctx.input.key_up("W")
                    moving = False
                    self.ctx.input.key_press("S", hold_ms=120)
                    self.ctx.input.move_camera_by(0, 120)
                else:
                    gap = 960 - cx
                    if abs(gap) > 90:
                        self.ctx.input.move_camera_by(gap * 0.5, 0)
                self.ctx.sleep(500)
            self.log("[AutoOpenChest] 超时退出")
            return False
        finally:
            if moving:
                self.ctx.input.key_up("W")
            self.ctx.input.release_all()
