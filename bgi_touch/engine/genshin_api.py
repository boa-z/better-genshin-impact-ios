"""genshin 全局对象：游戏语义级助手（bettergi.d.ts 的 genshin.*）。

已实现：基础属性、returnMainUi（模板/启发式）、chooseTalkOption（OCR）。
依赖大地图定位/复杂流程的方法抛出带路线图指引的 NotImplementedError。
"""

from __future__ import annotations

from typing import Callable

from ..vision.coordinate import REF_HEIGHT, REF_WIDTH
from ..vision.ocr import get_ocr
from .context import GENSHIN_BUNDLE_ID, GameContext


def _todo(name: str):
    raise NotImplementedError(f"genshin.{name} 依赖大地图定位/专用流程，尚未移植（docs/ROADMAP.md）")


class GenshinApi:
    def __init__(self, ctx: GameContext, log: Callable[[str], None] = print):
        self.ctx = ctx
        self.log = log

    # ---- 属性（脚本假定 1080p 基准）----
    @property
    def width(self) -> int:
        return REF_WIDTH

    @property
    def height(self) -> int:
        return REF_HEIGHT

    @property
    def scaleTo1080PRatio(self) -> float:
        return 1.0

    @property
    def screenDpiScale(self) -> float:
        return 1.0

    # ---- 已实现 ----

    def returnMainUi(self, max_tries: int = 6) -> bool:
        """回到主界面：反复点关闭/返回位置，直到检测到主界面特征。

        主界面判定：左上小地图区域存在明显圆形边界（简化启发式）。
        """
        import cv2
        import numpy as np

        for _ in range(max_tries):
            frame = self.ctx.capture_bgr()
            if self._is_main_ui(frame):
                return True
            # 依次尝试：右上角 X、通用返回（左上）、ESC 映射（派蒙菜单开着时点它会关闭）
            self.ctx.input.click_ref(1870, 50)
            self.ctx.sleep(700)
            frame = self.ctx.capture_bgr()
            if self._is_main_ui(frame):
                return True
            self.ctx.input.click_ref(50, 50)
            self.ctx.sleep(700)
        self.log("[genshin] returnMainUi 未能确认主界面，请检查画面")
        return False

    def _is_main_ui(self, bgr) -> bool:
        import cv2
        import numpy as np

        mm = self.ctx.layout.buttons.get("minimapCenter")
        if mm is None:
            return False
        t = self.ctx.transform
        cx, cy = mm[0] * t.device_width, mm[1] * t.device_height
        r = int(0.075 * t.device_width)
        x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
        crop = bgr[y0:y0 + 2 * r, x0:x0 + 2 * r]
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=r,
                                   param1=120, param2=40,
                                   minRadius=int(r * 0.55), maxRadius=int(r * 0.95))
        return circles is not None

    def chooseTalkOption(self, option: str, skip_times: int = 10, is_orange: bool = False) -> bool:
        """OCR 对话选项并点击包含指定文本的一项。"""
        for _ in range(max(1, int(skip_times))):
            region = self.ctx.capture_region()
            hits = region.find_multi(
                __import__("bgi_touch.engine.recognition", fromlist=["RecognitionObject"])
                .RecognitionObject.ocr(1200, 300, 700, 700)
            )
            for h in hits:
                if str(option) in h.text:
                    h.click()
                    return True
            self.ctx.input.click_ref(960, 800)  # 点击继续对话
            self.ctx.sleep(800)
        return False

    def uid(self) -> str:
        items = get_ocr().recognize(self.ctx.capture_bgr())
        for it in items:
            if "UID" in it.text.upper():
                return "".join(ch for ch in it.text if ch.isdigit())
        return ""

    def relogin(self) -> None:
        self.ctx.device.stop_app(GENSHIN_BUNDLE_ID)
        self.ctx.sleep(3000)
        self.ctx.device.launch_app(GENSHIN_BUNDLE_ID)
        self.ctx.sleep(30000)
        self.ctx.input.click_ref(960, 540)  # 点击进入
        self.ctx.sleep(15000)

    # ---- 未移植（依赖大地图定位）----

    def tp(self, x, y, map_name=None, force=False):
        _todo("tp")

    def moveMapTo(self, x, y, country=None):
        _todo("moveMapTo")

    def tpToStatueOfTheSeven(self):
        _todo("tpToStatueOfTheSeven")

    def getPositionFromBigMap(self, map_name=None):
        _todo("getPositionFromBigMap")

    def getPositionFromMap(self, *args):
        _todo("getPositionFromMap")

    def getCameraOrientation(self):
        from ..pathing.executor import camera_orientation_deg
        deg = camera_orientation_deg(self.ctx, self.ctx.capture_bgr())
        if deg is None:
            raise RuntimeError("相机朝向检测失败（小地图不可见？）")
        return deg

    def switchParty(self, name):
        _todo("switchParty")

    def setBigMapZoomLevel(self, level):
        _todo("setBigMapZoomLevel")

    def getBigMapZoomLevel(self):
        _todo("getBigMapZoomLevel")

    def autoFishing(self, policy=None):
        _todo("autoFishing")

    def blessingOfTheWelkinMoon(self):
        _todo("blessingOfTheWelkinMoon")

    def claimBattlePassRewards(self):
        _todo("claimBattlePassRewards")

    def claimEncounterPointsRewards(self):
        _todo("claimEncounterPointsRewards")

    def goToAdventurersGuild(self, country):
        _todo("goToAdventurersGuild")

    def goToCraftingBench(self, country):
        _todo("goToCraftingBench")

    def setTime(self, hour, minute, skip=False):
        _todo("setTime")

    def wonderlandCycle(self):
        _todo("wonderlandCycle")

    def clearPartyCache(self):
        pass
