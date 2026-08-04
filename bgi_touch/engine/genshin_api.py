"""genshin 全局对象：游戏语义级助手（bettergi.d.ts 的 genshin.*）。

地图移动、传送、位置识别、角色槽位切换和自动钓鱼已接入 iOS 任务实现。
依赖完整 Windows 窗口状态机的奖励/城市导航接口仍保持显式未迁移状态。
"""

from __future__ import annotations

import json
from pathlib import Path
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
        self._tp_task = None
        self._big_locator = None
        self._positioner = None

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

    # ---- 地图与任务能力 ----

    def tp(self, x, y, map_name=None, force=False):
        """打开大地图并传送到 (x, y) 附近锚点。"""
        if self._tp_task is None:
            from ..pathing.tp import TpTask
            self._tp_task = TpTask(self.ctx, log=self.log)
        result = self._tp_task.tp(float(x), float(y))
        if result and self._positioner is not None:
            self._positioner.set_prior(float(x), float(y))
        return result

    def moveMapTo(self, x, y, country=None):
        if self._tp_task is None:
            from ..pathing.tp import TpTask
            self._tp_task = TpTask(self.ctx, log=self.log)
        return self._tp_task.move_map_to(float(x), float(y))

    def tpToStatueOfTheSeven(self):
        if self._tp_task is None:
            from ..pathing.tp import TpTask
            self._tp_task = TpTask(self.ctx, log=self.log)
        return self._tp_task.tp_to_statue()

    def getPositionFromBigMap(self, map_name=None):
        """大地图打开状态下，返回视野中心的世界坐标（Point2f）；失败抛错。"""
        if self._big_locator is None:
            from ..pathing.tp import BigMapLocator
            self._big_locator = BigMapLocator()
        view = self._big_locator.locate_view(self.ctx.capture_bgr())
        if view is None:
            raise RuntimeError("大地图视野匹配失败（地图未打开？）")
        from ..pathing.map_locator import MapConfig
        wx, wy = MapConfig().image_to_world(view[0] * 8, view[1] * 8)
        from .recognition import Point2f
        return Point2f(wx, wy)

    def getPositionFromMap(self, *args):
        """小地图定位当前世界坐标；失败返回 None（与原版一致）。"""
        if self._positioner is None:
            from ..pathing.positioner import MinimapPositioner
            self._positioner = MinimapPositioner(self.ctx)
        pos = self._positioner.get_position(self.ctx.capture_bgr())
        if pos is None:
            return None
        from .recognition import Point2f
        return Point2f(pos[0], pos[1])

    def getCameraOrientation(self):
        from ..pathing.executor import camera_orientation_deg
        deg = camera_orientation_deg(self.ctx, self.ctx.capture_bgr())
        if deg is None:
            raise RuntimeError("相机朝向检测失败（小地图不可见？）")
        return deg

    def switchParty(self, name):
        """Switch the active character by the local party name mapping.

        BetterGI's Windows implementation edits the full party composition.
        On iOS this compatible method selects the corresponding visible slot;
        `config/party.json` maps names to slots and can be replaced per user.
        """
        try:
            slot = int(name)
        except (TypeError, ValueError):
            config = Path(__file__).resolve().parents[2] / "config" / "party.json"
            if not config.exists():
                self.log("[genshin] 未找到 config/party.json")
                return False
            mapping = json.loads(config.read_text(encoding="utf-8"))
            slot = mapping.get(str(name))
            if isinstance(slot, dict):
                slot = slot.get("slot")
            if slot is None:
                self.log(f"[genshin] 队伍配置中未找到角色/槽位：{name}")
                return False
        if int(slot) not in (1, 2, 3, 4):
            return False
        self.ctx.input.switch_party_slot(int(slot))
        return True

    def setBigMapZoomLevel(self, level):
        if self._tp_task is None:
            from ..pathing.tp import TpTask
            self._tp_task = TpTask(self.ctx, log=self.log)
        return self._tp_task.set_big_map_zoom_level(float(level))

    def getBigMapZoomLevel(self):
        if self._tp_task is None:
            from ..pathing.tp import TpTask
            self._tp_task = TpTask(self.ctx, log=self.log)
        return self._tp_task.get_big_map_zoom_level()

    def autoFishing(self, policy=None):
        from ..tasks.dispatcher import TaskDispatcher
        return TaskDispatcher(self.ctx, log=self.log).run_auto_fishing_task(policy)

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
        if self._positioner is not None:
            self._positioner.reset()
