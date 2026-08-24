"""genshin 全局对象：游戏语义级助手（bettergi.d.ts 的 genshin.*）。

地图移动、传送、位置识别、角色槽位切换和自动钓鱼已接入 iOS 任务实现。
依赖完整 Windows 窗口状态机的奖励/城市导航接口仍保持显式未迁移状态。
"""

from __future__ import annotations

import json
import math
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
        self._big_locators = {}
        self._positioners = {}
        self._positioner = None  # compatibility alias for older callers
        self._party_slots = self._load_party_slots()

    def _load_party_slots(self) -> dict[str, int]:
        config = Path(__file__).resolve().parents[2] / "config" / "party.json"
        try:
            raw = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {str(name): int(slot) for name, slot in raw.items()
                if isinstance(slot, (int, float)) and int(slot) in (1, 2, 3, 4)}

    def _tp_for(self, map_name: str | None = None):
        from ..pathing.tp import TpTask
        from ..pathing.map_locator import resolve_map_name

        name = resolve_map_name(map_name)
        if name == "Teyvat" and self._tp_task is not None:
            return self._tp_task
        task = TpTask(self.ctx, log=self.log, map_name=name)
        if name == "Teyvat":
            self._tp_task = task
        return task

    def _positioner_for(self, map_name: str = "Teyvat"):
        from ..pathing.map_locator import resolve_map_name
        map_name = resolve_map_name(map_name)
        if map_name not in self._positioners:
            from ..pathing.positioner import MinimapPositioner
            self._positioners[map_name] = MinimapPositioner(self.ctx, map_name)
        self._positioner = self._positioners[map_name]
        return self._positioner

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
        from ..vision.game_ui import is_main_ui

        return is_main_ui(self.ctx, bgr)

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
        task = self._tp_for(map_name)
        last_error = None
        # Keep AutoPick/AutoSkip paused across retries as well. Restoring a
        # trigger during the one-second retry gap can press F on the map and
        # leave the next attempt with an already disturbed touch state.
        with task.exclusive_triggers():
            for attempt in range(3):
                try:
                    result = task.tp(float(x), float(y), force=bool(force))
                    if result:
                        self._positioner_for(str(map_name or "Teyvat")).set_prior(
                            float(x), float(y)
                        )
                    return result
                except RuntimeError as error:
                    last_error = error
                    if attempt >= 2:
                        raise
                    self.log(f"[tp] 传送确认失败，重试 {attempt + 1}/2：{error}")
                    self.ctx.sleep(1000)
        if last_error is not None:
            raise last_error
        raise RuntimeError("传送失败：未知原因")

    def moveMapTo(self, x, y, forceCountry=None):
        return self._tp_for().move_map_to(
            float(x), float(y), force_country=forceCountry,
        )

    def clickMapPoint(self, x, y, forceCountry=None):
        """移动到世界坐标并点击地图上的点，保持地图选择面板打开。"""
        return self._tp_for().click_map_point(
            float(x), float(y), force_country=forceCountry,
        )

    def moveIndependentMapTo(self, x, y, map_name, forceCountry=None):
        task = self._tp_for(str(map_name))
        return task.move_independent_map_to(
            float(x), float(y), str(map_name), force_country=forceCountry,
        )

    def tpToStatueOfTheSeven(self):
        return self._tp_for().tp_to_statue()

    def teleportToStatue(self):
        """Community-script alias for BetterGI's TpToStatueOfTheSeven."""
        return self.tpToStatueOfTheSeven()

    def getPositionFromBigMap(self, map_name=None):
        """大地图打开状态下，返回视野中心的世界坐标（Point2f）；失败抛错。"""
        from ..pathing.tp import BigMapLocator
        name = str(map_name or "Teyvat")
        locator = self._big_locators.get(name)
        if locator is None:
            locator = BigMapLocator(name)
            self._big_locators[name] = locator
            if name == "Teyvat":
                self._big_locator = locator
        view = locator.locate_view(self.ctx.capture_bgr())
        if view is None:
            raise RuntimeError("大地图视野匹配失败（地图未打开？）")
        wx, wy = locator.feature_to_world(view[0], view[1])
        from .recognition import Point2f
        return Point2f(wx, wy)

    def getPositionFromMap(self, *args):
        """小地图定位当前世界坐标，兼容原版的缓存和局部匹配重载。

        BetterGI exposes both ``(mapName, cacheTimeMs)`` and
        ``(mapName, x, y)``.  The latter is a local-match hint, not a cache
        duration; keeping the overload parsing explicit prevents a world X
        coordinate from accidentally becoming a multi-second cache timeout.
        """
        map_name = "Teyvat"
        cache_time_ms = 900
        prior: tuple[float, float] | None = None
        if args:
            first = args[0]
            if isinstance(first, str):
                map_name = first
                numeric = args[1:]
                if len(numeric) == 1 and isinstance(numeric[0], (int, float)):
                    cache_time_ms = int(numeric[0])
                elif len(numeric) >= 2 and all(
                    isinstance(value, (int, float)) for value in numeric[:2]
                ):
                    prior = (float(numeric[0]), float(numeric[1]))
                    if len(numeric) >= 3 and isinstance(numeric[2], (int, float)):
                        cache_time_ms = int(numeric[2])
            elif isinstance(first, (int, float)):
                cache_time_ms = int(first)
        positioner = self._positioner_for(map_name)
        if prior is not None:
            positioner.set_prior(*prior)
        pos = positioner.get_position_stable(
            self.ctx.capture_bgr(), cache_time_ms=max(0, cache_time_ms)
        )
        if pos is None:
            return None
        from .recognition import Point2f
        return Point2f(pos[0], pos[1])

    def getPositionFromMapWithMatchingMethod(self, *args):
        """Compatibility overload; local iOS assets currently use SIFT."""
        map_name = "Teyvat"
        matching_method = "SIFT"
        cache_time_ms = 900
        if args and isinstance(args[0], str):
            if len(args) == 1:
                matching_method = args[0]
            else:
                map_name, matching_method = args[0], str(args[1])
                if len(args) >= 3:
                    cache_time_ms = int(args[2])
        if str(matching_method).lower() != "sift":
            self.log(f"[genshin] 地图匹配方法 {matching_method} 暂未移植，回退 SIFT")
        return self.getPositionFromMap(map_name, cache_time_ms)

    def getCameraOrientation(self):
        from ..pathing.executor import camera_orientation_deg
        deg = camera_orientation_deg(self.ctx, self.ctx.capture_bgr())
        if deg is None:
            raise RuntimeError("相机朝向检测失败（小地图不可见？）")
        return deg

    def switchParty(self, name):
        """Switch the game's named party through the iOS party UI."""
        from .party import PartySwitcher

        try:
            return PartySwitcher(
                self.ctx,
                log=self.log,
                return_main_ui=self.returnMainUi,
            ).switch(str(name))
        except Exception as error:
            self.log(f"[genshin] 队伍切换失败：{error}")
            return False

    def switchCharacter(self, slot1="", slot2="", slot3="", slot4="",
                        usePhysicalSlots=True):
        """Store a script-requested party composition and select its first slot.

        DeviceHub can reproduce the game's control surface but cannot edit an
        in-game party through HID alone. Keeping the requested mapping on the
        context makes subsequent combat scripts use the same composition while
        returning ``False`` for unknown character names, matching BetterGI's
        failure contract.
        """
        requested = [str(value or "").strip() for value in (slot1, slot2, slot3, slot4)]
        available = dict(self._party_slots)
        for index, character in enumerate(requested, start=1):
            if not character:
                continue
            if character not in available:
                self.log(f"[genshin] 队伍配置中未找到角色：{character}")
                return False
            available[character] = index
        self._party_slots = available
        setattr(self.ctx, "party_slots", available)
        first = next((name for name in requested if name), None)
        if first:
            self.ctx.input.switch_party_slot(available[first])
        self.log("[genshin] 已更新脚本队伍槽位映射；iOS 端保留当前游戏队伍")
        return True

    def setBigMapZoomLevel(self, level):
        return self._tp_for().set_big_map_zoom_level(float(level))

    def getBigMapZoomLevel(self):
        return self._tp_for().get_big_map_zoom_level()

    def autoFishing(self, policy=None):
        from ..tasks.dispatcher import TaskDispatcher
        return TaskDispatcher(self.ctx, party_slots=self._party_slots,
                              log=self.log).run_auto_fishing_task(policy)

    def _find_text(self, *keywords: str, roi=(0, 0, 1920, 1080)):
        from .recognition import RecognitionObject

        hits = self.ctx.capture_region().find_multi(
            RecognitionObject.ocr(*roi), limit=40
        )
        wanted = tuple(str(keyword).replace(" ", "") for keyword in keywords)
        return next((hit for hit in hits
                     if any(term in hit.text.replace(" ", "") for term in wanted)), None)

    def _tap_text(self, *keywords: str, timeout_s: float = 8,
                   roi=(0, 0, 1920, 1080)) -> bool:
        import time

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            hit = self._find_text(*keywords, roi=roi)
            if hit is not None:
                hit.click()
                self.ctx.sleep(700)
                return True
            self.ctx.sleep(350)
        return False

    def _open_paimon_menu(self) -> None:
        self.ctx.input.key_press("ESCAPE")
        self.ctx.sleep(900)

    def _poi_route(self, kind: str, country: str) -> Path:
        country_alias = {
            "mondstadt": "蒙德", "liyue": "璃月", "inazuma": "稻妻",
            "sumeru": "须弥", "fontaine": "枫丹", "natlan": "纳塔",
            "nod-krai": "挪德卡莱", "nodkrai": "挪德卡莱",
        }
        key = country_alias.get(str(country).strip().lower(), str(country).strip())
        root = Path(__file__).resolve().parents[2]
        local_candidates = (
            root / "assets" / "pathing" / "poi" / f"{kind}_{key}.json",
            root / "assets" / "pathing" / f"{kind}_{key}.json",
        )
        original_root = root.parent / "better-genshin-impact"
        original_candidates = (
            original_root / "BetterGenshinImpact" / "GameTask" / "Common" /
            "Element" / "Assets" / "Json" / f"{kind}_{key}.json",
        )
        for candidate in (*local_candidates, *original_candidates):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"未找到 {country} 的 {kind} 路线；请导入原项目 Common/Element/Assets/Json 路线"
        )

    def _run_poi_route(self, kind: str, country: str, *talk_keywords: str) -> bool:
        from ..pathing.executor import PathingExecutor
        from ..pathing.model import PathingTask

        route = PathingTask.load(self._poi_route(kind, country))
        ok = PathingExecutor(
            self.ctx,
            party_slots=self._party_slots,
            log=self.log,
        ).run(route)
        if not ok:
            return False
        self.ctx.input.key_press("F")
        self.ctx.sleep(1200)
        if talk_keywords:
            self._tap_text(*talk_keywords, timeout_s=6)
        return True

    def blessingOfTheWelkinMoon(self):
        self._open_paimon_menu()
        opened = self._tap_text("空月祝福", "月卡", "Welkin", timeout_s=5)
        if not opened:
            self.ctx.input.key_press("ESCAPE")
            return False
        claimed = self._tap_text("领取", "Claim", timeout_s=5)
        self.ctx.input.key_press("ESCAPE")
        return claimed

    def claimBattlePassRewards(self):
        self._open_paimon_menu()
        opened = self._tap_text("纪行", "BattlePass", "Battle Pass", timeout_s=5)
        if not opened:
            self.ctx.input.key_press("ESCAPE")
            return False
        claimed = self._tap_text("一键领取", "领取", "Claim All", timeout_s=6)
        self.ctx.input.key_press("ESCAPE")
        return claimed

    def claimEncounterPointsRewards(self):
        self._open_paimon_menu()
        opened = self._tap_text("冒险之证", "历练点", "Adventurer Handbook", timeout_s=5)
        if not opened:
            self.ctx.input.key_press("ESCAPE")
            return False
        claimed = self._tap_text("领取", "Claim", timeout_s=6)
        self.ctx.input.key_press("ESCAPE")
        return claimed

    def claimMailRewards(self):
        """Open the Paimon mail page and claim all available attachments."""
        self._open_paimon_menu()
        opened = self._tap_text("邮件", "Mail", timeout_s=5)
        if not opened:
            self.ctx.input.key_press("ESCAPE")
            return False
        claimed = self._tap_text("全部领取", "领取全部", "Claim All", timeout_s=6)
        # Reward popups and the mail page can each require one close action.
        self.ctx.input.key_press("ESCAPE")
        self.ctx.sleep(400)
        self.returnMainUi(max_tries=4)
        return claimed

    def goToAdventurersGuild(self, country):
        return self._run_poi_route("冒险家协会", str(country), "凯瑟琳", "冒险家协会", "Catherine")

    def goToCraftingBench(self, country):
        return self._run_poi_route("合成台", str(country), "合成", "Craft")

    def goCraftResin(self, country):
        if not self.goToCraftingBench(country):
            return False
        return self._tap_text("浓缩树脂", "Condensed Resin", timeout_s=8)

    def craftMaterial(self, material_name, quantity, material_type=None):
        quantity = int(quantity)
        if quantity <= 0:
            raise ValueError("quantity 必须大于 0")
        selected = self._tap_text(str(material_name), timeout_s=8)
        if not selected:
            return {"success": False, "materialName": str(material_name), "crafted": 0}
        if material_type:
            self._tap_text(str(material_type), timeout_s=3)
        crafted = 0
        for _ in range(quantity):
            if not self._tap_text("合成", "Craft", timeout_s=5):
                break
            self._tap_text("确认", "Confirm", timeout_s=4)
            crafted += 1
        return {
            "success": crafted == quantity,
            "materialName": str(material_name),
            "crafted": crafted,
            "requested": quantity,
        }

    @staticmethod
    def _time_dial_gestures(
        hour: int,
        minute: int,
    ) -> tuple[list[tuple[float, float]], tuple[tuple[float, float], tuple[float, float]]]:
        """Build the circular time-picker gestures in the 1920x1080 space.

        The constants and the two-stage inner-to-outer drag mirror BetterGI's
        ``SetTimeTask``.  A same-point swipe cannot move the game's circular
        thumb, which was the previous iOS placeholder implementation.
        """
        total = (int(hour) * 60 + int(minute)) % (24 * 60)
        normalized_hour, normalized_minute = divmod(total, 60)
        end = (normalized_hour + 6) * 60 + normalized_minute - 20
        center_x, center_y = 1441.0, 501.6

        def point(radius: float, index: float) -> tuple[float, float]:
            angle = index * 3.141592653589793 / 720.0
            return (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle),
            )

        taps = [point(30.0, end + i * 1440.0 / 3.0) for i in (-2, -1, 0)]
        drag = (point(150.0, end + 5.0), point(300.0, end + 20.5))
        return taps, drag

    def setTime(self, hour, minute, skip=False):
        hour, minute = int(hour), int(minute)
        if not 0 <= hour <= 24 or not 0 <= minute <= 59:
            raise ValueError("时间必须为 hour 0-24、minute 0-59")
        self._open_paimon_menu()
        if not self._tap_text("时间", "Time", timeout_s=5):
            self.ctx.input.key_press("ESCAPE")
            return False
        t = self.ctx.transform
        taps, (start, end) = self._time_dial_gestures(hour, minute)
        for ref_x, ref_y in taps:
            x, y = t.to_device(ref_x, ref_y)
            self.ctx.device.tap(x, y, image_width=t.device_width,
                                image_height=t.device_height)
            self.ctx.sleep(50)
        x1, y1 = t.to_device(*start)
        x2, y2 = t.to_device(*end)
        self.ctx.device.multi_touch(
            [{"x1": x1, "y1": y1, "x2": x2, "y2": y2}],
            duration_ms=100,
            image_width=t.device_width,
            image_height=t.device_height,
        )
        self.ctx.sleep(100)
        # The confirm button is fixed in the 1080p game layout. Keep an OCR
        # fallback for translated clients whose button text/layout differs.
        if not self._tap_text("确认", "确定", "Confirm", timeout_s=1.5,
                              roi=(1250, 850, 500, 220)):
            self.ctx.input.click_ref(1500, 1000)
        if skip:
            self.ctx.sleep(100)
            self.ctx.input.click_ref(45, 715)
            self.ctx.sleep(100)
            self.ctx.input.click_ref(45, 715)
        self.ctx.sleep(3000 if not skip else 1200)
        return True

    def wonderlandCycle(self):
        self._open_paimon_menu()
        opened = self._tap_text("千星奇域", "Wonderland", timeout_s=5)
        if not opened:
            self.ctx.input.key_press("ESCAPE")
            return False
        self.ctx.sleep(8000)
        self.ctx.input.key_press("ESCAPE")
        self.ctx.sleep(1500)
        return True

    def clearPartyCache(self):
        for positioner in self._positioners.values():
            positioner.reset()
