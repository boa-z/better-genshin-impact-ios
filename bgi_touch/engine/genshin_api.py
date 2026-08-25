"""genshin 全局对象：游戏语义级助手（bettergi.d.ts 的 genshin.*）。

地图移动、传送、位置识别、角色槽位切换、奖励/城市导航和时间调整均已接入
iOS 任务实现；需要真机画面确认的接口保留明确的 OCR/路线失败结果。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

from ..config_values import as_bool
from ..vision.coordinate import REF_HEIGHT, REF_WIDTH
from ..vision.ocr import get_ocr
from .context import GENSHIN_BUNDLE_ID, GameContext


def _todo(name: str):
    raise NotImplementedError(f"genshin.{name} 依赖大地图定位/专用流程，尚未移植（docs/ROADMAP.md）")


_AUTOSKIP_ASSETS = Path(__file__).resolve().parents[2] / "assets" / "templates" / "autoskip"
_STYGIAN_ASSETS = Path(__file__).resolve().parents[2] / "assets" / "templates" / "stygian"


class NavigationInstanceApi:
    """Small ``NavigationInstance`` compatibility surface for JS scripts.

    BetterGI exposes this object as an implementation detail of ``genshin``.
    A few community scripts still use it to perform local minimap matching,
    so keep the same pixel-coordinate contract while routing the actual work
    through the iOS ``MinimapPositioner``.
    """

    def __init__(self, api: "GenshinApi"):
        self._api = api
        self._map_name = "Teyvat"

    @staticmethod
    def _frame(image_region, ctx: GameContext):
        if image_region is None:
            return ctx.capture_bgr()
        value = getattr(image_region, "__wrapped__", image_region)
        bgr = getattr(value, "bgr", None)
        if bgr is not None:
            return bgr
        to_image_region = getattr(value, "to_image_region", None)
        if callable(to_image_region):
            region = to_image_region()
            bgr = getattr(region, "bgr", None)
            if bgr is not None:
                return bgr
        raise TypeError("NavigationInstance 需要 ImageRegion 或可转换为 ImageRegion 的对象")

    def _positioner(self, map_name: str | None):
        self._map_name = str(map_name or self._map_name or "Teyvat")
        return self._api._positioner_for(self._map_name)

    def _check_matching_method(self, matching_method: str | None) -> None:
        if matching_method and str(matching_method).lower() != "sift":
            self._api.log(
                f"[genshin] 地图匹配方法 {matching_method} 暂未移植，回退 SIFT"
            )

    def reset(self) -> None:
        for positioner in self._api._positioners.values():
            positioner.reset()

    def setPrevPosition(self, x: float, y: float) -> None:
        """Set the previous position in native map-pixel coordinates."""
        positioner = self._positioner(self._map_name)
        setter = getattr(positioner, "set_prior_pixel", None)
        if callable(setter):
            setter(float(x), float(y))
            return
        # Keep compatibility with test doubles and older positioners.
        locator = positioner.locator
        locator.prev = (float(x), float(y))
        positioner._last_position = locator.config.image_to_world(float(x), float(y))
        positioner._last_fix_at = 0.0

    def getPosition(
        self,
        image_region=None,
        map_name: str = "Teyvat",
        matching_method: str = "SIFT",
    ):
        self._check_matching_method(matching_method)
        positioner = self._positioner(map_name)
        pixel = positioner.get_position_pixel(self._frame(image_region, self._api.ctx))
        if pixel is None:
            return None
        from .recognition import Point2f
        return Point2f(*pixel)

    def _stable_pixel(
        self,
        image_region,
        map_name: str,
        matching_method: str,
        cache_time_ms: int,
    ):
        self._check_matching_method(matching_method)
        positioner = self._positioner(map_name)
        world = positioner.get_position_stable(
            self._frame(image_region, self._api.ctx),
            cache_time_ms=max(0, int(cache_time_ms)),
        )
        if world is None:
            return None
        ix, iy = positioner.locator.config.world_to_image(*world)
        from .recognition import Point2f
        return Point2f(ix, iy)

    def getPositionStable(
        self,
        image_region=None,
        map_name: str = "Teyvat",
        matching_method: str = "SIFT",
    ):
        return self._stable_pixel(
            image_region, str(map_name or "Teyvat"), matching_method, 0
        )

    def getPositionStableByCache(
        self,
        image_region=None,
        map_name: str = "Teyvat",
        matching_method: str = "SIFT",
        cache_time_ms: int = 900,
    ):
        return self._stable_pixel(
            image_region,
            str(map_name or "Teyvat"),
            matching_method,
            cache_time_ms,
        )

    # ClearScript's host binding is case-insensitive, but direct Python users
    # and the JS facade both benefit from the canonical PascalCase aliases.
    Reset = reset
    SetPrevPosition = setPrevPosition
    GetPosition = getPosition
    GetPositionStable = getPositionStable
    GetPositionStableByCache = getPositionStableByCache


class GenshinApi:
    def __init__(
        self,
        ctx: GameContext,
        log: Callable[[str], None] = print,
        party_slots: dict[str, int] | None = None,
    ):
        self.ctx = ctx
        self.log = log
        self._tp_task = None
        self._big_locator = None
        self._big_locators = {}
        self._positioners = {}
        self._positioner = None  # compatibility alias for older callers
        self._party_slots = (
            party_slots if party_slots is not None else self._load_party_slots()
        )
        self._navigation_instance = None
        if party_slots is not None:
            setattr(self.ctx, "party_slots", self._party_slots)

    @property
    def lazyNavigationInstance(self) -> NavigationInstanceApi:
        if getattr(self, "_navigation_instance", None) is None:
            self._navigation_instance = NavigationInstanceApi(self)
        return self._navigation_instance

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

    # ---- 属性（脚本坐标仍以 1080p 为基准，尺寸属性返回实际捕获尺寸）----
    def _capture_dimensions(self) -> tuple[int, int]:
        """Return the latest native capture dimensions without taking a frame.

        BetterGI's ``Genshin.Width``/``Height`` describe the game capture
        rectangle, not the 1920x1080 reference space used by recognition
        assets.  ``GameContext.transform`` is updated whenever a frame is
        received, so it is the authoritative and screenshot-free source for
        JS property reads.  The reference fallback keeps lightweight host
        doubles and reconnecting contexts compatible.
        """
        transform = getattr(self.ctx, "transform", None)
        try:
            width = int(getattr(transform, "device_width"))
            height = int(getattr(transform, "device_height"))
        except (AttributeError, TypeError, ValueError):
            return REF_WIDTH, REF_HEIGHT
        if width <= 0 or height <= 0:
            return REF_WIDTH, REF_HEIGHT
        return width, height

    @property
    def width(self) -> int:
        return self._capture_dimensions()[0]

    @property
    def height(self) -> int:
        return self._capture_dimensions()[1]

    @property
    def scaleTo1080PRatio(self) -> float:
        return self.height / REF_HEIGHT

    @property
    def screenDpiScale(self) -> float:
        return 1.0

    # ---- 已实现 ----

    def _exit_door_recognition(self):
        """Return BetterGI's generic return-to-world recognition object."""
        from .recognition import Mat, RecognitionObject

        cached = getattr(self, "_exit_door_ro", None)
        if cached is False:
            return None
        if cached is not None:
            return cached

        asset = _STYGIAN_ASSETS / "exit_door.png"
        if not asset.is_file():
            self.log(f"[genshin] 缺少通用退出门模板：{asset}")
            self._exit_door_ro = False
            return None

        ro = RecognitionObject.template_match(Mat.from_file(str(asset)))
        ro.name = "BtnExitDoor"
        self._exit_door_ro = ro
        return ro

    def _find_exit_door(self, frame):
        from .recognition import ImageRegion

        ro = self._exit_door_recognition()
        if ro is None:
            return None
        return ImageRegion(self.ctx, frame).find(ro)

    def returnMainUi(self, max_tries: int = 8) -> bool:
        """Return to the gameplay HUD using BetterGI's stable exit flow.

        The desktop implementation sends ``ESCAPE`` first, then recognizes
        the generic exit-door control instead of assuming a fixed click
        position.  Each attempt reuses one capture for both door and HUD
        checks; a post-click capture is only needed after the door is found.
        """
        try:
            tries = max(0, int(max_tries))
        except (TypeError, ValueError):
            tries = 8

        frame = self.ctx.capture_bgr()
        if self._is_main_ui(frame):
            return True

        for _ in range(tries):
            self.ctx.input.key_press("ESCAPE")
            self.ctx.sleep(900)
            frame = self.ctx.capture_bgr()

            exit_door = self._find_exit_door(frame)
            if exit_door is not None and exit_door.is_exist():
                exit_door.click()
                self.log("[genshin] 点击通用退出门")
                self.ctx.sleep(5000)
                frame = self.ctx.capture_bgr()

            if self._is_main_ui(frame):
                return True

        # Match ReturnMainUiTask's last-resort keyboard sequence.  Confirm
        # the result so callers that depend on the bool contract do not
        # continue while a menu or modal dialog is still open.
        self.ctx.sleep(500)
        self.ctx.input.key_press("ENTER")
        self.ctx.sleep(500)
        self.ctx.input.key_press("ESCAPE")
        self.ctx.sleep(500)
        if self._is_main_ui(self.ctx.capture_bgr()):
            return True

        self.log("[genshin] returnMainUi 未能确认主界面，请检查画面")
        return False

    def _chat_exit_recognition(self):
        """Build BetterGI's reference-anchored chat-exit recognition object."""
        from .recognition import Mat, RecognitionObject, SearchOptions

        cached = getattr(self, "_chat_exit_ro", None)
        if cached is not None:
            return cached
        asset = _AUTOSKIP_ASSETS / "chat_exit.png"
        if not asset.is_file():
            self.log(f"[genshin] 缺少聊天退出模板：{asset}")
            self._chat_exit_ro = False
            return None
        ro = RecognitionObject.template_match(Mat.from_file(str(asset)))
        # The upstream recognition asset searches a 345x509 area around the
        # lower-right dialogue exit icon on a 1920x1080 reference canvas.  A
        # reference search keeps this correct on the iPhone's extra-wide safe
        # area instead of treating the rectangle as a centred crop.
        ro.ReferenceImageSize = (1920, 1080)
        ro.ReferenceBoundingBox = (1284, 785, 31, 39)
        ro.SearchOptions = SearchOptions(
            reference_search_box=(1182, 378, 345, 509),
        )
        ro.threshold = 0.72
        self._chat_exit_ro = ro
        return ro

    def _find_chat_exit(self, frame):
        from .recognition import ImageRegion

        ro = self._chat_exit_recognition()
        if not ro:
            return None
        return ImageRegion(self.ctx, frame).find(ro)

    def _is_talk_ui_frame(self, frame) -> bool:
        """Detect the two low-cost dialogue indicators from one frame."""
        from .recognition import ImageRegion, Mat, RecognitionObject

        cached = getattr(self, "_talk_ui_ros", None)
        if cached is None:
            option = RecognitionObject.template_match(
                Mat.from_file(str(_AUTOSKIP_ASSETS / "icon_option.png")),
                1000, 280, 850, 700,
            )
            option.threshold = 0.75
            auto = RecognitionObject.template_match(
                Mat.from_file(str(_AUTOSKIP_ASSETS / "stop_auto.png")),
                0, 0, 400, 140,
            )
            auto.threshold = 0.72
            cached = (option, auto)
            self._talk_ui_ros = cached
        region = ImageRegion(self.ctx, frame)
        option, auto = cached
        return bool(region.find_multi(option, limit=1) or region.find(auto).is_exist())

    def clickChatExitUntilMainUi(self, retry_times: int = 15) -> bool:
        """Click the dialogue exit control until the gameplay HUD returns.

        This is the portable equivalent of BetterGI's
        ``ChooseTalkOptionTask.ClickChatExitUntilMainUi``.  Every iteration
        owns exactly one frame: the chat-exit template, talk indicator and
        main-UI check all consume that same capture, so closing a dialogue
        cannot create a competing screenshot loop on DeviceHub.
        """
        try:
            retries = max(0, int(retry_times))
        except (TypeError, ValueError):
            retries = 15
        for _ in range(retries):
            frame = self.ctx.capture_bgr()
            if self._is_main_ui(frame):
                return True
            chat_exit = self._find_chat_exit(frame)
            if chat_exit is not None and chat_exit.is_exist():
                chat_exit.click()
                self.log("[genshin] 点击退出对话按钮")
                self.ctx.sleep(200)
            elif self._is_talk_ui_frame(frame):
                # A dialogue can be in a text-only step without the exit
                # icon; advance it once and let the next frame decide.
                self.ctx.input.key_press("SPACE")
            self.ctx.sleep(500)

        final_frame = self.ctx.capture_bgr()
        return self._is_main_ui(final_frame)

    click_chat_exit_until_main_ui = clickChatExitUntilMainUi

    def _is_main_ui(self, bgr) -> bool:
        from ..vision.game_ui import is_main_ui

        return is_main_ui(self.ctx, bgr)

    @staticmethod
    def _is_orange_option(option_region) -> bool:
        """Return whether an OCR option contains enough orange pixels.

        BetterGI uses the option's own OCR bounding box for this check rather
        than inspecting the whole dialogue panel.  Keeping the same HSV
        bounds and strict 10% threshold makes ``isOrange`` useful for scripts
        that select actionable (orange) dialogue entries.
        """
        import cv2
        import numpy as np

        bgr = getattr(option_region, "bgr", option_region)
        if bgr is None:
            return False
        bgr = np.asarray(bgr)
        if bgr.size == 0 or bgr.ndim < 2:
            return False
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.ndim != 3 or bgr.shape[2] not in (3, 4):
            return False
        if bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array((10, 150, 150), dtype=np.uint8),
            np.array((25, 255, 255), dtype=np.uint8),
        )
        return cv2.countNonZero(mask) / float(mask.size) > 0.1

    @staticmethod
    def _talk_option_region(region, hit):
        """Materialize an OCR hit without requesting another screenshot."""
        to_image_region = getattr(hit, "to_image_region", None)
        if callable(to_image_region):
            return to_image_region()
        to_image_region = getattr(hit, "toImageRegion", None)
        if callable(to_image_region):
            return to_image_region()
        return region.derive_crop(hit.x, hit.y, hit.width, hit.height)

    def chooseTalkOption(self, option: str, skip_times: int = 10, is_orange: bool = False) -> bool:
        """OCR 对话选项并点击包含指定文本的一项。

        ``is_orange`` follows BetterGI's ``isOrange`` argument.  A matching
        non-orange entry is ignored so a later duplicate with the same text
        can still be selected. If the only matching entry is non-orange,
        return immediately without sending the continue key; this is the
        mobile equivalent of BetterGI's ``FoundButNotOrange`` result and
        prevents an already-claimed daily option from advancing dialogue.
        """
        for _ in range(max(1, int(skip_times))):
            region = self._text_capture_region()
            hits = region.find_multi(
                __import__("bgi_touch.engine.recognition", fromlist=["RecognitionObject"])
                .RecognitionObject.ocr(1200, 300, 700, 700)
            )
            matched_non_orange = False
            for h in hits:
                if str(option) in str(getattr(h, "text", "") or ""):
                    if as_bool(is_orange):
                        try:
                            option_region = self._talk_option_region(region, h)
                            if not self._is_orange_option(option_region):
                                matched_non_orange = True
                                continue
                        except (AttributeError, TypeError, ValueError):
                            # A colour check failing closed is safer than
                            # clicking the wrong same-text dialogue option.
                            matched_non_orange = True
                            continue
                    h.click()
                    return True
            if matched_non_orange:
                return False
            self.ctx.input.click_ref(960, 800)  # 点击继续对话
            self.ctx.sleep(800)
        return False

    def uid(self) -> int:
        """Return the numeric UID contract exposed by BetterGI."""
        items = get_ocr().recognize(self.ctx.capture_bgr())
        for it in items:
            if "UID" in it.text.upper():
                digits = "".join(ch for ch in it.text if ch.isdigit())
                if digits:
                    return int(digits)
        return 0

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
        # BetterGI has an overload ``Tp(x, y, bool force)`` in addition to
        # ``Tp(x, y, string mapName, bool force)``.  JavaScript callers use
        # the short form frequently; without this normalization ``True`` was
        # passed to resolve_map_name and reported as an unknown map.
        if isinstance(map_name, bool):
            if force is False or force is None:
                force = map_name
            map_name = None
        task = self._tp_for(map_name)
        from ..pathing.tp import TeleportPanelNotOpenedError

        last_error = None
        # Keep AutoPick/AutoSkip paused across retries as well. Restoring a
        # trigger during the one-second retry gap can press F on the map and
        # leave the next attempt with an already disturbed touch state.
        with task.exclusive_triggers():
            for attempt in range(3):
                try:
                    result = task.tp(float(x), float(y), force=as_bool(force))
                    if result:
                        self._positioner_for(str(map_name or "Teyvat")).set_prior(
                            float(x), float(y)
                        )
                    return result
                except TeleportPanelNotOpenedError:
                    # The point was already clicked and did not open a panel.
                    # Retrying the same map coordinate only repeats the
                    # failed interaction, which is especially harmful when
                    # the point is locked on a partially explored account.
                    raise
                except RuntimeError as error:
                    last_error = error
                    if attempt >= 2:
                        raise
                    self.log(f"[tp] 传送流程失败，重试 {attempt + 1}/2：{error}")
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
            return None
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
        """Compatibility overload; local iOS assets currently use SIFT.

        BetterGI exposes both ``(matchingMethod)`` and
        ``(mapName, matchingMethod, cacheTimeMs)``.  The former is easy to
        mis-route because a single string can also look like a map name; it is
        explicitly the matching-method overload in the upstream API.
        """
        map_name = "Teyvat"
        matching_method = "SIFT"
        cache_time_ms = 900
        if args and isinstance(args[0], str):
            if len(args) == 1:
                matching_method = args[0]
            else:
                map_name, matching_method = str(args[0]), str(args[1])
                if len(args) >= 3:
                    cache_time_ms = int(args[2])
        elif args and isinstance(args[0], (int, float)):
            cache_time_ms = int(args[0])
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
            result = PartySwitcher(
                self.ctx,
                log=self.log,
                return_main_ui=self.returnMainUi,
            ).switch(str(name))
            if result:
                # The upstream task invalidates RunnerContext's combat scene
                # after a named party switch.  Force the next iOS combat or
                # getAvatars call to refresh the mobile HUD as well.
                self.clearPartyCache()
            return result
        except Exception as error:
            self.log(f"[genshin] 队伍切换失败：{error}")
            return False

    def switchCharacter(self, slot1="", slot2="", slot3="", slot4="",
                        usePhysicalSlots=True):
        """Rebuild requested party slots through the iOS party configuration UI."""
        from .party import CharacterSwitcher

        requested = [str(value or "").strip() for value in (slot1, slot2, slot3, slot4)]
        switcher = CharacterSwitcher(
            self.ctx,
            log=self.log,
            return_main_ui=self.returnMainUi,
        )
        try:
            result = switcher.switch_characters(
                requested,
                use_physical_slots=as_bool(usePhysicalSlots, True),
            )
        except Exception as error:
            self.log(f"[genshin] 角色队伍重组失败：{error}")
            return False
        if not result:
            return False

        available = dict(self._party_slots)
        raw_assignments = getattr(switcher, "last_assignments", None)
        if isinstance(raw_assignments, (list, tuple)):
            assignments = list(raw_assignments)
        else:
            # Keep lightweight host doubles and older integrations compatible
            # with the pre-co-op CharacterSwitcher contract.
            from .party import build_character_assignments

            assignments = build_character_assignments(
                requested,
                use_physical_slots=as_bool(usePhysicalSlots, True),
            )
        assigned_slots = {int(slot) for slot, _name in assignments}
        assigned_names = {str(name) for _slot, name in assignments}
        # Remove stale names from slots that the operation actually edited and
        # remove duplicate copies of a newly selected character.  This keeps
        # CombatScenes/JSON strategy lookups aligned with the co-op UI.
        for name, slot in list(available.items()):
            if int(slot) in assigned_slots or name in assigned_names:
                del available[name]
        for slot, character in assignments:
            available[str(character)] = int(slot)
        self._party_slots.clear()
        self._party_slots.update(available)
        setattr(self.ctx, "party_slots", self._party_slots)
        return True

    def setBigMapZoomLevel(self, level):
        return self._tp_for().set_big_map_zoom_level(float(level))

    def getBigMapZoomLevel(self):
        return self._tp_for().get_big_map_zoom_level()

    def autoFishing(self, policy=None):
        from ..tasks.dispatcher import TaskDispatcher
        return TaskDispatcher(self.ctx, party_slots=self._party_slots,
                              log=self.log).run_auto_fishing_task(policy)

    def _text_capture_region(self):
        """Return one OCR frame without competing with an active frame loop.

        Menu helpers are often called from a realtime-triggered script. A
        direct ``capture_region`` call in every polling iteration would create
        a second DeviceHub screenshot producer and delay the caller's frame
        stream. When TriggerLoop is active, consume its recent cached frame;
        standalone calls still use a fresh capture.
        """
        loop = getattr(self.ctx, "_trigger_loop", None)
        if loop is not None and bool(getattr(loop, "active", False)):
            cached_frame = getattr(self.ctx, "cached_frame", None)
            if callable(cached_frame):
                try:
                    frame, age = cached_frame()
                    interval = max(0.1, float(getattr(loop, "interval", 0.7)))
                    if frame is not None and float(age) <= max(1.5, interval * 3):
                        from .recognition import ImageRegion

                        return ImageRegion(self.ctx, frame)
                except (AttributeError, TypeError, ValueError):
                    # Lightweight hosts and reconnecting contexts may expose
                    # no valid cache tuple; use one direct frame instead.
                    pass
        return self.ctx.capture_region()

    def _find_text(self, *keywords: str, roi=(0, 0, 1920, 1080)):
        from .recognition import RecognitionObject

        wanted = tuple(
            str(keyword).replace(" ", "")
            for keyword in keywords
            if str(keyword or "").strip()
        )
        if not wanted:
            return None
        hits = self._text_capture_region().find_multi(
            RecognitionObject.ocr(*roi), limit=40
        )
        return next(
            (
                hit for hit in hits
                if any(
                    term in str(getattr(hit, "text", "") or "").replace(" ", "")
                    for term in wanted
                )
            ),
            None,
        )

    def _tap_text(self, *keywords: str, timeout_s: float = 8,
                   roi=(0, 0, 1920, 1080)) -> bool:
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
        """Run the upstream two-page battle-pass reward state machine."""
        from ..tasks.claim_rewards import ClaimBattlePassRewardsTask

        result = ClaimBattlePassRewardsTask(
            self.ctx,
            return_main_ui=self.returnMainUi,
            log=self.log,
        ).run()
        return bool(result.get("claimed"))

    def claimEncounterPointsRewards(self, timeout_s: float = 12.0):
        """Run the dedicated upstream-compatible encounter reward job."""
        from ..tasks.claim_encounter_rewards import ClaimEncounterPointsRewardsTask

        result = ClaimEncounterPointsRewardsTask(
            self.ctx,
            timeout_s=timeout_s,
            return_main_ui=self.returnMainUi,
            log=self.log,
        ).run()
        return bool(result.get("ok"))

    def claimMailRewards(self):
        """Run the upstream Paimon-menu/mail reward state machine."""
        from ..tasks.claim_rewards import ClaimMailRewardsTask

        result = ClaimMailRewardsTask(
            self.ctx,
            return_main_ui=self.returnMainUi,
            log=self.log,
        ).run()
        return bool(result.get("claimed"))

    def goToAdventurersGuild(
        self,
        country,
        daily_reward_party_name: str = "",
        only_do_once: bool = False,
        timeout_s: float = 180.0,
        encounter_timeout_s: float = 12.0,
        cancelled: Callable[[], bool] | None = None,
    ):
        """Navigate to Catherine and complete the upstream daily-reward job.

        ``genshin.goToAdventurersGuild(country)`` remains the script-compatible
        one-argument form. The optional arguments mirror
        ``GoToAdventurersGuildTask.Start`` for OneDragon and converted jobs.
        """
        from ..tasks.adventurers_guild import AdventurersGuildTask

        result = AdventurersGuildTask(
            self.ctx,
            str(country),
            daily_reward_party_name=daily_reward_party_name,
            only_do_once=as_bool(only_do_once, False),
            timeout_s=timeout_s,
            encounter_timeout_s=encounter_timeout_s,
            log=self.log,
            api=self,
        ).run(cancelled=cancelled)
        return bool(result.get("ok"))

    def goToCraftingBench(
        self,
        country,
        timeout_s: float = 180.0,
        cancelled: Callable[[], bool] | None = None,
    ):
        """Navigate to the configured crafting bench and enter its UI.

        Keep the original one-argument script shape while exposing timeout and
        cancellation hooks to converted dispatcher/OneDragon jobs.
        """
        from ..tasks.crafting_bench import CraftingBenchTask

        return CraftingBenchTask(
            self.ctx,
            str(country),
            timeout_s=timeout_s,
            party_slots=self._party_slots,
            route_resolver=self._poi_route,
            talk_detector=self._is_talk_ui_frame,
            log=self.log,
        ).go_to_crafting_bench(cancelled=cancelled)

    def goCraftResin(
        self,
        country,
        min_resin_to_keep: int = 0,
        timeout_s: float = 180.0,
        cancelled: Callable[[], bool] | None = None,
    ):
        """Navigate to a crafting bench and craft safe condensed-resin amount."""
        from ..tasks.crafting_bench import CraftingBenchTask

        return CraftingBenchTask(
            self.ctx,
            str(country),
            min_resin_to_keep=min_resin_to_keep,
            timeout_s=timeout_s,
            party_slots=self._party_slots,
            route_resolver=self._poi_route,
            talk_detector=self._is_talk_ui_frame,
            return_main_ui=self.returnMainUi,
            log=self.log,
        ).craft_resin(cancelled=cancelled)

    def craftMaterial(
        self,
        material_name,
        quantity,
        material_type=None,
        cancelled: Callable[[], bool] | None = None,
    ):
        """Craft a material in the current crafting page.

        The previous implementation clicked the generic ``合成`` label once
        per requested item and could therefore submit the wrong material or
        quantity.  Delegate to the upstream-shaped ItemV2/grid/slider task;
        the optional cancellation callback is used by dispatcher callers and
        is omitted from the public three-argument script form.
        """
        from ..tasks.craft_material import CraftMaterialTask

        return CraftMaterialTask(
            self.ctx,
            str(material_name),
            quantity,
            material_type,
            log=self.log,
        ).run(cancelled=cancelled)

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
        skip = as_bool(skip)
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
        from ..tasks.wonderland import WonderlandCycleTask

        return WonderlandCycleTask(self.ctx, log=self.log).run()

    def clearPartyCache(self):
        """Discard cached party recognition, matching RunnerContext semantics."""
        cached = getattr(self.ctx, "party_slots", None)
        if isinstance(cached, dict) and cached is not self._party_slots:
            cached.clear()
        self._party_slots.clear()
        setattr(self.ctx, "party_slots", self._party_slots)
