"""GameContext：设备连接 + 坐标变换 + 输入模拟的聚合，任务与脚本的运行环境。"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from ..device.client import DeviceClient
from ..device.config import DeviceHubConfig
from ..input.layout import DEFAULT_LAYOUT, DeviceHubProfile, ControlLayout
from ..input.simulator import InputSimulator
from ..vision.coordinate import ScreenTransform
from .recognition import ImageRegion

GENSHIN_BUNDLE_ID = "com.miHoYo.Yuanshen"
GENSHIN_BUNDLE_ID_ALIASES = (
    "com.miHoYo.Yuanshen",
    "com.miHoYo.GenshinImpact",
)
DEFAULT_KEYMAP_PROFILE = os.environ.get("BGI_KEYMAP_PROFILE", "Genshin-Impact-fixed-16by9")


class GameContext:
    def __init__(self, mcp_url: str | None = None, layout_path: str | Path = DEFAULT_LAYOUT,
                 keymap_profile: str | None = DEFAULT_KEYMAP_PROFILE,
                 keymap_profile_path: str | Path | None = None,
                 devicehub_config_path: str | Path | None = None,
                 device_id: str | None = None,
                 game_bundle_id: str | None = None):
        self._frame_lock = threading.Lock()
        self._last_frame: np.ndarray | None = None
        self._last_frame_at = 0.0
        self._frame_generation = 0
        # A realtime trigger can already own a decoded frame when a task starts
        # a menu/map gesture.  The TriggerLoop pause waits for that frame, but a
        # small context-level gate is still needed for the boundary between its
        # final scene check and the actual input edge.  Keep this state on the
        # context so nested task helpers and lightweight trigger doubles share
        # the same ownership contract without touching DeviceHub screenshots.
        self._input_exclusive_lock = threading.RLock()
        self._input_exclusive_depth = 0
        self.devicehub_config = DeviceHubConfig.load(devicehub_config_path)
        self._configured_game_bundle_id = (
            str(game_bundle_id).strip()
            if game_bundle_id is not None and str(game_bundle_id).strip()
            else self.devicehub_config.game_bundle_id
        )
        # iOS has no desktop-wide listener.  The WebUI KeyMouseHook bridge is
        # the equivalent event source, so carry the upstream setting on the
        # caller-owned context for every JS entry point (CLI/WebUI/ScriptGroup)
        # without adding another configuration channel.
        self.disable_input_monitor = self.devicehub_config.disable_input_monitor
        self.device = DeviceClient(
            mcp_url or self.devicehub_config.mcp_url,
            headless=self.devicehub_config.headless,
        )
        self.device_id = device_id or self.devicehub_config.device_id
        self.device.connect_device(self.device_id)
        status = self.device.status()
        if status.get("status") != "connected":
            raise RuntimeError(f"设备未连接（status={status.get('status')}），请检查 DeviceHub Mask")
        self._device_status_snapshot = dict(status)
        screen_size = status.get("screen_size")
        if isinstance(screen_size, (list, tuple)) and len(screen_size) == 2:
            w, h = screen_size
        else:
            # A reconnect can briefly expose a connected device before the
            # first video frame has populated screen_size. capture_bgr() below
            # replaces this provisional transform with the native frame size.
            print("[context] 初始 status 暂无 screen_size，先使用 1920x1080 基准等待首帧")
            w, h = 1920, 1080
        if h > w:
            w, h = h, w
        self.transform = ScreenTransform(int(w), int(h))
        profile = self._load_keymap_profile(keymap_profile, keymap_profile_path)
        self.keymap_profile_name = profile.name if profile else None
        self.game_bundle_id = self._resolve_game_bundle_id(
            self._configured_game_bundle_id,
            profile,
        )
        self.layout = ControlLayout.load(layout_path, devicehub_profile=profile)
        self.refresh_orientation(status)
        self.input = InputSimulator(
            self.device,
            self.layout,
            self.transform,
            bundle_id=self.game_bundle_id,
        )
        # status.screen_size may be a low-resolution stream size. A native frame
        # is the authoritative coordinate space for tap/swipe and profile matching.
        try:
            self.capture_bgr()
            self.refresh_orientation()
        except Exception as e:
            print(f"[context] 初始截图尺寸同步失败（后续重试）：{e}")
        self._start_orientation_watch()

    def _load_keymap_profile(self, name: str | None,
                             path: str | Path | None) -> DeviceHubProfile | None:
        if path is not None:
            local_path = Path(path).expanduser()
            profile = DeviceHubProfile.from_path(local_path)
            if name and profile.name and profile.name != name:
                raise ValueError(f"keymap profile 名称不匹配：期望 {name}，实际 {profile.name}")
            self._sync_local_profile(local_path, profile)
            return profile
        if not name:
            return None
        try:
            return DeviceHubProfile.from_dict(self.device.get_keymap_profile(name))
        except Exception as e:
            # The desktop app stores profiles outside the repository. When an
            # older headless server has not imported the profile yet, load the
            # canonical local file and best-effort install it through MCP 130.
            local_path = Path.home() / "Library" / "Application Support" / \
                "com.devicehub.mask" / "profiles" / f"{name}.json"
            if local_path.is_file():
                try:
                    profile = DeviceHubProfile.from_path(local_path)
                    self._sync_local_profile(local_path, profile)
                    return profile
                except Exception as local_error:
                    print(f"[context] 本地 DeviceHub profile 读取失败：{local_error}")
            # profile 是增强路径；服务器版本不支持时继续使用本地触控布局。
            print(f"[context] 无法读取 DeviceHub profile {name}，回退手势泵：{e}")
            return None

    def _sync_local_profile(self, path: Path, profile: DeviceHubProfile) -> None:
        """Import an explicitly selected local profile when the server supports it."""
        try:
            self.device.install_keymap_profile(path, overwrite=False)
        except Exception as error:
            # Existing profiles and older servers both legitimately fail here;
            # the local profile remains usable by the touch fallback.
            print(f"[context] DeviceHub profile 未同步（继续使用本地副本）：{error}")

    @staticmethod
    def _resolve_game_bundle_id(
        configured: str | None,
        profile: DeviceHubProfile | None,
    ) -> str:
        """Resolve the app ID shared by lifecycle calls and game input sessions.

        DeviceHub's stock profile differs between the Chinese and global iOS
        packages.  A profile is the most reliable local source because its
        ``bundleIdentifiers`` field is the exact app the mapping targets; an
        explicit config/CLI value remains authoritative for custom builds.
        """
        if configured is not None and str(configured).strip():
            return str(configured).strip()
        if profile is not None:
            for value in profile.bundle_identifiers:
                candidate = str(value).strip()
                if candidate:
                    return candidate
        return GENSHIN_BUNDLE_ID

    def _start_orientation_watch(self) -> None:
        """朝向看门狗：应用切换（游戏↔其他 App/重登）会改变服务器 tap 坐标空间，
        且感知有延迟——静态映射会在切换窗口内失效，因此持续轮询并热更新。"""
        import threading

        def watch():
            last = None
            while True:
                time.sleep(2.5)
                try:
                    st = self.device.status()
                    self._remember_device_status(st)
                    ori = str(st.get("orientation", ""))
                    if ori != last:
                        if last is not None:
                            print(f"[context] 屏幕朝向变化 {last} → {ori}，更新触控映射")
                        self.refresh_orientation(st)
                        last = ori
                except Exception:
                    pass  # 设备暂不可用时静默重试

        threading.Thread(target=watch, daemon=True, name="orientation-watch").start()

    @contextmanager
    def exclusive_input(self):
        """Temporarily prevent realtime triggers from emitting input edges.

        This is deliberately separate from the TriggerLoop lock.  It protects
        the last few instructions of a trigger callback even when a callback
        was already holding a decoded frame while a task began a map/menu
        operation.  The scope is re-entrant for nested teleport and task jobs.
        """
        with self._input_exclusive_lock:
            self._input_exclusive_depth += 1
        try:
            yield
        finally:
            with self._input_exclusive_lock:
                self._input_exclusive_depth = max(0, self._input_exclusive_depth - 1)

    @property
    def input_exclusive(self) -> bool:
        with self._input_exclusive_lock:
            return self._input_exclusive_depth > 0

    def _remember_device_status(self, status: dict | None) -> None:
        if isinstance(status, dict):
            self._device_status_snapshot = dict(status)

    def cached_device_status(self) -> dict:
        """Return the latest status observed by the context without MCP I/O."""
        return dict(getattr(self, "_device_status_snapshot", {}))

    def refresh_orientation(self, status: dict | None = None) -> None:
        """按服务器当前朝向决定 tap 坐标映射。

        实测（iPhone 13 Pro Max + DeviceHub Mask）：
        - 服务器 status.orientation 为 landscape-* 时，tap 坐标空间即横屏空间，
          与本项目逻辑空间一致，无需映射；
        - 为 portrait 时（服务器尚未感知游戏横屏），tap 空间是竖屏帧空间，
          需做 P.x = P_W - L.y, P.y = L.x 逆旋转映射；
        - 截图流的帧朝向独立于此（capture_bgr 按帧宽高自适应旋转）。
        游戏启动/切前台后服务器朝向可能变化，此时应再调用本方法。
        """
        status = status if status is not None else self.device.status()
        self._remember_device_status(status)
        screen_size = status.get("screen_size")
        # DeviceHub may briefly report a connected device with no frame size
        # while a foreground app is being relaunched or the stream is rebuilt.
        for _ in range(6):
            if isinstance(screen_size, (list, tuple)) and len(screen_size) == 2:
                break
            time.sleep(0.4)
            status = self.device.status()
            screen_size = status.get("screen_size")
        if not isinstance(screen_size, (list, tuple)) or len(screen_size) != 2:
            print("[context] DeviceHub status 暂无 screen_size，保留旧坐标映射")
            return
        w, h = screen_size
        if "landscape" in str(status.get("orientation", "")) or w >= h:
            self.device.set_coord_mapper(None)
        else:
            pw, ph = int(w), int(h)

            def portrait_mapper(x, y, iw=None, ih=None):
                # iw/ih are the logical landscape screenshot dimensions. The
                # portrait output dimensions are therefore ih/iw.
                if iw and ih:
                    return ih - y, x, ih, iw
                return pw - y, x, pw, ph

            self.device.set_coord_mapper(portrait_mapper)

    def sleep(self, ms: float) -> None:
        time.sleep(ms / 1000)

    def capture_bgr(self, *, after_version: int | None = None,
                    timeout_ms: int = 250) -> np.ndarray:
        if after_version is None:
            png = self.device.screenshot_png()
        else:
            png = self.device.observe_game_png(
                after_version=after_version,
                timeout_ms=timeout_ms,
                max_dim=0,
            )
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("截图解码失败")
        if img.shape[0] > img.shape[1]:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_w, frame_h = img.shape[1], img.shape[0]
        if (frame_w, frame_h) != (self.transform.device_width, self.transform.device_height):
            self.transform = ScreenTransform(frame_w, frame_h)
            if hasattr(self, "input"):
                self.input.set_transform(self.transform)
        t = self.transform
        if (img.shape[1], img.shape[0]) != (t.device_width, t.device_height):
            img = cv2.resize(img, (t.device_width, t.device_height))
        with self._frame_lock:
            self._last_frame = img
            self._last_frame_at = time.monotonic()
            self._frame_generation += 1
        return img

    def capture_bgr_after_frame(self, after_version: int | None = None,
                                timeout_ms: int = 250) -> np.ndarray:
        """Capture an ungridded frame after a low-latency input action."""
        return self.capture_bgr(after_version=after_version, timeout_ms=timeout_ms)

    def cached_frame(self) -> tuple[np.ndarray | None, float]:
        """Return a copy of the latest frame without requesting a device screenshot.

        Web preview callers use this path so their polling cannot add MCP screenshot
        requests while a task or a realtime trigger is consuming the capture stream.
        """
        with self._frame_lock:
            if self._last_frame is None:
                return None, float("inf")
            age = max(0.0, time.monotonic() - self._last_frame_at)
            return self._last_frame.copy(), age

    @property
    def frame_generation(self) -> int:
        """Monotonic token for the cached frame used by WebUI response caches."""

        with self._frame_lock:
            return int(getattr(self, "_frame_generation", 0))

    def capture_region(self) -> ImageRegion:
        return ImageRegion(self, self.capture_bgr())

    def launch_game(self, *, auto_enter: bool = True) -> None:
        self.device.launch_app(self.game_bundle_id, wait=True)
        self.sleep(3000)
        # 前台切换后 HID 注入通道可能失效（实测门界面点按只在重连后生效），
        # 启动游戏后主动重建设备通道再刷新朝向映射。
        try:
            self.device.reconnect_device()
            self.sleep(2000)
        except Exception as e:
            print(f"[context] 设备重连失败（忽略）：{e}")
        self.refresh_orientation()
        if auto_enter:
            self.enable_trigger("GameLoading")

    _trigger_loop = None

    @property
    def triggers(self):
        """懒加载的实时触发器帧循环（TriggerLoop）。"""
        if self._trigger_loop is None:
            from ..triggers.loop import TriggerLoop
            self._trigger_loop = TriggerLoop(self)
        return self._trigger_loop

    def enable_trigger(self, name: str, **kwargs) -> None:
        if name == "AutoPick":
            from ..triggers.autopick import AutoPickTrigger
            self.triggers.add(AutoPickTrigger(self, log=self.triggers.log, **kwargs))
        elif name == "AutoSkip":
            from ..triggers.autoskip import AutoSkipTrigger
            self.triggers.add(AutoSkipTrigger(self, log=self.triggers.log, **kwargs))
        elif name in ("AutoEat", "自动吃药"):
            from ..tasks.auto_eat import AutoEatTrigger
            self.triggers.add(AutoEatTrigger(self, log=self.triggers.log, **kwargs))
        elif name in ("MapMask", "地图遮罩"):
            from ..triggers.map_mask import MapMaskTrigger
            self.triggers.add(MapMaskTrigger(self, log=self.triggers.log, **kwargs))
        elif name in ("SkillCd", "技能冷却"):
            from ..triggers.skill_cd import SkillCdTrigger
            self.triggers.add(SkillCdTrigger(self, log=self.triggers.log, **kwargs))
        elif name in ("GameLoading", "自动开门"):
            from ..triggers.game_loading import GameLoadingTrigger
            self.triggers.add(GameLoadingTrigger(self, log=self.triggers.log, **kwargs))
        elif name in ("QuickTeleport", "快速传送"):
            from ..triggers.quick_teleport import QuickTeleportTrigger
            self.triggers.add(QuickTeleportTrigger(self, log=self.triggers.log, **kwargs))
        elif name in ("AutoFish", "AutoFishing", "自动钓鱼"):
            from ..triggers.autofishing import AutoFishingTrigger
            self.triggers.add(AutoFishingTrigger(self, log=self.triggers.log, **kwargs))
        else:
            raise ValueError(
                "未知触发器 "
                f"{name}（支持 AutoPick/AutoSkip/AutoEat/AutoFish/MapMask/SkillCd/"
                "GameLoading/QuickTeleport）"
            )
        self.triggers.start()

    def close(self) -> None:
        if self._trigger_loop is not None:
            self._trigger_loop.stop()
            self._trigger_loop.clear()
        self.input.release_all()
        self.device.close()
