"""GameContext：设备连接 + 坐标变换 + 输入模拟的聚合，任务与脚本的运行环境。"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from ..device.client import DEFAULT_URL, DeviceClient
from ..input.layout import DEFAULT_LAYOUT, ControlLayout
from ..input.simulator import InputSimulator
from ..vision.coordinate import ScreenTransform
from .recognition import ImageRegion

GENSHIN_BUNDLE_ID = "com.miHoYo.Yuanshen"


class GameContext:
    def __init__(self, mcp_url: str = DEFAULT_URL, layout_path: str | Path = DEFAULT_LAYOUT):
        self.device = DeviceClient(mcp_url)
        status = self.device.status()
        if status.get("status") != "connected":
            raise RuntimeError(f"设备未连接（status={status.get('status')}），请检查 DeviceHub Mask")
        w, h = status["screen_size"]
        if h > w:
            w, h = h, w
        self.transform = ScreenTransform(int(w), int(h))
        self.layout = ControlLayout.load(layout_path)
        self.refresh_orientation(status)
        self.input = InputSimulator(self.device, self.layout, self.transform)
        self._start_orientation_watch()

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
                    ori = str(st.get("orientation", ""))
                    if ori != last:
                        if last is not None:
                            print(f"[context] 屏幕朝向变化 {last} → {ori}，更新触控映射")
                        self.refresh_orientation(st)
                        last = ori
                except Exception:
                    pass  # 设备暂不可用时静默重试

        threading.Thread(target=watch, daemon=True, name="orientation-watch").start()

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
        status = status or self.device.status()
        w, h = status["screen_size"]
        if "landscape" in str(status.get("orientation", "")) or w >= h:
            self.device.set_coord_mapper(None)
        else:
            pw, ph = int(w), int(h)
            self.device.set_coord_mapper(lambda x, y: (pw - y, x, pw, ph))

    def sleep(self, ms: float) -> None:
        time.sleep(ms / 1000)

    def capture_bgr(self) -> np.ndarray:
        png = self.device.screenshot_png()
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("截图解码失败")
        if img.shape[0] > img.shape[1]:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        t = self.transform
        if (img.shape[1], img.shape[0]) != (t.device_width, t.device_height):
            img = cv2.resize(img, (t.device_width, t.device_height))
        return img

    def capture_region(self) -> ImageRegion:
        return ImageRegion(self, self.capture_bgr())

    def launch_game(self) -> None:
        self.device.launch_app(GENSHIN_BUNDLE_ID, wait=True)
        self.sleep(3000)
        # 前台切换后 HID 注入通道可能失效（实测门界面点按只在重连后生效），
        # 启动游戏后主动重建设备通道再刷新朝向映射。
        try:
            self.device.reconnect_device()
            self.sleep(2000)
        except Exception as e:
            print(f"[context] 设备重连失败（忽略）：{e}")
        self.refresh_orientation()

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
        else:
            raise ValueError(f"未知触发器 {name}（支持 AutoPick/AutoSkip）")
        self.triggers.start()

    def close(self) -> None:
        self.input.release_all()
        self.device.close()
