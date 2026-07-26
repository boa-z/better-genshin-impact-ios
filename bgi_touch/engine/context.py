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
        self._raw_portrait = h > w  # 设备帧是否为竖屏（横屏游戏内容旋转 90° 呈现）
        self._raw_size = (int(w), int(h))
        if self._raw_portrait:
            w, h = h, w
        self.transform = ScreenTransform(int(w), int(h))
        self.layout = ControlLayout.load(layout_path)
        if self._raw_portrait:
            # 横屏逻辑坐标 L → 竖屏截图坐标 P：横屏画面 = 竖屏帧逆时针转 90°，
            # 逆变换为 P.x = P_W - L.y，P.y = L.x
            pw, ph = self._raw_size
            self.device.set_coord_mapper(lambda x, y: (pw - y, x, pw, ph))
        self.input = InputSimulator(self.device, self.layout, self.transform)

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

    def close(self) -> None:
        self.input.release_all()
        self.device.close()
