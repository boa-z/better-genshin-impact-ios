"""1920x1080 脚本坐标系 ↔ 设备屏幕坐标系。

BetterGI 脚本与模板资产以 1920x1080 横屏窗口为基准。iPhone 比 16:9 更宽
（如 2816x1296 ≈ 19.5:9），且原神移动端 HUD 元素贴边锚定，因此按高度等比
缩放后，x 需要按元素落在参考屏的哪一侧做锚点重定位。
"""

from __future__ import annotations

from dataclasses import dataclass

REF_WIDTH = 1920
REF_HEIGHT = 1080


@dataclass(frozen=True)
class ScreenTransform:
    device_width: int
    device_height: int

    @property
    def scale(self) -> float:
        return self.device_height / REF_HEIGHT

    def resolve_anchor(self, ref_x: float, anchor: str = "auto") -> str:
        if anchor != "auto":
            return anchor
        if ref_x < REF_WIDTH / 3:
            return "left"
        if ref_x > REF_WIDTH * 2 / 3:
            return "right"
        return "center"

    def to_device(self, ref_x: float, ref_y: float, anchor: str = "auto") -> tuple[float, float]:
        s = self.scale
        y = ref_y * s
        match self.resolve_anchor(ref_x, anchor):
            case "left":
                x = ref_x * s
            case "right":
                x = self.device_width - (REF_WIDTH - ref_x) * s
            case _:
                x = self.device_width / 2 + (ref_x - REF_WIDTH / 2) * s
        return x, y

    def resolve_device_anchor(self, dev_x: float, anchor: str = "auto") -> str:
        if anchor != "auto":
            return anchor
        s = self.scale
        edge = REF_WIDTH / 3 * s
        extra = max(0.0, self.device_width - REF_WIDTH * s)
        cutoff = edge + extra / 4
        if dev_x < cutoff:
            return "left"
        if dev_x > self.device_width - cutoff:
            return "right"
        return "center"

    def to_ref(self, dev_x: float, dev_y: float,
               anchor: str = "auto") -> tuple[float, float]:
        s = self.scale
        match self.resolve_device_anchor(dev_x, anchor):
            case "left":
                x = dev_x / s
            case "right":
                x = REF_WIDTH - (self.device_width - dev_x) / s
            case _:
                x = REF_WIDTH / 2 + (dev_x - self.device_width / 2) / s
        return x, dev_y / s

    def scale_len(self, ref_len: float) -> float:
        return ref_len * self.scale
