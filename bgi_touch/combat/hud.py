"""战斗 HUD 视觉判定：按名切人、技能就绪、（粗略）战斗结束。"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from ..engine.context import GameContext
from ..engine.recognition import RecognitionObject


class TeamSwitcher:
    """OCR 右侧队伍名牌按角色名切人（无需预配置 party.json）。"""

    def __init__(self, ctx: GameContext, log: Callable[[str], None] = print):
        self.ctx = ctx
        self.log = log

    def switch_by_name(self, name: str) -> bool:
        region = self.ctx.capture_region()
        # 名牌在头像左侧：OCR 右侧竖条（ref 空间）
        hits = region.find_multi(RecognitionObject.ocr(1560, 120, 360, 480), limit=8)
        for h in hits:
            if name in h.text:
                # 点名牌同行的头像列位置
                rows = [(abs(h.y - r[1] * 1080), i) for i, r in enumerate(
                    [self.ctx.layout.buttons.get(f"partyRow{i}", (0.96, 0)) for i in (1, 2, 3)], 1)]
                row = min(rows)[1]
                self.ctx.input.tap_button(f"partyRow{row}")
                self.ctx.sleep(700)
                return True
        return False  # 名字未出现 = 已是当前角色或不在队伍


def is_skill_ready(ctx: GameContext, frame: np.ndarray | None = None) -> bool:
    """元素战技就绪判定：冷却时按钮加暗色遮罩，亮度显著降低。"""
    if frame is None:
        frame = ctx.capture_bgr()
    nx, ny = ctx.layout.buttons["skill"]
    h, w = frame.shape[:2]
    r = int(0.028 * w)
    x, y = int(nx * w), int(ny * h)
    crop = frame[max(0, y - r):y + r, max(0, x - r):x + r]
    if crop.size == 0:
        return True
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 2].mean()) > 120  # 阈值经验值，必要时按机型标定


def enemies_nearby(ctx: GameContext, frame: np.ndarray | None = None) -> bool:
    """粗略战斗中判定：画面上方中部是否存在敌方血条（红/白细横条）。

    原版用经验条/YOLO 检测战斗结束；此为轻量启发式，供 combat `check` 使用。
    """
    if frame is None:
        frame = ctx.capture_bgr()
    h, w = frame.shape[:2]
    band = frame[int(0.04 * h):int(0.30 * h), int(0.25 * w):int(0.75 * w)]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    # 血条红：H≈0/180 高饱和
    red = ((hsv[..., 0] < 8) | (hsv[..., 0] > 172)) & (hsv[..., 1] > 150) & (hsv[..., 2] > 120)
    rows = red.sum(axis=1)
    return bool((rows > band.shape[1] * 0.05).any())
