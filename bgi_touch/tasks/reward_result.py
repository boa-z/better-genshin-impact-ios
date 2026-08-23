"""BetterGI-compatible reward result card recognition for the iOS client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np

from ..engine.context import GameContext
from ..vision.item_recognizer import ItemIconRecognizer
from .inventory_grid import recognize_inventory_count


REWARD_BAND = (220, 444, 1480, 220)


@dataclass(frozen=True)
class RewardItem:
    name: str
    quantity: int
    quality_level: int = -1


def crop_reward_band(ctx: GameContext, frame: np.ndarray) -> np.ndarray:
    """Normalize the centered reward strip to BetterGI's 1480x220 space."""
    x, y, width, height = REWARD_BAND
    scale = ctx.transform.scale
    x0, y0 = ctx.transform.to_device(x, y, anchor="center")
    crop = frame[
        max(0, round(y0)):min(frame.shape[0], round(y0 + height * scale)),
        max(0, round(x0)):min(frame.shape[1], round(x0 + width * scale)),
    ]
    if crop.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)


def detect_reward_card_rects(band_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Port RewardResultRecognizer.DetectCardRects from current BetterGI."""
    if band_bgr.shape[:2] != (220, 1480):
        return []
    hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 190), (179, 20, 249))
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    strips = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if not 68.75 <= width <= 168.75:
            continue
        if not 15.3 <= height <= 76.5 or area < 1200:
            continue
        if y + height / 2 < 88:
            continue
        strips.append((x, y, width, height))
    if not strips:
        return []
    strips.sort(key=lambda rect: rect[1] + rect[3])
    median_bottom = strips[len(strips) // 2][1] + strips[len(strips) // 2][3]
    strips = [
        rect for rect in strips
        if abs(rect[1] + rect[3] - median_bottom) <= 76.5
    ]
    if not strips:
        return []
    strips.sort(key=lambda rect: rect[1] + rect[3])
    row_bottom = strips[len(strips) // 2][1] + strips[len(strips) // 2][3]
    cards = []
    for x, _y, width, _height in strips:
        card_x = x + width // 2 - 125 // 2
        card_y = row_bottom - 153
        if card_x >= 0 and card_y >= 0 and card_x + 125 <= 1480 and card_y + 153 <= 220:
            cards.append((card_x, card_y, 125, 153))
    return sorted(cards)


def merge_reward_summary(summary: dict[str, int], rewards: Iterable[RewardItem]) -> None:
    for reward in rewards:
        summary[reward.name] = summary.get(reward.name, 0) + max(1, reward.quantity)


class RewardResultRecognizer:
    def __init__(self, ctx: GameContext, log: Callable[[str], None] = print):
        self.ctx = ctx
        self.log = log
        self.items = ItemIconRecognizer()

    def _recognize_page(self, frame: np.ndarray) -> list[RewardItem]:
        band = crop_reward_band(self.ctx, frame)
        cards = detect_reward_card_rects(band)
        rewards = []
        for index, (x, y, width, height) in enumerate(cards):
            card = band[y:y + height, x:x + width]
            match = self.items.match(card[:125, :125])
            if match.score < 0.75:
                self.log(
                    f"[Reward] 跳过低置信度卡片 {index + 1}: "
                    f"{match.name or '未知'} ({match.score:.2f})"
                )
                continue
            count = recognize_inventory_count(card)
            quantity = count.count if count.count >= 0 else 1
            if count.count < 0:
                self.log(
                    f"[Reward] {match.name} 数量识别失败（{count.reason}/"
                    f"{count.raw_text}），按 1 个计入"
                )
            rewards.append(RewardItem(match.name, quantity, match.quality_level))
        return rewards

    @staticmethod
    def _duplicate_prefix(current: list[RewardItem], previous: list[RewardItem]) -> int:
        duplicate_count = 0
        tail = previous[-10:]
        for reward in current[:len(previous)]:
            if reward not in tail:
                break
            duplicate_count += 1
        return duplicate_count

    def _next_page(self) -> None:
        scale = self.ctx.transform.scale
        x0, y0 = self.ctx.transform.to_device(1650, 540, anchor="center")
        x1, _ = self.ctx.transform.to_device(250, 540, anchor="center")
        self.ctx.device.swipe(
            x0, y0, x1, y0,
            duration_ms=700,
            image_width=self.ctx.transform.device_width,
            image_height=self.ctx.transform.device_height,
        )
        self.ctx.sleep(1200)

    def recognize_multi_page(self, max_pages: int = 3) -> dict[str, int]:
        summary: dict[str, int] = {}
        previous: list[RewardItem] | None = None
        for page in range(1, max(1, int(max_pages)) + 1):
            if page > 1:
                self._next_page()
            rewards = self._recognize_page(self.ctx.capture_bgr())
            if not rewards:
                self.log("[Reward] 未识别到奖励卡片，结束本轮识别")
                break
            duplicate_count = self._duplicate_prefix(rewards, previous) if previous else 0
            merge_reward_summary(summary, rewards[duplicate_count:])
            if duplicate_count:
                break
            previous = rewards
        self.log(f"[Reward] 本轮识别到 {sum(summary.values())} 个奖励：{summary}")
        return summary
