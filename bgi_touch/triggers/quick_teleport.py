"""BetterGI QuickTeleport trigger adapted to the mobile big-map UI."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from ..engine.recognition import Mat, RecognitionObject
from ..vision.game_ui import is_big_map_ui

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "templates" / "quick_teleport"
OPTION_ICONS = (
    "TeleportWaypoint", "StatueOfTheSeven", "Domain", "Domain2",
    "ObsidianTotemPole", "PortableWaypoint", "Mansion", "SubSpaceWaypoint",
    "NodKraiMeetingPoint", "TabletOfTona", "MarkTransPointMoonTower",
)


class QuickTeleportTrigger:
    name = "QuickTeleport"

    def __init__(
        self,
        ctx,
        *,
        teleport_list_click_delay_ms: int = 200,
        wait_teleport_panel_delay_ms: int = 50,
        hotkey_tp_enabled: bool = False,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
        big_map_detector=is_big_map_ui,
    ):
        self.ctx = ctx
        self.enabled = True
        self.log = log
        self._clock = clock
        self._big_map_detector = big_map_detector
        self.teleport_list_click_delay_ms = max(0, min(5000, int(teleport_list_click_delay_ms)))
        self.wait_teleport_panel_delay_ms = max(0, min(5000, int(wait_teleport_panel_delay_ms)))
        self.hotkey_tp_enabled = bool(hotkey_tp_enabled)
        self._armed_until = float("inf") if not self.hotkey_tp_enabled else float("-inf")
        self._last_execute_at = float("-inf")
        self._last_option_at = float("-inf")
        self._last_teleport_at = float("-inf")
        self._teleport = self._template("GoTeleport", (1440, 960, 120, 120), 0.70)
        self._map_close = self._template("MapCloseButton", (1600, 0, 320, 140), 0.65)
        self._map_choose = self._template("MapChoose", (1440, 0, 300, 100), 0.72)
        self._option_templates = [
            self._template(name, (1281, 80, 639, 900), 0.70 if name == "TeleportWaypoint" else 0.78)
            for name in OPTION_ICONS
        ]

    @staticmethod
    def _load(name: str) -> Mat:
        return Mat.from_file(str(ASSETS / f"{name}.png"))

    @classmethod
    def _template(cls, name: str, roi, threshold: float) -> RecognitionObject:
        ro = RecognitionObject.template_match(cls._load(name), *roi)
        ro.threshold = threshold
        ro.name = name
        return ro

    def activate(self, duration_s: float = 1.5) -> None:
        """Arm one manual quick-teleport tick for WebUI/hotkey integrations."""
        self._armed_until = self._clock() + max(0.3, float(duration_s))

    def on_frame(self, region) -> None:
        if not self.enabled:
            return
        now = self._clock()
        if now - self._last_execute_at < 0.3:
            return
        if self.hotkey_tp_enabled and now > self._armed_until:
            return
        self._last_execute_at = now

        teleport = region.find(self._teleport)
        if teleport.is_exist():
            if now - self._last_teleport_at >= 0.8:
                teleport.click()
                self._last_teleport_at = now
                self._armed_until = float("-inf") if self.hotkey_tp_enabled else float("inf")
                self.log("[QuickTeleport] 点击传送")
            return

        in_big_map = bool(self._big_map_detector(self.ctx, region.bgr))
        map_close = region.find(self._map_close).is_exist()
        map_choose = region.find(self._map_choose).is_exist()
        if (map_close or map_choose) and in_big_map:
            # Plain map view: no selected point and no overlapping-point list.
            return

        candidates = []
        for template in self._option_templates:
            candidates.extend(region.find_multi(template, limit=8))
        candidates.sort(key=lambda hit: hit.y)
        for icon in candidates:
            text = region.find(RecognitionObject.ocr(
                icon.x + icon.width,
                max(0.0, icon.y - 8),
                240,
                icon.height + 16,
            ))
            label = "".join(str(text.text or "").replace(">", "").split())
            if len(label) <= 1:
                continue
            if now - self._last_option_at < 0.5:
                return
            if self.teleport_list_click_delay_ms:
                self.ctx.sleep(self.teleport_list_click_delay_ms)
            self.ctx.input.click_ref(
                icon.x + icon.width + 100,
                icon.y + icon.height / 2,
            )
            self._last_option_at = now
            self.log(f"[QuickTeleport] 点击候选点：{label}")
            # Do not force a second capture. TriggerLoop's next frame checks
            # the teleport panel after the configured settle delay naturally.
            if self.wait_teleport_panel_delay_ms:
                self._last_execute_at += self.wait_teleport_panel_delay_ms / 1000
            return

    def close(self) -> None:
        self.enabled = False
