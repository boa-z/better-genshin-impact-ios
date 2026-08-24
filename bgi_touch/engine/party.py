"""触控版队伍配置切换。

BetterGI 在桌面端通过 ``SwitchPartyTask`` 识别队伍配置按钮、OCR 队伍名并
滚动列表。本模块保留同一套界面语义，输入改为 DeviceHub 的 L 键、触控滑动
和 OCR，供 ``genshin.switchParty`` 使用。
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable

from .recognition import Mat, RecognitionObject

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WHITE_CONFIRM_TEMPLATE = (
    PROJECT_ROOT / "assets" / "templates" / "artifact_salvage" / "btn_white_confirm.png"
)

# The bottom-left party bar and the team list use stable 1080p reference areas
# in the upstream UI. ScreenTransform keeps these coordinates safe on the
# iPhone 13 Pro Max wide canvas.
PARTY_NAME_ROI = (34, 950, 420, 130)
PARTY_TITLE_ROI = (0, 0, 520, 150)
PARTY_LIST_ROI = (0, 80, 980, 820)
PARTY_LIST_CONFIRM_ROI = (0, 760, 960, 320)
PARTY_PAGE_CONFIRM_ROI = (960, 760, 960, 320)
PARTY_SELECTOR_POINT = (140, 1020)
PARTY_LIST_MAX_PAGES = 16
PARTY_OPEN_TIMEOUT_S = 7.0
PARTY_LIST_SETTLE_MS = 450
CHARACTER_GRID_ROI = (24, 86, 766, 743)
CHARACTER_SLOT_POINTS = ((470, 550), (800, 550), (1130, 550), (1460, 550))
CHARACTER_CONFIRM_ROI = (350, 950, 180, 90)
CHARACTER_LIST_MAX_PAGES = 12


def _compact_text(value: str) -> str:
    return (
        str(value or "")
        .replace("\r", "")
        .replace("\n", "")
        .replace(" ", "")
        .replace("\t", "")
        .replace('"', "")
        .replace("“", "")
        .replace("”", "")
        .strip()
    )


class PartySwitcher:
    """Switch a named in-game party and leave the game on the main UI."""

    def __init__(
        self,
        ctx,
        *,
        log: Callable[[str], None] = print,
        return_main_ui: Callable[[], bool] | None = None,
    ):
        self.ctx = ctx
        self.log = log
        self.return_main_ui = return_main_ui
        self._confirm_template = None

    @staticmethod
    def _matches(text: str, pattern: str) -> bool:
        text = _compact_text(text)
        pattern = _compact_text(pattern)
        if not text or not pattern:
            return False
        try:
            return re.search(pattern, text) is not None
        except re.error:
            return pattern in text

    def _ocr(self, roi: tuple[float, float, float, float], limit: int = 60):
        return self.ctx.capture_region().find_multi(
            RecognitionObject.ocr(*roi), limit=limit,
        )

    def _current_team_name(self) -> str:
        hits = self._ocr(PARTY_NAME_ROI, limit=10)
        return _compact_text("".join(str(hit.text) for hit in hits))

    def _party_setup_visible(self) -> bool:
        title = "".join(str(hit.text) for hit in self._ocr(PARTY_TITLE_ROI, limit=20))
        if "队伍配置" in _compact_text(title):
            return True
        # The custom team name is the useful fallback when the title is partly
        # hidden by a loading animation or localized differently.
        return bool(self._current_team_name())

    def _wait_for_party_setup(self) -> bool:
        deadline = time.monotonic() + PARTY_OPEN_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                if self._party_setup_visible():
                    return True
            except Exception as error:
                self.log(f"[genshin] 队伍配置 OCR 失败，继续等待：{error}")
            self.ctx.sleep(300)
        self.log("[genshin] 未能打开队伍配置界面")
        return False

    def _open_team_list(self) -> None:
        # PartyBtnChooseView lives in the left-most 1/7 of the bottom bar in
        # the upstream layout. KeyL itself opens the party page; this tap opens
        # the named-team selector inside that page.
        self.ctx.input.click_ref(*PARTY_SELECTOR_POINT)
        self.ctx.sleep(PARTY_LIST_SETTLE_MS)

    def _confirm_recognition(self, roi):
        if self._confirm_template is None:
            self._confirm_template = Mat.from_file(str(WHITE_CONFIRM_TEMPLATE))
        recognition = RecognitionObject.template_match(
            self._confirm_template, *roi,
        )
        recognition.use_3_channels = True
        recognition.threshold = 0.72
        return self.ctx.capture_region().find(recognition)

    def _click_confirm(self, roi) -> bool:
        try:
            hit = self._confirm_recognition(roi)
            if hit.is_exist():
                hit.click()
                self.ctx.sleep(400)
                return True
        except Exception as error:
            self.log(f"[genshin] 队伍确认按钮识别失败：{error}")
        return False

    def _scroll_team_list(self) -> bool:
        vertical_scroll = getattr(self.ctx.input, "vertical_scroll", None)
        if not callable(vertical_scroll):
            return False
        # Negative BetterGI wheel direction maps to a finger-up gesture, which
        # advances toward later team entries in the mobile list.
        vertical_scroll(-3)
        self.ctx.sleep(450)
        return True

    def _leave_main_ui(self) -> bool:
        if callable(self.return_main_ui):
            return bool(self.return_main_ui())
        for _ in range(4):
            self.ctx.input.key_press("ESCAPE")
            self.ctx.sleep(500)
        return True

    def switch(self, party_name: str) -> bool:
        target = _compact_text(party_name)
        if not target:
            self.log("[genshin] 队伍名称为空")
            return False

        main_ui_returned = False

        def leave_main_ui() -> bool:
            nonlocal main_ui_returned
            if main_ui_returned:
                return True
            main_ui_returned = True
            return self._leave_main_ui()

        if callable(self.return_main_ui) and not self.return_main_ui():
            return False

        try:
            self.ctx.input.key_press("L")
            if not self._wait_for_party_setup():
                return False

            current = self._current_team_name()
            self.log(f"[genshin] 当前队伍：{current or '未识别'}，目标：{party_name}")
            if self._matches(current, target):
                self.log(f"[genshin] 当前队伍已是目标队伍：{party_name}")
                return leave_main_ui()

            self._open_team_list()
            for page in range(PARTY_LIST_MAX_PAGES):
                hits = self._ocr(PARTY_LIST_ROI, limit=80)
                target_hit = next(
                    (hit for hit in hits if self._matches(hit.text, target)),
                    None,
                )
                if target_hit is not None:
                    self.log(f"[genshin] 找到队伍：{target_hit.text.strip()}")
                    target_hit.click()
                    self.ctx.sleep(PARTY_LIST_SETTLE_MS)
                    if not self._click_confirm(PARTY_LIST_CONFIRM_ROI):
                        self.log("[genshin] 队伍列表确认失败")
                        return False
                    if not self._click_confirm(PARTY_PAGE_CONFIRM_ROI):
                        self.log("[genshin] 队伍页面确认失败")
                        return False
                    self.log(f"[genshin] 已切换队伍：{party_name}")
                    return leave_main_ui()

                if not hits or not self._scroll_team_list():
                    break
                self.log(f"[genshin] 队伍列表未找到目标，继续第 {page + 2} 页")

            self.log(f"[genshin] 未找到队伍：{party_name}")
            return False
        finally:
            # The caller may invoke another task immediately after a failed
            # OCR/page transition. Always close an opened list before leaving.
            if not main_ui_returned:
                self._leave_main_ui()


class CharacterSwitcher(PartySwitcher):
    """Replace selected physical party slots with OCR-matched characters."""

    def _click_character_confirm(self) -> bool:
        # BetterGI's role state machine uses the small bottom-left “更换” /
        # “加入” text region. OCR is preferred; the fixed reference point is a
        # safe fallback after a character card has already been selected.
        for hit in self._ocr(CHARACTER_CONFIRM_ROI, limit=10):
            if self._matches(hit.text, "更换") or self._matches(hit.text, "加入"):
                hit.click()
                self.ctx.sleep(PARTY_LIST_SETTLE_MS)
                return True
        self.ctx.input.click_ref(425, 1018)
        self.ctx.sleep(PARTY_LIST_SETTLE_MS)
        return True

    def _select_character(self, name: str) -> bool:
        for page in range(CHARACTER_LIST_MAX_PAGES):
            hits = self._ocr(CHARACTER_GRID_ROI, limit=120)
            target = next(
                (hit for hit in hits if self._matches(hit.text, name)),
                None,
            )
            if target is not None:
                self.log(f"[genshin] 找到角色：{target.text.strip()}")
                target.click()
                self.ctx.sleep(PARTY_LIST_SETTLE_MS)
                return self._click_character_confirm()
            if not hits or not self._scroll_team_list():
                break
            self.log(f"[genshin] 角色列表未找到 {name}，继续第 {page + 2} 页")
        return False

    def switch_characters(
        self,
        roles: list[str],
        *,
        use_physical_slots: bool = True,
    ) -> bool:
        requested = [_compact_text(role) for role in roles]
        if not any(requested):
            self.log("[genshin] 未指定需要切换的角色")
            return False
        if len([role for role in requested if role]) != len({role for role in requested if role}):
            self.log("[genshin] 同一角色不能同时指定到多个队伍槽位")
            return False

        assignments = (
            [(index + 1, role) for index, role in enumerate(requested) if role]
            if use_physical_slots
            else [
                (slot, role)
                for slot, role in zip(
                    (1, 2, 3, 4), (role for role in requested if role), strict=False,
                )
            ]
        )
        main_ui_returned = False

        def leave_main_ui() -> bool:
            nonlocal main_ui_returned
            if main_ui_returned:
                return True
            main_ui_returned = True
            return self._leave_main_ui()

        if callable(self.return_main_ui) and not self.return_main_ui():
            return False
        try:
            self.ctx.input.key_press("L")
            if not self._wait_for_party_setup():
                return False
            for slot, name in assignments:
                if slot > len(CHARACTER_SLOT_POINTS):
                    return False
                self.ctx.input.click_ref(*CHARACTER_SLOT_POINTS[slot - 1])
                self.ctx.sleep(PARTY_LIST_SETTLE_MS)
                if not self._select_character(name):
                    self.log(f"[genshin] 未找到角色：{name}")
                    return False
            self.log("[genshin] 角色队伍重组完成")
            return leave_main_ui()
        finally:
            if not main_ui_returned:
                self._leave_main_ui()
