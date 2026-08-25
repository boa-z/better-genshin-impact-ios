"""触控版队伍配置切换。

BetterGI 在桌面端通过 ``SwitchPartyTask`` 识别队伍配置按钮、OCR 队伍名并
滚动列表。本模块保留同一套界面语义，输入改为 DeviceHub 的 L 键、触控滑动
和 OCR，供 ``genshin.switchParty`` 使用。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .recognition import Mat, RecognitionObject

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WHITE_CONFIRM_TEMPLATE = (
    PROJECT_ROOT / "assets" / "templates" / "artifact_salvage" / "btn_white_confirm.png"
)
SWITCH_CHARACTER_ASSETS = PROJECT_ROOT / "assets" / "templates" / "switch_character"

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

# BetterGI detects these small markers before opening the party configuration
# page.  Keep the reference-space ROIs identical to the upstream assets while
# allowing ScreenTransform to scale them for the iPhone capture.
MULTI_PLAYER_ROI = (1536, 216, 384, 324)
PLAYER_MARKER_ROI = (0, 0, 480, 270)


@dataclass(frozen=True)
class PartyControlLayout:
    """Physical team slots that the current player can edit.

    ``physical_slots`` is deliberately ordered by the logical slot exposed to
    BetterGI scripts.  For example, the second player in a two-player world
    controls physical slots ``(3, 4)``; a script using
    ``usePhysicalSlots=false`` maps its first requested character to slot 3.
    """

    player_count: int
    player_index: int
    physical_slots: tuple[int, ...]


def physical_slots_for_players(player_count: int, player_index: int) -> tuple[int, ...]:
    """Return the physical slots controlled by one player in co-op mode."""

    try:
        count = int(player_count)
        index = int(player_index)
    except (TypeError, ValueError) as error:
        raise ValueError("联机人数和玩家编号必须是整数") from error
    if not 1 <= count <= 4 or not 1 <= index <= count:
        raise ValueError(f"不支持的联机状态：{count} 人、{index}P")
    mapping = {
        (1, 1): (1, 2, 3, 4),
        (2, 1): (1, 2),
        (2, 2): (3, 4),
        (3, 1): (1, 2),
        (3, 2): (3,),
        (3, 3): (4,),
        (4, 1): (1,),
        (4, 2): (2,),
        (4, 3): (3,),
        (4, 4): (4,),
    }
    return mapping[(count, index)]


def build_character_assignments(
    roles: Sequence[str],
    *,
    use_physical_slots: bool = True,
    physical_slots: Iterable[int] = (1, 2, 3, 4),
) -> list[tuple[int, str]]:
    """Map BetterGI's four arguments to physical slots.

    Empty arguments are skipped.  In physical-slot mode an argument keeps its
    original slot and is ignored when that slot belongs to another co-op
    player.  In controllable-order mode the non-empty arguments are assigned
    in order to the current player's physical slots.
    """

    slots: list[int] = []
    for raw_slot in physical_slots:
        try:
            slot = int(raw_slot)
        except (TypeError, ValueError):
            continue
        if 1 <= slot <= 4 and slot not in slots:
            slots.append(slot)
    if not slots:
        slots = [1, 2, 3, 4]

    requested = [_compact_text(role) for role in roles]
    if use_physical_slots:
        allowed = set(slots)
        return [
            (index + 1, role)
            for index, role in enumerate(requested[:4])
            if role and index + 1 in allowed
        ]
    return [
        (slot, role)
        for slot, role in zip(slots, (role for role in requested[:4] if role))
    ]


def _context_value(ctx: Any, *names: str) -> Any:
    for name in names:
        try:
            value = getattr(ctx, name)
        except (AttributeError, TypeError):
            continue
        if value is not None:
            return value
    return None


def _coerce_slots(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return None
    result: list[int] = []
    for raw in value:
        try:
            slot = int(raw)
        except (TypeError, ValueError):
            return None
        if not 1 <= slot <= 4 or slot in result:
            return None
        result.append(slot)
    return tuple(result) if result else None


def _template_exists(region: Any, filename: str, roi: tuple[int, int, int, int]) -> bool:
    path = SWITCH_CHARACTER_ASSETS / filename
    if not path.is_file():
        return False
    try:
        recognition = RecognitionObject.template_match(
            Mat.from_file(str(path)), *roi,
        )
        recognition.use_3_channels = True
        recognition.threshold = 0.72
        return bool(region.find(recognition).is_exist())
    except (OSError, TypeError, ValueError):
        return False


def detect_party_control_layout(
    ctx: Any,
    region: Any | None = None,
    *,
    log: Callable[[str], None] = print,
) -> PartyControlLayout:
    """Detect the current co-op player and its editable physical slots.

    Explicit context values are supported for headless/offline hosts and for
    devices whose HUD theme needs manual calibration.  On a real device the
    same frame is checked for BetterGI's ``P``/``1P`` and ``nP_top_left``
    templates.  Ambiguous or unavailable recognition safely falls back to the
    four solo slots, preserving the old single-player behavior.
    """

    direct = _coerce_slots(_context_value(
        ctx, "party_control_slots", "coop_physical_slots", "operable_party_slots",
    ))
    raw_count = _context_value(ctx, "coop_player_count", "multi_game_player_count")
    raw_index = _context_value(ctx, "coop_player_index", "player_index")
    try:
        if direct is not None:
            return PartyControlLayout(
                1 if direct == (1, 2, 3, 4) else max(2, len(direct)),
                1,
                direct,
            )
        if raw_count is not None and raw_index is not None:
            slots = physical_slots_for_players(int(raw_count), int(raw_index))
            return PartyControlLayout(int(raw_count), int(raw_index), slots)
    except (TypeError, ValueError, KeyError) as error:
        log(f"[genshin] 联机队伍配置无效，回退单机槽位：{error}")

    default = PartyControlLayout(1, 1, (1, 2, 3, 4))
    if region is None:
        try:
            region = ctx.capture_region()
        except Exception:
            return default

    try:
        if _template_exists(region, "stand_alone_icon.png", PLAYER_MARKER_ROI):
            return default

        # The upstream P marker appears once for every other player.  The
        # current account is the additional 1P host, or is identified by the
        # nP_top_left marker when joining another player's world.
        p_path = SWITCH_CHARACTER_ASSETS / "p.png"
        other_players = 0
        if p_path.is_file():
            recognition = RecognitionObject.template_match(
                Mat.from_file(str(p_path)), *MULTI_PLAYER_ROI,
            )
            recognition.use_3_channels = True
            recognition.threshold = 0.72
            other_players = len(region.find_multi(recognition, limit=4))

        matched_indices = [
            index for index in range(1, 5)
            if _template_exists(region, f"{index}P_top_left.png", PLAYER_MARKER_ROI)
        ]
        if not other_players and not matched_indices:
            return default

        player_count = other_players + 1
        if not 1 <= player_count <= 4:
            log(f"[genshin] 联机人数识别异常：{player_count}，回退单机槽位")
            return default
        if len(matched_indices) != 1:
            log(
                "[genshin] 联机玩家编号识别不唯一："
                f"{matched_indices or '无'}，回退单机槽位"
            )
            return default
        player_index = matched_indices[0]
        if player_index > player_count:
            log(
                f"[genshin] 玩家编号 {player_index}P 超出联机人数 {player_count}，"
                "回退单机槽位"
            )
            return default
        slots = physical_slots_for_players(player_count, player_index)
        log(
            f"[genshin] 识别联机队伍：{player_count} 人，当前 {player_index}P，"
            f"可控物理槽位 {slots}"
        )
        return PartyControlLayout(player_count, player_index, slots)
    except Exception as error:
        log(f"[genshin] 联机队伍识别失败，回退单机槽位：{error}")
        return default


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Exposed to GenshinApi so the runtime party cache uses the same
        # physical-slot mapping that was actually clicked on the page.
        self.last_assignments: list[tuple[int, str]] = []

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

        main_ui_returned = False

        def leave_main_ui() -> bool:
            nonlocal main_ui_returned
            if main_ui_returned:
                return True
            main_ui_returned = True
            return self._leave_main_ui()

        if callable(self.return_main_ui) and not self.return_main_ui():
            return False

        control_layout = detect_party_control_layout(
            self.ctx,
            log=self.log,
        )
        assignments = build_character_assignments(
            requested,
            use_physical_slots=bool(use_physical_slots),
            physical_slots=control_layout.physical_slots,
        )
        self.last_assignments = list(assignments)
        ignored = [
            index + 1 for index, role in enumerate(requested[:4])
            if role and (index + 1, role) not in assignments
        ]
        if ignored:
            self.log(
                "[genshin] 当前账号不可操作的角色槽位已忽略："
                + ",".join(str(slot) for slot in ignored)
            )
        try:
            self.ctx.input.key_press("L")
            if not self._wait_for_party_setup():
                return False
            for slot, name in assignments:
                if not 1 <= slot <= len(CHARACTER_SLOT_POINTS):
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
