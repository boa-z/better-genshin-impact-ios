"""BetterGI AutoStygianOnslaught state machine for the iOS touch runtime.

The event is driven from the main world by UI state instead of requiring an
account-specific path. Every transition reuses one captured frame for OCR and
template matching so preview clients do not compete with the task for frames.
``route_path`` remains an optional override for existing scripts.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..engine.context import GameContext
from ..engine.genshin_api import GenshinApi
from ..engine.recognition import ImageRegion, Mat, RecognitionObject, Region
from ..pathing.executor import PathingExecutor
from ..pathing.model import PathingTask
from .auto_fight import AutoFightTask
from .common_jobs import exclusive_realtime_triggers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_ROOT = PROJECT_ROOT / "assets" / "templates" / "stygian"


class StygianState(str, Enum):
    UNKNOWN = "Unknown"
    MAIN_WORLD = "MainWorld"
    EVENT_MENU = "EventMenu"
    STYGIAN_PAGE = "StygianOnslaughtPage"
    TELEPORT_MAP = "TeleportMap"
    DOMAIN_ENTRANCE = "DomainEntrance"
    DIFFICULTY_SELECT = "DifficultySelect"
    DOMAIN_LOADING = "DomainLoading"
    DOMAIN_LOBBY = "DomainLobby"
    BOSS_SELECT = "BossSelect"
    BATTLE_ARENA = "BattleArena"
    BATTLE_LOADING = "BattleLoading"
    IN_BATTLE = "InBattle"
    BATTLE_RESULT_WIN = "BattleResultWin"
    BATTLE_RESULT_LOSE = "BattleResultLose"
    LEYLINE_FLOWER_PROMPT = "LeylineFlowerPrompt"
    RESIN_SELECT = "ResinSelect"
    CONTINUE_OR_EXIT = "ContinueOrExit"
    EXITING = "Exiting"


@dataclass(frozen=True)
class StygianSignals:
    """Template and positional signals used by the pure state detector."""

    paimon_menu: bool = False
    teleport_button: bool = False
    leyline_disorder: bool = False
    inventory: bool = False
    white_cancel: bool = False
    white_confirm: bool = False
    continue_button: bool = False
    exit_button: bool = False
    black_confirm: bool = False
    exit_door: bool = False
    event_menu_title: bool = False
    event_page_title: bool = False
    domain_entrance_title: bool = False


@dataclass
class StygianSnapshot:
    region: ImageRegion
    hits: list[Region]
    texts: tuple[str, ...]
    signals: StygianSignals


@dataclass
class ResinUseRecord:
    name: str
    remaining: int
    maximum: int


@dataclass(frozen=True)
class ResinClaim:
    success: bool
    is_last: bool
    name: str = ""
    exhausted: bool = False


_RESIN_NAMES = ("浓缩树脂", "原粹树脂", "须臾树脂", "脆弱树脂")
_FULL_WIDTH_NUMBERS = str.maketrans("０１２３４５６７８９", "0123456789")


def _clean_text(text: object) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def detect_stygian_state(
    texts: Iterable[str], signals: StygianSignals = StygianSignals(),
) -> StygianState:
    """Classify a single frame using upstream detector priority."""

    normalized = tuple(_clean_text(text) for text in texts)
    joined = "".join(normalized)

    def has(*words: str) -> bool:
        return any(word in text for word in words for text in normalized)

    if (signals.continue_button and signals.exit_button) or (
        has("继续挑战", "继续") and has("退出秘境", "退出")
    ):
        return StygianState.CONTINUE_OR_EXIT
    if signals.teleport_button:
        return StygianState.TELEPORT_MAP
    if signals.leyline_disorder and signals.inventory:
        return StygianState.DOMAIN_LOBBY
    if signals.leyline_disorder and not signals.inventory:
        return StygianState.BATTLE_ARENA
    if signals.paimon_menu:
        return StygianState.MAIN_WORLD
    if (signals.white_cancel and has("返回")) or (
        has("挑战成功", "挑战达成", "挑战完成") and has("返回")
    ):
        return StygianState.BATTLE_RESULT_WIN
    if (signals.white_confirm and has("挑战失败", "重新挑战")) or has("挑战失败"):
        return StygianState.BATTLE_RESULT_LOSE
    if "地脉之花" in joined and any(name in joined for name in _RESIN_NAMES):
        return StygianState.RESIN_SELECT
    if "地脉之花" in joined:
        return StygianState.LEYLINE_FLOWER_PROMPT
    if has("角色预览") and has("开始挑战"):
        return StygianState.BOSS_SELECT
    if has("单人挑战"):
        return StygianState.DIFFICULTY_SELECT
    if signals.domain_entrance_title:
        return StygianState.DOMAIN_ENTRANCE
    if signals.event_menu_title:
        return StygianState.EVENT_MENU
    if signals.event_page_title:
        return StygianState.STYGIAN_PAGE
    return StygianState.UNKNOWN


def build_resin_plan(
    *, specify: bool, priority: Sequence[str] | None = None,
    original: int = 0, condensed: int = 0, transient: int = 0, fragile: int = 0,
) -> list[ResinUseRecord]:
    """Build a fixed-use plan while honoring BetterGI's priority setting."""

    if not specify:
        return []
    counts = {
        "原粹树脂": max(0, int(original)),
        "浓缩树脂": max(0, int(condensed)),
        "须臾树脂": max(0, int(transient)),
        "脆弱树脂": max(0, int(fragile)),
    }
    wanted: list[str] = []
    for raw in priority or ("浓缩树脂", "原粹树脂"):
        name = _clean_text(raw)
        if name in counts and name not in wanted:
            wanted.append(name)
    for name in _RESIN_NAMES:
        if counts[name] > 0 and name not in wanted:
            wanted.append(name)
    plan = [
        ResinUseRecord(name, counts[name], counts[name])
        for name in wanted if counts[name] > 0
    ]
    if not plan:
        raise ValueError("选择了指定树脂刷取次数，请至少配置一种树脂的刷取次数")
    return plan


def resin_count_from_lines(lines: Iterable[str], resin_name: str) -> int | None:
    """Best-effort count extraction from a reward panel OCR line."""

    normalized = [_clean_text(line).translate(_FULL_WIDTH_NUMBERS) for line in lines]
    for index, line in enumerate(normalized):
        if resin_name not in line:
            continue
        candidates = [line]
        if index + 1 < len(normalized):
            candidates.append(normalized[index + 1])
        for candidate in candidates:
            suffix = candidate.split(resin_name, 1)[-1] if resin_name in candidate else candidate
            match = re.search(r"(?<!\d)(\d{1,3})(?:/\d+)?", suffix)
            if match:
                return int(match.group(1))
    return None


def choose_stygian_resin(
    lines: Iterable[str], plan: Sequence[ResinUseRecord] | None = None,
) -> str | None:
    """Choose a visible resin option using automatic or fixed-use policy."""

    joined = "".join(_clean_text(line) for line in lines)
    original_visible = "原粹树脂" in joined and not (
        "数量不足" in joined or "补充原粹树脂" in joined
    )
    visible = {
        "浓缩树脂": "浓缩树脂" in joined,
        "原粹树脂": original_visible,
        "须臾树脂": "须臾树脂" in joined,
        "脆弱树脂": "脆弱树脂" in joined,
    }
    if plan is not None:
        return next(
            (record.name for record in plan if record.remaining > 0 and visible[record.name]),
            None,
        )
    return next((name for name in ("浓缩树脂", "原粹树脂") if visible[name]), None)


class AutoStygianOnslaughtTask:
    _TEMPLATE_ROI = {
        "paimon_menu": (0, 0, 240, 180),
        "teleport_button": (1370, 760, 550, 320),
        "leyline_disorder": (0, 0, 500, 260),
        "inventory": (1650, 760, 270, 320),
        "white_cancel": (500, 760, 920, 320),
        "white_confirm": (500, 760, 920, 320),
        "continue_button": (760, 760, 520, 320),
        "exit_button": (300, 760, 520, 320),
        "black_confirm": (500, 760, 920, 320),
        "exit_door": (0, 0, 1920, 1080),
    }

    def __init__(
        self,
        ctx: GameContext,
        *,
        route_path: str | Path | None = None,
        boss_num: int = 1,
        combat_strategy_path: str | None = None,
        timeout_s: float = 360.0,
        party_slots: dict[str, int] | None = None,
        auto_artifact_salvage: bool = False,
        specify_resin_use: bool = False,
        resin_priority_list: Sequence[str] | None = None,
        original_resin_use_count: int = 0,
        condensed_resin_use_count: int = 0,
        transient_resin_use_count: int = 0,
        fragile_resin_use_count: int = 0,
        fight_team_name: str = "",
        artifact_salvage_options: dict | None = None,
        max_battle_failures: int = 20,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.route_path = Path(route_path).expanduser() if route_path else None
        self.boss_num = int(boss_num)
        self.log = log
        self.auto_artifact_salvage = bool(auto_artifact_salvage)
        self.specify_resin_use = bool(specify_resin_use)
        self.resin_plan = build_resin_plan(
            specify=self.specify_resin_use,
            priority=resin_priority_list,
            original=original_resin_use_count,
            condensed=condensed_resin_use_count,
            transient=transient_resin_use_count,
            fragile=fragile_resin_use_count,
        )
        self.fight_team_name = str(fight_team_name or "")
        self.artifact_salvage_options = dict(artifact_salvage_options or {})
        self.max_battle_failures = max(1, int(max_battle_failures))
        self.party_slots = party_slots or {}
        self.fight = AutoFightTask(
            ctx,
            combat_strategy_path=combat_strategy_path,
            timeout_s=max(30.0, float(timeout_s)),
            party_slots=self.party_slots,
            log=log,
        )
        self.api = GenshinApi(ctx, log=log)
        self._templates: dict[str, Mat] = {}
        self._last_state = StygianState.UNKNOWN

    def _validate(self) -> None:
        if self.boss_num not in (1, 2, 3):
            raise ValueError("AutoStygianOnslaught.bossNum 必须为 1、2 或 3")
        if self.route_path is not None and not self.route_path.is_file():
            raise FileNotFoundError(f"幽境危战入口路线不存在：{self.route_path}")
        missing = [
            name for name in self._TEMPLATE_ROI
            if not (TEMPLATE_ROOT / f"{name}.png").is_file()
        ]
        if missing:
            raise FileNotFoundError("缺少幽境危战识别模板：" + ", ".join(missing))

    def _ro(self, name: str) -> RecognitionObject:
        if name not in self._templates:
            self._templates[name] = Mat.from_file(str(TEMPLATE_ROOT / f"{name}.png"))
        ro = RecognitionObject.template_match(self._templates[name])
        ro.threshold = 0.76 if name in ("white_cancel", "white_confirm") else 0.8
        ro.roi = self._TEMPLATE_ROI[name]
        return ro

    @staticmethod
    def _inside(hit: Region, x: float, y: float, w: float, h: float) -> bool:
        return x <= hit.x <= x + w and y <= hit.y <= y + h

    def _capture_snapshot(self) -> StygianSnapshot:
        """Capture once and reuse the frame for all state signals."""

        region = self.ctx.capture_region()
        hits = region.find_multi(RecognitionObject.ocr(0, 0, 1920, 1080), limit=100)
        hits.sort(key=lambda hit: (hit.y, hit.x))
        texts = tuple(hit.text for hit in hits)
        event_hits = [hit for hit in hits if "幽境危战" in _clean_text(hit.text)]
        signals = StygianSignals(
            **{
                name: region.find(self._ro(name)).is_exist()
                for name in self._TEMPLATE_ROI
            },
            event_menu_title=any(
                "活动一览" in _clean_text(hit.text)
                and self._inside(hit, 80, 80, 500, 160)
                for hit in hits
            ),
            event_page_title=any(
                self._inside(hit, 1000, 200, 600, 240) for hit in event_hits
            ),
            domain_entrance_title=any(
                self._inside(hit, 1120, 420, 500, 240) for hit in event_hits
            ),
        )
        return StygianSnapshot(region, hits, texts, signals)

    def _state(self, snapshot: StygianSnapshot) -> StygianState:
        state = detect_stygian_state(snapshot.texts, snapshot.signals)
        if state != self._last_state:
            self.log(f"[AutoStygianOnslaught] 状态 {self._last_state.value} → {state.value}")
            self._last_state = state
        return state

    @staticmethod
    def _cancelled(callback: Callable[[], bool] | None) -> bool:
        return bool(callback and callback())

    def _wait_for(
        self,
        states: StygianState | Iterable[StygianState],
        *,
        timeout_s: float,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[StygianState, StygianSnapshot | None]:
        wanted = {states} if isinstance(states, StygianState) else set(states)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._cancelled(cancelled):
                return StygianState.UNKNOWN, None
            snapshot = self._capture_snapshot()
            state = self._state(snapshot)
            if state in wanted:
                return state, snapshot
            self.ctx.sleep(350)
        return StygianState.UNKNOWN, None

    @staticmethod
    def _find_text(snapshot: StygianSnapshot, *keywords: str, roi=None) -> Region | None:
        for hit in snapshot.hits:
            if roi is not None and not AutoStygianOnslaughtTask._inside(hit, *roi):
                continue
            text = _clean_text(hit.text)
            if any(_clean_text(keyword) in text for keyword in keywords):
                return hit
        return None

    def _tap_text(self, snapshot: StygianSnapshot, *keywords: str, roi=None) -> bool:
        hit = self._find_text(snapshot, *keywords, roi=roi)
        if hit is None:
            return False
        hit.click()
        self.ctx.sleep(350)
        return True

    def _tap_template(self, snapshot: StygianSnapshot, name: str, *, double=False) -> bool:
        hit = snapshot.region.find(self._ro(name))
        if not hit.is_exist():
            return False
        hit.click()
        if double:
            self.ctx.sleep(60)
            hit.click()
        self.ctx.sleep(350)
        return True

    def _swipe_ref(
        self, start: tuple[float, float], end: tuple[float, float], duration_ms=650,
    ) -> None:
        t = self.ctx.transform
        x1, y1 = t.to_device(*start)
        x2, y2 = t.to_device(*end)
        self.ctx.device.swipe(
            x1, y1, x2, y2, duration_ms=duration_ms,
            image_width=t.device_width, image_height=t.device_height,
        )
        self.ctx.sleep(500)

    def _open_event_page(
        self, cancelled: Callable[[], bool] | None,
    ) -> StygianSnapshot | None:
        self.log("[AutoStygianOnslaught] 打开活动菜单")
        self.ctx.input.key_press("F5")
        state, snapshot = self._wait_for(
            (StygianState.EVENT_MENU, StygianState.STYGIAN_PAGE),
            timeout_s=15, cancelled=cancelled,
        )
        if state == StygianState.STYGIAN_PAGE:
            return snapshot
        if state != StygianState.EVENT_MENU:
            self.log("[AutoStygianOnslaught] 未能打开活动菜单")
            return None

        gestures = (
            ((343, 720), (343, 330)),
            ((343, 330), (343, 720)),
            ((343, 720), (343, 330)),
        )
        for start, end in gestures:
            assert snapshot is not None
            target = self._find_text(
                snapshot, "幽境危战", roi=(160, 170, 380, 730)
            )
            if target is not None:
                target.click()
                self.ctx.sleep(500)
                state, snapshot = self._wait_for(
                    StygianState.STYGIAN_PAGE, timeout_s=8, cancelled=cancelled,
                )
                return snapshot if state == StygianState.STYGIAN_PAGE else None
            self._swipe_ref(start, end)
            snapshot = self._capture_snapshot()
        self.log("[AutoStygianOnslaught] 活动列表中未找到幽境危战")
        return None

    def _navigate_event(self, cancelled: Callable[[], bool] | None) -> bool:
        if self.route_path is not None:
            self.log(f"[AutoStygianOnslaught] 执行自定义入口路线 {self.route_path.name}")
            if not PathingExecutor(
                self.ctx, party_slots=self.party_slots, log=self.log,
            ).run(PathingTask.load(self.route_path)):
                return False
            self.ctx.input.key_press("F")
        else:
            snapshot = self._open_event_page(cancelled)
            if snapshot is None:
                return False
            detail = "".join(_clean_text(text) for text in snapshot.texts)
            if "紊乱爆发期" in detail and "已结束" in detail:
                self.log("[AutoStygianOnslaught] 紊乱爆发期已结束")
                self.ctx.input.key_press("ESCAPE")
                return False
            if not self._tap_text(
                snapshot, "前往挑战", roi=(850, 600, 1000, 450)
            ):
                self.log("[AutoStygianOnslaught] 未找到前往挑战按钮")
                return False
            state, snapshot = self._wait_for(
                (StygianState.TELEPORT_MAP, StygianState.DOMAIN_ENTRANCE),
                timeout_s=15, cancelled=cancelled,
            )
            if state == StygianState.TELEPORT_MAP and snapshot is not None:
                if not self._tap_template(snapshot, "teleport_button"):
                    return False
                state, snapshot = self._wait_for(
                    StygianState.DOMAIN_ENTRANCE,
                    timeout_s=25,
                    cancelled=cancelled,
                )
            if state != StygianState.DOMAIN_ENTRANCE:
                self.log("[AutoStygianOnslaught] 未到达幽境危战入口")
                return False
            self.ctx.input.key_press("F")

        state, snapshot = self._wait_for(
            StygianState.DIFFICULTY_SELECT, timeout_s=18, cancelled=cancelled,
        )
        if state != StygianState.DIFFICULTY_SELECT or snapshot is None:
            self.log("[AutoStygianOnslaught] 未进入难度选择界面")
            return False
        if not self._select_hard_mode(snapshot, cancelled):
            return False
        state, _ = self._wait_for(
            StygianState.DOMAIN_LOBBY, timeout_s=45, cancelled=cancelled,
        )
        return state == StygianState.DOMAIN_LOBBY

    def _select_hard_mode(
        self, snapshot: StygianSnapshot, cancelled: Callable[[], bool] | None,
    ) -> bool:
        self.log("[AutoStygianOnslaught] 选择困难难度")
        hard = self._find_text(snapshot, "困难", roi=(850, 70, 1000, 450))
        if hard is None:
            mode = self._find_text(
                snapshot, "至危挑战", "常规挑战", roi=(850, 70, 1000, 450)
            )
            if mode is not None:
                self.ctx.input.click_ref(
                    mode.x + mode.width + 260, mode.y + mode.height / 2
                )
            else:
                self.ctx.input.click_ref(1300, 190)
            self.ctx.sleep(500)
            snapshot = self._capture_snapshot()
            hard = self._find_text(snapshot, "困难")
            if hard is not None:
                hard.click()
                self.ctx.sleep(500)
                snapshot = self._capture_snapshot()
        if not self._tap_text(
            snapshot, "单人挑战", roi=(900, 680, 1000, 400)
        ) and not self._tap_template(snapshot, "white_confirm"):
            self.log("[AutoStygianOnslaught] 未找到单人挑战确认按钮")
            return False
        return not self._cancelled(cancelled)

    def _walk_to_interaction(
        self, *, timeout_s: float, cancelled: Callable[[], bool] | None,
    ) -> StygianState:
        deadline = time.monotonic() + timeout_s
        moving = False
        try:
            while time.monotonic() < deadline:
                if self._cancelled(cancelled):
                    return StygianState.UNKNOWN
                snapshot = self._capture_snapshot()
                state = self._state(snapshot)
                if state in (StygianState.BOSS_SELECT, StygianState.RESIN_SELECT):
                    return state
                interact = self._find_text(
                    snapshot,
                    "激活", "接触", "地脉之花",
                    roi=(800, 250, 900, 650),
                )
                if state == StygianState.LEYLINE_FLOWER_PROMPT or interact is not None:
                    if moving:
                        self.ctx.input.key_up("W")
                        moving = False
                    self.ctx.input.key_press("F")
                    self.ctx.sleep(700)
                    continue
                if not moving:
                    self.ctx.input.key_down("W")
                    moving = True
                self.ctx.sleep(450)
        finally:
            if moving:
                self.ctx.input.key_up("W")
        return StygianState.UNKNOWN

    def _start_boss(
        self, snapshot: StygianSnapshot, cancelled: Callable[[], bool] | None,
    ) -> bool:
        positions = {1: (196, 346), 2: (237, 541), 3: (203, 728)}
        self.log(f"[AutoStygianOnslaught] 选择 Boss {self.boss_num}")
        self.ctx.input.click_ref(*positions[self.boss_num])
        self.ctx.sleep(350)
        snapshot = self._capture_snapshot()
        if not self._tap_text(
            snapshot, "开始挑战", roi=(900, 650, 1000, 430)
        ) and not self._tap_template(snapshot, "white_confirm"):
            return False
        state, _ = self._wait_for(
            StygianState.BATTLE_ARENA, timeout_s=45, cancelled=cancelled,
        )
        return state == StygianState.BATTLE_ARENA

    def _fight_round(self, cancelled: Callable[[], bool] | None) -> StygianState:
        self.ctx.input.key_down("W")
        self.ctx.sleep(1200)
        self.ctx.input.key_up("W")
        self.fight.run(cancelled=cancelled)
        state, _ = self._wait_for(
            (StygianState.BATTLE_RESULT_WIN, StygianState.BATTLE_RESULT_LOSE),
            timeout_s=75,
            cancelled=cancelled,
        )
        return state

    def _return_from_result(
        self, state: StygianState, cancelled: Callable[[], bool] | None,
    ) -> tuple[StygianState, StygianSnapshot | None]:
        snapshot = self._capture_snapshot()
        if state == StygianState.BATTLE_RESULT_LOSE:
            self.log("[AutoStygianOnslaught] 挑战失败，返回 Boss 选择")
            if not self._tap_template(snapshot, "white_confirm"):
                self._tap_text(snapshot, "重新挑战", "确认")
            return self._wait_for(
                StygianState.BOSS_SELECT, timeout_s=30, cancelled=cancelled,
            )
        self.log("[AutoStygianOnslaught] 挑战成功，返回门厅")
        if not self._tap_template(snapshot, "white_cancel"):
            self._tap_text(snapshot, "返回")
        return self._wait_for(
            StygianState.DOMAIN_LOBBY, timeout_s=35, cancelled=cancelled,
        )

    def _walk_to_flower(
        self, cancelled: Callable[[], bool] | None,
    ) -> StygianSnapshot | None:
        self.ctx.input.key_down("W")
        self.ctx.sleep(220)
        self.ctx.input.key_up("W")
        deadline = time.monotonic() + 45
        moving = False
        box = RecognitionObject.template_match(Mat.from_file(str(
            PROJECT_ROOT / "assets" / "templates" / "leyline" / "box.png"
        )))
        box.threshold = 0.72
        box.roi = (250, 80, 1420, 850)
        try:
            while time.monotonic() < deadline:
                if self._cancelled(cancelled):
                    return None
                snapshot = self._capture_snapshot()
                state = self._state(snapshot)
                if state == StygianState.RESIN_SELECT:
                    return snapshot
                interact = self._find_text(
                    snapshot, "接触", "地脉之花", roi=(700, 200, 1000, 700)
                )
                if state == StygianState.LEYLINE_FLOWER_PROMPT or interact is not None:
                    if moving:
                        self.ctx.input.key_up("W")
                        moving = False
                    self.ctx.input.key_press("F")
                    self.ctx.sleep(700)
                    continue
                icon = snapshot.region.find(box)
                if not icon.is_exist():
                    if moving:
                        self.ctx.input.key_up("W")
                        moving = False
                    self.ctx.input.move_camera_by(180, 0)
                    self.ctx.sleep(300)
                    continue
                offset = icon.x + icon.width / 2 - 960
                if abs(offset) > 100:
                    if moving:
                        self.ctx.input.key_up("W")
                        moving = False
                    self.ctx.input.move_camera_by(max(-300, min(300, offset)), 0)
                    self.ctx.sleep(300)
                    continue
                if not moving:
                    self.ctx.input.key_down("W")
                    moving = True
                self.ctx.sleep(450)
        finally:
            if moving:
                self.ctx.input.key_up("W")
        self.log("[AutoStygianOnslaught] 未找到地脉花")
        return None

    @staticmethod
    def _line_texts(snapshot: StygianSnapshot) -> list[str]:
        lines: list[list[Region]] = []
        for hit in snapshot.hits:
            line = next(
                (items for items in lines if abs(items[0].y - hit.y) <= max(18, hit.height)),
                None,
            )
            if line is None:
                lines.append([hit])
            else:
                line.append(hit)
        return [
            "".join(item.text for item in sorted(line, key=lambda item: item.x))
            for line in lines
        ]

    def _press_resin(self, snapshot: StygianSnapshot, resin_name: str) -> bool:
        resin = self._find_text(snapshot, resin_name, roi=(350, 150, 1200, 760))
        if resin is None:
            return False
        uses = [
            hit for hit in snapshot.hits
            if "使用" in _clean_text(hit.text) and hit.x > 960
        ]
        use = min(
            uses,
            key=lambda hit: abs(
                (hit.y + hit.height / 2) - (resin.y + resin.height / 2)
            ),
            default=None,
        )
        if use is None or abs(
            (use.y + use.height / 2) - (resin.y + resin.height / 2)
        ) > 55:
            return False
        use.click()
        self.ctx.sleep(60)
        use.click()
        self.ctx.sleep(900)
        return True

    def _claim_resin(self, snapshot: StygianSnapshot) -> ResinClaim:
        lines = self._line_texts(snapshot)
        plan = self.resin_plan if self.specify_resin_use else None
        name = choose_stygian_resin(lines, plan)
        if name is None:
            self.log("[AutoStygianOnslaught] 树脂耗尽或无法满足指定领奖次数")
            return ResinClaim(False, True, exhausted=True)
        before = resin_count_from_lines(lines, name)
        if not self._press_resin(snapshot, name):
            self.log(f"[AutoStygianOnslaught] 未找到{name}对应的使用按钮")
            return ResinClaim(False, True, name)

        if self.specify_resin_use:
            record = next(record for record in self.resin_plan if record.name == name)
            record.remaining -= 1
            used = record.maximum - record.remaining
            self.log(f"[AutoStygianOnslaught] {name} 刷取 {used}/{record.maximum}")
            last = sum(record.remaining for record in self.resin_plan) <= 0
        else:
            condensed = resin_count_from_lines(lines, "浓缩树脂")
            original = resin_count_from_lines(lines, "原粹树脂")
            if name == "浓缩树脂" and condensed is not None:
                condensed = max(0, condensed - 1)
            if name == "原粹树脂" and original is not None:
                original = max(0, original - 20)
            last = (
                condensed is not None and original is not None
                and condensed <= 0 and original < 20
            )
            suffix = f"（识别数量 {before}）" if before is not None else ""
            self.log(f"[AutoStygianOnslaught] 使用{name}{suffix}")
        return ResinClaim(True, last, name)

    def _handle_continue(
        self, claim: ResinClaim, cancelled: Callable[[], bool] | None,
    ) -> tuple[bool, StygianState, StygianSnapshot | None]:
        state, snapshot = self._wait_for(
            (StygianState.CONTINUE_OR_EXIT, StygianState.DOMAIN_LOBBY),
            timeout_s=15,
            cancelled=cancelled,
        )
        if state == StygianState.UNKNOWN:
            self.ctx.input.click_ref(960, 540)
            state, snapshot = self._wait_for(
                (StygianState.CONTINUE_OR_EXIT, StygianState.DOMAIN_LOBBY),
                timeout_s=8,
                cancelled=cancelled,
            )
        if not claim.success or claim.is_last:
            if state == StygianState.CONTINUE_OR_EXIT and snapshot is not None:
                if not self._tap_template(snapshot, "exit_button"):
                    self._tap_text(snapshot, "退出秘境", "退出")
            return False, state, snapshot
        if state == StygianState.CONTINUE_OR_EXIT and snapshot is not None:
            if not self._tap_template(snapshot, "continue_button", double=True):
                self._tap_text(snapshot, "继续挑战", "继续")
            state, snapshot = self._wait_for(
                (StygianState.BATTLE_ARENA, StygianState.DOMAIN_LOBBY),
                timeout_s=45,
                cancelled=cancelled,
            )
        return state != StygianState.UNKNOWN, state, snapshot

    def _exit_domain(self, cancelled: Callable[[], bool] | None) -> None:
        state, _ = self._wait_for(
            StygianState.MAIN_WORLD, timeout_s=8, cancelled=cancelled,
        )
        if state == StygianState.MAIN_WORLD:
            return
        self.ctx.input.key_press("ESCAPE")
        self.ctx.sleep(500)
        snapshot = self._capture_snapshot()
        if not self._tap_template(snapshot, "exit_door") and not self._tap_text(
            snapshot, "退出秘境", "退出挑战"
        ):
            self.ctx.input.key_press("ESCAPE")
            self.ctx.sleep(500)
            snapshot = self._capture_snapshot()
            if not self._tap_template(snapshot, "exit_door"):
                self._tap_text(snapshot, "退出秘境", "退出挑战")
        self.ctx.sleep(600)
        snapshot = self._capture_snapshot()
        if not self._tap_template(snapshot, "black_confirm"):
            self._tap_text(snapshot, "确认", "确定")
        self._wait_for(
            StygianState.MAIN_WORLD, timeout_s=30, cancelled=cancelled,
        )

    def _artifact_salvage(self, cancelled: Callable[[], bool] | None) -> None:
        if not self.auto_artifact_salvage or self._cancelled(cancelled):
            return
        from .artifact_salvage import AutoArtifactSalvageTask

        options = {
            "star": 4,
            "confirm_quick_salvage": False,
            "confirm_salvage": False,
            **self.artifact_salvage_options,
        }
        AutoArtifactSalvageTask(
            self.ctx, log=self.log, **options
        ).run(cancelled=cancelled)

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        """Run the event flow while owning the shared frame/input channel."""
        with exclusive_realtime_triggers(self.ctx):
            return self._run_impl(cancelled)

    def _run_impl(self, cancelled: Callable[[], bool] | None = None) -> bool:
        self._validate()
        completed = False
        self.log("[AutoStygianOnslaught] 开始")
        try:
            if not self.api.returnMainUi():
                return False
            if self.fight_team_name and not self.api.switchParty(self.fight_team_name):
                self.log(
                    f"[AutoStygianOnslaught] 未能切换战斗队 "
                    f"{self.fight_team_name}，保持当前队伍"
                )
            if not self._navigate_event(cancelled):
                return False

            state = self._walk_to_interaction(
                timeout_s=25, cancelled=cancelled
            )
            if state != StygianState.BOSS_SELECT:
                self.log("[AutoStygianOnslaught] 未到达 Boss 选择界面")
                return False
            snapshot = self._capture_snapshot()
            failures = 0
            round_no = 0
            while not self._cancelled(cancelled):
                if state == StygianState.BOSS_SELECT:
                    if not self._start_boss(snapshot, cancelled):
                        return False
                    state = StygianState.BATTLE_ARENA
                elif state == StygianState.DOMAIN_LOBBY:
                    state = self._walk_to_interaction(
                        timeout_s=25, cancelled=cancelled
                    )
                    if state != StygianState.BOSS_SELECT:
                        return False
                    snapshot = self._capture_snapshot()
                    continue

                round_no += 1
                self.log(f"[AutoStygianOnslaught] 第 {round_no} 轮战斗")
                result = self._fight_round(cancelled)
                if result == StygianState.UNKNOWN:
                    self.log("[AutoStygianOnslaught] 战斗结果检测超时")
                    return False
                state, snapshot = self._return_from_result(result, cancelled)
                if result == StygianState.BATTLE_RESULT_LOSE:
                    failures += 1
                    if failures >= self.max_battle_failures or snapshot is None:
                        self.log("[AutoStygianOnslaught] 连续挑战失败次数达到上限")
                        return False
                    continue
                failures = 0
                if state != StygianState.DOMAIN_LOBBY:
                    return False
                reward = self._walk_to_flower(cancelled)
                if reward is None:
                    return False
                claim = self._claim_resin(reward)
                should_continue, state, snapshot = self._handle_continue(
                    claim, cancelled
                )
                if not should_continue:
                    completed = claim.success or claim.exhausted
                    break
                if state == StygianState.BATTLE_ARENA:
                    continue
                if state != StygianState.DOMAIN_LOBBY or snapshot is None:
                    return False

            self._exit_domain(cancelled)
            if completed:
                self.ctx.sleep(3000)
                self._artifact_salvage(cancelled)
            self.log(
                f"[AutoStygianOnslaught] {'完成' if completed else '已取消'}"
            )
            return completed
        finally:
            self.ctx.input.release_all()
