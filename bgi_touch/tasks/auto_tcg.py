"""BetterGI AutoGeniusInvokation parser, recognition and duel executor."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

from ..engine.context import GameContext
from ..engine.recognition import ImageRegion, Mat, RecognitionObject, Region
from .tcg_state import (
    ELEMENT_FROM_CHINESE,
    TcgCharacter,
    TcgCommand,
    TcgElement,
    TcgPhase,
    TcgSkill,
    effective_skill_cost,
    next_living_character,
    reroll_indices,
    tuning_card_count,
    wanted_elements,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_DIR = PROJECT_ROOT / "assets" / "tcg" / "1920x1080"
DEFAULT_CARD_CONFIG = PROJECT_ROOT / "assets" / "tcg" / "tcg_character_card.json"

_TEMPLATE_FILES = {
    "confirm": "other/确定.png",
    "round_end": "other/回合结束.png",
    "tuning_confirm": "other/元素调和.png",
    "duel_end": "other/退出挑战.png",
    "opponent": "other/对方行动中.png",
    "end_phase": "other/回合结算阶段.png",
    "dice_lack": "other/元素骰子不足.png",
    "taken_out": "other/角色死亡.png",
    "character_pick": "other/出战角色.png",
    "hp_upper": "other/角色血量上方.png",
    "defeated": "other/角色被打败.png",
    "frozen": "other/角色状态_冻结.png",
    "bubble": "other/角色状态_水泡.png",
}

_TEMPLATE_ROIS = {
    "round_end": (0, 0, 384, 1080),
    "opponent": (0, 0, 384, 1080),
    "end_phase": (0, 0, 384, 1080),
    "duel_end": (0, 540, 960, 540),
    "dice_lack": (960, 0, 960, 1080),
    "character_pick": (960, 540, 960, 540),
    "tuning_confirm": (0, 540, 1920, 540),
}


def _parse_element(value: str, *, line_no: int) -> TcgElement:
    key = value.strip().replace("元素", "").replace("骰子", "")
    try:
        return ELEMENT_FROM_CHINESE[key]
    except KeyError as exc:
        where = f"第 {line_no} 行" if line_no else "默认卡牌配置"
        raise ValueError(f"七圣召唤{where}元素无效：{value}") from exc


def _parse_skill(value: str, *, line_no: int) -> TcgSkill:
    match = re.fullmatch(r"技能\s*(\d+)\s*消耗\s*=\s*(.+)", value.strip())
    if not match:
        raise ValueError(f"七圣召唤策略第 {line_no} 行技能定义无效：{value}")
    index = int(match.group(1))
    if not 1 <= index <= 5:
        raise ValueError(f"七圣召唤技能序号必须为 1-5：第 {line_no} 行")
    costs = [part.strip() for part in match.group(2).split("+")]
    first = re.fullmatch(r"(\d+)\s*([^\d]+)", costs[0])
    if not first:
        raise ValueError(f"七圣召唤策略第 {line_no} 行技能消耗无效：{value}")
    specific = int(first.group(1))
    element = _parse_element(first.group(2), line_no=line_no)
    any_cost = 0
    for part in costs[1:]:
        any_match = re.fullmatch(r"(\d+)\s*(?:任意|无色(?:元素)?|任意骰子)", part)
        if not any_match:
            raise ValueError(f"七圣召唤策略第 {line_no} 行杂色骰子消耗无效：{part}")
        any_cost += int(any_match.group(1))
    return TcgSkill(index, element, specific, any_cost)


def _load_default_character(name: str, index: int, path: Path) -> TcgCharacter:
    if not path.is_file():
        raise ValueError(
            f"角色【{name}】没有内联技能定义，且缺少 {path}；"
            "请运行 tools/fetch_map_assets.py --tcg"
        )
    cards = json.loads(path.read_text(encoding="utf-8-sig"))
    card = next((item for item in cards if item.get("name") == name), None)
    if card is None:
        alias_path = path.with_name("combat_avatar.json")
        if alias_path.is_file():
            avatars = json.loads(alias_path.read_text(encoding="utf-8-sig"))
            standard_name = next(
                (
                    str(item.get("name", ""))
                    for item in avatars
                    if name in item.get("alias", [])
                ),
                "",
            )
            card = next(
                (item for item in cards if item.get("name") == standard_name), None
            )
    if card is None:
        raise ValueError(f"角色【{name}】暂无默认卡牌定义，请在策略中完整定义")
    element = _parse_element(str(card["element"]), line_no=0)
    skills: dict[int, TcgSkill] = {}
    usable = [
        item for item in card.get("skills", [])
        if "被动技能" not in item.get("skillTag", [])
    ]
    # BetterGI numbers action buttons from right to left.
    for skill_index, item in enumerate(reversed(usable), start=1):
        specific = any_cost = 0
        skill_element = element
        for cost in item.get("cost", []):
            cost_type = str(cost.get("type", ""))
            count = int(cost.get("count", 0))
            if cost_type in ("无色元素", "任意元素"):
                any_cost += count
            elif cost_type != "充能":
                skill_element = _parse_element(cost_type, line_no=0)
                specific += count
        skills[skill_index] = TcgSkill(
            skill_index, skill_element, specific, any_cost, str(item.get("name", ""))
        )
    return TcgCharacter(index, name, element, skills)


def parse_tcg_strategy(
    text: str, *, card_config_path: str | Path = DEFAULT_CARD_CONFIG
) -> tuple[dict[int, TcgCharacter], list[TcgCommand]]:
    """Parse the current BetterGI Chinese TCG strategy format."""
    stage = ""
    characters: dict[int, TcgCharacter] = {}
    commands: list[TcgCommand] = []
    for line_no, raw in enumerate(str(text).splitlines(), start=1):
        line = raw.strip()
        if not line or line == "---" or line.startswith("//"):
            continue
        if line in ("角色定义:", "策略定义:"):
            stage = line
            continue
        if stage == "角色定义:":
            match = re.fullmatch(
                r"角色\s*(\d+)\s*=\s*([^|{]+?)(?:\|([^|{]+)\{(.+)\})?", line
            )
            if not match:
                raise ValueError(f"七圣召唤策略第 {line_no} 行角色定义无效")
            index = int(match.group(1))
            if index not in (1, 2, 3):
                raise ValueError(f"七圣召唤角色序号必须为 1-3：第 {line_no} 行")
            name = match.group(2).strip()
            if match.group(3) is None:
                character = _load_default_character(name, index, Path(card_config_path))
            else:
                element = _parse_element(match.group(3), line_no=line_no)
                skills = {
                    skill.index: skill
                    for skill in (
                        _parse_skill(part, line_no=line_no)
                        for part in match.group(4).split(",")
                    )
                }
                character = TcgCharacter(index, name, element, skills)
            characters[index] = character
            continue
        if stage == "策略定义:":
            parts = line.split()
            if not 3 <= len(parts) <= 4 or parts[1] != "使用":
                raise ValueError(f"七圣召唤策略第 {line_no} 行行动格式无效")
            skill_match = re.fullmatch(r"技能(\d+)", parts[2])
            if not skill_match or not 1 <= int(skill_match.group(1)) <= 5:
                raise ValueError(f"七圣召唤技能编号无效：第 {line_no} 行")
            delta = 0
            if len(parts) == 4:
                delta_match = re.fullmatch(r"骰子(增加|减少)(\d+)", parts[3])
                if not delta_match:
                    raise ValueError(f"七圣召唤骰子参数无效：第 {line_no} 行")
                delta = int(delta_match.group(2)) * (
                    1 if delta_match.group(1) == "增加" else -1
                )
            commands.append(TcgCommand(parts[0], int(skill_match.group(1)), delta))
            continue
        raise ValueError(f"七圣召唤策略第 {line_no} 行位于未知段落：{stage}")
    if set(characters) != {1, 2, 3}:
        raise ValueError("七圣召唤策略必须定义角色 1、2、3")
    by_name = {character.name: character for character in characters.values()}
    for command in commands:
        character = by_name.get(command.character)
        if character is None:
            raise ValueError(f"七圣召唤策略引用了未定义角色：{command.character}")
        if command.skill not in character.skills:
            raise ValueError(f"角色【{character.name}】没有定义技能{command.skill}")
    if not commands:
        raise ValueError("七圣召唤策略没有行动命令")
    return characters, commands


class TcgRecognizer:
    """Single-frame template/OCR recognizer for every duel phase."""

    def __init__(self, ctx: GameContext, asset_dir: str | Path = DEFAULT_ASSET_DIR):
        self.ctx = ctx
        self.asset_dir = Path(asset_dir)
        if (self.asset_dir / "1920x1080").is_dir():
            self.asset_dir /= "1920x1080"
        self.templates: dict[str, Mat] = {}
        for key, relative in _TEMPLATE_FILES.items():
            path = self.asset_dir / relative
            if path.is_file():
                self.templates[key] = Mat.from_file(str(path))
        self.roll_templates = self._load_dice_templates("roll")
        self.action_templates = self._load_dice_templates("action")

    def require_assets(self) -> None:
        required = set(_TEMPLATE_FILES)
        missing = sorted(required - self.templates.keys())
        if missing or len(self.roll_templates) != 8 or len(self.action_templates) != 8:
            raise FileNotFoundError(
                f"七圣召唤识别资产不完整：{self.asset_dir}；"
                "请运行 tools/fetch_map_assets.py --tcg"
            )

    def _load_dice_templates(self, phase: str) -> dict[TcgElement, Mat]:
        result: dict[TcgElement, Mat] = {}
        for element in TcgElement:
            path = self.asset_dir / "dice" / f"{phase}_{element.value}.png"
            if path.is_file():
                result[element] = Mat.from_file(str(path))
        return result

    def _find_all(
        self, frame: np.ndarray, key: str, *, threshold: float = 0.8, limit: int = 20
    ) -> list[Region]:
        template = self.templates.get(key)
        if template is None:
            return []
        roi = _TEMPLATE_ROIS.get(key)
        ro = RecognitionObject.template_match(
            template, *(roi if roi is not None else (None, None, None, None))
        )
        ro.threshold = threshold
        return ImageRegion(self.ctx, frame).find_multi(ro, limit=limit)

    def find(
        self, frame: np.ndarray, key: str, *, threshold: float = 0.8
    ) -> Region | None:
        matches = self._find_all(frame, key, threshold=threshold, limit=1)
        return matches[0] if matches else None

    def phase(self, frame: np.ndarray) -> TcgPhase:
        checks = (
            ("duel_end", TcgPhase.DUEL_END),
            ("character_pick", TcgPhase.CHARACTER_PICK),
            ("taken_out", TcgPhase.CHARACTER_TAKEN_OUT),
            ("round_end", TcgPhase.MY_ACTION),
            ("opponent", TcgPhase.OPPONENT_ACTION),
            ("end_phase", TcgPhase.END_PHASE),
        )
        for key, phase in checks:
            if self.find(frame, key):
                return phase
        confirm = self.find(frame, "confirm")
        if confirm:
            return TcgPhase.ROLL if len(self.roll_dice(frame)) == 8 else TcgPhase.PREPARE
        return TcgPhase.UNKNOWN

    @staticmethod
    def _overlap(a: Region, b: Region) -> float:
        ax2, ay2 = a.dx + a.dw, a.dy + a.dh
        bx2, by2 = b.dx + b.dw, b.dy + b.dh
        area = max(0.0, min(ax2, bx2) - max(a.dx, b.dx)) * max(
            0.0, min(ay2, by2) - max(a.dy, b.dy)
        )
        return area / max(1.0, a.dw * a.dh + b.dw * b.dh - area)

    def _dice_matches(
        self,
        frame: np.ndarray,
        templates: Mapping[TcgElement, Mat],
        roi=None,
        threshold=0.7,
    ) -> list[tuple[TcgElement, Region]]:
        candidates: list[tuple[TcgElement, Region]] = []
        image = ImageRegion(self.ctx, frame)
        for element, template in templates.items():
            ro = RecognitionObject.template_match(
                template, *(roi if roi is not None else (None, None, None, None))
            )
            ro.threshold = threshold
            candidates.extend((element, hit) for hit in image.find_multi(ro, limit=20))
        selected: list[tuple[TcgElement, Region]] = []
        for candidate in sorted(candidates, key=lambda item: item[1].score, reverse=True):
            if all(self._overlap(candidate[1], existing[1]) < 0.35 for existing in selected):
                selected.append(candidate)
        return sorted(selected, key=lambda item: (item[1].y, item[1].x))

    def roll_dice(self, frame: np.ndarray) -> list[tuple[TcgElement, Region]]:
        return self._dice_matches(frame, self.roll_templates, threshold=0.73)

    def action_dice(self, frame: np.ndarray) -> Counter[TcgElement]:
        matches = self._dice_matches(
            frame, self.action_templates, roi=(1536, 0, 384, 1080)
        )
        return Counter(element for element, _ in matches)

    def dice_count_ocr(self, frame: np.ndarray) -> int | None:
        text = ImageRegion(self.ctx, frame).derive_crop(68, 642, 40, 36).ocr_text()
        replacements = {
            "①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5",
            "⑥": "6", "⑦": "7", "⑧": "8", "⑨": "9", "⑩": "10",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        match = re.search(r"\d+", text.replace(" ", ""))
        return int(match.group()) if match else None

    def defeated_characters(
        self,
        frame: np.ndarray,
        card_rects: Mapping[int, tuple[float, float, float, float]],
    ) -> set[int]:
        defeated: set[int] = set()
        for hit in self._find_all(frame, "defeated", threshold=0.8):
            cx, cy = hit.x + hit.width / 2, hit.y + hit.height / 2
            for index, (x, y, w, h) in card_rects.items():
                if x <= cx <= x + w and y <= cy <= y + h:
                    defeated.add(index)
        return defeated

    def active_character(
        self,
        frame: np.ndarray,
        card_rects: Mapping[int, tuple[float, float, float, float]],
    ) -> int | None:
        hits = self._find_all(frame, "hp_upper", threshold=0.75)
        if not hits:
            return None
        hit = min(hits, key=lambda item: item.y)
        cx = hit.x + hit.width / 2
        return min(
            card_rects,
            key=lambda index: abs(
                cx - (card_rects[index][0] + card_rects[index][2] / 2)
            ),
        )

    def character_statuses(
        self, frame: np.ndarray, card_rect: tuple[float, float, float, float]
    ) -> set[str]:
        statuses: set[str] = set()
        x, y, w, h = card_rect
        for key in ("frozen", "bubble"):
            for hit in self._find_all(frame, key, threshold=0.78):
                cx, cy = hit.x + hit.width / 2, hit.y + hit.height / 2
                if x - 20 <= cx <= x + w + 40 and y - 20 <= cy <= y + h + 20:
                    statuses.add(key)
        return statuses


class AutoGeniusInvokationTask:
    def __init__(
        self,
        ctx: GameContext,
        strategy: str,
        *,
        character_points: Mapping[int, tuple[float, float]] | None = None,
        skill_points: Mapping[int, tuple[float, float]] | None = None,
        max_commands: int | None = None,
        max_rounds: int = 20,
        timeout_s: float = 900.0,
        asset_dir: str | Path | None = DEFAULT_ASSET_DIR,
        card_config_path: str | Path | None = DEFAULT_CARD_CONFIG,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.characters, parsed_commands = parse_tcg_strategy(
            strategy,
            card_config_path=(
                card_config_path if card_config_path is not None else DEFAULT_CARD_CONFIG
            ),
        )
        self.commands = list(
            parsed_commands[:max_commands] if max_commands else parsed_commands
        )
        self.card_rects = {
            1: (667.0, 632.0, 165.0, 282.0),
            2: (877.0, 632.0, 165.0, 282.0),
            3: (1088.0, 632.0, 165.0, 282.0),
        }
        self.character_points = character_points or {
            index: (x + w / 2, y + h / 2)
            for index, (x, y, w, h) in self.card_rects.items()
        }
        self.skill_points = skill_points or {
            index: (1920 - 100 * index, 960) for index in range(1, 6)
        }
        self.max_rounds = max(1, int(max_rounds))
        self.timeout_s = max(1.0, float(timeout_s))
        self.log = log
        self.recognizer = TcgRecognizer(
            ctx, asset_dir if asset_dir is not None else DEFAULT_ASSET_DIR
        )
        self.current_character: int | None = None
        self.current_dice = 0
        self.current_cards = 0

    def _cancelled(
        self, deadline: float, callback: Callable[[], bool] | None
    ) -> bool:
        return time.monotonic() >= deadline or bool(callback and callback())

    def _capture(self) -> np.ndarray:
        return self.ctx.capture_bgr()

    def _click_character(self, index: int, *, double: bool = False) -> None:
        point = self.character_points[index]
        self.ctx.input.click_ref(*point)
        if double:
            self.ctx.sleep(250)
            self.ctx.input.click_ref(*point)

    def _swipe_ref(
        self, x1: float, y1: float, x2: float, y2: float, duration_ms=350
    ) -> None:
        dx1, dy1 = self.ctx.transform.to_device(x1, y1)
        dx2, dy2 = self.ctx.transform.to_device(x2, y2)
        self.ctx.device.swipe(
            dx1, dy1, dx2, dy2, duration_ms=duration_ms,
            image_width=self.ctx.transform.device_width,
            image_height=self.ctx.transform.device_height,
        )

    def _wait_for_phase(
        self,
        phases: Iterable[TcgPhase],
        deadline: float,
        cancelled: Callable[[], bool] | None,
        timeout_s: float = 60.0,
    ) -> tuple[TcgPhase, np.ndarray]:
        expected = set(phases)
        local_deadline = min(deadline, time.monotonic() + timeout_s)
        stable: TcgPhase | None = None
        stable_count = 0
        last_frame: np.ndarray | None = None
        while time.monotonic() < local_deadline and not self._cancelled(deadline, cancelled):
            last_frame = self._capture()
            phase = self.recognizer.phase(last_frame)
            if phase == TcgPhase.DUEL_END:
                return phase, last_frame
            if phase == TcgPhase.CHARACTER_TAKEN_OUT:
                self._recover_defeated(last_frame)
                stable = None
                stable_count = 0
            elif phase in expected:
                stable_count = stable_count + 1 if stable == phase else 1
                stable = phase
                if stable_count >= 2:
                    return phase, last_frame
            else:
                stable = None
                stable_count = 0
            self.ctx.sleep(650)
        return (
            TcgPhase.UNKNOWN,
            last_frame if last_frame is not None else self._capture(),
        )

    def _prepare(
        self, deadline: float, cancelled: Callable[[], bool] | None
    ) -> bool:
        self.log("[AutoGeniusInvokation] 选择初始手牌")
        local_deadline = min(deadline, time.monotonic() + 45)
        while time.monotonic() < local_deadline and not self._cancelled(deadline, cancelled):
            frame = self._capture()
            phase = self.recognizer.phase(frame)
            if phase in (
                TcgPhase.CHARACTER_PICK, TcgPhase.ROLL, TcgPhase.MY_ACTION
            ):
                break
            if phase == TcgPhase.DUEL_END:
                return False
            confirm = self.recognizer.find(frame, "confirm")
            if confirm:
                confirm.click()
            self.ctx.sleep(900)
        first = next_living_character(self.commands, self.characters)
        if first is None:
            return False
        frame = self._capture()
        if self.recognizer.phase(frame) == TcgPhase.CHARACTER_PICK:
            self.log(f"[AutoGeniusInvokation] 初始出战：{first.name}")
            self._click_character(first.index, double=True)
            self.current_character = first.index
            self.ctx.sleep(1200)
        elif self.current_character is None:
            # Resuming after card selection has no historical active-character
            # state. BetterGI's initial active character is the first command's.
            self.current_character = first.index
        return True

    def _reroll(self, frame: np.ndarray) -> bool:
        matches = self.recognizer.roll_dice(frame)
        if len(matches) != 8:
            return False
        wanted = wanted_elements(
            self.commands,
            self.characters,
            dice_count=8,
            current_character=self.current_character,
        )
        dice = [element for element, _ in matches]
        indexes = reroll_indices(dice, wanted)
        self.log(
            "[AutoGeniusInvokation] 投骰保留："
            + "/".join(sorted(element.value for element in wanted))
            + f"，重投 {len(indexes)} 枚"
        )
        for index in indexes:
            matches[index][1].click()
            self.ctx.sleep(100)
        confirm = self.recognizer.find(self._capture(), "confirm")
        if not confirm:
            return False
        confirm.click()
        self.ctx.sleep(1200)
        return True

    def _refresh_character_state(self, frame: np.ndarray) -> None:
        defeated = self.recognizer.defeated_characters(frame, self.card_rects)
        for index, character in self.characters.items():
            character.defeated = index in defeated
            character.active = False
            character.statuses.clear()
        active = self.recognizer.active_character(frame, self.card_rects)
        if active is not None and not self.characters[active].defeated:
            self.current_character = active
        if self.current_character is not None:
            character = self.characters[self.current_character]
            character.active = True
            character.statuses = self.recognizer.character_statuses(
                frame, self.card_rects[self.current_character]
            )

    def _recover_defeated(self, frame: np.ndarray) -> bool:
        taken_out_index = self.current_character
        self._refresh_character_state(frame)
        if taken_out_index is not None:
            self.characters[taken_out_index].defeated = True
        character = next_living_character(self.commands, self.characters)
        if character is None:
            return False
        self.log(f"[AutoGeniusInvokation] 角色阵亡，重新出战：{character.name}")
        self._click_character(character.index, double=True)
        self.current_character = character.index
        self.ctx.sleep(1800)
        return True

    def _switch_character(self, character: TcgCharacter) -> None:
        self.ctx.input.click_ref(960, 540)
        self.ctx.sleep(500)
        self._click_character(character.index)
        self.ctx.sleep(300)
        self.ctx.input.click_ref(1820, 960)
        self.ctx.sleep(700)
        self.ctx.input.click_ref(1820, 960)
        self.current_character = character.index
        self.current_dice -= 1

    def _tune_one_card(self) -> bool:
        count = max(1, min(self.current_cards, 10))
        starts = {
            10: (570, 120), 9: (570, 130), 8: (600, 145),
            7: (630, 160), 6: (620, 200), 5: (720, 200),
            4: (820, 200), 3: (920, 200), 2: (1020, 200), 1: (1120, 200),
        }
        start, spacing = starts[count]
        self.ctx.input.click_ref(960, 1030)
        self.ctx.sleep(700)
        for card_index in range(count):
            if card_index:
                self.ctx.input.click_ref(960, 1030)
                self.ctx.sleep(350)
            self._swipe_ref(start + spacing * card_index, 1030, 1870, 540)
            self.ctx.sleep(650)
            confirm = self.recognizer.find(
                self._capture(), "tuning_confirm", threshold=0.9
            )
            if confirm:
                confirm.click()
                self.current_cards -= 1
                self.ctx.sleep(1000)
                return True
            self.ctx.input.click_ref(960, 540)
            self.ctx.sleep(350)
        return False

    def _use_skill(
        self, command: TcgCommand, skill: TcgSkill, frame: np.ndarray
    ) -> bool:
        dice = self.recognizer.action_dice(frame)
        for _ in range(5):
            if sum(dice.values()) == self.current_dice:
                break
            self.ctx.sleep(350)
            dice = self.recognizer.action_dice(self._capture())
        if sum(dice.values()) != self.current_dice:
            self.log(
                f"[AutoGeniusInvokation] 行动骰子仅识别 {sum(dice.values())}/"
                f"{self.current_dice}，本回合停止行动"
            )
            return False
        needed = tuning_card_count(dice, skill)
        if needed > self.current_cards:
            return False
        for _ in range(needed):
            if not self._tune_one_card():
                return False
        point = self.skill_points[command.skill]
        self.ctx.input.click_ref(960, 540)
        self.ctx.sleep(350)
        self.ctx.input.click_ref(*point)
        self.ctx.sleep(1100)
        warning = self.recognizer.find(self._capture(), "dice_lack")
        if warning:
            self.ctx.input.click_ref(960, 540)
            return False
        self.ctx.input.click_ref(*point)
        self.ctx.sleep(350)
        self.ctx.input.click_ref(960, 540)
        self.current_dice -= effective_skill_cost(skill, command.dice_delta)
        return True

    def _execute_one_action(self, frame: np.ndarray) -> bool:
        self._refresh_character_state(frame)
        by_name = {character.name: character for character in self.characters.values()}
        for command in list(self.commands):
            character = by_name[command.character]
            if character.defeated or character.statuses:
                continue
            if self.current_character is not None:
                current = self.characters[self.current_character]
                if current.statuses and current.index == character.index:
                    continue
            if self.current_character != character.index:
                if self.current_dice < 1:
                    return False
                self.log(f"[AutoGeniusInvokation] 切换角色：{character.name}")
                self._switch_character(character)
                return True
            skill = character.skills[command.skill]
            cost = effective_skill_cost(skill, command.dice_delta)
            if cost > self.current_dice:
                return False
            if not self._use_skill(command, skill, frame):
                self.log(
                    f"[AutoGeniusInvokation] 手牌或元素不足，跳过："
                    f"{character.name} 技能{command.skill}"
                )
                return False
            self.commands.remove(command)
            self.log(
                f"[AutoGeniusInvokation] {character.name} 使用技能{command.skill}，"
                f"剩余骰子 {self.current_dice}"
            )
            return True
        return False

    def _end_round(self, frame: np.ndarray) -> None:
        button = self.recognizer.find(frame, "round_end")
        if button:
            button.click()
            self.ctx.sleep(700)
            button.click()
        self.ctx.input.click_ref(960, 540)

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        self.recognizer.require_assets()
        deadline = time.monotonic() + self.timeout_s
        self.log(f"[AutoGeniusInvokation] 执行 {len(self.commands)} 条策略")
        if not self._prepare(deadline, cancelled):
            return False
        for round_number in range(1, self.max_rounds + 1):
            if self._cancelled(deadline, cancelled):
                return False
            if not self.commands:
                self.log("[AutoGeniusInvokation] 策略指令已全部完成")
                return True
            self.current_cards = 5 if round_number == 1 else self.current_cards + 2
            self.current_dice = 8
            for character in self.characters.values():
                character.statuses.clear()
            self.log(f"[AutoGeniusInvokation] -------- 第 {round_number} 回合 --------")
            phase, frame = self._wait_for_phase(
                {TcgPhase.ROLL, TcgPhase.MY_ACTION},
                deadline,
                cancelled,
                timeout_s=45,
            )
            if phase == TcgPhase.DUEL_END:
                return True
            if phase == TcgPhase.ROLL:
                if not self._reroll(frame):
                    raise RuntimeError("七圣召唤投骰识别不完整，未执行重投")
                phase, frame = self._wait_for_phase(
                    {TcgPhase.MY_ACTION}, deadline, cancelled, timeout_s=65
                )
            if phase == TcgPhase.DUEL_END:
                return True
            if phase != TcgPhase.MY_ACTION:
                raise TimeoutError("等待七圣召唤我方行动阶段超时")
            while self.commands and not self._cancelled(deadline, cancelled):
                ocr_count = self.recognizer.dice_count_ocr(frame)
                if ocr_count is not None and abs(ocr_count - self.current_dice) <= 4:
                    self.current_dice = ocr_count
                if self.current_dice <= 0 or not self._execute_one_action(frame):
                    break
                phase, frame = self._wait_for_phase(
                    {TcgPhase.MY_ACTION}, deadline, cancelled, timeout_s=65
                )
                if phase == TcgPhase.DUEL_END:
                    return True
                if phase != TcgPhase.MY_ACTION:
                    raise TimeoutError("等待七圣召唤对方行动结束超时")
            if not self.commands:
                return True
            self._end_round(frame)
            phase, _ = self._wait_for_phase(
                {TcgPhase.ROLL, TcgPhase.MY_ACTION},
                deadline,
                cancelled,
                timeout_s=90,
            )
            if phase == TcgPhase.DUEL_END:
                return True
            if phase == TcgPhase.UNKNOWN:
                raise TimeoutError("等待七圣召唤回合结算超时")
        self.log(f"[AutoGeniusInvokation] 达到最大回合数 {self.max_rounds}")
        return not self.commands
