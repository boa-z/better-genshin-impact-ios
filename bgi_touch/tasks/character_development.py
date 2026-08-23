"""Character development reader migrated from BetterGI's state machine."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from ..engine.context import GameContext
from ..engine.genshin_api import GenshinApi
from ..engine.recognition import Mat, RecognitionObject
from ..vision.ocr import get_ocr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS = PROJECT_ROOT / "assets" / "templates" / "character_development"
DATA = PROJECT_ROOT / "assets" / "data"

CHARACTER_GRID_ROI = (40, 76, 641, 897)
WEAPON_TYPES = {
    "1": "单手剑", "10": "法器", "11": "双手剑", "12": "弓", "13": "长柄武器"
}


@dataclass
class CharacterDevelopmentResult:
    CharacterName: str = ""
    ElementType: str | None = None
    Level: int | None = None
    LevelLimit: int | None = None
    WeaponName: str | None = None
    WeaponLevel: int | None = None
    WeaponLevelLimit: int | None = None
    AttackLevel: int | None = None
    AttackHasBonus: bool | None = None
    SkillLevel: int | None = None
    SkillHasBonus: bool | None = None
    BurstLevel: int | None = None
    BurstHasBonus: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CharacterMetadata:
    name: str
    aliases: tuple[str, ...]
    element: str | None
    weapon_type: str | None


@lru_cache(maxsize=1)
def load_character_metadata() -> dict[str, CharacterMetadata]:
    avatars = json.loads((DATA / "combat_avatar.json").read_text(encoding="utf-8"))
    elements = json.loads((DATA / "avatar_elements.json").read_text(encoding="utf-8-sig"))
    element_by_name = {
        str(name): str(element)
        for element, names in elements.items()
        for name in names
    }
    output: dict[str, CharacterMetadata] = {}
    for raw in avatars:
        name = str(raw.get("name", "")).strip()
        if not name:
            continue
        aliases = tuple(dict.fromkeys([name, *(str(item) for item in raw.get("alias", []))]))
        metadata = CharacterMetadata(
            name,
            aliases,
            element_by_name.get(name),
            WEAPON_TYPES.get(str(raw.get("weapon", ""))),
        )
        for alias in aliases:
            output[alias.casefold()] = metadata
    return output


def normalize_character_name(value: str) -> str:
    name = str(value or "").strip()
    if not name:
        raise ValueError("角色名不能为空")
    metadata = load_character_metadata().get(name.casefold())
    return metadata.name if metadata else name


def normalize_character_names(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ValueError("多角色接口需要字符串集合或 JS Array，单个字符串请使用 getCharacter")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("角色名参数必须是字符串集合或 JS Array") from error
    if not items:
        raise ValueError("角色名集合不能为空")
    if any(not isinstance(item, str) for item in items):
        raise ValueError("角色名集合中的每个元素都必须是字符串")
    return tuple(normalize_character_name(item) for item in items)


def parse_character_categories(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return "属性", "武器", "天赋"
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("读取分类不能为空字符串")
        raw = re.split(r"[;；,，]+", value)
    else:
        raw = [str(item) for item in value]
    aliases = {
        "属性": "属性", "屬性": "属性", "attribute": "属性", "attributes": "属性",
        "武器": "武器", "weapon": "武器", "weapons": "武器",
        "天赋": "天赋", "天賦": "天赋", "talent": "天赋", "talents": "天赋",
    }
    output = []
    for item in raw:
        key = item.strip().casefold()
        if key not in aliases:
            raise ValueError(f"未知的角色信息分类：{item}")
        if aliases[key] not in output:
            output.append(aliases[key])
    if not output:
        raise ValueError("至少需要指定一个读取分类")
    return tuple(output)


def try_parse_level_pair(text: str) -> tuple[int, int] | None:
    values = [int(value) for value in re.findall(r"\d+", str(text))]
    if len(values) < 2 or values[0] <= 0 or values[1] <= 0:
        return None
    return values[0], values[1]


def normalize_talent_type(text: str) -> str:
    normalized = str(text).replace(" ", "")
    aliases = {
        "普通攻击": ("普通攻击", "普通攻擊", "Normal Attack"),
        "元素战技": ("元素战技", "元素戰技", "Elemental Skill"),
        "元素爆发": ("元素爆发", "元素爆發", "Elemental Burst"),
    }
    for result, values in aliases.items():
        if any(value.casefold() in normalized.casefold() for value in values):
            return result
    return ""


def try_parse_talent_level(text: str) -> int | None:
    match = re.search(r"\d+", str(text))
    value = int(match.group()) if match else 0
    return value if value > 0 else None


def has_talent_bonus(text: str) -> bool:
    return bool(re.search(r"天赋\s*等级\s*[+＋]\s*3", str(text)))


def apply_talent_result(
    result: CharacterDevelopmentResult, talent_type: str, level: int, has_bonus: bool
) -> None:
    if talent_type == "普通攻击":
        result.AttackLevel, result.AttackHasBonus = level, has_bonus
    elif talent_type == "元素战技":
        result.SkillLevel, result.SkillHasBonus = level, has_bonus
    elif talent_type == "元素爆发":
        result.BurstLevel, result.BurstHasBonus = level, has_bonus
    else:
        raise ValueError(f"未知天赋类型：{talent_type}")


@dataclass(frozen=True)
class CharacterCard:
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2


def build_character_cards(
    bottom_rects: Iterable[tuple[int, int, int, int]],
    grid_size: tuple[int, int],
    scale: float,
) -> list[CharacterCard]:
    rects = list(bottom_rects)
    if not rects:
        return []
    card_width = max(1, round(115 * scale))
    card_height = max(1, round(140 * scale))
    x_tolerance = max(3, card_width // 3)
    y_tolerance = max(3, card_height // 3)

    def corrected(values: list[int], tolerance: int) -> list[int]:
        ordered = sorted((value, index) for index, value in enumerate(values))
        output = [0] * len(values)
        group: list[tuple[int, int]] = []

        def flush() -> None:
            if not group:
                return
            values_ = sorted(value for value, _ in group)
            median = values_[len(values_) // 2]
            for _, index in group:
                output[index] = median
            group.clear()

        for entry in ordered:
            if group and entry[0] - group[-1][0] > tolerance:
                flush()
            group.append(entry)
        flush()
        return output

    rights = corrected([x + width for x, _, width, _ in rects], x_tolerance)
    bottoms = corrected([y + height for _, y, _, height in rects], y_tolerance)
    seen = set()
    cards = []
    grid_width, grid_height = grid_size
    for right, bottom in zip(rights, bottoms):
        if (right, bottom) in seen:
            continue
        seen.add((right, bottom))
        x, y = right - card_width, bottom - card_height
        if x >= 0 and y >= 0 and x + card_width <= grid_width and y + card_height <= grid_height:
            cards.append(CharacterCard(x, y, card_width, card_height))
    return sorted(cards, key=lambda card: (card.y, card.x))


def detect_character_cards(grid_bgr: np.ndarray, scale: float) -> list[CharacterCard]:
    if grid_bgr.size == 0:
        return []
    hsv = cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 12, 233), (35, 16, 237))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area, max_area = 2000 * scale * scale, 3000 * scale * scale
    rects = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if min_area <= area <= max_area:
            rects.append((x, y, width, height))
    return build_character_cards(rects, (grid_bgr.shape[1], grid_bgr.shape[0]), scale)


class CharacterDevelopmentTask:
    def __init__(
        self,
        ctx: GameContext,
        *,
        max_pages: int = 30,
        timeout_s: float = 900,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.max_pages = max(1, min(100, int(max_pages)))
        self.timeout_s = max(60.0, float(timeout_s))
        self.log = log
        self._menu = Mat.from_file(str(ASSETS / "menu.png"))
        self._filter = Mat.from_file(str(ASSETS / "filter.png"))

    def _template(self, template: Mat, threshold=0.76):
        ro = RecognitionObject.template_match(template, 0, 850, 500, 230)
        ro.threshold = threshold
        return self.ctx.capture_region().find(ro)

    def _crop_ref(self, frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = roi
        transform = self.ctx.transform
        anchor = transform.resolve_anchor(x)
        x0, y0 = transform.to_device(x, y, anchor)
        x1, y1 = transform.to_device(x + width, y + height, anchor)
        left, right = sorted((round(x0), round(x1)))
        top, bottom = sorted((round(y0), round(y1)))
        return frame[max(0, top):bottom, max(0, left):right]

    def _ocr(self, roi: tuple[int, int, int, int], frame: np.ndarray | None = None) -> str:
        frame = self.ctx.capture_bgr() if frame is None else frame
        items = get_ocr().recognize(self._crop_ref(frame, roi))
        items.sort(key=lambda item: (item.y, item.x))
        return " ".join(item.text for item in items).strip()

    def _find_text(self, words: tuple[str, ...], roi=(0, 0, 1920, 1080)):
        hits = self.ctx.capture_region().find_multi(RecognitionObject.ocr(*roi), limit=80)
        for hit in hits:
            text = hit.text.replace(" ", "").casefold()
            if any(word.replace(" ", "").casefold() in text for word in words):
                return hit
        return None

    def _click_text(self, words: tuple[str, ...], roi=(0, 0, 1920, 1080)) -> bool:
        hit = self._find_text(words, roi)
        if hit is None:
            return False
        hit.click()
        self.ctx.sleep(420)
        return True

    def _open_overview(self) -> bool:
        if self._template(self._menu).is_exist():
            return True
        if not GenshinApi(self.ctx, log=self.log).returnMainUi():
            return False
        self.ctx.input.key_press("C")
        self.ctx.sleep(1000)
        return self._template(self._menu).is_exist()

    def _open_character_list(self) -> bool:
        if self._template(self._filter).is_exist():
            return True
        if not self._open_overview():
            return False
        menu = self._template(self._menu)
        if not menu.is_exist():
            return False
        menu.click()
        self.ctx.sleep(600)
        return self._template(self._filter).is_exist()

    def _grid_bounds(self) -> tuple[int, int, int, int]:
        scale = self.ctx.transform.scale
        return tuple(round(value * scale) for value in CHARACTER_GRID_ROI)

    def _cards(self, frame: np.ndarray) -> list[CharacterCard]:
        x, y, width, height = self._grid_bounds()
        local = detect_character_cards(frame[y:y + height, x:x + width], self.ctx.transform.scale)
        return [CharacterCard(card.x + x, card.y + y, card.width, card.height) for card in local]

    def _tap_card(self, card: CharacterCard) -> None:
        x, y = card.center
        transform = self.ctx.transform
        self.ctx.device.tap(
            x, y, image_width=transform.device_width, image_height=transform.device_height
        )

    def _scroll_cards(self) -> None:
        transform = self.ctx.transform
        x1, y1 = transform.to_device(360, 880, "left")
        x2, y2 = transform.to_device(360, 180, "left")
        self.ctx.device.swipe(
            x1, y1, x2, y2, duration_ms=550,
            image_width=transform.device_width, image_height=transform.device_height,
        )
        self.ctx.sleep(650)

    def _grid_fingerprint(self, frame: np.ndarray) -> np.ndarray:
        x, y, width, height = self._grid_bounds()
        gray = cv2.cvtColor(frame[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (32, 44), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _candidate_names(name: str) -> tuple[str, ...]:
        if name == "旅行者":
            return "空", "荧", "旅行者"
        metadata = load_character_metadata().get(name.casefold())
        return metadata.aliases if metadata else (name,)

    def _find_and_select_character(
        self,
        name: str,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> CharacterMetadata | None:
        candidates = self._candidate_names(name)
        metadata = load_character_metadata().get(name.casefold())
        for page in range(1, self.max_pages + 1):
            if time.monotonic() >= deadline or (cancelled and cancelled()):
                return None
            frame = self.ctx.capture_bgr()
            cards = self._cards(frame)
            if not cards:
                self.log(f"[CharacterDevelopment] 第 {page} 页未检测到角色卡")
                return None
            for card in cards:
                card_text_items = get_ocr().recognize(frame[
                    card.y + round(card.height * 0.72):card.y + card.height,
                    card.x:card.x + card.width,
                ])
                card_text = "".join(item.text for item in card_text_items)
                self._tap_card(card)
                self.ctx.sleep(260)
                display = self._ocr((1466, 131, 244, 38))
                if any(candidate in display or candidate in card_text for candidate in candidates):
                    self.log(f"[CharacterDevelopment] 已选择角色 {name}（{display or card_text}）")
                    self.ctx.input.key_press("ESCAPE")
                    self.ctx.sleep(450)
                    return metadata or CharacterMetadata(name, candidates, None, None)
            if page >= self.max_pages:
                break
            before = self._grid_fingerprint(self.ctx.capture_bgr())
            self._scroll_cards()
            after = self._grid_fingerprint(self.ctx.capture_bgr())
            if float(np.mean(cv2.absdiff(before, after))) < 1.5:
                break
        return None

    def _stable_value(
        self,
        reader: Callable[[], Any | None],
        field_name: str,
        deadline: float,
    ) -> Any:
        last = object()
        consecutive = 0
        for _ in range(10):
            if time.monotonic() >= deadline:
                break
            value = reader()
            if value is not None and value == last:
                consecutive += 1
            elif value is not None:
                last, consecutive = value, 1
            else:
                consecutive = 0
            if consecutive >= 3:
                return value
            self.ctx.sleep(200)
        raise RuntimeError(f"{field_name} OCR 未能连续三次稳定")

    def _read_attribute(self, result: CharacterDevelopmentResult, deadline: float) -> None:
        level, limit = self._stable_value(
            lambda: try_parse_level_pair(self._ocr((1467, 207, 172, 35))),
            "角色等级", deadline,
        )
        result.Level, result.LevelLimit = level, limit

    def _read_weapon(self, result: CharacterDevelopmentResult, deadline: float) -> None:
        name = self._stable_value(
            lambda: self._ocr((1465, 132, 346, 42)).strip() or None,
            "武器名称", deadline,
        )
        level, limit = self._stable_value(
            lambda: try_parse_level_pair(self._ocr((1464, 319, 147, 73))),
            "武器等级", deadline,
        )
        result.WeaponName = name
        result.WeaponLevel, result.WeaponLevelLimit = level, limit

    def _talent_points(self) -> list[tuple[float, float]]:
        hits = self.ctx.capture_region().find_multi(
            RecognitionObject.ocr(1536, 0, 384, 1080), limit=80
        )
        points = []
        min_distance = 30 * self.ctx.transform.scale
        for hit in sorted(hits, key=lambda item: item.dy):
            if "lv" not in hit.text.casefold():
                continue
            point = hit.dx + hit.dw / 2, hit.dy + hit.dh / 2
            if all(abs(existing[1] - point[1]) > min_distance for existing in points):
                points.append(point)
        return points if len(points) == 3 else []

    def _read_talent_value(self) -> tuple[str, int, bool] | None:
        talent_type = normalize_talent_type(self._ocr((242, 13, 120, 40)))
        level = try_parse_talent_level(self._ocr((250, 160, 85, 40)))
        if not talent_type or level is None:
            return None
        return talent_type, level, has_talent_bonus(self._ocr((35, 275, 180, 50)))

    def _read_talents(self, result: CharacterDevelopmentResult, deadline: float) -> None:
        points = []
        for _ in range(6):
            points = self._talent_points()
            if points:
                break
            self.ctx.sleep(250)
        if len(points) != 3:
            raise RuntimeError("未能定位普通攻击、元素战技和元素爆发三个天赋")
        seen = set()
        transform = self.ctx.transform
        for x, y in points:
            self.ctx.device.tap(
                x, y, image_width=transform.device_width, image_height=transform.device_height
            )
            self.ctx.sleep(350)
            talent_type, level, bonus = self._stable_value(
                self._read_talent_value, "天赋详情", deadline
            )
            if talent_type in seen:
                raise RuntimeError(f"重复识别天赋类型：{talent_type}")
            seen.add(talent_type)
            apply_talent_result(result, talent_type, level, bonus)
        if seen != {"普通攻击", "元素战技", "元素爆发"}:
            raise RuntimeError("未能完整识别三个战斗天赋")
        self.ctx.input.key_press("ESCAPE")
        self.ctx.sleep(350)

    def run(
        self,
        character_names: Iterable[str],
        categories: str | Iterable[str] | None = None,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        names = tuple(normalize_character_name(name) for name in character_names)
        if not names:
            raise ValueError("至少需要指定一个角色")
        requested = parse_character_categories(categories)
        deadline = time.monotonic() + self.timeout_s
        results = []
        try:
            for index, name in enumerate(names):
                if cancelled and cancelled():
                    break
                if index and not self._click_text(("属性", "屬性", "Attributes"), (0, 0, 450, 1080)):
                    raise RuntimeError("切换下一角色前无法返回属性页")
                if not self._open_character_list():
                    raise RuntimeError("无法打开角色列表")
                metadata = self._find_and_select_character(name, deadline, cancelled)
                if metadata is None:
                    raise RuntimeError(f"未找到目标角色 {name}")
                result = CharacterDevelopmentResult(name, ElementType=metadata.element)
                for category in requested:
                    words = {
                        "属性": ("属性", "屬性", "Attributes"),
                        "武器": ("武器", "Weapon"),
                        "天赋": ("天赋", "天賦", "Talents"),
                    }[category]
                    if not self._click_text(words, (0, 0, 450, 1080)):
                        raise RuntimeError(f"无法打开角色{category}页")
                    if category == "属性":
                        self._read_attribute(result, deadline)
                    elif category == "武器":
                        self._read_weapon(result, deadline)
                    else:
                        self._read_talents(result, deadline)
                results.append(result.to_dict())
            return results
        finally:
            GenshinApi(self.ctx, log=self.log).returnMainUi()

    def get_character(self, character_name: str, categories: str | None = None) -> dict[str, Any]:
        return self.run([normalize_character_name(character_name)], categories)[0]

    def get_multi_characters(
        self, character_names: object, categories: str | None = None
    ) -> list[dict[str, Any]]:
        return self.run(normalize_character_names(character_names), categories)

    getCharacter = get_character
    GetCharacter = get_character
    getMultiCharacters = get_multi_characters
    GetMultiCharacters = get_multi_characters
