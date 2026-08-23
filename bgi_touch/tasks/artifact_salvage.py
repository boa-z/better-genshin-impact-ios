"""BetterGI artifact-salvage task adapted to iOS touch input.

The destructive actions are deliberately gated.  Quick-select and five-star
rules can both prepare a review selection, but final salvage only happens when
the corresponding confirmation option is explicitly enabled.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from ..engine.context import GameContext
from ..engine.genshin_api import GenshinApi
from ..engine.recognition import Mat, RecognitionObject
from ..vision.ocr import OcrItem, get_ocr


ASSETS = Path(__file__).resolve().parents[2] / "assets" / "templates" / "artifact_salvage"

GRID_ROI = (48, 106, 1267, 768)
GRID_COLUMNS = 9
GRID_ROWS = 4


class ArtifactStatus(str, Enum):
    NONE = "none"
    LOCKED = "locked"
    SELECTED = "selected"


@dataclass(frozen=True)
class ArtifactAffix:
    Type: str
    Value: float
    IsUnactivated: bool = False


@dataclass(frozen=True)
class ArtifactStat:
    Name: str
    MainAffix: ArtifactAffix
    MinorAffixes: tuple[ArtifactAffix, ...]
    Level: int
    AllText: str = ""

    def js_value(self) -> dict:
        value = asdict(self)
        value.pop("AllText", None)
        value["MinorAffixes"] = [asdict(item) for item in self.MinorAffixes]
        return value


_AFFIX_NAMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CRITDMG", ("暴击伤害", "暴擊傷害", "CRIT DMG", "CRIT Damage")),
    ("CRITRate", ("暴击率", "暴擊率", "CRIT Rate")),
    ("EnergyRecharge", ("元素充能效率", "元素充能效能", "Energy Recharge")),
    ("ElementalMastery", ("元素精通", "Elemental Mastery")),
    ("HealingBonus", ("治疗加成", "治療加成", "Healing Bonus")),
    ("PhysicalDMGBonus", ("物理伤害加成", "物理傷害加成", "Physical DMG Bonus")),
    ("PyroDMGBonus", ("火元素伤害加成", "火元素傷害加成", "Pyro DMG Bonus")),
    ("HydroDMGBonus", ("水元素伤害加成", "水元素傷害加成", "Hydro DMG Bonus")),
    ("DendroDMGBonus", ("草元素伤害加成", "草元素傷害加成", "Dendro DMG Bonus")),
    ("ElectroDMGBonus", ("雷元素伤害加成", "雷元素傷害加成", "Electro DMG Bonus")),
    ("AnemoDMGBonus", ("风元素伤害加成", "風元素傷害加成", "Anemo DMG Bonus")),
    ("CryoDMGBonus", ("冰元素伤害加成", "冰元素傷害加成", "Cryo DMG Bonus")),
    ("GeoDMGBonus", ("岩元素伤害加成", "岩元素傷害加成", "Geo DMG Bonus")),
    ("ATK", ("攻击力", "攻擊力", "ATK")),
    ("DEF", ("防御力", "防禦力", "DEF")),
    ("HP", ("生命值", "HP")),
)


def _clean_ocr_line(value: str) -> str:
    return (
        str(value).strip().replace("，", ",").replace("。", ".")
        .replace("％", "%").replace("＋", "+").replace("：", ":")
    )


def _affix_type(text: str, *, percent: bool = False) -> str | None:
    normalized = text.casefold()
    for kind, aliases in _AFFIX_NAMES:
        if any(alias.casefold() in normalized for alias in aliases):
            if percent and kind in ("ATK", "DEF", "HP"):
                return f"{kind}Percent"
            return kind
    return None


def _number(text: str) -> float:
    value = re.sub(r"[^0-9.,-]", "", text).replace(",", "")
    if not value:
        raise ValueError(f"未识别的词条数值：{text}")
    return float(value)


def parse_artifact_stat_text(value: str | Iterable[str]) -> ArtifactStat:
    """Parse BetterGI's detail-card OCR result into its ArtifactStat contract."""
    source = value.splitlines() if isinstance(value, str) else value
    lines = [_clean_ocr_line(line) for line in source if _clean_ocr_line(line)]
    if not lines:
        raise ValueError("圣遗物详情 OCR 为空")

    level = 0
    for line in lines:
        match = re.fullmatch(r"\+\s*(\d{1,2})", line)
        if match:
            level = int(match.group(1))
            if level > 20:
                raise ValueError(f"圣遗物等级无效：{level}")
            break

    minor: list[ArtifactAffix] = []
    minor_indexes: set[int] = set()
    minor_pattern = re.compile(r"^(.+?)[+:]\s*([0-9][0-9.,]*)\s*(%)?.*$")
    for index, line in enumerate(lines):
        match = minor_pattern.match(line)
        if not match:
            continue
        kind = _affix_type(match.group(1), percent=bool(match.group(3)))
        if kind is None:
            continue
        minor.append(ArtifactAffix(kind, _number(match.group(2))))
        minor_indexes.add(index)

    main: ArtifactAffix | None = None
    main_index = -1
    for index, line in enumerate(lines):
        if index in minor_indexes or re.fullmatch(r"\+\s*\d{1,2}", line):
            continue
        kind = _affix_type(line, percent="%" in line)
        if kind is None:
            continue
        same_line = re.search(r"([0-9][0-9.,]*)\s*(%)?", line)
        value_text = same_line.group(1) if same_line else None
        percent = bool(same_line and same_line.group(2)) or "%" in line
        if value_text is None and index + 1 < len(lines):
            following = re.fullmatch(r"([0-9][0-9.,]*)\s*(%)?", lines[index + 1])
            if following:
                value_text = following.group(1)
                percent = bool(following.group(2))
        if value_text is not None:
            if percent and kind in ("ATK", "DEF", "HP"):
                kind = f"{kind}Percent"
            main = ArtifactAffix(kind, _number(value_text))
            main_index = index
            break
    if main is None:
        raise ValueError("未识别到圣遗物主词条")

    name = ""
    for index, line in enumerate(lines):
        if index >= main_index:
            break
        if not re.search(r"\d", line) and _affix_type(line) is None:
            name = line
            break
    if not name:
        name = lines[0]
    return ArtifactStat(name, main, tuple(minor), level, "\n".join(lines))


_NODE_RULE_HARNESS = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
const sandbox = Object.create(null);
sandbox.ArtifactStat = payload.artifact;
vm.createContext(sandbox, {codeGeneration: {strings: false, wasm: false}});
const compiled = new vm.Script(payload.script, {filename: 'artifact-rule.js'});
compiled.runInContext(sandbox, {timeout: payload.timeoutMs});
if (!Object.prototype.hasOwnProperty.call(sandbox, 'Output')) {
  throw new Error('JavaScript没有设置Output输出');
}
if (typeof sandbox.Output !== 'boolean') {
  throw new Error('JavaScript的Output输出不是布尔类型');
}
process.stdout.write(JSON.stringify({output: sandbox.Output}));
"""


def evaluate_artifact_javascript(
    artifact: ArtifactStat,
    javascript: str,
    *,
    timeout_ms: int = 300,
) -> bool:
    """Run the upstream Output-based rule in an isolated, time-limited Node VM."""
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("五星圣遗物 JavaScript 筛选需要安装 Node.js")
    payload = json.dumps(
        {
            "artifact": artifact.js_value(),
            "script": str(javascript),
            "timeoutMs": max(10, min(2000, int(timeout_ms))),
        },
        ensure_ascii=False,
    )
    try:
        result = subprocess.run(
            [node, "-e", _NODE_RULE_HARNESS],
            input=payload,
            text=True,
            capture_output=True,
            timeout=max(1.0, timeout_ms / 1000 + 0.8),
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError("圣遗物 JavaScript 筛选超时") from error
    if result.returncode != 0:
        lines = (result.stderr or result.stdout).strip().splitlines()
        message = next(
            (line.strip() for line in lines if "Error" in line or "timed out" in line),
            lines[-1].strip() if lines else "圣遗物 JavaScript 执行失败",
        )
        raise RuntimeError(message)
    output = json.loads(result.stdout or "{}").get("output")
    if not isinstance(output, bool):
        raise RuntimeError("圣遗物 JavaScript 未返回布尔 Output")
    return output


def detect_artifact_status(cell_bgr: np.ndarray) -> ArtifactStatus:
    """Port BetterGI's pink-lock/green-selection HSV detector."""
    if cell_bgr.size == 0:
        return ArtifactStatus.NONE
    upper = cell_bgr[:max(1, round(cell_bgr.shape[0] * 0.19))]
    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV_FULL)

    def mask(hue_deg: float, saturation: float, sat_delta: int, val_delta: int) -> np.ndarray:
        hue = hue_deg / 360 * 255
        center_s = saturation * 255
        low = np.array([max(0, hue - 3), max(0, center_s - sat_delta), 255 - val_delta])
        high = np.array([min(255, hue + 3), min(255, center_s + sat_delta), 255])
        return cv2.inRange(hsv, low, high)

    pink = mask(9, 0.54, 25, 25)
    points = cv2.findNonZero(pink)
    if points is not None:
        points = np.asarray(points, dtype=np.int32).reshape(-1, 2)
        left_points = points[points[:, 0] < pink.shape[1] * 0.2]
        if left_points.size:
            _, _, width, height = cv2.boundingRect(left_points)
            if width > pink.shape[1] * 0.07 and height > pink.shape[0] * 0.3:
                return ArtifactStatus.LOCKED

    green = mask(80, 0.76, 10, 5)
    points = cv2.findNonZero(green)
    if points is not None:
        _, _, width, height = cv2.boundingRect(points)
        if width > green.shape[1] * 0.2 and height > green.shape[0] * 0.8:
            return ArtifactStatus.SELECTED
    return ArtifactStatus.NONE


def parse_artifact_set_filter(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = re.split(r"[,，;；|\n]+", value)
    else:
        parts = [str(item) for item in value]
    return tuple(dict.fromkeys(part.strip() for part in parts if part.strip()))


class AutoArtifactSalvageTask:
    def __init__(
        self,
        ctx: GameContext,
        *,
        star: int = 4,
        javascript: str | None = None,
        artifact_set_filter: str | Iterable[str] | None = None,
        max_num_to_check: int = 100,
        recognition_failure_policy: str = "Skip",
        confirm_quick_salvage: bool = False,
        confirm_salvage: bool = False,
        max_pages: int = 12,
        timeout_s: float = 300,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.star = int(star)
        if self.star not in (1, 2, 3, 4):
            raise ValueError("AutoArtifactSalvage.star 必须在 1 到 4 之间")
        self.javascript = str(javascript).strip() if javascript else None
        self.artifact_sets = parse_artifact_set_filter(artifact_set_filter)
        self.max_num_to_check = max(1, min(2000, int(max_num_to_check)))
        policy = str(recognition_failure_policy or "Skip").split(".")[-1].casefold()
        if policy not in ("skip", "abort", "跳过", "中止"):
            raise ValueError("recognitionFailurePolicy 仅支持 Skip 或 Abort")
        self.skip_recognition_failure = policy in ("skip", "跳过")
        self.confirm_quick_salvage = bool(confirm_quick_salvage)
        self.confirm_salvage = bool(confirm_salvage)
        self.max_pages = max(1, min(100, int(max_pages)))
        self.timeout_s = max(30.0, float(timeout_s))
        self.log = log
        self._set_filter_applied = False
        self._templates = {
            name: Mat.from_file(str(ASSETS / f"{name}.png"))
            for name in (
                "bag_artifact_checked", "bag_artifact_unchecked",
                "btn_artifact_salvage", "btn_artifact_salvage_confirm",
                "btn_white_confirm", "btn_black_confirm",
            )
        }

    @staticmethod
    def _clean(text: str) -> str:
        return str(text).replace(" ", "").replace("\u3000", "")

    def _find_template(self, name: str, roi=None, threshold: float = 0.8):
        ro = RecognitionObject.template_match(
            self._templates[name], *(roi if roi is not None else (None,) * 4)
        )
        ro.threshold = threshold
        return self.ctx.capture_region().find(ro)

    def _find_text(self, words: tuple[str, ...], roi=(0, 0, 1920, 1080)):
        for hit in self.ctx.capture_region().find_multi(RecognitionObject.ocr(*roi), limit=60):
            text = self._clean(hit.text)
            if any(self._clean(word).casefold() in text.casefold() for word in words):
                return hit
        return None

    def _open_salvage(self, deadline: float) -> bool:
        if not GenshinApi(self.ctx, log=self.log).returnMainUi():
            return False
        self.ctx.input.key_press("B")
        self.ctx.sleep(900)
        for _ in range(6):
            if time.monotonic() >= deadline:
                return False
            checked = self._find_template("bag_artifact_checked", (0, 0, 1920, 120), 0.76)
            if checked.is_exist():
                break
            unchecked = self._find_template("bag_artifact_unchecked", (0, 0, 1920, 120), 0.82)
            if unchecked.is_exist():
                unchecked.click()
                self.ctx.sleep(600)
                break
            confirm = self._find_text(("确认", "确定", "Confirm"), (500, 350, 920, 650))
            if confirm is not None:
                confirm.click()
            else:
                self.ctx.input.key_press("B")
            self.ctx.sleep(500)
        else:
            return False
        salvage = self._find_template("btn_artifact_salvage", (0, 850, 1920, 230), 0.76)
        if not salvage.is_exist():
            self.log("[AutoArtifactSalvage] 未找到背包中的圣遗物分解按钮")
            return False
        salvage.click()
        self.ctx.sleep(800)
        return True

    def _quick_select(self) -> bool:
        quick = self._find_text(("快速选择", "快速選擇", "Quick Select"), (0, 850, 700, 230))
        if quick is None:
            self.log("[AutoArtifactSalvage] 未找到快速选择按钮")
            return False
        quick.click()
        self.ctx.sleep(450)
        for index in range(self.star, 4):
            number = index + 1
            option = self._find_text(
                (f"{number}星圣遗物", f"{number}星聖遺物", f"{number}-Star"),
                (0, 0, 600, 1080),
            )
            if option is None:
                self.log(f"[AutoArtifactSalvage] 未找到 {number} 星反选项")
                return False
            option.click()
            self.ctx.sleep(220)
        confirm = self._find_template("btn_white_confirm", threshold=0.76)
        if not confirm.is_exist():
            self.log("[AutoArtifactSalvage] 未找到快速选择确认按钮")
            return False
        confirm.click()
        self.ctx.sleep(800)
        return True

    def _confirm_current_selection(self) -> bool:
        salvage = self._find_template(
            "btn_artifact_salvage_confirm", (1250, 850, 670, 230), 0.76
        )
        if not salvage.is_exist():
            self.log("[AutoArtifactSalvage] 当前没有可分解的选中项")
            return False
        salvage.click()
        self.ctx.sleep(650)
        confirm = self._find_template("btn_black_confirm", threshold=0.76)
        if not confirm.is_exist():
            self.log("[AutoArtifactSalvage] 未找到最终黑色确认按钮，已停止")
            return False
        confirm.click()
        self.ctx.sleep(700)
        return True

    def _grid_bounds(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        scale = self.ctx.transform.scale
        x, y, width, height = GRID_ROI
        return round(x * scale), round(y * scale), round(width * scale), round(height * scale)

    def _cell_crop(self, frame: np.ndarray, column: int, row: int) -> np.ndarray:
        x, y, width, height = self._grid_bounds(frame)
        cell_w, cell_h = width / GRID_COLUMNS, height / GRID_ROWS
        margin_x, margin_y = cell_w * 0.03, cell_h * 0.03
        x0 = round(x + column * cell_w + margin_x)
        y0 = round(y + row * cell_h + margin_y)
        x1 = round(x + (column + 1) * cell_w - margin_x)
        y1 = round(y + (row + 1) * cell_h - margin_y)
        return frame[y0:y1, x0:x1]

    def _tap_cell(self, column: int, row: int) -> None:
        x, y, width, height = GRID_ROI
        self.ctx.input.click_ref(
            x + (column + 0.5) * width / GRID_COLUMNS,
            y + (row + 0.5) * height / GRID_ROWS,
        )

    @staticmethod
    def _detail_crop(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        x0, y0 = round(width * 0.70), round(height * 0.112)
        return frame[y0:round(y0 + height * 0.50), x0:round(x0 + width * 0.275)]

    @staticmethod
    def _page_fingerprint(frame: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = bounds
        gray = cv2.cvtColor(frame[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (48, 28), interpolation=cv2.INTER_AREA)

    def _scroll_grid(self) -> None:
        t = self.ctx.transform
        x1, y1 = t.to_device(650, 800, "left")
        x2, y2 = t.to_device(650, 190, "left")
        self.ctx.device.swipe(
            x1, y1, x2, y2,
            duration_ms=550,
            image_width=t.device_width,
            image_height=t.device_height,
        )
        self.ctx.sleep(650)

    def _read_artifact(self, frame: np.ndarray) -> ArtifactStat:
        items: list[OcrItem] = get_ocr().recognize(self._detail_crop(frame))
        confident = [item for item in items if item.confidence >= 0.45]
        confident.sort(key=lambda item: (item.y, item.x))
        return parse_artifact_stat_text(item.text for item in confident)

    def _matches_set(self, artifact: ArtifactStat) -> bool:
        return self._set_filter_applied or not self.artifact_sets or any(
            name.casefold() in artifact.AllText.casefold() for name in self.artifact_sets
        )

    def _apply_set_filter(self, deadline: float) -> bool:
        """Use the game's own set-name filter before scanning five-star cards."""
        if not self.artifact_sets:
            return True
        button = self._find_text(("筛选", "篩選", "Filter"), (950, 0, 970, 1080))
        if button is None:
            template = self._find_template("btn_white_confirm", threshold=0.74)
            if template.is_exist():
                button = template
        if button is None or not button.is_exist():
            self.log("[AutoArtifactSalvage] 未找到圣遗物筛选按钮")
            return False
        button.click()
        self.ctx.sleep(450)

        category = self._find_text(
            ("所属套装", "所屬套裝", "Artifact Set"), (0, 0, 700, 400)
        )
        if category is not None:
            category.click()
        else:
            # Same reference point used by BetterGI for the set category.
            self.ctx.input.click_ref(315, 190)
        self.ctx.sleep(650)

        remaining = set(self.artifact_sets)
        for _ in range(16):
            if not remaining or time.monotonic() >= deadline:
                break
            hits = self.ctx.capture_region().find_multi(
                RecognitionObject.ocr(0, 70, 1420, 900), limit=100
            )
            for hit in hits:
                hit_text = self._clean(hit.text).casefold()
                match = next(
                    (
                        name for name in remaining
                        if self._clean(name).casefold() in hit_text
                        or (len(hit_text) >= 3 and hit_text in self._clean(name).casefold())
                    ),
                    None,
                )
                if match is None:
                    continue
                hit.click()
                remaining.remove(match)
                self.ctx.sleep(100)
            if remaining:
                transform = self.ctx.transform
                x1, y1 = transform.to_device(700, 820, "left")
                x2, y2 = transform.to_device(700, 240, "left")
                self.ctx.device.swipe(
                    x1, y1, x2, y2,
                    duration_ms=450,
                    image_width=transform.device_width,
                    image_height=transform.device_height,
                )
                self.ctx.sleep(450)
        if remaining:
            self.log(
                "[AutoArtifactSalvage] 套装筛选未找到：" + ", ".join(sorted(remaining))
            )
            return False

        confirm = self._find_template("btn_white_confirm", threshold=0.74)
        if not confirm.is_exist():
            self.log("[AutoArtifactSalvage] 未找到套装筛选确认按钮")
            return False
        confirm.click()
        self.ctx.sleep(650)
        confirm = self._find_template("btn_white_confirm", threshold=0.74)
        if confirm.is_exist():
            confirm.click()
            self.ctx.sleep(500)
        self._set_filter_applied = True
        return True

    def _scan_five_star(
        self,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[int, int]:
        checked = selected = 0
        for page in range(self.max_pages):
            frame = self.ctx.capture_bgr()
            for row in range(GRID_ROWS):
                for column in range(GRID_COLUMNS):
                    if checked >= self.max_num_to_check or time.monotonic() >= deadline:
                        return checked, selected
                    if cancelled and cancelled():
                        return checked, selected
                    if detect_artifact_status(self._cell_crop(frame, column, row)) != ArtifactStatus.NONE:
                        continue
                    self._tap_cell(column, row)
                    self.ctx.sleep(260)
                    current = self.ctx.capture_bgr()
                    if detect_artifact_status(self._cell_crop(current, column, row)) != ArtifactStatus.SELECTED:
                        frame = current
                        continue
                    checked += 1
                    keep = False
                    try:
                        artifact = self._read_artifact(current)
                        keep = self._matches_set(artifact) and evaluate_artifact_javascript(
                            artifact, self.javascript or "Output = false;"
                        )
                        self.log(
                            f"[AutoArtifactSalvage] 检查 {checked}/{self.max_num_to_check}: "
                            f"{artifact.Name} -> {'保留选中' if keep else '跳过'}"
                        )
                    except Exception as error:
                        if not self.skip_recognition_failure:
                            raise
                        self.log(f"[AutoArtifactSalvage] 识别失败，跳过当前圣遗物：{error}")
                    if keep:
                        selected += 1
                    else:
                        self._tap_cell(column, row)
                        self.ctx.sleep(120)
                    frame = self.ctx.capture_bgr()
            if checked >= self.max_num_to_check or page + 1 >= self.max_pages:
                break
            before = self._page_fingerprint(frame, self._grid_bounds(frame))
            self._scroll_grid()
            after_frame = self.ctx.capture_bgr()
            after = self._page_fingerprint(after_frame, self._grid_bounds(after_frame))
            if float(np.mean(cv2.absdiff(before, after))) < 1.5:
                break
        return checked, selected

    def run(self, cancelled: Callable[[], bool] | None = None) -> dict[str, int | bool]:
        deadline = time.monotonic() + self.timeout_s
        if not self._open_salvage(deadline):
            return {"ok": False, "quickSelected": False, "checked": 0, "selected": 0}

        result: dict[str, int | bool] = {
            "ok": True, "quickSelected": False, "quickSalvaged": False,
            "checked": 0, "selected": 0, "salvaged": False,
        }
        if not self._quick_select():
            result["ok"] = False
            return result
        result["quickSelected"] = True
        if not self.confirm_quick_salvage:
            self.log(
                "[AutoArtifactSalvage] 已完成低星快速选择并停在复查界面；"
                "需要显式设置 confirmQuickSalvage=true 才会分解"
            )
            return result
        result["quickSalvaged"] = self._confirm_current_selection()
        if not self.javascript:
            if result["quickSalvaged"]:
                self.ctx.input.click_ref(960, 540)
                self.ctx.sleep(350)
                GenshinApi(self.ctx, log=self.log).returnMainUi()
            return result

        if result["quickSalvaged"]:
            self.ctx.input.click_ref(960, 540)
            self.ctx.sleep(700)
        if self.artifact_sets and not self._apply_set_filter(deadline):
            raise RuntimeError("无法安全应用 artifactSetFilter，已停止五星筛选")
        checked, selected = self._scan_five_star(deadline, cancelled)
        result["checked"], result["selected"] = checked, selected
        if selected and self.confirm_salvage:
            result["salvaged"] = self._confirm_current_selection()
        else:
            self.log(
                f"[AutoArtifactSalvage] 五星筛选完成：检查 {checked}，选中 {selected}；"
                "当前停留在复查界面"
            )
        return result
