"""BetterGI boss and ley-line encounter loops for the iOS touch runtime.

``AutoBossTask`` mirrors the current upstream policy contract: it can run a
fixed number of successful claims or continue until resin is exhausted,
supports the official route variants, locates the Trounce Blossom, and
optionally aggregates reward cards. ``AutoLeyLineTask`` keeps the smaller
reusable route/combat/claim loop until its own navigator is ported.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..engine.context import GameContext
from ..engine.genshin_api import GenshinApi
from ..engine.recognition import Mat, RecognitionObject
from ..macro.keymouse import MacroPlayer, load_keymouse
from ..pathing.executor import PathingExecutor
from ..pathing.model import PathingTask
from .auto_fight import AutoFightTask


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOSS_ROUTE_ROOT = PROJECT_ROOT / "assets" / "pathing" / "boss"
BOSS_ASSET_ROOT = PROJECT_ROOT / "assets" / "autoboss"

BOSS_COUNTRIES: dict[str, tuple[str, ...]] = {
    "蒙德": ("急冻树", "无相之雷", "守望者·堕天"),
    "璃月": ("爆炎树", "纯水精灵", "古岩龙蜥", "无相之岩", "遗迹巨蛇", "隐山猊兽"),
    "稻妻": ("无相之火", "恒常机关阵列", "雷音权现", "魔偶剑鬼", "无相之水"),
    "须弥": ("掣电树", "半永恒统辖矩阵", "翠翎恐蕈", "风蚀沙虫", "无相之草", "深罪浸礼者", "兆载永劫龙兽"),
    "枫丹": ("歌裴莉娅的葬送", "科培琉司的劫罚", "实验性场力发生装置", "魔像督军", "千年珍珠骏麟", "水形幻人", "铁甲熔火帝皇"),
    "纳塔": ("金焰绒翼龙暴君", "灵觉隐修的迷者", "秘源机兵·构型械", "秘源机兵·统御械", "熔岩辉龙像", "深邃摹结株", "贪食匿叶龙山王"),
    "挪德卡莱": ("蕴光月守宫", "深黯魇语之主", "超重型陆巡舰·机动战垒", "霜夜巡天灵主", "蕴光月幻蝶", "重拳出击鸭"),
}
SUPPORTED_BOSSES = frozenset(name for names in BOSS_COUNTRIES.values() for name in names)
TALK_TO_START_BOSSES = frozenset(("歌裴莉娅的葬送", "科培琉司的劫罚", "纯水精灵", "重拳出击鸭"))
NO_PATHING_SUPPORT_BOSSES = frozenset(("蕴光月守宫", "超重型陆巡舰·机动战垒", "蕴光月幻蝶"))

_FULL_WIDTH_NUMBERS = str.maketrans("０１２３４５６７８９：", "0123456789:")


@dataclass(frozen=True)
class OriginalResinInfo:
    count: int
    limit: int


@dataclass(frozen=True)
class BossRunPolicy:
    """Pure run policy shared by dispatcher, tests, and the live task."""

    specify_run_count: bool = False
    run_count: int = 1

    def __post_init__(self) -> None:
        if self.specify_run_count and self.run_count < 1:
            raise ValueError("指定讨伐次数必须大于 0")

    def should_continue(self, successful_claims: int) -> bool:
        return not self.specify_run_count or successful_claims < self.run_count


def normalize_resin_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").translate(_FULL_WIDTH_NUMBERS))


def parse_resin_limit(text: str) -> int:
    """Parse BetterGI's current/limit OCR text using its last-three-digits rule."""

    digits = re.sub(r"\D", "", normalize_resin_text(text))
    if len(digits) < 3:
        raise ValueError(f"原粹树脂上限 OCR 失败：{text}")
    limit = int(digits[-3:])
    if limit <= 0:
        raise ValueError(f"原粹树脂上限解析失败：{text}")
    return limit


def parse_full_recovery_seconds(text: str) -> int:
    normalized = normalize_resin_text(text)
    if "原粹树脂已完全恢复" in normalized:
        return 0
    match = re.search(r"全部恢复(?P<time>\d{1,3}:\d{2}:\d{2})", normalized)
    if match is None:
        matches = re.findall(r"\d{1,3}:\d{2}:\d{2}", normalized)
        if not matches:
            raise ValueError(f"未识别到全部恢复时间：{text}")
        value = matches[-1]
    else:
        value = match.group("time")
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    if minutes > 59 or seconds > 59:
        raise ValueError(f"树脂恢复时间格式无效：{value}")
    return hours * 3600 + minutes * 60 + seconds


def calculate_current_resin(limit: int, recovery_seconds: int) -> OriginalResinInfo:
    if limit <= 0 or recovery_seconds < 0:
        raise ValueError("树脂上限和恢复时间必须为非负有效值")
    missing = math.ceil(recovery_seconds / (8 * 60))
    if missing > limit:
        raise ValueError(f"计算缺失树脂 {missing} 超过树脂上限 {limit}")
    return OriginalResinInfo(limit - missing, limit)


def calculate_supplemental_resin_quantity(
    current: int, limit: int, available: int | None = None, *, max_quick_use: int = 20,
) -> int:
    """How many 60-resin consumables fit without overflowing the resin cap."""

    quantity = max(0, (int(limit) - int(current)) // 60)
    if available is not None:
        quantity = min(quantity, max(0, int(available)))
    return min(quantity, max(0, int(max_quick_use)))


class AutoEncounterTask:
    """Portable fallback loop currently used by ley-line tasks."""

    def __init__(
        self,
        ctx: GameContext,
        *,
        name: str,
        route_path: str | Path | None = None,
        rounds: int = 1,
        combat_strategy_path: str | None = None,
        timeout_s: float = 240,
        party_slots: dict[str, int] | None = None,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.name = name
        self.route_path = Path(route_path).expanduser() if route_path else None
        self.rounds = max(1, int(rounds))
        self.party_slots = party_slots or {}
        self.log = log
        self.fight = AutoFightTask(
            ctx,
            combat_strategy_path=combat_strategy_path,
            timeout_s=max(30, float(timeout_s)),
            party_slots=self.party_slots,
            log=log,
        )

    def _route(self) -> PathingTask:
        if self.route_path is None:
            raise FileNotFoundError(
                f"{self.name} 未配置 routePath/pathingFile；请提供 BetterGI 地图追踪路线"
            )
        if not self.route_path.is_file():
            raise FileNotFoundError(f"{self.name} 路线不存在：{self.route_path}")
        return PathingTask.load(self.route_path)

    def _pathing(self, route: PathingTask) -> bool:
        return PathingExecutor(
            self.ctx, party_slots=self.party_slots, log=self.log,
        ).run(route)

    def _claim_reward(self) -> bool:
        self.ctx.input.key_press("F")
        self.ctx.sleep(1600)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            hits = self.ctx.capture_region().find_multi(
                RecognitionObject.ocr(900, 350, 900, 650), limit=30
            )
            for hit in hits:
                text = hit.text.replace(" ", "")
                if any(term in text for term in (
                    "使用浓缩树脂", "使用原粹树脂", "浓缩树脂", "原粹树脂",
                    "领取奖励", "收取奖励", "Claim Reward",
                )):
                    hit.click()
                    self.ctx.sleep(1400)
                    return True
            self.ctx.sleep(350)
        return False

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        for round_no in range(1, self.rounds + 1):
            if cancelled and cancelled():
                return False
            self.log(f"[{self.name}] 第 {round_no}/{self.rounds} 轮")
            if not self._pathing(self._route()):
                return False
            if cancelled and cancelled():
                return False
            if not self.fight.run(cancelled=cancelled):
                return False
            self._claim_reward()
        self.log(f"[{self.name}] 完成")
        return True


class AutoBossTask(AutoEncounterTask):
    REWARD_TEMPLATE = "box.png"

    def __init__(
        self,
        ctx: GameContext,
        *,
        boss_name: str = "",
        route_path: str | Path | None = None,
        specify_run_count: bool = False,
        rounds: int = 1,
        use_transient_resin: bool = False,
        use_fragile_resin: bool = False,
        revive_retry_count: int = 3,
        return_to_statue_after_each_round: bool = False,
        reward_recognition_enabled: bool = False,
        reward_max_pages: int = 3,
        team_name: str = "",
        **kwargs,
    ):
        super().__init__(ctx, name="AutoBoss", route_path=route_path, rounds=rounds, **kwargs)
        self.boss_name = str(boss_name or "")
        self.policy = BossRunPolicy(bool(specify_run_count), int(rounds))
        self.use_transient_resin = bool(use_transient_resin)
        self.use_fragile_resin = bool(use_fragile_resin)
        self.revive_retry_count = int(revive_retry_count)
        self.return_to_statue_after_each_round = bool(return_to_statue_after_each_round)
        self.reward_recognition_enabled = bool(reward_recognition_enabled)
        self.reward_max_pages = max(1, int(reward_max_pages))
        self.team_name = str(team_name or "")
        self.reward_summary: dict[str, int] = {}
        self._reward_recognizer = None
        self._reward_ro = None
        self._reward_template_missing = False
        self._api = GenshinApi(ctx, log=self.log)

    def _validate(self) -> None:
        if not self.boss_name and self.route_path is None:
            raise ValueError("请选择需要讨伐的首领，或提供 routePath/pathingFile")
        if self.boss_name and self.boss_name not in SUPPORTED_BOSSES and self.route_path is None:
            raise ValueError(f"暂不支持首领：{self.boss_name}")
        if self.revive_retry_count < 0:
            raise ValueError("角色死亡后重试次数不能小于 0")
        if not self.policy.specify_run_count and (
            self.use_transient_resin or self.use_fragile_resin
        ):
            raise ValueError("只有指定讨伐次数模式才能开启须臾树脂或脆弱树脂补充")
        for path in self._required_routes():
            if not path.is_file():
                raise FileNotFoundError(f"未找到首领路线文件：{path.name}")

    def _required_routes(self) -> list[Path]:
        if self.route_path is not None:
            return [self.route_path]
        if self.boss_name in NO_PATHING_SUPPORT_BOSSES:
            return [
                BOSS_ROUTE_ROOT / f"{self.boss_name}强制传送.json",
                BOSS_ROUTE_ROOT / f"{self.boss_name}键鼠前往.json",
            ]
        routes = [BOSS_ROUTE_ROOT / f"{self.boss_name}前往.json"]
        if self.boss_name in TALK_TO_START_BOSSES:
            routes.append(BOSS_ROUTE_ROOT / f"{self.boss_name}战斗后快速前往.json")
        return routes

    def _route_at(self, path: Path) -> bool:
        self._api.returnMainUi()
        return self._pathing(PathingTask.load(path))

    def _navigate_to_boss(self) -> bool:
        target = self.boss_name or (self.route_path.name if self.route_path else "自定义首领")
        self.log(f"[AutoBoss] 前往 {target}")
        if self.route_path is not None:
            return self._route_at(self.route_path)
        if self.boss_name in NO_PATHING_SUPPORT_BOSSES:
            if not self._route_at(BOSS_ROUTE_ROOT / f"{self.boss_name}强制传送.json"):
                return False
            MacroPlayer(self.ctx.input, sleep=self.ctx.sleep, log=self.log).play(
                load_keymouse(BOSS_ROUTE_ROOT / f"{self.boss_name}键鼠前往.json")
            )
            return True
        return self._route_at(BOSS_ROUTE_ROOT / f"{self.boss_name}前往.json")

    def _template(self, name: str, roi=None) -> RecognitionObject:
        path = BOSS_ASSET_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(
                f"缺少 AutoBoss 识别模板 {name}；请运行 "
                "tools/fetch_map_assets.py --auto-boss"
            )
        ro = RecognitionObject.template_match(Mat.from_file(str(path)))
        ro.threshold = 0.8
        ro.roi = roi
        return ro

    @staticmethod
    def _ocr_text(region, roi: tuple[float, float, float, float]) -> str:
        hits = region.find_multi(RecognitionObject.ocr(*roi), limit=40)
        hits.sort(key=lambda hit: (hit.y, hit.x))
        return "".join(hit.text for hit in hits)

    def _recognize_original_resin_info(self) -> OriginalResinInfo:
        region = self.ctx.capture_region()
        icon = region.find(self._template(
            "original_resin_top_icon.png", (1200, 25, 580, 50)
        ))
        if not icon.is_exist():
            raise RuntimeError("未找到原粹树脂图标")
        icon.click()
        self.ctx.sleep(500)
        clicked = self.ctx.capture_region()
        count_text = self._ocr_text(
            clicked, (icon.x + icon.width + 25, 37, 120, 28)
        )
        limit = parse_resin_limit(count_text)
        recovery_text = self._ocr_text(
            clicked, (max(0, icon.x - 13), icon.y + icon.height + 29, 220, 150)
        )
        info = calculate_current_resin(
            limit, parse_full_recovery_seconds(recovery_text)
        )
        self.log(f"[AutoBoss] 当前原粹树脂 {info.count}/{info.limit}")
        return info

    def _tap_text(
        self, text: str, roi: tuple[float, float, float, float], timeout_s: float = 3,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            hits = self.ctx.capture_region().find_multi(
                RecognitionObject.ocr(*roi), limit=30
            )
            target = next((hit for hit in hits if text in hit.text.replace(" ", "")), None)
            if target is not None:
                target.click()
                self.ctx.sleep(400)
                return True
            self.ctx.sleep(250)
        return False

    def _wait_and_close_obtain_dialog(self) -> bool:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            hits = self.ctx.capture_region().find_multi(
                RecognitionObject.ocr(820, 230, 300, 160), limit=20
            )
            obtain = next((hit for hit in hits if "获得" in hit.text), None)
            if obtain is not None:
                obtain.click()
                self.ctx.sleep(800)
                return True
            self.ctx.sleep(250)
        return False

    def _try_use_supplemental_resin(self, resin: OriginalResinInfo) -> bool:
        target_quantity = calculate_supplemental_resin_quantity(
            resin.count, resin.limit
        )
        if target_quantity <= 0:
            self.log("[AutoBoss] 使用补充树脂会超过上限，停止补充")
            return False

        region = self.ctx.capture_region()
        title = self._ocr_text(region, (800, 220, 350, 100))
        if "补充原粹树脂" not in normalize_resin_text(title):
            button = region.find(self._template(
                "open_resin_supplement_pane_button.png", (1200, 25, 580, 50)
            ))
            if not button.is_exist():
                self.log("[AutoBoss] 未找到补充原粹树脂入口")
                return False
            button.click()
            self.ctx.sleep(700)

        options: list[tuple[str, str]] = []
        if self.use_transient_resin:
            options.append(("须臾树脂", "transient_resin_in_supplement_pane.png"))
        if self.use_fragile_resin:
            options.append(("脆弱树脂", "fragile_resin_in_supplement_pane.png"))

        for resin_name, template_name in options:
            panel = self.ctx.capture_region()
            icon = panel.find(self._template(template_name, (644, 378, 620, 192)))
            if not icon.is_exist():
                self.log(f"[AutoBoss] 补充面板未找到{resin_name}")
                continue
            icon.click()
            self.ctx.sleep(400)
            if not self._tap_text("使用", (1080, 680, 250, 170), timeout_s=3):
                continue

            quick = normalize_resin_text(self._ocr_text(
                self.ctx.capture_region(), (820, 220, 400, 160)
            ))
            used_quantity = 1
            if "快捷使用" in quick:
                available_text = self._ocr_text(
                    self.ctx.capture_region(), (1170, 600, 130, 80)
                )
                available = max(
                    (int(value) for value in re.findall(r"\d+", normalize_resin_text(available_text))),
                    default=0,
                )
                used_quantity = calculate_supplemental_resin_quantity(
                    resin.count, resin.limit, available
                )
                if used_quantity <= 0:
                    self.log(f"[AutoBoss] 未识别到可用的{resin_name}")
                    return False
                plus = self.ctx.capture_region().find(self._template(
                    "increase_resin_usage_quantity_button.png", (1240, 590, 110, 100)
                ))
                for _ in range(used_quantity - 1):
                    if not plus.is_exist():
                        return False
                    plus.click()
                    self.ctx.sleep(180)
                if not self._tap_text("使用", (1080, 680, 250, 170), timeout_s=3):
                    return False

            if not self._wait_and_close_obtain_dialog():
                self.log(f"[AutoBoss] 使用{resin_name}后未识别到获得界面")
                return False
            self.log(f"[AutoBoss] 已使用 {used_quantity} 个{resin_name}")
            return True
        self.log("[AutoBoss] 未找到可用的须臾树脂或脆弱树脂")
        return False

    def _ensure_resin_before_round(self) -> bool:
        """Preflight the 40-resin boss cost; recognition failures defer to claim UI."""

        try:
            if not self._api._tp_for().open_map():
                raise RuntimeError("未能打开大地图")
            resin = self._recognize_original_resin_info()
            if resin.count >= 40:
                return True
            if not self.policy.specify_run_count:
                self.log("[AutoBoss] 树脂不足 40，树脂耗尽模式结束")
                return False
            if not (self.use_transient_resin or self.use_fragile_resin):
                self.log("[AutoBoss] 树脂不足且未开启补充树脂")
                return False
            if not self._try_use_supplemental_resin(resin):
                return False
            refreshed = self._recognize_original_resin_info()
            return refreshed.count >= 40
        except Exception as error:
            self.log(f"[AutoBoss] 战前树脂预检失败，交由领奖界面兜底：{error}")
            return True
        finally:
            self._api.returnMainUi()

    def _reward_template(self) -> RecognitionObject | None:
        if self._reward_ro is not None:
            return self._reward_ro
        if self._reward_template_missing:
            return None
        path = BOSS_ASSET_ROOT / self.REWARD_TEMPLATE
        if not path.is_file():
            self._reward_template_missing = True
            self.log(
                "[AutoBoss] 缺少征讨之花模板；请运行 "
                "tools/fetch_map_assets.py --auto-boss，使用 OCR/直走回退"
            )
            return None
        ro = self._template(self.REWARD_TEMPLATE, (0, 80, 1920, 850))
        self._reward_ro = ro
        return ro

    def _reward_panel_state(self, region=None) -> str | None:
        region = region or self.ctx.capture_region()
        hits = region.find_multi(RecognitionObject.ocr(820, 700, 400, 120), limit=15)
        text = "".join(hit.text.replace(" ", "") for hit in hits)
        if "补充原粹树脂" in text or ("补充" in text and "树脂" in text):
            return "insufficient"
        if "使用原粹树脂" in text:
            return "ready"
        return None

    def _interaction_visible(self, region=None) -> bool:
        region = region or self.ctx.capture_region()
        hits = region.find_multi(RecognitionObject.ocr(1180, 260, 420, 500), limit=20)
        return any("征讨之花" in hit.text or "接触" in hit.text for hit in hits)

    def _navigate_to_reward(
        self, cancelled: Callable[[], bool] | None = None, timeout_s: float = 20,
    ) -> bool:
        """Touch equivalent of upstream's camera/movement/reward prompt race."""

        deadline = time.monotonic() + timeout_s
        moving = False
        last_interact = 0.0
        camera_misses = 0
        try:
            while time.monotonic() < deadline:
                if cancelled and cancelled():
                    return False
                region = self.ctx.capture_region()
                if self._reward_panel_state(region) is not None:
                    return True
                now = time.monotonic()
                if self._interaction_visible(region) and now - last_interact >= 0.3:
                    if moving:
                        self.ctx.input.key_up("W")
                        moving = False
                    self.ctx.input.key_press("F")
                    last_interact = now
                    self.ctx.sleep(400)
                    continue

                ro = self._reward_template()
                if ro is None:
                    if not moving:
                        self.ctx.input.key_down("W")
                        moving = True
                    self.ctx.sleep(800)
                    self.ctx.input.key_press("SPACE")
                    continue
                hit = region.find(ro)
                if hit is not None and hit.is_exist():
                    camera_misses = 0
                    center_x = hit.x + hit.width / 2
                    center_y = hit.y + hit.height / 2
                    if center_y > 600:
                        self.ctx.input.move_camera_by(0, 380)
                        self.ctx.sleep(250)
                        continue
                    offset = center_x - 960
                    if abs(offset) > 90:
                        if moving:
                            self.ctx.input.key_up("W")
                            moving = False
                        self.ctx.input.move_camera_by(offset, 0)
                        self.ctx.sleep(300)
                        continue
                else:
                    camera_misses += 1
                    if moving:
                        self.ctx.input.key_up("W")
                        moving = False
                    self.ctx.input.move_camera_by(180 if camera_misses % 8 else -720, 0)
                    self.ctx.sleep(250)
                    continue

                if not moving:
                    self.ctx.input.key_down("W")
                    moving = True
                self.ctx.sleep(800)
                self.ctx.input.key_press("SPACE")
        finally:
            if moving:
                self.ctx.input.key_up("W")
        self.log("[AutoBoss] 超时未找到征讨之花领奖界面")
        return False

    def _recognize_rewards(self, cancelled: Callable[[], bool] | None = None) -> None:
        if not self.reward_recognition_enabled:
            return
        from .reward_result import RewardResultRecognizer, crop_reward_band, detect_reward_card_rects

        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            if cancelled and cancelled():
                return
            if detect_reward_card_rects(crop_reward_band(self.ctx, self.ctx.capture_bgr())):
                break
            self.ctx.sleep(300)
        else:
            self.log("[AutoBoss] 奖励结果页未就绪，跳过本轮奖励识别")
            return
        try:
            if self._reward_recognizer is None:
                self._reward_recognizer = RewardResultRecognizer(self.ctx, log=self.log)
            rewards = self._reward_recognizer.recognize_multi_page(self.reward_max_pages)
        except Exception as error:
            self.log(f"[AutoBoss] 奖励识别失败，跳过本轮汇总：{error}")
            return
        for name, quantity in rewards.items():
            self.reward_summary[name] = self.reward_summary.get(name, 0) + quantity

    def _claim_boss_reward(
        self, cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if cancelled and cancelled():
                return False
            hits = self.ctx.capture_region().find_multi(
                RecognitionObject.ocr(820, 700, 400, 120), limit=20
            )
            for hit in hits:
                text = hit.text.replace(" ", "")
                if "补充原粹树脂" in text or ("补充" in text and "树脂" in text):
                    self.log("[AutoBoss] 原粹树脂不足，结束讨伐")
                    self.ctx.input.key_press("ESCAPE")
                    self.ctx.sleep(500)
                    self._api.returnMainUi()
                    return False
                if "使用原粹树脂" in text:
                    hit.click()
                    self.ctx.sleep(1200)
                    self._recognize_rewards(cancelled)
                    self._close_reward_result(cancelled)
                    return True
            self.ctx.sleep(300)
        self.log("[AutoBoss] 未能点击“使用原粹树脂”")
        return False

    def _close_reward_result(self, cancelled: Callable[[], bool] | None = None) -> bool:
        for attempt in range(20):
            if cancelled and cancelled():
                return False
            region = self.ctx.capture_region()
            if self._api._is_main_ui(region.bgr):
                return True
            hits = region.find_multi(RecognitionObject.ocr(800, 900, 400, 150), limit=10)
            if hits:
                hits[0].click()
            elif attempt > 5:
                self.ctx.input.click_ref(960, 540)
            self.ctx.sleep(300)
        return self._api.returnMainUi()

    def _is_character_dead(self) -> bool:
        hits = self.ctx.capture_region().find_multi(
            RecognitionObject.ocr(500, 250, 920, 650), limit=30
        )
        text = "".join(hit.text.replace(" ", "") for hit in hits)
        return any(term in text for term in ("角色倒下", "角色死亡", "复苏", "Revive"))

    def _reposition(self) -> bool:
        if self.boss_name in TALK_TO_START_BOSSES and self.route_path is None:
            return self._route_at(BOSS_ROUTE_ROOT / f"{self.boss_name}战斗后快速前往.json")
        if self.boss_name in NO_PATHING_SUPPORT_BOSSES and self.route_path is None:
            return self._navigate_to_boss()
        source = self.route_path or BOSS_ROUTE_ROOT / f"{self.boss_name}前往.json"
        original = PathingTask.load(source)
        if not original.positions:
            raise ValueError(f"首领路线缺少路径点：{source}")
        task = PathingTask(
            name=f"{original.name} 战后重新定位",
            map_name=original.map_name,
            positions=[original.positions[-1]],
            info=original.info,
            config=original.config,
            map_match_method=original.map_match_method,
            realtime_triggers=original.realtime_triggers,
            farming_info=original.farming_info,
        )
        if not self._pathing(task):
            return False
        self.ctx.sleep(4000)
        return True

    def _prepare(self) -> None:
        self._api.returnMainUi()
        if self.team_name and not self._api.switchParty(self.team_name):
            self.log(f"[AutoBoss] 未能切换队伍 {self.team_name}，保持当前队伍")

    def _run_loop(self, cancelled: Callable[[], bool] | None = None) -> dict[str, int]:
        self._prepare()
        successful_claims = 0
        should_navigate = True
        death_retries = 0
        while self.policy.should_continue(successful_claims):
            if cancelled and cancelled():
                break
            self.log(f"[AutoBoss] 开始第 {successful_claims + 1} 次讨伐 {self.boss_name}")
            if not self._ensure_resin_before_round():
                break
            if should_navigate and not self._navigate_to_boss():
                break
            if not self.fight.run(cancelled=cancelled):
                if self._is_character_dead() and death_retries < self.revive_retry_count:
                    death_retries += 1
                    self.log(
                        f"[AutoBoss] 角色死亡，返回七天神像并重试 "
                        f"{death_retries}/{self.revive_retry_count}"
                    )
                    self._api.tpToStatueOfTheSeven()
                    self.ctx.sleep(3000)
                    should_navigate = True
                    continue
                break
            if not self._navigate_to_reward(cancelled):
                break
            if not self._claim_boss_reward(cancelled):
                break
            successful_claims += 1
            death_retries = 0
            if not self.policy.should_continue(successful_claims):
                break
            if self.return_to_statue_after_each_round:
                self.log("[AutoBoss] 返回七天神像")
                self._api.tpToStatueOfTheSeven()
                self.ctx.sleep(3000)
                should_navigate = True
            else:
                should_navigate = False
                if not self._reposition():
                    break
        self.log(f"[AutoBoss] 完成，成功领取 {successful_claims} 次：{self.reward_summary}")
        return dict(self.reward_summary)

    def run(self, cancelled: Callable[[], bool] | None = None) -> dict[str, int]:
        self._validate()
        self.reward_summary.clear()
        try:
            return self._run_loop(cancelled)
        finally:
            self.ctx.input.release_all()


class AutoLeyLineTask(AutoEncounterTask):
    def __init__(self, ctx: GameContext, *, route_path: str | Path | None = None, **kwargs):
        super().__init__(ctx, name="AutoLeyLine", route_path=route_path, **kwargs)
