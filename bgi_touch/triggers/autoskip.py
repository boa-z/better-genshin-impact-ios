"""自动剧情触发器（原版 AutoSkip 的移动端适配）。

检测条件（防误触，先判定在对话中）：
- 对话选项图标 icon_option 模板命中 → 点选项（可偏好含指定文本/最上面一项）
- 或左上角自动播放指示（stop_auto 模板）命中 → 点屏幕中下部推进对话
- 对话结束后的有限窗口内，稳定识别并关闭普通页、道具页和初见角色横幅

模板来自原版 AutoSkip/Assets（1080p 基准，识别层自动缩放）。
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..engine.context import GameContext
from ..engine.recognition import ImageRegion, Mat, RecognitionObject
from ..vision.game_ui import is_big_map_ui, is_main_ui

TEMPLATES = Path(__file__).resolve().parents[2] / "assets" / "templates" / "autoskip"
HANGOUT_CONFIG = Path(__file__).resolve().parents[2] / "assets" / "config" / "autoskip" / "hangout.json"
AUTOPICK_TEMPLATES = Path(__file__).resolve().parents[2] / "assets" / "templates" / "autopick"


def _compact_text(value: object) -> str:
    """Normalize OCR text in the same way as BetterGI's StringUtils helper."""

    return "".join(str(value or "").split())


def _split_keywords(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Sequence[object] = value.replace("\r", "\n").replace("；", ";").split(";")
        expanded: list[str] = []
        for item in values:
            expanded.extend(str(item).split("\n"))
        values = expanded
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        values = (value,)
    return tuple(
        text for item in values
        if (text := _compact_text(item))
    )


def _load_hangout_config(path: Path = HANGOUT_CONFIG) -> dict[str, tuple[str, ...]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, tuple[str, ...]] = {}
    for name, values in raw.items():
        keywords = _split_keywords(values)
        if keywords:
            result[_compact_text(name)] = keywords
    return result


class AutoSkipTrigger:
    name = "AutoSkip"

    # A Wi-Fi iPhone screenshot can take 3.5-5.6 seconds. Two-frame stable
    # confirmation therefore needs a slightly wider window than desktop's 10s.
    POPUP_WINDOW_S = 15.0
    PAGE_CLOSE_STABLE_S = 0.2
    BLACK_CLICK_INTERVAL_S = 1.2
    ITEM_CLICK_INTERVAL_S = 1.0
    HANGOUT_CLICK_INTERVAL_S = 1.2
    DAILY_REWARD_WINDOW_S = 10.0
    SUBMIT_WINDOW_S = 3.0
    SUBMIT_STAGE_TIMEOUT_S = 4.0

    def __init__(self, ctx: GameContext, prefer_text: str | None = None,
                 priority_texts: list[str] | None = None,
                 click_option: str = "优先选择第一个选项",
                 quickly_skip: bool = True,
                 skip_built_in_options: bool = False,
                 after_choose_delay_ms: int = 0,
                 before_confirm_delay_ms: int = 0,
                 close_popup_pages: bool = True,
                 auto_re_explore_enabled: bool = True,
                 auto_get_daily_rewards_enabled: bool = True,
                 auto_wait_dialogue_option_voice_enabled: bool = False,
                 dialogue_option_voice_max_wait_seconds: int = 30,
                 default_pause_texts: Sequence[str] | str | None = None,
                 pause_texts: Sequence[str] | str | None = None,
                 select_texts: Sequence[str] | str | None = None,
                 auto_hangout_event_enabled: bool = False,
                 auto_hangout_end_choose: str = "",
                 auto_hangout_choose_option_sleep_delay: int = 0,
                 auto_hangout_press_skip_enabled: bool = True,
                 hangout_options: Mapping[str, Sequence[str]] | None = None,
                 hangout_config_path: str | Path | None = None,
                 submit_goods_enabled: bool = True,
                 use_interaction_key: bool = False,
                 interaction_key: str = "F",
                 voice_waiter: Callable[[int], bool] | None = None,
                 log: Callable[[str], None] = print,
                 clock: Callable[[], float] = time.monotonic,
                 main_ui_detector: Callable[[object, np.ndarray], bool] = is_main_ui,
                 big_map_detector: Callable[[object, np.ndarray], bool] = is_big_map_ui):
        self.ctx = ctx
        self.enabled = True
        self.priority_texts = list(priority_texts or ([prefer_text] if prefer_text else []))
        self.click_option = click_option
        self.quickly_skip = quickly_skip
        self.skip_built_in_options = skip_built_in_options
        self.after_choose_delay_ms = max(0, int(after_choose_delay_ms))
        self.before_confirm_delay_ms = max(0, int(before_confirm_delay_ms))
        self.close_popup_pages = bool(close_popup_pages)
        self.auto_re_explore_enabled = bool(auto_re_explore_enabled)
        self.auto_get_daily_rewards_enabled = bool(auto_get_daily_rewards_enabled)
        self.auto_wait_dialogue_option_voice_enabled = bool(
            auto_wait_dialogue_option_voice_enabled
        )
        self.dialogue_option_voice_max_wait_seconds = max(
            0, min(600, int(dialogue_option_voice_max_wait_seconds))
        )
        self.default_pause_texts = _split_keywords(default_pause_texts)
        self.pause_texts = _split_keywords(pause_texts)
        self.select_texts = _split_keywords(select_texts)
        self.auto_hangout_event_enabled = bool(auto_hangout_event_enabled)
        self.auto_hangout_end_choose = _compact_text(auto_hangout_end_choose)
        self.auto_hangout_choose_option_sleep_delay = max(
            0, int(auto_hangout_choose_option_sleep_delay)
        )
        self.auto_hangout_press_skip_enabled = bool(auto_hangout_press_skip_enabled)
        self.hangout_options = self._normalize_hangout_options(
            hangout_options,
            hangout_config_path,
        )
        self.submit_goods_enabled = bool(submit_goods_enabled)
        self.use_interaction_key = bool(use_interaction_key)
        self.interaction_key = str(interaction_key or "F")
        self.voice_waiter = voice_waiter
        self._option_delay_ms = max(0, int(after_choose_delay_ms))
        self._option_wait_until: float | None = None
        self._before_confirm_until: float | None = None
        self._option_delay_consumed = False
        self.log = log
        self._clock = clock
        self._main_ui_detector = main_ui_detector
        self._big_map_detector = big_map_detector
        self._last_dialogue_at: float | None = None
        self._page_close_seen_at: float | None = None
        self._last_black_click_at = float("-inf")
        self._last_item_click_at = float("-inf")
        self._last_hangout_click_at = float("-inf")
        self._daily_reward_until = float("-inf")
        self._submit_stage: str | None = None
        self._submit_stage_until = float("-inf")
        self._last_submit_at = float("-inf")
        # 选项图标出现在屏幕右侧偏下（ref 空间 ROI 收窄降误报）
        self.ro_option = RecognitionObject.template_match(
            Mat.from_file(str(TEMPLATES / "icon_option.png")), 1000, 280, 850, 700)
        self.ro_option.threshold = 0.75
        # 对话中的左上"自动播放"指示
        self.ro_auto = RecognitionObject.template_match(
            Mat.from_file(str(TEMPLATES / "stop_auto.png")), 0, 0, 400, 140)
        self.ro_auto.threshold = 0.75
        self.ro_exclamation = self._optional_template(
            "icon_exclamation.png", 1000, 280, 850, 700, threshold=0.75,
        )
        self.ro_interaction = self._optional_template(
            "F.png", 1000, 280, 850, 700, threshold=0.75,
            root=AUTOPICK_TEMPLATES,
        )
        self.ro_page_close = RecognitionObject.template_match(
            Mat.from_file(str(TEMPLATES / "page_close.png")), 1600, 0, 320, 160)
        self.ro_page_close.threshold = 0.72
        self.ro_popup_guards = []
        for name in ("guiding_notes.png", "chat_history.png", "valiant_chronicles.png"):
            guard = RecognitionObject.template_match(
                Mat.from_file(str(TEMPLATES / name)), 0, 0, 320, 180)
            guard.threshold = 0.75
            self.ro_popup_guards.append(guard)
        self.ro_primogem = self._optional_template(
            "primogem.png", 0, 360, 1920, 360, threshold=0.76,
        )
        self.ro_daily_confirm = tuple(
            recognition for name in ("comfirm_btn1.png", "comfirm_btn2.png")
            if (recognition := self._optional_template(name, 500, 550, 920, 480, threshold=0.72))
        )
        self.ro_hangout_selected = self._optional_template(
            "hangout_selected.png", threshold=0.78,
        )
        self.ro_hangout_unselected = self._optional_template(
            "hangout_unselected.png", threshold=0.78,
        )
        self.ro_hangout_skip = self._optional_template(
            "hangout_skip.png", 0, 0, 400, 180, threshold=0.78,
        )
        self.ro_submit_exclamation = self._optional_template(
            "submit_icon_exclamation.png", 0, 0, 1920, 270, threshold=0.78,
        )
        self.ro_submit_goods = self._optional_template(
            "submit_goods.png", 0, 0, 960, 360, threshold=0.86,
        )
        self.ro_submit_black_confirm = self._optional_template(
            "btn_black_confirm.png", 500, 550, 920, 480, threshold=0.76,
            root=Path(__file__).resolve().parents[2] / "assets" / "templates" / "artifact_salvage",
        )
        self.ro_submit_white_confirm = self._optional_template(
            "btn_white_confirm.png", 500, 550, 920, 480, threshold=0.74,
            root=Path(__file__).resolve().parents[2] / "assets" / "templates" / "artifact_salvage",
        )

    @staticmethod
    def _normalize_hangout_options(
        values: Mapping[str, Sequence[str]] | None,
        path: str | Path | None,
    ) -> dict[str, tuple[str, ...]]:
        if values is None:
            loaded = _load_hangout_config(
                Path(path).expanduser() if path else HANGOUT_CONFIG
            )
            return loaded
        result: dict[str, tuple[str, ...]] = {}
        for name, options in values.items():
            key = _compact_text(name)
            keywords = _split_keywords(options)
            if key and keywords:
                result[key] = keywords
        return result

    @staticmethod
    def _optional_template(
        name: str,
        x: float | None = None,
        y: float | None = None,
        w: float | None = None,
        h: float | None = None,
        *,
        threshold: float = 0.8,
        root: Path = TEMPLATES,
    ) -> RecognitionObject | None:
        path = root / name
        if not path.is_file():
            return None
        try:
            recognition = RecognitionObject.template_match(
                Mat.from_file(str(path)), x, y, w, h,
            )
            recognition.threshold = threshold
            return recognition
        except (OSError, ValueError):
            return None

    def on_frame(self, region: ImageRegion) -> None:
        if not self.enabled:
            return
        now = self._clock()

        # These two flows are deliberately advanced by later frames instead of
        # taking a fresh screenshot after every click.  DeviceHub's stream is
        # caller-owned; a trigger must never become a second screenshot loop.
        if self._advance_submit_goods(region, now):
            return
        if self._handle_daily_reward(region, now):
            return

        if self.auto_hangout_event_enabled and now - self._last_hangout_click_at >= self.HANGOUT_CLICK_INTERVAL_S:
            if self._choose_hangout_option(region, now):
                return

        options = region.find_multi(self.ro_option, limit=6)
        if options:
            self._last_dialogue_at = now
            self._page_close_seen_at = None
            if (
                self.use_interaction_key
                and not self.priority_texts
                and not self.select_texts
                and not self.pause_texts
                and not self.default_pause_texts
            ):
                self._choose_option_with_key(options, now)
            else:
                self._choose_option(region, options)
            return

        # Some 5.2+ conversations expose only the red exclamation marker.
        # It is an explicit interaction affordance, so it is safe to click
        # without falling through to the generic "continue" action.
        if self.ro_exclamation is not None:
            exclamation = region.find(self.ro_exclamation)
            if exclamation.is_exist():
                self._last_dialogue_at = now
                self._page_close_seen_at = None
                if self.click_option != "不选择选项" and self._prepare_option_click(now):
                    exclamation.click()
                    self.log("[AutoSkip] 点击感叹号对话选项")
                return

        if self.use_interaction_key and self.ro_interaction is not None:
            interaction = region.find(self.ro_interaction)
            if interaction.is_exist():
                if self.click_option != "不选择选项" and self._before_confirm_ready(now):
                    self.ctx.input.key_press(self.interaction_key)
                    self.log("[AutoSkip] 交互键推进对话")
                self._last_dialogue_at = now
                self._page_close_seen_at = None
                return

        auto_playing = region.find(self.ro_auto).is_exist()
        if auto_playing:
            self._last_dialogue_at = now
            self._page_close_seen_at = None
            if self.quickly_skip:
                # 对话进行中且无选项 → 点中下部推进。交互键模式用于
                # 后台/无鼠标语义的脚本，普通模式继续使用触控点按。
                if self._before_confirm_ready(now):
                    if self.use_interaction_key:
                        self.ctx.input.key_press("SPACE")
                    else:
                        self.ctx.input.click_ref(960, 820)
            return

        if (
            self.close_popup_pages
            and self._last_dialogue_at is not None
            and now - self._last_dialogue_at <= self.POPUP_WINDOW_S
        ):
            if self._handle_post_dialogue_popup(region, now):
                return
        else:
            self._page_close_seen_at = None

        self._click_black_screen(region, now)

    def _choose_option(self, region: ImageRegion, options: list) -> bool:
        """Apply BetterGI's custom-priority then default-order semantics."""

        options = sorted(options, key=lambda option: option.y)
        chosen = None
        texts: list[str] = []
        for option in options:
            texts.append(self._option_text(region, option))

        if self.priority_texts:
            for preferred in self.priority_texts:
                chosen = next((
                    option for option, text in zip(options, texts)
                    if preferred in text
                ), None)
                if chosen is not None:
                    break

        # Upstream custom priorities still apply when the default policy is
        # "不选择选项". SkipBuiltInClickOptions only disables built-in keyword
        # lists; it must not disable custom/default selection entirely.
        if chosen is None and not self.skip_built_in_options:
            for option, text in zip(options, texts):
                if any(keyword in text for keyword in self.select_texts):
                    chosen = option
                    break
            if chosen is None and any(
                any(keyword in text for keyword in self.pause_texts)
                for text in texts
            ):
                self.log("[AutoSkip] 命中自定义暂停选项，保留对话")
                return False

            if chosen is None and any(
                any(keyword in text for keyword in self.default_pause_texts)
                for text in texts
            ):
                self.log("[AutoSkip] 命中内置暂停选项，保留对话")
                return False

        if chosen is None:
            if self.click_option == "不选择选项":
                return False
            if self.click_option == "优先选择最后一个选项":
                chosen = options[-1]
            elif self.click_option == "随机选择选项":
                chosen = random.choice(options)
            else:
                chosen = options[0]

        if not self._prepare_option_click(self._clock()):
            return True

        self.log(f"[AutoSkip] 点击对话选项 @({chosen.x:.0f},{chosen.y:.0f})")
        chosen.click()
        chosen_index = options.index(chosen)
        chosen_text = texts[chosen_index] if texts else ""
        self._after_option_click(chosen_text)
        return True

    def _option_text(self, region: ImageRegion, option) -> str:
        line = region.find(RecognitionObject.ocr(
            option.x + 30, option.y - 12, 800, 60
        ))
        return _compact_text(line.text if line.is_exist() else "")

    def _choose_option_with_key(self, options: list, now: float) -> bool:
        if self.click_option == "不选择选项" or not self._before_confirm_ready(now):
            return False
        if self.click_option == "优先选择最后一个选项":
            # Move the in-game focus to the last visible choice before
            # confirming it.  The mobile keymap uses W/S for the same list
            # navigation semantics as the desktop interaction dialog.
            for _ in range(max(0, len(options) - 1)):
                self.ctx.input.key_press("S")
        elif self.click_option == "随机选择选项":
            for _ in range(random.randrange(len(options))):
                self.ctx.input.key_press("S")
        self.ctx.input.key_press(self.interaction_key)
        self.log("[AutoSkip] 交互键选择对话选项")
        return True

    def _prepare_option_click(self, now: float) -> bool:
        """Apply delay/voice policy without capturing another frame.

        DeviceHub currently exposes no game-audio loopback stream.  An
        optional callable can be supplied by a host that has one; otherwise
        the setting is retained as a compatibility no-op and logged once.
        """

        wait_until = getattr(self, "_option_wait_until", None)
        if wait_until is not None:
            if now < wait_until:
                return False
            self._option_wait_until = None

        if self.auto_wait_dialogue_option_voice_enabled:
            waiter = getattr(self, "voice_waiter", None)
            if callable(waiter):
                try:
                    if not bool(waiter(self.dialogue_option_voice_max_wait_seconds)):
                        return False
                except Exception as error:
                    self.log(f"[AutoSkip] 语音等待失败，继续点选：{error}")
            elif not getattr(self, "_voice_unavailable_logged", False):
                self.log("[AutoSkip] DeviceHub 无游戏音频回环，跳过语音等待")
                self._voice_unavailable_logged = True

        if self.after_choose_delay_ms and not self._option_delay_consumed:
            self._option_wait_until = now + self._option_delay_ms / 1000
            self._option_delay_consumed = True
            return False
        return True

    def _before_confirm_ready(self, now: float) -> bool:
        wait_until = getattr(self, "_before_confirm_until", None)
        if wait_until is not None:
            if now < wait_until:
                return False
            self._before_confirm_until = None
            return True
        if self.before_confirm_delay_ms:
            self._before_confirm_until = now + self.before_confirm_delay_ms / 1000
            return False
        return True

    def _after_option_click(self, chosen_text: str) -> None:
        self._option_delay_consumed = False
        if "每日" in chosen_text or "委托" in chosen_text:
            if self.auto_get_daily_rewards_enabled:
                self._daily_reward_until = self._clock() + self.DAILY_REWARD_WINDOW_S
                self.log("[AutoSkip] 已点击每日委托选项，等待奖励确认")
        if self.auto_re_explore_enabled and ("探索" in chosen_text or "派遣" in chosen_text):
            # The expedition task owns the page after the option click.  It
            # pauses the shared trigger loop, so this does not create a second
            # capture producer.
            self.ctx.sleep(800)
            from ..tasks.expedition import OneKeyExpeditionTask
            OneKeyExpeditionTask(self.ctx, log=self.log).run()

    def _handle_post_dialogue_popup(self, region: ImageRegion, now: float) -> bool:
        page_close = region.find(self.ro_page_close)
        if page_close.is_exist():
            protected = (
                self._main_ui_detector(self.ctx, region.bgr)
                or self._big_map_detector(self.ctx, region.bgr)
                or any(region.find(guard).is_exist() for guard in self.ro_popup_guards)
            )
            if protected:
                self._page_close_seen_at = None
                return False
            if self._page_close_seen_at is None:
                self._page_close_seen_at = now
                return False
            if now - self._page_close_seen_at < self.PAGE_CLOSE_STABLE_S:
                return False
            page_close.click()
            self._page_close_seen_at = None
            self._last_dialogue_at = now
            self.log("[AutoSkip] 关闭剧情弹出页")
            return True
        self._page_close_seen_at = None

        is_main = self._main_ui_detector(self.ctx, region.bgr)
        is_big_map = self._big_map_detector(self.ctx, region.bgr)
        if is_main or is_big_map:
            return False
        if self._close_item_popup(region.bgr, now):
            return True
        if self._close_character_popup(region.bgr, now):
            return True
        if self.submit_goods_enabled and now - self._last_dialogue_at <= self.SUBMIT_WINDOW_S:
            return self._start_submit_goods(region, now)
        return False

    def _handle_daily_reward(self, region: ImageRegion, now: float) -> bool:
        """Confirm the post-commission prompt and dismiss the primogem card."""

        if now > self._daily_reward_until:
            self._daily_reward_until = float("-inf")
            return False

        # The confirmation appears before the reward card on current clients.
        # Check both upstream variants because the black/white button changed
        # between game UI revisions.
        for recognition in self.ro_daily_confirm:
            hit = region.find(recognition)
            if hit.is_exist():
                hit.click()
                self.log("[AutoSkip] 确认每日委托奖励提示")
                return True

        if self.ro_primogem is not None:
            hit = region.find(self.ro_primogem)
            if hit.is_exist():
                # BetterGI clicks the fixed lower-center reward dismissal
                # point, not the primogem icon itself.
                self.ctx.input.click_ref(960, 900)
                self._daily_reward_until = float("-inf")
                self.log("[AutoSkip] 关闭每日委托原石奖励弹窗")
                return True
        return False

    def _choose_hangout_option(self, region: ImageRegion, now: float) -> bool:
        """Choose an unvisited hangout branch, honoring the configured ending."""

        selected = []
        unselected = []
        if self.ro_hangout_selected is not None:
            selected = region.find_multi(self.ro_hangout_selected, limit=20)
        if self.ro_hangout_unselected is not None:
            unselected = region.find_multi(self.ro_hangout_unselected, limit=20)

        choices = [(hit, True) for hit in selected] + [(hit, False) for hit in unselected]
        if choices:
            choices.sort(key=lambda item: float(getattr(item[0], "y", 0)))
            text_choices = [(hit, is_selected, self._option_text(region, hit))
                            for hit, is_selected in choices]
            target = None
            branch_keywords = self.hangout_options.get(self.auto_hangout_end_choose, ())
            if branch_keywords:
                target = next(
                    (
                        item for item in text_choices
                        if any(keyword in item[2] for keyword in branch_keywords)
                    ),
                    None,
                )
            if target is None:
                target = next((item for item in text_choices if not item[1]), None)
            if target is None:
                target = text_choices[0]

            hit, _is_selected, text = target
            if self.auto_hangout_choose_option_sleep_delay:
                self.ctx.sleep(self.auto_hangout_choose_option_sleep_delay)
            hit.click()
            self._last_hangout_click_at = now
            self._last_dialogue_at = now
            self.log(f"[AutoSkip] 邀约选项: {text}")
            return True

        if self.auto_hangout_press_skip_enabled and self.ro_hangout_skip is not None:
            skip = region.find(self.ro_hangout_skip)
            if skip.is_exist():
                skip.click()
                self._last_hangout_click_at = now
                self._last_dialogue_at = now
                self.log("[AutoSkip] 邀约点击跳过")
                return True
        return False

    def _advance_submit_goods(self, region: ImageRegion, now: float) -> bool:
        """Advance the submit-goods flow using one shared frame per tick."""

        if self._submit_stage is None:
            return False
        if now > self._submit_stage_until:
            self.log("[AutoSkip] 提交物品流程超时，清理状态")
            self._submit_stage = None
            return False

        if self._submit_stage == "item_confirm":
            if self.ro_submit_black_confirm is not None:
                confirm = region.find(self.ro_submit_black_confirm)
                if confirm.is_exist():
                    confirm.click()
                    self._submit_stage = "delivery_confirm"
                    self._submit_stage_until = now + self.SUBMIT_STAGE_TIMEOUT_S
                    self.log("[AutoSkip] 提交物品：放入物品")
                    return True
            return True

        if self._submit_stage == "delivery_confirm":
            if self.ro_submit_white_confirm is not None:
                confirm = region.find(self.ro_submit_white_confirm)
                if confirm.is_exist():
                    confirm.click()
                    self._submit_stage = None
                    self._last_submit_at = now
                    self.log("[AutoSkip] 提交物品：交付")
                    return True
            return True
        self._submit_stage = None
        return False

    def _start_submit_goods(self, region: ImageRegion, now: float) -> bool:
        if now - self._last_submit_at < self.SUBMIT_STAGE_TIMEOUT_S:
            return False
        if self.ro_submit_exclamation is None:
            return False
        exclamation = region.find(self.ro_submit_exclamation)
        if not exclamation.is_exist():
            return False

        goods = []
        if self.ro_submit_goods is not None:
            goods = region.find_multi(self.ro_submit_goods, limit=4)
        if goods:
            goods[0].click()
        else:
            rects = self._find_submit_item_rects(region.bgr)
            if not rects:
                return False
            x, y, width, height = rects[0]
            self.ctx.input.click_ref(x + width / 2, y + height / 2)
        self._submit_stage = "item_confirm"
        self._submit_stage_until = now + self.SUBMIT_STAGE_TIMEOUT_S
        self._last_submit_at = now
        self.log("[AutoSkip] 提交物品：选择物品")
        return True

    def _find_submit_item_rects(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Find the beige item cells used by submit-goods dialogs."""

        reference = self._reference_frame(bgr)
        hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
        # The upstream color is BGR (233, 229, 220); tolerate brightness and
        # JPEG/Wi-Fi compression variation while retaining a fairly narrow hue.
        mask = cv2.inRange(
            hsv,
            np.array((8, 8, 175), dtype=np.uint8),
            np.array((35, 100, 255), dtype=np.uint8),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width < 20 or height < 20 or y > 420:
                continue
            if width / max(1, height) > 5 or height / max(1, width) > 5:
                continue
            rects.append((x, y, width, height))
        return sorted(rects, key=lambda rect: (rect[1], rect[0]))[:4]

    def _reference_frame(self, bgr: np.ndarray) -> np.ndarray:
        """Center-crop wide iPhone frames into BetterGI's 1920x1080 space."""

        height, width = bgr.shape[:2]
        target_ratio = 16 / 9
        if width / max(1, height) >= target_ratio:
            content_width = max(1, round(height * target_ratio))
            left = max(0, (width - content_width) // 2)
            crop = bgr[:, left:left + content_width]
        else:
            content_height = max(1, round(width / target_ratio))
            top = max(0, (height - content_height) // 2)
            crop = bgr[top:top + content_height, :]
        return cv2.resize(crop, (1920, 1080), interpolation=cv2.INTER_AREA)

    def _close_item_popup(self, bgr: np.ndarray, now: float) -> bool:
        if now - self._last_item_click_at < self.ITEM_CLICK_INTERVAL_S:
            return False
        reference = self._reference_frame(bgr)
        crop = reference[980:1060, 945:975]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, np.array((0, 240, 229)), np.array((25, 255, 255)))
        blue = cv2.inRange(hsv, np.array((90, 156, 145)), np.array((99, 208, 253)))
        contours = []
        for mask in (yellow, blue):
            found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(found)
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if not (10 <= area <= 50 and len(approx) == 3):
                continue
            x, y, width, height = cv2.boundingRect(approx)
            self.ctx.input.click_ref(945 + x + width / 2, 980 + y + height / 2)
            self._last_item_click_at = now
            self._last_dialogue_at = now
            self.log(f"[AutoSkip] 关闭道具弹出页（三角面积 {area:.0f}）")
            return True
        return False

    def _close_character_popup(self, bgr: np.ndarray, now: float) -> bool:
        reference = self._reference_frame(bgr).copy()
        cv2.rectangle(reference, (240, 395), (540, 445), (229, 241, 245), -1)
        cv2.rectangle(reference, (290, 660), (500, 700), (101, 82, 74), -1)
        hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
        light = cv2.inRange(hsv, np.array((18, 16, 234)), np.array((27, 19, 250)))
        dark = cv2.inRange(hsv, np.array((101, 57, 95)), np.array((118, 85, 106)))
        combined = cv2.bitwise_or(light, dark)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image_area = reference.shape[0] * reference.shape[1]
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if height == 0:
                continue
            area_ratio = width * height / image_area
            aspect_ratio = width / height
            if not (0.24 < area_ratio < 0.3 and 5.6 <= aspect_ratio <= 7.2):
                continue
            if y <= 1080 * 0.3 or y + height >= 1080 * 0.7:
                continue
            if cv2.countNonZero(light[y:y + height, x:x + width]) == 0:
                continue
            if cv2.countNonZero(dark[y:y + height, x:x + width]) == 0:
                continue
            self.ctx.input.click_ref(100, 100)
            self._last_dialogue_at = now
            self.log("[AutoSkip] 关闭初见角色弹出页")
            return True
        return False

    def _click_black_screen(self, region: ImageRegion, now: float) -> bool:
        if now - self._last_black_click_at < self.BLACK_CLICK_INTERVAL_S:
            return False
        gray = cv2.cvtColor(region.bgr, cv2.COLOR_BGR2GRAY)
        band = gray[gray.shape[0] // 3:gray.shape[0] * 2 // 3]
        if band.size == 0:
            return False
        black_rate = float(np.count_nonzero(band == 0)) / band.size
        if not 0.5 <= black_rate < 0.98999:
            return False
        self.ctx.input.click_ref(960, 540)
        self._last_black_click_at = now
        self.log(f"[AutoSkip] 点击黑屏（黑色比例 {black_rate:.3f}）")
        return True
