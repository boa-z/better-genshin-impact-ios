from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import cv2
import numpy as np


class Hit:
    def __init__(self, *, x=0, y=0, text="", exists=True):
        self.x = x
        self.y = y
        self.text = text
        self._exists = exists
        self.click = Mock()

    def is_exist(self):
        return self._exists


class Clock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


class Region:
    def __init__(self, trigger, *, bgr=None, options=None, page_close=None,
                 auto=False, guards=False, texts=None, exclamation=False,
                 interaction=False, primogem=False, confirms=None,
                 hangout_selected=None, hangout_unselected=None,
                 hangout_skip=False, submit_exclamation=False, submit_goods=None,
                 submit_black_confirm=False, submit_white_confirm=False):
        self.trigger = trigger
        self.bgr = bgr if bgr is not None else np.full((1080, 1920, 3), 255, np.uint8)
        self.options = options or []
        self.page_close = page_close or Hit(exists=False)
        self.auto = Hit(exists=auto)
        self.guards = guards
        self.texts = texts or {}
        self.exclamation = Hit(exists=exclamation)
        self.interaction = Hit(exists=interaction)
        self.primogem = Hit(exists=primogem)
        self.confirms = confirms or {}
        self.hangout_selected = list(hangout_selected or [])
        self.hangout_unselected = list(hangout_unselected or [])
        self.hangout_skip = Hit(exists=hangout_skip)
        self.submit_exclamation = Hit(exists=submit_exclamation)
        self.submit_goods = list(submit_goods or [])
        self.submit_black_confirm = Hit(exists=submit_black_confirm)
        self.submit_white_confirm = Hit(exists=submit_white_confirm)

    def find_multi(self, recognition, limit=10):
        if recognition is self.trigger.ro_option:
            return list(self.options)
        if recognition is getattr(self.trigger, "ro_hangout_selected", None):
            return list(self.hangout_selected)
        if recognition is getattr(self.trigger, "ro_hangout_unselected", None):
            return list(self.hangout_unselected)
        if recognition is getattr(self.trigger, "ro_submit_goods", None):
            return list(self.submit_goods)
        return []

    def find(self, recognition):
        if recognition is self.trigger.ro_auto:
            return self.auto
        if recognition is getattr(self.trigger, "ro_exclamation", None):
            return self.exclamation
        if recognition is getattr(self.trigger, "ro_interaction", None):
            return self.interaction
        if recognition is self.trigger.ro_page_close:
            return self.page_close
        if recognition is getattr(self.trigger, "ro_primogem", None):
            return self.primogem
        if recognition in getattr(self.trigger, "ro_daily_confirm", ()):
            return self.confirms.get(recognition, Hit(exists=False))
        if recognition is getattr(self.trigger, "ro_hangout_skip", None):
            return self.hangout_skip
        if recognition is getattr(self.trigger, "ro_submit_exclamation", None):
            return self.submit_exclamation
        if recognition is getattr(self.trigger, "ro_submit_black_confirm", None):
            return self.submit_black_confirm
        if recognition is getattr(self.trigger, "ro_submit_white_confirm", None):
            return self.submit_white_confirm
        if recognition in self.trigger.ro_popup_guards:
            return Hit(exists=self.guards)
        if recognition.recognition_type == "Ocr":
            option_y = recognition.roi[1] + 12
            return Hit(text=self.texts.get(option_y, ""), exists=option_y in self.texts)
        return Hit(exists=False)


def make_trigger(**kwargs):
    from bgi_touch.triggers.autoskip import AutoSkipTrigger
    from bgi_touch.vision.coordinate import ScreenTransform

    clock = kwargs.pop("clock", Clock())
    ctx = SimpleNamespace(
        input=SimpleNamespace(click_ref=Mock(), key_press=Mock()),
        device=SimpleNamespace(tap=Mock()),
        sleep=Mock(),
        transform=ScreenTransform(1920, 1080),
    )
    trigger = AutoSkipTrigger(
        ctx,
        clock=clock,
        main_ui_detector=kwargs.pop("main_ui_detector", lambda _ctx, _frame: False),
        big_map_detector=kwargs.pop("big_map_detector", lambda _ctx, _frame: False),
        log=Mock(),
        **kwargs,
    )
    return trigger, ctx, clock


def test_autoskip_orders_options_by_screen_position_even_when_builtins_disabled():
    trigger, _ctx, _clock = make_trigger(skip_built_in_options=True)
    bottom, top, middle = Hit(x=1200, y=700), Hit(x=1200, y=300), Hit(x=1200, y=500)
    region = Region(trigger, options=[bottom, top, middle])

    trigger.on_frame(region)

    top.click.assert_called_once_with()
    middle.click.assert_not_called()
    bottom.click.assert_not_called()


def test_autoskip_custom_priority_still_applies_to_no_default_selection():
    trigger, _ctx, _clock = make_trigger(
        click_option="不选择选项", priority_texts=["重要"]
    )
    first, target = Hit(x=1200, y=300), Hit(x=1200, y=500)
    region = Region(
        trigger,
        options=[first, target],
        texts={300: "普通选项", 500: "这是重要选项"},
    )

    trigger.on_frame(region)

    first.click.assert_not_called()
    target.click.assert_called_once_with()


def test_autoskip_expedition_option_runs_one_key_task():
    from bgi_touch.tasks.expedition import OneKeyExpeditionTask

    trigger, ctx, _clock = make_trigger(auto_re_explore_enabled=True)
    option = Hit(x=1200, y=500)
    region = Region(trigger, options=[option], texts={500: "探索派遣"})

    with patch.object(OneKeyExpeditionTask, "run", return_value=True) as run:
        trigger.on_frame(region)

    option.click.assert_called_once_with()
    ctx.sleep.assert_called_with(800)
    run.assert_called_once_with()


def test_autoskip_can_disable_expedition_automation():
    from bgi_touch.tasks.expedition import OneKeyExpeditionTask

    trigger, _ctx, _clock = make_trigger(auto_re_explore_enabled=False)
    option = Hit(x=1200, y=500)
    region = Region(trigger, options=[option], texts={500: "探索派遣"})

    with patch.object(OneKeyExpeditionTask, "run") as run:
        trigger.on_frame(region)

    option.click.assert_called_once_with()
    run.assert_not_called()


def test_autoskip_page_close_requires_recent_dialogue_and_stable_detection():
    clock = Clock(1.0)
    trigger, _ctx, _clock = make_trigger(clock=clock)
    close = Hit(x=1800, y=40)
    region = Region(trigger, page_close=close)

    trigger.on_frame(region)
    close.click.assert_not_called()

    trigger._last_dialogue_at = clock.value
    trigger.on_frame(region)
    close.click.assert_not_called()
    clock.value += 0.25
    trigger.on_frame(region)
    close.click.assert_called_once_with()


def test_autoskip_never_closes_protected_or_big_map_pages():
    clock = Clock(1.0)
    trigger, _ctx, _clock = make_trigger(clock=clock)
    trigger._last_dialogue_at = clock.value
    protected_close = Hit()
    protected = Region(trigger, page_close=protected_close, guards=True)
    trigger.on_frame(protected)
    clock.value += 1
    trigger.on_frame(protected)
    protected_close.click.assert_not_called()

    map_trigger, _ctx, map_clock = make_trigger(
        clock=Clock(1.0), big_map_detector=lambda _ctx, _frame: True
    )
    map_trigger._last_dialogue_at = map_clock.value
    map_close = Hit()
    big_map = Region(map_trigger, page_close=map_close)
    map_trigger.on_frame(big_map)
    map_clock.value += 1
    map_trigger.on_frame(big_map)
    map_close.click.assert_not_called()


def test_autoskip_real_templates_close_plain_page_but_keep_guarded_page():
    from bgi_touch.engine.recognition import ImageRegion

    clock = Clock(1.0)
    trigger, ctx, _clock = make_trigger(clock=clock)
    trigger._last_dialogue_at = clock.value
    rng = np.random.default_rng(7)
    plain = rng.integers(16, 240, (1080, 1920, 3), dtype=np.uint8)
    close_template = trigger.ro_page_close.template.bgr
    height, width = close_template.shape[:2]
    plain[40:40 + height, 1740:1740 + width] = close_template
    region = ImageRegion(ctx, plain)

    trigger.on_frame(region)
    clock.value += 0.25
    trigger.on_frame(region)
    assert ctx.device.tap.call_count == 1

    guard_trigger, guard_ctx, guard_clock = make_trigger(clock=Clock(1.0))
    guard_trigger._last_dialogue_at = guard_clock.value
    guarded = plain.copy()
    guard_template = guard_trigger.ro_popup_guards[0].template.bgr
    guard_h, guard_w = guard_template.shape[:2]
    guarded[30:30 + guard_h, 60:60 + guard_w] = guard_template
    guard_region = ImageRegion(guard_ctx, guarded)
    guard_trigger.on_frame(guard_region)
    guard_clock.value += 1
    guard_trigger.on_frame(guard_region)
    guard_ctx.device.tap.assert_not_called()


def test_autoskip_clicks_bottom_triangle_item_popup():
    trigger, ctx, clock = make_trigger(clock=Clock(2.0))
    trigger._last_dialogue_at = clock.value
    frame = np.zeros((1080, 1920, 3), np.uint8)
    yellow = cv2.cvtColor(np.uint8([[[15, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    triangle = np.array([[955, 990], [965, 990], [960, 998]], np.int32)
    cv2.fillPoly(frame, [triangle], tuple(int(value) for value in yellow))

    trigger.on_frame(Region(trigger, bgr=frame))

    x, y = ctx.input.click_ref.call_args.args
    assert 955 <= x <= 965
    assert 990 <= y <= 998


def test_autoskip_closes_character_intro_banner_by_two_color_geometry():
    trigger, ctx, clock = make_trigger(clock=Clock(2.0))
    trigger._last_dialogue_at = clock.value
    frame = np.zeros((1080, 1920, 3), np.uint8)
    light = cv2.cvtColor(np.uint8([[[22, 18, 242]]]), cv2.COLOR_HSV2BGR)[0, 0]
    dark = cv2.cvtColor(np.uint8([[[110, 70, 100]]]), cv2.COLOR_HSV2BGR)[0, 0]
    frame[380:532, 45:1875] = light
    frame[532:685, 45:1875] = dark

    trigger.on_frame(Region(trigger, bgr=frame))

    ctx.input.click_ref.assert_called_once_with(100, 100)


def test_autoskip_black_screen_click_is_throttled_and_skips_full_black():
    trigger, ctx, clock = make_trigger(clock=Clock(2.0))
    frame = np.full((1080, 1920, 3), 255, np.uint8)
    frame[360:720, :1200] = 0
    region = Region(trigger, bgr=frame)

    trigger.on_frame(region)
    ctx.input.click_ref.assert_called_once_with(960, 540)
    trigger.on_frame(region)
    assert ctx.input.click_ref.call_count == 1
    clock.value += 1.3
    trigger.on_frame(Region(trigger, bgr=np.zeros_like(frame)))
    assert ctx.input.click_ref.call_count == 1


def test_autoskip_honors_custom_pause_and_select_keyword_lists():
    trigger, _ctx, _clock = make_trigger(
        pause_texts=["稍后"], select_texts=["立即"],
        click_option="优先选择最后一个选项",
    )
    first = Hit(x=1200, y=300)
    pause = Hit(x=1200, y=500)
    region = Region(
        trigger,
        options=[first, pause],
        texts={300: "立即开始", 500: "稍后再说"},
    )

    trigger.on_frame(region)

    first.click.assert_called_once_with()
    pause.click.assert_not_called()

    first.click.reset_mock()
    trigger = make_trigger(pause_texts=["稍后"])[0]
    pause = Hit(x=1200, y=500)
    region = Region(trigger, options=[pause], texts={500: "稍后再说"})
    trigger.on_frame(region)
    pause.click.assert_not_called()


def test_autoskip_daily_reward_dismissal_uses_current_frame_only():
    trigger, ctx, clock = make_trigger(clock=Clock(3.0))
    trigger._daily_reward_until = clock.value + 10
    region = Region(trigger, primogem=True)

    trigger.on_frame(region)

    ctx.input.click_ref.assert_called_once_with(960, 900)
    assert trigger._daily_reward_until == float("-inf")


def test_autoskip_hangout_prefers_configured_ending_keyword():
    trigger, _ctx, clock = make_trigger(
        clock=Clock(4.0), auto_hangout_event_enabled=True,
        auto_hangout_end_choose="结局A",
        hangout_options={"结局A": ["走左边"]},
    )
    visited = Hit(x=900, y=400)
    target = Hit(x=900, y=520)
    region = Region(
        trigger,
        hangout_selected=[visited],
        hangout_unselected=[target],
        texts={412: "走右边", 532: "走左边"},
    )

    trigger.on_frame(region)

    target.click.assert_called_once_with()
    visited.click.assert_not_called()
    assert trigger._last_hangout_click_at == clock.value


def test_autoskip_interaction_key_mode_does_not_click_stale_option_region():
    trigger, ctx, _clock = make_trigger(use_interaction_key=True)
    option = Hit(x=1200, y=500)
    region = Region(trigger, options=[option])

    trigger.on_frame(region)

    ctx.input.key_press.assert_called_once_with("F")
    option.click.assert_not_called()


def test_autoskip_submit_goods_is_a_three_frame_state_machine():
    trigger, _ctx, clock = make_trigger(clock=Clock(5.0))
    exclamation = Region(trigger, submit_exclamation=True,
                         submit_goods=[Hit(x=700, y=300)])

    trigger._last_dialogue_at = clock.value
    trigger.on_frame(exclamation)
    goods = exclamation.submit_goods[0]
    goods.click.assert_called_once_with()

    black = Region(trigger, submit_black_confirm=True)
    clock.value += 0.5
    trigger.on_frame(black)
    black.submit_black_confirm.click.assert_called_once_with()

    white = Region(trigger, submit_white_confirm=True)
    clock.value += 0.5
    trigger.on_frame(white)
    white.submit_white_confirm.click.assert_called_once_with()
    assert trigger._submit_stage is None
