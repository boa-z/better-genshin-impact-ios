from types import SimpleNamespace
from unittest.mock import Mock

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
                 auto=False, guards=False, texts=None):
        self.trigger = trigger
        self.bgr = bgr if bgr is not None else np.full((1080, 1920, 3), 255, np.uint8)
        self.options = options or []
        self.page_close = page_close or Hit(exists=False)
        self.auto = Hit(exists=auto)
        self.guards = guards
        self.texts = texts or {}

    def find_multi(self, recognition, limit=10):
        return list(self.options) if recognition is self.trigger.ro_option else []

    def find(self, recognition):
        if recognition is self.trigger.ro_auto:
            return self.auto
        if recognition is self.trigger.ro_page_close:
            return self.page_close
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
        input=SimpleNamespace(click_ref=Mock()),
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
