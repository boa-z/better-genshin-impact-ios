from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


class Clock:
    value = 100.0

    def __call__(self):
        return self.value


class Hit:
    def __init__(self, exists=False, text=""):
        self._exists = exists
        self.text = text
        self.click = Mock()

    def is_exist(self):
        return self._exists


def make_trigger(*, main=False):
    from bgi_touch.triggers.game_loading import GameLoadingTrigger

    clock = Clock()
    ctx = SimpleNamespace(
        input=SimpleNamespace(click_ref=Mock()),
        _trigger_loop=None,
    )
    trigger = GameLoadingTrigger(
        ctx,
        clock=clock,
        main_ui_detector=lambda _ctx, _frame: main,
        log=Mock(),
    )
    trigger._template = lambda name, *_args, **_kwargs: name
    return clock, ctx, trigger


def test_game_loading_clicks_enter_from_existing_trigger_frame():
    _, ctx, trigger = make_trigger()
    enter = Hit(True)

    class Region:
        bgr = np.zeros((4, 8, 3), dtype=np.uint8)

        @staticmethod
        def find(ro):
            return enter if ro == "enter_game" else Hit(False)

        @staticmethod
        def find_multi(_ro, limit=40):
            raise AssertionError("enter match should avoid OCR")

    trigger.on_frame(Region())

    enter.click.assert_called_once_with()
    ctx.input.click_ref.assert_not_called()
    assert "点击进入游戏" in trigger.log.call_args.args[0]


def test_game_loading_closes_age_prompt_and_reward_popups_safely():
    clock, ctx, trigger = make_trigger()
    confirm = Hit(True, "确认")
    age = Hit(True, "适龄提示")

    class Region:
        bgr = np.zeros((4, 8, 3), dtype=np.uint8)

        @staticmethod
        def find(_ro):
            return Hit(False)

        @staticmethod
        def find_multi(_ro, limit=40):
            return [age, confirm]

    trigger.on_frame(Region())
    confirm.click.assert_called_once_with()

    clock.value += 2.1
    Region.find_multi = staticmethod(lambda _ro, limit=40: [Hit(True, "空月祝福")])
    trigger.on_frame(Region())
    ctx.input.click_ref.assert_called_once_with(960, 820)


def test_game_loading_stops_capture_work_after_main_ui():
    _, ctx, trigger = make_trigger(main=True)
    loop = SimpleNamespace(get=Mock(return_value=trigger), remove=Mock())
    ctx._trigger_loop = loop
    region = SimpleNamespace(bgr=np.zeros((4, 8, 3), dtype=np.uint8))

    trigger.on_frame(region)

    assert trigger.enabled is False
    loop.remove.assert_called_once_with("GameLoading")


def test_game_loading_timer_maps_bettergi_name():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    ctx = SimpleNamespace(triggers=SimpleNamespace(clear=Mock()), enable_trigger=Mock())
    TaskDispatcher(ctx).add_timer({
        "name": "自动开门",
        "config": {"TimeoutSeconds": 90},
    })

    ctx.triggers.clear.assert_called_once_with()
    ctx.enable_trigger.assert_called_once_with("GameLoading", timeout_s=90.0)
