from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


class Clock:
    value = 100.0

    def __call__(self):
        return self.value


class Hit:
    def __init__(self, exists=False, *, x=0, y=0, width=20, height=20, text=""):
        self._exists = exists
        self.x, self.y = x, y
        self.width, self.height = width, height
        self.text = text
        self.click = Mock()

    def is_exist(self):
        return self._exists


def make_trigger(*, hotkey=False, in_map=True):
    from bgi_touch.triggers.quick_teleport import QuickTeleportTrigger

    clock = Clock()
    ctx = SimpleNamespace(
        input=SimpleNamespace(click_ref=Mock()),
        sleep=Mock(),
        capture_bgr=Mock(side_effect=AssertionError("trigger must not capture")),
    )
    trigger = QuickTeleportTrigger(
        ctx,
        hotkey_tp_enabled=hotkey,
        clock=clock,
        big_map_detector=lambda _ctx, _frame: in_map,
        log=Mock(),
    )
    return clock, ctx, trigger


def test_quick_teleport_clicks_visible_teleport_button_without_capture():
    _, ctx, trigger = make_trigger()
    teleport = Hit(True)

    class Region:
        bgr = np.zeros((4, 8, 3), dtype=np.uint8)

        @staticmethod
        def find(ro):
            return teleport if ro is trigger._teleport else Hit(False)

        @staticmethod
        def find_multi(_ro, limit=8):
            return []

    trigger.on_frame(Region())

    teleport.click.assert_called_once_with()
    ctx.capture_bgr.assert_not_called()
    assert "点击传送" in trigger.log.call_args.args[0]


def test_quick_teleport_selects_top_valid_icon_then_waits_for_next_frame():
    _, ctx, trigger = make_trigger(in_map=False)
    invalid = Hit(True, x=1300, y=180)
    valid = Hit(True, x=1300, y=260, width=24, height=30)
    text = Hit(True, text="  传送锚点 > ")

    class Region:
        bgr = np.zeros((4, 8, 3), dtype=np.uint8)

        @staticmethod
        def find(ro):
            if getattr(ro, "recognition_type", "") == "Ocr":
                return Hit(True, text="") if ro.roi[1] < 200 else text
            return Hit(False)

        @staticmethod
        def find_multi(ro, limit=8):
            if ro is trigger._option_templates[0]:
                return [invalid, valid]
            return []

    trigger.on_frame(Region())

    ctx.input.click_ref.assert_called_once_with(1424, 275.0)
    ctx.sleep.assert_called_once_with(200)
    ctx.capture_bgr.assert_not_called()
    assert "传送锚点" in trigger.log.call_args.args[0]


def test_quick_teleport_plain_map_does_not_click_arbitrary_content():
    _, ctx, trigger = make_trigger(in_map=True)

    class Region:
        bgr = np.zeros((4, 8, 3), dtype=np.uint8)

        @staticmethod
        def find(ro):
            return Hit(ro is trigger._map_close)

        @staticmethod
        def find_multi(_ro, limit=8):
            raise AssertionError("plain map must not scan/click candidate list")

    trigger.on_frame(Region())
    ctx.input.click_ref.assert_not_called()


def test_quick_teleport_manual_mode_requires_activation():
    _, _, trigger = make_trigger(hotkey=True)
    teleport = Hit(True)
    region = SimpleNamespace(
        bgr=np.zeros((4, 8, 3), dtype=np.uint8),
        find=lambda _ro: teleport,
        find_multi=lambda _ro, limit=8: [],
    )

    trigger.on_frame(region)
    teleport.click.assert_not_called()
    trigger.activate()
    trigger.on_frame(region)
    teleport.click.assert_called_once_with()


def test_quick_teleport_timer_maps_bettergi_configuration():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    ctx = SimpleNamespace(triggers=SimpleNamespace(clear=Mock()), enable_trigger=Mock())
    TaskDispatcher(ctx).add_timer({
        "name": "快速传送",
        "config": {
            "TeleportListClickDelay": 120,
            "WaitTeleportPanelDelay": 80,
            "HotkeyTpEnabled": True,
        },
    })

    ctx.triggers.clear.assert_called_once_with()
    ctx.enable_trigger.assert_called_once_with(
        "QuickTeleport",
        teleport_list_click_delay_ms=120,
        wait_teleport_panel_delay_ms=80,
        hotkey_tp_enabled=True,
    )


def test_webui_exposes_quick_teleport_toggle_and_manual_tick():
    page = (
        Path(__file__).parents[1] / "bgi_touch" / "webui" / "static" / "index.html"
    ).read_text(encoding="utf-8")
    assert 'id="trigQuickTp"' in page
    assert "/api/quick-teleport/tick" in page
