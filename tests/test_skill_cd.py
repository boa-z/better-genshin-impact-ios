from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


class Clock:
    def __init__(self, value: float = 100.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


class EventInput:
    def __init__(self):
        self._active_slot = 1
        self.listener = None
        self.unsubscribed = False

    def subscribe(self, listener):
        self.listener = listener

        def unsubscribe():
            self.unsubscribed = True
            self.listener = None

        return unsubscribe

    def emit(self, clock: Clock, event_type: str, **values):
        assert self.listener is not None
        if event_type == "party_switch":
            self._active_slot = values["to_slot"]
        self.listener({"type": event_type, "timestamp": clock(), **values})


class EmptyOcr:
    @staticmethod
    def recognize(_frame):
        return []


def make_trigger(clock, *, team=None, ocr=None, in_game=True, **kwargs):
    from bgi_touch.triggers.skill_cd import SkillCdTrigger

    input_simulator = EventInput()
    ctx = SimpleNamespace(
        input=input_simulator,
        layout=SimpleNamespace(buttons={"skill": (0.72, 0.894)}),
        _last_frame_at=clock(),
    )
    party = team or {"钟离": 1, "那维莱特": 2, "纳西妲": 3, "芙宁娜": 4}
    trigger = SkillCdTrigger(
        ctx,
        party_slots=party,
        clock=clock,
        ocr=ocr or EmptyOcr(),
        main_ui_detector=lambda _ctx, _frame: in_game,
        log=lambda _message: None,
        **kwargs,
    )
    return ctx, input_simulator, trigger


def gameplay_frame(trigger, ctx, clock):
    ctx._last_frame_at = clock()
    trigger.on_frame(SimpleNamespace(bgr=np.zeros((1080, 1920, 3), dtype=np.uint8)))


def test_skill_cd_uses_custom_fallback_and_monotonic_deadline():
    clock = Clock()
    ctx, input_simulator, trigger = make_trigger(
        clock,
        trigger_on_skill_use=True,
        custom_cd_list=[{"RoleName": "钟离", "CdValueText": "8.4"}],
    )
    gameplay_frame(trigger, ctx, clock)

    input_simulator.emit(clock, "key_press", key="E")
    assert trigger.state.snapshot()["team"][0]["remaining"] == pytest.approx(8.4)

    clock.value += 2.25
    snapshot = trigger.state.snapshot()
    assert snapshot["team"][0]["remaining"] == pytest.approx(6.2)
    assert snapshot["team"][0]["ready"] is False


def test_skill_cd_switch_uses_cached_frame_ocr_and_compensates_frame_age():
    clock = Clock()
    ocr = SimpleNamespace(recognize=Mock(return_value=[SimpleNamespace(text="9.8")]))
    ctx, input_simulator, trigger = make_trigger(clock, ocr=ocr)
    gameplay_frame(trigger, ctx, clock)
    clock.value += 1.0

    input_simulator.emit(
        clock, "party_switch", from_slot=1, to_slot=2,
    )

    snapshot = trigger.state.snapshot()
    assert snapshot["activeSlot"] == 2
    assert snapshot["team"][0]["remaining"] == pytest.approx(8.8)
    assert ocr.recognize.call_count == 1


def test_skill_cd_switch_only_falls_back_after_recent_e_edge():
    clock = Clock()
    ctx, input_simulator, trigger = make_trigger(
        clock,
        custom_cd_list=[{"roleName": "钟离", "cdValueText": "12"}],
    )
    gameplay_frame(trigger, ctx, clock)
    input_simulator.emit(clock, "party_switch", from_slot=1, to_slot=2)
    assert trigger.state.snapshot()["team"][0]["remaining"] == 0

    input_simulator.emit(clock, "party_switch", from_slot=2, to_slot=1)
    assert trigger.state.snapshot()["team"][1]["remaining"] == 0

    input_simulator.emit(clock, "key_down", key="E")
    clock.value += 0.2
    input_simulator.emit(clock, "party_switch", from_slot=1, to_slot=2)
    assert trigger.state.snapshot()["team"][0]["remaining"] == pytest.approx(12.0)


def test_skill_cd_blank_custom_rule_uses_avatar_default():
    from bgi_touch.triggers.skill_cd import _avatar_cooldowns

    clock = Clock()
    ctx, input_simulator, trigger = make_trigger(
        clock,
        trigger_on_skill_use=True,
        custom_cd_list=[{"roleName": "钟离", "cdValueText": ""}],
    )
    gameplay_frame(trigger, ctx, clock)
    input_simulator.emit(clock, "key_press", key="E")

    assert trigger.state.snapshot()["team"][0]["remaining"] == pytest.approx(
        _avatar_cooldowns()["钟离"]
    )


def test_skill_cd_hides_incomplete_team_and_debounces_scene_leave():
    clock = Clock()
    ctx, _, trigger = make_trigger(
        clock,
        team={"钟离": 1, "那维莱特": 2, "纳西妲": 3},
    )
    gameplay_frame(trigger, ctx, clock)
    assert trigger.state.snapshot()["visible"] is False

    trigger._team[3] = "芙宁娜"
    gameplay_frame(trigger, ctx, clock)
    assert trigger.state.snapshot()["visible"] is True

    trigger._main_ui_detector = lambda _ctx, _frame: False
    clock.value += 0.2
    gameplay_frame(trigger, ctx, clock)
    assert trigger.state.snapshot()["visible"] is True
    clock.value += 0.81
    gameplay_frame(trigger, ctx, clock)
    assert trigger.state.snapshot()["visible"] is False


def test_skill_cd_close_unsubscribes_and_deactivates_state():
    clock = Clock()
    _, input_simulator, trigger = make_trigger(clock)
    trigger.close()

    assert input_simulator.unsubscribed is True
    assert trigger.state.snapshot()["active"] is False


def test_input_simulator_publishes_completed_skill_and_party_edges():
    from bgi_touch.input.layout import ControlLayout
    from bgi_touch.input.simulator import InputSimulator
    from bgi_touch.vision.coordinate import ScreenTransform

    layout = ControlLayout.load(
        Path(__file__).parents[1] / "config" / "controls" / "genshin-default.json"
    )
    device = SimpleNamespace(tap=Mock(), multi_touch=Mock())
    simulator = InputSimulator(device, layout, ScreenTransform(1920, 1080))
    events = []
    unsubscribe = simulator.subscribe(events.append)

    simulator.key_press("E")
    simulator.switch_party_slot(2)
    unsubscribe()
    simulator.key_press("E")

    assert [(item["type"], item.get("key")) for item in events] == [
        ("key_press", "E"),
        ("party_switch", None),
    ]
    assert events[1]["from_slot"] == 1
    assert events[1]["to_slot"] == 2


def test_skill_cd_timer_maps_bettergi_configuration():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    ctx = SimpleNamespace(triggers=SimpleNamespace(clear=Mock()), enable_trigger=Mock())
    TaskDispatcher(ctx, party_slots={"钟离": 1}).add_timer({
        "name": "技能冷却",
        "config": {
            "CustomCdList": [{"RoleName": "钟离", "CdValueText": "9"}],
            "TriggerOnSkillUse": True,
            "HideWhenZero": True,
            "PX": 1500,
            "PY": 220,
            "Gap": 88,
            "Scale": 1.2,
        },
    })

    ctx.triggers.clear.assert_called_once_with()
    kwargs = ctx.enable_trigger.call_args.kwargs
    assert ctx.enable_trigger.call_args.args == ("SkillCd",)
    assert kwargs["party_slots"] == {"钟离": 1}
    assert kwargs["custom_cd_list"][0]["RoleName"] == "钟离"
    assert kwargs["trigger_on_skill_use"] is True
    assert kwargs["hide_when_zero"] is True
    assert (kwargs["p_x"], kwargs["p_y"], kwargs["gap"], kwargs["scale"]) == (
        1500.0, 220.0, 88.0, 1.2,
    )


def test_skill_cd_api_polling_does_not_connect_or_capture(monkeypatch):
    from bgi_touch.webui import server

    monkeypatch.setattr(server, "_ctx", None)
    connect = Mock(side_effect=AssertionError("read-only SkillCd polling must not connect"))
    monkeypatch.setattr(server, "get_ctx", connect)

    state = server.api_skill_cd()

    assert state["active"] is False
    assert state["scene"] == "disabled"
    connect.assert_not_called()


def test_webui_contains_cached_skill_cd_overlay():
    page = (
        Path(__file__).parents[1] / "bgi_touch" / "webui" / "static" / "index.html"
    ).read_text(encoding="utf-8")

    assert 'id="skillCdLayer"' in page
    assert "fetch('/api/skill-cd'" in page
    assert "不触碰 DeviceHub 截图器" in page
    assert "refXToPreview" in page
