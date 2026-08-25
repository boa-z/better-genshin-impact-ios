import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest


class _Input:
    def __init__(self):
        self._active_slot = 1
        self.switch_party_slot = Mock(side_effect=self._switch)
        self.key_press = Mock()
        self.key_down = Mock()
        self.key_up = Mock()
        self.attack = Mock()
        self.charged_attack = Mock()
        self.move_camera_by = Mock()
        self.attack_down = Mock()
        self.attack_up = Mock()
        self.button_down = Mock()
        self.button_up = Mock()
        self.release_all = Mock()

    def _switch(self, slot):
        self._active_slot = int(slot)


def test_js_combat_scenes_and_avatar_drive_touch_input(tmp_path):
    pytest.importorskip("pythonmonkey")

    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    (tmp_path / "main.js").write_text(
        """
const scenes = new CombatScenes();
scenes.initializeTeam(captureGameRegion());
const avatars = scenes.getAvatars();
const second = scenes.selectAvatar('夜兰');
second.manualSkillCd = 8;
second.switch();
second.useSkill(true);
const cd = second.getSkillCdSeconds();
second.attack(200);
second.charge(450);
second.dash(300);
second.jump();
second.walk('w', 250);
second.moveCamera(12, -8);
second.mouseDown('right');
second.mouseUp('right');
scenes.updateActionSchedulerByCd('钟离,12;夜兰,9;枫原万叶');
const parsed = Avatar.parseActionSchedulerByCd('夜兰', '夜兰,6;钟离,12');
const current = scenes.currentAvatar();
scenes.afterTask();
const taskValue = await new Task(resolve => resolve(42));
const rect = host.newObj(OpenCvSharp.Rect, 1, 2, 3, 4);
const hostArray = host.newArr(Number, 3);
return JSON.stringify({
  count: avatars.Count,
  names: Array.from(avatars, x => x.name),
  current, active: scenes.lastActiveAvatarIndex,
  cdPositive: cd > 7.5 && cd <= 8,
  configuredCd: second.manualSkillCd,
  parsed, taskValue, rect: [rect.X, rect.Y, rect.Width, rect.Height],
  hostArrayLength: hostArray.length
});
""",
        encoding="utf-8",
    )
    input_simulator = _Input()
    ctx = SimpleNamespace(
        input=input_simulator,
        device=SimpleNamespace(paste_text=Mock(), tap=Mock()),
        transform=ScreenTransform(1920, 1080),
        capture_bgr=lambda: np.zeros((1080, 1920, 3), dtype=np.uint8),
        sleep=Mock(),
    )

    result = json.loads(JsScriptRuntime(
        ctx,
        tmp_path,
        party_slots={"钟离": 1, "夜兰": 2, "枫原万叶": 3, "纳西妲": 4},
    ).run())

    assert result == {
        "count": 4,
        "names": ["钟离", "夜兰", "枫原万叶", "纳西妲"],
        "current": "夜兰",
        "active": 2,
        "cdPositive": True,
        "configuredCd": 9,
        "parsed": 6,
        "taskValue": 42,
        "rect": [1, 2, 3, 4],
        "hostArrayLength": 3,
    }
    input_simulator.switch_party_slot.assert_called_once_with(2)
    assert input_simulator.key_press.call_args_list == [
        call("E", hold_ms=900),
        call("LSHIFT", hold_ms=300),
        call("SPACE"),
    ]
    assert input_simulator.attack.call_count == 2
    input_simulator.charged_attack.assert_called_once_with(450)
    input_simulator.key_down.assert_called_once_with("W")
    input_simulator.key_up.assert_called_once_with("W")
    input_simulator.move_camera_by.assert_called_once_with(12.0, -8.0)
    input_simulator.button_down.assert_called_once_with("sprint")
    input_simulator.button_up.assert_called_once_with("sprint")
    input_simulator.release_all.assert_called_once_with()


def test_avatar_ready_waits_for_mobile_hud_instead_of_skill_cooldown(monkeypatch):
    import bgi_touch.engine.combat_host as module

    ctx = SimpleNamespace()
    scenes = SimpleNamespace(ctx=ctx)
    avatar = module.Avatar(scenes, "钟离", 1)
    wait_for_hud = Mock(return_value=True)
    monkeypatch.setattr(module, "wait_for_party_hud", wait_for_hud)

    avatar.ready()

    wait_for_hud.assert_called_once()
    assert wait_for_hud.call_args.args == (ctx,)
    assert callable(wait_for_hud.call_args.kwargs["cancelled"])


def test_combat_ready_uses_hud_callback_without_polling_skill_cd():
    from bgi_touch.combat.dsl import CombatCommand, CombatExecutor

    hud_ready = Mock(side_effect=[False, True])
    skill_ready = Mock(side_effect=AssertionError("ready must not mean skill CD"))
    executor = CombatExecutor(
        SimpleNamespace(),
        sleep=lambda _milliseconds: None,
        hud_ready=hud_ready,
        skill_ready=skill_ready,
    )

    executor.exec(CombatCommand("ready"))

    assert hud_ready.call_count == 2
    skill_ready.assert_not_called()


def test_party_hud_readiness_prefers_fresh_shared_frame(monkeypatch):
    import bgi_touch.combat.hud as module

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ctx = SimpleNamespace(
        cached_frame=Mock(return_value=(frame, 0.05)),
        capture_bgr=Mock(side_effect=AssertionError("must use shared frame")),
    )

    assert module._cached_or_captured_frame(ctx) is frame
    ctx.capture_bgr.assert_not_called()


def test_party_hud_readiness_accepts_recognized_mobile_name(monkeypatch):
    import bgi_touch.combat.hud as module

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    class Region:
        def find_multi(self, _recognition, *, limit):
            assert limit == 8
            return [SimpleNamespace(text="夜兰")]

    monkeypatch.setattr(module, "ImageRegion", lambda _ctx, _frame: Region())
    ctx = SimpleNamespace()

    assert module.is_party_hud_ready(ctx, frame)
