import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


def _context():
    from bgi_touch.vision.coordinate import ScreenTransform

    return SimpleNamespace(
        transform=ScreenTransform(1920, 1080),
        device=SimpleNamespace(tap=Mock(), paste_text=Mock()),
        input=SimpleNamespace(
            key_down=Mock(), key_up=Mock(), key_press=Mock(),
            click_ref=Mock(), move_camera_by=Mock(), attack=Mock(),
            attack_down=Mock(), attack_up=Mock(), button_down=Mock(),
            button_up=Mock(), release_all=Mock(),
        ),
        sleep=lambda _ms: None,
        capture_bgr=lambda: np.zeros((1080, 1920, 3), dtype=np.uint8),
    )


def test_task_constructors_keep_bettergi_property_aliases(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    (tmp_path / "main.js").write_text(
        """
const timer = new RealtimeTimer('AutoPick', { forceInteraction: true });
timer.Name = 'AutoSkip';
timer.Interval = 75;
const task = new SoloTask('AutoFight', { Timeout: 10 });
task.Name = 'AutoDomain';
const fight = new AutoFightParam('combat.txt');
AutoFightParam.SwimmingEnabled = true;
const finish = new AutoFightParam.FightFinishDetectConfig();
finish.FastCheckEnabled = true;
return JSON.stringify({
  timer: [timer.name, timer.Name, timer.interval, timer.Interval, timer.Config.forceInteraction],
  task: [task.name, task.Name, task.Config.Timeout],
  swimming: AutoFightParam.SwimmingEnabled,
  finish: finish.fastCheckEnabled
});
""",
        encoding="utf-8",
    )

    result = json.loads(JsScriptRuntime(_context(), tmp_path).run())
    assert result == {
        "timer": ["AutoSkip", "AutoSkip", 75, 75, True],
        "task": ["AutoDomain", "AutoDomain", 10],
        "swimming": True,
        "finish": True,
    }
