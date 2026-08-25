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


def test_dispatcher_exposes_all_migrated_specialized_task_entrypoints(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    (tmp_path / "main.js").write_text(
        """
return JSON.stringify({
  wood: typeof dispatcher.runAutoWoodTask,
  grid: typeof dispatcher.runGetGridIconsTask,
  compare: typeof dispatcher.runInventoryCountComparisonTask,
  character: typeof dispatcher.runCharacterDevelopmentTask,
  daily: typeof dispatcher.runCheckRewardsTask,
  pot: typeof dispatcher.runSereniteaPotRewardsTask,
  group: typeof dispatcher.runScriptGroupTask,
  music: typeof dispatcher.runMusicPlayerTask,
  shell: typeof dispatcher.runShellTask
});
""",
        encoding="utf-8",
    )

    result = json.loads(JsScriptRuntime(_context(), tmp_path).run())

    assert result == {
        "wood": "function",
        "grid": "function",
        "compare": "function",
        "character": "function",
    "daily": "function",
    "pot": "function",
    "group": "function",
        "music": "function",
        "shell": "function",
    }


def test_dispatcher_supports_bettergi_pascal_case_entrypoints(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    (tmp_path / "main.js").write_text(
        """
return JSON.stringify({
  runTask: typeof dispatcher.runTask,
  RunTask: typeof dispatcher.RunTask,
  domain: typeof dispatcher.RunAutoDomainTask,
  fight: typeof dispatcher.RunAutoFightTask,
  leyLine: typeof dispatcher.RunAutoLeyLineOutcropTask,
  expedition: typeof dispatcher.RunOneKeyExpeditionTask,
  pot: typeof dispatcher.RunGoToSereniteaPotTask,
  daily: typeof dispatcher.RunCheckRewardsTask,
  shell: typeof dispatcher.RunShellTask,
  combat: typeof dispatcher.RunCombatScript,
  addTimer: typeof dispatcher.AddTimer,
  addTrigger: typeof dispatcher.AddTrigger,
  clear: typeof dispatcher.ClearAllTriggers,
  tokenSource: typeof dispatcher.GetLinkedCancellationTokenSource,
  token: typeof dispatcher.GetLinkedCancellationToken
});
""",
        encoding="utf-8",
    )

    result = json.loads(JsScriptRuntime(_context(), tmp_path).run())

    assert result == {
        "runTask": "function",
        "RunTask": "function",
        "domain": "function",
        "fight": "function",
        "leyLine": "function",
    "expedition": "function",
    "pot": "function",
        "daily": "function",
        "shell": "function",
        "combat": "function",
        "addTimer": "function",
        "addTrigger": "function",
        "clear": "function",
        "tokenSource": "function",
        "token": "function",
    }


def test_dispatcher_linked_cancellation_tokens_follow_runtime_stop(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    (tmp_path / "main.js").write_text(
        """
const token = dispatcher.getLinkedCancellationToken();
const source = dispatcher.GetLinkedCancellationTokenSource();
return JSON.stringify({
  token: token.isCancellationRequested,
  tokenPascal: token.IsCancellationRequested,
  source: source.isCancellationRequested,
  sourcePascal: source.IsCancellationRequested,
  canCancel: token.canBeCanceled
});
""",
        encoding="utf-8",
    )

    runtime = JsScriptRuntime(_context(), tmp_path)
    runtime.cancelled = True

    result = json.loads(runtime.run())

    assert result == {
        "token": True,
        "tokenPascal": True,
        "source": True,
        "sourcePascal": True,
        "canCancel": True,
    }


def test_js_add_trigger_forwards_realtime_timer_config(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    (tmp_path / "main.js").write_text(
        """
dispatcher.AddTrigger(new RealtimeTimer('AutoPick', {
  forceInteraction: true,
  pickKey: 'G',
  mode: 'Blacklist',
  blackList: ['进入']
}));
return 'ok';
""",
        encoding="utf-8",
    )
    ctx = _context()
    ctx.triggers = SimpleNamespace(clear=Mock())
    ctx.enable_trigger = Mock()

    assert JsScriptRuntime(ctx, tmp_path).run() == "ok"

    ctx.triggers.clear.assert_not_called()
    assert ctx.enable_trigger.call_args == (("AutoPick",), {
        "force_interaction": True,
        "pick_key": "G",
        "mode": "Blacklist",
        "text_list": None,
        "whitelist": None,
        "blacklist": ["进入"],
        "fuzzy_blacklist": None,
        "whitelist_exclusions": None,
        "blacklist_mode_pick_enabled": False,
        "whitelist_mode_do_not_pick_enabled": True,
    })


def test_dispatcher_exposes_common_job_entrypoints(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    (tmp_path / "main.js").write_text(
        """
return JSON.stringify({
  walk: typeof dispatcher.runWalkToFTask,
  scan: typeof dispatcher.runScanPickTask,
  lower: typeof dispatcher.runLowerHeadThenWalkToTask,
  walkPascal: typeof dispatcher.RunWalkToFTask,
  scanPascal: typeof dispatcher.RunScanPickTask,
  lowerPascal: typeof dispatcher.RunLowerHeadThenWalkToTask
});
""",
        encoding="utf-8",
    )

    result = json.loads(JsScriptRuntime(_context(), tmp_path).run())
    assert result == {
        "walk": "function",
        "scan": "function",
        "lower": "function",
        "walkPascal": "function",
        "scanPascal": "function",
        "lowerPascal": "function",
    }


def test_dispatcher_exposes_genshin_common_job_entrypoints(tmp_path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    (tmp_path / "main.js").write_text(
        """
return JSON.stringify({
  welkin: typeof dispatcher.runBlessingOfTheWelkinMoonTask,
  battlePass: typeof dispatcher.runClaimBattlePassRewardsTask,
  mail: typeof dispatcher.runClaimMailRewardsTask,
  crafting: typeof dispatcher.runCraftMaterialTask,
  time: typeof dispatcher.runSetTimeTask,
  relogin: typeof dispatcher.runReloginTask,
  linnea: typeof dispatcher.runLinneaMiningTask,
  welkinPascal: typeof dispatcher.RunBlessingOfTheWelkinMoonTask,
  timePascal: typeof dispatcher.RunSetTimeTask,
  linneaPascal: typeof dispatcher.RunLinneaMiningTask
});
""",
        encoding="utf-8",
    )

    result = json.loads(JsScriptRuntime(_context(), tmp_path).run())
    assert result == {
        "welkin": "function",
        "battlePass": "function",
        "mail": "function",
        "crafting": "function",
        "time": "function",
        "relogin": "function",
        "linnea": "function",
        "welkinPascal": "function",
        "timePascal": "function",
        "linneaPascal": "function",
    }
