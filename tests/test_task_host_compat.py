import json
import threading
import time
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
  accuracy: typeof dispatcher.runGridIconsAccuracyTestTask,
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
        "accuracy": "function",
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
  accuracy: typeof dispatcher.RunGridIconsAccuracyTestTask,
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
        "accuracy": "function",
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


def test_dispatcher_task_returns_concurrent_promise_and_snapshots_parameters(
    tmp_path, monkeypatch,
):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    calls = []

    def fake_auto_fight(self, param, ct=None):
        calls.append((param, ct))
        time.sleep(0.08)
        return {"timeout": param["timeout"], "worker": threading.current_thread().name}

    monkeypatch.setattr(TaskDispatcher, "run_auto_fight_task", fake_auto_fight)
    (tmp_path / "main.js").write_text(
        """
const param = new AutoFightParam('snapshot.txt');
param.Timeout = 321;
const task = dispatcher.runAutoFightTask(param);
const isPromise = task instanceof Promise;
param.Timeout = 999;
let ticks = 0;
const timer = setInterval(() => ticks++, 5);
const result = await task;
clearInterval(timer);
return JSON.stringify({isPromise, result, ticks});
""",
        encoding="utf-8",
    )

    result = json.loads(JsScriptRuntime(_context(), tmp_path).run())

    assert result["isPromise"] is True
    assert result["result"]["timeout"] == 321
    assert result["ticks"] >= 3
    assert len(calls) == 1
    assert calls[0][0] == {
        "combatStrategyPath": "snapshot.txt",
        "timeout": 321,
        "fightFinishDetectEnabled": False,
        "finishDetectConfig": {
            "battleEndProgressBarColor": "",
            "battleEndProgressBarColorTolerance": "",
            "fastCheckEnabled": False,
            "fastCheckParams": "",
            "checkAfterSwitchAvatar": False,
            "checkEndDelay": "0.4;钟离,1.4;",
            "beforeDetectDelay": "0.4",
            "rotateFindEnemyEnabled": False,
            "skipFightEndCheckWhenEnemyVisible": False,
            "blockCheckBeforeBattleSeconds": 0,
            "paimonEndCheckEnabled": True,
            "paimonEndCheckDelay": 0.075,
        },
        "pickDropsAfterFightEnabled": False,
        "pickDropsAfterFightSeconds": 15,
        "kazuhaPickupEnabled": True,
        "kazuhaPartyName": "",
        "actionSchedulerByCd": "",
        "onlyPickEliteDropsMode": "",
        "battleThresholdForLoot": -1,
        "guardianAvatar": "",
        "guardianCombatSkip": False,
        "guardianAvatarHold": False,
        "checkBeforeBurst": False,
        "isFirstCheck": True,
        "rotaryFactor": 10,
        "burstEnabled": False,
        "qinDoublePickUp": False,
    }


def test_dispatcher_task_rejection_is_catchable_from_js(tmp_path, monkeypatch):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    def fake_auto_fight(self, _param, _ct=None):
        raise ValueError("模拟任务失败")

    monkeypatch.setattr(TaskDispatcher, "run_auto_fight_task", fake_auto_fight)
    (tmp_path / "main.js").write_text(
        """
const message = await dispatcher.runAutoFightTask({}).catch(error => error.message);
return message;
""",
        encoding="utf-8",
    )

    assert JsScriptRuntime(_context(), tmp_path).run() == "ValueError: 模拟任务失败"


def test_dispatcher_task_uses_external_cancellation_token(tmp_path, monkeypatch):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    started = threading.Event()

    def fake_auto_fight(self, _param, ct=None):
        started.set()
        while not ct.isCancellationRequested:
            time.sleep(0.002)
        raise RuntimeError("任务已取消")

    monkeypatch.setattr(TaskDispatcher, "run_auto_fight_task", fake_auto_fight)
    (tmp_path / "main.js").write_text(
        """
const cts = new CancellationTokenSource();
const task = dispatcher.runAutoFightTask({}, cts.Token)
  .catch(error => error.message);
setTimeout(() => cts.cancel(), 25);
return await task;
""",
        encoding="utf-8",
    )

    result = JsScriptRuntime(_context(), tmp_path).run()

    assert started.is_set()
    assert result == "RuntimeError: 任务已取消"


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
