import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


def test_js_strategy_file_uses_restricted_configured_root(tmp_path):
    pytest.importorskip("pythonmonkey")

    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    strategy_root = tmp_path / "strategies"
    (strategy_root / "群友分享").mkdir(parents=True)
    (strategy_root / "群友分享" / "策略.txt").write_text("attack(1)", encoding="utf-8")
    script_root = tmp_path / "script"
    script_root.mkdir()
    (script_root / "main.js").write_text(
        """
let escaped = false;
try { strategyFile.ReadPathSync("../"); } catch (_) { escaped = true; }
const roots = strategyFile.ReadPathSync(".");
const children = strategyFile.readPathSync("群友分享");
return JSON.stringify({
  roots,
  children,
  folder: strategyFile.IsFolder("群友分享"),
  file: strategyFile.isFile("群友分享/策略.txt"),
  exists: strategyFile.IsExists("群友分享/策略.txt"),
  escaped
});
""",
        encoding="utf-8",
    )
    ctx = SimpleNamespace(
        input=SimpleNamespace(),
        device=SimpleNamespace(paste_text=Mock()),
        transform=ScreenTransform(1920, 1080),
        sleep=lambda _ms: None,
    )

    result = json.loads(JsScriptRuntime(
        ctx, script_root, strategy_roots=[strategy_root]
    ).run())

    assert result == {
        "roots": ["群友分享"],
        "children": ["群友分享/策略.txt"],
        "folder": True,
        "file": True,
        "exists": True,
        "escaped": True,
    }


def test_dispatcher_resolves_script_and_shared_combat_strategies(tmp_path):
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    script_root = tmp_path / "script"
    shared_root = tmp_path / "combat"
    script_root.mkdir()
    shared_root.mkdir()
    local = script_root / "local.txt"
    shared = shared_root / "shared.txt"
    local.write_text("attack(1)", encoding="utf-8")
    shared.write_text("attack(2)", encoding="utf-8")
    dispatcher = TaskDispatcher(
        object(), strategy_roots=[script_root, shared_root],
        restrict_strategy_roots=True,
    )

    with patch("bgi_touch.tasks.auto_fight.AutoFightTask") as task:
        task.return_value.run.return_value = True
        assert dispatcher.run_auto_fight_task({"combatStrategyPath": "local.txt"})
        assert task.call_args.kwargs["combat_strategy_path"] == str(local.resolve())

        assert dispatcher.run_auto_fight_task({"combatStrategyPath": "shared"})
        assert task.call_args.kwargs["combat_strategy_path"] == str(shared.resolve())

        outside = tmp_path / "outside.txt"
        outside.write_text("attack(9)", encoding="utf-8")
        assert dispatcher.run_auto_fight_task({"combatStrategyPath": "../outside.txt"})
        assert task.call_args.kwargs["combat_strategy_path"] is None
