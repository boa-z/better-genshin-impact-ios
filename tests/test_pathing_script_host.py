import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


def _context():
    from bgi_touch.vision.coordinate import ScreenTransform

    return SimpleNamespace(
        input=SimpleNamespace(
            key_down=Mock(), key_up=Mock(), key_press=Mock(), click_ref=Mock(),
            move_camera_by=Mock(), attack=Mock(), attack_down=Mock(),
            attack_up=Mock(), button_down=Mock(), button_up=Mock(),
            tap_button=Mock(), drag_ref=Mock(), release_all=Mock(),
        ),
        device=SimpleNamespace(paste_text=Mock(), tap=Mock()),
        transform=ScreenTransform(1920, 1080),
        sleep=lambda _ms: None,
    )


def _route(name: str) -> str:
    return json.dumps({
        "info": {"name": name},
        "positions": [
            {"id": 1, "x": 10, "y": 20, "type": "target", "move_mode": "walk"}
        ],
    }, ensure_ascii=False)


def test_pathing_script_uses_shared_auto_pathing_root(tmp_path: Path):
    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime

    script_root = tmp_path / "script"
    pathing_root = tmp_path / "subscribed"
    script_root.mkdir()
    (pathing_root / "nested").mkdir(parents=True)
    (script_root / "nested").mkdir()
    (script_root / "nested" / "route.json").write_text(
        _route("脚本内路线"), encoding="utf-8"
    )
    (pathing_root / "nested" / "route.json").write_text(
        _route("订阅路线"), encoding="utf-8"
    )
    (pathing_root / "root.json").write_text(_route("根路线"), encoding="utf-8")
    (script_root / "manifest.json").write_text(
        json.dumps({"main": "main.js"}), encoding="utf-8"
    )
    (script_root / "main.js").write_text(
        """
(async function () {
  const entries = pathingScript.ReadPathSync(".");
  const text = pathingScript.ReadTextSync("nested\\\\route.json");
  const escaped = pathingScript.ReadTextSync("../secret.json");
  await pathingScript.RunFileFromUser("nested/route.json");
  return JSON.stringify({
    entries,
    exists: pathingScript.IsExists("nested/route.json"),
    file: pathingScript.IsFile("nested/route.json"),
    folder: pathingScript.IsFolder("nested"),
    missing: pathingScript.IsExists("missing.json"),
    routeName: JSON.parse(text).info.name,
    escaped
  });
})();
""",
        encoding="utf-8",
    )

    with patch("bgi_touch.engine.js_runtime.PathingExecutor") as executor:
        executor.return_value.run.return_value = True
        result = json.loads(JsScriptRuntime(
            _context(), script_root, pathing_root=pathing_root,
            log=lambda _message: None,
        ).run())

    assert result == {
        "entries": ["nested", "root.json"],
        "exists": True,
        "file": True,
        "folder": True,
        "missing": False,
        "routeName": "订阅路线",
        "escaped": "",
    }
    task = executor.return_value.run.call_args.args[0]
    assert task.name == "订阅路线"
    assert Path(task.source_path).is_relative_to(pathing_root)


def test_script_group_passes_its_pathing_root_to_javascript(tmp_path: Path):
    from bgi_touch.tasks.script_group import (
        ScriptGroup,
        ScriptGroupProject,
        ScriptGroupRoots,
        ScriptGroupRunner,
    )

    js = tmp_path / "js" / "demo"
    js.mkdir(parents=True)
    roots = ScriptGroupRoots.build(
        javascript=tmp_path / "js", pathing=tmp_path / "routes"
    )
    runner = ScriptGroupRunner(
        _context(),
        [ScriptGroup("group", [ScriptGroupProject("demo", "demo")])],
        roots=roots,
    )

    with patch("bgi_touch.engine.js_runtime.JsScriptRuntime") as runtime:
        runner._execute_project(runner.groups[0], runner.groups[0].projects[0])

    assert runtime.call_args.kwargs["pathing_root"] == roots.pathing
