import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


def _write_group(path: Path, projects: list[dict], **extra) -> Path:
    path.write_text(json.dumps({
        "Name": "每日配置",
        "Config": extra.pop("Config", {}),
        "Projects": projects,
        **extra,
    }, ensure_ascii=False), encoding="utf-8")
    return path


def _ctx():
    return SimpleNamespace(
        triggers=SimpleNamespace(clear=Mock()),
        input=SimpleNamespace(release_all=Mock()),
    )


def test_script_group_loads_original_bettergi_schema_and_skips_disabled(tmp_path: Path):
    from bgi_touch.tasks.script_group import ScriptGroupRunner

    source = _write_group(tmp_path / "daily.json", [
        {
            "Name": "采集脚本",
            "FolderName": "collect",
            "Type": "Javascript",
            "Status": "Enabled",
            "Schedule": "Daily",
            "RunNum": 2,
            "JsScriptSettingsObject": {"count": 3},
        },
        {
            "Name": "disabled.json",
            "FolderName": "routes",
            "Type": "Pathing",
            "Status": "Disabled",
        },
    ])

    runner = ScriptGroupRunner.load(None, [source])
    description = runner.describe()

    assert description["groups"] == ["每日配置"]
    assert description["count"] == 1
    assert description["projects"][0] == {
        "group": "每日配置",
        "index": 0,
        "name": "采集脚本",
        "folderName": "collect",
        "type": "Javascript",
        "runNum": 2,
    }
    assert runner.groups[0].projects[0].settings == {"count": 3}


def test_script_group_progress_resumes_at_failed_project(tmp_path: Path):
    from bgi_touch.tasks.script_group import ScriptGroupRunner
    from bgi_touch.tasks.task_progress import TaskProgressStore

    source = _write_group(tmp_path / "daily.json", [
        {"Name": "A", "Type": "Shell"},
        {"Name": "B", "Type": "Shell"},
        {"Name": "C", "Type": "Shell"},
    ])
    store = TaskProgressStore(tmp_path / "progress", log=lambda _message: None)
    first_calls = []
    first = ScriptGroupRunner.load(
        _ctx(), [source], progress_store=store, continue_on_error=False,
        log=lambda _message: None,
    )

    def fail_b(_group, project):
        first_calls.append(project.name)
        if project.name == "B":
            raise RuntimeError("boom")

    first._execute_project = fail_b
    with pytest.raises(RuntimeError, match="每日配置/B"):
        first.run()

    active = store.load_active()
    assert first_calls == ["A", "B"]
    assert len(active) == 1
    assert active[0].last_success.name == "A"
    assert active[0].current.name == "B"
    assert active[0].current.status == 2

    resumed_calls = []
    resumed = ScriptGroupRunner.load(
        _ctx(), [source], progress_store=store, continue_on_error=False,
        log=lambda _message: None,
    )
    resumed._execute_project = lambda _group, project: resumed_calls.append(project.name)
    result = resumed.run(resume=active[0])

    assert resumed_calls == ["B", "C"]
    assert result["status"] == "completed"
    assert result["skipped"] == 1
    assert result["completed"] == 2
    assert store.load(result["progress"]).end_time is not None


def test_script_group_dispatches_javascript_keymouse_pathing_and_shell(tmp_path: Path):
    from bgi_touch.tasks.script_group import (
        ScriptGroup,
        ScriptGroupProject,
        ScriptGroupRoots,
        ScriptGroupRunner,
    )

    js = tmp_path / "js" / "demo"
    macro = tmp_path / "macro"
    pathing = tmp_path / "pathing" / "folder"
    js.mkdir(parents=True)
    macro.mkdir()
    pathing.mkdir(parents=True)
    (js / "manifest.json").write_text("{}", encoding="utf-8")
    (macro / "macro.json").write_text('{"macroEvents": []}', encoding="utf-8")
    route_file = pathing / "route.json"
    route_file.write_text(json.dumps({
        "info": {"name": "route"},
        "positions": [{"id": 1, "x": 0, "y": 0, "type": "target"}],
    }), encoding="utf-8")
    group = ScriptGroup(
        "测试组",
        [
            ScriptGroupProject("JS", "demo", "Javascript", settings={"n": 2}),
            ScriptGroupProject("macro.json", "", "KeyMouse"),
            ScriptGroupProject("route.json", "folder", "Pathing"),
            ScriptGroupProject("echo ok", "", "Shell"),
        ],
        config={
            "enableShellConfig": True,
            "shellConfig": {"timeoutSeconds": 3},
            "PathingConfig": {
                "Enabled": True,
                "HurryOnAvatar": "夜兰",
                "HurryOnFrameInterval": 80,
            },
        },
    )
    ctx = SimpleNamespace(
        input=SimpleNamespace(release_all=Mock()),
        sleep=Mock(),
        triggers=SimpleNamespace(clear=Mock()),
    )
    runner = ScriptGroupRunner(
        ctx,
        [group],
        roots=ScriptGroupRoots.build(javascript=tmp_path / "js",
                                     key_mouse=macro, pathing=tmp_path / "pathing"),
    )

    with patch("bgi_touch.engine.js_runtime.JsScriptRuntime") as runtime, \
            patch("bgi_touch.macro.keymouse.MacroPlayer") as player, \
            patch("bgi_touch.pathing.executor.PathingExecutor") as executor, \
            patch("bgi_touch.tasks.dispatcher.TaskDispatcher") as dispatcher:
        executor.return_value.run.return_value = True
        dispatcher.return_value.run_shell_task.return_value = {"status": "completed"}
        for project in group.projects:
            runner._execute_project(group, project)

    assert runtime.call_args.args[1] == js
    assert runtime.call_args.kwargs["settings"] == {"n": 2}
    player.return_value.play.assert_called_once_with({"macroEvents": []})
    assert executor.call_args.kwargs["farming_route_info"]["group_name"] == "测试组"
    assert executor.call_args.kwargs["pathing_config"].hurry_on_avatar == "夜兰"
    assert runtime.call_args.kwargs["pathing_config"]["PathingConfig"]["HurryOnAvatar"] == "夜兰"
    shell = dispatcher.return_value.run_shell_task.call_args.args[0]
    assert shell["command"] == "echo ok"
    assert shell["timeoutSeconds"] == 3
    assert shell["disable"] is False


def test_script_group_dispatcher_accepts_multiple_config_files(tmp_path: Path):
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.script_group.ScriptGroupRunner.load") as load:
        load.return_value.run.return_value = {"status": "completed"}
        result = TaskDispatcher(object()).run_script_group_task({
            "configFiles": ["one.json", "two.json"],
            "progressDirectory": str(tmp_path),
            "continueOnError": False,
            "resume": "20260824010203",
        })

    assert result == {"status": "completed"}
    assert load.call_args.args[1] == ["one.json", "two.json"]
    assert load.call_args.kwargs["continue_on_error"] is False
    load.return_value.run.assert_called_once_with(resume="20260824010203")


def test_task_progress_store_avoids_same_second_name_collisions(tmp_path: Path):
    from bgi_touch.tasks.task_progress import TaskProgressStore

    store = TaskProgressStore(tmp_path)
    first = store.create(["A"])
    store.save(first)
    second = store.create(["B"])

    assert second.name != first.name
    assert second.name.startswith(first.name + "-")
    assert store.load(first.name + ".json").script_group_names == ["A"]


def test_script_group_cli_dry_run_does_not_connect_device(tmp_path: Path, capsys):
    from bgi_touch.cli import cmd_group

    source = _write_group(tmp_path / "daily.json", [
        {"Name": "echo ok", "Type": "Shell", "Status": "Enabled"},
    ])
    args = SimpleNamespace(
        files=[str(source)], script_root=None, macro_root=None, pathing_root=None,
        progress_dir=str(tmp_path / "progress"), stop_on_error=False,
        records_dir=str(tmp_path / "records"), dry_run=True, resume=None,
    )
    with patch("bgi_touch.cli._context") as context:
        assert cmd_group(args) == 0
    context.assert_not_called()
    assert json.loads(capsys.readouterr().out)["count"] == 1


def test_execution_records_apply_bettergi_policy_boundary_and_gap(tmp_path: Path):
    from bgi_touch.tasks.execution_records import (
        CompletionSkipRule,
        ExecutionRecord,
        ExecutionRecordStore,
    )

    zone = timezone(timedelta(hours=8))
    started = datetime(2026, 8, 24, 5, 0, tzinfo=zone)
    store = ExecutionRecordStore(tmp_path / "records")
    record = ExecutionRecord.start(
        "每日配置", "路线.json", "精英", "Pathing", now=started,
    )
    store.save(record)
    store.finish(record, True, now=started + timedelta(minutes=10))

    daily = CompletionSkipRule(enabled=True, boundary_hour=4)
    assert store.should_skip(
        "每日配置", "路线.json", "精英", "Pathing", daily,
        now=started + timedelta(hours=2),
    )[0]
    assert not store.should_skip(
        "其他配置", "路线.json", "精英", "Pathing", daily,
        now=started + timedelta(hours=2),
    )[0]
    assert not store.should_skip(
        "每日配置", "路线.json", "精英", "Pathing", daily,
        now=started + timedelta(days=1),
    )[0]

    gap = CompletionSkipRule(
        enabled=True,
        policy="SameNameSkipPolicy",
        boundary_hour=-1,
        last_run_gap_s=3600,
        reference_point="EndTime",
    )
    assert store.should_skip(
        "任意组", "路线.json", "任意目录", "Pathing", gap,
        now=started + timedelta(minutes=40),
    )[0]
    assert not store.should_skip(
        "任意组", "路线.json", "任意目录", "Pathing", gap,
        now=started + timedelta(hours=2),
    )[0]


def test_script_group_schedule_rule_supports_skip_hour_and_cycle():
    from bgi_touch.tasks.execution_records import TaskScheduleRule

    zone = timezone(timedelta(hours=8))
    skip_hour = TaskScheduleRule.from_group_config({
        "PathingConfig": {"Enabled": True, "SkipDuring": "5"},
    })
    assert skip_hour.skip_reason(now=datetime(2026, 8, 24, 5, tzinfo=zone))
    assert skip_hour.skip_reason(now=datetime(2026, 8, 24, 6, tzinfo=zone)) is None

    cycle = TaskScheduleRule.from_group_config({
        "PathingConfig": {
            "Enabled": True,
            "TaskCycleConfig": {
                "Enable": True,
                "BoundaryTime": 4,
                "Cycle": 3,
                "Index": 2,
            },
        },
    })
    # 1970-01-02 is day 1: (1 % 3) + 1 == 2.
    assert cycle.skip_reason(now=datetime(1970, 1, 2, 5, tzinfo=zone)) is None
    assert cycle.skip_reason(now=datetime(1970, 1, 3, 5, tzinfo=zone))


def test_script_group_skips_successful_project_in_same_completion_window(tmp_path: Path):
    from bgi_touch.tasks.execution_records import ExecutionRecordStore
    from bgi_touch.tasks.script_group import ScriptGroupRunner
    from bgi_touch.tasks.task_progress import TaskProgressStore

    source = _write_group(
        tmp_path / "daily.json",
        [{"Name": "A", "Type": "Shell", "FolderName": "folder"}],
        Config={
            "PathingConfig": {
                "TaskCompletionSkipRuleConfig": {
                    "Enable": True,
                    "SkipPolicy": "GroupPhysicalPathSkipPolicy",
                    "BoundaryTime": 4,
                },
            },
        },
    )
    records = ExecutionRecordStore(tmp_path / "records")
    calls = []
    first = ScriptGroupRunner.load(
        _ctx(), [source],
        progress_store=TaskProgressStore(tmp_path / "progress1"),
        execution_store=records,
        log=lambda _message: None,
    )
    first._execute_project = lambda _group, project: calls.append(project.name)
    assert first.run()["completed"] == 1

    second = ScriptGroupRunner.load(
        _ctx(), [source],
        progress_store=TaskProgressStore(tmp_path / "progress2"),
        execution_store=records,
        log=lambda _message: None,
    )
    second._execute_project = lambda _group, project: calls.append(project.name)
    result = second.run()

    assert calls == ["A"]
    assert result["completed"] == 0
    assert result["skipped"] == 1
    raw = json.loads(next((tmp_path / "records").glob("*.json")).read_text())
    assert raw["execution_records"][0]["is_successful"] is True
