from datetime import datetime, timezone

import pytest


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        (True, False, True),
        (False, True, False),
        (1, False, True),
        (0, True, False),
        (" true ", False, True),
        ("FALSE", True, False),
        ("是", False, True),
        ("否", True, False),
        ("unknown", True, True),
        (None, True, True),
    ],
)
def test_as_bool_decodes_bettergi_variants(value, default, expected):
    from bgi_touch.config_values import as_bool

    assert as_bool(value, default) is expected


def test_pathing_config_string_booleans_are_not_truthy_strings():
    from bgi_touch.pathing.model import PathingTask

    task = PathingTask.parse({
        "config": {"realtimeTriggers": {"AutoPick": "false", "AutoSkip": "1"}},
        "positions": [{
            "id": 1,
            "x": 0,
            "y": 0,
            "pointExtParams": {"enableMonsterLootSplit": "0"},
        }],
    })

    assert task.realtime_triggers == {"AutoPick": False, "AutoSkip": True}
    assert task.positions[0].enable_monster_loot_split is False


def test_script_progress_and_execution_records_decode_string_booleans():
    from bgi_touch.tasks.execution_records import ExecutionRecord
    from bgi_touch.tasks.script_group import ScriptGroupProject, ScriptGroupRunner
    from bgi_touch.tasks.task_progress import ProjectProgress, TaskProgress

    project = ScriptGroupProject.from_mapping({
        "Name": "通知关闭",
        "AllowJsNotification": "false",
    })
    progress = TaskProgress.from_mapping({
        "scriptGroupNames": [],
        "loop": "true",
    })
    project_progress = ProjectProgress.from_mapping({"taskEnd": "true"})
    record = ExecutionRecord.from_mapping({
        "guid": "test",
        "startTime": datetime.now(timezone.utc).isoformat(),
        "isSuccessful": "false",
    })
    runner = ScriptGroupRunner(None, [], continue_on_error="false")

    assert project.allow_js_notification is False
    assert progress.loop is True
    assert project_progress is not None and project_progress.task_end is True
    assert record is not None and record.successful is False
    assert runner.continue_on_error is False
