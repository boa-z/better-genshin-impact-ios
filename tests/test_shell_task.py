import json
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import patch

from bgi_touch.tasks.shell_task import ShellHostConfig, ShellTask


def _config(tmp_path, *, enabled):
    path = tmp_path / "config" / "shell.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "enabled": enabled,
        "timeoutSeconds": 2,
        "output": True,
        "workingDirectory": "..",
        "maxOutputChars": 1000,
    }), encoding="utf-8")
    return path


def _python(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def test_shell_task_is_disabled_by_default(tmp_path):
    config = _config(tmp_path, enabled=False)
    with patch("bgi_touch.tasks.shell_task.subprocess.Popen") as popen:
        result = ShellTask("echo unsafe", config_path=config).run()
    assert result["status"] == "disabled"
    popen.assert_not_called()


def test_shell_task_captures_cross_platform_output(tmp_path):
    config = _config(tmp_path, enabled=True)
    result = ShellTask(
        _python("print('兼容输出')"), config_path=config, log=lambda _m: None,
    ).run()
    assert result["status"] == "completed"
    assert result["return_code"] == 0
    assert result["output"].strip() == "兼容输出"


def test_shell_task_timeout_terminates_process_group(tmp_path):
    config = _config(tmp_path, enabled=True)
    result = ShellTask(
        _python("import time; time.sleep(5)"),
        config_path=config,
        timeout_s=0.15,
        log=lambda _m: None,
    ).run()
    assert result["status"] == "timeout"
    assert result["timed_out"] is True
    assert result["return_code"] is not None


def test_shell_config_resolves_relative_working_directory(tmp_path):
    path = _config(tmp_path, enabled=True)
    config = ShellHostConfig.load(path)
    assert config.enabled is True
    assert config.working_directory == tmp_path.resolve()
    assert config.timeout_s == 2


def test_shell_dispatcher_maps_bettergi_parameters():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.shell_task.ShellTask") as task:
        task.return_value.run.return_value = {"status": "completed"}
        result = TaskDispatcher(object()).run_shell_task({
            "command": "echo ok",
            "timeoutSeconds": 7,
            "noWindow": False,
            "output": False,
            "disable": True,
            "workingDirectory": "/tmp",
        })
    assert result == {"status": "completed"}
    assert task.call_args.args == ("echo ok",)
    assert task.call_args.kwargs["timeout_s"] == 7
    assert task.call_args.kwargs["no_window"] is False
    assert task.call_args.kwargs["output"] is False
    assert task.call_args.kwargs["disable"] is True
    assert task.call_args.kwargs["working_directory"] == "/tmp"


def test_shell_dispatcher_is_available_to_one_dragon():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    assert "Shell" in TaskDispatcher.IMPLEMENTED


def test_shell_cli_does_not_connect_devicehub():
    from bgi_touch.cli import cmd_task

    args = SimpleNamespace(
        name="Shell", config="{}", config_file=None,
    )
    with patch("bgi_touch.cli._context") as context:
        assert cmd_task(args) == 0
    context.assert_not_called()
