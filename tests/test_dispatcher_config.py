from types import SimpleNamespace
from unittest.mock import Mock, patch


def test_shell_dispatcher_coerces_string_boolean_values():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.shell_task.ShellTask") as task:
        task.return_value.run.return_value = {"status": "completed"}
        result = TaskDispatcher(object()).run_shell_task({
            "command": "echo ok",
            "noWindow": "false",
            "output": "false",
            "disable": "false",
        })

    assert result == {"status": "completed"}
    assert task.call_args.kwargs["no_window"] is False
    assert task.call_args.kwargs["output"] is False
    assert task.call_args.kwargs["disable"] is False


def test_music_player_dispatcher_coerces_string_boolean_values():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.music_player.MusicPlayerTask") as task:
        task.return_value.run.return_value = {"status": "completed"}
        result = TaskDispatcher(object()).run_music_player_task({
            "file": "score.json",
            "useCustomBpm": "false",
            "customBpm": 88,
            "autoSwitchInstrument": "false",
        })

    assert result == {"status": "completed"}
    assert task.call_args.kwargs["custom_bpm"] is None
    assert task.call_args.kwargs["auto_switch_instrument"] is False


def test_auto_fishing_dispatcher_coerces_string_boolean_values():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.auto_fishing.AutoFishingTask") as task:
        task.return_value.run.return_value = True
        result = TaskDispatcher(object()).run_auto_fishing_task({
            "autoThrowRodEnabled": "false",
            "isCoop": "false",
            "quitOnFinish": "false",
        })

    assert result is True
    kwargs = task.call_args.kwargs
    assert kwargs["auto_throw_rod_enabled"] is False
    assert kwargs["coop"] is False
    assert kwargs["quit_on_finish"] is False


def test_timer_dispatcher_coerces_string_boolean_values():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    ctx = SimpleNamespace(
        triggers=SimpleNamespace(clear=Mock()),
        enable_trigger=Mock(),
    )
    dispatcher = TaskDispatcher(ctx)

    dispatcher.add_timer({
        "name": "AutoPick",
        "config": {
            "forceInteraction": "false",
            "blacklistModePickEnabled": "false",
            "whitelistModeDoNotPickEnabled": "false",
        },
    })
    assert ctx.enable_trigger.call_args.kwargs["force_interaction"] is False
    assert ctx.enable_trigger.call_args.kwargs["blacklist_mode_pick_enabled"] is False
    assert ctx.enable_trigger.call_args.kwargs[
        "whitelist_mode_do_not_pick_enabled"
    ] is False

    ctx.enable_trigger.reset_mock()
    dispatcher.add_timer({
        "name": "QuickTeleport",
        "config": {"hotkeyTpEnabled": "false"},
    })
    assert ctx.enable_trigger.call_args.kwargs["hotkey_tp_enabled"] is False
