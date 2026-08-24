import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, call


class _Input:
    def __init__(self, slot=1):
        self._active_slot = slot
        self.key_up = Mock()
        self.attack_up = Mock()
        self.button_up = Mock()


class _Executor:
    def __init__(self, callback=None):
        self.actions = []
        self.callback = callback

    def exec(self, command):
        self.actions.append(command.action)
        if self.callback:
            self.callback(command)


def _write_macro(path, script, *, priority=0, second=""):
    path.write_text(json.dumps([{
        "name": "可莉",
        "scriptContent1": script,
        "scriptContent2": second,
        "macroPriority": priority,
    }], ensure_ascii=False), encoding="utf-8")


def _task(tmp_path, script, **kwargs):
    from bgi_touch.combat.one_key_fight import OneKeyFightTask

    path = tmp_path / "avatar_macro.json"
    _write_macro(path, script)
    ctx = SimpleNamespace(input=_Input(), sleep=lambda _ms: None)
    task = OneKeyFightTask(
        ctx, party_slots={"可莉": 1}, macro_path=path,
        default_macro_path=path, log=lambda _message: None, **kwargs,
    )
    return task, ctx, path


def test_parse_one_key_macro_round_clauses_and_ranges():
    from bgi_touch.combat.one_key_fight import parse_one_key_script

    commands = parse_one_key_script(
        "round(1,3-5),keydown(w),wait(0.08),keyup(w)|attack(0.2)"
    )
    assert [command.action for command in commands] == [
        "keydown", "wait", "keyup", "attack",
    ]
    assert commands[0].activating_rounds == (1, 3, 4, 5)
    assert commands[2].activating_rounds == (1, 3, 4, 5)
    assert commands[3].activating_rounds == ()


def test_avatar_macro_priority_and_case_insensitive_fields(tmp_path):
    from bgi_touch.combat.one_key_fight import load_avatar_macros

    path = tmp_path / "macro.json"
    path.write_text(json.dumps([{
        "Name": "刻晴", "ScriptContent1": "attack(0.1)",
        "SCRIPT_CONTENT2": "e", "MacroPriority": 2,
    }], ensure_ascii=False), encoding="utf-8")
    loaded = load_avatar_macros(path, global_priority=1)
    assert [command.action for command in loaded["刻晴"]] == ["e"]


def test_hold_on_stops_before_following_command_and_releases_key(tmp_path):
    from bgi_touch.combat.one_key_fight import HOLD_ON_MODE

    task, ctx, _ = _task(
        tmp_path, "keydown(w),wait(1),keyup(w)", mode=HOLD_ON_MODE,
    )
    wait_entered = threading.Event()
    unblock = threading.Event()

    def execute(command):
        if command.action == "wait":
            wait_entered.set()
            unblock.wait(1)

    executor = _Executor(execute)
    task.executor = executor
    assert task.key_down()
    assert wait_entered.wait(1)
    assert task.key_up()
    ctx.input.key_up.assert_called_once_with("w")
    unblock.set()
    assert task.wait(1)
    assert executor.actions == ["keydown", "wait"]


def test_hold_finish_completes_current_round_after_key_up(tmp_path):
    from bgi_touch.combat.one_key_fight import HOLD_FINISH_MODE

    task, _, _ = _task(
        tmp_path, "keydown(w),wait(1),keyup(w)", mode=HOLD_FINISH_MODE,
    )
    wait_entered = threading.Event()
    unblock = threading.Event()

    def execute(command):
        if command.action == "wait":
            wait_entered.set()
            unblock.wait(1)

    executor = _Executor(execute)
    task.executor = executor
    task.key_down()
    assert wait_entered.wait(1)
    task.key_up()
    unblock.set()
    assert task.wait(1)
    assert executor.actions == ["keydown", "wait", "keyup"]


def test_tick_second_press_cancels_after_current_round(tmp_path):
    from bgi_touch.combat.one_key_fight import TICK_MODE

    task, _, _ = _task(tmp_path, "attack,wait(1)", mode=TICK_MODE)
    wait_entered = threading.Event()
    unblock = threading.Event()

    def execute(command):
        if command.action == "wait":
            wait_entered.set()
            unblock.wait(1)

    executor = _Executor(execute)
    task.executor = executor
    task.key_down()
    task.key_up()
    assert wait_entered.wait(1)
    assert task.key_down()
    task.key_up()
    unblock.set()
    assert task.wait(1)
    assert executor.actions == ["attack", "wait"]


def test_macro_file_hot_reload_and_character_selection(tmp_path):
    from bgi_touch.combat.one_key_fight import HOLD_ON_MODE

    task, _, path = _task(tmp_path, "attack", mode=HOLD_ON_MODE)
    task.reload_if_needed()
    assert [command.action for command in task._macros["可莉"]] == ["attack"]
    time.sleep(0.002)
    _write_macro(path, "e")
    assert task.reload_if_needed()
    assert [command.action for command in task._macros["可莉"]] == ["e"]


def test_dsl_mouse_down_up_maps_to_touch_holds():
    from bgi_touch.combat.dsl import CombatExecutor, parse_combat_script

    input_sim = SimpleNamespace(
        attack_down=Mock(), attack_up=Mock(), button_down=Mock(), button_up=Mock(),
    )
    executor = CombatExecutor(input_sim, sleep=lambda _ms: None)
    commands = parse_combat_script(
        "mousedown(left),mouseup(left),mousedown(right),mouseup(right)"
    )[0].commands
    for command in commands:
        executor.exec(command)
    input_sim.attack_down.assert_called_once_with()
    input_sim.attack_up.assert_called_once_with()
    assert input_sim.button_down.call_args_list == [call("aim")]
    assert input_sim.button_up.call_args_list == [call("aim")]


def test_upstream_default_macro_corpus_is_fully_supported():
    from bgi_touch.combat.dsl import KNOWN_ACTIONS
    from bgi_touch.combat.one_key_fight import DEFAULT_MACRO_PATH, load_avatar_macros

    macros = load_avatar_macros(DEFAULT_MACRO_PATH)
    commands = [command for values in macros.values() for command in values]
    assert len(macros) == 10
    assert len(commands) == 111
    assert {command.action for command in commands} <= KNOWN_ACTIONS


def test_js_runtime_exposes_one_key_fight_and_stops_it(tmp_path):
    import pytest

    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    macro = tmp_path / "macro.json"
    _write_macro(macro, "")
    (tmp_path / "main.js").write_text(
        "oneKeyFight.KeyDown(); oneKeyFight.KeyUp(); return true;",
        encoding="utf-8",
    )
    input_sim = _Input()
    input_sim.release_all = Mock()
    ctx = SimpleNamespace(
        input=input_sim,
        device=SimpleNamespace(paste_text=Mock()),
        transform=ScreenTransform(1920, 1080),
        sleep=lambda _ms: None,
    )
    runtime = JsScriptRuntime(
        ctx, tmp_path,
        settings={"avatarMacroPath": str(macro)},
        party_slots={"可莉": 1}, log=lambda _message: None,
    )
    assert runtime.run() is True
    assert not runtime._one_key_fight.running
