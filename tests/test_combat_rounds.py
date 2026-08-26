from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest


def test_text_combat_parser_applies_round_markers_per_pipe_clause():
    from bgi_touch.combat.dsl import parse_combat_script

    lines = parse_combat_script(
        "钟离 round(1, 3-5), keypress(E)|round(2), keypress(Q)|keypress(SPACE)"
    )

    assert len(lines) == 1
    assert lines[0].character == "钟离"
    assert [command.action for command in lines[0].commands] == [
        "keypress", "keypress", "keypress",
    ]
    assert [command.params for command in lines[0].commands] == [
        ["E"], ["Q"], ["SPACE"],
    ]
    assert lines[0].commands[0].activating_rounds == (1, 3, 4, 5)
    assert lines[0].commands[1].activating_rounds == (2,)
    assert lines[0].commands[2].activating_rounds == ()


def test_text_combat_parser_splits_top_level_semicolons_and_avatar_aliases():
    from bgi_touch.combat.dsl import parse_combat_script

    lines = parse_combat_script("钟离 e;万叶 q; wait(0.1)")

    assert [(line.character, [command.action for command in line.commands]) for line in lines] == [
        ("钟离", ["e"]),
        ("枫原万叶", ["q"]),
        (None, ["wait"]),
    ]


def test_text_combat_parser_uses_dispatcher_default_avatar_for_unprefixed_lines():
    from bgi_touch.combat.dsl import parse_combat_script

    lines = parse_combat_script(
        "keypress(E); wait(0.1)",
        default_avatar="万叶",
    )

    assert [line.character for line in lines] == ["枫原万叶", "枫原万叶"]


@pytest.mark.parametrize(
    "script",
    [
        "round",
        "round()",
        "round(0)",
        "round(3-1)",
        "round(1-3-5)",
        "round(nope)",
    ],
)
def test_text_combat_parser_rejects_invalid_round_markers(script):
    from bgi_touch.combat.dsl import parse_combat_script

    with pytest.raises(ValueError):
        parse_combat_script(script)


def test_combat_executor_filters_rounds_and_does_not_switch_inactive_lines():
    from bgi_touch.combat.dsl import CombatExecutor

    input_sim = SimpleNamespace(
        key_press=Mock(),
        release_all=Mock(),
    )
    end_check = Mock(side_effect=[False, True])
    executor = CombatExecutor(
        input_sim,
        sleep=lambda _milliseconds: None,
        party_slots={"钟离": 1, "可莉": 2},
        check_combat_end=end_check,
    )

    executor.run(
        "钟离 round(2), keypress(E)\n可莉 round(1), keypress(Q)|keypress(SPACE)",
        loop_until_end=True,
    )

    assert input_sim.key_press.call_args_list == [
        call("2"), call("Q"), call("SPACE"),
        call("1"), call("E"), call("2"), call("SPACE"),
    ]
    input_sim.release_all.assert_called_once_with()
    assert end_check.call_count == 2


def test_auto_fight_text_loop_advances_round_number():
    from bgi_touch.combat.dsl import parse_combat_script
    from bgi_touch.tasks.auto_fight import AutoFightTask

    class Executor:
        def __init__(self):
            self.actions = []

        def switch_to(self, character):
            self.actions.append(("switch", character))

        def exec(self, command):
            self.actions.append(("command", command.action, tuple(command.params)))

    executor = Executor()
    task = object.__new__(AutoFightTask)
    task.timeout_s = 5
    task.log = lambda _message: None
    task.lines = parse_combat_script(
        "钟离 round(1), keypress(E)|round(2), keypress(Q)"
    )
    task.executor = executor
    task.finish_detect_enabled = True
    task.finish_detector = SimpleNamespace(
        config=SimpleNamespace(check_after_switch_avatar=False),
        should_fast_check=lambda _previous: False,
    )
    task.experience_detector = SimpleNamespace(available=False)
    task.ctx = SimpleNamespace(input=SimpleNamespace(release_all=Mock()))
    task._start_battle = lambda: None
    task._finish_fight = Mock(return_value=True)
    task._check_fight_end = Mock(side_effect=[False, True])
    task._battle_count = 0

    assert task._run_txt() is True
    assert executor.actions == [
        ("switch", "钟离"),
        ("command", "keypress", ("E",)),
        ("switch", "钟离"),
        ("command", "keypress", ("Q",)),
    ]
    task.ctx.input.release_all.assert_called_once_with()
