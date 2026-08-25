from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest


def _strategy(**overrides):
    value = {
        "Info": {
            "Name": "测试策略",
            "Author": "BetterGI",
            "Config": {"mode": "test"},
            "PreActions": ["钟离 e"],
        },
        "Actions": [
            {
                "Name": "钟离-e-开场",
                "Character": "钟离",
                "Action": "e(hold), attack(0.1)",
                "Condition": {"Expression": "since()>2"},
                "Index": 3,
                "EnsureCast": True,
                "MorePriorities": [
                    {"Expression": "q-ready()", "Priority": 1},
                ],
            },
        ],
    }
    value.update(overrides)
    return value


def test_parse_json_strategy_accepts_bettergi_field_casing():
    from bgi_touch.combat.json_strategy import parse_json_strategy

    parsed = parse_json_strategy(json.dumps(_strategy(), ensure_ascii=False))
    assert parsed.info.name == "测试策略"
    assert parsed.info.pre_actions == ("钟离 e",)
    assert parsed.character_names == ("钟离",)
    assert parsed.actions[0].ensure_cast
    assert parsed.actions[0].more_priorities[0].priority == 1

    camel = {
        "info": {"name": "camel", "preActions": []},
        "actions": [{
            "name": "普攻", "character": "", "action": "attack(1)",
            "condition": {"expression": "true"}, "index": 1,
            "ensureCast": False, "morePriorities": [],
        }],
    }
    assert parse_json_strategy(camel).actions[0].condition.expression == "true"


def test_community_decimal_action_name_is_an_exact_identifier():
    from bgi_touch.combat.json_strategy import ConditionEvaluator, parse_json_strategy

    value = _strategy()
    value["Actions"][0]["Name"] = "木偶-喷冰4.15"
    value["Actions"][0]["Condition"]["Expression"] = "since(木偶-喷冰4.15)>3"
    parsed = parse_json_strategy(value)
    evaluator = ConditionEvaluator(action_names=[parsed.actions[0].name])

    assert evaluator.evaluate(
        parsed.actions[0].condition.expression,
        parsed.actions[0].index,
        action_name=parsed.actions[0].name,
    )


def test_formulaic_community_strategy_conditions_remain_compatible():
    """Exercise the bare-function syntax used by current community JSON files."""
    from bgi_touch.combat.json_strategy import ConditionEvaluator, parse_json_strategy

    value = {
        "info": {"name": "公式化锄地语料", "preActions": []},
        "actions": [
            {
                "name": "检查战斗结束",
                "character": "",
                "action": "check,w(0.05),click(middle),scroll(10)",
                "condition": {"expression": "t>1.5 && since>0.7"},
                "index": 1,
            },
            {
                "name": "芙芙-e",
                "character": "芙宁娜",
                "action": "e,wait(0.3)",
                "condition": {"expression": "since>28 && e-ready"},
                "index": 21,
                "morePriorities": [
                    {"expression": "since>20 && e-ready", "priority": 30},
                ],
            },
            {
                "name": "满芙普攻",
                "character": "芙宁娜",
                "action": "click",
                "condition": {
                    "expression": "(since(21)<10) && (count(27, t-10, t)<2)",
                },
                "index": 27,
            },
            {
                "name": "木偶-喷冰4.15",
                "character": "桑多涅",
                "action": "charge(4.15)",
                "condition": {"expression": "count=0 || since(38)>since(44)"},
                "index": 38,
            },
        ],
    }

    parsed = parse_json_strategy(value)
    messages = []
    evaluator = ConditionEvaluator(
        action_names=[action.name for action in parsed.actions],
        party_names=["芙宁娜", "桑多涅"],
        q_ready=lambda *_: True,
        e_cd=lambda *_: 0.0,
        clock=_Clock(),
        log=messages.append,
    )

    for action in parsed.actions:
        expressions = [action.condition.expression]
        expressions.extend(item.expression for item in action.more_priorities)
        for expression in expressions:
            evaluator.evaluate(expression, action.index, action.character, action.name)

    assert not messages
    assert evaluator.evaluate(
        "since(木偶-喷冰4.15)>3", 27, "芙宁娜", "满芙普攻"
    )


@pytest.mark.parametrize(
    "name",
    ["true", "FALSE", "12", "动作 名", "动作,名", "动作+名", "since", "动作_1"],
)
def test_parse_json_strategy_rejects_names_that_conditions_cannot_reference(name):
    from bgi_touch.combat.json_strategy import StrategyFormatError, parse_json_strategy

    value = _strategy()
    value["Actions"][0]["Name"] = name
    with pytest.raises(StrategyFormatError, match="动作名称"):
        parse_json_strategy(value)


def test_expand_priorities_is_stable_and_filters_known_party():
    from bgi_touch.combat.json_strategy import expand_priorities, parse_json_strategy

    value = _strategy()
    value["Actions"].append({
        "Name": "异队动作", "Character": "温迪", "Action": "q",
        "Condition": {"Expression": "true"}, "Index": 0,
        "MorePriorities": [],
    })
    value["Actions"].append({
        "Name": "同优先级", "Character": "钟离", "Action": "attack(1)",
        "Condition": {"Expression": "true"}, "Index": 1,
        "MorePriorities": [],
    })
    entries = expand_priorities(parse_json_strategy(value), {"钟离": 1})

    assert [(entry.action.name, entry.priority) for entry in entries] == [
        ("钟离-e-开场", 1),
        ("同优先级", 1),
        ("钟离-e-开场", 3),
    ]


class _Clock:
    def __init__(self, value=100.0):
        self.value = value

    def __call__(self):
        return self.value


def test_condition_evaluator_arithmetic_boolean_and_short_circuit():
    from bgi_touch.combat.json_strategy import ConditionEvaluator

    messages = []
    evaluator = ConditionEvaluator(clock=_Clock(), log=messages.append)
    assert evaluator.evaluate("1 + 2 * 3 = 7 && !(2>3)", 1)
    assert evaluator.evaluate("true || unknown()", 1)
    assert evaluator.evaluate("false && unknown()", 1) is False
    assert evaluator.evaluate("1/0=0", 1)
    assert not messages


def test_condition_evaluator_history_by_name_index_and_time_window():
    from bgi_touch.combat.json_strategy import ConditionEvaluator

    clock = _Clock()
    evaluator = ConditionEvaluator(
        action_names=["钟离-e-开场", "同序号动作"], clock=clock,
    )
    assert evaluator.evaluate("since()>999", 3, "钟离", "钟离-e-开场")
    assert evaluator.evaluate("last-exec(2)", 3, "钟离", "钟离-e-开场")

    evaluator.update_last_exec_time(3, "钟离-e-开场")
    clock.value += 3
    evaluator.update_last_exec_time(3, "同序号动作")
    clock.value += 2

    assert evaluator.evaluate("since()=5", 3, "钟离", "钟离-e-开场")
    assert evaluator.evaluate("since(3)=2", 9)
    assert evaluator.evaluate("since(钟离-e-开场)=5", 9)
    assert evaluator.evaluate("count(3)=2", 9)
    assert evaluator.evaluate("count(3,2,t)=1", 9)
    assert evaluator.evaluate("last-exec(4,true,钟离-e-开场)", 9)
    assert evaluator.evaluate("last-exec(6,false,钟离-e-开场)", 9)
    assert evaluator.evaluate("min(since(3),10)=2 && max(1,2,3)=3", 9)


def test_condition_evaluator_visual_callbacks_share_cycle_frame():
    from bgi_touch.combat.json_strategy import ConditionEvaluator

    frame = object()
    q_ready = Mock(return_value=True)
    e_cd = Mock(return_value=2.5)
    low_hp = Mock(return_value=True)
    evaluator = ConditionEvaluator(
        party_names=["钟离", "芙宁娜"],
        active_character=lambda: "钟离",
        q_ready=q_ready,
        e_cd=e_cd,
        low_hp=low_hp,
        last_check=lambda: 4.0,
    )
    evaluator.set_frame(frame)

    assert evaluator.evaluate(
        "in-party(钟离) && onfield() && q-ready() && q-ready(钟离) "
        "&& !e-ready() && e-cd()=2.5 && low-hp() && last-check()>3",
        1, "钟离", "动作",
    )
    q_ready.assert_called_once_with("钟离", frame)
    e_cd.assert_called_once_with("钟离", frame)
    low_hp.assert_called_once_with(frame)

    evaluator.set_frame(object())
    assert evaluator.evaluate("q-ready()", 1, "钟离", "动作")
    assert q_ready.call_count == 2


def test_condition_errors_fail_closed_and_unknown_action_name_is_not_infinity():
    from bgi_touch.combat.json_strategy import ConditionEvaluator

    messages = []
    evaluator = ConditionEvaluator(action_names=["正确动作"], log=messages.append)
    assert not evaluator.evaluate("since(拼错动作)>1", 1, action_name="正确动作")
    assert not evaluator.evaluate("1 2", 1)
    assert len(messages) == 2


def test_auto_fight_dispatches_json_and_executes_only_first_matching_action(tmp_path):
    from bgi_touch.tasks.auto_fight import AutoFightTask

    value = _strategy()
    value["Info"]["PreActions"] = []
    value["Actions"] = [
        {
            "Name": "第一动作", "Character": "钟离", "Action": "e",
            "Condition": {"Expression": "true"}, "Index": 1,
            "MorePriorities": [],
        },
        {
            "Name": "第二动作", "Character": "钟离", "Action": "q",
            "Condition": {"Expression": "true"}, "Index": 2,
            "MorePriorities": [],
        },
    ]
    strategy_path = tmp_path / "combat.json"
    strategy_path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    input_sim = SimpleNamespace(_active_slot=1, release_all=Mock(), attack=Mock())
    ctx = SimpleNamespace(
        input=input_sim,
        capture_bgr=Mock(return_value=np.zeros((108, 192, 3), dtype=np.uint8)),
        sleep=Mock(),
        layout=SimpleNamespace(buttons={"burst": (0.65, 0.92)}),
        _trigger_loop=None,
    )
    executor = Mock()
    detector = Mock()
    detector.config = SimpleNamespace(check_after_switch_avatar=False)
    detector.last_check_at = 0.0
    detector.should_fast_check.return_value = False

    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 3

    with patch("bgi_touch.tasks.auto_fight.CombatExecutor.for_context", return_value=executor), \
            patch("bgi_touch.tasks.auto_fight.FightFinishDetector", return_value=detector):
        task = AutoFightTask(
            ctx, str(strategy_path), timeout_s=10, party_slots={"钟离": 1},
        )
        assert task.json_strategy is not None
        assert not task.run(cancelled=cancelled)

    executor.switch_to.assert_called_once_with("钟离")
    assert executor.exec.call_count == 1
    assert executor.exec.call_args.args[0].action == "e"
    ctx.capture_bgr.assert_called_once_with()
    detector.start_battle.assert_called_once_with()
    input_sim.release_all.assert_called()


def test_converter_validates_and_copies_json_combat_strategy(tmp_path):
    from bgi_touch.converter.convert import convert_any, detect_kind

    source = tmp_path / "strategy.json"
    source.write_text(json.dumps(_strategy(), ensure_ascii=False), encoding="utf-8")
    assert detect_kind(source) == "combat"

    result = convert_any(source, tmp_path / "output")
    assert result["kind"] == "combat"
    assert result["actions"] == 1
    assert (tmp_path / "output" / "combat" / "strategy.json").is_file()


def test_combat_cli_routes_json_to_auto_fight_and_closes_context():
    from bgi_touch import cli

    ctx = Mock()
    task = Mock()
    task.return_value.run.return_value = True
    args = SimpleNamespace(file="strategy.json", timeout=45)
    with patch("bgi_touch.cli._context", return_value=ctx), \
            patch("bgi_touch.cli._load_party", return_value={"钟离": 1}), \
            patch("bgi_touch.tasks.auto_fight.AutoFightTask", task):
        assert cli.cmd_combat(args) == 0

    task.assert_called_once_with(
        ctx, "strategy.json", timeout_s=45, party_slots={"钟离": 1},
    )
    ctx.close.assert_called_once_with()
