from unittest.mock import patch

import pytest

from bgi_touch.tasks.auto_stygian import (
    ResinUseRecord,
    StygianSignals,
    StygianState,
    build_resin_plan,
    choose_stygian_resin,
    detect_stygian_state,
    resin_count_from_lines,
)


def test_stygian_state_detector_uses_upstream_priority():
    assert detect_stygian_state(
        ["继续挑战", "退出秘境", "挑战成功", "返回"],
        StygianSignals(
            continue_button=True,
            exit_button=True,
            white_cancel=True,
        ),
    ) == StygianState.CONTINUE_OR_EXIT
    assert detect_stygian_state(
        ["地脉之花", "使用浓缩树脂"]
    ) == StygianState.RESIN_SELECT
    assert detect_stygian_state(
        ["角色预览", "开始挑战"]
    ) == StygianState.BOSS_SELECT
    assert detect_stygian_state(
        [], StygianSignals(leyline_disorder=True, inventory=True)
    ) == StygianState.DOMAIN_LOBBY
    assert detect_stygian_state(
        [], StygianSignals(leyline_disorder=True)
    ) == StygianState.BATTLE_ARENA


def test_stygian_navigation_state_uses_positional_signals():
    assert detect_stygian_state(
        ["幽境危战"], StygianSignals(domain_entrance_title=True)
    ) == StygianState.DOMAIN_ENTRANCE
    assert detect_stygian_state(
        ["活动一览"], StygianSignals(event_menu_title=True)
    ) == StygianState.EVENT_MENU
    assert detect_stygian_state(
        ["幽境危战"], StygianSignals(event_page_title=True)
    ) == StygianState.STYGIAN_PAGE


def test_stygian_resin_plan_honors_priority_and_appends_configured_types():
    plan = build_resin_plan(
        specify=True,
        priority=["原粹树脂", "浓缩树脂"],
        original=2,
        condensed=1,
        transient=3,
    )
    assert [(item.name, item.remaining) for item in plan] == [
        ("原粹树脂", 2),
        ("浓缩树脂", 1),
        ("须臾树脂", 3),
    ]
    with pytest.raises(ValueError, match="至少配置"):
        build_resin_plan(specify=True)
    assert build_resin_plan(specify=False) == []


def test_stygian_resin_choice_and_count_parsing():
    assert resin_count_from_lines(["浓缩树脂 ２", "原粹树脂 160/200"], "浓缩树脂") == 2
    assert resin_count_from_lines(["浓缩树脂 ２", "原粹树脂 160/200"], "原粹树脂") == 160
    assert choose_stygian_resin(["浓缩树脂 2", "原粹树脂 160"]) == "浓缩树脂"
    plan = [
        ResinUseRecord("原粹树脂", 1, 1),
        ResinUseRecord("浓缩树脂", 2, 2),
    ]
    assert choose_stygian_resin(
        ["使用浓缩树脂", "使用原粹树脂"], plan
    ) == "原粹树脂"
    assert choose_stygian_resin(
        ["补充原粹树脂", "浓缩树脂 1"], plan
    ) == "浓缩树脂"


def test_stygian_dispatcher_maps_current_bettergi_parameters():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    config = {
        "bossNum": 3,
        "autoArtifactSalvage": True,
        "specifyResinUse": True,
        "resinPriorityList": ["原粹树脂", "浓缩树脂"],
        "originalResinUseCount": 2,
        "condensedResinUseCount": 1,
        "transientResinUseCount": 3,
        "fragileResinUseCount": 4,
        "fightTeamName": "幽境队",
        "combatScriptBagPath": "combat.txt",
        "confirmQuickSalvage": True,
        "confirmArtifactSalvage": True,
        "maxBattleFailures": 7,
        "timeout": 480,
    }
    with patch("bgi_touch.tasks.auto_stygian.AutoStygianOnslaughtTask") as task:
        task.return_value.run.return_value = True
        assert TaskDispatcher(object()).run_auto_stygian_onslaught_task(config)
    kwargs = task.call_args.kwargs
    assert kwargs["route_path"] is None
    assert kwargs["boss_num"] == 3
    assert kwargs["combat_strategy_path"] == "combat.txt"
    assert kwargs["timeout_s"] == 480
    assert kwargs["auto_artifact_salvage"] is True
    assert kwargs["specify_resin_use"] is True
    assert kwargs["resin_priority_list"] == ["原粹树脂", "浓缩树脂"]
    assert kwargs["original_resin_use_count"] == 2
    assert kwargs["condensed_resin_use_count"] == 1
    assert kwargs["transient_resin_use_count"] == 3
    assert kwargs["fragile_resin_use_count"] == 4
    assert kwargs["fight_team_name"] == "幽境队"
    assert kwargs["max_battle_failures"] == 7
    assert kwargs["artifact_salvage_options"]["confirm_salvage"] is True
