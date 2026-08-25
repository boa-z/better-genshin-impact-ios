from types import SimpleNamespace
from unittest.mock import Mock, call, patch


class _FakeApi:
    def __init__(self):
        self.returnMainUi = Mock(return_value=True)
        self.switchParty = Mock(return_value=True)
        self.claimEncounterPointsRewards = Mock(return_value=False)
        self.chooseTalkOption = Mock()
        self.clickChatExitUntilMainUi = Mock(return_value=True)


def _task(*, api=None, config=None):
    from bgi_touch.tasks.adventurers_guild import AdventurersGuildTask

    ctx = SimpleNamespace(
        sleep=Mock(),
        input=SimpleNamespace(key_press=Mock(), release_all=Mock()),
    )
    task = AdventurersGuildTask(
        ctx,
        "枫丹",
        api=api or _FakeApi(),
        config=config,
        log=Mock(),
    )
    return task, ctx


def test_adventurers_guild_config_coerces_upstream_options():
    from bgi_touch.tasks.adventurers_guild import AdventurersGuildConfig

    config = AdventurersGuildConfig.from_mapping(raw={
        "country": "Fontaine",
        "dailyRewardPartyName": "好感队",
        "onlyDoOnce": "true",
        "timeoutSeconds": "240",
        "talkSkipTimes": "6",
    })

    assert config.country == "Fontaine"
    assert config.daily_reward_party_name == "好感队"
    assert config.only_do_once is True
    assert config.timeout_s == 240
    assert config.talk_skip_times == 6


def test_adventurers_guild_runs_daily_reward_and_expedition_in_one_dialogue_flow():
    api = _FakeApi()
    task, _ctx = _task(api=api)
    task._return_main_ui = Mock(return_value=True)
    task._route = Mock(return_value=True)
    task._press_catherine_until_talk = Mock(side_effect=[True, True])
    task._choose_option = Mock(side_effect=[True, True])
    task._click_black_confirm = Mock(return_value=True)
    task._close_dialogue = Mock(side_effect=[True, True])

    with patch(
        "bgi_touch.tasks.adventurers_guild.OneKeyExpeditionTask"
    ) as expedition:
        expedition.return_value.run.return_value = True
        result = task.run()

    assert result["ok"] is True
    assert result["routed"] is True
    assert result["dailyReward"] == "selected"
    assert result["dailyConfirmation"] is True
    assert result["expedition"] == "selected"
    assert result["expeditionCompleted"] is True
    api.claimEncounterPointsRewards.assert_called_once_with(12.0)
    expedition.return_value.run.assert_called_once()
    assert task._close_dialogue.call_count == 2


def test_adventurers_guild_only_do_once_skips_encounter_points_and_keeps_route_flow():
    api = _FakeApi()
    task, _ctx = _task(api=api, config={"onlyDoOnce": "true"})
    task._return_main_ui = Mock(return_value=True)
    task._route = Mock(return_value=True)
    task._press_catherine_until_talk = Mock(return_value=True)
    task._choose_option = Mock(return_value=False)
    task._close_dialogue = Mock(return_value=True)

    result = task.run()

    assert result["ok"] is True
    assert result["onlyDoOnce"] is True
    assert result["dailyReward"] == "unavailable"
    assert result["expedition"] == "unavailable"
    api.claimEncounterPointsRewards.assert_not_called()
    assert task._route.call_count == 1


def test_adventurers_guild_keeps_cleanup_inside_realtime_exclusive_scope():
    api = _FakeApi()
    task, _ctx = _task(api=api)
    task._return_main_ui = Mock(return_value=True)
    task._route = Mock(return_value=True)
    task._press_catherine_until_talk = Mock(return_value=True)
    task._choose_option = Mock(return_value=False)
    task._close_dialogue = Mock(return_value=True)
    events = []

    class Scope:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_args):
            events.append("exit")

    with patch(
        "bgi_touch.tasks.adventurers_guild.exclusive_realtime_triggers",
        return_value=Scope(),
    ) as isolated:
        result = task.run()

    assert result["ok"] is True
    isolated.assert_called_once_with(task.ctx)
    assert events == ["enter", "exit"]
    assert task._close_dialogue.call_count == 1


def test_adventurers_guild_cancellation_after_route_still_closes_dialogue():
    api = _FakeApi()
    task, _ctx = _task(api=api)
    task._return_main_ui = Mock(return_value=True)
    task._route = Mock(return_value=True)
    task._press_catherine_until_talk = Mock(return_value=True)
    task._choose_option = Mock(return_value=False)
    task._close_dialogue = Mock(return_value=True)

    # The early checks allow the task to reach the route; a later check
    # interrupts before selecting the next dialogue option.
    cancelled = iter((False, False, True))
    result = task.run(cancelled=lambda: next(cancelled))

    assert result["ok"] is False
    assert result["cancelled"] is True
    task._close_dialogue.assert_called_once_with(None)


def test_dispatcher_passes_adventurers_guild_options():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    api = Mock()
    api.goToAdventurersGuild.return_value = True
    dispatcher = TaskDispatcher(object())
    dispatcher._genshin_api = Mock(return_value=api)

    assert dispatcher.run_task({
        "name": "GoToAdventurersGuildTask",
        "config": {
            "country": "Fontaine",
            "dailyRewardPartyName": "好感队",
            "onlyDoOnce": "true",
            "timeoutSeconds": 300,
            "encounterTimeoutSeconds": 9,
        },
    }) is True

    api.goToAdventurersGuild.assert_called_once_with(
        "Fontaine",
        daily_reward_party_name="好感队",
        only_do_once=True,
        timeout_s=300.0,
        encounter_timeout_s=9.0,
    )


def test_one_dragon_daily_rewards_does_not_claim_encounter_points_twice():
    from bgi_touch.tasks.one_dragon import OneDragonFlowTask, OneDragonItem

    class Dispatcher:
        pass

    flow = OneDragonFlowTask(
        SimpleNamespace(),
        {
            "adventurersGuildCountry": "枫丹",
            "dailyRewardPartyName": "好感队",
        },
        Dispatcher(),
        log=lambda _message: None,
    )
    flow.genshin = Mock()
    flow.genshin.goToAdventurersGuild.return_value = True
    flow.genshin.claimBattlePassRewards.return_value = True

    assert flow._run_builtin(OneDragonItem("daily", "领取每日奖励", True)) is True
    flow.genshin.goToAdventurersGuild.assert_called_once_with(
        "枫丹", daily_reward_party_name="好感队",
    )
    flow.genshin.claimEncounterPointsRewards.assert_not_called()
    flow.genshin.claimBattlePassRewards.assert_called_once_with()
