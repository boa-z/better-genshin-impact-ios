from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, call, patch


def test_serenitea_reward_config_supports_upstream_shop_schedule_prefix():
    from bgi_touch.tasks.serenitea_rewards import SereniteaPotRewardConfig

    config = SereniteaPotRewardConfig.from_mapping({
        "enterWithPot": "true",
        "claimTrust": "false",
        "secretTreasureObjects": ["星期一", "布匹", "须臾树脂", "布匹"],
        "leaveAfter": "0",
        "timeoutSeconds": "300",
        "searchTurns": "9",
    })

    assert config.enter_with_pot is True
    assert config.claim_trust is False
    assert config.shop_schedule == "星期一"
    assert config.shop_items == ("布匹", "须臾树脂")
    assert config.leave_after is False
    assert config.timeout_s == 300
    assert config.search_turns == 9


def test_serenitea_shop_schedule_obeys_four_am_reset():
    from bgi_touch.tasks.serenitea_rewards import is_shop_schedule_active

    monday_0359 = datetime(2026, 8, 24, 3, 59, tzinfo=timezone.utc)
    monday_0400 = datetime(2026, 8, 24, 4, 0, tzinfo=timezone.utc)
    assert is_shop_schedule_active("星期日", now=monday_0359)
    assert is_shop_schedule_active("星期一", now=monday_0400)
    assert not is_shop_schedule_active("星期二", now=monday_0400)
    assert is_shop_schedule_active("每天重复", now=monday_0400)
    # Unknown values fail closed so a typo cannot enable a purchase.
    assert not is_shop_schedule_active("星期八", now=monday_0400)


def test_serenitea_approach_reuses_one_frame_and_releases_forward_key():
    from bgi_touch.tasks.serenitea_rewards import SereniteaPotRewardsTask

    target = SimpleNamespace(text="阿圆", x=940, y=180, width=40, height=40)
    region = SimpleNamespace(find_multi=Mock(return_value=[target]))
    input_sim = SimpleNamespace(
        key_down=Mock(), key_up=Mock(), key_press=Mock(),
        move_camera_by=Mock(), release_all=Mock(),
    )
    ctx = SimpleNamespace(
        input=input_sim,
        capture_region=Mock(return_value=region),
        sleep=Mock(),
    )
    task = SereniteaPotRewardsTask(ctx, config={"searchTurns": 1})
    task._interaction = SimpleNamespace(visible=Mock(return_value=True))

    assert task._find_and_approach_npc(10**9, None)
    ctx.capture_region.assert_called_once_with()
    assert input_sim.key_down.call_args_list == [call("W")]
    assert input_sim.key_up.call_args_list == [call("W")]
    assert input_sim.key_press.call_args_list == [call("F")]


def test_serenitea_reward_run_keeps_shop_purchase_explicit_and_restores_triggers():
    from bgi_touch.tasks.serenitea_rewards import SereniteaPotRewardsTask

    input_sim = SimpleNamespace(release_all=Mock())
    ctx = SimpleNamespace(input=input_sim, sleep=Mock())
    task = SereniteaPotRewardsTask(ctx, config={"shopItems": []})
    task._enter = Mock(return_value=True)
    task._find_and_approach_npc = Mock(return_value=True)
    task._choose = Mock(return_value=True)
    task._buy_shop_items = Mock(return_value=[])
    task._dismiss_reward_popup = Mock()

    api = SimpleNamespace(clickChatExitUntilMainUi=Mock(return_value=True))
    with patch("bgi_touch.tasks.serenitea_rewards.GenshinApi", return_value=api), \
         patch("bgi_touch.tasks.quick_serenitea.QuickSereniteaPotTask") as quick:
        quick.return_value.run.return_value = True
        result = task.run()

    assert result["status"] == "completed"
    assert result["trustClaimed"] is True
    task._buy_shop_items.assert_called_once()
    api.clickChatExitUntilMainUi.assert_called_once_with(retry_times=8)
    input_sim.release_all.assert_called_once_with()


def test_dispatcher_and_one_dragon_use_full_serenitea_reward_entrypoint():
    from bgi_touch.tasks.dispatcher import TaskDispatcher
    from bgi_touch.tasks.one_dragon import OneDragonFlowTask, OneDragonItem

    with patch("bgi_touch.tasks.serenitea_rewards.SereniteaPotRewardsTask") as task:
        task.return_value.run.return_value = {"status": "completed"}
        dispatcher = TaskDispatcher(object())
        assert dispatcher.run_task({
            "name": "GoToSereniteaPotTask",
            "config": {"claimTrust": True},
        }) == {"status": "completed"}
        task.assert_called_once()

    class Dispatcher:
        def __init__(self):
            self.config = None

        def run_serenitea_pot_rewards_task(self, config):
            self.config = config
            return {"status": "completed"}

    dispatcher = Dispatcher()
    flow = OneDragonFlowTask(SimpleNamespace(), {
        "taskConfigs": {"pot": {"shopItems": ["布匹"]}},
    }, dispatcher, log=lambda _message: None)
    result = flow._run_builtin(OneDragonItem("pot", "领取尘歌壶奖励", True))

    assert result == {"status": "completed"}
    assert dispatcher.config == {"shopItems": ["布匹"]}
