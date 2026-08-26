from types import SimpleNamespace
from unittest.mock import Mock


def _ctx():
    return SimpleNamespace(
        input=SimpleNamespace(
            key_press=Mock(),
            move_camera_by=Mock(),
            release_all=Mock(),
        ),
        sleep=Mock(),
    )


def test_post_fight_pickup_gates_kazuha_on_elite_experience():
    from bgi_touch.combat.pickup import PostFightPickup, PostFightPickupConfig

    ctx = _ctx()
    executor = Mock()
    task = PostFightPickup(
        ctx,
        party_slots={"枫原万叶": 3},
        config=PostFightPickupConfig(exp_based_pickup_enabled=True),
        executor=executor,
    )

    skipped = task.run(elite_detected=False)
    assert skipped.picker is None
    executor.switch_to.assert_not_called()

    picked = task.run(elite_detected=True)
    assert picked.picker == "枫原万叶"
    executor.switch_to.assert_called_once_with("枫原万叶")
    assert [call.args[0].action for call in executor.exec.call_args_list] == [
        "e", "attack",
    ]
    ctx.input.release_all.assert_called_once_with()


def test_post_fight_pickup_runs_scan_without_kazuha():
    from bgi_touch.combat.pickup import PostFightPickup, PostFightPickupConfig

    class Clock:
        value = 0.0

        def __call__(self):
            self.value += 0.4
            return self.value

    ctx = _ctx()
    result = PostFightPickup(
        ctx,
        config=PostFightPickupConfig(
            kazuha_pickup_enabled=False,
            pick_drops_after_fight_enabled=True,
            pick_drops_after_fight_seconds=1,
        ),
        clock=Clock(),
    ).run()

    assert result.scan_requested is True
    assert ctx.input.key_press.call_count >= 1
    assert ctx.input.move_camera_by.call_count >= 1


def test_post_fight_pickup_config_accepts_bettergi_names_and_clamps_seconds():
    from bgi_touch.combat.pickup import PostFightPickupConfig

    config = PostFightPickupConfig.from_mapping({
        "KazuhaPickupEnabled": False,
        "PickDropsAfterFightEnabled": True,
        "PickDropsAfterFightSeconds": 999,
        "ExpBasedPickupEnabled": True,
        "BattleThresholdForLoot": "3",
        "QinDoublePickUp": "true",
    })
    assert config.kazuha_pickup_enabled is False
    assert config.pick_drops_after_fight_enabled is True
    assert config.pick_drops_after_fight_seconds == 120
    assert config.exp_based_pickup_enabled is True
    assert config.battle_threshold_for_loot == 3
    assert config.qin_double_pick_up is True


def test_pathing_monster_policy_disables_post_fight_pickup_for_non_elite():
    from bgi_touch.combat.pickup import PostFightPickupConfig

    config = PostFightPickupConfig.from_mapping({
        "onlyPickEliteDropsMode": "DisableAutoPickupForNonElite",
        "kazuhaPickupEnabled": True,
        "pickDropsAfterFightEnabled": True,
    })
    effective, suspend = config.apply_pathing_monster_policy(
        enabled=True,
        monster_tag="normal",
    )

    assert effective.kazuha_pickup_enabled is False
    assert effective.pick_drops_after_fight_enabled is False
    assert suspend is True

    elite, elite_suspend = config.apply_pathing_monster_policy(
        enabled=True,
        monster_tag="legendary",
    )
    assert elite == config
    assert elite_suspend is False


def test_dispatcher_passes_post_fight_options_to_auto_fight():
    from unittest.mock import patch

    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.auto_fight.AutoFightTask") as task:
        task.return_value.run.return_value = True
        assert TaskDispatcher(object()).run_auto_fight_task({
            "kazuhaPickupEnabled": False,
            "pickDropsAfterFightEnabled": True,
            "pickDropsAfterFightSeconds": 9,
            "expBasedPickupEnabled": True,
        })

    config = task.call_args.kwargs["post_fight_config"]
    assert config.kazuha_pickup_enabled is False
    assert config.pick_drops_after_fight_enabled is True
    assert config.pick_drops_after_fight_seconds == 9
    assert config.exp_based_pickup_enabled is True


def test_dispatcher_suspends_only_autopick_for_non_elite_route_fight():
    from types import SimpleNamespace
    from unittest.mock import patch
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    trigger = SimpleNamespace(enabled=True)
    loop = SimpleNamespace(get=lambda name: trigger if name == "AutoPick" else None)
    ctx = SimpleNamespace(_trigger_loop=loop)
    observed = []

    with patch("bgi_touch.tasks.auto_fight.AutoFightTask") as task:
        task.return_value.run.side_effect = lambda **_kwargs: observed.append(
            trigger.enabled
        ) or True
        assert TaskDispatcher(ctx).run_auto_fight_task({
            "onlyPickEliteDropsMode": "DisableAutoPickupForNonElite",
            "_pathingEnableMonsterLootSplit": True,
            "_pathingMonsterTag": "normal",
        })

    assert observed == [False]
    assert trigger.enabled is True
