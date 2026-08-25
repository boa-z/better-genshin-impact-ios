from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch


def _ctx(*, frames=None):
    input_simulator = SimpleNamespace(
        key_down=Mock(),
        key_up=Mock(),
        key_press=Mock(),
        click_ref=Mock(),
        release_all=Mock(),
    )
    value = SimpleNamespace(
        input=input_simulator,
        sleep=Mock(),
        capture_bgr=Mock(side_effect=frames or [object()]),
    )
    return value, input_simulator


def test_parse_resin_inventory_supports_combined_and_split_ocr_lines():
    from bgi_touch.tasks.crafting_bench import parse_resin_inventory

    assert parse_resin_inventory(["原粹树脂 １６０/２００", "浓缩树脂 ２/５"]) == (160, 2)
    assert parse_resin_inventory(["原粹树脂", "160/200", "浓缩树脂", "2/5"]) == (160, 2)
    assert parse_resin_inventory(["Original Resin 40/200", "Condensed Resin 0/5"]) == (40, 0)


def test_calculate_condensed_resin_crafts_respects_keep_and_capacity():
    from bgi_touch.tasks.crafting_bench import calculate_condensed_resin_crafts

    assert calculate_condensed_resin_crafts(160, 0, 20) == 2
    assert calculate_condensed_resin_crafts(300, 4, 0) == 1
    assert calculate_condensed_resin_crafts(50, 0, 0) == 0
    assert calculate_condensed_resin_crafts(1000, 5, 0) == 0
    assert calculate_condensed_resin_crafts(1000, 0, 400) == 5


def test_crafting_route_applies_upstream_party_settings_and_owns_final_interaction():
    from bgi_touch.tasks.crafting_bench import CraftingBenchTask

    ctx, input_simulator = _ctx(frames=[object(), object(), object()])
    detector = Mock()
    detector.visible.return_value = True
    route = SimpleNamespace(realtime_triggers={"AutoPick": True})
    executor = Mock()
    executor.return_value.run.return_value = True

    with patch("bgi_touch.tasks.crafting_bench.PathingTask.load", return_value=route), \
         patch("bgi_touch.tasks.crafting_bench.PathingExecutor", executor):
        task = CraftingBenchTask(
            ctx,
            "Fontaine",
            route_resolver=lambda _kind, _country: Path("craft.json"),
            talk_detector=Mock(side_effect=[False, True]),
            interaction_detector=detector,
        )
        task._is_crafting_screen = Mock(return_value=True)
        assert task.go_to_crafting_bench()

    assert route.realtime_triggers == {}
    assert executor.call_args.kwargs["pathing_config"] == {
        "enabled": True,
        "autoSkipEnabled": True,
        "autoRunEnabled": False,
    }
    assert input_simulator.key_press.call_args_list == [call("F")]


def test_crafting_interaction_uses_backward_step_after_two_failed_presses():
    from bgi_touch.tasks.crafting_bench import CraftingBenchTask

    ctx, input_simulator = _ctx(frames=[object(), object(), object(), object()])
    detector = Mock()
    detector.visible.return_value = False
    task = CraftingBenchTask(
        ctx,
        "枫丹",
        talk_detector=Mock(return_value=False),
        interaction_detector=detector,
    )

    assert not task._press_interaction_until_talk(
        __import__("time").monotonic() + 5,
        lambda: False,
    )
    assert input_simulator.key_press.call_args_list == [call("F"), call("F"), call("F")]
    assert input_simulator.key_down.call_args_list == [call("S")]
    assert input_simulator.key_up.call_args_list == [call("S")]


def test_craft_resin_resets_quantity_and_confirms_both_dialogs():
    from bgi_touch.tasks.crafting_bench import CraftingBenchTask

    ctx, input_simulator = _ctx()
    return_main = Mock(return_value=True)
    task = CraftingBenchTask(
        ctx,
        "枫丹",
        min_resin_to_keep=0,
        return_main_ui=return_main,
    )
    task._click_condensed_resin = Mock(return_value=True)
    task._wait_resin_counts = Mock(return_value=(300, 0))
    task._wait_click_confirm = Mock(side_effect=[True, True])

    assert task._craft_resin_once(__import__("time").monotonic() + 5, lambda: False)
    assert input_simulator.click_ref.call_args_list[:5] == [call(1074, 672)] * 5
    assert input_simulator.click_ref.call_args_list[5:] == [call(1614, 672)] * 4
    assert task._wait_click_confirm.call_args_list[0].args[1] == "white_confirm"
    assert task._wait_click_confirm.call_args_list[1].args[1] == "black_confirm"
    return_main.assert_called_once_with()


def test_dispatcher_passes_resin_policy_to_genshin_api():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.engine.genshin_api.GenshinApi") as api_type:
        api_type.return_value.goCraftResin.return_value = True
        dispatcher = TaskDispatcher(object())
        assert dispatcher.run_task({
            "name": "GoCraftResin",
            "config": {
                "country": "Mondstadt",
                "minResinToKeep": "60",
                "timeoutSeconds": 120,
            },
        })

    api_type.return_value.goCraftResin.assert_called_once_with(
        "Mondstadt",
        min_resin_to_keep=60,
        timeout_s=120.0,
    )
