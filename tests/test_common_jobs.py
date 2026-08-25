from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import pytest


def _input():
    return SimpleNamespace(
        key_down=Mock(),
        key_up=Mock(),
        key_press=Mock(),
        move_camera_by=Mock(),
        tap_button=Mock(),
        release_all=Mock(),
    )


def test_walk_to_f_releases_movement_and_presses_interaction():
    from bgi_touch.tasks.common_jobs import WalkToFTask

    input_simulator = _input()
    ctx = SimpleNamespace(
        input=input_simulator,
        capture_region=Mock(return_value=object()),
        sleep=Mock(),
    )
    detector = Mock(side_effect=[False, True])

    with patch("bgi_touch.tasks.common_jobs.time.monotonic", side_effect=[0, 0, 1]):
        assert WalkToFTask(
            ctx,
            run_to_f=True,
            detector=detector,
            timeout_s=10,
            poll_interval_ms=100,
        ).run()

    assert input_simulator.key_down.call_args_list == [call("W"), call("LSHIFT")]
    assert input_simulator.key_press.call_args_list == [call("F")]
    assert input_simulator.key_up.call_args_list == [call("W"), call("LSHIFT")]
    assert detector.call_count == 2


def test_walk_to_f_cancellation_always_clears_held_keys():
    from bgi_touch.tasks.common_jobs import WalkToFTask

    input_simulator = _input()
    ctx = SimpleNamespace(
        input=input_simulator,
        capture_region=Mock(),
        sleep=Mock(),
    )

    assert not WalkToFTask(ctx, run_to_f=True).run(cancelled=lambda: True)
    assert input_simulator.key_up.call_args_list == [call("W"), call("LSHIFT")]
    ctx.capture_region.assert_not_called()


def test_scan_pick_fallback_reuses_no_extra_frame_producer():
    from bgi_touch.tasks.common_jobs import ScanPickTask

    input_simulator = _input()
    ctx = SimpleNamespace(
        input=input_simulator,
        sleep=Mock(),
    )

    clock = Mock(side_effect=[0, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert ScanPickTask(
        ctx,
        seconds=0.5,
        sweep_interval_ms=200,
        clock=clock,
    ).run()

    # The fallback is camera/input-only: it does not call capture_bgr and
    # therefore cannot compete with TriggerLoop's caller-owned screenshot.
    assert not hasattr(ctx, "capture_bgr")
    assert input_simulator.move_camera_by.call_count >= 1
    input_simulator.release_all.assert_called_once_with()
    assert input_simulator.key_up.call_args_list[-4:] == [
        call("W"), call("A"), call("S"), call("D"),
    ]


def test_scan_pick_restores_owned_autopick_trigger():
    from bgi_touch.tasks.common_jobs import ScanPickTask

    input_simulator = _input()
    old_trigger = object()

    class Loop:
        triggers = [old_trigger]

        def get(self, name):
            assert name == "AutoPick"
            return None

        def replace(self, value):
            self.restored = list(value)

        def stop(self):
            self.stopped = True

    loop = Loop()
    ctx = SimpleNamespace(
        input=input_simulator,
        sleep=Mock(),
        triggers=loop,
        enable_trigger=Mock(),
    )
    clock = Mock(side_effect=[0, 0, 0.2, 0.4, 0.6, 1.0])

    assert ScanPickTask(ctx, seconds=0.4, sweep_interval_ms=200, clock=clock).run()
    ctx.enable_trigger.assert_called_once_with("AutoPick")
    assert loop.restored == [old_trigger]


def test_scan_pick_moves_towards_closest_reference_item():
    from bgi_touch.tasks.common_jobs import PickItem, ScanPickTask

    input_simulator = _input()
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
    ctx = SimpleNamespace(
        input=input_simulator,
        sleep=Mock(),
        capture_bgr=Mock(return_value=frames[0]),
        transform=SimpleNamespace(to_ref=lambda x, y: (x, y)),
    )
    clock = Mock(side_effect=[0, 0, 0.1, 1.0])
    task = ScanPickTask(
        ctx,
        seconds=0.5,
        sweep_interval_ms=500,
        detector=lambda _frame: [
            PickItem(300, 300, 40, 40),
            PickItem(940, 660, 40, 40),
        ],
        clock=clock,
    )

    assert task.run()
    assert call("W") in input_simulator.key_down.call_args_list
    assert call("S") in input_simulator.key_up.call_args_list
    assert call("F") in input_simulator.key_press.call_args_list


def test_dispatcher_maps_common_job_parameters():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.common_jobs.WalkToFTask") as walk, \
         patch("bgi_touch.tasks.common_jobs.ScanPickTask") as scan, \
         patch("bgi_touch.tasks.common_jobs.LowerHeadThenWalkToTask") as lower:
        walk.return_value.run.return_value = True
        scan.return_value.run.return_value = True
        lower.return_value.run.return_value = True
        dispatcher = TaskDispatcher(object())

        assert dispatcher.run_task({"name": "WalkToFTask", "config": {
            "timeoutMilliseconds": 2500,
            "needPress": False,
            "runToF": True,
        }})
        assert dispatcher.run_task({"name": "ScanPick", "config": {
            "scanSeconds": 9,
            "sweepIntervalMilliseconds": 300,
        }})
        assert dispatcher.run_task({"name": "LowerHeadThenWalkToTask", "config": {
            "targetMatName": "custom.png",
            "timeoutMilliseconds": 8000,
        }})

    assert walk.call_args.kwargs["timeout_s"] == 2.5
    assert walk.call_args.kwargs["need_press"] is False
    assert walk.call_args.kwargs["run_to_f"] is True
    assert scan.call_args.kwargs["seconds"] == 9
    assert scan.call_args.kwargs["sweep_interval_ms"] == 300
    assert lower.call_args.args[1] == "custom.png"
    assert lower.call_args.kwargs["timeout_s"] == 8


def test_dispatcher_maps_genshin_common_jobs_and_reuses_facade():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.engine.genshin_api.GenshinApi") as api_type:
        api = api_type.return_value
        api.claimMailRewards.return_value = True
        api.goToCraftingBench.return_value = True
        api.craftMaterial.return_value = {
            "success": True, "materialName": "浓缩树脂", "crafted": 2,
        }
        api.setTime.return_value = True
        dispatcher = TaskDispatcher(object())

        assert dispatcher.run_task({"name": "ClaimMailRewardsTask", "config": {}})
        assert dispatcher.run_task({"name": "GoToCraftingBench", "config": {
            "country": "Mondstadt",
        }})
        assert dispatcher.run_task({"name": "CraftMaterialTask", "config": {
            "materialName": "浓缩树脂", "quantity": 2,
        }})["crafted"] == 2
        assert dispatcher.run_task({"name": "SetTimeTask", "config": {
            "hour": "6", "minute": 30, "skip": True,
        }})

    api_type.assert_called_once()
    api.claimMailRewards.assert_called_once_with()
    api.goToCraftingBench.assert_called_once_with("Mondstadt")
    api.craftMaterial.assert_called_once_with("浓缩树脂", 2, None)
    api.setTime.assert_called_once_with(6, 30, True)


def test_dispatcher_validates_genshin_common_job_parameters():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    dispatcher = TaskDispatcher(object())
    with pytest.raises(ValueError, match="需要 country"):
        dispatcher.run_task({"name": "GoToCraftingBench", "config": {}})
    with pytest.raises(ValueError, match="需要有效的 quantity"):
        dispatcher.run_task({"name": "CraftMaterial", "config": {
            "materialName": "浓缩树脂", "quantity": "bad",
        }})


def test_converter_reports_genshin_common_job_surface():
    from bgi_touch.converter.convert import SUPPORTED

    assert {
        "genshin.blessingOfTheWelkinMoon",
        "genshin.goToAdventurersGuild",
        "genshin.goToCraftingBench",
        "genshin.craftMaterial",
        "genshin.setTime",
        "dispatcher.runClaimMailRewardsTask",
        "dispatcher.runCraftMaterialTask",
        "dispatcher.runSetTimeTask",
    } <= set(SUPPORTED)
