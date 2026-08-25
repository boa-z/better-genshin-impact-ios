from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


def _context():
    from bgi_touch.vision.coordinate import ScreenTransform

    return SimpleNamespace(
        transform=ScreenTransform(1920, 1080),
        input=SimpleNamespace(
            attack=Mock(), key_press=Mock(), release_all=Mock(),
        ),
        sleep=lambda _ms: None,
        capture_bgr=lambda: np.zeros((1080, 1920, 3), dtype=np.uint8),
    )


def test_parse_wood_statistics_accepts_combined_and_item_ocr():
    from bgi_touch.tasks.auto_wood import parse_wood_statistics

    assert parse_wood_statistics("获得\n竹节×30\n杉木x20") == {
        "竹节": 30,
        "杉木": 20,
    }
    assert parse_wood_statistics([
        SimpleNamespace(text="获得"),
        SimpleNamespace(text="竹节×30"),
        SimpleNamespace(text="杉木×20"),
    ]) == {"竹节": 30, "杉木": 20}
    assert parse_wood_statistics("获得\n未知木×999\nUID×123") == {}


def test_wood_statistics_accumulates_and_stops_at_per_material_limit():
    from bgi_touch.tasks.auto_wood import WoodStatistics

    stats = WoodStatistics(daily_max_count=40)
    stats.record("获得\n竹节×20\n杉木×20")
    assert not stats.reached_max_count
    stats.record("获得\n竹节×20\n杉木×20")

    assert stats.totals == {"竹节": 40, "杉木": 40}
    assert stats.reached_max_count


def test_wood_task_uses_wonderland_refresh_and_stops_when_stats_reach_limit():
    from bgi_touch.tasks.auto_wood import AutoWoodTask

    ctx = _context()
    with patch("bgi_touch.tasks.auto_wood.GenshinApi") as api_type:
        api_type.return_value.wonderlandCycle.return_value = True
        task = AutoWoodTask(
            ctx,
            rounds=4,
            per_round_attacks=0,
            wood_daily_max_count=60,
            wood_count_ocr_enabled=True,
            ocr_final_round=True,
            gadget_check_enabled=False,
            ocr_provider=lambda _crop: "获得\n竹节×30\n杉木×20",
            log=lambda _message: None,
        )

        assert task.run()

    assert task.wood_totals == {"竹节": 90, "杉木": 60}
    assert ctx.input.key_press.call_count == 3
    assert api_type.return_value.wonderlandCycle.call_count == 2


def test_wood_task_stops_after_consecutive_empty_ocr_results():
    from bgi_touch.tasks.auto_wood import AutoWoodTask

    ctx = _context()
    with patch("bgi_touch.tasks.auto_wood.GenshinApi") as api_type:
        api_type.return_value.wonderlandCycle.return_value = True
        task = AutoWoodTask(
            ctx,
            rounds=8,
            per_round_attacks=0,
            wood_count_ocr_enabled=True,
            ocr_final_round=True,
            gadget_check_enabled=False,
            ocr_provider=lambda _crop: "获得",
            log=lambda _message: None,
        )

        assert task.run()

    assert task.nothing_count == 3
    assert ctx.input.key_press.call_count == 3
    assert api_type.return_value.wonderlandCycle.call_count == 2


def test_wood_round_zero_keeps_bettergi_unlimited_round_semantics():
    from bgi_touch.tasks.auto_wood import AutoWoodTask

    task = AutoWoodTask(
        _context(),
        rounds=0,
        wood_daily_max_count=0,
        gadget_check_enabled=False,
    )

    assert task.rounds == 9999
    assert task.wood_daily_max_count == 9999


def test_dispatcher_maps_full_auto_wood_configuration():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.auto_wood.AutoWoodTask") as task_type:
        task_type.return_value.run.return_value = True
        result = TaskDispatcher(object()).run_task({
            "name": "AutoWood",
            "config": {
                "woodRoundNum": 0,
                "perRoundAttacks": 0,
                "woodDailyMaxCount": 120,
                "woodCountOcrEnabled": True,
                "useWonderlandRefresh": False,
                "afterZSleepDelay": 450,
                "woodOcrEmptyLimit": 4,
                "woodOcrTimeoutMs": 1800,
                "woodOcrPollIntervalMs": 100,
                "gadgetCheckEnabled": False,
                "gadgetCheckStrict": False,
                "reloginBetween": True,
            },
        })

    assert result is True
    kwargs = task_type.call_args.kwargs
    assert kwargs["rounds"] == 0
    assert kwargs["per_round_attacks"] == 0
    assert kwargs["wood_daily_max_count"] == 120
    assert kwargs["wood_count_ocr_enabled"] is True
    assert kwargs["use_wonderland_refresh"] is False
    assert kwargs["after_z_sleep_delay_ms"] == 450
    assert kwargs["empty_ocr_limit"] == 4
    assert kwargs["ocr_timeout_ms"] == 1800
    assert kwargs["ocr_poll_interval_ms"] == 100
    assert kwargs["gadget_check_enabled"] is False
    assert kwargs["gadget_check_strict"] is False
    assert kwargs["relogin_between"] is True

