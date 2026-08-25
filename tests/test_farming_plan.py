import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from bgi_touch.pathing.farming import (
    DailyFarmingData,
    FarmingRecord,
    FarmingRouteInfo,
    FarmingSession,
    FarmingStatsRecorder,
)
from bgi_touch.tasks.travel_diary import (
    DiaryPage,
    GameInfo,
    TravelDiaryUpdate,
    TravelDiaryStore,
    ActionItem,
)
from bgi_touch.pathing.model import PathingTask


TZ8 = timezone(timedelta(hours=8))


def _config(tmp_path, *, enabled=True, elite_cap=2, mob_cap=10, miyoushe_enabled=False):
    path = tmp_path / "config" / "farming.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "enabled": enabled,
        "dailyEliteCap": elite_cap,
        "dailyMobCap": mob_cap,
        "serverTimezone": 8,
        "logDirectory": "../records",
        "miyousheDataConfig": {
            "enabled": miyoushe_enabled,
            "dailyEliteCap": 400,
            "dailyMobCap": 2000,
        },
    }), encoding="utf-8")
    return path


def test_farming_session_accepts_bettergi_field_spellings():
    session = FarmingSession.from_mapping({
        "allowFarmingCount": "true",
        "normal_mob_count": "12.5",
        "eliteMobCount": 3,
        "primary_target": "ELITE",
        "durationSeconds": 9,
        "elite_details": "遗迹守卫",
        "totalMora": 600,
    })
    assert session.allow_farming_count is True
    assert session.normal_mob_count == 12.5
    assert session.elite_mob_count == 3
    assert session.primary_target == "elite"
    assert session.duration_seconds == 9
    assert session.elite_details == "遗迹守卫"
    assert session.total_mora == 600


def test_farming_stats_day_changes_at_server_four_am(tmp_path):
    recorder = FarmingStatsRecorder(_config(tmp_path))
    assert recorder.stats_date(datetime(2026, 8, 23, 3, 59, tzinfo=TZ8)).isoformat() == "2026-08-22"
    assert recorder.stats_date(datetime(2026, 8, 23, 4, 0, tzinfo=TZ8)).isoformat() == "2026-08-23"


def test_farming_limit_matches_primary_target_policy(tmp_path):
    fixed = datetime(2026, 8, 23, 12, 0, tzinfo=TZ8)
    recorder = FarmingStatsRecorder(_config(tmp_path), now=lambda: fixed, log=lambda _m: None)
    recorder.record(
        FarmingSession(True, normal_mob_count=5, elite_mob_count=2),
        FarmingRouteInfo(project_name="seed.json"),
    )
    elite = FarmingSession(
        True, normal_mob_count=3, elite_mob_count=1, primary_target="elite"
    )
    decision = recorder.check_limit(elite)
    assert decision.skip is True
    assert "精英超上限:2/2" in decision.message
    assert "脚本主目标为精英" in decision.message

    normal = FarmingSession(
        True, normal_mob_count=3, elite_mob_count=1, primary_target="normal"
    )
    assert recorder.check_limit(normal).skip is False
    disabled = FarmingSession(
        True, normal_mob_count=0, elite_mob_count=0, primary_target="disable"
    )
    assert recorder.check_limit(disabled).skip is False


def test_farming_miyoushe_totals_only_add_records_after_cutoff():
    cutoff = datetime(2026, 8, 23, 10, 0, tzinfo=TZ8)
    data = DailyFarmingData(
        records=[
            FarmingRecord(elite_mob_count=8, normal_mob_count=20,
                          timestamp=cutoff - timedelta(minutes=1)),
            FarmingRecord(elite_mob_count=2, normal_mob_count=5,
                          timestamp=cutoff + timedelta(minutes=1)),
        ],
        miyoushe_total_normal_mob_count=100,
        miyoushe_total_elite_mob_count=30,
        travels_diary_detail_manager_update_time=cutoff,
    )
    assert data.final_totals(TZ8) == (32, 105)


def test_pathing_executor_skips_route_at_farming_limit(tmp_path):
    from bgi_touch.pathing.executor import PathingExecutor

    config = _config(tmp_path, enabled=True, elite_cap=0, mob_cap=2000)
    ctx = SimpleNamespace(input=Mock(), sleep=lambda _ms: None)
    task = PathingTask(
        "精英路线", "Teyvat", [],
        farming_info={
            "allow_farming_count": True,
            "normal_mob_count": 1,
            "elite_mob_count": 1,
            "primary_target": "elite",
        },
    )
    logs = []
    assert PathingExecutor(
        ctx, positioner=object(), farming_config_path=config, log=logs.append,
    ).run(task)
    assert any("精英超上限" in line and "跳过此任务" in line for line in logs)
    ctx.input.release_all.assert_not_called()


def test_pathing_executor_records_successful_farming_route(tmp_path):
    from bgi_touch.pathing.executor import PathingExecutor

    fixed = datetime(2026, 8, 23, 12, 0, tzinfo=TZ8)
    recorder = FarmingStatsRecorder(
        _config(tmp_path, enabled=False), now=lambda: fixed, log=lambda _m: None
    )
    ctx = SimpleNamespace(input=Mock(), sleep=lambda _ms: None)
    task = PathingTask(
        "锄地一号", "Teyvat", [],
        farming_info={
            "allow_farming_count": True,
            "normal_mob_count": 9,
            "elite_mob_count": 2,
        },
        source_path=str(tmp_path / "敌人与魔物" / "锄地一号.json"),
    )
    assert PathingExecutor(
        ctx, positioner=object(), farming_recorder=recorder, log=lambda _m: None,
    ).run(task)
    data = recorder.read_daily_data()
    assert (data.total_elite_mob_count, data.total_normal_mob_count) == (2, 9)
    assert data.records[0].project_name == "锄地一号.json"
    assert data.records[0].folder_name == "敌人与魔物"
    ctx.input.release_all.assert_called_once()


class _FakeTravelDiaryUpdater:
    def __init__(self, store):
        self.store = store
        self.calls = []

    def update(self, cookie, *, role_index=0):
        self.calls.append((cookie, role_index))
        return TravelDiaryUpdate(
            GameInfo("hk4e_cn", "cn_gf01", "100000001"), (), ()
        )


def test_miyoushe_update_requires_explicit_cookie_and_projects_diary(tmp_path):
    config = _config(tmp_path, enabled=True, miyoushe_enabled=True)
    store = TravelDiaryStore(tmp_path / "diary")
    store.write("100000001", 2026, 8, DiaryPage(items=[
        ActionItem(37, "", "2026-08-23 10:00:00", 100),
        ActionItem(37, "", "2026-08-23 11:00:00", 1200),
        ActionItem(28, "", "2026-08-23 12:00:00", 10),
    ]))
    updater = _FakeTravelDiaryUpdater(store)
    recorder = FarmingStatsRecorder(
        config,
        now=lambda: datetime(2026, 8, 23, 12, tzinfo=TZ8),
        log=lambda _message: None,
        travel_diary_updater=updater,
        cookie_provider=lambda: "ltoken=secret",
    )
    assert recorder.update_miyoushe_data() is True
    assert updater.calls == [("ltoken=secret", 0)]
    data = recorder.read_daily_data()
    assert data.miyoushe_total_elite_mob_count == 2
    assert data.miyoushe_total_normal_mob_count == 1
    assert data.travels_diary_detail_manager_update_time == datetime(
        2026, 8, 23, 12, tzinfo=TZ8
    )
    assert "secret" not in recorder.data_path().read_text(encoding="utf-8")


def test_miyoushe_update_does_not_call_updater_without_cookie(tmp_path):
    config = _config(tmp_path, enabled=True, miyoushe_enabled=True)
    updater = _FakeTravelDiaryUpdater(TravelDiaryStore(tmp_path / "diary"))
    recorder = FarmingStatsRecorder(
        config,
        now=lambda: datetime(2026, 8, 23, 12, tzinfo=TZ8),
        log=lambda _message: None,
        travel_diary_updater=updater,
        cookie_provider=lambda: "",
    )
    assert recorder.update_miyoushe_data() is False
    assert recorder.maybe_update_miyoushe() is False
    assert updater.calls == []
