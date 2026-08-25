import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bgi_touch.tasks.travel_diary import (
    ActionItem,
    DiaryPage,
    GameInfo,
    MoraStatistics,
    NoLoginError,
    TravelDiaryError,
    TravelDiaryStore,
    TravelDiaryUpdate,
    TravelDiaryUpdater,
    YsClient,
    current_and_previous_months,
    diary_months_for_day,
    load_today_action_items,
    parse_diary_time,
)


TZ8 = timezone(timedelta(hours=8))


def _item(action_id, stamp, num, action=""):
    return ActionItem(action_id, action, stamp, num)


def _payload(items=None, *, message="", retcode=0, **data):
    return {
        "retcode": retcode,
        "message": message,
        "data": {
            "uid": 10001,
            "region": "os_usa",
            "account_id": 20002,
            "nickname": "旅行者",
            "date": "2026-08-25",
            "month": 8,
            "optional_month": [8, 7],
            "data_month": 8,
            "page": 1,
            "list": [item.to_dict() for item in (items or [])],
            **data,
        },
    }


class _Response:
    status = 200

    def __init__(self, value):
        self.value = value
        self.closed = False

    def read(self):
        if isinstance(self.value, bytes):
            return self.value
        return json.dumps(self.value, ensure_ascii=False).encode("utf-8")

    def close(self):
        self.closed = True


class _Opener:
    def __init__(self, *values):
        self.values = list(values)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        return _Response(self.values.pop(0))


def test_secret2_and_signed_request_headers_are_stable():
    opener = _Opener({"retcode": 0, "message": "", "data": {"list": [
        {
            "game_biz": "hk4e_cn",
            "region": "cn_gf01",
            "game_uid": "100000001",
            "is_chosen": "false",
            "is_official": 0,
        },
    ]}})
    client = YsClient(
        opener=opener,
        clock=lambda: 1_700_000_000,
        rand=lambda _low, _high: 123456,
    )
    url = "https://example.test/path?b=2&a=1"
    raw = "salt=xV8v4Qu54lUKrEYFZkJhB8cuOh9Asafs&t=1700000000&r=123456&b=&q=a=1&b=2"
    expected = "1700000000,123456," + hashlib.md5(raw.encode()).hexdigest()
    assert client.create_secret2(url) == expected

    roles = client.get_game_roles("ltoken=secret")
    assert roles[0].game_uid == "100000001"
    assert roles[0].is_chosen is False
    headers = {key.casefold(): value for key, value in opener.requests[0][0].header_items()}
    assert headers["cookie"] == "ltoken=secret"
    role_request_secret = client.create_secret2(client.ROLES_URL)
    assert headers["ds"] == role_request_secret
    assert headers["x-rpc-client_type".casefold()] == "5"
    assert opener.requests[0][1] == 30.0


@pytest.mark.parametrize(
    "value, expected",
    [
        (b"not-json", "米游社响应不是有效 JSON"),
        ({"message": "ok", "data": {}}, "米游社响应缺少 retcode"),
    ],
)
def test_request_rejects_malformed_payloads(value, expected):
    client = YsClient(opener=_Opener(value))
    with pytest.raises(TravelDiaryError, match=expected):
        client._request("https://example.test", "cookie")


def test_request_reports_expired_login_and_api_errors():
    with pytest.raises(NoLoginError):
        YsClient(opener=_Opener({"retcode": -1, "message": "未登录", "data": {}}))._request(
            "https://example.test", "cookie"
        )
    with pytest.raises(TravelDiaryError, match="retcode=-123"):
        YsClient(opener=_Opener({"retcode": -123, "message": "拒绝访问", "data": {}}))._request(
            "https://example.test", "cookie"
        )


def test_diary_page_handles_optional_fields_and_bool_values():
    page = DiaryPage.from_payload({
        "retcode": 0,
        "message": "",
        "data": {
            "page": True,
            "optional_month": "8",
            "list": [{"action_id": 37, "num": True, "time": None}],
        },
    })
    assert page.page == 1
    assert page.optional_month == []
    assert page.items[0].num == 0
    assert parse_diary_time("2026/08/25 04:00:00", "Asia/Shanghai") == datetime(
        2026, 8, 25, 4, tzinfo=TZ8
    )
    assert parse_diary_time("not-a-time") is None


class _PagedClient(YsClient):
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get_travels_diary_detail_page(self, role, cookie, month, *, kind=2, page=1, limit=100):
        self.calls.append(page)
        return self.pages[page - 1] if page <= len(self.pages) else DiaryPage()


def test_detail_pagination_stops_at_incremental_cutoff():
    role = GameInfo("hk4e_cn", "cn_gf01", "100000001")
    client = _PagedClient([
        DiaryPage(items=[
            _item(37, "2026-08-25 11:00:00", 100),
            _item(37, "2026-08-25 10:00:00", 100),
        ]),
        DiaryPage(items=[_item(37, "2026-08-25 09:00:00", 100)]),
    ])
    result = client.get_travels_diary_detail(
        role,
        "cookie",
        8,
        limit=2,
        last_action=_item(37, "2026-08-25 10:00:00", 100),
    )
    assert client.calls == [1]
    assert [item.time for item in result.items] == ["2026-08-25 11:00:00"]


def test_detail_filters_cutoff_even_when_first_page_is_short():
    role = GameInfo("hk4e_cn", "cn_gf01", "100000001")
    client = _PagedClient([DiaryPage(items=[
        _item(37, "2026-08-25 11:00:00", 100),
        _item(37, "2026-08-25 09:00:00", 100),
    ])])
    result = client.get_travels_diary_detail(
        role,
        "cookie",
        8,
        limit=100,
        last_action=_item(37, "2026-08-25 10:00:00", 100),
    )
    assert client.calls == [1]
    assert [item.time for item in result.items] == ["2026-08-25 11:00:00"]


def test_store_round_trip_corrupt_cache_and_stat_race(tmp_path):
    store = TravelDiaryStore(tmp_path)
    page = DiaryPage(month=8, items=[_item(37, "2026-08-25 10:00:00", 200)])
    path = store.write("100000001", 2026, 8, page)
    assert store.read("100000001", 2026, 8).items[0].num == 200
    path.write_text("{broken", encoding="utf-8")
    assert store.read("100000001", 2026, 8) is None

    class _BrokenPath:
        def is_file(self):
            return True

        def stat(self):
            raise OSError("file disappeared")

    assert store.modified_in_month(_BrokenPath(), datetime(2026, 8, 25, tzinfo=TZ8)) is False


def test_month_windows_and_four_am_boundary():
    january = datetime(2026, 1, 15, 12, tzinfo=TZ8)
    assert current_and_previous_months(january, count=3, tz=TZ8) == [
        (2026, 1), (2025, 12), (2025, 11)
    ]
    assert diary_months_for_day(datetime(2026, 8, 1, 3, 59, tzinfo=TZ8), tz=TZ8) == [
        (2026, 7), (2026, 8)
    ]
    assert diary_months_for_day(datetime(2026, 8, 1, 4, tzinfo=TZ8), tz=TZ8) == [
        (2026, 8)
    ]


def test_today_action_items_uses_server_four_am_window(tmp_path):
    store = TravelDiaryStore(tmp_path)
    store.write("100000001", 2026, 7, DiaryPage(items=[
        _item(37, "2026-07-31 03:59:00", 100),
        _item(37, "2026-07-31 04:00:00", 200),
    ]))
    store.write("100000001", 2026, 8, DiaryPage(items=[
        _item(28, "2026-08-01 03:30:00", 10),
    ]))
    items = load_today_action_items(
        store,
        "100000001",
        now=datetime(2026, 8, 1, 3, 45, tzinfo=TZ8),
        tz=TZ8,
    )
    assert [item.time for item in items] == [
        "2026-07-31 04:00:00", "2026-08-01 03:30:00"
    ]


def test_mora_statistics_matches_bettergi_counters():
    stats = MoraStatistics((
        _item(37, "2026-08-25 10:00:00", 100),
        _item(37, "2026-08-25 11:00:00", 200),
        _item(37, "2026-08-25 12:00:00", 1200),
        _item(37, "2026-08-25 13:00:00", 3000),
        _item(28, "2026-08-25 14:00:00", 50),
        _item(39, "2026-08-25 15:00:00", 60),
    ))
    assert stats.elite_statistics == 3
    assert stats.elite_game_statistics == 6
    assert stats.elite_mora == 4400
    assert stats.small_monster_statistics == 1
    assert stats.small_monster_mora == 100
    assert stats.total_mora_killing_monsters == 4500
    assert stats.other_mora == 110
    assert stats.all_mora == 4610
    assert stats.elite_details == "200*1, 1200*1, 3000*1"
    assert stats.small_monster_details == "10*1"
    assert stats.emergency_bonus == "50(1/10)"
    assert stats.chest_reward == "60(1/10)"
    assert stats.last_elite_time == "2026-08-25 13:00:00"


class _UpdaterClient:
    def __init__(self):
        self.calls = []

    def get_game_roles(self, cookie):
        self.calls.append(("roles", cookie))
        return [GameInfo("hk4e_cn", "cn_gf01", "100000001")]

    def get_travels_diary_detail(self, role, cookie, month, **kwargs):
        self.calls.append((month, kwargs.get("last_action")))
        return DiaryPage(items=[_item(37, "2026-08-24 10:00:00", 100)])


def test_updater_reuses_oldest_and_recently_modified_previous_cache(tmp_path):
    store = TravelDiaryStore(tmp_path)
    for year, month, stamp in ((2026, 6, "2026-06-20 10:00:00"),
                               (2026, 7, "2026-07-20 10:00:00"),
                               (2026, 8, "2026-08-20 10:00:00")):
        path = store.write("100000001", year, month, DiaryPage(items=[
            _item(37, stamp, 100),
        ]))
        if month == 7:
            modified = datetime(2026, 8, 1, 0, 0, tzinfo=TZ8).timestamp()
            os.utime(path, (modified, modified))

    client = _UpdaterClient()
    updater = TravelDiaryUpdater(
        client=client,
        store=store,
        now=lambda: datetime(2026, 8, 25, 12, tzinfo=TZ8),
        tz=TZ8,
        log=lambda _message: None,
    )
    result = updater.update("cookie")
    assert result.updated_months == ((2026, 8),)
    assert result.reused_months == ((2026, 6), (2026, 7))
    assert [item[0] for item in client.calls] == ["roles", 8]
    assert client.calls[1][1].time == "2026-08-20 10:00:00"
    cached = store.read("100000001", 2026, 8)
    assert [item.time for item in cached.items] == [
        "2026-08-24 10:00:00", "2026-08-20 10:00:00"
    ]
