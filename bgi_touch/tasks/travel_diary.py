"""米游社旅行札记数据与锄地统计。

BetterGI 的桌面版把旅行札记请求、最近三个月缓存和凌晨 4 点统计窗口
放在 ``TravelsDiaryDetailManager``/``YsClient`` 中。本模块保留相同的数据
契约，但使用同步标准库 HTTP，便于 CLI、锄地限制和离线测试复用；它不会
创建 GameContext，也不会触碰 DeviceHub 截图通道。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIARY_ROOT = PROJECT_ROOT / "log" / "logparse"
DEFAULT_SERVER_TIMEZONE = "Asia/Shanghai"
DETAIL_ACTION_IDS = frozenset({37, 28, 39})


class TravelDiaryError(RuntimeError):
    """旅行札记 HTTP/API 响应错误。"""


class NoLoginError(TravelDiaryError):
    """米游社 Cookie 已失效或未登录。"""


def _value(raw: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    folded = {
        str(key).replace("_", "").casefold(): value
        for key, value in raw.items()
    }
    for name in names:
        key = name.replace("_", "").casefold()
        if key in folded:
            return folded[key]
    return default


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = False) -> bool:
    """Decode the API's bool/0/1/string variants without ``bool('false')``."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on", "是"}:
            return True
        if normalized in {"false", "0", "no", "off", "否", ""}:
            return False
    return default


def _text(value: Any, default: str = "") -> str:
    if value is None or isinstance(value, (Mapping, list, tuple, set)):
        return default
    return str(value)


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    return [_int(item) for item in value if not isinstance(item, bool)]


def _server_tz(value: str | int | float | timezone | None = None):
    if isinstance(value, timezone):
        return value
    if isinstance(value, (int, float)):
        return timezone(timedelta(hours=float(value)))
    name = str(value or DEFAULT_SERVER_TIMEZONE)
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        try:
            return timezone(timedelta(hours=float(name)))
        except ValueError:
            return timezone(timedelta(hours=8))


def parse_diary_time(value: Any, tz=None) -> datetime | None:
    """Parse the API's ISO/space-separated timestamps as server-local time."""

    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        parsed = None
        for candidate in (text, text.replace("/", "-")):
            try:
                parsed = datetime.fromisoformat(candidate)
                break
            except ValueError:
                continue
        if parsed is None:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
        if parsed is None:
            return None
    zone = _server_tz(tz)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=zone)
    return parsed.astimezone(zone)


@dataclass(frozen=True)
class GameInfo:
    game_biz: str
    region: str
    game_uid: str
    nickname: str = ""
    level: int = 0
    is_chosen: bool = False
    region_name: str = ""
    is_official: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GameInfo":
        return cls(
            game_biz=_text(_value(raw, "game_biz", "gameBiz", default="")),
            region=_text(_value(raw, "region", default="")),
            game_uid=_text(_value(raw, "game_uid", "gameUid", default="")),
            nickname=_text(_value(raw, "nickname", default="")),
            level=_int(_value(raw, "level", default=0)),
            is_chosen=_bool(_value(raw, "is_chosen", "isChosen", default=False)),
            region_name=_text(_value(raw, "region_name", "regionName", default="")),
            is_official=_bool(_value(raw, "is_official", "isOfficial", default=False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_biz": self.game_biz,
            "region": self.region,
            "game_uid": self.game_uid,
            "nickname": self.nickname,
            "level": self.level,
            "is_chosen": self.is_chosen,
            "region_name": self.region_name,
            "is_official": self.is_official,
        }


@dataclass(frozen=True)
class ActionItem:
    action_id: int
    action: str
    time: str
    num: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ActionItem":
        return cls(
            action_id=_int(_value(raw, "action_id", "actionId", default=0)),
            action=_text(_value(raw, "action", default="")),
            time=_text(_value(raw, "time", default="")),
            num=_int(_value(raw, "num", default=0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action": self.action,
            "time": self.time,
            "num": self.num,
        }

    @property
    def parsed_time(self) -> datetime | None:
        return parse_diary_time(self.time)

    def key(self) -> tuple[int, str, int, str]:
        return self.action_id, self.time, self.num, self.action


@dataclass
class DiaryPage:
    """Decoded ``monthDetail`` response, retaining API metadata."""

    retcode: int = 0
    message: str = ""
    uid: int = 0
    region: str = ""
    account_id: int = 0
    nickname: str = ""
    date: str = ""
    month: int = 0
    optional_month: list[int] = field(default_factory=list)
    data_month: int = 0
    page: int = 1
    items: list[ActionItem] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DiaryPage":
        if not isinstance(payload, Mapping):
            raise TypeError("旅行札记响应必须是对象")
        data = payload.get("data") or {}
        if not isinstance(data, Mapping):
            data = {}
        raw_items = data.get("list", [])
        if not isinstance(raw_items, list):
            raw_items = []
        return cls(
            retcode=_int(payload.get("retcode", 0)),
            message=_text(payload.get("message", "")),
            uid=_int(_value(data, "uid", default=0)),
            region=_text(_value(data, "region", default="")),
            account_id=_int(_value(data, "account_id", "accountId", default=0)),
            nickname=_text(_value(data, "nickname", default="")),
            date=_text(_value(data, "date", default="")),
            month=_int(_value(data, "month", default=0)),
            optional_month=_int_list(_value(
                data, "optional_month", "optionalMonth", default=[]
            )),
            data_month=_int(_value(data, "data_month", "dataMonth", default=0)),
            page=_int(_value(data, "page", default=1), 1),
            items=[ActionItem.from_mapping(item) for item in raw_items if isinstance(item, Mapping)],
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "retcode": self.retcode,
            "message": self.message,
            "data": {
                "uid": self.uid,
                "region": self.region,
                "account_id": self.account_id,
                "nickname": self.nickname,
                "date": self.date,
                "month": self.month,
                "optional_month": list(self.optional_month),
                "data_month": self.data_month,
                "page": self.page,
                "list": [item.to_dict() for item in self.items],
            },
        }


class YsClient:
    """Small synchronous client for the two public travel-diary endpoints."""

    APP_VERSION = "2.71.1"
    API_SALT_2 = "xV8v4Qu54lUKrEYFZkJhB8cuOh9Asafs"
    ROLES_URL = (
        "https://api-takumi.mihoyo.com/binding/api/"
        "getUserGameRolesByCookie?game_biz=hk4e_cn"
    )
    DIARY_BASE_URL = "https://hk4e-api.mihoyo.com/event/ys_ledger"

    def __init__(
        self,
        *,
        opener: Callable[..., Any] = urlopen,
        timeout_s: float = 30.0,
        clock: Callable[[], float] = time.time,
        rand: Callable[[int, int], int] | None = None,
        user_agent: str | None = None,
    ):
        self.opener = opener
        self.timeout_s = float(timeout_s)
        self.clock = clock
        self.rand = rand or random.SystemRandom().randint
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Linux; Android 13; Pixel 5 Build/TQ3A.230901.001; wv) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/118.0.0.0 "
            f"Mobile Safari/537.36 miHoYoBBS/{self.APP_VERSION}"
        )

    def create_secret2(self, url: str) -> str:
        timestamp = int(self.clock())
        random_value = str(self.rand(100000, 200000))
        query = urlsplit(url).query
        normalized_query = "&".join(sorted(query.split("&"))) if query else ""
        raw = (
            f"salt={self.API_SALT_2}&t={timestamp}&r={random_value}"
            f"&b=&q={normalized_query}"
        )
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return f"{timestamp},{random_value},{digest}"

    def _request(self, url: str, cookie: str, *, signed: bool = False) -> Mapping[str, Any]:
        cookie = str(cookie or "").strip()
        if not cookie:
            raise ValueError("米游社 Cookie 不能为空")
        headers = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
            "Cookie": cookie,
            "Referer": "https://webstatic.mihoyo.com/",
            "X-Requested-With": "com.mihoyo.hyperion",
            "x-rpc-client_type": "5",
        }
        if signed:
            headers["DS"] = self.create_secret2(url)
            headers["x-rpc-app_version"] = self.APP_VERSION
        request = Request(url, headers=headers, method="GET")
        try:
            response = self.opener(request, timeout=self.timeout_s)
            try:
                status = getattr(response, "status", None)
                if isinstance(status, int) and status >= 400:
                    raise TravelDiaryError(f"米游社 HTTP 请求失败：{status}")
                raw = response.read()
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        except TravelDiaryError:
            raise
        except Exception as error:
            raise TravelDiaryError(f"米游社请求失败：{error}") from error
        try:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except (TypeError, ValueError) as error:
            raise TravelDiaryError("米游社响应不是有效 JSON") from error
        if not isinstance(payload, Mapping):
            raise TravelDiaryError("米游社响应根节点不是对象")
        message = _text(payload.get("message", ""))
        if message == "未登录" or "未登录" in message:
            raise NoLoginError("米游社 Cookie 未登录或已过期")
        if "retcode" not in payload:
            raise TravelDiaryError("米游社响应缺少 retcode")
        retcode = _int(payload.get("retcode", 0))
        if retcode != 0:
            raise TravelDiaryError(f"米游社接口错误：retcode={retcode}，{message}")
        return payload

    def get_game_roles(self, cookie: str) -> list[GameInfo]:
        payload = self._request(self.ROLES_URL, cookie, signed=True)
        data = payload.get("data") or {}
        raw_list = data.get("list", []) if isinstance(data, Mapping) else []
        if not isinstance(raw_list, list):
            raw_list = []
        roles = [
            role for item in raw_list
            if isinstance(item, Mapping)
            for role in [GameInfo.from_mapping(item)]
            if role.game_uid and role.region
        ]
        if not roles:
            raise TravelDiaryError("米游社未返回原神账号角色")
        return roles

    def _diary_url(self, role: GameInfo, *, month: int, page: int, limit: int, kind: int) -> str:
        return (
            f"{self.DIARY_BASE_URL}/monthDetail?page={int(page)}&month={int(month)}"
            f"&limit={int(limit)}&type={int(kind)}&bind_uid={role.game_uid}"
            f"&bind_region={role.region}&bbs_presentation_style=fullscreen"
            "&bbs_auth_required=true&utm_source=bbs&utm_medium=mys&utm_campaign=icon"
        )

    def get_travels_diary_detail_page(
        self,
        role: GameInfo,
        cookie: str,
        month: int,
        *,
        kind: int = 2,
        page: int = 1,
        limit: int = 100,
    ) -> DiaryPage:
        if not 1 <= int(page):
            raise ValueError("旅行札记页码必须从 1 开始")
        if not 1 <= int(limit) <= 100:
            raise ValueError("旅行札记每页数量必须在 1-100 之间")
        payload = self._request(
            self._diary_url(role, month=month, page=page, limit=limit, kind=kind),
            cookie,
        )
        return DiaryPage.from_payload(payload)

    def get_travels_diary_detail(
        self,
        role: GameInfo,
        cookie: str,
        month: int,
        *,
        kind: int = 2,
        limit: int = 100,
        last_action: ActionItem | None = None,
        max_pages: int = 100,
    ) -> DiaryPage:
        first = self.get_travels_diary_detail_page(
            role, cookie, month, kind=kind, page=1, limit=limit,
        )
        combined = list(first.items)
        cutoff = parse_diary_time(last_action.time) if last_action else None

        def reached_cutoff(items: list[ActionItem]) -> bool:
            if cutoff is None or not items:
                return False
            timestamps = [
                stamp for item in items
                if (stamp := parse_diary_time(item.time)) is not None
            ]
            return bool(timestamps) and min(timestamps) <= cutoff

        def after_cutoff(items: list[ActionItem]) -> list[ActionItem]:
            if cutoff is None:
                return items
            return [
                item for item in items
                if (stamp := parse_diary_time(item.time)) is None or stamp > cutoff
            ]

        if reached_cutoff(combined):
            combined = after_cutoff(combined)
            first.items = combined
            return first
        if len(first.items) < limit:
            first.items = after_cutoff(combined)
            return first

        for page in range(2, max_pages + 1):
            current = self.get_travels_diary_detail_page(
                role, cookie, month, kind=kind, page=page, limit=limit,
            )
            if not current.items:
                break
            combined.extend(current.items)
            if reached_cutoff(combined):
                break
            if len(current.items) < limit:
                break
        if cutoff is not None:
            combined = after_cutoff(combined)
        first.items = combined
        return first


class TravelDiaryStore:
    """JSON cache compatible with BetterGI's ``year_month.json`` files."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or DEFAULT_DIARY_ROOT).expanduser().resolve()

    def path(self, game_uid: str, year: int, month: int) -> Path:
        return self.root / str(game_uid) / "travelsdiarydetail" / f"{int(year)}_{int(month)}.json"

    def read(self, game_uid: str, year: int, month: int) -> DiaryPage | None:
        try:
            path = self.path(game_uid, year, month)
            if not path.is_file():
                return None
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            return DiaryPage.from_payload(payload) if isinstance(payload, Mapping) else None
        except (OSError, UnicodeError, ValueError, TypeError, OverflowError):
            return None

    def write(self, game_uid: str, year: int, month: int, page: DiaryPage) -> Path:
        path = self.path(game_uid, year, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(page.to_payload(), ensure_ascii=False, indent=2) + "\n"
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        finally:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        return path

    def modified_in_month(self, path: Path, now: datetime) -> bool:
        try:
            if not path.is_file():
                return False
            zone = now.tzinfo or timezone.utc
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=zone)
            start = now.astimezone(zone).replace(
                day=1, hour=0, minute=0, second=0, microsecond=0,
            )
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)
            return start <= modified < end
        except (OSError, OverflowError, ValueError):
            return False


def current_and_previous_months(
    now: datetime | None = None,
    *,
    count: int = 3,
    tz: str | int | float | timezone | None = None,
) -> list[tuple[int, int]]:
    if count < 1:
        return []
    zone = _server_tz(tz)
    current = (now or datetime.now(zone))
    if current.tzinfo is None:
        current = current.replace(tzinfo=zone)
    current = current.astimezone(zone)
    year, month = current.year, current.month
    result: list[tuple[int, int]] = []
    for offset in range(count):
        value = month - offset
        result.append((year + (value - 1) // 12, (value - 1) % 12 + 1))
    return result


def diary_months_for_day(now: datetime | None = None, *, tz=None) -> list[tuple[int, int]]:
    zone = _server_tz(tz)
    current = (now or datetime.now(zone))
    if current.tzinfo is None:
        current = current.replace(tzinfo=zone)
    current = current.astimezone(zone)
    months = [(current.year, current.month)]
    if current.day == 1 and current.hour < 4:
        months.insert(0, current_and_previous_months(current, count=2, tz=zone)[1])
    return months


def load_action_items(
    store: TravelDiaryStore,
    game_uid: str,
    months: Iterable[tuple[int, int]],
    *,
    action_ids: Iterable[int] = DETAIL_ACTION_IDS,
    tz=None,
) -> list[ActionItem]:
    allowed = {
        int(value) for value in action_ids
        if not isinstance(value, bool)
    }
    items: list[ActionItem] = []
    seen: set[tuple[int, str, int, str]] = set()
    for year, month in months:
        page = store.read(game_uid, year, month)
        if page is not None:
            for item in page.items:
                if item.action_id in allowed and item.key() not in seen:
                    items.append(item)
                    seen.add(item.key())
    zone = _server_tz(tz)
    return sorted(
        items,
        key=lambda item: parse_diary_time(item.time, zone) or datetime.min.replace(tzinfo=zone),
    )


def load_today_action_items(
    store: TravelDiaryStore,
    game_uid: str,
    *,
    now: datetime | None = None,
    tz=None,
) -> list[ActionItem]:
    zone = _server_tz(tz)
    current = now or datetime.now(zone)
    if current.tzinfo is None:
        current = current.replace(tzinfo=zone)
    current = current.astimezone(zone)
    start = current.replace(hour=4, minute=0, second=0, microsecond=0)
    if current < start:
        start -= timedelta(days=1)
    end = start + timedelta(days=1)
    return [
        item for item in load_action_items(
            store, game_uid, diary_months_for_day(current, tz=zone), tz=zone,
        )
        if (stamp := parse_diary_time(item.time, zone)) is not None and start <= stamp < end
    ]


@dataclass(frozen=True)
class MoraStatistics:
    """Travel-diary rewards projected to BetterGI's farming counters."""

    action_items: tuple[ActionItem, ...] = ()

    @property
    def monster_action_items(self) -> list[ActionItem]:
        return [item for item in self.action_items if item.action_id == 37]

    @property
    def elite_monster_action_items(self) -> list[ActionItem]:
        return [item for item in self.monster_action_items if item.num >= 200]

    @property
    def small_monster_action_items(self) -> list[ActionItem]:
        return [item for item in self.monster_action_items if item.num < 200]

    @property
    def emergency_bonus(self) -> str:
        items = [item for item in self.action_items if item.action_id == 28]
        if not items:
            return ""
        suffix = "" if len(items) >= 10 else f"({len(items)}/10)"
        return f"{sum(item.num for item in items)}{suffix}"

    @property
    def chest_reward(self) -> str:
        items = [item for item in self.action_items if item.action_id == 39]
        if not items:
            return ""
        suffix = "" if len(items) >= 10 else f"({len(items)}/10)"
        return f"{sum(item.num for item in items)}{suffix}"

    @property
    def elite_statistics(self) -> int:
        return len(self.elite_monster_action_items)

    @property
    def elite_game_statistics(self) -> int:
        return sum(3 if item.num >= 3000 else 2 if item.num >= 1200 else 1
                   for item in self.elite_monster_action_items)

    @property
    def elite_mora(self) -> int:
        return sum(item.num for item in self.elite_monster_action_items)

    @property
    def small_monster_statistics(self) -> int:
        return len(self.small_monster_action_items)

    @property
    def small_monster_mora(self) -> int:
        return sum(item.num for item in self.small_monster_action_items)

    @property
    def total_mora_killing_monsters(self) -> int:
        return sum(item.num for item in self.monster_action_items)

    @property
    def elite_details(self) -> str:
        groups: dict[int, int] = {}
        for item in self.elite_monster_action_items:
            groups[item.num] = groups.get(item.num, 0) + 1
        return ", ".join(f"{num}*{count}" for num, count in sorted(groups.items()))

    @property
    def small_monster_details(self) -> str:
        groups: dict[int, int] = {}
        for item in self.small_monster_action_items:
            groups[item.num // 10] = groups.get(item.num // 10, 0) + 1
        return ", ".join(f"{num}*{count}" for num, count in sorted(groups.items()))

    @property
    def other_mora(self) -> int:
        return sum(item.num for item in self.action_items if item.action_id != 37)

    @property
    def all_mora(self) -> int:
        return sum(item.num for item in self.action_items)

    @property
    def last_elite_time(self) -> str | None:
        values = [item for item in self.elite_monster_action_items if item.parsed_time]
        return max(values, key=lambda item: item.parsed_time).time if values else None

    @property
    def last_small_time(self) -> str | None:
        values = [item for item in self.small_monster_action_items if item.parsed_time]
        return max(values, key=lambda item: item.parsed_time).time if values else None

    def get_filter(self, predicate: Callable[[ActionItem], bool]) -> "MoraStatistics":
        return MoraStatistics(tuple(item for item in self.action_items if predicate(item)))


@dataclass(frozen=True)
class TravelDiaryUpdate:
    game_info: GameInfo
    updated_months: tuple[tuple[int, int], ...]
    reused_months: tuple[tuple[int, int], ...]


class TravelDiaryUpdater:
    """Fetch/cache the current and previous two diary months."""

    def __init__(
        self,
        client: YsClient | None = None,
        store: TravelDiaryStore | None = None,
        *,
        now: Callable[[], datetime] | None = None,
        tz: str | int | float | timezone | None = None,
        log: Callable[[str], None] = print,
    ):
        self.client = client or YsClient()
        self.store = store or TravelDiaryStore()
        self.tz = _server_tz(tz)
        self.now = now or (lambda: datetime.now(self.tz))
        self.log = log

    def update(self, cookie: str, *, role_index: int = 0) -> TravelDiaryUpdate:
        roles = self.client.get_game_roles(cookie)
        if not 0 <= int(role_index) < len(roles):
            raise IndexError(f"米游社角色索引越界：{role_index}")
        role = roles[int(role_index)]
        current = self.now()
        if current.tzinfo is None:
            current = current.replace(tzinfo=self.tz)
        current = current.astimezone(self.tz)
        months = list(reversed(current_and_previous_months(current, count=3, tz=self.tz)))
        updated: list[tuple[int, int]] = []
        reused: list[tuple[int, int]] = []

        for index, (year, month) in enumerate(months):
            path = self.store.path(role.game_uid, year, month)
            cached = self.store.read(role.game_uid, year, month)
            if cached is not None:
                # BetterGI deliberately leaves the oldest cached month alone;
                # it cannot receive new records once it is outside the current
                # and previous month window.
                if index == 0:
                    reused.append((year, month))
                    continue
                if index == 1 and self.store.modified_in_month(path, current):
                    reused.append((year, month))
                    continue
                timestamps = [
                    item for item in cached.items
                    if item.parsed_time is not None
                ]
                last = max(timestamps, key=lambda item: item.parsed_time) if timestamps else None
                page = self.client.get_travels_diary_detail(
                    role, cookie, month, last_action=last,
                )
                page.items = self._merge_items(page.items, cached.items)
            else:
                page = self.client.get_travels_diary_detail(role, cookie, month)
            self.store.write(role.game_uid, year, month, page)
            updated.append((year, month))
            self.log(f"[travel-diary] 已更新 {year}_{month}")
        return TravelDiaryUpdate(role, tuple(updated), tuple(reused))

    @staticmethod
    def _merge_items(new_items: list[ActionItem], old_items: list[ActionItem]) -> list[ActionItem]:
        result: list[ActionItem] = []
        seen: set[tuple[int, str, int, str]] = set()
        for item in [*new_items, *old_items]:
            key = item.key()
            if key not in seen:
                result.append(item)
                seen.add(key)
        return result


def cookie_from_environment() -> str:
    """Read the opt-in cookie without ever echoing it in logs or reports."""

    return os.environ.get("BGI_MIYOUSHE_COOKIE", "").strip()
