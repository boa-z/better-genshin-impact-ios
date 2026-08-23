"""BetterGI farming-plan limits and daily pathing statistics.

BetterGI stores this metadata in each pathing JSON under ``farming_info``.
Limit checks are optional, but successful routes with counting enabled are
always recorded, matching ``ScriptService`` and ``ScriptGroupProject``.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "farming.json"


def _field(raw: Mapping, *names: str, default=None):
    for name in names:
        if name in raw:
            return raw[name]
    return default


def _bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in ("true", "1", "yes", "on", "是"):
            return True
        if normalized in ("false", "0", "no", "off", "否", ""):
            return False
    if value is None:
        return default
    return bool(value)


def _number(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _datetime(value, tz) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            parsed = datetime.min
    else:
        parsed = datetime.min
    if parsed == datetime.min:
        return parsed.replace(tzinfo=tz)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=tz)
    return parsed.astimezone(tz)


@dataclass(frozen=True)
class FarmingConfig:
    enabled: bool = False
    daily_elite_cap: float = 400
    daily_mob_cap: float = 2000
    log_directory: Path = PROJECT_ROOT / "log" / "FarmingPlan"
    server_timezone: str | int | float = "Asia/Shanghai"
    miyoushe_enabled: bool = False
    miyoushe_daily_elite_cap: float = 400
    miyoushe_daily_mob_cap: float = 2000

    @classmethod
    def load(cls, path: str | Path | None = None) -> "FarmingConfig":
        config_path = Path(
            path or os.getenv("BGI_FARMING_CONFIG") or DEFAULT_CONFIG
        ).expanduser()
        if not config_path.is_file():
            return cls()
        raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise ValueError("锄地规划配置根节点必须是对象")
        mys = _field(
            raw, "miyousheDataConfig", "miyoushe_data_config", default={}
        )
        if not isinstance(mys, dict):
            mys = {}
        raw_log = _field(
            raw, "logDirectory", "log_directory", default="../log/FarmingPlan"
        )
        log_directory = Path(str(raw_log)).expanduser()
        if not log_directory.is_absolute():
            log_directory = (config_path.parent / log_directory).resolve()
        return cls(
            enabled=_bool(_field(raw, "enabled", default=False)),
            daily_elite_cap=max(0, _number(_field(
                raw, "dailyEliteCap", "daily_elite_cap", default=400
            ), 400)),
            daily_mob_cap=max(0, _number(_field(
                raw, "dailyMobCap", "daily_mob_cap", default=2000
            ), 2000)),
            log_directory=log_directory,
            server_timezone=_field(
                raw, "serverTimezone", "server_timezone", default="Asia/Shanghai"
            ),
            miyoushe_enabled=_bool(_field(mys, "enabled", default=False)),
            miyoushe_daily_elite_cap=max(0, _number(_field(
                mys, "dailyEliteCap", "daily_elite_cap", default=400
            ), 400)),
            miyoushe_daily_mob_cap=max(0, _number(_field(
                mys, "dailyMobCap", "daily_mob_cap", default=2000
            ), 2000)),
        )

    def tzinfo(self):
        if isinstance(self.server_timezone, (int, float)):
            return timezone(timedelta(hours=float(self.server_timezone)))
        value = str(self.server_timezone or "Asia/Shanghai")
        try:
            return ZoneInfo(value)
        except ZoneInfoNotFoundError:
            try:
                return timezone(timedelta(hours=float(value)))
            except ValueError:
                return timezone(timedelta(hours=8))


@dataclass
class FarmingSession:
    allow_farming_count: bool = False
    normal_mob_count: float = 0
    elite_mob_count: float = 0
    primary_target: str = ""
    duration_seconds: float = 0
    elite_details: str = ""
    total_mora: float = 0

    @classmethod
    def from_mapping(cls, raw: Mapping | None) -> "FarmingSession":
        raw = raw if isinstance(raw, Mapping) else {}
        return cls(
            allow_farming_count=_bool(_field(
                raw, "allow_farming_count", "allowFarmingCount", default=False
            )),
            normal_mob_count=max(0, _number(_field(
                raw, "normal_mob_count", "normalMobCount", default=0
            ))),
            elite_mob_count=max(0, _number(_field(
                raw, "elite_mob_count", "eliteMobCount", default=0
            ))),
            primary_target=str(_field(
                raw, "primary_target", "primaryTarget", default=""
            ) or "").strip().casefold(),
            duration_seconds=max(0, _number(_field(
                raw, "duration_seconds", "durationSeconds", default=0
            ))),
            elite_details=str(_field(
                raw, "elite_details", "eliteDetails", default=""
            ) or ""),
            total_mora=max(0, _number(_field(
                raw, "total_mora", "totalMora", default=0
            ))),
        )


@dataclass(frozen=True)
class FarmingRouteInfo:
    group_name: str = ""
    project_name: str = ""
    folder_name: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping | None) -> "FarmingRouteInfo":
        raw = raw if isinstance(raw, Mapping) else {}
        return cls(
            group_name=str(_field(raw, "group_name", "groupName", default="") or ""),
            project_name=str(_field(
                raw, "project_name", "projectName", default=""
            ) or ""),
            folder_name=str(_field(
                raw, "folder_name", "folderName", default=""
            ) or ""),
        )


@dataclass
class FarmingRecord:
    group_name: str = ""
    project_name: str = ""
    folder_name: str = ""
    normal_mob_count: float = 0
    elite_mob_count: float = 0
    timestamp: datetime | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping, tz) -> "FarmingRecord":
        route = FarmingRouteInfo.from_mapping(raw)
        return cls(
            route.group_name,
            route.project_name,
            route.folder_name,
            max(0, _number(_field(raw, "normal_mob_count", "normalMobCount", default=0))),
            max(0, _number(_field(raw, "elite_mob_count", "eliteMobCount", default=0))),
            _datetime(_field(raw, "timestamp", default=""), tz),
        )

    def to_dict(self) -> dict:
        timestamp = self.timestamp or datetime.now(timezone.utc)
        return {
            "group_name": self.group_name,
            "project_name": self.project_name,
            "folder_name": self.folder_name,
            "normal_mob_count": self.normal_mob_count,
            "elite_mob_count": self.elite_mob_count,
            "timestamp": timestamp.isoformat(),
        }


@dataclass
class DailyFarmingData:
    total_normal_mob_count: float = 0
    total_elite_mob_count: float = 0
    records: list[FarmingRecord] = field(default_factory=list)
    miyoushe_total_normal_mob_count: float = 0
    miyoushe_total_elite_mob_count: float = 0
    last_miyoushe_update_time: datetime | None = None
    travels_diary_detail_manager_update_time: datetime | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping | None, tz) -> "DailyFarmingData":
        raw = raw if isinstance(raw, Mapping) else {}
        records = raw.get("records", [])
        if not isinstance(records, list):
            records = []
        return cls(
            total_normal_mob_count=max(0, _number(_field(
                raw, "total_normal_mob_count", "totalNormalMobCount", default=0
            ))),
            total_elite_mob_count=max(0, _number(_field(
                raw, "total_elite_mob_count", "totalEliteMobCount", default=0
            ))),
            records=[
                FarmingRecord.from_mapping(item, tz)
                for item in records if isinstance(item, Mapping)
            ],
            miyoushe_total_normal_mob_count=max(0, _number(_field(
                raw,
                "miyoushe_total_normal_mob_count", "miyousheTotalNormalMobCount",
                default=0,
            ))),
            miyoushe_total_elite_mob_count=max(0, _number(_field(
                raw,
                "miyoushe_total_elite_mob_count", "miyousheTotalEliteMobCount",
                default=0,
            ))),
            last_miyoushe_update_time=_datetime(_field(
                raw, "last_miyoushe_update_time", "lastMiyousheUpdateTime", default=""
            ), tz),
            travels_diary_detail_manager_update_time=_datetime(_field(
                raw,
                "travels_diary_detail_manager_update_time",
                "travelsDiaryDetailManagerUpdateTime",
                default="",
            ), tz),
        )

    def final_totals(self, tz) -> tuple[float, float]:
        if self.miyoushe_total_elite_mob_count + self.miyoushe_total_normal_mob_count <= 0:
            return self.total_elite_mob_count, self.total_normal_mob_count
        cutoff = self.travels_diary_detail_manager_update_time or datetime.min.replace(tzinfo=tz)
        elite = self.miyoushe_total_elite_mob_count
        normal = self.miyoushe_total_normal_mob_count
        for record in self.records:
            timestamp = record.timestamp or datetime.min.replace(tzinfo=tz)
            if timestamp > cutoff:
                elite += record.elite_mob_count
                normal += record.normal_mob_count
        return elite, normal

    def to_dict(self) -> dict:
        def dt(value: datetime | None) -> str:
            return value.isoformat() if value is not None else ""

        return {
            "total_normal_mob_count": self.total_normal_mob_count,
            "total_elite_mob_count": self.total_elite_mob_count,
            "records": [record.to_dict() for record in self.records],
            "miyoushe_total_normal_mob_count": self.miyoushe_total_normal_mob_count,
            "miyoushe_total_elite_mob_count": self.miyoushe_total_elite_mob_count,
            "last_miyoushe_update_time": dt(self.last_miyoushe_update_time),
            "travels_diary_detail_manager_update_time": dt(
                self.travels_diary_detail_manager_update_time
            ),
        }


@dataclass(frozen=True)
class FarmingLimitDecision:
    skip: bool
    message: str = ""


class FarmingStatsRecorder:
    _write_lock = threading.Lock()

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        now: Callable[[], datetime] | None = None,
        log: Callable[[str], None] = print,
    ):
        self.config = FarmingConfig.load(config_path)
        self.tz = self.config.tzinfo()
        self._now = now or (lambda: datetime.now(self.tz))
        self.log = log

    def current_time(self) -> datetime:
        value = self._now()
        if value.tzinfo is None:
            return value.replace(tzinfo=self.tz)
        return value.astimezone(self.tz)

    def stats_date(self, current: datetime | None = None):
        value = current or self.current_time()
        if value.tzinfo is None:
            value = value.replace(tzinfo=self.tz)
        return (value.astimezone(self.tz) - timedelta(hours=4)).date()

    def data_path(self, current: datetime | None = None) -> Path:
        return self.config.log_directory / f"{self.stats_date(current):%Y%m%d}.json"

    def read_daily_data(self, current: datetime | None = None) -> DailyFarmingData:
        path = self.data_path(current)
        if not path.is_file():
            return DailyFarmingData()
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            return DailyFarmingData.from_mapping(raw, self.tz)
        except (OSError, ValueError, TypeError):
            self.log(f"[farming] 每日统计损坏，重新创建：{path}")
            return DailyFarmingData()

    def final_caps(self, data: DailyFarmingData) -> tuple[float, float]:
        if data.miyoushe_total_elite_mob_count + data.miyoushe_total_normal_mob_count > 0:
            return (
                self.config.miyoushe_daily_elite_cap,
                self.config.miyoushe_daily_mob_cap,
            )
        return self.config.daily_elite_cap, self.config.daily_mob_cap

    def check_limit(self, session: FarmingSession) -> FarmingLimitDecision:
        if (
            not self.config.enabled
            or not session.allow_farming_count
            or session.primary_target == "disable"
        ):
            return FarmingLimitDecision(False)
        data = self.read_daily_data()
        total_elite, total_normal = data.final_totals(self.tz)
        elite_cap, normal_cap = self.final_caps(data)
        elite_over = total_elite >= elite_cap
        normal_over = total_normal >= normal_cap
        messages: list[str] = []
        if elite_over:
            messages.append(f"精英超上限:{total_elite:g}/{elite_cap:g}")
        if normal_over:
            messages.append(f"小怪超上限:{total_normal:g}/{normal_cap:g}")
        if elite_over and normal_over:
            return FarmingLimitDecision(True, ",".join(messages))
        if session.normal_mob_count == 0 and session.elite_mob_count == 0:
            messages.append("精英和小怪计数都为0，请确认配置")
            return FarmingLimitDecision(True, ",".join(messages))
        if (
            session.primary_target == "elite" and session.elite_mob_count == 0
        ) or (
            session.primary_target == "normal" and session.normal_mob_count == 0
        ):
            messages.append("主目标计数为0，请确认配置")
            return FarmingLimitDecision(True, ",".join(messages))

        skip = (
            session.primary_target == "elite" and elite_over
        ) or (
            session.primary_target == "normal" and normal_over
        ) or (
            elite_over and session.normal_mob_count == 0
        ) or (
            normal_over and session.elite_mob_count == 0
        )
        if session.primary_target == "elite" and elite_over and session.normal_mob_count > 0:
            messages.append("脚本主目标为精英")
        if session.primary_target == "normal" and normal_over and session.elite_mob_count > 0:
            messages.append("脚本主目标为小怪")
        return FarmingLimitDecision(skip, ",".join(messages))

    def _save(self, path: Path, data: DailyFarmingData) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data.to_dict(), ensure_ascii=False, indent=2) + "\n"
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            stream.write(payload)
            temporary = Path(stream.name)
        temporary.replace(path)

    def record(self, session: FarmingSession, route: FarmingRouteInfo) -> DailyFarmingData:
        now = self.current_time()
        path = self.data_path(now)
        with self._write_lock:
            data = self.read_daily_data(now)
            if session.allow_farming_count:
                data.total_normal_mob_count += session.normal_mob_count
                data.total_elite_mob_count += session.elite_mob_count
            data.records.append(FarmingRecord(
                route.group_name,
                route.project_name,
                route.folder_name,
                session.normal_mob_count if session.allow_farming_count else 0,
                session.elite_mob_count if session.allow_farming_count else 0,
                now,
            ))
            self._save(path, data)
        total_elite, total_normal = data.final_totals(self.tz)
        elite_cap, normal_cap = self.final_caps(data)
        suffix = "(合并米游社数据)" if (
            data.miyoushe_total_elite_mob_count
            + data.miyoushe_total_normal_mob_count > 0
        ) else ""
        self.log(
            f"[farming] 锄地进度:[小怪:{total_normal:g}/{normal_cap:g},"
            f"精英:{total_elite:g}/{elite_cap:g}]{suffix}"
        )
        return data
