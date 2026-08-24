"""BetterGI ScriptGroup completion records and skip policies."""

from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORD_DIR = PROJECT_ROOT / "log" / "ExecutionRecords"
SERVER_TIMEZONE = timezone(timedelta(hours=8))


def _value(raw: Mapping[str, Any], *names: str, default=None):
    folded = {str(key).replace("_", "").casefold(): value for key, value in raw.items()}
    for name in names:
        key = name.replace("_", "").casefold()
        if key in folded:
            return folded[key]
    return default


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return bool(value)


def _datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class CompletionSkipRule:
    enabled: bool = False
    policy: str = "GroupPhysicalPathSkipPolicy"
    boundary_hour: int = 4
    server_time_boundary: bool = False
    last_run_gap_s: int = -1
    reference_point: str = "EndTime"

    @classmethod
    def from_group_config(cls, group_config: Mapping[str, Any]) -> "CompletionSkipRule":
        pathing = _mapping(_value(group_config, "pathingConfig", default={}))
        raw = _mapping(_value(pathing, "taskCompletionSkipRuleConfig", default={}))
        try:
            boundary = int(_value(raw, "boundaryTime", default=4))
        except (TypeError, ValueError):
            boundary = 4
        try:
            gap = int(_value(raw, "lastRunGapSeconds", default=-1))
        except (TypeError, ValueError):
            gap = -1
        return cls(
            enabled=_as_bool(_value(raw, "enable", "enabled", default=False)),
            policy=str(_value(
                raw, "skipPolicy", default="GroupPhysicalPathSkipPolicy",
            ) or "GroupPhysicalPathSkipPolicy"),
            boundary_hour=boundary,
            server_time_boundary=_as_bool(_value(
                raw, "isBoundaryTimeBasedOnServerTime", default=False,
            )),
            last_run_gap_s=gap,
            reference_point=str(_value(raw, "referencePoint", default="EndTime") or "EndTime"),
        )

    @property
    def valid(self) -> bool:
        return self.enabled and (
            0 <= self.boundary_hour <= 23 or self.last_run_gap_s >= 0
        )


@dataclass(frozen=True)
class TaskScheduleRule:
    enabled: bool = False
    skip_hour: int | None = None
    cycle_enabled: bool = False
    boundary_hour: int = 0
    server_time_boundary: bool = False
    cycle: int = 1
    index: int = 1

    @classmethod
    def from_group_config(cls, group_config: Mapping[str, Any]) -> "TaskScheduleRule":
        pathing = _mapping(_value(group_config, "pathingConfig", default={}))
        raw_cycle = _mapping(_value(pathing, "taskCycleConfig", default={}))
        try:
            skip_hour = int(_value(pathing, "skipDuring", default=""))
            if not 0 <= skip_hour <= 23:
                skip_hour = None
        except (TypeError, ValueError):
            skip_hour = None
        def number(name: str, default: int) -> int:
            try:
                return int(_value(raw_cycle, name, default=default))
            except (TypeError, ValueError):
                return default
        return cls(
            enabled=_as_bool(_value(pathing, "enabled", default=False)),
            skip_hour=skip_hour,
            cycle_enabled=_as_bool(_value(raw_cycle, "enable", "enabled", default=False)),
            boundary_hour=number("boundaryTime", 0),
            server_time_boundary=_as_bool(_value(
                raw_cycle, "isBoundaryTimeBasedOnServerTime", default=False,
            )),
            cycle=number("cycle", 1),
            index=number("index", 1),
        )

    def skip_reason(self, *, now: datetime | None = None) -> str | None:
        if not self.enabled:
            return None
        current = now or datetime.now().astimezone()
        if self.skip_hour is not None and current.astimezone().hour == self.skip_hour:
            return "任务已到禁止执行时段"
        if not self.cycle_enabled:
            return None
        if self.cycle <= 0 or not 0 <= self.boundary_hour <= 23:
            return None
        zone = SERVER_TIMEZONE if self.server_time_boundary else current.astimezone().tzinfo
        adjusted = current.astimezone(zone)
        boundary = adjusted.replace(
            hour=self.boundary_hour, minute=0, second=0, microsecond=0,
        )
        if adjusted < boundary:
            adjusted -= timedelta(days=1)
        order = (adjusted.date() - datetime(1970, 1, 1).date()).days % self.cycle + 1
        if order != self.index:
            return f"任务不在执行周期（当前值 {order} != 配置值 {self.index}）"
        return None


@dataclass(frozen=True)
class ExecutionRecord:
    guid: str
    group_name: str
    project_name: str
    folder_name: str
    type: str
    start_time: datetime
    end_time: datetime | None = None
    server_start_time: datetime | None = None
    server_end_time: datetime | None = None
    successful: bool = False

    @classmethod
    def start(cls, group_name: str, project_name: str, folder_name: str,
              project_type: str, *, now: datetime | None = None) -> "ExecutionRecord":
        local_now = now or datetime.now().astimezone()
        if local_now.tzinfo is None:
            local_now = local_now.astimezone()
        return cls(
            str(uuid.uuid4()), group_name, project_name, folder_name, project_type,
            local_now, server_start_time=local_now.astimezone(SERVER_TIMEZONE),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExecutionRecord | None":
        start = _datetime(_value(raw, "start_time", "startTime"))
        if start is None:
            return None
        return cls(
            guid=str(_value(raw, "guid", "id", default="") or ""),
            group_name=str(_value(raw, "group_name", "groupName", default="") or ""),
            project_name=str(_value(raw, "project_name", "projectName", default="") or ""),
            folder_name=str(_value(raw, "folder_name", "folderName", default="") or ""),
            type=str(_value(raw, "type", default="") or ""),
            server_start_time=_datetime(_value(raw, "server_start_time", "serverStartTime")),
            start_time=start,
            server_end_time=_datetime(_value(raw, "server_end_time", "serverEndTime")),
            end_time=_datetime(_value(raw, "end_time", "endTime")),
            successful=bool(_value(raw, "is_successful", "isSuccessful", default=False)),
        )

    def to_mapping(self) -> dict[str, Any]:
        def stamp(value: datetime | None):
            return value.isoformat() if value is not None else None

        return {
            "guid": self.guid,
            "group_name": self.group_name,
            "project_name": self.project_name,
            "folder_name": self.folder_name,
            "type": self.type,
            "server_start_time": stamp(self.server_start_time),
            "start_time": stamp(self.start_time),
            "server_end_time": stamp(self.server_end_time),
            "end_time": stamp(self.end_time),
            "is_successful": self.successful,
        }


class ExecutionRecordStore:
    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory or DEFAULT_RECORD_DIR).expanduser().resolve()

    def _path(self, record: ExecutionRecord) -> Path:
        return self.directory / f"{record.start_time:%Y%m%d}.json"

    def _read(self, path: Path) -> list[ExecutionRecord]:
        if not path.is_file():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        values = _value(raw, "execution_records", "executionRecords", default=[]) \
            if isinstance(raw, Mapping) else []
        if not isinstance(values, list):
            return []
        return [
            record for value in values if isinstance(value, Mapping)
            if (record := ExecutionRecord.from_mapping(value)) is not None
        ]

    def save(self, record: ExecutionRecord) -> None:
        path = self._path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = self._read(path)
        for index, existing in enumerate(records):
            if existing.guid == record.guid:
                records[index] = record
                break
        else:
            records.append(record)
        payload = json.dumps({
            "name": path.stem,
            "execution_records": [item.to_mapping() for item in records],
        }, ensure_ascii=False, indent=2)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        finally:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass

    def finish(self, record: ExecutionRecord, successful: bool,
               *, now: datetime | None = None) -> ExecutionRecord:
        local_now = now or datetime.now().astimezone()
        if local_now.tzinfo is None:
            local_now = local_now.astimezone()
        finished = replace(
            record,
            end_time=local_now,
            server_end_time=local_now.astimezone(SERVER_TIMEZONE),
            successful=bool(successful),
        )
        self.save(finished)
        return finished

    def recent(self, rule: CompletionSkipRule,
               *, now: datetime | None = None) -> list[ExecutionRecord]:
        current = now or datetime.now().astimezone()
        days = 2 if 0 <= rule.boundary_hour <= 23 else 1
        if rule.last_run_gap_s >= 0:
            days = max(1, math.ceil(rule.last_run_gap_s / 86400) + 1)
        records: list[ExecutionRecord] = []
        for offset in range(days):
            path = self.directory / f"{current - timedelta(days=offset):%Y%m%d}.json"
            records.extend(reversed(self._read(path)))
        return records

    @staticmethod
    def _aware(value: datetime, zone) -> datetime:
        return value.replace(tzinfo=zone) if value.tzinfo is None else value.astimezone(zone)

    @classmethod
    def _within_boundary(cls, value: datetime, boundary_hour: int,
                         current: datetime, zone) -> bool:
        now = cls._aware(current, zone)
        target = cls._aware(value, zone)
        start = now.replace(hour=boundary_hour, minute=0, second=0, microsecond=0)
        if now < start:
            start -= timedelta(days=1)
        return start <= target < start + timedelta(days=1)

    def should_skip(self, group_name: str, project_name: str, folder_name: str,
                    project_type: str, rule: CompletionSkipRule,
                    *, now: datetime | None = None) -> tuple[bool, str]:
        if not rule.valid:
            return False, ""
        current = now or datetime.now().astimezone()
        boundary_enabled = 0 <= rule.boundary_hour <= 23
        policies = {
            "GroupPhysicalPathSkipPolicy", "PhysicalPathSkipPolicy", "SameNameSkipPolicy",
        }
        if rule.policy not in policies:
            return False, f"未预期的跳过策略：{rule.policy}"
        for record in self.recent(rule, now=current):
            if not record.successful:
                continue
            if record.type != project_type or record.project_name != project_name:
                continue
            reference = record.start_time if rule.reference_point == "StartTime" else record.end_time
            if reference is None:
                continue
            reference = self._aware(reference, current.astimezone().tzinfo)
            local_now = self._aware(current, reference.tzinfo)
            if rule.last_run_gap_s >= 0 and (local_now - reference).total_seconds() > rule.last_run_gap_s:
                continue
            if boundary_enabled:
                zone = SERVER_TIMEZONE if rule.server_time_boundary else local_now.tzinfo
                boundary_value = record.server_start_time if rule.server_time_boundary else record.start_time
                if boundary_value is None or not self._within_boundary(
                    boundary_value, rule.boundary_hour, current, zone,
                ):
                    continue
            if rule.policy == "GroupPhysicalPathSkipPolicy":
                matched = record.group_name == group_name and record.folder_name == folder_name
                reason = "组和物理路径匹配一致"
            elif rule.policy == "PhysicalPathSkipPolicy":
                matched = record.folder_name == folder_name
                reason = "物理路径相同"
            else:
                matched = True
                reason = "名称相同"
            if matched:
                detail = f"检查出满足跳过条件: {reason}"
                if rule.last_run_gap_s >= 0:
                    next_time = reference + timedelta(seconds=rule.last_run_gap_s)
                    detail += f"，需在 {next_time:%Y-%m-%d %H:%M:%S} 之后才能开始执行"
                elif boundary_enabled:
                    detail += f"，需在下一日 {rule.boundary_hour} 点后才能开始执行"
                return True, f"{detail}，匹配记录 GUID={record.guid}"
        return False, ""
