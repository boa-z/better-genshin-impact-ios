"""Portable BetterGI log analysis.

The desktop project turns its log files into a report containing configuration
groups, script durations, picked items and common pathing failures.  This
module keeps that useful, non-visual part of the feature available on iOS.
It deliberately returns structured data instead of generating a WPF/HTML
window, so the same parser can be used by the CLI, WebUI and tests without
touching DeviceHub or competing with the game screenshot loop.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from .travel_diary import ActionItem, DiaryPage, MoraStatistics, parse_diary_time


_HEADER_RE = re.compile(
    r"^\[(?P<time>\d{2}:\d{2}:\d{2})(?:\.(?P<fraction>\d+))?\]"
    r"\s+\[[^\]]+\](?:\s+\[(?P<instance>[A-Za-z][A-Za-z0-9]*:S\d+:P\d+:T\d+)\])?"
)
_TIME_ONLY_RE = re.compile(r"^\[(?P<time>\d{2}:\d{2}:\d{2})(?:\.(?P<fraction>\d+))?\]")
_FILE_DATE_RE = re.compile(r"better-genshin-impact(?P<date>\d{8})(?:_\d+)?\.log$", re.I)
_ISO_DATE_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})")

_GROUP_START_RE = re.compile(
    r"配置组\s*[\"“](?P<name>.+?)[\"”]\s*加载完成，共(?P<count>\d+)个脚本"
)
_GROUP_END_RE = re.compile(r"配置组\s*[\"“](?P<name>.+?)[\"”]\s*执行结束")
_SCRIPT_START_RE = re.compile(
    r"→\s*开始执行\s*(?:地图追踪任务|JS脚本|键鼠脚本|Shell任务|脚本)\s*[:：]\s*[\"“](?P<name>.+?)[\"”]"
)
_SCRIPT_END_RE = re.compile(r"→\s*脚本执行结束\s*[:：]\s*[\"“](?P<name>.+?)[\"”]")
_TELEPORT_RE = re.compile(r"传送失败，重试\s*(?P<count>\d+)\s*次")
_ERROR_RE = re.compile(r"执行脚本时发生异常\s*[:：]\s*[\"“](?P<message>.+?)[\"”]")
_PICK_RE = re.compile(r"交互或拾取\s*[：:]\s*[\"“](?P<name>.+?)[\"”]")


def _parse_date(value: date | datetime | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"无法解析日志日期: {value}")


def _date_from_path(path: Path) -> date:
    match = _FILE_DATE_RE.search(path.name)
    if match:
        return datetime.strptime(match.group("date"), "%Y%m%d").date()
    match = _ISO_DATE_RE.search(path.name)
    if match:
        return date.fromisoformat(match.group("date"))
    # A manually exported log may not follow BetterGI's file name.  mtime is
    # a deterministic and useful fallback, while callers can always override
    # it with ``(path, date)`` when the source date matters.
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).date()
    except OSError:
        return date.today()


def _timestamp(day: date, value: str, fraction: str | None = None) -> datetime:
    current = datetime.combine(day, datetime.strptime(value, "%H:%M:%S").time())
    if fraction:
        # datetime accepts microseconds only.  Preserve both short and long
        # log fractions by right-padding and truncating to six digits.
        current += timedelta(microseconds=int(fraction[:6].ljust(6, "0")))
    return current


def _compact(value: str) -> str:
    return " ".join(str(value).split()).strip()


@dataclass
class FaultScenario:
    """Failure counters attached to one configuration-group project."""

    pathing_success_end: bool = True
    revive_count: int = 0
    teleport_fail_count: int = 0
    stuck_count: int = 0
    retry_count: int = 0
    battle_timeout_count: int = 0
    error_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "pathingSuccessEnd": self.pathing_success_end,
            "reviveCount": self.revive_count,
            "teleportFailCount": self.teleport_fail_count,
            "stuckCount": self.stuck_count,
            "retryCount": self.retry_count,
            "battleTimeoutCount": self.battle_timeout_count,
            "errCount": self.error_count,
        }


@dataclass
class ConfigTaskLog:
    name: str
    start_date: datetime | None = None
    end_date: datetime | None = None
    picks: dict[str, int] = field(default_factory=dict)
    fault: FaultScenario = field(default_factory=FaultScenario)
    is_merger: bool = False

    @property
    def duration_seconds(self) -> float:
        if self.start_date is None or self.end_date is None:
            return 0.0
        return max(0.0, (self.end_date - self.start_date).total_seconds())

    def add_pick(self, value: str) -> None:
        name = _compact(value)
        if name:
            self.picks[name] = self.picks.get(name, 0) + 1

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "isMerger": self.is_merger,
            "startDate": self.start_date.isoformat() if self.start_date else None,
            "endDate": self.end_date.isoformat() if self.end_date else None,
            "durationSeconds": round(self.duration_seconds, 3),
            "picks": dict(sorted(self.picks.items())),
            "fault": self.fault.to_dict(),
        }


@dataclass
class ConfigGroupLog:
    name: str
    start_date: datetime | None = None
    end_date: datetime | None = None
    declared_script_count: int | None = None
    tasks: list[ConfigTaskLog] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        if self.start_date is None or self.end_date is None:
            return 0.0
        return max(0.0, (self.end_date - self.start_date).total_seconds())

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "startDate": self.start_date.isoformat() if self.start_date else None,
            "endDate": self.end_date.isoformat() if self.end_date else None,
            "durationSeconds": round(self.duration_seconds, 3),
            "declaredScriptCount": self.declared_script_count,
            "tasks": [task.to_dict() for task in self.tasks],
        }


@dataclass(frozen=True)
class MoraDayStatistics:
    """Travel-diary statistics for a BetterGI day (04:00 to 04:00)."""

    day: date
    statistics: MoraStatistics


@dataclass(frozen=True)
class _LogLine:
    text: str
    source: str
    day: date
    timestamp: datetime | None
    instance: str


@dataclass
class LogParseReport:
    """Structured result returned by :func:`parse_log_files`."""

    groups: list[ConfigGroupLog] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    mora_items: tuple[ActionItem, ...] = ()

    @property
    def tasks(self) -> list[ConfigTaskLog]:
        return [task for group in self.groups for task in group.tasks]

    @property
    def total_duration_seconds(self) -> float:
        return sum(group.duration_seconds for group in self.groups)

    @property
    def pick_totals(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for task in self.tasks:
            for name, count in task.picks.items():
                result[name] = result.get(name, 0) + count
        return dict(sorted(result.items()))

    @property
    def fault_totals(self) -> dict[str, int]:
        fields = {
            "reviveCount": "revive_count",
            "teleportFailCount": "teleport_fail_count",
            "stuckCount": "stuck_count",
            "retryCount": "retry_count",
            "battleTimeoutCount": "battle_timeout_count",
            "errCount": "error_count",
        }
        return {
            output_name: sum(int(getattr(task.fault, field_name)) for task in self.tasks)
            for output_name, field_name in fields.items()
        }

    @property
    def mora_statistics(self) -> MoraStatistics:
        return MoraStatistics(tuple(self.mora_items))

    @property
    def mora_day_statistics(self) -> list[MoraDayStatistics]:
        grouped: dict[date, list[ActionItem]] = defaultdict(list)
        for item in self.mora_items:
            stamp = item.parsed_time
            if stamp is not None:
                grouped[(stamp - timedelta(hours=4)).date()].append(item)
        return [
            MoraDayStatistics(day, MoraStatistics(tuple(grouped[day])))
            for day in sorted(grouped)
        ]

    def mora_between(
        self,
        start: datetime | None,
        end: datetime | None,
    ) -> MoraStatistics:
        """Return diary actions inside an inclusive log time window."""

        if start is None or end is None:
            return MoraStatistics(())
        values: list[ActionItem] = []
        start_value = start.replace(tzinfo=None)
        end_value = end.replace(tzinfo=None)
        for item in self.mora_items:
            stamp = item.parsed_time
            if stamp is None:
                continue
            stamp_value = stamp.replace(tzinfo=None)
            if start_value <= stamp_value <= end_value:
                values.append(item)
        return MoraStatistics(tuple(values))

    def to_dict(self) -> dict[str, object]:
        payload = {
            "sources": list(self.sources),
            "groups": [group.to_dict() for group in self.groups],
            "summary": {
                "groupCount": len(self.groups),
                "taskCount": len(self.tasks),
                "totalDurationSeconds": round(self.total_duration_seconds, 3),
                "pickTotals": self.pick_totals,
                "faultTotals": self.fault_totals,
            },
        }
        if self.mora_items:
            payload["mora"] = _mora_payload(self.mora_statistics, self.mora_day_statistics)
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_html(self, *, title: str = "日志分析", include_faults: bool = True) -> str:
        return render_html(self, title=title, include_faults=include_faults)


def _iter_lines(path: Path, day: date) -> Iterator[_LogLine]:
    try:
        stream = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return
    with stream:
        for raw in stream:
            text = raw.rstrip("\r\n")
            header = _HEADER_RE.match(text)
            if header:
                timestamp = _timestamp(
                    day, header.group("time"), header.group("fraction")
                )
                instance = header.group("instance") or ""
                message = text[header.end():].strip()
            else:
                timestamp_match = _TIME_ONLY_RE.match(text)
                timestamp = (
                    _timestamp(
                        day,
                        timestamp_match.group("time"),
                        timestamp_match.group("fraction"),
                    )
                    if timestamp_match
                    else None
                )
                instance = ""
                message = text[timestamp_match.end():].strip() if timestamp_match else text
            yield _LogLine(message, str(path), day, timestamp, instance)


def _coerce_source(source: str | Path | tuple[str | Path, date | datetime | str]) -> tuple[Path, date]:
    if isinstance(source, tuple):
        if len(source) != 2:
            raise ValueError("日志来源元组必须是 (path, date)")
        path, raw_day = source
        day = _parse_date(raw_day)
        if day is None:
            raise ValueError("日志日期不能为空")
        return Path(path).expanduser(), day
    path = Path(source).expanduser()
    return path, _date_from_path(path)


def _event_time(item: _LogLine, previous: datetime | None) -> datetime | None:
    return item.timestamp or previous


def _close_task(task: ConfigTaskLog | None, fallback: datetime | None) -> None:
    if task is not None and task.end_date is None:
        task.end_date = fallback or task.start_date


def _close_group(group: ConfigGroupLog | None, fallback: datetime | None) -> None:
    if group is None:
        return
    _close_task(group.tasks[-1] if group.tasks else None, fallback)
    if group.end_date is None:
        group.end_date = (
            group.tasks[-1].end_date if group.tasks and group.tasks[-1].end_date else fallback
        ) or group.start_date


def _parse_instance(lines: Sequence[_LogLine]) -> list[ConfigGroupLog]:
    groups: list[ConfigGroupLog] = []
    current_group: ConfigGroupLog | None = None
    current_task: ConfigTaskLog | None = None
    previous_time: datetime | None = None

    for item in lines:
        event_time = _event_time(item, previous_time)
        if item.timestamp is not None:
            previous_time = item.timestamp
        text = item.text

        match = _GROUP_START_RE.search(text)
        if match:
            _close_group(current_group, event_time)
            current_group = ConfigGroupLog(
                name=_compact(match.group("name")),
                start_date=event_time,
                declared_script_count=int(match.group("count")),
            )
            groups.append(current_group)
            current_task = None
            continue

        if current_group is None:
            continue

        match = _GROUP_END_RE.search(text)
        if match and _compact(match.group("name")) == current_group.name:
            _close_task(current_task, event_time)
            current_group.end_date = event_time or current_group.end_date
            current_task = None
            current_group = None
            continue

        match = _SCRIPT_START_RE.search(text)
        if match:
            _close_task(current_task, event_time)
            current_task = ConfigTaskLog(
                name=_compact(match.group("name")), start_date=event_time,
            )
            current_group.tasks.append(current_task)
            continue

        if current_task is None:
            continue

        match = _SCRIPT_END_RE.search(text)
        if match and _compact(match.group("name")) == current_task.name:
            current_task.end_date = event_time or current_task.end_date
            current_task = None
            continue

        if "此追踪脚本未正常走完" in text:
            current_task.fault.pathing_success_end = False
        if text.endswith("前往七天神像复活"):
            current_task.fault.revive_count += 1
        match = _TELEPORT_RE.search(text)
        if match:
            # BetterGI reports the retry ordinal.  Keep the largest ordinal
            # when a route contains more than one failed teleport.
            current_task.fault.teleport_fail_count = max(
                current_task.fault.teleport_fail_count, int(match.group("count"))
            )
        if text == "战斗超时结束":
            current_task.fault.battle_timeout_count += 1
        if text.endswith("重试一次路线或放弃此路线！"):
            current_task.fault.retry_count += 1
        if text == "疑似卡死，尝试脱离...":
            current_task.fault.stuck_count += 1
        if _ERROR_RE.search(text) or text.startswith("出错") or text.startswith("Uncaught Error"):
            current_task.fault.error_count += 1
        match = _PICK_RE.search(text)
        if match:
            current_task.add_pick(match.group("name"))

    _close_group(current_group, previous_time)
    return groups


def parse_log_files(
    sources: Iterable[str | Path | tuple[str | Path, date | datetime | str]],
    *,
    mora_items: Iterable[ActionItem] = (),
) -> LogParseReport:
    """Parse one or more BetterGI log files without connecting to a device.

    A source can be a path (the date is inferred from the standard BetterGI
    filename) or ``(path, date)`` for manually exported/renamed logs.  Lines
    carrying an instance id are parsed independently, matching the desktop
    analyser's multi-instance behaviour.
    """

    grouped: dict[str, list[_LogLine]] = defaultdict(list)
    paths: list[str] = []
    for source in sources:
        path, day = _coerce_source(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(str(path))
        for item in _iter_lines(path, day):
            grouped[item.instance].append(item)

    groups: list[ConfigGroupLog] = []
    for lines in grouped.values():
        lines.sort(key=lambda item: (item.timestamp or datetime.min, item.source))
        groups.extend(_parse_instance(lines))
    groups.sort(key=lambda group: group.start_date or datetime.min)
    return LogParseReport(
        groups=groups,
        sources=paths,
        mora_items=tuple(item for item in mora_items if isinstance(item, ActionItem)),
    )


def load_diary_action_items(sources: Iterable[str | Path]) -> list[ActionItem]:
    """Load deduplicated detail actions from local travel-diary cache files.

    BetterGI stores each month as an API response containing ``data.list``.
    Accepting a plain list as well makes the helper useful for exported
    fixtures while keeping this path entirely offline.
    """

    items: list[ActionItem] = []
    seen: set[tuple[int, str, int, str]] = set()
    for raw_source in sources:
        path = Path(raw_source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError(f"无法读取旅行札记缓存：{path}") from error
        page: DiaryPage | None = None
        if isinstance(payload, Mapping):
            page = DiaryPage.from_payload(payload)
        elif isinstance(payload, list):
            page = DiaryPage(items=[
                ActionItem.from_mapping(item)
                for item in payload if isinstance(item, Mapping)
            ])
        if page is None:
            raise ValueError(f"旅行札记缓存格式无效：{path}")
        for item in page.items:
            if item.action_id not in {28, 37, 39} or item.key() in seen:
                continue
            seen.add(item.key())
            items.append(item)
    return sorted(
        items,
        key=lambda item: (
            item.parsed_time.timestamp() if item.parsed_time is not None else float("-inf")
        ),
    )


def discover_log_files(folder: str | Path) -> list[tuple[Path, date]]:
    """Return BetterGI logs in chronological order with their source dates."""

    root = Path(folder).expanduser()
    if not root.is_dir():
        return []
    result: list[tuple[Path, date]] = []
    for path in root.iterdir():
        match = _FILE_DATE_RE.fullmatch(path.name)
        if not match or not path.is_file():
            continue
        result.append((path, datetime.strptime(match.group("date"), "%Y%m%d").date()))
    return sorted(result, key=lambda item: (item[1], item[0].name))


def format_duration(total_seconds: float) -> str:
    """Format seconds like the BetterGI desktop report."""

    if total_seconds < 0:
        raise ValueError("seconds cannot be negative")
    hours, remainder = divmod(float(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    result = ""
    if hours > 0:
        result += f"{int(hours)}小时"
    if minutes > 0 or hours > 0:
        result += f"{int(minutes)}分钟"
    if seconds > 0 or (hours == 0 and minutes == 0):
        result += f"{int(seconds)}秒" if seconds.is_integer() else f"{seconds:.2f}秒"
    return result


def render_text(report: LogParseReport) -> str:
    """Render a compact terminal report for humans."""

    lines = [
        f"配置组 {len(report.groups)} 个，脚本 {len(report.tasks)} 个，"
        f"总耗时 {format_duration(report.total_duration_seconds)}",
    ]
    for group in report.groups:
        group_duration = format_duration(group.duration_seconds)
        lines.append(f"- {group.name} ({group_duration})")
        for task in group.tasks:
            status = "成功" if task.fault.pathing_success_end else "异常结束"
            lines.append(
                f"  - {task.name}: {format_duration(task.duration_seconds)}，{status}"
            )
            if task.picks:
                picks = ", ".join(f"{name}×{count}" for name, count in task.picks.items())
                lines.append(f"    拾取: {picks}")
            fault = task.fault
            fault_count = sum((
                fault.revive_count, fault.teleport_fail_count, fault.stuck_count,
                fault.retry_count, fault.battle_timeout_count, fault.error_count,
            ))
            if fault_count:
                lines.append(
                    "    故障: "
                    f"复活 {fault.revive_count}、传送重试 {fault.teleport_fail_count}、"
                    f"脱困 {fault.stuck_count}、路线重试 {fault.retry_count}、"
                    f"战斗超时 {fault.battle_timeout_count}、异常 {fault.error_count}"
                )
    if report.pick_totals:
        lines.append("拾取汇总: " + ", ".join(
            f"{name}×{count}" for name, count in report.pick_totals.items()
        ))
    return "\n".join(lines)


def _html_datetime(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _html_cell(value: object) -> str:
    return escape(str(value), quote=True)


def _mora_statistics_payload(statistics: MoraStatistics) -> dict[str, object]:
    return {
        "smallMonsterCount": statistics.small_monster_statistics,
        "smallMonsterDetails": statistics.small_monster_details,
        "eliteCount": statistics.elite_game_statistics,
        "eliteDetails": statistics.elite_details,
        "totalMora": statistics.total_mora_killing_monsters,
        "emergencyMora": statistics.emergency_bonus,
        "chestMora": statistics.chest_reward,
    }


def _mora_payload(
    statistics: MoraStatistics,
    days: Sequence[MoraDayStatistics],
) -> dict[str, object]:
    return {
        "itemCount": len(statistics.action_items),
        "totals": _mora_statistics_payload(statistics),
        "days": [
            {"date": item.day.isoformat(), **_mora_statistics_payload(item.statistics)}
            for item in days
        ],
    }


def _html_sort_header(label: str, sort_type: str) -> str:
    return (
        f'<th class="sortable" data-sort-type="{_html_cell(sort_type)}">'
        f"{_html_cell(label)}</th>"
    )


def _html_sort_cell(
    value: object,
    *,
    sort_type: str | None = None,
    sort_value: object | None = None,
    class_name: str = "",
) -> str:
    classes = f' class="{_html_cell(class_name)}"' if class_name else ""
    if sort_type is None:
        return f"<td{classes}>{_html_cell(value)}</td>"
    raw_sort_value = value if sort_value is None else sort_value
    return (
        f'<td{classes} data-sort-type="{_html_cell(sort_type)}" '
        f'data-sort="{_html_cell(raw_sort_value)}">{_html_cell(value)}</td>'
    )


def _fault_total(fault: FaultScenario) -> int:
    return sum((
        fault.revive_count,
        fault.teleport_fail_count,
        fault.stuck_count,
        fault.retry_count,
        fault.battle_timeout_count,
        fault.error_count,
    ))


def _render_fault_cells(fault: FaultScenario) -> str:
    values = (
        fault.revive_count,
        fault.retry_count,
        fault.stuck_count,
        fault.battle_timeout_count,
        fault.teleport_fail_count,
        fault.error_count,
    )
    return "".join(
        _html_sort_cell(value or "", sort_type="number", sort_value=value)
        for value in values
    )


_SORT_SCRIPT = r"""
(function () {
  "use strict";
  function cellValue(row, column, type) {
    var cell = row.cells[column];
    if (!cell) return type === "number" || type === "date" || type === "time" ? 0 : "";
    var raw = cell.getAttribute("data-sort");
    if (raw === null) raw = (cell.textContent || "").trim();
    if (type === "number" || type === "time") {
      var number = Number(String(raw).replace(/[^0-9.+-]/g, ""));
      return Number.isFinite(number) ? number : 0;
    }
    if (type === "date") {
      var dateNumber = Number(raw);
      if (Number.isFinite(dateNumber) && raw !== "") return dateNumber;
      var parsed = Date.parse(raw);
      return Number.isFinite(parsed) ? parsed : 0;
    }
    return String(raw).toLocaleLowerCase();
  }
  function sortTable(table, header) {
    var body = table.tBodies[0];
    if (!body) return;
    var headers = Array.prototype.slice.call(table.querySelectorAll("thead th"));
    var column = headers.indexOf(header);
    if (column < 0) return;
    var type = header.getAttribute("data-sort-type") || "string";
    var oldColumn = table.getAttribute("data-sort-column");
    var oldDirection = table.getAttribute("data-sort-direction");
    var direction = oldColumn === String(column) && oldDirection === "asc" ? "desc" : "asc";
    table.setAttribute("data-sort-column", String(column));
    table.setAttribute("data-sort-direction", direction);
    headers.forEach(function (item) {
      item.classList.remove("sort-asc", "sort-desc");
    });
    header.classList.add("sort-" + direction);

    var rows = Array.prototype.slice.call(body.rows);
    var blocks = [];
    var fixed = [];
    var current = null;
    rows.forEach(function (row) {
      if (row.getAttribute("data-sortable") === "true") {
        current = { main: row, details: [] };
        blocks.push(current);
      } else if (current && row.getAttribute("data-ignore-sort") !== "true") {
        current.details.push(row);
      } else {
        fixed.push(row);
      }
    });
    if (!blocks.length) return;
    blocks.sort(function (left, right) {
      var a = cellValue(left.main, column, type);
      var b = cellValue(right.main, column, type);
      var result = a < b ? -1 : (a > b ? 1 : 0);
      return direction === "asc" ? result : -result;
    });
    rows.forEach(function (row) { row.remove(); });
    blocks.forEach(function (block) {
      body.appendChild(block.main);
      block.details.forEach(function (row) { body.appendChild(row); });
    });
    fixed.forEach(function (row) { body.appendChild(row); });
  }
  document.addEventListener("click", function (event) {
    var header = event.target.closest("th[data-sort-type]");
    if (header) sortTable(header.closest("table"), header);
  });
})();
"""


def render_html(
    report: LogParseReport,
    *,
    title: str = "日志分析",
    include_faults: bool = True,
) -> str:
    """Render a self-contained BetterGI-style log report.

    The desktop application embeds CSS/JavaScript assets in a WPF HTML mask.
    The iOS port deliberately emits a standalone document instead: it can be
    saved with shell redirection, opened on any platform, and does not load
    remote resources or request another DeviceHub frame.
    """

    fault_headers = (
        "".join(_html_sort_header(label, "number") for label in (
            "复活", "路线重试", "疑似卡死", "战斗超时", "传送失败", "异常",
        ))
        if include_faults else ""
    )
    css = """
:root { color-scheme: light dark; font-family: -apple-system, BlinkMacSystemFont,
  "Segoe UI", sans-serif; }
body { margin: 0; padding: 24px; background: #f5f7fb; color: #1f2937; }
main { max-width: 1400px; margin: 0 auto; }
h1 { margin: 0 0 18px; }
h2 { margin: 28px 0 10px; color: #234b76; }
.summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px; margin-bottom: 18px; }
.card { background: white; border: 1px solid #dbe3ef; border-radius: 8px;
  padding: 14px; box-shadow: 0 1px 2px #0000000d; }
.card .label { display: block; color: #64748b; font-size: 0.85rem; }
.card .value { display: block; margin-top: 4px; font-size: 1.3rem; font-weight: 650; }
.table-wrap { overflow-x: auto; background: white; border: 1px solid #dbe3ef;
  border-radius: 8px; }
table { width: 100%; border-collapse: collapse; min-width: 760px; }
th, td { padding: 8px 10px; border-bottom: 1px solid #e5eaf1; text-align: left;
  vertical-align: top; }
th { position: sticky; top: 0; z-index: 1; background: #e8f1fb; color: #234b76; }
th.sortable { cursor: pointer; user-select: none; }
th.sortable::after { content: "↕"; margin-left: .35em; color: #7890aa; }
th.sortable.sort-asc::after { content: "↑"; color: #234b76; }
th.sortable.sort-desc::after { content: "↓"; color: #234b76; }
tr:last-child td { border-bottom: 0; }
tr.failed td:first-child { color: #b42318; font-weight: 650; }
tr.sub-row td { background: #fafcff; color: #475569; font-size: 0.92rem; }
.badge { display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: .8rem;
  background: #dcfce7; color: #166534; }
.badge.failed { background: #fee2e2; color: #991b1b; }
.sources { color: #64748b; font-size: .9rem; white-space: pre-wrap; }
@media (prefers-color-scheme: dark) {
  body { background: #111827; color: #e5e7eb; }
  .card, .table-wrap { background: #1f2937; border-color: #374151; }
  th { background: #243b53; color: #dbeafe; }
  th, td { border-color: #374151; }
  tr.sub-row td { background: #1b2533; color: #cbd5e1; }
  h2 { color: #93c5fd; }
}
"""

    summary = report.to_dict()["summary"]
    cards = [
        ("配置组", summary["groupCount"]),
        ("脚本", summary["taskCount"]),
        ("总耗时", format_duration(report.total_duration_seconds)),
        ("拾取种类", len(report.pick_totals)),
        ("故障次数", sum(report.fault_totals.values())),
    ]
    if report.mora_items:
        cards.append(("锄地摩拉", report.mora_statistics.total_mora_killing_monsters))
    card_html = "".join(
        f'<div class="card"><span class="label">{_html_cell(label)}</span>'
        f'<span class="value">{_html_cell(value)}</span></div>'
        for label, value in cards
    )

    pick_html = ""
    if report.pick_totals:
        rows = "".join(
            f'<tr data-sortable="true"><td>{_html_cell(name)}</td>'
            f'{_html_sort_cell(count, sort_type="number", sort_value=count)}</tr>'
            for name, count in report.pick_totals.items()
        )
        pick_html = (
            "<h2>拾取汇总</h2><div class=\"table-wrap\"><table>"
            f"<thead><tr>{_html_sort_header('物品', 'string')}"
            f"{_html_sort_header('次数', 'number')}</tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    mora_html = ""
    if report.mora_items:
        daily_rows: list[str] = []
        daily_columns = (
            ("日期", "date"),
            ("小怪数量", "number"),
            ("小怪详细(摩拉/10)", "string"),
            ("最后小怪时间", "date"),
            ("精英数量", "number"),
            ("精英详细", "string"),
            ("最后精英时间", "date"),
            ("总计锄地摩拉", "number"),
            ("突发事件获取摩拉", "number"),
            ("宝箱奖励（狗粮附带）", "number"),
        )
        for daily in reversed(report.mora_day_statistics):
            stats = daily.statistics
            last_small = stats.last_small_time or ""
            last_elite = stats.last_elite_time or ""
            daily_rows.append(
                '<tr data-sortable="true">'
                + _html_sort_cell(daily.day.isoformat(), sort_type="date", sort_value=daily.day.toordinal())
                + _html_sort_cell(stats.small_monster_statistics, sort_type="number")
                + _html_sort_cell(stats.small_monster_details)
                + _html_sort_cell(last_small, sort_type="date", sort_value=last_small)
                + _html_sort_cell(stats.elite_game_statistics, sort_type="number")
                + _html_sort_cell(stats.elite_details)
                + _html_sort_cell(last_elite, sort_type="date", sort_value=last_elite)
                + _html_sort_cell(stats.total_mora_killing_monsters, sort_type="number")
                + _html_sort_cell(
                    stats.emergency_bonus,
                    sort_type="number",
                    sort_value=sum(item.num for item in stats.action_items if item.action_id == 28),
                )
                + _html_sort_cell(
                    stats.chest_reward,
                    sort_type="number",
                    sort_value=sum(item.num for item in stats.action_items if item.action_id == 39),
                )
                + "</tr>"
            )
        daily_headers = "".join(_html_sort_header(label, sort_type) for label, sort_type in daily_columns)
        mora_html = (
            "<h2>按日摩拉收益统计</h2><div class=\"table-wrap\"><table>"
            f"<thead><tr>{daily_headers}</tr></thead>"
            f"<tbody>{''.join(daily_rows)}</tbody></table></div>"
        )

    group_html: list[str] = []
    has_mora = bool(report.mora_items)
    mora_task_columns = (
        ("小怪", "number"),
        ("小怪详细(摩拉/10)", "string"),
        ("精英", "number"),
        ("精英详细", "string"),
        ("锄地摩拉", "number"),
        ("摩拉（每秒）", "number"),
    )
    for group in report.groups:
        declared = (
            f"，声明脚本 {group.declared_script_count} 个"
            if group.declared_script_count is not None else ""
        )
        rows: list[str] = []
        group_mora = report.mora_between(group.start_date, group.end_date)
        for task in group.tasks:
            failed = not task.fault.pathing_success_end
            status = (
                '<span class="badge failed">异常结束</span>'
                if failed else '<span class="badge">完成</span>'
            )
            row_class = ' class="failed"' if failed else ""
            cells = [
                _html_sort_cell(task.name, sort_type="string"),
                _html_sort_cell(
                    _html_datetime(task.start_date),
                    sort_type="date",
                    sort_value=task.start_date.timestamp() if task.start_date else 0,
                ),
                _html_sort_cell(
                    _html_datetime(task.end_date),
                    sort_type="date",
                    sort_value=task.end_date.timestamp() if task.end_date else 0,
                ),
                _html_sort_cell(
                    format_duration(task.duration_seconds),
                    sort_type="time",
                    sort_value=task.duration_seconds,
                ),
                (
                    f'<td data-sort-type="string" data-sort="{_html_cell("异常结束" if failed else "完成")}">'
                    f"{status}</td>"
                ),
            ]
            if include_faults:
                cells.append(_render_fault_cells(task.fault))
            task_mora = report.mora_between(task.start_date, task.end_date)
            if has_mora:
                per_second = (
                    task_mora.total_mora_killing_monsters / task.duration_seconds
                    if task.duration_seconds > 0 else 0
                )
                cells.extend([
                    _html_sort_cell(task_mora.small_monster_statistics, sort_type="number"),
                    _html_sort_cell(task_mora.small_monster_details),
                    _html_sort_cell(task_mora.elite_game_statistics, sort_type="number"),
                    _html_sort_cell(task_mora.elite_details),
                    _html_sort_cell(
                        task_mora.total_mora_killing_monsters,
                        sort_type="number",
                    ),
                    _html_sort_cell(
                        f"{per_second:.2f}", sort_type="number", sort_value=per_second,
                    ),
                ])
            rows.append(f'<tr{row_class} data-sortable="true">{"".join(cells)}</tr>')
            if task.picks:
                picks = "，".join(
                    f"{_html_cell(name)}×{_html_cell(count)}"
                    for name, count in sorted(task.picks.items())
                )
                rows.append(
                    f'<tr class="sub-row" data-sort-detail="true"><td colspan="{5 + (6 if include_faults else 0) + (6 if has_mora else 0)}">'
                    f"拾取物：{picks}</td></tr>"
                )
            if include_faults and _fault_total(task.fault):
                rows.append(
                    f'<tr class="sub-row" data-sort-detail="true"><td colspan="{5 + (6 if include_faults else 0) + (6 if has_mora else 0)}">'
                    f"故障计数：{_html_cell(task.fault.to_dict())}</td></tr>"
                )
        headers = (
            _html_sort_header("任务名称", "string")
            + _html_sort_header("开始时间", "date")
            + _html_sort_header("结束时间", "date")
            + _html_sort_header("任务耗时", "time")
            + _html_sort_header("状态", "string")
        ) + fault_headers
        if has_mora:
            headers += "".join(_html_sort_header(label, sort_type) for label, sort_type in mora_task_columns)
        total_columns = 5 + (6 if include_faults else 0) + (6 if has_mora else 0)
        if has_mora:
            rows.append(
                f'<tr data-ignore-sort="true"><td colspan="{total_columns}">'
                f"锄地总计：小怪 {group_mora.small_monster_statistics}，"
                f"精英 {group_mora.elite_game_statistics}，"
                f"合计锄地摩拉 {group_mora.total_mora_killing_monsters}"
                "</td></tr>"
            )
        tbody = "".join(rows) or f'<tr><td colspan="{total_columns}">无脚本</td></tr>'
        group_html.append(
            f"<h2>配置组：{_html_cell(group.name)}</h2>"
            f"<p>耗时 {_html_cell(format_duration(group.duration_seconds))}"
            f"{_html_cell(declared)}</p>"
            f'<div class="table-wrap"><table><thead><tr>{headers}</tr></thead>'
            f"<tbody>{tbody}</tbody></table></div>"
        )

    source_html = ""
    if report.sources:
        source_html = (
            '<h2>来源</h2><div class="sources">'
            + _html_cell("\n".join(report.sources))
            + "</div>"
        )
    return (
        "<!doctype html>\n<html lang=\"zh-CN\"><head>"
        "<meta charset=\"utf-8\"><meta name=\"viewport\" "
        "content=\"width=device-width, initial-scale=1\">"
        f"<title>{_html_cell(title)}</title><style>{css}</style></head><body><main>"
        f"<h1>{_html_cell(title)}</h1><div class=\"summary\">{card_html}</div>"
        f"{source_html}{mora_html}{pick_html}{''.join(group_html)}"
        f"</main><script>{_SORT_SCRIPT}</script></body></html>\n"
    )
