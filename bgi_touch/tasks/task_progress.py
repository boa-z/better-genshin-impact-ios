"""Atomic ScriptGroup progress compatible with BetterGI's TaskProgress files."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROGRESS_DIR = PROJECT_ROOT / "log" / "task_progress"
_PROGRESS_NAME = re.compile(r"^\d{14}(?:-\d+)?$")


def _value(raw: Mapping[str, Any], *names: str, default=None):
    folded = {str(key).replace("_", "").casefold(): value for key, value in raw.items()}
    for name in names:
        key = name.replace("_", "").casefold()
        if key in folded:
            return folded[key]
    return default


def _timestamp(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


@dataclass
class ProjectProgress:
    group_name: str = ""
    index: int = 0
    name: str = ""
    folder_name: str = ""
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str | None = None
    status: int = 1
    task_end: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "ProjectProgress | None":
        if not isinstance(raw, Mapping):
            return None
        return cls(
            group_name=str(_value(raw, "groupName", default="") or ""),
            index=int(_value(raw, "index", default=0) or 0),
            name=str(_value(raw, "name", "projectName", default="") or ""),
            folder_name=str(_value(raw, "folderName", default="") or ""),
            start_time=_timestamp(_value(raw, "startTime")) or datetime.now().isoformat(),
            end_time=_timestamp(_value(raw, "endTime")),
            status=int(_value(raw, "status", default=1) or 1),
            task_end=bool(_value(raw, "taskEnd", default=False)),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "groupName": self.group_name,
            "taskEnd": self.task_end,
            "index": self.index,
            "name": self.name,
            "folderName": self.folder_name,
            "startTime": self.start_time,
            "endTime": self.end_time,
            "status": self.status,
        }


@dataclass
class TaskProgress:
    script_group_names: list[str]
    name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str | None = None
    last_script_group_name: str | None = None
    last_success: ProjectProgress | None = None
    current_group_name: str | None = None
    current: ProjectProgress | None = None
    history: list[ProjectProgress] = field(default_factory=list)
    loop: bool = False
    loop_count: int = 0
    consecutive_failure_count: int = 0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TaskProgress":
        history = _value(raw, "history", default=[])
        return cls(
            script_group_names=[
                str(value) for value in _value(raw, "scriptGroupNames", default=[]) or []
            ],
            name=str(_value(raw, "name", default="") or datetime.now().strftime("%Y%m%d%H%M%S")),
            start_time=_timestamp(_value(raw, "startTime")) or datetime.now().isoformat(),
            end_time=_timestamp(_value(raw, "endTime")),
            last_script_group_name=_value(raw, "lastScriptGroupName"),
            last_success=ProjectProgress.from_mapping(
                _value(raw, "lastSuccessScriptGroupProjectInfo", "lastSuccess")
            ),
            current_group_name=_value(raw, "currentScriptGroupName", "currentGroupName"),
            current=ProjectProgress.from_mapping(
                _value(raw, "currentScriptGroupProjectInfo", "current")
            ),
            history=[
                item for value in (history if isinstance(history, list) else [])
                if (item := ProjectProgress.from_mapping(value)) is not None
            ],
            loop=bool(_value(raw, "loop", default=False)),
            loop_count=int(_value(raw, "loopCount", default=0) or 0),
            consecutive_failure_count=int(
                _value(raw, "consecutiveFailureCount", default=0) or 0
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "scriptGroupNames": self.script_group_names,
            "lastScriptGroupName": self.last_script_group_name,
            "lastSuccessScriptGroupProjectInfo": (
                self.last_success.to_mapping() if self.last_success else None
            ),
            "currentScriptGroupName": self.current_group_name,
            "currentScriptGroupProjectInfo": self.current.to_mapping() if self.current else None,
            "name": self.name,
            "startTime": self.start_time,
            "endTime": self.end_time,
            "history": [item.to_mapping() for item in self.history],
            "loop": self.loop,
            "loopCount": self.loop_count,
            "consecutiveFailureCount": self.consecutive_failure_count,
        }

    def begin(self, group_name: str, index: int, name: str, folder_name: str) -> None:
        self.current_group_name = group_name
        self.current = ProjectProgress(
            group_name=group_name,
            index=index,
            name=name,
            folder_name=folder_name,
        )

    def finish_current(self, success: bool) -> None:
        if self.current is None:
            return
        self.current.end_time = datetime.now().isoformat()
        self.current.task_end = True
        self.current.status = 1 if success else 2
        self.history.append(self.current)
        if success:
            self.last_script_group_name = self.current.group_name
            self.last_success = self.current
            self.consecutive_failure_count = 0
        else:
            self.consecutive_failure_count += 1


class TaskProgressStore:
    def __init__(self, directory: str | Path | None = None,
                 log: Callable[[str], None] = print):
        self.directory = Path(directory or DEFAULT_PROGRESS_DIR).expanduser().resolve()
        self.log = log

    def path_for(self, name: str) -> Path:
        if not _PROGRESS_NAME.fullmatch(str(name)):
            raise ValueError("TaskProgress 名称必须是 14 位时间戳（可带冲突序号）")
        return self.directory / f"{name}.json"

    def create(self, script_group_names: Iterable[str]) -> TaskProgress:
        base = datetime.now().strftime("%Y%m%d%H%M%S")
        name = base
        suffix = 0
        while self.path_for(name).exists():
            suffix += 1
            name = f"{base}-{suffix}"
        return TaskProgress([str(value) for value in script_group_names], name=name)

    def save(self, progress: TaskProgress) -> Path:
        path = self.path_for(progress.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(progress.to_mapping(), ensure_ascii=False, indent=2)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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
        return path

    def load(self, source: str | Path) -> TaskProgress:
        path = Path(source).expanduser()
        if not path.exists() and path.parent == Path("."):
            name = path.stem if path.suffix == ".json" else str(path)
            path = self.path_for(name)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"TaskProgress 根节点必须是对象：{path}")
        return TaskProgress.from_mapping(raw)

    def load_active(self, *, now: datetime | None = None) -> list[TaskProgress]:
        self.directory.mkdir(parents=True, exist_ok=True)
        current = now or datetime.now()
        active: list[tuple[float, TaskProgress]] = []
        for path in self.directory.glob("*.json"):
            if not _PROGRESS_NAME.fullmatch(path.stem):
                continue
            try:
                modified = datetime.fromtimestamp(path.stat().st_mtime)
                if current - modified > timedelta(days=3):
                    path.unlink()
                    continue
                progress = self.load(path)
                if progress.end_time is None:
                    active.append((path.stat().st_mtime, progress))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                self.log(f"[TaskProgress] 忽略无效进度 {path.name}: {error}")
        return [item for _, item in sorted(active, key=lambda value: value[0], reverse=True)]


def resume_flat_index(progress: TaskProgress,
                      projects: Iterable[tuple[str, int, str, str]]) -> int:
    """Return the first project to retry/run in a flattened group sequence."""
    values = list(projects)
    if progress.end_time is not None:
        return len(values)
    current = progress.current
    if current is not None and (not current.task_end or current.status == 2):
        for flat_index, value in enumerate(values):
            if value == (current.group_name, current.index, current.name, current.folder_name):
                return flat_index
    previous = progress.last_success
    if previous is not None:
        for flat_index, value in enumerate(values):
            if value == (previous.group_name, previous.index, previous.name, previous.folder_name):
                return flat_index + 1
    return 0
