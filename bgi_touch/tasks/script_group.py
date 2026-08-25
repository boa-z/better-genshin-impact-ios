"""BetterGI ScriptGroup parser and cross-platform project scheduler."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..config_values import as_bool
from .execution_records import (
    CompletionSkipRule,
    ExecutionRecord,
    ExecutionRecordStore,
    TaskScheduleRule,
)
from .task_progress import TaskProgress, TaskProgressStore, resume_flat_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _value(raw: Mapping[str, Any], *names: str, default=None):
    folded = {str(key).replace("_", "").casefold(): value for key, value in raw.items()}
    for name in names:
        key = name.replace("_", "").casefold()
        if key in folded:
            return folded[key]
    return default


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class ScriptGroupRoots:
    javascript: Path = PROJECT_ROOT / "scripts" / "js"
    key_mouse: Path = PROJECT_ROOT / "scripts" / "keymouse"
    pathing: Path = PROJECT_ROOT / "scripts" / "pathing"

    @classmethod
    def build(cls, *, javascript=None, key_mouse=None, pathing=None) -> "ScriptGroupRoots":
        def path(value, default):
            return Path(value or default).expanduser().resolve()

        return cls(
            javascript=path(javascript, cls.javascript),
            key_mouse=path(key_mouse, cls.key_mouse),
            pathing=path(pathing, cls.pathing),
        )


@dataclass
class ScriptGroupProject:
    name: str
    folder_name: str = ""
    type: str = "Javascript"
    status: str = "Enabled"
    schedule: str = "Daily"
    run_num: int = 1
    settings: dict[str, Any] = field(default_factory=dict)
    allow_js_notification: bool = True
    allow_js_http_hash: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ScriptGroupProject":
        name = str(_value(raw, "name", default="") or "").strip()
        if not name:
            raise ValueError("ScriptGroup 项目缺少 Name")
        project_type = str(_value(raw, "type", default="Javascript") or "Javascript").strip()
        aliases = {
            "javascript": "Javascript", "js": "Javascript",
            "keymouse": "KeyMouse", "macro": "KeyMouse",
            "pathing": "Pathing", "path": "Pathing",
            "shell": "Shell",
        }
        normalized_type = aliases.get(project_type.casefold())
        if normalized_type is None:
            raise ValueError(f"不支持的 ScriptGroup 项目类型：{project_type}")
        settings = _mapping(_value(raw, "jsScriptSettingsObject", "settings", default={}))
        try:
            run_num = max(1, min(100, int(_value(raw, "runNum", default=1) or 1)))
        except (TypeError, ValueError):
            run_num = 1
        return cls(
            name=name,
            folder_name=str(_value(raw, "folderName", default="") or "").strip(),
            type=normalized_type,
            status=str(_value(raw, "status", default="Enabled") or "Enabled"),
            schedule=str(_value(raw, "schedule", default="Daily") or "Daily"),
            run_num=run_num,
            settings=settings,
            allow_js_notification=as_bool(_value(
                raw, "allowJsNotification", default=True,
            ), True),
            allow_js_http_hash=str(_value(raw, "allowJsHTTPHash", default="") or ""),
        )

    @property
    def enabled(self) -> bool:
        return self.status.strip().casefold() not in {
            "disabled", "disable", "false", "0", "禁用",
        }


@dataclass
class ScriptGroup:
    name: str
    projects: list[ScriptGroupProject]
    config: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, source_path=None) -> "ScriptGroup":
        projects = _value(raw, "projects", default=[])
        if not isinstance(projects, list):
            raise ValueError("ScriptGroup.Projects 必须是数组")
        source = Path(source_path).expanduser().resolve() if source_path else None
        name = str(_value(raw, "name", default=source.stem if source else "") or "").strip()
        if not name:
            raise ValueError("ScriptGroup 缺少 Name")
        return cls(
            name=name,
            projects=[
                ScriptGroupProject.from_mapping(value)
                for value in projects if isinstance(value, Mapping)
            ],
            config=_mapping(_value(raw, "config", default={})),
            source_path=source,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ScriptGroup":
        source = Path(path).expanduser().resolve()
        raw = json.loads(source.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"ScriptGroup 根节点必须是对象：{source}")
        return cls.from_mapping(raw, source_path=source)


class ScriptGroupCancelled(RuntimeError):
    pass


class ScriptGroupRunner:
    def __init__(
        self,
        ctx,
        groups: Sequence[ScriptGroup],
        *,
        roots: ScriptGroupRoots | None = None,
        party_slots: dict[str, int] | None = None,
        progress_store: TaskProgressStore | None = None,
        execution_store: ExecutionRecordStore | None = None,
        continue_on_error: bool = True,
        cancelled: Callable[[], bool] | None = None,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.groups = list(groups)
        self.roots = roots or ScriptGroupRoots.build()
        self.party_slots = party_slots or {}
        self.progress_store = progress_store or TaskProgressStore(log=log)
        self.execution_store = execution_store or ExecutionRecordStore()
        self.continue_on_error = as_bool(continue_on_error, True)
        self.cancelled = cancelled or (lambda: False)
        self.log = log

    @classmethod
    def load(cls, ctx, paths: Sequence[str | Path], **kwargs) -> "ScriptGroupRunner":
        return cls(ctx, [ScriptGroup.load(path) for path in paths], **kwargs)

    def _enabled(self):
        for group in self.groups:
            for index, project in enumerate(group.projects):
                if project.enabled:
                    yield group, index, project

    def describe(self) -> dict[str, Any]:
        projects = [
            {
                "group": group.name,
                "index": index,
                "name": project.name,
                "folderName": project.folder_name,
                "type": project.type,
                "runNum": project.run_num,
            }
            for group, index, project in self._enabled()
        ]
        return {
            "groups": [group.name for group in self.groups],
            "projects": projects,
            "count": len(projects),
        }

    def run(self, *, resume: TaskProgress | str | Path | None = None) -> dict[str, Any]:
        enabled = list(self._enabled())
        identities = [
            (group.name, index, project.name, project.folder_name)
            for group, index, project in enabled
        ]
        if isinstance(resume, TaskProgress):
            progress = resume
        elif resume is not None:
            progress = self.progress_store.load(resume)
        else:
            progress = self.progress_store.create(group.name for group in self.groups)
        start_index = resume_flat_index(progress, identities) if resume is not None else 0
        self.progress_store.save(progress)

        completed = 0
        failed = 0
        skipped = start_index
        try:
            for group, project_index, project in enabled[start_index:]:
                if self.cancelled():
                    raise ScriptGroupCancelled("配置组执行已取消")
                schedule_reason = TaskScheduleRule.from_group_config(
                    group.config
                ).skip_reason()
                if schedule_reason:
                    skipped += 1
                    self.log(f"[ScriptGroup] {project.name}: {schedule_reason}，跳过此任务")
                    continue
                skip_rule = CompletionSkipRule.from_group_config(group.config)
                should_skip, skip_reason = self.execution_store.should_skip(
                    group.name,
                    project.name,
                    project.folder_name,
                    project.type,
                    skip_rule,
                )
                if should_skip:
                    skipped += 1
                    self.log(f"[ScriptGroup] {project.name}: {skip_reason}，跳过此任务")
                    continue
                progress.begin(group.name, project_index, project.name, project.folder_name)
                self.progress_store.save(progress)
                success = True
                error_text = None
                self.log(
                    f"[ScriptGroup] {group.name} {project_index + 1}/{len(group.projects)} "
                    f"{project.type}: {project.name}"
                )
                for run_index in range(project.run_num):
                    if self.cancelled():
                        raise ScriptGroupCancelled("配置组执行已取消")
                    execution_record = None
                    if skip_rule.enabled:
                        execution_record = ExecutionRecord.start(
                            group.name,
                            project.name,
                            project.folder_name,
                            project.type,
                        )
                        self.execution_store.save(execution_record)
                    try:
                        self._execute_project(group, project)
                    except ScriptGroupCancelled:
                        if execution_record is not None:
                            self.execution_store.finish(execution_record, False)
                        raise
                    except Exception as error:
                        if execution_record is not None:
                            self.execution_store.finish(execution_record, False)
                        success = False
                        error_text = str(error)
                        self.log(
                            f"[ScriptGroup] {project.name} 第 {run_index + 1}/{project.run_num} 次失败：{error}"
                        )
                        break
                    else:
                        if execution_record is not None:
                            self.execution_store.finish(execution_record, True)
                progress.finish_current(success)
                self.progress_store.save(progress)
                if success:
                    completed += 1
                else:
                    failed += 1
                    if not self.continue_on_error:
                        raise RuntimeError(
                            f"配置组项目失败：{group.name}/{project.name}: {error_text}"
                        )
            progress.end_time = datetime.now().isoformat()
            progress.current = None
            progress.current_group_name = None
            self.progress_store.save(progress)
            return {
                "status": "completed" if failed == 0 else "completed_with_errors",
                "progress": str(self.progress_store.path_for(progress.name)),
                "completed": completed,
                "failed": failed,
                "skipped": skipped,
            }
        except BaseException:
            self.progress_store.save(progress)
            raise
        finally:
            if self.ctx is not None:
                triggers = getattr(self.ctx, "triggers", None)
                if callable(getattr(triggers, "clear", None)):
                    triggers.clear()
                input_sim = getattr(self.ctx, "input", None)
                if callable(getattr(input_sim, "release_all", None)):
                    input_sim.release_all()

    def _candidate(self, group: ScriptGroup, root: Path, *parts: str) -> Path:
        relative = Path(*[part for part in parts if part])
        candidates = []
        if relative.is_absolute():
            candidates.append(relative)
        if group.source_path is not None:
            candidates.append(group.source_path.parent / relative)
        candidates.append(root / relative)
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if resolved.exists():
                return resolved
        return candidates[-1].expanduser().resolve()

    def _execute_project(self, group: ScriptGroup, project: ScriptGroupProject) -> None:
        if project.type == "Javascript":
            from ..engine.js_runtime import JsScriptRuntime

            folder = project.folder_name or project.name
            path = self._candidate(group, self.roots.javascript, folder)
            if not path.is_dir():
                raise FileNotFoundError(f"JS 脚本目录不存在：{path}")
            JsScriptRuntime(
                self.ctx,
                path,
                settings=project.settings,
                party_slots=self.party_slots,
                pathing_root=self.roots.pathing,
                pathing_config=group.config,
                log=self.log,
            ).run()
            return
        if project.type == "KeyMouse":
            from ..macro.keymouse import MacroPlayer, load_keymouse

            path = self._candidate(
                group, self.roots.key_mouse, project.folder_name, project.name
            )
            if not path.is_file():
                raise FileNotFoundError(f"键鼠脚本不存在：{path}")
            MacroPlayer(self.ctx.input, sleep=self.ctx.sleep, log=self.log).play(
                load_keymouse(path)
            )
            return
        if project.type == "Pathing":
            from ..pathing.executor import PathingExecutor
            from ..pathing.model import PathingTask
            from ..pathing.party_config import PathingPartyConfig

            path = self._candidate(
                group, self.roots.pathing, project.folder_name, project.name
            )
            if not path.is_file():
                raise FileNotFoundError(f"地图追踪脚本不存在：{path}")
            task = PathingTask.load(path)
            pathing_config = PathingPartyConfig.from_mapping(group.config)
            ok = PathingExecutor(
                self.ctx,
                party_slots=self.party_slots,
                pathing_config=pathing_config,
                farming_route_info={
                    "group_name": group.name,
                    "project_name": project.name,
                    "folder_name": project.folder_name,
                },
                log=self.log,
            ).run(task)
            if not ok:
                raise RuntimeError("地图追踪脚本未正常走完")
            return
        if project.type == "Shell":
            from .dispatcher import TaskDispatcher

            group_config = _mapping(_value(group.config, "shellConfig", default={}))
            result = TaskDispatcher(None, log=self.log).run_shell_task({
                **group_config,
                "command": project.name,
                "disable": not as_bool(_value(
                    group.config, "enableShellConfig", default=False,
                )),
            })
            if result.get("status") in {"failed", "timeout"}:
                raise RuntimeError(
                    f"Shell {result['status']} (return_code={result.get('return_code')})"
                )
            return
        raise ValueError(f"不支持的 ScriptGroup 项目类型：{project.type}")
