"""Cross-platform BetterGI ShellTask with an explicit host safety gate."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "shell.json"


def _field(raw: dict, *names: str, default=None):
    for name in names:
        if name in raw:
            return raw[name]
    return default


def _bool(value, default=False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in ("1", "true", "yes", "on", "是"):
            return True
        if normalized in ("0", "false", "no", "off", "否", ""):
            return False
    return default if value is None else bool(value)


@dataclass(frozen=True)
class ShellHostConfig:
    enabled: bool = False
    timeout_s: float = 60
    no_window: bool = True
    output: bool = True
    working_directory: Path = PROJECT_ROOT
    max_output_chars: int = 200_000

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ShellHostConfig":
        config_path = Path(
            path or os.getenv("BGI_SHELL_CONFIG") or DEFAULT_CONFIG
        ).expanduser()
        raw = {}
        if config_path.is_file():
            value = json.loads(config_path.read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict):
                raise ValueError("Shell 配置根节点必须是对象")
            raw = value
        enabled = _bool(_field(raw, "enabled", default=False))
        env_enabled = os.getenv("BGI_SHELL_ENABLED")
        if env_enabled is not None:
            enabled = _bool(env_enabled)
        working = Path(str(_field(
            raw, "workingDirectory", "working_directory", default=".."
        ))).expanduser()
        if not working.is_absolute():
            working = (config_path.parent / working).resolve()
        return cls(
            enabled=enabled,
            timeout_s=float(_field(
                raw, "timeoutSeconds", "timeout", default=60
            ) or 0),
            no_window=_bool(_field(raw, "noWindow", "no_window", default=True), True),
            output=_bool(_field(raw, "output", default=True), True),
            working_directory=working,
            max_output_chars=max(1000, min(2_000_000, int(_field(
                raw, "maxOutputChars", "max_output_chars", default=200_000
            ) or 200_000))),
        )


@dataclass(frozen=True)
class ShellResult:
    command: str
    status: str
    pid: int | None = None
    return_code: int | None = None
    output: str = ""
    timed_out: bool = False
    cancelled: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class ShellTask:
    def __init__(
        self,
        command: str,
        *,
        config_path: str | Path | None = None,
        timeout_s: float | None = None,
        no_window: bool | None = None,
        output: bool | None = None,
        disable: bool = False,
        working_directory: str | Path | None = None,
        log: Callable[[str], None] = print,
    ):
        self.command = str(command or "")
        self.config = ShellHostConfig.load(config_path)
        self.timeout_s = self.config.timeout_s if timeout_s is None else float(timeout_s)
        self.no_window = self.config.no_window if no_window is None else bool(no_window)
        self.output_enabled = self.config.output if output is None else bool(output)
        self.disable = bool(disable)
        self.working_directory = (
            Path(working_directory).expanduser().resolve()
            if working_directory else self.config.working_directory
        )
        self.log = log

    @staticmethod
    def _argv(command: str) -> list[str]:
        if os.name == "nt":
            return [os.environ.get("ComSpec", "cmd.exe"), "/d", "/s", "/c", command]
        shell = os.environ.get("SHELL") or "/bin/sh"
        return [shell, "-lc", command]

    def _popen_kwargs(self, wait: bool) -> dict:
        kwargs = {
            "cwd": str(self.working_directory),
            "stdin": subprocess.DEVNULL,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if wait and self.output_enabled:
            kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        else:
            kwargs.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.name == "nt":
            creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if self.no_window:
                creation_flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
            kwargs["creationflags"] = creation_flags
        else:
            kwargs["start_new_session"] = True
        return kwargs

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=1.5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

    def run(self, cancelled: Callable[[], bool] | None = None) -> dict:
        if not self.config.enabled or self.disable:
            self.log("[Shell] Shell 任务已由 config/shell.json 禁用")
            return ShellResult(self.command, "disabled").to_dict()
        if not self.command.strip():
            self.log("[Shell] 命令为空，跳过")
            return ShellResult(self.command, "empty").to_dict()
        if not self.working_directory.is_dir():
            raise NotADirectoryError(f"Shell 工作目录不存在：{self.working_directory}")
        if cancelled and cancelled():
            return ShellResult(self.command, "cancelled", cancelled=True).to_dict()

        wait = self.timeout_s > 0
        self.log(
            f"[Shell] 执行：{self.command}；"
            + (f"超时 {self.timeout_s:g}s" if wait else "不等待")
        )
        process = subprocess.Popen(
            self._argv(self.command), **self._popen_kwargs(wait)
        )
        if not wait:
            return ShellResult(
                self.command, "started", pid=process.pid
            ).to_dict()

        deadline = time.monotonic() + self.timeout_s
        output = ""
        while True:
            if cancelled and cancelled():
                self._terminate(process)
                self.log(f"[Shell] 已取消：{self.command}")
                return ShellResult(
                    self.command,
                    "cancelled",
                    pid=process.pid,
                    return_code=process.returncode,
                    cancelled=True,
                ).to_dict()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate(process)
                self.log(f"[Shell] 执行超时：{self.command}")
                return ShellResult(
                    self.command,
                    "timeout",
                    pid=process.pid,
                    return_code=process.returncode,
                    timed_out=True,
                ).to_dict()
            try:
                output, _ = process.communicate(timeout=min(0.2, remaining))
                break
            except subprocess.TimeoutExpired:
                continue

        output = output or ""
        if len(output) > self.config.max_output_chars:
            output = output[:self.config.max_output_chars] + "\n…[输出已截断]"
        status = "completed" if process.returncode == 0 else "failed"
        if self.output_enabled and output:
            self.log(f"[Shell] 输出：{output.rstrip()}")
        self.log(f"[Shell] 结束，退出码 {process.returncode}")
        return ShellResult(
            self.command,
            status,
            pid=process.pid,
            return_code=process.returncode,
            output=output if self.output_enabled else "",
        ).to_dict()
