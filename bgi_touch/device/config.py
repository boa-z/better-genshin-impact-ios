"""DeviceHub MCP and headless host configuration."""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "devicehub.json"
DEFAULT_MCP_URL = "http://127.0.0.1:8009/mcp"


def _first(raw: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in raw:
            return raw[name]
    return default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    return default


def _as_args(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(shlex.split(value))
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return ()


def _resolve_path(value: Any, base_dir: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


@dataclass(frozen=True)
class HeadlessConfig:
    """How bgi-touch may start a local ``devicehub-headless`` host."""

    executable: Path | None = None
    working_directory: Path | None = None
    args: tuple[str, ...] = ()
    auto_start: bool = True
    startup_timeout_s: float = 20.0
    shutdown_on_exit: bool = True

    def command(self, mcp_url: str) -> list[str]:
        if self.executable is None:
            raise ValueError("未配置 devicehub-mask headless 程序位置")
        args = list(self.args)
        if not any(arg == "--mcp-listen" or arg.startswith("--mcp-listen=") for arg in args):
            parsed = urlparse(mcp_url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 8009
            args.extend(("--mcp-listen", f"{host}:{port}"))
        return [str(self.executable), *args]


@dataclass(frozen=True)
class DeviceHubConfig:
    mcp_url: str = DEFAULT_MCP_URL
    device_id: str | None = None
    headless: HeadlessConfig = HeadlessConfig()
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "DeviceHubConfig":
        configured_path = path or os.environ.get("BGI_DEVICEHUB_CONFIG")
        config_path = Path(configured_path).expanduser() if configured_path else DEFAULT_CONFIG_PATH
        config_path = config_path.resolve()
        if not config_path.exists():
            return cls(
                mcp_url=os.environ.get("BGI_MCP_URL", DEFAULT_MCP_URL),
                device_id=os.environ.get("BGI_DEVICE_ID") or None,
                path=config_path,
            )

        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"DeviceHub 配置 JSON 无效：{config_path}: {error}") from error
        if not isinstance(raw, dict):
            raise ValueError(f"DeviceHub 配置根节点必须是对象：{config_path}")

        configured_url = _first(raw, "mcp_url", "mcpUrl", "url")
        mcp_url = os.environ.get("BGI_MCP_URL") or str(configured_url or DEFAULT_MCP_URL)
        configured_device = _first(raw, "device_id", "deviceId", "udid", "device")
        device_id = os.environ.get("BGI_DEVICE_ID") or (
            str(configured_device).strip() if configured_device else None
        )
        nested = _first(raw, "headless", "devicehub_headless", "devicehubHeadless", default={})
        if isinstance(nested, str):
            nested = {"executable": nested}
        if not isinstance(nested, dict):
            nested = {}

        executable_value = _first(
            nested,
            "executable",
            "executable_path",
            "executablePath",
            "program",
            "program_path",
            "programPath",
            "path",
        )
        if executable_value is None:
            executable_value = _first(raw, "headless_path", "headlessPath", "devicehubHeadlessPath")
        executable = _resolve_path(executable_value, config_path.parent)
        working_directory = _resolve_path(
            _first(nested, "working_directory", "workingDirectory", "cwd"),
            config_path.parent,
        )
        raw_timeout = _first(
            nested,
            "startup_timeout_seconds",
            "startupTimeoutSeconds",
            "startup_timeout",
            "startupTimeout",
            default=20,
        )
        try:
            timeout = max(1.0, min(120.0, float(raw_timeout)))
        except (TypeError, ValueError):
            timeout = 20.0
        auto_start = _as_bool(
            _first(nested, "auto_start", "autoStart", default=_first(raw, "auto_start", "autoStart", default=True)),
            True,
        )
        shutdown_on_exit = _as_bool(
            _first(nested, "shutdown_on_exit", "shutdownOnExit", default=True),
            True,
        )
        return cls(
            mcp_url=mcp_url,
            device_id=device_id,
            headless=HeadlessConfig(
                executable=executable,
                working_directory=working_directory,
                args=_as_args(_first(nested, "args", "arguments", default=())),
                auto_start=auto_start,
                startup_timeout_s=timeout,
                shutdown_on_exit=shutdown_on_exit,
            ),
            path=config_path,
        )
