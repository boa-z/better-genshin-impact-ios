import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch


def test_devicehub_config_loads_headless_path_and_resolves_command(tmp_path: Path):
    from bgi_touch.device.config import DeviceHubConfig

    binary = tmp_path / "bin" / "devicehub-headless"
    config_path = tmp_path / "config" / "devicehub.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps({
        "mcpUrl": "http://127.0.0.1:9010/mcp",
        "headless": {
            "executable": "../bin/devicehub-headless",
            "workingDirectory": "../bin",
            "args": ["--listen", "127.0.0.1:8080"],
            "startupTimeoutSeconds": 7,
            "shutdownOnExit": False,
        },
    }), encoding="utf-8")

    config = DeviceHubConfig.load(config_path)
    assert config.mcp_url == "http://127.0.0.1:9010/mcp"
    assert config.headless.executable == binary.resolve()
    assert config.headless.working_directory == binary.parent.resolve()
    assert config.headless.startup_timeout_s == 7
    assert not config.headless.shutdown_on_exit
    assert config.headless.command(config.mcp_url) == [
        str(binary.resolve()), "--listen", "127.0.0.1:8080",
        "--mcp-listen", "127.0.0.1:9010",
    ]


def test_devicehub_config_without_file_keeps_default_mcp_url(tmp_path: Path):
    from bgi_touch.device.config import DeviceHubConfig, DEFAULT_MCP_URL

    config = DeviceHubConfig.load(tmp_path / "missing.json")
    assert config.mcp_url == DEFAULT_MCP_URL
    assert config.headless.executable is None


def test_headless_start_uses_configured_executable_and_working_directory(tmp_path: Path):
    from bgi_touch.device.client import DeviceClient
    from bgi_touch.device.config import HeadlessConfig

    client = DeviceClient.__new__(DeviceClient)
    client.url = "http://127.0.0.1:9010/mcp"
    client.headless = HeadlessConfig(
        executable=Path(sys.executable),
        working_directory=tmp_path,
        args=("--fake-devicehub-arg",),
    )
    client._headless_process = None
    process = Mock()

    with patch("bgi_touch.device.client.subprocess.Popen", return_value=process) as popen:
        client._start_headless()

    assert popen.call_args.args[0] == [
        sys.executable,
        "--fake-devicehub-arg",
        "--mcp-listen",
        "127.0.0.1:9010",
    ]
    assert popen.call_args.kwargs["cwd"] == str(tmp_path)
    assert client._headless_process is process
