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


def test_devicehub_config_loads_device_id_and_environment_overrides_it(
        tmp_path: Path, monkeypatch):
    from bgi_touch.device.config import DeviceHubConfig

    config_path = tmp_path / "devicehub.json"
    config_path.write_text(json.dumps({
        "deviceId": "iphone-from-config::wifi",
        "disableInputMonitor": True,
    }), encoding="utf-8")

    monkeypatch.delenv("BGI_DEVICE_ID", raising=False)
    assert DeviceHubConfig.load(config_path).device_id == "iphone-from-config::wifi"

    monkeypatch.setenv("BGI_DEVICE_ID", "iphone-from-env::usb")
    assert DeviceHubConfig.load(config_path).device_id == "iphone-from-env::usb"


def test_devicehub_config_loads_game_bundle_id_and_environment_override(
        tmp_path: Path, monkeypatch):
    from bgi_touch.device.config import DeviceHubConfig

    config_path = tmp_path / "devicehub.json"
    config_path.write_text(json.dumps({
        "gameBundleId": "com.example.genshin",
    }), encoding="utf-8")

    monkeypatch.delenv("BGI_GAME_BUNDLE_ID", raising=False)
    monkeypatch.delenv("BGI_GENSHIN_BUNDLE_ID", raising=False)
    assert DeviceHubConfig.load(config_path).game_bundle_id == "com.example.genshin"

    monkeypatch.setenv("BGI_GAME_BUNDLE_ID", "com.example.from-env")
    assert DeviceHubConfig.load(config_path).game_bundle_id == "com.example.from-env"


def test_devicehub_config_loads_disable_input_monitor_and_environment_override(
        tmp_path: Path, monkeypatch):
    from bgi_touch.device.config import DeviceHubConfig

    config_path = tmp_path / "devicehub.json"
    config_path.write_text(json.dumps({"disableInputMonitor": True}), encoding="utf-8")

    monkeypatch.delenv("BGI_DISABLE_INPUT_MONITOR", raising=False)
    assert DeviceHubConfig.load(config_path).disable_input_monitor is True

    monkeypatch.setenv("BGI_DISABLE_INPUT_MONITOR", "false")
    assert DeviceHubConfig.load(config_path).disable_input_monitor is False


def test_game_context_passes_exact_device_id_to_devicehub(tmp_path: Path):
    from bgi_touch.engine.context import GameContext

    config_path = tmp_path / "devicehub.json"
    config_path.write_text(json.dumps({
        "deviceId": "iphone-from-config::wifi",
        "disableInputMonitor": True,
    }), encoding="utf-8")
    device = Mock()
    device.status.return_value = {
        "status": "connected",
        "screen_size": [2778, 1284],
    }

    with patch("bgi_touch.engine.context.DeviceClient", return_value=device), \
            patch.object(GameContext, "capture_bgr", return_value=None), \
            patch.object(GameContext, "_start_orientation_watch"):
        context = GameContext(
            devicehub_config_path=config_path,
            device_id="iphone-from-cli::wifi",
            keymap_profile=None,
        )

    assert context.device_id == "iphone-from-cli::wifi"
    assert context.disable_input_monitor is True
    device.connect_device.assert_called_once_with("iphone-from-cli::wifi")


def test_game_context_prefers_profile_bundle_id_for_lifecycle_and_game_session(
        tmp_path: Path):
    from bgi_touch.engine.context import GameContext

    profile_path = tmp_path / "Genshin.json"
    profile_path.write_text(json.dumps({
        "version": 2,
        "name": "Genshin",
        "bundleIdentifiers": ["com.miHoYo.GenshinImpact"],
        "mappings": [],
    }), encoding="utf-8")
    device = Mock()
    device.status.return_value = {
        "status": "connected",
        "screen_size": [2778, 1284],
    }

    with patch("bgi_touch.engine.context.DeviceClient", return_value=device), \
            patch.object(GameContext, "capture_bgr", return_value=None), \
            patch.object(GameContext, "_start_orientation_watch"):
        context = GameContext(
            keymap_profile=None,
            keymap_profile_path=profile_path,
        )

    assert context.game_bundle_id == "com.miHoYo.GenshinImpact"
    context.sleep = Mock()
    context.launch_game(auto_enter=False)
    device.launch_app.assert_called_once_with(
        "com.miHoYo.GenshinImpact", wait=True,
    )
    context.input._ensure_profile_session()
    assert device.start_game_session.call_args.kwargs["bundle_id"] == (
        "com.miHoYo.GenshinImpact"
    )
    context.input.release_all()


def test_explicit_game_bundle_id_overrides_profile(tmp_path: Path):
    from bgi_touch.engine.context import GameContext

    profile_path = tmp_path / "Genshin.json"
    profile_path.write_text(json.dumps({
        "version": 2,
        "name": "Genshin",
        "bundleIdentifiers": ["com.miHoYo.GenshinImpact"],
        "mappings": [],
    }), encoding="utf-8")
    device = Mock()
    device.status.return_value = {
        "status": "connected",
        "screen_size": [2778, 1284],
    }

    with patch("bgi_touch.engine.context.DeviceClient", return_value=device), \
            patch.object(GameContext, "capture_bgr", return_value=None), \
            patch.object(GameContext, "_start_orientation_watch"):
        context = GameContext(
            keymap_profile=None,
            keymap_profile_path=profile_path,
            game_bundle_id="com.example.custom-genshin",
        )

    assert context.game_bundle_id == "com.example.custom-genshin"


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


def test_device_client_uses_exact_selection_id_and_tracks_frame_versions():
    from bgi_touch.device.client import DeviceClient, ToolResult

    client = DeviceClient.__new__(DeviceClient)
    calls = []

    def fake_call(name, **kwargs):
        calls.append((name, kwargs))
        if name == "status":
            return ToolResult({
                "device_id": "udid::usb",
                "screen_size": [1296, 2816],
            }, None, None)
        if name == "connect_device":
            return ToolResult({"connected": True}, None, None)
        if name == "wait_for_frame":
            return ToolResult({"frame_version": 42, "ready": True}, None, None)
        raise AssertionError(name)

    client.call = fake_call
    client._last_frame_version = None
    assert client.connect_device() == {"connected": True}
    assert calls[1] == ("connect_device", {"udid": "udid::usb"})
    assert client.wait_for_frame(after_version=41, timeout_ms=500)["frame_version"] == 42


def test_device_client_prefers_action_after_cursor_and_never_rolls_back():
    import threading
    from types import SimpleNamespace

    from bgi_touch.device.client import DeviceClient

    client = DeviceClient.__new__(DeviceClient)
    client._lock = threading.Lock()
    client._session = SimpleNamespace()
    client._session.call_tool = Mock()
    client._reconnect = Mock(side_effect=AssertionError("不应重连"))
    client._last_frame_version = None

    def response(payload):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(payload))],
            isError=False,
        )

    client._session.call_tool.side_effect = [
        object(), object(), object(),
    ]
    client._run = Mock(side_effect=[
        response({
            "frame_version_before": 41,
            "frame_version_after": 44,
        }),
        response({"frame_version": 42}),
        response({"frameVersionAfter": "47"}),
    ])

    client.call("swipe")
    assert client.last_frame_version == 44
    # A delayed screenshot response must not move an observation cursor back.
    client.call("screenshot")
    assert client.last_frame_version == 44
    client.call("tap")
    assert client.last_frame_version == 47


def test_device_client_portrait_mapper_preserves_landscape_dimensions():
    from bgi_touch.device.client import DeviceClient

    client = DeviceClient.__new__(DeviceClient)
    client._mapper = lambda x, y, iw=None, ih=None: (ih - y, x, ih, iw)
    assert client._map(100, 200, 2778, 1284) == (1084, 100, 1284, 2778)


def test_device_client_selects_only_device_but_not_ambiguous_devices():
    import pytest

    from bgi_touch.device.client import DeviceClient, DeviceError, ToolResult

    client = DeviceClient.__new__(DeviceClient)
    connected = []
    devices = [{
        "active": False,
        "id": "iphone::wifi",
        "name": "iPhone 13 Pro Max",
    }]

    def fake_call(name, **kwargs):
        if name == "status":
            return ToolResult({"device_id": None, "active_udid": None}, None, None)
        if name == "list_devices":
            return ToolResult({"devices": devices}, None, None)
        if name == "connect_device":
            connected.append(kwargs["udid"])
            return ToolResult({"connected": True}, None, None)
        raise AssertionError(name)

    client.call = fake_call
    assert client.connect_device() == {"connected": True}
    assert connected == ["iphone::wifi"]

    devices.append({"active": False, "id": "tablet::wifi", "name": "Tablet"})
    with pytest.raises(DeviceError, match="未返回可连接的设备 ID"):
        client.connect_device()
