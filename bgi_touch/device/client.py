"""devicehub-mask MCP 客户端（同步外观）。

BetterGI 脚本与任务代码是同步调用风格（原版是 ClearScript 绑定同步 C# 方法），
因此在后台线程跑一个 asyncio 事件循环持有 MCP 会话，对外提供阻塞式方法。
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import threading
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession

# The MCP Python SDK renamed this context manager in 2.0.  DeviceHub Mask
# still speaks the same Streamable HTTP protocol, so keep the client usable
# with both the 1.x dependency used by the original port and current 2.x
# runtimes instead of making an otherwise unrelated SDK upgrade break import.
try:
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:  # pragma: no cover - selected by the installed SDK version
    from mcp.client.streamable_http import streamable_http_client as streamablehttp_client

from .config import DEFAULT_MCP_URL, HeadlessConfig

DEFAULT_URL = DEFAULT_MCP_URL
CALL_TIMEOUT_S = 120
# DeviceHub status endpoints are substantially slower than local frame-cache
# reads on an iPhone/Wi-Fi session.  Keep a short cache for observers such as
# the WebUI and orientation watcher; explicit reconnect/connection paths use
# ``force=True`` below.  Screenshot and input tools are never cached here.
STATUS_CACHE_TTL_S = 8.0
APP_STATUS_CACHE_TTL_S = 8.0


@dataclass
class ToolResult:
    json: Any | None
    image: bytes | None
    text: str | None


class DeviceClient:
    """所有坐标均为截图像素坐标；动作调用附带 image_width/height 保证确定性缩放。"""

    def __init__(self, url: str = DEFAULT_URL, headless: HeadlessConfig | None = None):
        self.url = url
        self.headless = headless
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-loop")
        self._thread.start()
        self._session: ClientSession | None = None
        self._close_event: asyncio.Event | None = None
        self._session_task = None
        self._lock = threading.Lock()
        self._status_cache_lock = threading.RLock()
        self._status_cache: dict | None = None
        self._status_cache_at = 0.0
        self._app_status_cache: dict[str, tuple[dict, float]] = {}
        self._mapper = None
        self._last_image_size: tuple[int, int] | None = None
        self._last_frame_version: int | None = None
        self._game_session_id: str | None = None
        self._headless_process: subprocess.Popen | None = None
        try:
            self._start_session()
        except Exception:
            if headless is None or not headless.auto_start or headless.executable is None:
                raise
            self._start_headless()
            try:
                self._start_session_until_ready(headless.startup_timeout_s)
            except Exception:
                self._stop_headless()
                raise

    # ---- plumbing ----
    # anyio 的 task group 必须在进入它的同一协程内退出，因此把整个会话生命
    # 周期放进一个驻留协程：连接→就绪→等待关闭事件→原地退出。close/重连
    # 只是置位事件并等待该协程结束，避免跨任务关闭导致 cancel scope 报错。

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=CALL_TIMEOUT_S)

    def _start_session(self) -> None:
        import concurrent.futures

        ready: concurrent.futures.Future = concurrent.futures.Future()

        async def runner():
            self._close_event = asyncio.Event()
            try:
                async with AsyncExitStack() as stack:
                    read, write, _ = await stack.enter_async_context(streamablehttp_client(self.url))
                    session = await stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                    self._session = session
                    ready.set_result(None)
                    await self._close_event.wait()
            except Exception as e:
                if not ready.done():
                    ready.set_exception(e)
            finally:
                self._session = None

        self._session_task = asyncio.run_coroutine_threadsafe(runner(), self._loop)
        ready.result(timeout=CALL_TIMEOUT_S)

    def _start_session_until_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while True:
            try:
                self._start_session()
                return
            except Exception as error:
                last_error = error
                self._end_session()
                if time.monotonic() >= deadline:
                    raise last_error
                time.sleep(0.25)

    def _start_headless(self) -> None:
        assert self.headless is not None
        executable = self.headless.executable
        assert executable is not None
        if executable.is_dir():
            candidates = (executable / "devicehub-headless", executable / "devicehub-headless.exe")
            executable = next((candidate for candidate in candidates if candidate.is_file()), executable)
        if not executable.is_file():
            raise FileNotFoundError(f"devicehub-mask headless 程序不存在：{executable}")
        if os.name != "nt" and not os.access(executable, os.X_OK):
            raise PermissionError(f"devicehub-mask headless 程序不可执行：{executable}")
        cwd = self.headless.working_directory or executable.parent
        if not cwd.is_dir():
            raise NotADirectoryError(f"headless 工作目录不存在：{cwd}")
        command = HeadlessConfig(
            executable=executable,
            working_directory=cwd,
            args=self.headless.args,
            auto_start=self.headless.auto_start,
            startup_timeout_s=self.headless.startup_timeout_s,
            shutdown_on_exit=self.headless.shutdown_on_exit,
        ).command(self.url)
        print(f"[device] MCP 不可用，启动 headless：{' '.join(command)}")
        self._headless_process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
        )

    def _end_session(self) -> None:
        if self._close_event is not None:
            self._loop.call_soon_threadsafe(self._close_event.set)
        if self._session_task is not None:
            try:
                self._session_task.result(timeout=10)
            except Exception:
                pass
            self._session_task = None

    def close(self) -> None:
        """关闭 MCP 会话前释放持久化游戏输入。"""
        try:
            self.stop_game_session()
        except Exception:
            pass
        finally:
            self._end_session()
            self._stop_headless()
            self._loop.call_soon_threadsafe(self._loop.stop)

    def _stop_headless(self) -> None:
        process = self._headless_process
        self._headless_process = None
        if process is None or process.poll() is not None:
            return
        if self.headless is not None and not self.headless.shutdown_on_exit:
            self._headless_process = process
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _reconnect(self) -> None:
        """会话被服务器终止后重建（Streamable HTTP 会话可能因闲置被回收）。"""
        self._game_session_id = None
        # Frame cursors belong to the selected DeviceHub video session.  Do
        # not carry a cursor from the dead session into ``observe_game`` after
        # reconnecting: a newly-created stream may start at a lower value and
        # would otherwise wait forever for a frame version that can no longer
        # be produced.
        self._last_frame_version = None
        self._last_image_size = None
        self._invalidate_status_cache()
        self._end_session()
        self._start_session()

    def _ensure_status_cache(self) -> None:
        """为旧实例和轻量测试 double 补齐状态缓存字段。"""
        if not hasattr(self, "_status_cache_lock"):
            self._status_cache_lock = threading.RLock()
        if not hasattr(self, "_status_cache"):
            self._status_cache = None
        if not hasattr(self, "_status_cache_at"):
            self._status_cache_at = 0.0
        if not hasattr(self, "_app_status_cache"):
            self._app_status_cache = {}

    def _invalidate_status_cache(self) -> None:
        self._ensure_status_cache()
        with self._status_cache_lock:
            self._status_cache = None
            self._status_cache_at = 0.0
            self._app_status_cache.clear()

    # 这些关键字意味着请求未被处理（会话死亡/连接断开），重连后重试是安全的；
    # 工具级业务错误（isError 结果、设备侧超时）不在此列，不做自动重试。
    _TRANSIENT_KEYWORDS = ("session terminated", "session termination", "closedresource",
                           "connection", "disconnected", "broken", "404")

    def call(self, tool_name: str, **args: Any) -> ToolResult:
        args = {k: v for k, v in args.items() if v is not None}
        with self._lock:  # 串行化设备操作，保持手势顺序确定
            try:
                if self._session is None:
                    self._reconnect()
                result = self._run(self._session.call_tool(tool_name, args))
            except Exception as e:
                msg = f"{type(e).__name__}: {e}".lower()
                if not any(k in msg for k in self._TRANSIENT_KEYWORDS):
                    raise
                print(f"[device] MCP 会话失效（{e}），重连后重试 {tool_name}")
                self._reconnect()
                result = self._run(self._session.call_tool(tool_name, args))
        text: str | None = None
        parsed: Any | None = None
        image: bytes | None = None
        for block in result.content or []:
            btype = getattr(block, "type", None)
            if btype == "text" and text is None:
                text = block.text
                try:
                    parsed = json.loads(block.text)
                except (json.JSONDecodeError, TypeError):
                    parsed = None
            elif btype == "image" and image is None:
                image = base64.b64decode(block.data)
        if result.isError:
            raise DeviceError(f"{tool_name} 失败: {text or '未知错误'}")
        if isinstance(parsed, dict):
            self._record_frame_version(parsed)
        return ToolResult(json=parsed, image=image, text=text)

    def _record_frame_version(self, payload: dict[str, Any]) -> None:
        """Remember the newest cursor returned by any DeviceHub tool.

        Low-latency actions return ``frame_version_before`` and
        ``frame_version_after``.  Older code only consumed ``frame_version``
        and consequently seeded the next observation with the frame from
        before a swipe.  Prefer the ``*_after`` value, accept the camelCase
        spelling used by lightweight proxies, and never move the local cursor
        backwards when a delayed screenshot response arrives.
        """
        version = None
        for key in (
            "frame_version_after",
            "frameVersionAfter",
            "frame_version",
            "frameVersion",
        ):
            if key not in payload:
                continue
            try:
                candidate = int(payload[key])
            except (TypeError, ValueError, OverflowError):
                continue
            if candidate >= 0:
                version = candidate
                break
        if version is None:
            return
        current = getattr(self, "_last_frame_version", None)
        if not isinstance(current, int) or version >= current:
            self._last_frame_version = version

    # ---- 坐标映射（横屏逻辑空间 → 设备截图空间）----
    # 游戏横屏运行时截图流仍是竖屏帧（内容旋转 90°），tap 坐标空间跟随截图。
    # GameContext 依据帧朝向安装映射函数；未安装时坐标原样透传。

    def set_coord_mapper(self, fn) -> None:
        self._mapper = fn

    def _map(self, x: float, y: float, iw: int | None, ih: int | None):
        if self._mapper is None:
            return x, y, iw, ih
        try:
            return self._mapper(x, y, iw, ih)
        except TypeError:
            # 兼容旧的二维映射回调。
            return self._mapper(x, y)

    # ---- typed wrappers ----

    def status(self, *, force: bool = False) -> dict:
        self._ensure_status_cache()
        now = time.monotonic()
        with self._status_cache_lock:
            if (
                not force
                and self._status_cache is not None
                and now - self._status_cache_at < STATUS_CACHE_TTL_S
            ):
                return dict(self._status_cache)
        payload = self.call("status").json
        if not isinstance(payload, dict):
            raise DeviceError("status 未返回 JSON 对象")
        with self._status_cache_lock:
            self._status_cache = dict(payload)
            self._status_cache_at = time.monotonic()
        return dict(payload)

    def list_devices(self) -> dict:
        payload = self.call("list_devices").json
        if not isinstance(payload, dict):
            raise DeviceError("list_devices 未返回 JSON 对象")
        return payload

    def connect_device(self, selection_id: str | None = None) -> dict:
        """为当前 MCP 会话建立 active device session。"""
        if selection_id is None:
            status = self.status(force=True)
            selection_id = status.get("device_id") or status.get("active_udid")
        if not selection_id:
            devices = self.list_devices().get("devices", [])
            active = next((item for item in devices
                           if isinstance(item, dict) and item.get("active")), None)
            if isinstance(active, dict):
                selection_id = active.get("id") or active.get("udid")
            elif isinstance(devices, list):
                available = [
                    item for item in devices
                    if isinstance(item, dict) and (item.get("id") or item.get("udid"))
                ]
                # Fresh DeviceHub 140 sessions list a reachable Wi-Fi iPhone
                # before marking it active. Selecting the sole candidate is
                # deterministic; with multiple devices, keep refusing to guess.
                if len(available) == 1:
                    selection_id = available[0].get("id") or available[0].get("udid")
        if not selection_id:
            raise DeviceError("status 未返回可连接的设备 ID")
        # A device connection starts a fresh frame stream, even when the MCP
        # session itself remains alive.  Clear the old stream cursor before
        # the connect response is parsed so its first ``frame_version_after``
        # becomes the new baseline.
        self._last_frame_version = None
        self._last_image_size = None
        result = self.call("connect_device", udid=selection_id)
        self._invalidate_status_cache()
        if isinstance(result.json, dict):
            return result.json
        return {"message": result.text or "设备连接请求已发送", "selection_id": selection_id}

    def screenshot_png(self) -> bytes:
        r = self.call("screenshot", grid=False, max_dim=0)
        if not r.image:
            raise DeviceError("screenshot 未返回图像")
        if isinstance(r.json, dict):
            width, height = r.json.get("image_width"), r.json.get("image_height")
            if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
                self._last_image_size = (width, height)
        return r.image

    def observe_game_png(self, *, after_version: int | None = None,
                         timeout_ms: int = 250, max_dim: int = 0,
                         region: dict | None = None) -> bytes:
        """Capture an ungridded frame using DeviceHub 130's observation API.

        The method deliberately does not replace ``screenshot_png``: callers that
        depend on the exact legacy screenshot path can continue using it, while
        frame-driven tasks can wait for a newer decoded frame without polling.
        """
        try:
            result = self.call(
                "observe_game",
                after_version=after_version,
                timeout_ms=timeout_ms,
                max_dim=max_dim,
                region=region,
            )
        except Exception as error:
            if "unknown tool" not in str(error).lower() and "not found" not in str(error).lower():
                raise
            return self.screenshot_png()
        if not result.image:
            raise DeviceError("observe_game 未返回图像")
        if isinstance(result.json, dict):
            width = result.json.get("image_width")
            height = result.json.get("image_height")
            crop = result.json.get("crop")
            # A region observation has a different coordinate origin. Only
            # cache dimensions for full-screen frames.
            full = not isinstance(crop, dict) or (
                crop.get("x") == 0 and crop.get("y") == 0 and
                crop.get("width") == result.json.get("screen_width") and
                crop.get("height") == result.json.get("screen_height")
            )
            if full and isinstance(width, int) and isinstance(height, int):
                self._last_image_size = (width, height)
        return result.image

    def wait_for_frame(self, after_version: int | None = None,
                       timeout_ms: int = 2000) -> dict:
        """Wait for a decoded frame newer than ``after_version`` when supported."""
        result = self.call(
            "wait_for_frame",
            after_version=after_version,
            timeout_ms=timeout_ms,
        )
        payload = result.json
        if not isinstance(payload, dict):
            raise DeviceError("wait_for_frame 未返回 JSON 对象")
        return payload

    @property
    def last_image_size(self) -> tuple[int, int] | None:
        """最近一次原生截图尺寸（宽、高），不受状态接口缩略尺寸影响。"""
        return self._last_image_size

    @property
    def last_frame_version(self) -> int | None:
        return self._last_frame_version

    @property
    def game_session_id(self) -> str | None:
        """Locally cached persistent game-session ID, if it is still usable."""
        return self._game_session_id

    def _release_game_session_for_direct_input(self) -> None:
        """Cleanly end exclusive profile input before a legacy HID action.

        DeviceHub 140 invalidates a persistent game-control session whenever a
        direct tap/swipe/multi-touch action is sent.  Stopping it explicitly
        keeps the local cache in sync and avoids a later, misleading
        ``game control session not found`` heartbeat failure.
        """
        if self._game_session_id is None:
            return
        try:
            self.stop_game_session()
        except Exception:
            # The server may already have invalidated it; stop_game_session's
            # finally block still clears the local ID.
            pass

    def tap(self, x: float, y: float, *, hold_ms: int | None = None,
            image_width: int | None = None, image_height: int | None = None) -> None:
        self._release_game_session_for_direct_input()
        x, y, image_width, image_height = self._map(x, y, image_width, image_height)
        self.call("tap", x=x, y=y, hold_ms=hold_ms, wait_for_settle=False,
                  image_width=image_width, image_height=image_height)

    def swipe(self, x1: float, y1: float, x2: float, y2: float, *, duration_ms: int = 300,
              image_width: int | None = None, image_height: int | None = None) -> None:
        self._release_game_session_for_direct_input()
        source_width, source_height = image_width, image_height
        x1, y1, mapped_width, mapped_height = self._map(
            x1, y1, source_width, source_height
        )
        x2, y2, _, _ = self._map(x2, y2, source_width, source_height)
        self.call("swipe", x1=x1, y1=y1, x2=x2, y2=y2, duration_ms=duration_ms,
                  wait_for_settle=False, image_width=mapped_width, image_height=mapped_height)

    def multi_touch(self, contacts: list[dict], *, duration_ms: int = 250,
                    image_width: int | None = None, image_height: int | None = None) -> None:
        self._release_game_session_for_direct_input()
        mapped = []
        source_width, source_height = image_width, image_height
        mapped_width, mapped_height = source_width, source_height
        for c in contacts:
            x1, y1, mapped_width, mapped_height = self._map(
                c["x1"], c["y1"], source_width, source_height
            )
            x2, y2, _, _ = self._map(c["x2"], c["y2"], source_width, source_height)
            mapped.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        self.call("multi_touch", contacts=mapped, duration_ms=duration_ms,
                  image_width=mapped_width if contacts else source_width,
                  image_height=mapped_height if contacts else source_height)

    def press_button(self, button: str) -> None:
        self._release_game_session_for_direct_input()
        self.call("press_button", button=button)

    def press_key(self, key: str) -> None:
        self._release_game_session_for_direct_input()
        self.call("press_key", key=key)

    def app_switcher(self) -> None:
        self.call("app_switcher")

    def lock_device(self) -> None:
        self.call("lock_device")

    def launch_app(self, bundle_id: str, wait: bool = True) -> None:
        self.call("launch_app", bundle_id=bundle_id, wait_for_settle=wait)
        self._invalidate_status_cache()

    def stop_app(self, bundle_id: str) -> None:
        self.call("stop_app", bundle_id=bundle_id)
        self._invalidate_status_cache()

    def background_current_app(self) -> str:
        """将当前 App 移出前台；WDA 不可用时使用 Home 按钮。"""
        try:
            self.call("wda_background_app")
            return "backgrounded-wda"
        except Exception:
            self.press_button("home")
            return "backgrounded-home"

    def app_status(self, bundle_id: str, *, force: bool = False) -> dict:
        self._ensure_status_cache()
        now = time.monotonic()
        with self._status_cache_lock:
            cached = self._app_status_cache.get(bundle_id)
            if not force and cached is not None and now - cached[1] < APP_STATUS_CACHE_TTL_S:
                return dict(cached[0])
        result = self.call("app_status", bundle_id=bundle_id)
        payload = result.json
        if not isinstance(payload, dict):
            raise DeviceError(f"app_status 未返回 JSON: {result.text or '未知错误'}")
        with self._status_cache_lock:
            self._app_status_cache[bundle_id] = (dict(payload), time.monotonic())
        return dict(payload)

    def wait_for_app(self, bundle_id: str, state: str, *, timeout_ms: int = 5000) -> dict:
        payload = self.call(
            "wait_for_app", bundle_id=bundle_id, state=state, timeout_ms=timeout_ms
        ).json
        if not isinstance(payload, dict):
            raise DeviceError("wait_for_app 未返回 JSON 对象")
        return payload

    def wait_for_device_event(self, *, after_sequence: int | None = None,
                              timeout_ms: int = 10000) -> dict:
        payload = self.call(
            "wait_for_device_event",
            after_sequence=after_sequence,
            timeout_ms=timeout_ms,
        ).json
        if not isinstance(payload, dict):
            raise DeviceError("wait_for_device_event 未返回 JSON 对象")
        return payload

    def get_keymap_profile(self, name: str) -> dict:
        result = self.call("get_keymap_profile", name=name)
        if not isinstance(result.json, dict):
            raise DeviceError(f"get_keymap_profile 未返回 JSON: {result.text or '未知错误'}")
        return result.json

    def list_keymap_profiles(self) -> dict:
        payload = self.call("list_keymap_profiles").json
        if not isinstance(payload, dict):
            raise DeviceError("list_keymap_profiles 未返回 JSON 对象")
        return payload

    def save_keymap_profile(self, profile: dict, *, overwrite: bool = False) -> dict:
        """Persist a native v2 profile in the current DeviceHub repository."""
        if not isinstance(profile, dict):
            raise TypeError("profile 必须是 JSON 对象")
        name = profile.get("name")
        mappings = profile.get("mappings")
        if not isinstance(name, str) or not name:
            raise ValueError("profile 缺少 name")
        if not isinstance(mappings, list):
            raise ValueError("profile 缺少 mappings 数组")
        target = profile.get("targetResolution")
        if target is not None and not isinstance(target, dict):
            raise ValueError("profile.targetResolution 必须是对象")
        result = self.call(
            "save_keymap_profile",
            name=name,
            mappings=mappings,
            bundleIdentifiers=profile.get("bundleIdentifiers", []),
            targetResolution=target,
            hardwareBindings=profile.get("hardwareBindings"),
            overwrite=overwrite,
        )
        payload = result.json
        if not isinstance(payload, dict):
            return {"message": result.text or "profile 已保存"}
        return payload

    def install_keymap_profile(self, path: str | os.PathLike[str], *, overwrite: bool = True) -> dict:
        with open(path, "r", encoding="utf-8") as stream:
            profile = json.load(stream)
        return self.save_keymap_profile(profile, overwrite=overwrite)

    def run_keymap(self, profile_name: str, keys: list[str], *, hold_ms: int = 100,
                   allow_scripts: bool = False, wait_for_settle: bool = False) -> None:
        self._release_game_session_for_direct_input()
        self.call("run_keymap", profile_name=profile_name, keys=keys, hold_ms=hold_ms,
                  allow_scripts=allow_scripts, wait_for_settle=wait_for_settle)

    def start_game_session(self, profile_name: str, *, bundle_id: str | None = None,
                           lease_ms: int = 3000, require_resolution_match: bool = False,
                           allow_scripts: bool = False) -> str:
        if self._game_session_id is not None:
            return self._game_session_id
        result = self.call(
            "start_game_session",
            profile_name=profile_name,
            bundle_id=bundle_id,
            lease_ms=lease_ms,
            require_resolution_match=require_resolution_match,
            allow_scripts=allow_scripts,
        )
        payload = result.json if isinstance(result.json, dict) else {}
        session_id = payload.get("session_id") or payload.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise DeviceError(f"start_game_session 未返回 session_id: {result.text or '未知错误'}")
        self._game_session_id = session_id
        return session_id

    def set_game_input(self, session_id: str, keys: list[str], *, lease_ms: int = 3000) -> None:
        self.call("set_game_input", session_id=session_id, keys=keys, lease_ms=lease_ms)

    def stop_game_session(self, session_id: str | None = None) -> None:
        session_id = session_id or self._game_session_id
        if session_id is None:
            return
        try:
            self.call("stop_game_session", session_id=session_id)
        finally:
            if self._game_session_id == session_id:
                self._game_session_id = None

    def type_text(self, text: str) -> None:
        self.call("type_text", text=text)

    def reconnect_device(self) -> None:
        """重建设备通道并等待新帧。

        实测：前台应用切换（游戏冷启动/切回）后 HID 注入可能整体失效，
        重连设备通道可恢复——门界面点按只在 reconnect 后生效过。
        """
        st = self.status(force=True)
        udid = st.get("device_id") or st.get("active_udid")
        if udid:
            self.call("reconnect_device", udid=udid)
            self._invalidate_status_cache()

    # ---- WDA semantic helpers (optional; HID remains the primary path) ----

    def wda_device_state(self) -> dict:
        payload = self.call("wda_device_state").json
        if not isinstance(payload, dict):
            raise DeviceError("wda_device_state 未返回 JSON 对象")
        return payload

    def wda_unlock(self) -> dict:
        payload = self.call("wda_unlock").json
        if not isinstance(payload, dict):
            raise DeviceError("wda_unlock 未返回 JSON 对象")
        return payload

    def wda_ui_tree(self, *, max_characters: int | None = None) -> dict:
        payload = self.call("wda_ui_tree", max_characters=max_characters).json
        if not isinstance(payload, dict):
            raise DeviceError("wda_ui_tree 未返回 JSON 对象")
        return payload

    def wda_find_elements(self, using: str, value: str, *, limit: int = 10) -> dict:
        payload = self.call("wda_find_elements", using=using, value=value, limit=limit).json
        if not isinstance(payload, dict):
            raise DeviceError("wda_find_elements 未返回 JSON 对象")
        return payload

    def wda_wait_for_element(self, using: str, value: str, *, index: int = 0,
                             state: str = "present", timeout_ms: int = 5000) -> dict:
        payload = self.call(
            "wda_wait_for_element", using=using, value=value, index=index,
            state=state, timeout_ms=timeout_ms
        ).json
        if not isinstance(payload, dict):
            raise DeviceError("wda_wait_for_element 未返回 JSON 对象")
        return payload

    def wda_click(self, using: str, value: str, *, index: int = 0) -> dict:
        payload = self.call("wda_click", using=using, value=value, index=index).json
        if not isinstance(payload, dict):
            raise DeviceError("wda_click 未返回 JSON 对象")
        return payload

    def wda_type_text(self, text: str) -> None:
        self.call("wda_type_text", text=text)

    def paste_text(self, text: str) -> None:
        self.call("paste_text", text=text)


class DeviceError(RuntimeError):
    pass
