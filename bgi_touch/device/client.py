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
from mcp.client.streamable_http import streamablehttp_client

from .config import DEFAULT_MCP_URL, HeadlessConfig

DEFAULT_URL = DEFAULT_MCP_URL
CALL_TIMEOUT_S = 120


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
        self._mapper = None
        self._last_image_size: tuple[int, int] | None = None
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
        self._end_session()
        self._start_session()

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
        return ToolResult(json=parsed, image=image, text=text)

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

    def status(self) -> dict:
        return self.call("status").json

    def connect_device(self, selection_id: str | None = None) -> dict:
        """为当前 MCP 会话建立 active device session。"""
        if selection_id is None:
            status = self.status()
            selection_id = status.get("device_id") or status.get("active_udid")
        if not selection_id:
            raise DeviceError("status 未返回可连接的设备 ID")
        result = self.call("connect_device", udid=selection_id)
        return result.json if isinstance(result.json, dict) else {}

    def screenshot_png(self) -> bytes:
        r = self.call("screenshot", grid=False, max_dim=0)
        if not r.image:
            raise DeviceError("screenshot 未返回图像")
        if isinstance(r.json, dict):
            width, height = r.json.get("image_width"), r.json.get("image_height")
            if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
                self._last_image_size = (width, height)
        return r.image

    @property
    def last_image_size(self) -> tuple[int, int] | None:
        """最近一次原生截图尺寸（宽、高），不受状态接口缩略尺寸影响。"""
        return self._last_image_size

    def tap(self, x: float, y: float, *, hold_ms: int | None = None,
            image_width: int | None = None, image_height: int | None = None) -> None:
        x, y, image_width, image_height = self._map(x, y, image_width, image_height)
        self.call("tap", x=x, y=y, hold_ms=hold_ms, wait_for_settle=False,
                  image_width=image_width, image_height=image_height)

    def swipe(self, x1: float, y1: float, x2: float, y2: float, *, duration_ms: int = 300,
              image_width: int | None = None, image_height: int | None = None) -> None:
        source_width, source_height = image_width, image_height
        x1, y1, mapped_width, mapped_height = self._map(
            x1, y1, source_width, source_height
        )
        x2, y2, _, _ = self._map(x2, y2, source_width, source_height)
        self.call("swipe", x1=x1, y1=y1, x2=x2, y2=y2, duration_ms=duration_ms,
                  wait_for_settle=False, image_width=mapped_width, image_height=mapped_height)

    def multi_touch(self, contacts: list[dict], *, duration_ms: int = 250,
                    image_width: int | None = None, image_height: int | None = None) -> None:
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
        self.call("press_button", button=button)

    def launch_app(self, bundle_id: str, wait: bool = True) -> None:
        self.call("launch_app", bundle_id=bundle_id, wait_for_settle=wait)

    def stop_app(self, bundle_id: str) -> None:
        self.call("stop_app", bundle_id=bundle_id)

    def background_current_app(self) -> str:
        """将当前 App 移出前台；WDA 不可用时使用 Home 按钮。"""
        try:
            self.call("wda_background_app")
            return "backgrounded-wda"
        except Exception:
            self.press_button("home")
            return "backgrounded-home"

    def app_status(self, bundle_id: str) -> dict:
        return self.call("app_status", bundle_id=bundle_id).json

    def get_keymap_profile(self, name: str) -> dict:
        result = self.call("get_keymap_profile", name=name)
        if not isinstance(result.json, dict):
            raise DeviceError(f"get_keymap_profile 未返回 JSON: {result.text or '未知错误'}")
        return result.json

    def list_keymap_profiles(self) -> dict:
        return self.call("list_keymap_profiles").json

    def run_keymap(self, profile_name: str, keys: list[str], *, hold_ms: int = 100,
                   allow_scripts: bool = False, wait_for_settle: bool = False) -> None:
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
        st = self.status()
        udid = st.get("device_id") or st.get("active_udid")
        if udid:
            self.call("reconnect_device", udid=udid)

    def paste_text(self, text: str) -> None:
        self.call("paste_text", text=text)


class DeviceError(RuntimeError):
    pass
