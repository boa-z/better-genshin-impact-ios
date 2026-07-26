"""devicehub-mask MCP 客户端（同步外观）。

BetterGI 脚本与任务代码是同步调用风格（原版是 ClearScript 绑定同步 C# 方法），
因此在后台线程跑一个 asyncio 事件循环持有 MCP 会话，对外提供阻塞式方法。
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

DEFAULT_URL = "http://127.0.0.1:8009/mcp"
CALL_TIMEOUT_S = 120


@dataclass
class ToolResult:
    json: Any | None
    image: bytes | None
    text: str | None


class DeviceClient:
    """所有坐标均为截图像素坐标；动作调用附带 image_width/height 保证确定性缩放。"""

    def __init__(self, url: str = DEFAULT_URL):
        self.url = url
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-loop")
        self._thread.start()
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._lock = threading.Lock()
        self._run(self._connect())

    # ---- plumbing ----

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=CALL_TIMEOUT_S)

    async def _connect(self) -> None:
        self._stack = AsyncExitStack()
        read, write, _ = await self._stack.enter_async_context(streamablehttp_client(self.url))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    def close(self) -> None:
        if self._stack is not None:
            try:
                self._run(self._stack.aclose())
            except Exception:
                pass
            self._stack = None
        self._loop.call_soon_threadsafe(self._loop.stop)

    def call(self, name: str, **args: Any) -> ToolResult:
        args = {k: v for k, v in args.items() if v is not None}
        with self._lock:  # 串行化设备操作，保持手势顺序确定
            result = self._run(self._session.call_tool(name, args))
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
            raise DeviceError(f"{name} 失败: {text or '未知错误'}")
        return ToolResult(json=parsed, image=image, text=text)

    # ---- 坐标映射（横屏逻辑空间 → 设备截图空间）----
    # 游戏横屏运行时截图流仍是竖屏帧（内容旋转 90°），tap 坐标空间跟随截图。
    # GameContext 依据帧朝向安装映射函数；未安装时坐标原样透传。

    _mapper = None  # (x, y) -> (x, y, image_width, image_height)

    def set_coord_mapper(self, fn) -> None:
        self._mapper = fn

    def _map(self, x: float, y: float, iw: int | None, ih: int | None):
        if self._mapper is None:
            return x, y, iw, ih
        return self._mapper(x, y)

    # ---- typed wrappers ----

    def status(self) -> dict:
        return self.call("status").json

    def screenshot_png(self) -> bytes:
        r = self.call("screenshot", grid=False, max_dim=0)
        if not r.image:
            raise DeviceError("screenshot 未返回图像")
        return r.image

    def tap(self, x: float, y: float, *, hold_ms: int | None = None,
            image_width: int | None = None, image_height: int | None = None) -> None:
        x, y, image_width, image_height = self._map(x, y, image_width, image_height)
        self.call("tap", x=x, y=y, hold_ms=hold_ms, wait_for_settle=False,
                  image_width=image_width, image_height=image_height)

    def swipe(self, x1: float, y1: float, x2: float, y2: float, *, duration_ms: int = 300,
              image_width: int | None = None, image_height: int | None = None) -> None:
        x1, y1, image_width, image_height = self._map(x1, y1, image_width, image_height)
        x2, y2, _, _ = self._map(x2, y2, image_width, image_height)
        self.call("swipe", x1=x1, y1=y1, x2=x2, y2=y2, duration_ms=duration_ms,
                  wait_for_settle=False, image_width=image_width, image_height=image_height)

    def multi_touch(self, contacts: list[dict], *, duration_ms: int = 250,
                    image_width: int | None = None, image_height: int | None = None) -> None:
        mapped = []
        for c in contacts:
            x1, y1, image_width, image_height = self._map(c["x1"], c["y1"], image_width, image_height)
            x2, y2, _, _ = self._map(c["x2"], c["y2"], image_width, image_height)
            mapped.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        self.call("multi_touch", contacts=mapped, duration_ms=duration_ms,
                  image_width=image_width, image_height=image_height)

    def press_button(self, button: str) -> None:
        self.call("press_button", button=button)

    def launch_app(self, bundle_id: str, wait: bool = True) -> None:
        self.call("launch_app", bundle_id=bundle_id, wait_for_settle=wait)

    def stop_app(self, bundle_id: str) -> None:
        self.call("stop_app", bundle_id=bundle_id)

    def app_status(self, bundle_id: str) -> dict:
        return self.call("app_status", bundle_id=bundle_id).json

    def type_text(self, text: str) -> None:
        self.call("type_text", text=text)

    def paste_text(self, text: str) -> None:
        self.call("paste_text", text=text)


class DeviceError(RuntimeError):
    pass
