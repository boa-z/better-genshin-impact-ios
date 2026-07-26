"""实时触发器框架（原版 TaskTriggerDispatcher 的移植）。

原版以 ~50ms 定时器驱动；移动端截图往返 ~200-400ms，取 ~1-2fps 帧循环。
触发器实现 Trigger 协议：on_frame(region) 检测并自行执行点按。
帧循环独立线程运行，设备层内部锁保证与主任务的触控互不交错。
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Protocol

from ..engine.context import GameContext
from ..engine.recognition import ImageRegion


class Trigger(Protocol):
    name: str
    enabled: bool

    def on_frame(self, region: ImageRegion) -> None: ...


class TriggerLoop:
    def __init__(self, ctx: GameContext, interval_s: float = 0.7,
                 log: Callable[[str], None] = print):
        self.ctx = ctx
        self.interval = interval_s
        self.log = log
        self.triggers: list[Trigger] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add(self, trigger: Trigger) -> None:
        self.triggers = [t for t in self.triggers if t.name != trigger.name] + [trigger]
        self.log(f"[trigger] 启用 {trigger.name}")

    def clear(self) -> None:
        self.triggers = []

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="trigger-loop")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        self.log(f"[trigger] 帧循环启动（{1/self.interval:.1f} fps）")
        while not self._stop.is_set():
            t0 = time.monotonic()
            if self.triggers:
                try:
                    region = self.ctx.capture_region()
                    for tr in list(self.triggers):
                        if getattr(tr, "enabled", True):
                            tr.on_frame(region)
                except Exception as e:
                    # 帧循环不因单帧失败而退出（截到竖屏/设备重连等）
                    self.log(f"[trigger] 帧处理失败: {e}")
                    time.sleep(1.0)
            wait = self.interval - (time.monotonic() - t0)
            if wait > 0 and self._stop.wait(wait):
                break
        self.log("[trigger] 帧循环停止")
