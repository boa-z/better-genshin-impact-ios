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


TriggerState = tuple[list[Trigger], bool, int]


class TriggerLoop:
    def __init__(self, ctx: GameContext, interval_s: float = 0.7,
                 log: Callable[[str], None] = print):
        self.ctx = ctx
        self.interval = interval_s
        self.log = log
        self.triggers: list[Trigger] = []
        self._state_lock = threading.RLock()
        self._generation = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add(self, trigger: Trigger) -> None:
        with self._state_lock:
            self.triggers = [t for t in self.triggers if t.name != trigger.name] + [trigger]
            self._generation += 1
        self.log(f"[trigger] 启用 {trigger.name}")

    def clear(self) -> None:
        with self._state_lock:
            self.triggers = []
            self._generation += 1

    def remove(self, name: str) -> None:
        with self._state_lock:
            self.triggers = [t for t in self.triggers if t.name != name]
            self._generation += 1

    def replace(self, triggers: list[Trigger]) -> None:
        with self._state_lock:
            self.triggers = list(triggers)
            self._generation += 1

    def start(self) -> None:
        with self._state_lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="trigger-loop")
            self._thread.start()

    def stop(self) -> None:
        with self._state_lock:
            self._generation += 1
        self._stop.set()

    @property
    def active(self) -> bool:
        with self._state_lock:
            return bool(self.triggers and self._thread and self._thread.is_alive()
                        and not self._stop.is_set())

    def pause(self) -> TriggerState:
        """停止帧循环并等待当前截图/触发器调用退出。

        地图手势与实时触发器共用设备输入通道。仅清空列表仍可能让已经
        截完图的帧继续执行，因此这里同时设置停止事件并等待线程结束。
        """
        with self._state_lock:
            previous = list(self.triggers)
            was_active = bool(previous and self._thread and self._thread.is_alive()
                              and not self._stop.is_set())
            self.triggers = []
            self._stop.set()
            self._generation += 1
            generation = self._generation
            thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            # Device calls are serialized and bounded by DeviceClient's timeout.
            # A short bounded wait keeps a broken trigger from blocking teleport
            # forever; _run also checks the stop event before every action.
            thread.join(timeout=10.0)
        with self._state_lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None
        return previous, was_active, generation

    def resume(self, state: TriggerState) -> None:
        """恢复 pause 前仍在运行的触发器，除非外部已安装了新列表。"""
        previous, was_active, generation = state
        if not previous or not was_active:
            return
        with self._state_lock:
            if self.triggers or self._generation != generation:
                return
            self.triggers = list(previous)
            self._generation += 1
        self.start()

    def _run(self) -> None:
        self.log(f"[trigger] 帧循环启动（{1/self.interval:.1f} fps）")
        while not self._stop.is_set():
            t0 = time.monotonic()
            with self._state_lock:
                triggers = list(self.triggers)
            if triggers:
                try:
                    region = self.ctx.capture_region()
                    # pause() may have been called while capture_region was in
                    # flight. Do not let that already captured frame send input.
                    if self._stop.is_set():
                        break
                    for tr in triggers:
                        if self._stop.is_set():
                            break
                        if getattr(tr, "enabled", True):
                            tr.on_frame(region)
                except Exception as e:
                    # 帧循环不因单帧失败而退出（截到竖屏/设备重连等）
                    self.log(f"[trigger] 帧处理失败: {e}")
                    self._stop.wait(1.0)
            wait = self.interval - (time.monotonic() - t0)
            if wait > 0 and self._stop.wait(wait):
                break
        self.log("[trigger] 帧循环停止")
