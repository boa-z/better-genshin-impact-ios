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
        self._restart_requested = False

    def add(self, trigger: Trigger) -> None:
        with self._state_lock:
            removed = [t for t in self.triggers if t.name == trigger.name and t is not trigger]
            self.triggers = [t for t in self.triggers if t.name != trigger.name] + [trigger]
            self._generation += 1
        self._close(removed)
        self.log(f"[trigger] 启用 {trigger.name}")

    def clear(self) -> None:
        with self._state_lock:
            removed = self.triggers
            self.triggers = []
            self._generation += 1
        self._close(removed)

    def remove(self, name: str) -> None:
        with self._state_lock:
            removed = [t for t in self.triggers if t.name == name]
            self.triggers = [t for t in self.triggers if t.name != name]
            self._generation += 1
        self._close(removed)

    def replace(self, triggers: list[Trigger]) -> None:
        with self._state_lock:
            retained = {id(trigger) for trigger in triggers}
            removed = [trigger for trigger in self.triggers if id(trigger) not in retained]
            self.triggers = list(triggers)
            self._generation += 1
        self._close(removed)

    @staticmethod
    def _close(triggers: list[Trigger]) -> None:
        for trigger in triggers:
            close = getattr(trigger, "close", None)
            if callable(close):
                close()

    def start(self) -> None:
        with self._state_lock:
            if self._thread and self._thread.is_alive():
                if self._stop.is_set() and self.triggers:
                    # A screenshot may still be finishing after stop(). Keep
                    # the new trigger list and restart as soon as that producer
                    # exits instead of silently leaving it inactive.
                    self._restart_requested = True
                return
            self._stop.clear()
            self._restart_requested = False
            self._thread = threading.Thread(target=self._run, daemon=True, name="trigger-loop")
            self._thread.start()

    def stop(self) -> None:
        with self._state_lock:
            self._generation += 1
            self._restart_requested = False
        self._stop.set()

    @property
    def active(self) -> bool:
        with self._state_lock:
            return bool(self.triggers and self._thread and self._thread.is_alive()
                        and not self._stop.is_set())

    def get(self, name: str) -> Trigger | None:
        with self._state_lock:
            return next((trigger for trigger in self.triggers if trigger.name == name), None)

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
            self._restart_requested = False
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
        """恢复 pause 前的触发器，除非外部已安装了新列表。

        ``pause`` 也可能用于一个尚未启动帧线程、但已经配置好触发器的
        ``TriggerLoop``。这种情况常见于任务刚切换页面时：输入生产者还没
        start，任务仍然需要暂时独占输入。恢复时应保留原列表，但不要凭空
        启动此前未运行的线程。
        """
        previous, was_active, generation = state
        if not previous:
            return
        with self._state_lock:
            if self.triggers or self._generation != generation:
                return
            self.triggers = list(previous)
            self._generation += 1
        if was_active:
            self.start()

    def _run(self) -> None:
        self.log(f"[trigger] 帧循环启动（{1/self.interval:.1f} fps）")
        try:
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
                        exclusive = next(
                            (
                                tr for tr in triggers
                                if getattr(tr, "is_exclusive", False)
                                and getattr(tr, "enabled", True)
                            ),
                            None,
                        )
                        active_triggers = [exclusive] if exclusive is not None else triggers
                        for tr in active_triggers:
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
        finally:
            self.log("[trigger] 帧循环停止")
            current = threading.current_thread()
            with self._state_lock:
                if self._thread is current:
                    self._thread = None
                restart = self._restart_requested and bool(self.triggers)
                self._restart_requested = False
            if restart:
                self.start()
