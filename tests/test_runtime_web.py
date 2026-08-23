import threading
import time

import numpy as np
import pytest


def test_game_context_cached_frame_returns_copy_without_device_access():
    from bgi_touch.engine.context import GameContext

    ctx = object.__new__(GameContext)
    ctx._frame_lock = threading.Lock()
    ctx._last_frame = np.zeros((2, 3, 3), dtype=np.uint8)
    ctx._last_frame_at = time.monotonic()

    frame, age = ctx.cached_frame()
    assert frame is not ctx._last_frame
    assert frame.shape == (2, 3, 3)
    assert age >= 0
    frame[0, 0] = 255
    assert not ctx._last_frame[0, 0].any()


def test_trigger_loop_pause_waits_for_frame_and_resume_restores_trigger():
    from bgi_touch.triggers.loop import TriggerLoop

    frame_seen = threading.Event()
    calls = []

    class Context:
        def capture_region(self):
            return object()

    class Trigger:
        name = "AutoPick"
        enabled = True

        def on_frame(self, region):
            calls.append(region)
            frame_seen.set()

    loop = TriggerLoop(Context(), interval_s=0.01, log=lambda _: None)
    trigger = Trigger()
    loop.add(trigger)
    loop.start()
    assert frame_seen.wait(1.0)

    state = loop.pause()
    assert state[0] == [trigger]
    assert state[1] is True
    count = len(calls)
    time.sleep(0.05)
    assert len(calls) == count

    loop.resume(state)
    assert loop.active
    loop.pause()
    assert not loop.active


def test_dispatcher_passes_bettergi_force_interaction_config():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    class Triggers:
        def clear(self):
            pass

    class Context:
        triggers = Triggers()

        def __init__(self):
            self.calls = []

        def enable_trigger(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    ctx = Context()
    TaskDispatcher(ctx).add_timer({
        "name": "AutoPick",
        "config": {
            "forceInteraction": True,
            "mode": "Blacklist",
            "blackList": ["调查"],
            "fuzzyBlacklist": ["进入"],
            "whiteList": ["甜甜花"],
            "doNotPickList": ["薄荷"],
        },
    })
    assert ctx.calls == [(('AutoPick',), {
        "force_interaction": True,
        "mode": "Blacklist",
        "whitelist": ["甜甜花"],
        "blacklist": ["调查"],
        "fuzzy_blacklist": ["进入"],
        "whitelist_exclusions": ["薄荷"],
    })]


def test_autopick_defaults_to_recommended_whitelist_and_supports_blacklist():
    from bgi_touch.triggers.autopick import AutoPickTrigger

    trigger = AutoPickTrigger.__new__(AutoPickTrigger)
    trigger.mode = "Whitelist"
    trigger.whitelist = frozenset({"甜甜花", "薄荷"})
    trigger.blacklist = {"调查"}
    trigger.fuzzy_blacklist = ("进入",)

    assert trigger._should_pick("甜甜花")
    assert not trigger._should_pick("优兰尼娅湖")
    assert not trigger._should_pick("聚所")

    trigger.mode = "Blacklist"
    assert trigger._should_pick("甜甜花")
    assert not trigger._should_pick("调查")
    assert not trigger._should_pick("进入秘境")


def test_teleport_panel_wait_does_not_fail_on_first_empty_frame():
    from bgi_touch.pathing.tp import TpTask

    class Hit:
        text = ""

        def __init__(self, exists):
            self.exists = exists
            self.clicks = 0

        def is_exist(self):
            return self.exists

        def click(self):
            self.clicks += 1

    button = Hit(True)

    class Region:
        def __init__(self, ready):
            self.ready = ready

        def find(self, _):
            return button if self.ready else Hit(False)

        def find_multi(self, *_args, **_kwargs):
            return []

    class Context:
        def __init__(self):
            self.captures = 0

        def capture_region(self):
            self.captures += 1
            return Region(self.captures >= 2)

        def sleep(self, _):
            pass

    task = TpTask.__new__(TpTask)
    task.ctx = Context()
    task.log = lambda _: None
    task._go_teleport = object()

    assert task._find_and_tap_confirm(timeout_s=0.2, initial_delay_ms=0)
    assert task.ctx.captures == 2
    assert button.clicks == 1


def test_teleport_ambiguous_icons_use_only_one_precomputed_fallback():
    from types import SimpleNamespace

    from bgi_touch.pathing.tp import TpTask

    taps = []
    fallback = SimpleNamespace(clicks=0)
    fallback.click = lambda: setattr(fallback, "clicks", fallback.clicks + 1)
    ctx = SimpleNamespace(
        transform=SimpleNamespace(device_width=1000, device_height=500),
        device=SimpleNamespace(tap=lambda *args, **kwargs: taps.append((args, kwargs))),
    )
    task = TpTask.__new__(TpTask)
    task.ctx = ctx
    task.log = lambda _: None
    task._anchor_icons_near = lambda *_: [
        (20, fallback),
        (35, SimpleNamespace()),
    ]
    confirmations = iter((False, True))
    task._find_and_tap_confirm = lambda: next(confirmations)

    assert task._select_target_and_confirm(400, 250, 50)
    assert len(taps) == 1
    assert fallback.clicks == 1


def test_js_runtime_awaits_async_iife_and_restores_python_error_text(tmp_path):
    import pythonmonkey as pm

    from bgi_touch.engine.js_runtime import JsScriptRuntime

    def fail():
        raise RuntimeError("传送失败：未能完成锚点确认（迭代/超时耗尽）")

    pm.eval("globalThis")["bgi_test_fail"] = fail
    (tmp_path / "main.js").write_text(
        "// wrapper form used by BetterGI scripts\n"
        "(async function () { await bgi_test_fail(); })();",
        encoding="utf-8",
    )
    runtime = JsScriptRuntime.__new__(JsScriptRuntime)
    runtime.pm = pm
    runtime.script_dir = tmp_path
    runtime.manifest = {}

    with pytest.raises(RuntimeError, match="传送失败：未能完成锚点确认"):
        runtime.run()
