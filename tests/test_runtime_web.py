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
        "config": {"forceInteraction": True},
    })
    assert ctx.calls == [(('AutoPick',), {"force_interaction": True})]


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
