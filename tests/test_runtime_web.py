import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest


def test_webui_shutdown_closes_shared_context_once(monkeypatch):
    from unittest.mock import Mock

    from bgi_touch.webui import server

    context = Mock()
    runner = Mock()
    monkeypatch.setattr(server, "_ctx", context)
    monkeypatch.setattr(server, "runner", runner)

    server._shutdown_context()
    server._shutdown_context()

    assert runner.stop.call_count == 2
    context.close.assert_called_once_with()
    assert server._ctx is None


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


def test_webui_screenshot_reuses_cached_jpeg_and_supports_conditional_requests(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.webui import server

    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    ctx = SimpleNamespace(
        cached_frame=Mock(return_value=(frame.copy(), 0.05)),
        frame_generation=7,
    )
    monkeypatch.setattr(server, "_ctx", ctx)
    server._preview_jpeg_cache.clear()
    try:
        first = server.api_screenshot(w=6, q=60, if_none_match=None)
        assert first.status_code == 200
        assert first.headers["etag"]
        assert first.headers["cache-control"] == "private, max-age=0, must-revalidate"

        def unexpected_encode(*_args, **_kwargs):
            raise AssertionError("同一缓存帧不应重复 JPEG 编码")

        monkeypatch.setattr(server.cv2, "imencode", unexpected_encode)
        repeated = server.api_screenshot(w=6, q=60, if_none_match=None)
        assert repeated.status_code == 200
        assert repeated.body == first.body

        unchanged = server.api_screenshot(
            w=6, q=60, if_none_match=first.headers["etag"]
        )
        assert unchanged.status_code == 304
        assert unchanged.body in (b"", None)
    finally:
        server._preview_jpeg_cache.clear()


def test_webui_preview_polling_is_single_flight_and_pauses_in_background():
    from pathlib import Path

    page = (Path(__file__).parents[1] / "bgi_touch" / "webui" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "let shotInFlight = false" in page
    assert "if (shotInFlight) { shotPending = true; return; }" in page
    assert "document.hidden" in page
    assert "cache:'no-cache'" in page
    assert "w=1024&q=60" in page


def test_main_ui_uses_paimon_hud_marker_instead_of_minimap_circle():
    from bgi_touch.engine.recognition import ImageRegion
    from bgi_touch.vision.coordinate import ScreenTransform
    from bgi_touch.vision.game_ui import (
        MAP_CLOSE,
        MAP_FALLBACK_MARKERS,
        PAIMON_HUD,
        is_big_map_ui,
        is_main_ui,
    )

    template = PAIMON_HUD.template.bgr
    gameplay = np.zeros((1080, 1920, 3), dtype=np.uint8)
    gameplay[40:40 + template.shape[0], 110:110 + template.shape[1]] = template
    menu = np.zeros_like(gameplay)

    class Context:
        transform = ScreenTransform(1920, 1080)

    ctx = Context()
    assert is_main_ui(ctx, gameplay)
    assert not is_main_ui(ctx, menu)

    # AutoPick must share the same guard so a translucent menu cannot expose
    # the minimap and trigger OCR interaction clicks behind it.
    from bgi_touch.triggers.autopick import AutoPickTrigger
    trigger = AutoPickTrigger.__new__(AutoPickTrigger)
    trigger.ctx = ctx
    assert trigger._is_gameplay_frame(ImageRegion(ctx, gameplay))
    assert not trigger._is_gameplay_frame(ImageRegion(ctx, menu))

    # The translucent big map may retain the Paimon marker.  Its close button
    # must still win the scene classification so map labels cannot be treated
    # as pickup entries or trigger forced interaction.
    big_map = gameplay.copy()
    map_template = MAP_CLOSE.template.bgr
    map_h, map_w = map_template.shape[:2]
    big_map[40:40 + map_h, 1740:1740 + map_w] = map_template
    assert not trigger._is_gameplay_frame(ImageRegion(ctx, big_map))
    assert not is_main_ui(ctx, big_map)

    # A resampled close button may miss while another fixed map control still
    # matches.  The fallback markers must classify that frame consistently.
    if MAP_FALLBACK_MARKERS:
        fallback = gameplay.copy()
        marker = MAP_FALLBACK_MARKERS[0].template.bgr
        marker_h, marker_w = marker.shape[:2]
        fallback[24:24 + marker_h, 1500:1500 + marker_w] = marker
        assert is_big_map_ui(ctx, fallback)
        assert not is_main_ui(ctx, fallback)


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


def test_trigger_loop_pause_resume_preserves_configured_but_inactive_triggers():
    from bgi_touch.triggers.loop import TriggerLoop

    class Context:
        def capture_region(self):
            return object()

    class Trigger:
        name = "AutoPick"
        enabled = True

    loop = TriggerLoop(Context(), log=lambda _: None)
    trigger = Trigger()
    loop.add(trigger)

    state = loop.pause()
    assert state[0] == [trigger]
    assert state[1] is False
    assert loop.triggers == []

    loop.resume(state)
    assert loop.triggers == [trigger]
    assert not loop.active


def test_teleport_exclusive_scope_pauses_configured_inactive_trigger_loop():
    from bgi_touch.pathing.tp import TpTask
    from bgi_touch.triggers.loop import TriggerLoop

    class Context:
        def capture_region(self):
            return object()

    class Trigger:
        name = "AutoPick"
        enabled = True

    ctx = Context()
    loop = TriggerLoop(ctx, log=lambda _: None)
    trigger = Trigger()
    loop.add(trigger)
    ctx._trigger_loop = loop
    task = TpTask.__new__(TpTask)
    task.ctx = ctx

    assert not loop.active
    with task.exclusive_triggers():
        assert loop.triggers == []
        assert not loop.active

    assert loop.triggers == [trigger]
    assert not loop.active


def test_trigger_loop_exclusive_scopes_are_serialized_and_reentrant():
    from bgi_touch.triggers.loop import TriggerLoop

    loop = TriggerLoop(SimpleNamespace(), log=lambda _: None)
    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()
    order = []

    def first():
        with loop.exclusive():
            order.append("first-enter")
            entered.set()
            assert release.wait(1.0)
            with loop.exclusive():
                order.append("nested-enter")
            order.append("first-exit")

    def second():
        assert entered.wait(1.0)
        with loop.exclusive():
            order.append("second-enter")
            second_entered.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert entered.wait(1.0)
    time.sleep(0.03)
    assert not second_entered.is_set()
    release.set()
    first_thread.join(1.0)
    second_thread.join(1.0)

    assert order == ["first-enter", "nested-enter", "first-exit", "second-enter"]


def test_trigger_loop_runs_only_the_exclusive_trigger():
    from bgi_touch.triggers.loop import TriggerLoop

    calls = []

    class Context:
        def capture_region(self):
            return object()

    class Trigger:
        enabled = True

        def __init__(self, name, exclusive=False):
            self.name = name
            self.is_exclusive = exclusive

        def on_frame(self, _region):
            calls.append(self.name)

    loop = TriggerLoop(Context(), interval_s=0.01, log=lambda _: None)
    normal = Trigger("AutoPick")
    exclusive = Trigger("AutoFish", exclusive=True)
    loop.add(normal)
    loop.add(exclusive)
    loop.start()
    deadline = time.monotonic() + 1.0
    while not calls and time.monotonic() < deadline:
        time.sleep(0.01)
    loop.stop()
    assert calls
    assert set(calls) == {"AutoFish"}


def test_trigger_loop_honors_context_input_gate_at_frame_boundary():
    from bgi_touch.triggers.loop import TriggerLoop

    calls = []

    class Context:
        input_exclusive = False

        def capture_region(self):
            # Simulate a task acquiring ownership immediately after the frame
            # was decoded but before any trigger callback can emit input.
            self.input_exclusive = True
            return object()

    class Trigger:
        name = "AutoPick"
        enabled = True

        def on_frame(self, _region):
            calls.append("input")

    ctx = Context()
    loop = TriggerLoop(ctx, interval_s=0.01, log=lambda _: None)
    loop.add(Trigger())
    loop.start()
    time.sleep(0.05)
    loop.stop()
    assert calls == []


def test_autofishing_trigger_controls_bar_and_releases_on_scene_exit(monkeypatch):
    from bgi_touch.triggers.autofishing import AutoFishingTrigger

    calls = []

    class Input:
        def attack_down(self): calls.append("down")
        def attack_up(self): calls.append("up")
        def attack(self): calls.append("attack")
        def key_press(self, key): calls.append(key)

    ctx = SimpleNamespace(input=Input())
    trigger = AutoFishingTrigger(ctx, log=lambda _: None)
    region = SimpleNamespace(bgr=np.zeros((1080, 1920, 3), dtype=np.uint8))
    exit_hit = SimpleNamespace(is_exist=lambda: True)
    no_hit = SimpleNamespace(is_exist=lambda: False)
    current = {"exit_fishing": exit_hit, "lift_rod": no_hit, "wait_bite": no_hit, "Space": no_hit}
    trigger._find = lambda _region, name, _roi: current[name]
    monkeypatch.setattr(
        "bgi_touch.triggers.autofishing.get_fish_bar_rects",
        lambda _frame: [(10, 10, 4, 4), (20, 10, 80, 4)],
    )
    monkeypatch.setattr(
        "bgi_touch.triggers.autofishing.fish_bar_action",
        lambda _rects: "hold",
    )
    monkeypatch.setattr(
        "bgi_touch.triggers.autofishing.match_fish_bite_words",
        lambda _frame, _roi: False,
    )

    trigger.on_frame(region)
    assert trigger.is_exclusive is True
    assert calls == ["down"]

    current["exit_fishing"] = no_hit
    trigger.on_frame(region)
    assert trigger.is_exclusive is False
    assert calls == ["down", "up"]


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
            "pickKey": "G",
            "textList": ["调查"],
            "mode": "Blacklist",
            "blackList": ["调查"],
            "fuzzyBlacklist": ["进入"],
            "whiteList": ["甜甜花"],
            "doNotPickList": ["薄荷"],
            "blacklistModePickEnabled": True,
            "whitelistModeDoNotPickEnabled": False,
        },
    })
    assert ctx.calls == [(('AutoPick',), {
        "force_interaction": True,
        "pick_key": "G",
        "text_list": ["调查"],
        "mode": "Blacklist",
        "whitelist": ["甜甜花"],
        "blacklist": ["调查"],
        "fuzzy_blacklist": ["进入"],
        "whitelist_exclusions": ["薄荷"],
        "blacklist_mode_pick_enabled": True,
        "whitelist_mode_do_not_pick_enabled": False,
    })]


def test_autopick_defaults_to_recommended_whitelist_and_supports_blacklist():
    from bgi_touch.triggers.autopick import AutoPickTrigger

    trigger = AutoPickTrigger.__new__(AutoPickTrigger)
    trigger.mode = "Whitelist"
    trigger.whitelist = frozenset({"甜甜花", "薄荷"})
    trigger.blacklist = {"调查"}
    trigger.fuzzy_blacklist = ("进入",)
    trigger.blacklist_mode_pick_enabled = False
    trigger.blacklist_pick_list = frozenset()
    trigger.text_list = frozenset()

    assert trigger._should_pick("甜甜花")
    assert not trigger._should_pick("优兰尼娅湖")
    assert not trigger._should_pick("聚所")
    for text in (
        "叮铃", "眶螂", "蛋卷坊", "西风成垒", "望崖营壁",
        "魔女的花园", "月谕圣牌",
    ):
        assert not trigger._should_pick(text)

    trigger.mode = "Blacklist"
    assert trigger._should_pick("甜甜花")
    assert not trigger._should_pick("调查")
    assert not trigger._should_pick("进入秘境")


def test_autopick_uses_profile_key_for_mobile_candidates_and_external_text_list():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.triggers.autopick import AutoPickTrigger

    key_press = Mock()
    ctx = SimpleNamespace(input=SimpleNamespace(key_press=key_press))
    trigger = AutoPickTrigger(
        ctx,
        mode="Blacklist",
        text_list=["调查"],
        pick_key="G",
    )
    assert trigger._should_pick("调查")
    assert not trigger._should_pick("甜甜花")

    class Hit:
        def __init__(self, text, x, y):
            self.text = text
            self.x = x
            self.y = y
            self.clicks = 0

        def click(self):
            self.clicks += 1

    lower = Hit("调查", 1200, 520)
    upper = Hit("调查", 1200, 420)

    class Region:
        def find_multi(self, _recognition, *, limit):
            assert limit == 5
            return [lower, upper]

    trigger._is_gameplay_frame = lambda _region: True
    trigger.on_frame(Region())

    key_press.assert_called_once_with("G")
    assert upper.clicks == 0
    assert lower.clicks == 0

    # A throttled edge must not consume a newly recognized candidate.
    trigger._last_action_at = time.monotonic()
    trigger.on_frame(Region())
    assert key_press.call_count == 1
    assert trigger._last_text == "调查"


def test_autopick_uses_upstream_default_lists_and_mode_specific_overrides():
    from bgi_touch.triggers.autopick import AutoPickTrigger

    whitelist = AutoPickTrigger(
        object(),
        whitelist=["自定义采集物"],
        whitelist_exclusions=["甜甜花"],
    )
    assert whitelist._should_pick("自定义采集物")
    assert not whitelist._should_pick("甜甜花")

    blacklist = AutoPickTrigger(
        object(),
        mode="Blacklist",
        blacklist=["自定义机关"],
        whitelist=["进入"],
        blacklist_mode_pick_enabled=True,
    )
    assert len(blacklist.blacklist) >= 4900
    assert not blacklist._should_pick("自定义机关")
    assert not blacklist._should_pick("退出秘境")
    assert blacklist._should_pick("进入")


def test_autopick_reads_bettergi_user_pick_lists(tmp_path):
    from bgi_touch.triggers.autopick import AutoPickTrigger

    (tmp_path / "pick_black_lists.txt").write_text("自定义黑名单\n# 注释\n", encoding="utf-8")
    (tmp_path / "pick_fuzzy_black_lists.txt").write_text("危险区域\n", encoding="utf-8")
    (tmp_path / "pick_white_lists.txt").write_text("黑名单模式拾取\n", encoding="utf-8")

    trigger = AutoPickTrigger(
        object(),
        mode="Blacklist",
        blacklist_mode_pick_enabled=True,
        user_dir=tmp_path,
    )

    assert not trigger._should_pick("自定义黑名单")
    assert not trigger._should_pick("危险区域掉落")
    assert trigger._should_pick("黑名单模式拾取")


def test_autopick_requires_interaction_prompt_before_scanning_real_regions():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.triggers.autopick import AutoPickTrigger

    class Region:
        def find(self, _recognition):
            return SimpleNamespace(is_exist=lambda: False)

        find_multi = Mock(side_effect=AssertionError("OCR must wait for prompt"))

    trigger = AutoPickTrigger(SimpleNamespace(), mode="Blacklist")
    trigger._is_gameplay_frame = lambda _region: True
    trigger.on_frame(Region())
    trigger.log = Mock()
    trigger.log.assert_not_called()


def test_autopick_scrolls_hidden_interaction_list_from_shared_frame():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.triggers.autopick import AutoPickTrigger
    from bgi_touch.vision.coordinate import ScreenTransform

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for x, y, color in AutoPickTrigger._SCROLL_ICON_SAMPLES:
        frame[y, x] = color

    vertical_scroll = Mock()
    ctx = SimpleNamespace(
        input=SimpleNamespace(vertical_scroll=vertical_scroll),
        transform=ScreenTransform(1920, 1080),
        input_exclusive=False,
    )
    region = SimpleNamespace(bgr=frame)
    trigger = AutoPickTrigger.__new__(AutoPickTrigger)
    trigger.ctx = ctx
    trigger.enabled = True
    trigger.require_pick_prompt = True
    trigger._last_scroll_at = float("-inf")
    trigger._is_gameplay_frame = lambda _region: True
    trigger._find_pick_prompt = lambda _region: None
    trigger.log = Mock()

    trigger.on_frame(region)

    vertical_scroll.assert_called_once_with(2)
    trigger.log.assert_called_once_with("[AutoPick] 滚动交互列表")


def test_autopick_scroll_marker_requires_all_samples_and_is_throttled():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.triggers.autopick import AutoPickTrigger
    from bgi_touch.vision.coordinate import ScreenTransform

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for x, y, color in AutoPickTrigger._SCROLL_ICON_SAMPLES[:-1]:
        frame[y, x] = color
    scroll = Mock()
    ctx = SimpleNamespace(
        input=SimpleNamespace(vertical_scroll=scroll),
        transform=ScreenTransform(1920, 1080),
        input_exclusive=False,
    )
    trigger = AutoPickTrigger.__new__(AutoPickTrigger)
    trigger.ctx = ctx
    trigger.require_pick_prompt = True
    trigger._last_scroll_at = float("-inf")
    trigger._is_gameplay_frame = lambda _region: True
    trigger._find_pick_prompt = lambda _region: None
    trigger.log = Mock()
    region = SimpleNamespace(bgr=frame)

    trigger.on_frame(region)
    scroll.assert_not_called()

    for x, y, color in AutoPickTrigger._SCROLL_ICON_SAMPLES:
        frame[y, x] = color
    trigger.on_frame(region)
    trigger.on_frame(region)
    scroll.assert_called_once_with(2)


def test_autopick_rejects_chat_or_settings_icon_before_ocr():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.triggers.autopick import AutoPickTrigger

    prompt = SimpleNamespace(x=1090, y=420, height=32, is_exist=lambda: True)
    icon = SimpleNamespace(is_exist=lambda: True)

    class IconRegion:
        def find(self, _recognition):
            return icon

    class Region:
        def __init__(self):
            self.find_calls = 0
            self.find_multi = Mock(side_effect=AssertionError("excluded icon must stop OCR"))

        def find(self, _recognition):
            self.find_calls += 1
            return prompt

        def derive_crop(self, *_args):
            return IconRegion()

    trigger = AutoPickTrigger(SimpleNamespace(), mode="Blacklist")
    trigger._is_gameplay_frame = lambda _region: True
    trigger.on_frame(Region())


def test_autopick_processes_ocr_edge_noise_like_bettergi():
    from bgi_touch.triggers.autopick import _process_ocr_text

    assert _process_ocr_text("  [甜甜花 ") == "「甜甜花」"
    assert _process_ocr_text("...甜甜花!!!") == "甜甜花"
    assert _process_ocr_text("--") == ""


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


def test_teleport_selection_reuses_the_map_frame_for_icon_fallback():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.pathing.tp import TpTask

    map_region = object()
    ctx = SimpleNamespace(
        transform=SimpleNamespace(device_width=1000, device_height=500),
        device=SimpleNamespace(tap=Mock()),
    )
    task = TpTask.__new__(TpTask)
    task.ctx = ctx
    task.log = Mock()
    task._anchor_icons_near = Mock(return_value=[])
    task._find_and_tap_confirm = Mock(return_value=True)

    assert task._select_target_and_confirm(
        400,
        250,
        50,
        map_region=map_region,
    )

    task._anchor_icons_near.assert_called_once_with(
        400,
        250,
        100,
        region=map_region,
    )
    ctx.device.tap.assert_called_once_with(
        400,
        250,
        image_width=1000,
        image_height=500,
    )


def test_map_drag_returns_the_frame_after_its_own_gesture():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.pathing.tp import TpTask

    feedback = np.ones((4, 8, 3), dtype=np.uint8)
    device = SimpleNamespace(last_frame_version=41, swipe=Mock())
    ctx = SimpleNamespace(
        device=device,
        transform=SimpleNamespace(device_width=1000, device_height=600),
        capture_bgr_after_frame=Mock(return_value=feedback),
        sleep=Mock(),
    )
    task = TpTask.__new__(TpTask)
    task.ctx = ctx

    assert task._drag_map(120, -60) is feedback

    ctx.capture_bgr_after_frame.assert_called_once_with(41, timeout_ms=1800)
    ctx.sleep.assert_called_once_with(700)


def test_map_drag_uses_cursor_returned_by_the_final_swipe():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.pathing.tp import TpTask

    feedback = np.ones((4, 8, 3), dtype=np.uint8)
    device = SimpleNamespace(last_frame_version=41, swipe=Mock())

    def swipe_and_publish(*_args, **_kwargs):
        device.last_frame_version = 44

    device.swipe.side_effect = swipe_and_publish
    ctx = SimpleNamespace(
        device=device,
        transform=SimpleNamespace(device_width=1000, device_height=600),
        capture_bgr_after_frame=Mock(return_value=feedback),
        sleep=Mock(),
    )
    task = TpTask.__new__(TpTask)
    task.ctx = ctx

    assert task._drag_map(120, -60) is feedback
    ctx.capture_bgr_after_frame.assert_called_once_with(44, timeout_ms=1800)


def test_map_drag_seeds_a_frame_cursor_when_screenshot_has_no_version():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.pathing.tp import TpTask

    feedback = np.ones((4, 8, 3), dtype=np.uint8)
    device = SimpleNamespace(
        last_frame_version=None,
        wait_for_frame=Mock(return_value={"frame_version": 17}),
        swipe=Mock(),
    )
    ctx = SimpleNamespace(
        device=device,
        transform=SimpleNamespace(device_width=1000, device_height=600),
        capture_bgr_after_frame=Mock(return_value=feedback),
        sleep=Mock(),
    )
    task = TpTask.__new__(TpTask)
    task.ctx = ctx

    assert task._drag_map(120, -60) is feedback

    device.wait_for_frame.assert_called_once_with(
        after_version=None,
        timeout_ms=1200,
    )
    ctx.capture_bgr_after_frame.assert_called_once_with(17, timeout_ms=1800)


def test_map_drag_splits_long_gesture_without_intermediate_screenshots():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.pathing.tp import TpTask

    feedback = np.ones((4, 8, 3), dtype=np.uint8)
    device = SimpleNamespace(last_frame_version=9, swipe=Mock())
    ctx = SimpleNamespace(
        device=device,
        transform=SimpleNamespace(device_width=1920, device_height=1080),
        capture_bgr_after_frame=Mock(return_value=feedback),
        capture_bgr=Mock(),
        sleep=Mock(),
    )
    task = TpTask.__new__(TpTask)
    task.ctx = ctx

    assert task._drag_map(900, 0) is feedback

    # A long map move is divided into four central gestures, while the frame
    # stream is consumed only once after the complete gesture sequence.
    assert device.swipe.call_count == 4
    sent_dx = 0.0
    for swipe_call in device.swipe.call_args_list:
        x1, y1, x2, y2 = swipe_call.args[:4]
        sent_dx += x2 - x1
        assert 128 <= x1 <= 1792
        assert 128 <= x2 <= 1792
        assert 96 <= y1 <= 984
        assert 96 <= y2 <= 984
    assert sent_dx == pytest.approx(900)
    ctx.capture_bgr.assert_not_called()
    ctx.capture_bgr_after_frame.assert_called_once_with(9, timeout_ms=1800)
    assert [item.args[0] for item in ctx.sleep.call_args_list] == [90, 90, 90, 700]


def test_move_map_consumes_drag_feedback_without_requesting_an_extra_frame():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.pathing.tp import TpTask

    initial = np.zeros((4, 8, 3), dtype=np.uint8)
    feedback = np.ones((4, 8, 3), dtype=np.uint8)
    ctx = SimpleNamespace(
        transform=SimpleNamespace(device_width=100, device_height=60),
        capture_bgr=Mock(return_value=initial),
    )
    task = TpTask.__new__(TpTask)
    task.ctx = ctx
    task.log = Mock()
    task.open_map = Mock(return_value=True)
    task.big = SimpleNamespace(
        world_to_feature=Mock(return_value=(10.0, 0.0)),
        locate_view=Mock(side_effect=lambda frame: (
            (0.0, 0.0, 1.0) if frame is initial else (10.0, 0.0, 1.0)
        )),
    )
    task._drag_map = Mock(return_value=feedback)

    assert task._move_map_to(1, 2)

    ctx.capture_bgr.assert_called_once_with()
    task._drag_map.assert_called_once_with(-10.0, -0.0)
    assert task._last_located_frame is feedback


def test_map_move_refreshes_a_duplicate_post_gesture_frame_once():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.pathing.tp import TpTask

    initial = np.zeros((4, 8, 3), dtype=np.uint8)
    stale = np.ones((4, 8, 3), dtype=np.uint8)
    refreshed = np.full((4, 8, 3), 2, dtype=np.uint8)
    ctx = SimpleNamespace(
        transform=SimpleNamespace(device_width=100, device_height=60),
        capture_bgr=Mock(side_effect=[initial, refreshed]),
    )
    task = TpTask.__new__(TpTask)
    task.ctx = ctx
    task.log = Mock()
    task.open_map = Mock(return_value=True)
    task.big = SimpleNamespace(
        world_to_feature=Mock(return_value=(10.0, 0.0)),
        locate_view=Mock(side_effect=[
            (0.0, 0.0, 1.0),  # before the swipe
            (0.0, 0.0, 1.0),  # stale observation returned by the gesture
            (10.0, 0.0, 1.0),  # fresh frame after the one-off refresh
        ]),
    )
    task._drag_map = Mock(return_value=stale)

    assert task._move_map_to(1, 2)
    assert ctx.capture_bgr.call_count == 2
    task._drag_map.assert_called_once_with(-10.0, -0.0)


def test_map_move_does_not_repeat_swipe_while_feedback_frame_stays_stale():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.pathing.tp import TpTask

    initial = object()
    stale = object()
    ctx = SimpleNamespace(
        transform=SimpleNamespace(device_width=100, device_height=60),
        capture_bgr=Mock(side_effect=[initial, stale, stale, stale]),
        sleep=Mock(),
    )
    task = TpTask.__new__(TpTask)
    task.ctx = ctx
    task.log = Mock()
    task.open_map = Mock(return_value=True)
    task.big = SimpleNamespace(
        world_to_feature=Mock(return_value=(10.0, 0.0)),
        locate_view=Mock(return_value=(0.0, 0.0, 1.0)),
    )
    task._drag_map = Mock(return_value=stale)
    task._recover_device_channel = Mock(return_value=False)

    with pytest.raises(RuntimeError, match="地图移动失败"):
        task._move_map_view_to(
            1,
            2,
            timeout_s=5,
            log_prefix="[map] 迭代",
            max_iterations=4,
            error_message="地图移动失败",
        )

    # A stale observation must not turn into a stream of identical swipes.
    task._drag_map.assert_called_once_with(-10.0, -0.0)
    task._recover_device_channel.assert_called_once_with("连续拖动后地图视野未变化")


def test_map_move_stale_guard_prefers_the_device_frame_cursor():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.pathing.tp import TpTask

    initial = object()
    stale = object()
    refreshed = object()
    ctx = SimpleNamespace(
        transform=SimpleNamespace(device_width=100, device_height=60),
        device=SimpleNamespace(last_frame_version=42),
        capture_bgr=Mock(return_value=initial),
        capture_bgr_after_frame=Mock(return_value=refreshed),
    )
    task = TpTask.__new__(TpTask)
    task.ctx = ctx
    task.log = Mock()
    task.open_map = Mock(return_value=True)
    task.big = SimpleNamespace(
        world_to_feature=Mock(return_value=(10.0, 0.0)),
        locate_view=Mock(side_effect=[
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (10.0, 0.0, 1.0),
        ]),
    )
    task._drag_map = Mock(return_value=stale)

    assert task._move_map_to(1, 2)

    ctx.capture_bgr_after_frame.assert_called_once_with(42, timeout_ms=1800)
    ctx.capture_bgr.assert_called_once_with()
    task._drag_map.assert_called_once_with(-10.0, -0.0)


def test_map_move_flips_profile_swipe_direction_when_feedback_is_opposite():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.pathing.tp import TpTask

    frames = [object(), object(), object()]
    ctx = SimpleNamespace(
        transform=SimpleNamespace(device_width=100, device_height=60),
        capture_bgr=Mock(return_value=frames[0]),
    )
    task = TpTask.__new__(TpTask)
    task.ctx = ctx
    task.log = Mock()
    task.open_map = Mock(return_value=True)
    task.big = SimpleNamespace(
        world_to_feature=Mock(return_value=(10.0, 0.0)),
        locate_view=Mock(side_effect=[
            (0.0, 0.0, 1.0),
            (-5.0, 0.0, 1.0),  # the profile moved away from the target
            (10.0, 0.0, 1.0),
        ]),
    )
    task._drag_map = Mock(side_effect=[frames[1], frames[2]])

    assert task._move_map_to(1, 2)
    assert task._drag_map.call_args_list[0].args == (-10.0, -0.0)
    assert task._drag_map.call_args_list[1].args == (15.0, 0.0)
    assert any("拖动方向相反" in call.args[0] for call in task.log.call_args_list)


def test_map_move_flips_direction_when_target_distance_does_not_shrink():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.pathing.tp import TpTask

    frames = [object(), object(), object()]
    ctx = SimpleNamespace(
        transform=SimpleNamespace(device_width=100, device_height=60),
        capture_bgr=Mock(return_value=frames[0]),
    )
    task = TpTask.__new__(TpTask)
    task.ctx = ctx
    task.log = Mock()
    task.open_map = Mock(return_value=True)
    task.big = SimpleNamespace(
        world_to_feature=Mock(return_value=(10.0, 0.0)),
        locate_view=Mock(side_effect=[
            (0.0, 0.0, 1.0),
            (1.0, 10.0, 1.0),  # moves, but the target distance grows
            (10.0, 0.0, 1.0),
        ]),
    )
    task._drag_map = Mock(side_effect=[frames[1], frames[2]])

    assert task._move_map_to(1, 2)
    assert task._drag_map.call_args_list[0].args == (-10.0, -0.0)
    assert task._drag_map.call_args_list[1].args == (9.0, -10.0)
    assert any("目标距离未缩小" in call.args[0] for call in task.log.call_args_list)


def test_map_move_recovers_when_small_view_drift_does_not_converge():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.pathing.tp import TpTask

    frames = [object() for _ in range(5)]
    initial, retry_initial = frames[0], frames[4]
    ctx = SimpleNamespace(
        transform=SimpleNamespace(device_width=100, device_height=60),
        capture_bgr=Mock(side_effect=[initial, retry_initial]),
        sleep=Mock(),
    )
    task = TpTask.__new__(TpTask)
    task.ctx = ctx
    task.log = Mock()
    task.open_map = Mock(return_value=True)
    task.big = SimpleNamespace(
        world_to_feature=Mock(return_value=(100.0, 0.0)),
        locate_view=Mock(side_effect=[
            (0.0, 0.0, 1.0),
            (2.0, 2.0, 1.0),
            (4.0, 4.0, 1.0),
            (6.0, 6.0, 1.0),
            (0.0, 0.0, 1.0),  # fresh view after channel recovery
            (100.0, 0.0, 1.0),
        ]),
    )
    task._drag_map = Mock(side_effect=frames[1:])
    task._recover_device_channel = Mock(return_value=True)

    assert task._move_map_to(1, 2)

    task._recover_device_channel.assert_called_once_with("目标距离连续未缩小")
    # Three non-converging gestures are allowed before recovery, then the
    # recovered channel receives one fresh gesture and reaches the target.
    assert task._drag_map.call_count == 4
    assert ctx.capture_bgr.call_count == 2


def test_map_move_stops_after_failed_distance_progress_recovery():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.pathing.tp import TpTask

    frames = [object() for _ in range(4)]
    ctx = SimpleNamespace(
        transform=SimpleNamespace(device_width=100, device_height=60),
        capture_bgr=Mock(return_value=frames[0]),
        sleep=Mock(),
    )
    task = TpTask.__new__(TpTask)
    task.ctx = ctx
    task.log = Mock()
    task.open_map = Mock(return_value=True)
    task.big = SimpleNamespace(
        world_to_feature=Mock(return_value=(100.0, 0.0)),
        locate_view=Mock(side_effect=[
            (0.0, 0.0, 1.0),
            (2.0, 2.0, 1.0),
            (4.0, 4.0, 1.0),
            (6.0, 6.0, 1.0),
        ]),
    )
    task._drag_map = Mock(side_effect=frames[1:])
    task._recover_device_channel = Mock(return_value=False)

    with pytest.raises(RuntimeError, match="地图移动失败"):
        task._move_map_view_to(
            1,
            2,
            timeout_s=5,
            log_prefix="[map] 迭代",
            max_iterations=8,
            error_message="地图移动失败",
        )

    task._recover_device_channel.assert_called_once_with("目标距离连续未缩小")
    # Recovery failure must end the gesture loop instead of issuing another
    # swipe against the same, non-converging map view.
    assert task._drag_map.call_count == 3


def test_teleport_confirm_ignores_top_map_label_and_clicks_bottom_button():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.pathing.tp import TpTask

    class Hit:
        def __init__(self, text, y):
            self.text = text
            self.dx, self.dy, self.dw, self.dh = 700, y, 80, 30
            self.clicks = 0

        def is_exist(self):
            return True

        def click(self):
            self.clicks += 1

    top = Hit("传送", 40)
    bottom = Hit("传送", 400)

    class Region:
        def find(self, _):
            return SimpleNamespace(is_exist=lambda: False)

        def find_multi(self, *_args, **_kwargs):
            return [top, bottom]

    task = TpTask.__new__(TpTask)
    task.ctx = SimpleNamespace(
        transform=SimpleNamespace(device_width=1000, device_height=500),
        capture_region=Mock(return_value=Region()),
        sleep=Mock(),
    )
    task.log = Mock()
    task._go_teleport = object()

    assert task._find_and_tap_confirm(timeout_s=0.2, initial_delay_ms=0)
    assert top.clicks == 0
    assert bottom.clicks == 1


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
