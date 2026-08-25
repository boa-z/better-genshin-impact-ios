"""无设备回归测试：解析器、坐标系、宏转换、（有资产时）地图定位。

运行：.venv/bin/python -m pytest tests/ -q
"""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
ASSETS = Path(__file__).parents[1] / "assets" / "map"


# ---- 坐标系 ----

def test_screen_transform_anchors():
    from bgi_touch.vision.coordinate import ScreenTransform
    t = ScreenTransform(2816, 1296)
    assert t.scale == pytest.approx(1.2)
    # 左锚点：贴左边元素
    x, y = t.to_device(62, 19)
    assert x == pytest.approx(62 * 1.2)
    # 右锚点：贴右边元素随设备加宽外移
    x, _ = t.to_device(1870, 50)
    assert x == pytest.approx(2816 - (1920 - 1870) * 1.2)
    # 中锚点往返
    rx, ry = t.to_ref(*t.to_device(960, 540))
    assert (rx, ry) == (pytest.approx(960), pytest.approx(540))


def test_teyvat_world_image_roundtrip():
    from bgi_touch.pathing.map_locator import MapConfig
    c = MapConfig()
    ix, iy = c.world_to_image(0, 0)
    assert (ix, iy) == (32768, 16384)
    wx, wy = c.image_to_world(*c.world_to_image(-910.5, 2249.9))
    assert wx == pytest.approx(-910.5)
    assert wy == pytest.approx(2249.9)


# ---- 战斗 DSL ----

def test_combat_parser():
    from bgi_touch.combat.dsl import parse_combat_script
    lines = parse_combat_script(
        "// c\n钟离 s(0.2), e(hold), wait(0.2), click(middle)\nwait(1),keypress(VK_SPACE)\n")
    assert lines[0].character == "钟离"
    assert [c.action for c in lines[0].commands] == ["s", "e", "wait", "click"]
    assert lines[0].commands[1].params == ["hold"]
    assert lines[1].character is None
    assert lines[1].commands[1].params == ["VK_SPACE"]


def test_combat_parser_normalizes_upstream_method_aliases():
    from bgi_touch.combat.dsl import parse_combat_script

    lines = parse_combat_script(
        "普攻(0.1), 重击(0.2), 等待(0.1), 完成, 检测, r, j, verticalscroll(1)"
    )

    assert [command.action for command in lines[0].commands] == [
        "attack", "charge", "wait", "ready", "check", "aim", "jump", "scroll",
    ]


def test_combat_executor_supports_skill_flags_and_virtual_mouse_keys():
    from unittest.mock import Mock

    from bgi_touch.combat.dsl import CombatCommand, CombatExecutor

    input_sim = type("Input", (), {
        "key_press": Mock(), "key_down": Mock(), "key_up": Mock(),
        "attack": Mock(), "attack_down": Mock(), "attack_up": Mock(),
        "button_down": Mock(), "button_up": Mock(), "tap_button": Mock(),
    })()
    skill_ready = Mock(side_effect=[False, True])
    executor = CombatExecutor(
        input_sim, sleep=lambda _milliseconds: None, skill_ready=skill_ready,
    )

    executor.exec(CombatCommand("e", ["fast"]))
    executor.exec(CombatCommand("skill", ["wait"]))
    executor.exec(CombatCommand("keypress", ["VK_LBUTTON"]))
    executor.exec(CombatCommand("keypress", ["VK_RBUTTON"]))
    executor.exec(CombatCommand("keydown", ["VK_RBUTTON"]))
    executor.exec(CombatCommand("keyup", ["VK_RBUTTON"]))
    executor.exec(CombatCommand("keypress", ["VK_MBUTTON"]))

    assert skill_ready.call_count == 2
    assert input_sim.key_press.call_args_list == [
        (("E",), {"hold_ms": 80}),
        (("LSHIFT",), {}),
    ]
    input_sim.attack.assert_called_once_with()
    input_sim.button_down.assert_called_once_with("sprint")
    input_sim.button_up.assert_called_once_with("sprint")
    input_sim.tap_button.assert_called_once_with("elementalSight")


def test_combat_executor_checks_after_the_command():
    from bgi_touch.combat.dsl import CombatExecutor

    events = []
    input_sim = type("Input", (), {
        "key_press": lambda _self, key, **_kwargs: events.append(f"input:{key}"),
        "release_all": lambda _self: events.append("release"),
    })()

    def check():
        events.append("check")
        return True

    executor = CombatExecutor(input_sim, sleep=lambda _milliseconds: None,
                              check_combat_end=check)
    executor.run("keypress(E), check")

    assert events == ["input:E", "check", "release"]


def test_combat_executor_maps_keys():
    from unittest.mock import MagicMock
    from bgi_touch.combat.dsl import CombatExecutor
    inp = MagicMock()
    ex = CombatExecutor(inp, sleep=lambda ms: None)
    ex.run("e, q, jump, dash, w(0.01)")
    inp.key_press.assert_any_call("E", hold_ms=80)
    inp.key_press.assert_any_call("Q")
    inp.key_press.assert_any_call("SPACE")
    inp.key_down.assert_called_with("W")
    inp.key_up.assert_called_with("W")


# ---- 键鼠宏转换 ----

def test_keymouse_conversion():
    from bgi_touch.macro.keymouse import convert_keymouse
    macro = {
        "info": {"width": 1920, "height": 1080},
        "macroEvents": [
            {"type": 0, "keyCode": 87, "time": 0},
            {"type": 3, "mouseX": 50, "mouseY": 0, "time": 10},
            {"type": 3, "mouseX": 50, "mouseY": 0, "time": 60},
            {"type": 1, "keyCode": 87, "time": 500},
            {"type": 2, "mouseX": 960, "mouseY": 540, "time": 600},
            {"type": 4, "mouseButton": "Left", "time": 620},
            {"type": 5, "mouseButton": "Left", "time": 700},
        ],
    }
    events, warnings = convert_keymouse(macro)
    kinds = [e.kind for e in events]
    assert kinds == ["keyDown", "cameraBy", "keyUp", "tapRef"]
    cam = events[1]
    assert cam.x == pytest.approx(100)  # 两次相对移动合并
    assert not warnings


# ---- 键位规范化 ----

def test_key_normalization():
    from bgi_touch.input.layout import normalize_key
    assert normalize_key("VK_W") == "W"
    assert normalize_key("LeftShift") == "LSHIFT"
    assert normalize_key(" ") == "SPACE"
    assert normalize_key("esc") == "ESCAPE"
    assert normalize_key("KeyE") == "E"
    assert normalize_key("Digit4") == "4"


def test_devicehub_profile_maps_native_codes_to_bettergi_keys():
    from bgi_touch.input.layout import ControlLayout, DeviceHubProfile

    profile = {
        "version": 2,
        "name": "test-profile",
        "mappings": [
            {"type": "Press", "bind": ["Space"], "position": {"x": 0.88, "y": 0.66}},
            {"type": "Press", "bind": ["ShiftLeft"], "position": {"x": 0.88, "y": 0.86}},
            {"type": "Press", "bind": ["KeyE"], "position": {"x": 0.72, "y": 0.89}},
            {"type": "Press", "bind": ["KeyR"], "position": {"x": 0.697, "y": 0.746}},
            {"type": "Press", "bind": ["KeyX"], "position": {"x": 0.795, "y": 0.778}},
            {"type": "Press", "bind": ["ControlLeft"], "position": {"x": 0.958, "y": 0.951}},
            {"type": "Press", "bind": ["KeyO"], "position": {"x": 0.742, "y": 0.064}},
            {
                "type": "DirectionPad",
                "bind": {"up": ["KeyW"], "down": ["KeyS"],
                         "left": ["KeyA"], "right": ["KeyD"]},
                "position": {"x": 0.18, "y": 0.75},
            },
        ],
    }
    layout = ControlLayout.load(devicehub_profile=profile)

    assert isinstance(layout.devicehub_profile, DeviceHubProfile)
    assert layout.profile_key("W") == "KeyW"
    assert layout.profile_key("E") == "KeyE"
    # The supplied profile deliberately places Space on sprint and ShiftLeft on jump.
    assert layout.profile_key("SPACE") == "ShiftLeft"
    assert layout.profile_key("LSHIFT") == "Space"
    assert layout.profile_key_for_button("aim") == "KeyR"
    assert layout.profile_key_for_button("attack") == "KeyX"
    assert layout.profile_key("X") == "ControlLeft"
    assert layout.profile_key("F5") == "KeyO"

    class FakeDevice:
        def __init__(self):
            self.inputs = []
            self.stopped = []
            self.start_kwargs = None

        def start_game_session(self, profile_name, **kwargs):
            assert profile_name == "test-profile"
            self.start_kwargs = kwargs
            return "session-1"

        def set_game_input(self, session_id, keys, **kwargs):
            self.inputs.append((session_id, list(keys)))

        def stop_game_session(self, session_id):
            self.stopped.append(session_id)

    from bgi_touch.input.simulator import InputSimulator
    from bgi_touch.vision.coordinate import ScreenTransform
    device = FakeDevice()
    simulator = InputSimulator(device, layout, ScreenTransform(1920, 1080))
    simulator.key_down("W")
    simulator.key_down("LSHIFT")
    simulator.key_up("W")
    simulator.key_press("E", hold_ms=25)
    simulator.key_press("X", hold_ms=25)
    simulator.attack_down()
    simulator.attack_up()
    assert device.inputs[:3] == [
        ("session-1", ["KeyW"]),
        ("session-1", ["KeyW", "Space"]),
        ("session-1", ["Space"]),
    ]
    assert ("session-1", ["Space", "KeyE"]) in device.inputs
    assert ("session-1", ["Space", "ControlLeft"]) in device.inputs
    assert ("session-1", ["Space", "KeyX"]) in device.inputs
    assert device.inputs[-2:] == [
        ("session-1", ["Space", "KeyX"]),
        ("session-1", ["Space"]),
    ]
    simulator.release_all()
    assert device.stopped == ["session-1"]
    assert device.start_kwargs["lease_ms"] >= 15000


def test_expired_devicehub_game_session_can_be_rebuilt():
    from bgi_touch.input.simulator import InputSimulator

    class Device:
        def __init__(self):
            self.stopped = []

        def stop_game_session(self, session_id):
            self.stopped.append(session_id)

    simulator = InputSimulator.__new__(InputSimulator)
    simulator.device = Device()
    simulator._profile_failed = False
    simulator._profile_session_id = "expired"

    assert simulator._drop_profile_session(
        "expired", RuntimeError("game control session not found")
    )
    assert simulator._profile_session_id is None
    assert simulator._profile_failed is False
    assert simulator.device.stopped == ["expired"]

    simulator._profile_session_id = "broken"
    assert not simulator._drop_profile_session("broken", RuntimeError("HID unavailable"))
    assert simulator._profile_failed is True


def test_device_client_stops_persistent_session_before_direct_touch():
    from unittest.mock import Mock, call

    from bgi_touch.device.client import DeviceClient

    client = DeviceClient.__new__(DeviceClient)
    client._game_session_id = "game-1"
    client._mapper = None
    client.call = Mock()

    client.tap(100, 200, image_width=1920, image_height=1080)

    assert client._game_session_id is None
    assert client.call.call_args_list == [
        call("stop_game_session", session_id="game-1"),
        call(
            "tap",
            x=100,
            y=200,
            hold_ms=None,
            wait_for_settle=False,
            image_width=1920,
            image_height=1080,
        ),
    ]


def test_direct_camera_swipe_restarts_held_native_profile_session():
    from bgi_touch.input.layout import ControlLayout, DeviceHubProfile
    from bgi_touch.input.simulator import InputSimulator
    from bgi_touch.vision.coordinate import ScreenTransform

    profile = DeviceHubProfile.from_dict({
        "name": "test-profile",
        "mappings": [
            {"type": "DirectionPad", "bind": {"up": ["KeyW"]}},
        ],
    })
    layout = ControlLayout.load(
        Path(__file__).parents[1] / "config" / "controls" / "genshin-default.json",
        devicehub_profile=profile,
    )

    class Device:
        def __init__(self):
            self.game_session_id = None
            self.started = []
            self.inputs = []
            self.stopped = []
            self.swipes = []

        def start_game_session(self, _profile_name, **_kwargs):
            session_id = f"session-{len(self.started) + 1}"
            self.started.append(session_id)
            self.game_session_id = session_id
            return session_id

        def set_game_input(self, session_id, keys, **_kwargs):
            self.inputs.append((session_id, list(keys)))

        def stop_game_session(self, session_id):
            self.stopped.append(session_id)
            self.game_session_id = None

        def swipe(self, *args, **kwargs):
            if self.game_session_id is not None:
                self.stop_game_session(self.game_session_id)
            self.swipes.append((args, kwargs))

    device = Device()
    simulator = InputSimulator(device, layout, ScreenTransform(1920, 1080))
    simulator.key_down("W")
    simulator.move_camera_by(70, 0)

    assert device.started == ["session-1", "session-2"]
    assert device.stopped == ["session-1"]
    assert len(device.swipes) == 1
    assert simulator._profile_session_id == "session-2"
    assert device.inputs[-1] == ("session-2", ["KeyW"])
    simulator.release_all()


def test_native_ui_layout_overlay_changes_space_semantics():
    from bgi_touch.input.layout import ControlLayout

    layout = ControlLayout.load(Path(__file__).parents[1] / "config" / "controls" / "genshin-native-ui.json")
    assert layout.binding("SPACE")["button"] == "jump"
    assert layout.binding("SPACE")["profileCode"] == "Space"


def test_auto_cook_color_peak_detector():
    import cv2
    import numpy as np

    from bgi_touch.tasks.auto_cook import count_target_color, update_peak

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # TargetCookColor is RGB (255, 192, 64), while the fixture is BGR.
    frame[700:720, 700:900] = (64, 192, 255)
    assert count_target_color(frame, rect=(600, 660, 730, 190)) == 4000
    candidate = stable = None
    built = None
    for value in (1000, 1005, 995):
        candidate, stable, built = update_peak(value, candidate, stable or 0)
    assert built == 1005


def test_auto_fishing_bar_detector_and_controller():
    import numpy as np

    from bgi_touch.tasks.auto_fishing import fish_bar_action, get_fish_bar_rects

    frame = np.zeros((240, 800, 3), dtype=np.uint8)
    yellow = (192, 255, 255)  # RGB (255, 255, 192), HSV_FULL hue≈60°
    frame[100:120, 120:140] = yellow
    frame[100:120, 180:580] = yellow
    rects = get_fish_bar_rects(frame)
    assert len(rects) == 2
    assert fish_bar_action(rects) == "hold"


def test_auto_fishing_model_maps_current_labels_and_bait_policy():
    from bgi_touch.tasks.fishing_model import (
        choose_bait,
        fishpond_from_detections,
        rod_state,
    )
    from bgi_touch.vision.yolo import Detection

    detections = [
        Detection(100, 100, 50, 30, 0.9, 13),  # medaka
        Detection(180, 100, 50, 30, 0.8, 13),
        Detection(260, 100, 60, 35, 0.85, 20),  # stickleback
        Detection(210, 160, 70, 25, 0.95, 18),  # rod
    ]
    pond = fishpond_from_detections(detections, include_target=True)
    assert [fish.kind.chinese_name for fish in pond.fishes] == ["花鳉", "棘鱼", "花鳉"]
    assert choose_bait(pond.fishes) == "果酿饵"
    assert pond.rod == (210, 160, 70, 25)
    state = rod_state(pond.rod, pond.fishes[0], 800, 450)
    assert state == 2


def test_auto_fishing_dispatcher_maps_upstream_full_auto_parameters():
    from unittest.mock import patch

    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.auto_fishing.AutoFishingTask") as task:
        task.return_value.run.return_value = True
        result = TaskDispatcher(object()).run_auto_fishing_task({
            "autoThrowRodEnabled": True,
            "autoThrowRodTimeOut": 18,
            "wholeProcessTimeoutSeconds": 360,
            "targetCatches": 5,
            "fishingTimePolicy": 2,
            "quitOnFinish": False,
        })
    assert result is True
    kwargs = task.call_args.kwargs
    assert kwargs["auto_throw_rod_enabled"] is True
    assert kwargs["throw_rod_timeout_s"] == 18
    assert kwargs["timeout_s"] == 360
    assert kwargs["target_catches"] == 5
    assert kwargs["fishing_time_policy"] == 2
    assert kwargs["quit_on_finish"] is False


def test_auto_fishing_time_policy_matches_bettergi_schedule():
    import pytest

    from bgi_touch.tasks.auto_fishing import (
        FishingTimePolicy,
        fishing_hours,
        parse_fishing_time_policy,
    )

    assert parse_fishing_time_policy(0) == FishingTimePolicy.ALL
    assert parse_fishing_time_policy("白天") == FishingTimePolicy.DAYTIME
    assert parse_fishing_time_policy("Nighttime") == FishingTimePolicy.NIGHTTIME
    assert parse_fishing_time_policy("不调") == FishingTimePolicy.DONT_CHANGE
    assert fishing_hours(FishingTimePolicy.ALL) == (7, 19)
    assert fishing_hours(FishingTimePolicy.DAYTIME) == (7,)
    assert fishing_hours(FishingTimePolicy.NIGHTTIME) == (19,)
    assert fishing_hours(FishingTimePolicy.ALL, coop=True) == (None,)
    with pytest.raises(ValueError, match="fishingTimePolicy"):
        parse_fishing_time_policy("黄昏")


def test_auto_fishing_all_policy_runs_day_and_night_rounds():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.tasks.auto_fishing import AutoFishingTask

    release_all = Mock()
    ctx = SimpleNamespace(input=SimpleNamespace(release_all=release_all))
    task = AutoFishingTask(
        ctx,
        target_catches=0,
        fishing_time_policy="全天",
        auto_throw_rod_enabled=False,
        quit_on_finish=False,
        log=lambda _message: None,
    )
    task._set_time = Mock(return_value=True)
    task._run_fishing_round = Mock(side_effect=[(True, 2), (True, 3)])
    task._quit_fishing_mode = Mock()
    ctx.sleep = Mock()

    assert task.run() is True
    assert [call.args for call in task._set_time.call_args_list] == [(7,), (19,)]
    assert task._run_fishing_round.call_count == 2
    task._quit_fishing_mode.assert_called_once_with()
    release_all.assert_called_once_with()


def test_auto_eat_red_bar_detector_scales_from_full_frame():
    import numpy as np

    from bgi_touch.tasks.auto_eat import current_avatar_is_low_hp, red_bar_components

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[950:965, 800:1000] = (0, 0, 255)
    assert current_avatar_is_low_hp(frame)
    assert red_bar_components(frame)[0][2] >= 190

    narrow = np.zeros_like(frame)
    narrow[950:965, 800:840] = (0, 0, 255)
    assert not current_avatar_is_low_hp(narrow)


def test_auto_music_lane_detector_uses_device_points():
    import numpy as np

    from bgi_touch.tasks.auto_music import detect_music_lanes
    from bgi_touch.vision.coordinate import ScreenTransform

    transform = ScreenTransform(2816, 1296)
    points = [transform.to_device(x, 921) for x in (417, 628, 844, 1061, 1277, 1493)]
    frame = np.full((1296, 2816, 3), 255, dtype=np.uint8)
    for index in (0, 2, 5):
        x, y = (round(value) for value in points[index])
        frame[y - 3:y + 4, x - 3:x + 4, 0] = 0
    assert detect_music_lanes(
        frame,
        [round(x) for x, _ in points],
        round(points[0][1]),
    ) == (True, False, True, False, False, True)


def test_auto_album_normalizes_upstream_music_levels():
    from bgi_touch.tasks.auto_album import resolve_album_difficulties

    assert [level.name for level in resolve_album_difficulties(None)] == ["传说"]
    assert [level.name for level in resolve_album_difficulties("all")] == [
        "普通", "困难", "大师", "传说"
    ]
    assert [level.name for level in resolve_album_difficulties(["困难", "normal", "困难"])] == [
        "普通", "困难"
    ]
    with pytest.raises(ValueError, match="难度无效"):
        resolve_album_difficulties("不存在")


def test_auto_album_dispatcher_maps_bettergi_parameters():
    from unittest.mock import patch

    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.auto_album.AutoAlbumTask") as task:
        task.return_value.run.return_value = True
        assert TaskDispatcher(object()).run_task({
            "name": "AutoAlbum",
            "config": {
                "musicLevel": "all",
                "mustCanorusLevel": True,
                "songCount": 2,
                "trackTimeoutSeconds": 30,
            },
        })
    kwargs = task.call_args.kwargs
    assert kwargs["music_level"] == "all"
    assert kwargs["must_canorus_level"] is True
    assert kwargs["song_count"] == 2
    assert kwargs["track_timeout_s"] == 30


def test_redeem_code_normalizer_accepts_bettergi_objects_and_text():
    from bgi_touch.tasks.redeem_code import normalize_redeem_codes

    assert [item.code for item in normalize_redeem_codes("AAA\nBBB,AAA\n")] == [
        "AAA", "BBB"
    ]
    items = normalize_redeem_codes([
        {"Code": "CCC", "Items": "原石"},
        {"code": "DDD", "items": "摩拉"},
        "",
    ])
    assert [(item.code, item.items) for item in items] == [
        ("CCC", "原石"), ("DDD", "摩拉")
    ]


def test_redeem_code_normalizer_strips_announcement_urls():
    from bgi_touch.tasks.redeem_code import normalize_redeem_codes

    codes = normalize_redeem_codes(
        "AAA https://genshin.hoyoverse.com/gift?code=SHOULD_NOT_BE_PARSED\n"
        "BBB,https://example.com/redeem\nCCC"
    )

    assert [item.code for item in codes] == ["AAA", "BBB", "CCC"]


def test_quick_claim_candidates_keep_upstream_top_left_order():
    from types import SimpleNamespace

    from bgi_touch.tasks.quick_claim import QuickClaimRewardTask

    task = QuickClaimRewardTask.__new__(QuickClaimRewardTask)
    text = [SimpleNamespace(dx=500, dy=300), SimpleNamespace(dx=100, dy=100)]
    gifts = [SimpleNamespace(dx=50, dy=300)]
    task._find_multi = lambda region, name, threshold: text if name == "claim_text" else gifts
    candidates = task._find_candidates(object())
    assert [(name, hit.dx, hit.dy) for name, hit in candidates] == [
        ("领取", 100, 100),
        ("礼物领取", 50, 300),
        ("领取", 500, 300),
    ]


def test_new_quick_tasks_are_declared_and_validate_redeem_codes():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    assert TaskDispatcher.IMPLEMENTED >= {
        "QuickSereniteaPot", "QuickClaimReward", "QuickBuy", "UseRedemptionCode"
    }
    with pytest.raises(ValueError, match="至少一个"):
        TaskDispatcher(object()).run_use_redemption_code_task({"codes": []})


def test_quick_buy_uses_touch_slider_for_both_upstream_shop_layouts():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.tasks.quick_buy import QuickBuyTask
    from bgi_touch.vision.coordinate import ScreenTransform

    def context():
        return SimpleNamespace(
            transform=ScreenTransform(2816, 1296),
            device=SimpleNamespace(swipe=Mock()),
            input=SimpleNamespace(click_ref=Mock()),
            sleep=Mock(),
        )

    serenitea_ctx = context()
    assert QuickBuyTask(
        serenitea_ctx, serenitea=True, log=lambda _message: None
    ).run()
    serenitea_swipe = serenitea_ctx.device.swipe.call_args.args
    assert serenitea_swipe[2] > serenitea_swipe[0]
    assert [call.args for call in serenitea_ctx.input.click_ref.call_args_list] == [
        (1600, 1020), (960, 850)
    ]

    generic_ctx = context()
    assert QuickBuyTask(
        generic_ctx, serenitea=False, log=lambda _message: None
    ).run()
    generic_swipe = generic_ctx.device.swipe.call_args.args
    assert generic_swipe[2] > generic_swipe[0]
    assert [call.args for call in generic_ctx.input.click_ref.call_args_list] == [
        (1695, 1020), (1100, 780), (1695, 1020)
    ]


def test_quick_buy_requires_asset_or_explicit_shop_type():
    from types import SimpleNamespace

    from bgi_touch.tasks.quick_buy import QuickBuyTask

    task = QuickBuyTask.__new__(QuickBuyTask)
    task.serenitea = None
    task._coin = None
    task.ctx = SimpleNamespace(
        capture_region=lambda: SimpleNamespace(find_multi=lambda *_args, **_kwargs: [])
    )
    with pytest.raises(FileNotFoundError, match="显式配置"):
        task._is_serenitea_shop()


def test_artifact_stat_parser_preserves_bettergi_contract():
    from bgi_touch.tasks.artifact_salvage import parse_artifact_stat_text

    artifact = parse_artifact_stat_text([
        "异种的期许", "生之花", "生命值", "717", "+0",
        "元素精通+16", "元素充能效率+6.5%", "攻击力+5.8%", "防御力+23",
        "深廊终曲（2）",
    ])
    assert artifact.Name == "异种的期许"
    assert (artifact.MainAffix.Type, artifact.MainAffix.Value) == ("HP", 717)
    assert [(item.Type, item.Value) for item in artifact.MinorAffixes] == [
        ("ElementalMastery", 16),
        ("EnergyRecharge", 6.5),
        ("ATKPercent", 5.8),
        ("DEF", 23),
    ]
    assert artifact.Level == 0


def test_artifact_javascript_uses_output_contract_and_timeout():
    from bgi_touch.tasks.artifact_salvage import (
        ArtifactAffix,
        ArtifactStat,
        evaluate_artifact_javascript,
    )

    artifact = ArtifactStat(
        "test",
        ArtifactAffix("HP", 717),
        (ArtifactAffix("ATKPercent", 5.8), ArtifactAffix("DEF", 23)),
        0,
    )
    rule = """
var hasATK = Array.from(ArtifactStat.MinorAffixes).some(a => a.Type == 'ATKPercent');
var hasDEF = Array.from(ArtifactStat.MinorAffixes).some(a => a.Type == 'DEF');
Output = ArtifactStat.Level == 0 && hasATK && hasDEF;
"""
    assert evaluate_artifact_javascript(artifact, rule)
    with pytest.raises(RuntimeError, match="Output"):
        evaluate_artifact_javascript(artifact, "const answer = true;")
    with pytest.raises(RuntimeError, match="timed out|Script execution timed out"):
        evaluate_artifact_javascript(artifact, "while (true) {}", timeout_ms=20)


def test_js_runtime_exposes_bettergi_param_tokens_file_and_postmessage(tmp_path):
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    import cv2
    import numpy as np

    pytest.importorskip("pythonmonkey")
    from bgi_touch.engine.js_runtime import JsScriptRuntime
    from bgi_touch.vision.coordinate import ScreenTransform

    cv2.imwrite(str(tmp_path / "source.png"), np.full((8, 8, 3), 127, np.uint8))
    (tmp_path / "input.txt").write_text("兼容文本", encoding="utf-8")
    (tmp_path / "main.js").write_text(
        """
const fight = new AutoFightParam("strategy.txt");
fight.Timeout = 321;
fight.FinishDetectConfig.FastCheckEnabled = true;
const domain = new AutoDomainParam(2, "combat.txt");
domain.SetResinPriorityList("原粹树脂", "浓缩树脂");
const leyline = new AutoLeyLineOutcropParam(3, "枫丹", "启示之花");
leyline.FightConfig.Timeout = 188;
leyline.ScanDropsAfterRewardEnabled = true;
const stygian = new AutoStygianOnslaughtParam("route.json");
stygian.RoutePath = "route.json";
stygian.SetCombatStrategyPath("fight.txt");
const boss = new AutoBossParam();
boss.BossName = "急冻树";
boss.RunCount = 4;
const autoSkip = new AutoSkipConfig();
autoSkip.ClickChatOption = "优先选择最后一个选项";
autoSkip.CustomPriorityOptionsEnabled = true;
autoSkip.CustomPriorityOptions = "优先项;备选项";

let callbackCount = 0;
const cts = new CancellationTokenSource();
cts.Token.Register(() => callbackCount++);
cts.Cancel();
const linked = CancellationTokenSource.CreateLinkedTokenSource(cts.Token);

let readCallback = "";
await file.ReadText("input.txt", (error, data) => {
  if (error) throw new Error(error);
  readCallback = data;
});
let writeCallback = false;
await file.WriteText("callback.txt", "ok", (error, success) => {
  if (error) throw new Error(error);
  writeCallback = success;
});
const created = file.CreateDirectory("record/nested");
const mkdirAlias = file.mkdir("alias-dir");
file.WriteTextSync("record/old.txt", "rename me");
const renamed = file.RenamePathSync("record/old.txt", "renamed/new.txt");
const missingRename = file.renamePathSync("missing.txt", "renamed/missing.txt");
const escaped = file.CreateDirectory("../outside-script");
const mat = file.ReadImageMatWithResizeSync("source.png", 4, 5, 0);
const imageRegion = new ImageRegion(file.ReadImageMatSync("source.png"), 3, 4);
const imageFound = imageRegion.Find(
  RecognitionObject.TemplateMatch(file.ReadImageMatSync("source.png"))
).IsExist();
const imageSaved = file.WriteImageSync("written.png", mat);
const recognition = RecognitionObject.TemplateMatch(mat);
recognition.Threshold = 0.61;

const post = new PostMessage();
post.KeyDown("W");
post.KeyUp("W");

return JSON.stringify({
  timeout: fight.timeout,
  fast: fight.finishDetectConfig.fastCheckEnabled,
  rounds: domain.DomainRoundNum,
  resin: domain.resinPriorityList,
  leylineCount: leyline.Count,
  leylineTimeout: leyline.fightConfig.timeout,
  leylineScan: leyline.scanDropsAfterRewardEnabled,
  route: stygian.routePath,
  combat: stygian.CombatScriptBagPath,
  boss: boss.bossName,
  bossRuns: boss.runCount,
  autoSkipLast: autoSkip.isClickFirstChatOption() === false,
  autoSkipPriority: autoSkip.customPriorityOptions,
  cancelled: cts.IsCancellationRequested,
  linkedCancelled: linked.isCancellationRequested,
  callbackCount,
  readCallback,
  writeCallback,
  created,
  mkdirAlias,
  renamed,
  missingRename,
  escaped,
  imageSaved,
  imageSize: [mat.Width, mat.Height],
  imageRegion: [imageRegion.X, imageRegion.Y, imageRegion.Width, imageRegion.Height],
  imageFound,
  recognitionThreshold: recognition.threshold,
  serverOffset: ServerTime.GetServerTimeZoneOffset()
});
""",
        encoding="utf-8",
    )
    input_simulator = SimpleNamespace(
        key_down=Mock(), key_up=Mock(), key_press=Mock(),
        click_ref=Mock(), move_camera_by=Mock(), attack=Mock(),
        attack_down=Mock(), attack_up=Mock(), button_down=Mock(), button_up=Mock(),
    )
    ctx = SimpleNamespace(
        input=input_simulator,
        device=SimpleNamespace(paste_text=Mock()),
        transform=ScreenTransform(1920, 1080),
        sleep=lambda _ms: None,
    )
    result = json.loads(JsScriptRuntime(ctx, tmp_path).run())
    assert result["timeout"] == 321
    assert result["fast"] is True
    assert result["rounds"] == 2
    assert result["resin"] == ["原粹树脂", "浓缩树脂"]
    assert (result["leylineCount"], result["route"], result["combat"]) == (
        3, "route.json", "fight.txt"
    )
    assert result["leylineTimeout"] == 188
    assert result["leylineScan"] is True
    assert (result["boss"], result["bossRuns"]) == ("急冻树", 4)
    assert result["autoSkipLast"] is True
    assert result["autoSkipPriority"] == "优先项;备选项"
    assert result["cancelled"] is True
    assert result["linkedCancelled"] is True
    assert result["callbackCount"] == 1
    assert result["readCallback"] == "兼容文本"
    assert result["writeCallback"] is True
    assert result["created"] is True
    assert result["mkdirAlias"] is True
    assert result["renamed"] is True
    assert result["missingRename"] is False
    assert result["escaped"] is False
    assert result["imageSaved"] is True
    assert result["imageSize"] == [4, 5]
    assert result["imageRegion"] == [3, 4, 8, 8]
    assert result["imageFound"] is True
    assert result["recognitionThreshold"] == pytest.approx(0.61)
    assert isinstance(result["serverOffset"], int)
    assert (tmp_path / "callback.txt").read_text(encoding="utf-8") == "ok"
    assert (tmp_path / "record" / "nested").is_dir()
    assert (tmp_path / "alias-dir").is_dir()
    assert (tmp_path / "renamed" / "new.txt").read_text(encoding="utf-8") == "rename me"
    assert not (tmp_path.parent / "outside-script").exists()
    input_simulator.key_down.assert_called_once_with("W")
    input_simulator.key_up.assert_called_once_with("W")


def test_genshin_teleport_to_statue_alias():
    from unittest.mock import MagicMock

    from bgi_touch.engine.genshin_api import GenshinApi

    api = GenshinApi(MagicMock())
    api.tpToStatueOfTheSeven = MagicMock(return_value=True)

    assert api.teleportToStatue()
    api.tpToStatueOfTheSeven.assert_called_once_with()


def test_dispatcher_maps_bettergi_auto_skip_config():
    from unittest.mock import MagicMock

    from bgi_touch.tasks.dispatcher import TaskDispatcher

    ctx = MagicMock()
    TaskDispatcher(ctx).add_timer({
        "name": "AutoSkip",
        "config": {
            "ClickChatOption": "优先选择最后一个选项",
            "CustomPriorityOptionsEnabled": True,
            "CustomPriorityOptions": "优先项; 备选项\n第三项",
            "QuicklySkipConversationsEnabled": False,
            "SkipBuiltInClickOptions": True,
            "AfterChooseOptionSleepDelay": 120,
            "BeforeClickConfirmDelay": 80,
        },
    })

    ctx.triggers.clear.assert_called_once_with()
    ctx.enable_trigger.assert_called_once_with(
        "AutoSkip",
        click_option="优先选择最后一个选项",
        priority_texts=["优先项", "备选项", "第三项"],
        quickly_skip=False,
        skip_built_in_options=True,
        after_choose_delay_ms=120,
        before_confirm_delay_ms=80,
        close_popup_pages=True,
        auto_re_explore_enabled=True,
        auto_get_daily_rewards_enabled=True,
        auto_wait_dialogue_option_voice_enabled=False,
        dialogue_option_voice_max_wait_seconds=30,
        default_pause_texts=None,
        pause_texts=None,
        select_texts=None,
        auto_hangout_event_enabled=False,
        auto_hangout_end_choose="",
        auto_hangout_choose_option_sleep_delay=0,
        auto_hangout_press_skip_enabled=True,
        hangout_config_path=None,
        submit_goods_enabled=True,
        use_interaction_key=False,
        interaction_key="F",
    )


def test_auto_fight_dispatcher_keeps_upstream_timeout_seconds():
    from unittest.mock import patch

    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.auto_fight.AutoFightTask") as task:
        task.return_value.run.return_value = True
        assert TaskDispatcher(object()).run_auto_fight_task({"timeout": 321})
    assert task.call_args.kwargs["timeout_s"] == 321


def test_artifact_status_detector_matches_upstream_hsv_markers():
    import cv2
    import numpy as np

    from bgi_touch.tasks.artifact_salvage import ArtifactStatus, detect_artifact_status

    def bgr(hue_degrees, saturation):
        hsv = np.uint8([[[round(hue_degrees / 360 * 255), round(saturation * 255), 255]]])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)[0, 0]

    locked = np.zeros((180, 140, 3), dtype=np.uint8)
    locked[2:25, 2:18] = bgr(9, 0.54)
    assert detect_artifact_status(locked) == ArtifactStatus.LOCKED

    selected = np.zeros((180, 140, 3), dtype=np.uint8)
    selected[1:33, 90:135] = bgr(80, 0.76)
    assert detect_artifact_status(selected) == ArtifactStatus.SELECTED


def test_artifact_dispatcher_maps_upstream_and_safety_parameters():
    from unittest.mock import patch

    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.artifact_salvage.AutoArtifactSalvageTask") as task:
        task.return_value.run.return_value = {"ok": True}
        result = TaskDispatcher(object()).run_task({
            "name": "AutoArtifactSalvage",
            "config": {
                "maxArtifactStar": "3",
                "javaScript": "Output = true;",
                "artifactSetFilter": "绝缘之旗印",
                "maxNumToCheck": 12,
                "recognitionFailurePolicy": "Abort",
                "confirmQuickSalvage": True,
                "confirmSalvage": False,
            },
        })
    assert result == {"ok": True}
    kwargs = task.call_args.kwargs
    assert kwargs["star"] == 3
    assert kwargs["max_num_to_check"] == 12
    assert kwargs["recognition_failure_policy"] == "Abort"
    assert kwargs["confirm_quick_salvage"] is True
    assert kwargs["confirm_salvage"] is False


def test_inventory_and_artifact_tasks_use_exclusive_realtime_trigger_scope(tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from bgi_touch.tasks.artifact_salvage import AutoArtifactSalvageTask
    from bgi_touch.tasks.inventory_grid import GetGridIconsTask

    ctx = SimpleNamespace()
    scanner = Mock()
    scanner.category = SimpleNamespace(name="Materials")

    with patch(
        "bgi_touch.tasks.inventory_grid.InventoryGridScanner",
        return_value=scanner,
    ), patch(
        "bgi_touch.tasks.inventory_grid.exclusive_realtime_triggers",
        return_value=nullcontext(),
    ) as inventory_scope:
        task = GetGridIconsTask(ctx, "Materials", output_dir=tmp_path / "grid-icons")
        with patch.object(task, "_run_impl", return_value=[]) as run_impl:
            assert task.run() == []
        inventory_scope.assert_called_once_with(ctx)
        run_impl.assert_called_once_with(None)

    with patch(
        "bgi_touch.tasks.artifact_salvage.exclusive_realtime_triggers",
        return_value=nullcontext(),
    ) as salvage_scope:
        task = AutoArtifactSalvageTask(ctx)
        with patch.object(task, "_run_impl", return_value={"ok": True}) as run_impl:
            assert task.run() == {"ok": True}
        salvage_scope.assert_called_once_with(ctx)
        run_impl.assert_called_once_with(None)


def test_inventory_category_aliases_and_grid_detector():
    import cv2
    import numpy as np

    from bgi_touch.tasks.inventory_grid import (
        detect_artifact_set_filter_cells,
        detect_inventory_cells,
        inventory_category,
    )

    assert inventory_category("GridScreenName.Materials").name == "Materials"
    assert inventory_category("养成道具").name == "CharacterDevelopmentItems"
    with pytest.raises(ValueError, match="不支持"):
        inventory_category("不存在")

    grid = np.zeros((420, 1171, 3), dtype=np.uint8)
    for row, y in enumerate((15, 200)):
        for column, x in enumerate((12, 158, 304)):
            cv2.rectangle(grid, (x, y), (x + 124, y + 152), (220, 220, 220), 3)
    cells = detect_inventory_cells(grid, columns=8)
    assert len(cells) == 6
    assert {(cell.row, cell.column) for cell in cells} == {
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)
    }

    filter_grid = np.zeros((300, 1300, 3), dtype=np.uint8)
    for y in (20, 110):
        cv2.rectangle(filter_grid, (20, y), (610, y + 68), (220, 220, 220), 2)
        cv2.rectangle(filter_grid, (670, y), (1260, y + 68), (220, 220, 220), 2)
    filter_cells = detect_artifact_set_filter_cells(filter_grid)
    assert len(filter_cells) == 4
    assert {(cell.row, cell.column) for cell in filter_cells} == {
        (0, 0), (0, 1), (1, 0), (1, 1)
    }


def test_inventory_count_parser_and_narrow_one_correction():
    import numpy as np
    from unittest.mock import patch

    from bgi_touch.tasks.inventory_grid import parse_inventory_count, recognize_inventory_count

    assert parse_inventory_count("１２３") == 123
    assert parse_inventory_count("12a") is None

    cell = np.full((153, 125, 3), 240, dtype=np.uint8)
    cell[132:149, 60:63] = 80
    with patch("bgi_touch.tasks.inventory_grid._ocr_text", return_value=""):
        result = recognize_inventory_count(cell)
    assert result.count == 1
    assert result.reason == "NARROW_ONE"


def test_count_inventory_dispatcher_preserves_single_and_multi_contract():
    from unittest.mock import patch

    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.inventory_grid.CountInventoryItemTask") as task:
        task.return_value.run.return_value = {"萃凝晶": 42}
        result = TaskDispatcher(object()).run_task({
            "name": "CountInventoryItem",
            "config": {
                "gridScreenName": "Materials",
                "itemNames": ["萃凝晶"],
                "iconRecognitionMode": "Item",
                "maxPages": 3,
            },
        })
    assert result == {"萃凝晶": 42}
    kwargs = task.call_args.kwargs
    assert kwargs["item_names"] == ["萃凝晶"]
    assert kwargs["icon_recognition_mode"] == "Item"
    assert kwargs["max_pages"] == 3


def test_character_development_metadata_and_parsers():
    from bgi_touch.tasks.character_development import (
        CharacterDevelopmentResult,
        apply_talent_result,
        has_talent_bonus,
        load_character_metadata,
        normalize_character_name,
        normalize_talent_type,
        parse_character_categories,
        try_parse_level_pair,
        try_parse_talent_level,
    )

    assert normalize_character_name("岩王爷") == "钟离"
    metadata = load_character_metadata()["钟离"]
    assert metadata.element == "岩"
    assert metadata.weapon_type == "长柄武器"
    assert parse_character_categories(None) == ("属性", "武器", "天赋")
    assert parse_character_categories("weapon;天赋") == ("武器", "天赋")
    with pytest.raises(ValueError, match="未知"):
        parse_character_categories("不存在")
    assert try_parse_level_pair("Lv. 80 / 90") == (80, 90)
    assert try_parse_talent_level("Lv. 13") == 13
    assert normalize_talent_type("元素战技") == "元素战技"
    assert has_talent_bonus("天赋等级 + 3")

    result = CharacterDevelopmentResult("钟离")
    apply_talent_result(result, "元素爆发", 13, True)
    assert (result.BurstLevel, result.BurstHasBonus) == (13, True)


def test_character_card_builder_corrects_noisy_bottom_regions():
    from bgi_touch.tasks.character_development import build_character_cards

    cards = build_character_cards(
        [(15, 120, 100, 20), (139, 119, 100, 21), (14, 262, 101, 19)],
        (641, 897),
        1.0,
    )
    assert len(cards) == 3
    assert [(card.x, card.y) for card in cards] == [(0, 0), (124, 0), (0, 141)]


def test_character_development_dispatcher_maps_single_character():
    from unittest.mock import patch

    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.character_development.CharacterDevelopmentTask") as task:
        task.return_value.run.return_value = [{"CharacterName": "钟离", "Level": 90}]
        result = TaskDispatcher(object()).run_task({
            "name": "CharacterDevelopment",
            "config": {
                "characterName": "钟离",
                "categories": "属性;武器",
                "maxPages": 4,
            },
        })
    assert result == {"CharacterName": "钟离", "Level": 90}
    assert task.call_args.kwargs["max_pages"] == 4
    task.return_value.run.assert_called_once()


def test_reward_card_detector_ports_bettergi_strip_geometry():
    import cv2
    import numpy as np

    from bgi_touch.tasks.reward_result import detect_reward_card_rects

    band = np.zeros((220, 1480, 3), dtype=np.uint8)
    # Low-saturation, V=220 strips match the quantity/name band mask.
    for x in (300, 450, 600):
        cv2.rectangle(band, (x, 155), (x + 100, 184), (220, 220, 220), -1)
    cards = detect_reward_card_rects(band)
    assert len(cards) == 3
    assert [card[0] for card in cards] == [288, 438, 588]
    assert all(card[1:] == (32, 125, 153) for card in cards)


def test_reward_summary_and_duplicate_prefix_match_bettergi_contract():
    from bgi_touch.tasks.reward_result import (
        RewardItem,
        RewardResultRecognizer,
        merge_reward_summary,
    )

    summary = {"摩拉": 1000}
    merge_reward_summary(summary, [RewardItem("摩拉", 500), RewardItem("冒险阅历", -1)])
    assert summary == {"摩拉": 1500, "冒险阅历": 1}
    previous = [RewardItem("摩拉", 1000), RewardItem("好感经验", 20)]
    current = [RewardItem("好感经验", 20), RewardItem("冒险阅历", 100)]
    assert RewardResultRecognizer._duplicate_prefix(current, previous) == 1


def test_auto_domain_dispatcher_maps_reward_recognition_options():
    from unittest.mock import patch

    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.auto_domain.AutoDomainTask") as task:
        task.return_value.run.return_value = {"摩拉": 12000}
        result = TaskDispatcher(object()).run_auto_domain_task({
            "domainRoundNum": 2,
            "rewardRecognitionEnabled": True,
            "rewardMaxPages": 4,
        })
    assert result == {"摩拉": 12000}
    assert task.call_args.kwargs["reward_recognition_enabled"] is True
    assert task.call_args.kwargs["reward_max_pages"] == 4


def test_auto_boss_resin_policy_matches_upstream_contract():
    from bgi_touch.tasks.auto_encounter import (
        BossRunPolicy,
        OriginalResinInfo,
        calculate_current_resin,
        calculate_supplemental_resin_quantity,
        parse_full_recovery_seconds,
        parse_resin_limit,
    )

    fixed = BossRunPolicy(specify_run_count=True, run_count=3)
    assert [fixed.should_continue(count) for count in range(5)] == [
        True, True, True, False, False,
    ]
    assert BossRunPolicy().should_continue(999)
    with pytest.raises(ValueError, match="必须大于"):
        BossRunPolicy(specify_run_count=True, run_count=0)

    assert parse_resin_limit(" 3０ / ２００ ") == 200
    recovery = parse_full_recovery_seconds("全部恢复 22：40：01")
    assert recovery == 22 * 3600 + 40 * 60 + 1
    assert calculate_current_resin(200, recovery) == OriginalResinInfo(29, 200)
    assert calculate_current_resin(200, parse_full_recovery_seconds("原粹树脂已完全恢复")) == OriginalResinInfo(200, 200)
    assert calculate_supplemental_resin_quantity(20, 200) == 3
    assert calculate_supplemental_resin_quantity(20, 200, available=2) == 2
    assert calculate_supplemental_resin_quantity(0, 2000, available=99) == 20


def test_auto_boss_dispatcher_maps_current_bettergi_parameters():
    from unittest.mock import patch

    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.auto_encounter.AutoBossTask") as task:
        task.return_value.run.return_value = {"摩拉": 6000}
        result = TaskDispatcher(object()).run_auto_boss_task({
            "bossName": "急冻树",
            "specifyRunCount": True,
            "runCount": 4,
            "useTransientResin": True,
            "useFragileResin": True,
            "reviveRetryCount": 5,
            "returnToStatueAfterEachRound": True,
            "rewardRecognitionEnabled": True,
            "rewardMaxPages": 4,
            "teamName": "讨伐队",
            "timeout": 321,
        })

    assert result == {"摩拉": 6000}
    kwargs = task.call_args.kwargs
    assert kwargs["route_path"] is None
    assert kwargs["specify_run_count"] is True
    assert kwargs["rounds"] == 4
    assert kwargs["use_transient_resin"] is True
    assert kwargs["use_fragile_resin"] is True
    assert kwargs["revive_retry_count"] == 5
    assert kwargs["return_to_statue_after_each_round"] is True
    assert kwargs["reward_recognition_enabled"] is True
    assert kwargs["reward_max_pages"] == 4
    assert kwargs["team_name"] == "讨伐队"
    assert kwargs["timeout_s"] == 321


def test_auto_boss_official_route_inventory_is_complete():
    from bgi_touch.tasks.auto_encounter import (
        BOSS_ROUTE_ROOT,
        NO_PATHING_SUPPORT_BOSSES,
        SUPPORTED_BOSSES,
        TALK_TO_START_BOSSES,
    )

    assert len(SUPPORTED_BOSSES) == 41
    required = []
    for boss_name in SUPPORTED_BOSSES:
        if boss_name in NO_PATHING_SUPPORT_BOSSES:
            required.extend((
                BOSS_ROUTE_ROOT / f"{boss_name}强制传送.json",
                BOSS_ROUTE_ROOT / f"{boss_name}键鼠前往.json",
            ))
        else:
            required.append(BOSS_ROUTE_ROOT / f"{boss_name}前往.json")
            if boss_name in TALK_TO_START_BOSSES:
                required.append(BOSS_ROUTE_ROOT / f"{boss_name}战斗后快速前往.json")
    assert not [path.name for path in required if not path.is_file()]


def test_auto_leyline_graph_and_official_routes_are_complete():
    from bgi_touch.tasks.auto_leyline import LeyLineRouteGraph
    from bgi_touch.pathing.model import PathingTask

    graph = LeyLineRouteGraph.load()
    blossoms = [node for node in graph.nodes.values() if node.kind == "blossom"]
    assert len(graph.nodes) == 382
    assert len(graph.edges) == 378
    assert len(blossoms) == 269
    missing = []
    for blossom in blossoms:
        nearest = graph.nearest_blossom(
            blossom.x, blossom.y, country=blossom.region[:2], threshold=1,
        )
        assert nearest is not None
        assert (nearest.x, nearest.y) == (blossom.x, blossom.y)
        plan = graph.shortest_plan(blossom)
        assert plan is not None and plan.routes
        for route in plan.routes:
            if not graph.resolve_route(route).is_file():
                missing.append(route)
        target = graph.resolve_route(graph.target_route(plan.routes[-1]))
        if not target.is_file():
            missing.append(str(target))
    assert missing == []
    # Parse every bundled base/target/rerun file, including upstream BOM files.
    route_root = graph.resolve_route("")
    route_files = list(route_root.rglob("*.json"))
    assert len(route_files) == 632
    assert sum(len(PathingTask.load(path).positions) for path in route_files) == 1711


def test_auto_leyline_resin_count_and_reward_priority_match_upstream():
    from bgi_touch.tasks.auto_leyline import (
        LeyLineResinCounts,
        calculate_leyline_run_count,
        choose_leyline_reward_resins,
    )

    count = calculate_leyline_run_count(
        LeyLineResinCounts(original=100, condensed=3, transient=2, fragile=4),
        use_transient=True, use_fragile=False,
    )
    assert (count.total, count.original, count.condensed, count.transient, count.fragile) == (
        8, 3, 3, 2, 0,
    )
    assert choose_leyline_reward_resins(
        ["浓缩树脂", "原粹树脂 20"], use_transient=False, use_fragile=False,
    ) == ["浓缩树脂", "原粹树脂"]
    assert choose_leyline_reward_resins(
        ["双倍产出", "原粹树脂 20"], use_transient=True, use_fragile=True,
    ) == ["原粹树脂"]
    assert choose_leyline_reward_resins(
        ["补充原粹树脂", "须臾树脂", "脆弱树脂"],
        use_transient=True, use_fragile=False,
    ) == ["须臾树脂"]


def test_auto_leyline_dispatcher_maps_current_parameters():
    from unittest.mock import patch

    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.auto_leyline.AutoLeyLineOutcropTask") as task:
        task.return_value.run.return_value = True
        result = TaskDispatcher(object()).run_auto_leyline_task({
            "count": 6,
            "country": "纳塔",
            "leyLineOutcropType": "藏金之花",
            "openModeCountMin": True,
            "isResinExhaustionMode": True,
            "useAdventurerHandbook": True,
            "friendshipTeam": "好感队",
            "team": "战斗队",
            "useFragileResin": True,
            "useTransientResin": True,
            "scanDropsAfterRewardEnabled": True,
            "scanDropsAfterRewardSeconds": 25,
            "fightConfig": {"timeout": 188},
            "oneDragonMode": True,
        })

    assert result is True
    kwargs = task.call_args.kwargs
    assert kwargs["count"] == 6
    assert kwargs["country"] == "纳塔"
    assert kwargs["ley_line_type"] == "藏金之花"
    assert kwargs["open_mode_count_min"] is True
    assert kwargs["resin_exhaustion_mode"] is True
    assert kwargs["use_adventurer_handbook"] is True
    assert kwargs["friendship_team"] == "好感队"
    assert kwargs["team"] == "战斗队"
    assert kwargs["use_fragile_resin"] is True
    assert kwargs["use_transient_resin"] is True
    assert kwargs["scan_drops_after_reward_enabled"] is True
    assert kwargs["scan_drops_after_reward_seconds"] == 25
    assert kwargs["timeout_s"] == 188
    assert kwargs["one_dragon_mode"] is True


def test_one_dragon_parser_supports_ids_duplicates_and_next_task():
    from bgi_touch.tasks.one_dragon import parse_one_dragon_items

    items = parse_one_dragon_items({
        "TaskEnabledList": {"a": True, "b": False, "c": True},
        "TaskOrder": ["c", "a", "b"],
        "TaskDefinitions": {"a": "自动秘境", "b": "领取邮件", "c": "自动秘境"},
        "NextTaskId": "a",
    })
    assert [(item.id, item.name, item.enabled) for item in items] == [
        ("a", "自动秘境", True),
        ("b", "领取邮件", False),
    ]


def test_one_dragon_parser_supports_legacy_name_keys():
    from bgi_touch.tasks.one_dragon import parse_one_dragon_items

    items = parse_one_dragon_items({
        "taskEnabledList": {"领取邮件": True, "自动秘境": False},
    })
    assert [item.name for item in items] == ["领取邮件", "自动秘境"]


def test_one_dragon_parser_coerces_string_boolean_values():
    from bgi_touch.tasks.one_dragon import parse_one_dragon_items

    items = parse_one_dragon_items({
        "taskEnabledList": {"enabled": "true", "disabled": "false", "zero": 0},
        "taskOrder": ["enabled", "disabled", "zero"],
        "taskDefinitions": {
            "enabled": "自动秘境",
            "disabled": "领取邮件",
            "zero": "自动地脉花",
        },
    })
    assert [item.enabled for item in items] == [True, False, False]


def test_one_dragon_builtin_parameters_coerce_string_booleans():
    from types import SimpleNamespace

    from bgi_touch.tasks.one_dragon import OneDragonFlowTask, OneDragonItem

    class Dispatcher:
        def run_auto_boss_task(self, config):
            self.config = config
            return True

    dispatcher = Dispatcher()
    task = OneDragonFlowTask(
        SimpleNamespace(),
        {
            "autoBossSpecifyRunCount": "false",
            "autoBossUseTransientResin": "true",
            "autoBossUseFragileResin": "0",
            "autoBossReturnToStatueAfterEachRound": "yes",
            "autoBossRewardRecognitionEnabled": "false",
        },
        dispatcher,
        continue_on_error="false",
        close_game_on_completion="0",
        log=lambda _message: None,
    )
    assert task.continue_on_error is False
    assert task.close_game_on_completion is False
    task._run_builtin(OneDragonItem("boss", "自动首领讨伐", True))
    assert dispatcher.config["specifyRunCount"] is False
    assert dispatcher.config["useTransientResin"] is True
    assert dispatcher.config["useFragileResin"] is False
    assert dispatcher.config["returnToStatueAfterEachRound"] is True
    assert dispatcher.config["rewardRecognitionEnabled"] is False


def test_one_dragon_runner_dispatches_custom_task_configs_and_continues():
    from types import SimpleNamespace

    from bgi_touch.tasks.one_dragon import OneDragonFlowTask

    calls = []

    class Dispatcher:
        IMPLEMENTED = {"AutoCook", "AutoFishing", "OneDragon"}

        def run_task(self, task):
            calls.append(task)
            return task["name"] != "AutoCook"

    ctx = SimpleNamespace(sleep=lambda _ms: None)
    config = {
        "name": "离线一条龙",
        "taskEnabledList": {"x": True, "y": True},
        "taskOrder": ["x", "y"],
        "taskDefinitions": {"x": "自定义烹饪", "y": "自定义钓鱼"},
        "taskConfigs": {
            "x": {"taskName": "AutoCook", "timeoutSeconds": 10},
            "y": {"taskName": "AutoFishing", "targetCatches": 2},
        },
    }
    result = OneDragonFlowTask(ctx, config, Dispatcher(), log=lambda _: None).run()
    assert [call["name"] for call in calls] == ["AutoCook", "AutoFishing"]
    assert result["completed"] == ["y"]
    assert "x" in result["failed"]


def test_one_dragon_maps_current_auto_boss_config():
    from types import SimpleNamespace

    from bgi_touch.tasks.one_dragon import OneDragonFlowTask, OneDragonItem

    class Dispatcher:
        def run_auto_boss_task(self, config):
            self.config = config
            return {"摩拉": 6000}

    dispatcher = Dispatcher()
    task = OneDragonFlowTask(
        SimpleNamespace(),
        {
            "autoBossName": "急冻树",
            "autoBossStrategyName": "讨伐策略",
            "autoBossTeamName": "讨伐队",
            "autoBossSpecifyRunCount": True,
            "autoBossRunCount": 3,
            "autoBossUseTransientResin": True,
            "autoBossUseFragileResin": True,
            "autoBossReviveRetryCount": 4,
            "autoBossReturnToStatueAfterEachRound": True,
            "autoBossRewardRecognitionEnabled": True,
            "autoBossTimeout": 360,
        },
        dispatcher,
        log=lambda _: None,
    )

    assert task._run_builtin(OneDragonItem("boss", "自动首领讨伐", True)) == {"摩拉": 6000}
    assert dispatcher.config == {
        "bossName": "急冻树",
        "strategyName": "讨伐策略",
        "teamName": "讨伐队",
        "specifyRunCount": True,
        "runCount": 3,
        "useTransientResin": True,
        "useFragileResin": True,
        "reviveRetryCount": 4,
        "returnToStatueAfterEachRound": True,
        "rewardRecognitionEnabled": True,
        "timeout": 360,
    }


def test_one_dragon_maps_daily_leyline_config():
    from datetime import datetime, timedelta
    from types import SimpleNamespace

    from bgi_touch.tasks.one_dragon import OneDragonFlowTask, OneDragonItem

    day = (datetime.now().astimezone() - timedelta(hours=4)).strftime("%A")

    class Dispatcher:
        def run_auto_leyline_task(self, config):
            self.config = config
            return True

    dispatcher = Dispatcher()
    task = OneDragonFlowTask(
        SimpleNamespace(),
        {
            f"leyLineRun{day}": True,
            f"leyLine{day}Type": "藏金之花",
            f"leyLine{day}Country": "纳塔",
            "leyLineRunCount": 5,
            "leyLineResinExhaustionMode": True,
            "leyLineOpenModeCountMin": True,
            "leyLineOneDragonMode": True,
        },
        dispatcher,
        log=lambda _: None,
    )

    assert task._run_builtin(OneDragonItem("leyline", "自动地脉花", True)) is True
    assert dispatcher.config == {
        "count": 5,
        "leyLineOutcropType": "藏金之花",
        "country": "纳塔",
        "isResinExhaustionMode": True,
        "openModeCountMin": True,
        "oneDragonMode": True,
    }


def test_tcg_strategy_parser_matches_bettergi_format():
    from bgi_touch.tasks.auto_tcg import parse_tcg_strategy
    from bgi_touch.tasks.tcg_state import TcgElement

    characters, commands = parse_tcg_strategy(
        """角色定义:
角色1=刻晴|雷{技能1消耗=4雷骰子}
角色2=雷神|雷{技能3消耗=1雷骰子+2任意}
角色3=甘雨|冰{技能2消耗=5冰骰子}
---
策略定义:
刻晴 使用 技能1
雷神 使用 技能3 骰子减少1
甘雨 使用 技能2 骰子增加2
"""
    )
    assert characters[1].name == "刻晴"
    assert characters[1].element == TcgElement.ELECTRO
    assert characters[1].skills[1].specific_element_cost == 4
    assert characters[2].skills[3].any_element_cost == 2
    assert characters[2].skills[3].all_cost == 3
    assert [command.skill for command in commands] == [1, 3, 2]
    assert [command.dice_delta for command in commands] == [0, -1, 2]


def test_tcg_strategy_parser_loads_default_character_metadata(tmp_path):
    import json

    from bgi_touch.tasks.auto_tcg import parse_tcg_strategy
    from bgi_touch.tasks.tcg_state import TcgElement

    cards = [{
        "name": "测试角色",
        "element": "火元素",
        "skills": [
            {
                "name": "普通攻击",
                "skillTag": ["普通攻击"],
                "cost": [
                    {"type": "火元素", "count": 1},
                    {"type": "无色元素", "count": 2},
                ],
            },
            {
                "name": "元素战技",
                "skillTag": ["元素战技"],
                "cost": [{"type": "火元素", "count": 3}],
            },
            {
                "name": "元素爆发",
                "skillTag": ["元素爆发"],
                "cost": [
                    {"type": "火元素", "count": 3},
                    {"type": "充能", "count": 2},
                ],
            },
        ],
    }]
    config = tmp_path / "tcg_character_card.json"
    config.write_text(json.dumps(cards, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "combat_avatar.json").write_text(
        json.dumps([{"name": "测试角色", "alias": ["测试别名"]}], ensure_ascii=False),
        encoding="utf-8",
    )
    characters, commands = parse_tcg_strategy(
        """角色定义:
角色1=测试别名
角色2=测试角色|火{技能1消耗=3火骰子}
角色3=另一个角色|火{技能1消耗=3火骰子}
策略定义:
测试别名 使用 技能1
""",
        card_config_path=config,
    )
    assert characters[1].element == TcgElement.PYRO
    assert characters[1].name == "测试别名"
    assert characters[1].skills[1].name == "元素爆发"
    assert characters[1].skills[3].any_element_cost == 2
    assert commands[0].skill == 1


def test_tcg_round_planning_reroll_tuning_and_defeat_order():
    from bgi_touch.tasks.tcg_state import (
        TcgCharacter,
        TcgCommand,
        TcgElement,
        TcgSkill,
        effective_skill_cost,
        next_living_character,
        reroll_indices,
        tuning_card_count,
        wanted_elements,
    )

    characters = {
        1: TcgCharacter(1, "雷", TcgElement.ELECTRO, {
            1: TcgSkill(1, TcgElement.ELECTRO, 1, 2),
        }),
        2: TcgCharacter(2, "冰", TcgElement.CRYO, {
            1: TcgSkill(1, TcgElement.CRYO, 3),
        }),
        3: TcgCharacter(3, "水", TcgElement.HYDRO, {
            1: TcgSkill(1, TcgElement.HYDRO, 3),
        }),
    }
    commands = [
        TcgCommand("雷", 1, -1),
        TcgCommand("冰", 1),
        TcgCommand("水", 1),
    ]
    assert wanted_elements(commands, characters, current_character=1) == {
        TcgElement.OMNI,
        TcgElement.ELECTRO,
        TcgElement.CRYO,
    }
    dice = [
        TcgElement.OMNI,
        TcgElement.ELECTRO,
        TcgElement.PYRO,
        TcgElement.CRYO,
    ]
    assert reroll_indices(dice, {TcgElement.ELECTRO}) == [2, 3]
    assert tuning_card_count(
        {TcgElement.OMNI: 1, TcgElement.CRYO: 1}, characters[2].skills[1]
    ) == 1
    assert effective_skill_cost(characters[1].skills[1], -1) == 2
    characters[1].defeated = True
    assert next_living_character(commands, characters) is characters[2]


def test_tcg_strategy_rejects_undefined_skill():
    from bgi_touch.tasks.auto_tcg import parse_tcg_strategy

    with pytest.raises(ValueError, match="没有定义技能2"):
        parse_tcg_strategy(
            """角色定义:
角色1=甲|火{技能1消耗=3火骰子}
角色2=乙|水{技能1消耗=3水骰子}
角色3=丙|冰{技能1消耗=3冰骰子}
策略定义:
甲 使用 技能2
"""
        )


@pytest.mark.parametrize(
    ("visible", "dice_count", "expected"),
    [
        ({"duel_end", "round_end"}, 0, "duel_end"),
        ({"taken_out", "round_end"}, 0, "character_taken_out"),
        ({"round_end", "opponent"}, 0, "my_action"),
        ({"opponent"}, 0, "opponent_action"),
        ({"end_phase"}, 0, "end_phase"),
        ({"confirm"}, 8, "roll"),
        ({"confirm"}, 6, "prepare"),
        (set(), 0, "unknown"),
    ],
)
def test_tcg_phase_transition_priority(visible, dice_count, expected):
    import numpy as np

    from bgi_touch.tasks.auto_tcg import TcgRecognizer

    recognizer = object.__new__(TcgRecognizer)
    recognizer.find = lambda _frame, key: key if key in visible else None
    recognizer.roll_dice = lambda _frame: [(None, None)] * dice_count
    assert recognizer.phase(np.zeros((1, 1, 3), dtype=np.uint8)).value == expected


def test_task_dispatcher_declares_migrated_core_tasks():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    assert TaskDispatcher.IMPLEMENTED >= {
        "AutoFight", "AutoWood", "AutoDomain", "AutoCook", "AutoFishing", "AutoOpenChest",
        "AutoBoss", "AutoLeyLine", "AutoLeyLineOutcrop",
        "AutoEat", "AutoMusicGame", "AutoGeniusInvokation", "AutoStygianOnslaught",
        "AutoAlbum", "CheckRewards",
    }
    with pytest.raises(NotImplementedError, match="尚未移植"):
        TaskDispatcher(object()).run_task({"name": "UnknownSoloTask", "config": {}})


def test_new_task_dispatcher_validates_upstream_parameter_contracts():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    dispatcher = TaskDispatcher(object())
    with pytest.raises(ValueError, match="不能同时指定"):
        dispatcher.run_auto_eat_task({"foodName": "甜甜花酿鸡", "foodEffectType": 1})
    with pytest.raises(ValueError, match="需要 strategy"):
        dispatcher.run_auto_genius_invokation_task({})
    from unittest.mock import patch

    with patch("bgi_touch.tasks.auto_stygian.AutoStygianOnslaughtTask") as task:
        task.return_value.run.return_value = True
        assert dispatcher.run_auto_stygian_onslaught_task({})
    assert task.call_args.kwargs["route_path"] is None


def test_genshin_map_local_match_overload_keeps_world_hint_separate_from_cache():
    from unittest.mock import Mock

    from bgi_touch.engine.genshin_api import GenshinApi

    api = GenshinApi.__new__(GenshinApi)
    positioner = Mock()
    positioner.get_position_stable.return_value = (1.0, 2.0)
    api._positioner_for = Mock(return_value=positioner)
    api.ctx = Mock()
    api.ctx.capture_bgr.return_value = object()
    result = api.getPositionFromMap("Teyvat", 4328.0, 3960.0)
    assert (result.x, result.y) == (1.0, 2.0)
    positioner.set_prior.assert_called_once_with(4328.0, 3960.0)
    positioner.get_position_stable.assert_called_once_with(
        api.ctx.capture_bgr.return_value, cache_time_ms=900
    )


def test_time_dial_gesture_is_a_real_circular_drag():
    from bgi_touch.engine.genshin_api import GenshinApi

    taps, (start, end) = GenshinApi._time_dial_gestures(6, 30)
    assert len(taps) == 3
    assert start != end
    assert start[0] > 1000
    assert end[1] != start[1]


def test_pathing_model_preserves_bettergi_extensions():
    from bgi_touch.pathing.model import PathingTask

    task = PathingTask.parse({
        "info": {
            "name": "兼容路线",
            "mapName": "Teyvat",
            "mapMatchMethod": "SIFT",
        },
        "config": {"realtimeTriggers": {"AutoPick": False, "AutoSkip": True}},
        "positions": [{
            "id": 7,
            "x": 10,
            "y": 20,
            "type": "target",
            "moveMode": "run",
            "action": "log_output",
            "actionParams": "arrived",
            "pointExtParams": {
                "monsterTag": "elite",
                "enableMonsterLootSplit": True,
                "misidentification": {
                    "type": ["unrecognized", "pathTooFar"],
                    "handlingMode": "previousDetectedPoint",
                    "arrivalTime": 1200,
                },
            },
        }],
    })

    task.validate()
    assert task.map_name == "Teyvat"
    assert task.realtime_triggers == {"AutoPick": False, "AutoSkip": True}
    point = task.positions[0]
    assert point.move_mode == "run"
    assert point.monster_tag == "elite"
    assert point.enable_monster_loot_split
    assert point.misidentification.types == ["unrecognized", "pathTooFar"]
    assert point.misidentification.arrival_time == 1200


def test_minimap_stable_position_rejects_jump_after_global_retry():
    from bgi_touch.pathing.positioner import MinimapPositioner

    positioner = object.__new__(MinimapPositioner)
    positioner._last_position = (0.0, 0.0)
    positioner._last_fix_at = 0.0

    class Locator:
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    positioner.locator = Locator()
    positions = iter([(1000.0, 1000.0), (1000.0, 1000.0)])
    positioner.get_position = lambda frame: next(positions)

    assert positioner.get_position_stable(object(), cache_time_ms=0, max_jump=100) is None
    assert positioner.locator.reset_count == 1


def test_pathing_executor_reaches_waypoint_and_runs_action():
    from unittest.mock import MagicMock

    from bgi_touch.pathing.executor import PathingExecutor
    from bgi_touch.pathing.model import PathingTask

    ctx = MagicMock()
    ctx.input = MagicMock()
    ctx.sleep = lambda ms: None
    positioner = MagicMock()
    positioner.get_position_stable.return_value = (10.0, 20.0)
    task = PathingTask.parse({
        "info": {"name": "offline", "map_name": "Teyvat"},
        "config": {"realtime_triggers": {}},
        "positions": [{
            "id": 1, "x": 10, "y": 20, "type": "target",
            "action": "log_output", "action_params": "done",
        }],
    })
    logs = []

    assert PathingExecutor(ctx, positioner=positioner, log=logs.append).run(task)
    assert any("done" in line for line in logs)
    ctx.input.release_all.assert_called_once()


def test_pathing_executor_stops_when_action_is_not_supported():
    from types import SimpleNamespace

    from bgi_touch.pathing.executor import PathingExecutor
    from bgi_touch.pathing.model import Waypoint

    executor = PathingExecutor.__new__(PathingExecutor)
    executor.actions = SimpleNamespace(run=lambda _waypoint: False)
    waypoint = Waypoint(
        id=7, x=10, y=20, type="target", move_mode="walk",
        action="future_action",
    )

    with pytest.raises(RuntimeError, match="future_action 未支持"):
        executor._do_action(waypoint)


def test_pathing_executor_handles_four_leaf_before_movement():
    from unittest.mock import MagicMock

    from bgi_touch.pathing.executor import PathingExecutor
    from bgi_touch.pathing.model import PathingTask

    ctx = MagicMock()
    ctx.input = MagicMock()
    ctx.sleep = lambda ms: None
    positioner = MagicMock()
    executor = PathingExecutor(ctx, positioner=positioner, log=lambda _: None)
    executor._face_to = MagicMock()
    executor._do_action = MagicMock()
    executor._move_to = MagicMock()
    task = PathingTask.parse({
        "info": {"name": "leaf", "map_name": "Teyvat"},
        "config": {"realtime_triggers": {}},
        "positions": [{
            "id": 1, "x": 10, "y": 20, "type": "path",
            "action": "up_down_grab_leaf",
        }],
    })

    assert executor.run(task)
    executor._face_to.assert_called_once_with(task.positions[0])
    executor._do_action.assert_called_once_with(task.positions[0])
    executor._move_to.assert_not_called()


def test_pathing_executor_enables_all_migrated_realtime_triggers():
    from unittest.mock import MagicMock

    from bgi_touch.pathing.executor import PathingExecutor
    from bgi_touch.pathing.model import PathingTask

    ctx = MagicMock()
    ctx.input = MagicMock()
    ctx.sleep = lambda _ms: None
    ctx._trigger_loop = None
    task = PathingTask.parse({
        "info": {"name": "triggers", "map_name": "Teyvat"},
        "config": {"realtimeTriggers": {
            "AutoPick": True,
            "AutoSkip": True,
            "AutoEat": True,
            "MapMask": True,
            "SkillCd": True,
            "GameLoading": True,
            "QuickTeleport": True,
            "AutoFishing": True,
        }},
        "positions": [],
    })

    executor = PathingExecutor(
        ctx,
        party_slots={"钟离": 1},
        log=lambda _message: None,
    )
    enabled, previous = executor._enable_realtime_triggers(task)

    assert previous is None
    assert enabled == [
        "AutoPick", "AutoSkip", "AutoEat", "MapMask", "SkillCd",
        "GameLoading", "QuickTeleport", "AutoFish",
    ]
    assert ctx.enable_trigger.call_args_list[3].kwargs == {
        "map_name": "Teyvat", "mini_map_enabled": True,
    }
    assert ctx.enable_trigger.call_args_list[4].kwargs == {
        "party_slots": {"钟离": 1},
    }


# ---- 地图定位（需要资产与夹具，缺则跳过）----

@pytest.mark.skipif(not (ASSETS / "Teyvat" / "Teyvat_0_2048_SIFT.kp.bin").exists(),
                    reason="地图资产未下载")
def test_minimap_localization_mondstadt():
    import cv2
    from bgi_touch.pathing.map_locator import MapLocator
    mm = cv2.imread(str(FIXTURES / "minimap-mondstadt.png"), cv2.IMREAD_GRAYSCALE)
    assert mm is not None
    loc = MapLocator()
    p = loc.locate_pixel(mm)
    assert p is not None
    wx, wy = loc.config.image_to_world(*p)
    # 录制时角色在蒙德城（约 -910, 2250）
    assert wx == pytest.approx(-910, abs=15)
    assert wy == pytest.approx(2250, abs=15)
