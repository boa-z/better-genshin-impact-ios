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

    class FakeDevice:
        def __init__(self):
            self.inputs = []
            self.stopped = []

        def start_game_session(self, profile_name, **kwargs):
            assert profile_name == "test-profile"
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
    assert ("session-1", ["Space", "KeyX"]) in device.inputs
    assert device.inputs[-2:] == [
        ("session-1", ["Space", "KeyX"]),
        ("session-1", ["Space"]),
    ]
    simulator.release_all()
    assert device.stopped == ["session-1"]


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
        "QuickSereniteaPot", "QuickClaimReward", "UseRedemptionCode"
    }
    with pytest.raises(ValueError, match="至少一个"):
        TaskDispatcher(object()).run_use_redemption_code_task({"codes": []})


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


def test_tcg_strategy_parser_matches_bettergi_format():
    from bgi_touch.tasks.auto_tcg import parse_tcg_strategy

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
    assert [command.skill for command in commands] == [1, 3, 2]
    assert [command.dice_delta for command in commands] == [0, -1, 2]


def test_task_dispatcher_declares_migrated_core_tasks():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    assert TaskDispatcher.IMPLEMENTED >= {
        "AutoFight", "AutoWood", "AutoDomain", "AutoCook", "AutoFishing", "AutoOpenChest",
        "AutoBoss", "AutoLeyLine", "AutoLeyLineOutcrop",
        "AutoEat", "AutoMusicGame", "AutoGeniusInvokation", "AutoStygianOnslaught",
        "AutoAlbum",
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
    with pytest.raises(FileNotFoundError, match="routePath"):
        dispatcher.run_auto_stygian_onslaught_task({})


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
