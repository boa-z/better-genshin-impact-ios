from bgi_touch.pathing.hurry_on import HurryOnController
from bgi_touch.pathing.party_config import PathingPartyConfig


def test_yelan_hurry_cast_is_cooldown_paced_without_extra_frame_requests():
    config = PathingPartyConfig.from_mapping({
        "HurryOnAvatar": "夜兰",
        "HurryOnFrameInterval": 100,
        "Distance": 45,
    })
    controller = HurryOnController(config, {"夜兰": 2})

    assert controller.enabled is True
    assert controller.start() == "夜兰"
    first = controller.tick(distance=120, move_mode="run", now=0.0)
    throttled = controller.tick(distance=120, move_mode="run", now=0.05)
    ready = controller.tick(distance=120, move_mode="run", now=0.2)
    after_cd = controller.tick(distance=120, move_mode="run", now=10.3)

    assert first.press_skill is True
    assert throttled.press_skill is False
    assert ready.press_skill is False
    assert after_cd.press_skill is True


def test_mavuika_jump_and_safe_dismount_decisions_are_deterministic():
    config = PathingPartyConfig.from_mapping({
        "HurryOnAvatar": "玛薇卡",
        "SwitchToWalkEnabled": True,
        "Distance": 45,
        "ApproachStopDistance": 20,
        "MwkJumpFlyDistance": 70,
        "MwkJumpFlyIntervalSeconds": 1,
    })
    controller = HurryOnController(
        config,
        {"玛薇卡": 1, "钟离": 2},
    )
    controller.start()

    far = controller.tick(distance=100, move_mode="run", now=0.0)
    near = controller.tick(distance=15, move_mode="run", now=1.1)

    assert far.press_skill is True
    assert far.press_jump is True
    assert near.switch_to_walk is True
    assert controller.walk_avatar == "钟离"


def test_mavuika_sprint_jump_count_is_limited_per_hurry_segment():
    config = PathingPartyConfig.from_mapping({
        "HurryOnAvatar": "玛薇卡",
        "Distance": 45,
        "MwkJumpFlyDistance": 70,
        "MwkJumpFlyIntervalSeconds": 1,
        "MwkJumpFlySprintCount": 2,
    })
    controller = HurryOnController(config, {"玛薇卡": 1})
    controller.start()

    first = controller.tick(distance=100, move_mode="run", now=0.0)
    second = controller.tick(distance=100, move_mode="run", now=1.1)
    third = controller.tick(distance=100, move_mode="run", now=2.2)

    assert first.sprint_jump is True
    assert second.sprint_jump is True
    assert third.press_jump is True
    assert third.sprint_jump is False


def test_mavuika_sprint_jump_is_disabled_with_sprint_suppression():
    config = PathingPartyConfig.from_mapping({
        "HurryOnAvatar": "玛薇卡",
        "MwkJumpFlyDistance": 70,
        "MwkJumpFlySprintCount": 2,
        "MwkDisableSprintEnabled": True,
    })
    controller = HurryOnController(config, {"玛薇卡": 1})
    action = controller.tick(distance=100, move_mode="run", now=0.0)

    assert action.press_jump is True
    assert action.sprint_jump is False


def test_hurry_never_applies_to_walk_segments():
    config = PathingPartyConfig.from_mapping({"HurryOnAvatar": "法尔伽"})
    controller = HurryOnController(config, {"法尔伽": 1})
    controller.start()

    action = controller.tick(distance=200, move_mode="walk", now=0.0)

    assert action.handled is False
    assert action.press_skill is False


def test_continuous_hurry_keeps_running_through_a_plain_run_waypoint():
    config = PathingPartyConfig.from_mapping({
        "HurryOnAvatar": "流浪者",
        "SwitchToWalkEnabled": True,
        "TravelMode": "连续赶路",
        "Distance": 45,
        "ApproachStopDistance": 25,
    })
    controller = HurryOnController(config, {"流浪者": 1, "钟离": 2})
    controller.start()

    action = controller.tick(
        distance=20,
        move_mode="run",
        next_distance=80,
        next_type="path",
        next_move_mode="run",
        now=0.0,
    )

    assert action.switch_to_walk is False


def test_continuous_hurry_approaches_before_non_running_boundary():
    config = PathingPartyConfig.from_mapping({
        "HurryOnAvatar": "流浪者",
        "SwitchToWalkEnabled": True,
        "TravelMode": "连续赶路",
        "Distance": 45,
        "ApproachStopDistance": 25,
    })
    controller = HurryOnController(config, {"流浪者": 1, "钟离": 2})
    controller.start()

    action = controller.tick(
        distance=20,
        move_mode="run",
        next_distance=80,
        next_type="path",
        next_move_mode="walk",
        now=0.0,
    )

    assert action.switch_to_walk is True


def test_continuous_hurry_uses_character_turn_threshold():
    config = PathingPartyConfig.from_mapping({
        "HurryOnAvatar": "流浪者",
        "SwitchToWalkEnabled": True,
        "TravelMode": "连续赶路",
        "Distance": 45,
        "ApproachStopDistance": 25,
    })
    controller = HurryOnController(config, {"流浪者": 1, "钟离": 2})
    controller.start()

    action = controller.tick(
        distance=20,
        move_mode="run",
        next_distance=80,
        next_type="path",
        next_move_mode="run",
        turn_angle=45,
        now=0.0,
    )

    assert action.switch_to_walk is True


def test_flight_hurry_uses_shared_motion_status_to_hold_sprint():
    config = PathingPartyConfig.from_mapping({
        "HurryOnAvatar": "恰斯卡",
        "Distance": 45,
    })
    controller = HurryOnController(config, {"恰斯卡": 1})
    controller.start()

    started = controller.tick(
        distance=120,
        move_mode="run",
        motion_status="normal",
        now=0.0,
    )
    flying = controller.tick(
        distance=120,
        move_mode="run",
        motion_status="fly",
        now=0.2,
    )
    near_flying = controller.tick(
        distance=40,
        move_mode="run",
        motion_status="fly",
        now=0.4,
    )

    assert started.press_skill is True
    assert flying.hold_sprint is True
    assert flying.suppress_sprint is True
    assert near_flying.hold_sprint is True


def test_flight_hurry_stops_with_elemental_skill_at_approach_boundary():
    config = PathingPartyConfig.from_mapping({
        "HurryOnAvatar": "流浪者",
        "Distance": 45,
        "ApproachStopDistance": 25,
    })
    controller = HurryOnController(config, {"流浪者": 1})
    controller.start()

    controller.tick(
        distance=120,
        move_mode="run",
        motion_status="fly",
        now=0.0,
    )
    action = controller.tick(
        distance=20,
        move_mode="run",
        motion_status="fly",
        now=0.2,
    )

    assert action.stop_flying is True
    assert action.press_skill is False
