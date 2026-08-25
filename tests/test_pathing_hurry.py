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


def test_hurry_never_applies_to_walk_segments():
    config = PathingPartyConfig.from_mapping({"HurryOnAvatar": "法尔伽"})
    controller = HurryOnController(config, {"法尔伽": 1})
    controller.start()

    action = controller.tick(distance=200, move_mode="walk", now=0.0)

    assert action.handled is False
    assert action.press_skill is False
