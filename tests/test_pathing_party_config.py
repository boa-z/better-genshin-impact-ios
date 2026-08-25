from bgi_touch.pathing.party_config import PathingPartyConfig


def test_pathing_party_config_accepts_bettergi_camel_case_and_clamps_related_values():
    config = PathingPartyConfig.from_mapping({
        "PathingConfig": {
            "Enabled": True,
            "Distance": 40,
            "ApproachStopDistance": 90,
            "HurryOnAvatar": "夜兰",
            "HurryOnFrameInterval": 999,
            "MwkJumpFlyDistance": 20,
            "MwkJumpFlyIntervalSeconds": 0,
            "MwkDisableSprintEnabled": "true",
        }
    })

    assert config.distance == 40
    assert config.approach_stop_distance == 40
    assert config.hurry_on_avatar == "夜兰"
    assert config.hurry_on_frame_interval == 150
    assert config.effective_hurry_frame_interval_ms == 150
    assert config.mwk_jump_fly_distance == 41
    assert config.mwk_jump_fly_interval_seconds == 0.1
    assert config.mwk_disable_sprint_enabled is True


def test_pathing_party_config_keeps_unknown_hurry_selection_disabled():
    config = PathingPartyConfig.from_mapping({"HurryOnAvatar": "不存在的角色"})

    assert config.hurry_on_avatar == ""
    assert config.resolve_hurry_avatar({"夜兰": 1}) is None


def test_auto_hurry_prefers_main_avatar_then_upstream_priority():
    config = PathingPartyConfig.from_mapping({
        "HurryOnAvatar": "自动",
        "MainAvatarIndex": "2",
    })
    party = {"钟离": 1, "夜兰": 2, "玛薇卡": 3}

    assert config.resolve_hurry_avatar(party) == "夜兰"

    fallback = PathingPartyConfig.from_mapping({"HurryOnAvatar": "自动"})
    assert fallback.resolve_hurry_avatar(party) == "玛薇卡"


def test_walk_avatar_prefers_main_slot_and_excludes_hurry_blacklist():
    config = PathingPartyConfig.from_mapping({"MainAvatarIndex": "3"})
    party = {"玛薇卡": 1, "希诺宁": 2, "钟离": 3, "夜兰": 4}

    assert config.resolve_walk_avatar(party, "玛薇卡") == "钟离"
