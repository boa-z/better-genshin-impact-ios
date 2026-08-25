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


def test_pathing_party_config_parses_nested_auto_eat_defaults():
    config = PathingPartyConfig.from_mapping({
        "PathingConfig": {
            "AutoEatConfig": {
                "DefaultAtkBoostingDishName": "攻击料理",
                "DefaultAdventurersDishName": "冒险料理",
                "DefaultDefBoostingDishName": "防御料理",
            },
        },
    })

    assert config.auto_eat_config.default_atk_boosting_dish_name == "攻击料理"
    assert config.auto_eat_config.default_adventurers_dish_name == "冒险料理"
    assert config.auto_eat_config.default_def_boosting_dish_name == "防御料理"


def test_pathing_party_config_uses_upstream_default_attack_dish():
    config = PathingPartyConfig.from_mapping({"PathingConfig": {}})

    assert config.auto_eat_config.default_atk_boosting_dish_name == "炸萝卜丸子"


def test_pathing_party_config_parses_recovery_timing_and_gadget_interval():
    config = PathingPartyConfig.from_mapping({
        "PathingConfig": {
            "OnlyInTeleportRecover": True,
            "UseGadgetIntervalMs": "2500",
        },
    })
    never = PathingPartyConfig.from_mapping({
        "RecoverTiming": "Never",
        "UseGadgetIntervalMs": -1,
    })

    assert config.recover_timing == "OnlyTeleport"
    assert config.use_gadget_interval_ms == 2500
    assert never.recover_timing == "Never"
    assert never.use_gadget_interval_ms == 0
