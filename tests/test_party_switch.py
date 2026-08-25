from types import SimpleNamespace
from unittest.mock import Mock, call


class _Hit:
    def __init__(self, text="", exists=True):
        self.text = text
        self._exists = exists
        self.clicks = 0

    def is_exist(self):
        return self._exists

    def click(self):
        self.clicks += 1


def test_party_switcher_uses_ocr_list_and_two_confirmations():
    import bgi_touch.engine.party as module

    title = _Hit("队伍配置")
    current = _Hit("日常队")
    target = _Hit("采集队")
    confirm_hits = [_Hit(), _Hit()]

    class Region:
        def find_multi(self, ro, *, limit):
            if ro.roi == module.PARTY_TITLE_ROI:
                return [title]
            if ro.roi == module.PARTY_NAME_ROI:
                return [current]
            if ro.roi == module.PARTY_LIST_ROI:
                return [target]
            return []

        def find(self, _ro):
            return confirm_hits.pop(0)

    input_controller = SimpleNamespace(
        key_press=Mock(),
        click_ref=Mock(),
        vertical_scroll=Mock(),
    )
    ctx = SimpleNamespace(
        input=input_controller,
        capture_region=Mock(return_value=Region()),
        sleep=Mock(),
    )
    return_main = Mock(return_value=True)
    switcher = module.PartySwitcher(ctx, log=Mock(), return_main_ui=return_main)

    assert switcher.switch("采集队")

    input_controller.key_press.assert_called_once_with("L")
    input_controller.click_ref.assert_called_once_with(*module.PARTY_SELECTOR_POINT)
    assert target.clicks == 1
    assert len(confirm_hits) == 0
    assert return_main.call_count == 2


def test_party_switcher_scrolls_until_target_is_found():
    import bgi_touch.engine.party as module

    calls = {"list": 0}

    class Region:
        def find_multi(self, ro, *, limit):
            if ro.roi == module.PARTY_TITLE_ROI:
                return [_Hit("队伍配置")]
            if ro.roi == module.PARTY_NAME_ROI:
                return [_Hit("当前队伍")]
            if ro.roi == module.PARTY_LIST_ROI:
                calls["list"] += 1
                return [_Hit("其他队伍")] if calls["list"] == 1 else [_Hit("目标队伍")]
            return []

        def find(self, _ro):
            return _Hit()

    input_controller = SimpleNamespace(
        key_press=Mock(), click_ref=Mock(), vertical_scroll=Mock(),
    )
    ctx = SimpleNamespace(
        input=input_controller,
        capture_region=Mock(return_value=Region()),
        sleep=Mock(),
    )
    switcher = module.PartySwitcher(ctx, log=Mock(), return_main_ui=Mock(return_value=True))

    assert switcher.switch("目标队伍")
    input_controller.vertical_scroll.assert_called_once_with(-3)


def test_coop_player_slot_mapping_matches_bettergi_semantics():
    from bgi_touch.engine.party import physical_slots_for_players

    assert physical_slots_for_players(1, 1) == (1, 2, 3, 4)
    assert physical_slots_for_players(2, 1) == (1, 2)
    assert physical_slots_for_players(2, 2) == (3, 4)
    assert physical_slots_for_players(3, 1) == (1, 2)
    assert physical_slots_for_players(3, 2) == (3,)
    assert physical_slots_for_players(3, 3) == (4,)
    assert physical_slots_for_players(4, 4) == (4,)


def test_coop_character_arguments_map_to_current_players_physical_slots():
    from bgi_touch.engine.party import build_character_assignments

    assert build_character_assignments(
        ["甲", "乙", "", ""],
        use_physical_slots=False,
        physical_slots=(3, 4),
    ) == [(3, "甲"), (4, "乙")]
    assert build_character_assignments(
        ["", "乙", "", "甲"],
        use_physical_slots=True,
        physical_slots=(3, 4),
    ) == [(4, "甲")]


def test_party_control_layout_prefers_explicit_headless_context_values():
    from bgi_touch.engine.party import detect_party_control_layout

    ctx = SimpleNamespace(coop_player_count=2, coop_player_index=2)
    layout = detect_party_control_layout(ctx)

    assert layout.player_count == 2
    assert layout.player_index == 2
    assert layout.physical_slots == (3, 4)


def test_party_control_layout_uses_upstream_markers_without_extra_capture(monkeypatch):
    import bgi_touch.engine.party as module

    marker_hits = {"2P_top_left.png"}
    monkeypatch.setattr(
        module,
        "_template_exists",
        lambda _region, filename, _roi: filename in marker_hits,
    )
    region = SimpleNamespace(
        find_multi=Mock(return_value=[object(), object()]),
    )
    ctx = SimpleNamespace()

    layout = module.detect_party_control_layout(ctx, region)

    assert layout == module.PartyControlLayout(3, 2, (3,))
    region.find_multi.assert_called_once()


def test_default_layout_exposes_devicehub_party_setup_key():
    from bgi_touch.input.layout import ControlLayout

    layout = ControlLayout.load()
    assert layout.binding("L") == {
        "type": "button",
        "button": "partySetup",
        "profileCode": "KeyL",
    }
    assert layout.buttons["partySetup"] == (0.065, 0.499)


def test_character_switcher_selects_slot_by_ocr_and_confirms_change():
    import bgi_touch.engine.party as module

    title = _Hit("队伍配置")
    character = _Hit("胡桃")
    confirm = _Hit("更换")

    class Region:
        def find_multi(self, ro, *, limit):
            if ro.roi == module.PARTY_TITLE_ROI:
                return [title]
            if ro.roi == module.CHARACTER_GRID_ROI:
                return [character]
            if ro.roi == module.CHARACTER_CONFIRM_ROI:
                return [confirm]
            return []

    input_controller = SimpleNamespace(key_press=Mock(), click_ref=Mock())
    ctx = SimpleNamespace(
        input=input_controller,
        capture_region=Mock(return_value=Region()),
        sleep=Mock(),
    )
    return_main = Mock(return_value=True)

    assert module.CharacterSwitcher(
        ctx, log=Mock(), return_main_ui=return_main,
    ).switch_characters(["胡桃", "", "", ""])

    input_controller.key_press.assert_called_once_with("L")
    assert input_controller.click_ref.call_args.args == module.CHARACTER_SLOT_POINTS[0]
    assert character.clicks == 1
    assert confirm.clicks == 1
    assert return_main.call_count == 2


def test_character_switcher_uses_coop_physical_slots_for_logical_arguments():
    import bgi_touch.engine.party as module

    title = _Hit("队伍配置")
    first = _Hit("胡桃")
    second = _Hit("夜兰")
    confirm = _Hit("更换")

    class Region:
        def find_multi(self, ro, *, limit):
            if ro.roi == module.PARTY_TITLE_ROI:
                return [title]
            if ro.roi == module.CHARACTER_GRID_ROI:
                return [first if first.clicks == 0 else second]
            if ro.roi == module.CHARACTER_CONFIRM_ROI:
                return [confirm]
            return []

    input_controller = SimpleNamespace(key_press=Mock(), click_ref=Mock())
    ctx = SimpleNamespace(
        input=input_controller,
        capture_region=Mock(return_value=Region()),
        sleep=Mock(),
        coop_player_count=2,
        coop_player_index=2,
    )
    return_main = Mock(return_value=True)

    switcher = module.CharacterSwitcher(
        ctx, log=Mock(), return_main_ui=return_main,
    )
    assert switcher.switch_characters(
        ["胡桃", "夜兰", "", ""], use_physical_slots=False,
    )

    assert switcher.last_assignments == [(3, "胡桃"), (4, "夜兰")]
    assert input_controller.click_ref.call_args_list[:2] == [
        call(*module.CHARACTER_SLOT_POINTS[2]),
        call(*module.CHARACTER_SLOT_POINTS[3]),
    ]


def test_genshin_switch_character_updates_runtime_party_mapping(monkeypatch):
    from bgi_touch.engine.genshin_api import GenshinApi

    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace()
    api.log = Mock()
    api._party_slots = {"旧角色": 1}
    fake_switcher = Mock()
    fake_switcher.switch_characters.return_value = True
    monkeypatch.setattr(
        "bgi_touch.engine.party.CharacterSwitcher",
        Mock(return_value=fake_switcher),
    )
    api.returnMainUi = Mock(return_value=True)

    assert api.switchCharacter("胡桃", "夜兰", "", "", False)

    fake_switcher.switch_characters.assert_called_once_with(
        ["胡桃", "夜兰", "", ""], use_physical_slots=False,
    )
    assert api.ctx.party_slots["胡桃"] == 1
    assert api.ctx.party_slots["夜兰"] == 2


def test_genshin_switch_character_decodes_string_physical_slot_flag(monkeypatch):
    from bgi_touch.engine.genshin_api import GenshinApi

    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace()
    api.log = Mock()
    api._party_slots = {}
    fake_switcher = Mock()
    fake_switcher.switch_characters.return_value = True
    fake_switcher.last_assignments = [(1, "胡桃")]
    monkeypatch.setattr(
        "bgi_touch.engine.party.CharacterSwitcher",
        Mock(return_value=fake_switcher),
    )
    api.returnMainUi = Mock(return_value=True)

    assert api.switchCharacter("胡桃", "", "", "", "false")

    fake_switcher.switch_characters.assert_called_once_with(
        ["胡桃", "", "", ""], use_physical_slots=False,
    )
