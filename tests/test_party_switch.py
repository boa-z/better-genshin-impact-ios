from types import SimpleNamespace
from unittest.mock import Mock


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


def test_default_layout_exposes_devicehub_party_setup_key():
    from bgi_touch.input.layout import ControlLayout

    layout = ControlLayout.load()
    assert layout.binding("L") == {
        "type": "button",
        "button": "partySetup",
        "profileCode": "KeyL",
    }
    assert layout.buttons["partySetup"] == (0.065, 0.499)
