from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch


def _executor(config):
    from bgi_touch.pathing.executor import PathingExecutor

    ctx = SimpleNamespace(input=Mock(), sleep=lambda _ms: None)
    return PathingExecutor(ctx, pathing_config=config, log=lambda _message: None)


def test_route_party_switch_retries_from_statue_after_in_place_failure():
    executor = _executor({
        "PartyName": "采集队",
        "IsVisitStatueBeforeSwitchParty": False,
    })
    fake_tp = Mock()
    fake_tp.exclusive_triggers.return_value = nullcontext()
    fake_switcher = Mock()
    fake_switcher.switch.side_effect = [False, True]
    fake_api = Mock()
    fake_api.returnMainUi.return_value = True

    with patch("bgi_touch.pathing.tp.TpTask", return_value=fake_tp), \
            patch("bgi_touch.engine.party.PartySwitcher", return_value=fake_switcher), \
            patch("bgi_touch.engine.genshin_api.GenshinApi", return_value=fake_api):
        assert executor._switch_party_for_route() is True

    fake_tp.tp_to_statue.assert_called_once_with()
    assert fake_switcher.switch.call_count == 2


def test_route_party_switch_can_force_statue_before_first_attempt():
    executor = _executor({
        "PartyName": "战斗队",
        "IsVisitStatueBeforeSwitchParty": True,
    })
    fake_tp = Mock()
    fake_tp.exclusive_triggers.return_value = nullcontext()
    fake_switcher = Mock()
    fake_switcher.switch.return_value = True
    fake_api = Mock()
    fake_api.returnMainUi.return_value = True

    with patch("bgi_touch.pathing.tp.TpTask", return_value=fake_tp), \
            patch("bgi_touch.engine.party.PartySwitcher", return_value=fake_switcher), \
            patch("bgi_touch.engine.genshin_api.GenshinApi", return_value=fake_api):
        assert executor._switch_party_for_route() is True

    fake_tp.tp_to_statue.assert_called_once_with()
    fake_switcher.switch.assert_called_once_with("战斗队")
