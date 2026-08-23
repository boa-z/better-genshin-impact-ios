from unittest.mock import MagicMock, patch

import pytest

from bgi_touch.pathing.actions import PathingActionRunner
from bgi_touch.pathing.model import Waypoint


def _waypoint(action: str, params: str = "") -> Waypoint:
    return Waypoint(
        id=1,
        x=0,
        y=0,
        type="target",
        move_mode="walk",
        action=action,
        action_params=params,
    )


def _runner() -> PathingActionRunner:
    ctx = MagicMock()
    ctx.sleep = MagicMock()
    return PathingActionRunner(ctx, log=MagicMock())


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ("7:00", (7, 0, True)),
        ("07:05", (7, 5, True)),
        ("19:30:false", (19, 30, False)),
        ("24:00:TRUE", (24, 0, True)),
    ],
)
def test_set_time_action_maps_bettergi_parameters(params, expected):
    api = MagicMock()
    api.setTime.return_value = True

    with patch("bgi_touch.engine.genshin_api.GenshinApi", return_value=api):
        assert _runner().run(_waypoint("set_time", params))

    api.setTime.assert_called_once_with(*expected)


@pytest.mark.parametrize(
    "params",
    ["", "7", "07:0", "07:60", "25:00", "07:00:yes", "07:00:true:extra"],
)
def test_set_time_action_rejects_invalid_parameters(params):
    with pytest.raises(ValueError, match="set_time"):
        _runner().run(_waypoint("set_time", params))


def test_set_time_action_reports_ui_failure():
    api = MagicMock()
    api.setTime.return_value = False

    with patch("bgi_touch.engine.genshin_api.GenshinApi", return_value=api):
        with pytest.raises(RuntimeError, match="set_time 失败"):
            _runner().run(_waypoint("set_time", "07:00"))


def test_wonderland_cycle_action_calls_genshin_api():
    api = MagicMock()
    api.wonderlandCycle.return_value = True

    with patch("bgi_touch.engine.genshin_api.GenshinApi", return_value=api):
        assert _runner().run(_waypoint("wonderland_cycle"))

    api.wonderlandCycle.assert_called_once_with()


def test_wonderland_cycle_action_reports_ui_failure():
    api = MagicMock()
    api.wonderlandCycle.return_value = False

    with patch("bgi_touch.engine.genshin_api.GenshinApi", return_value=api):
        with pytest.raises(RuntimeError, match="wonderland_cycle 失败"):
            _runner().run(_waypoint("wonderland_cycle"))
