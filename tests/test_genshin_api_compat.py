from types import SimpleNamespace
from unittest.mock import MagicMock, Mock


def _api_with_task(task):
    from bgi_touch.engine.genshin_api import GenshinApi

    api = GenshinApi.__new__(GenshinApi)
    api.ctx = SimpleNamespace()
    api.log = Mock()
    api._tp_for = Mock(return_value=task)
    api._positioner_for = Mock()
    return api


def test_genshin_tp_forwards_force_to_touch_task():
    task = MagicMock()
    task.tp.return_value = True
    api = _api_with_task(task)

    assert api.tp(4328, 3960, "Teyvat", True)

    task.tp.assert_called_once_with(4328.0, 3960.0, force=True)
    api._positioner_for.return_value.set_prior.assert_called_once_with(4328.0, 3960.0)


def test_genshin_map_helpers_forward_force_country():
    task = MagicMock()
    api = _api_with_task(task)

    api.moveMapTo(1, 2, forceCountry="Mondstadt")
    api.clickMapPoint(3, 4, "璃月")
    api.moveIndependentMapTo(5, 6, "层岩巨渊", "璃月")

    task.move_map_to.assert_called_once_with(1.0, 2.0, force_country="Mondstadt")
    task.click_map_point.assert_called_once_with(3.0, 4.0, force_country="璃月")
    task.move_independent_map_to.assert_called_once_with(
        5.0, 6.0, "层岩巨渊", force_country="璃月"
    )


def test_country_coordinate_helpers_match_upstream_centers():
    from bgi_touch.pathing.map_locator import (
        nearest_teyvat_country,
        resolve_country_name,
    )

    assert resolve_country_name("Mondstadt") == "蒙德"
    assert resolve_country_name("Nod-Krai") == "挪德卡莱"
    assert nearest_teyvat_country(-876, 2278) == "蒙德"
    assert nearest_teyvat_country(4515, 3631) == "枫丹"


def test_area_switch_normalizes_force_country_alias():
    import bgi_touch.pathing.tp as module

    hit = SimpleNamespace(text="蒙德", y=720, click=Mock())
    region = SimpleNamespace(find_multi=Mock(return_value=[hit]))
    ctx = SimpleNamespace(
        input=SimpleNamespace(click_ref=Mock()),
        sleep=Mock(),
        capture_region=Mock(return_value=region),
    )
    task = module.TpTask.__new__(module.TpTask)
    task.ctx = ctx
    task.log = Mock()
    task.map_name = "Teyvat"

    assert task._switch_area("Mondstadt", timeout_s=0.05)

    hit.click.assert_called_once_with()
    assert task._selected_area == "蒙德"
    assert any("蒙德" in call.args[0] for call in task.log.call_args_list)
