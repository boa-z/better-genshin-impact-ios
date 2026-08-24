from pathlib import Path


def test_upstream_tp_asset_indexes_world_coordinates_and_transfer_coordinates():
    from bgi_touch.pathing.teleport_points import TeleportPointStore

    path = Path(__file__).resolve().parents[1] / "assets" / "data" / "tp.json"
    store = TeleportPointStore(path)

    # GiTpPosition exposes Position[2] as X and Position[0] as Y.
    point = store.nearest_point("Teyvat", -874.22, 1993.73)

    assert point is not None
    assert point.point_type == "TeleportWaypoint"
    assert point.country == "蒙德"
    assert point.x == -874.22
    assert point.y == 1993.73
    assert point.transfer_x == -867.23413
    assert point.transfer_y == 1992.874


def test_tp_asset_contains_independent_map_scenes():
    from bgi_touch.pathing.teleport_points import TeleportPointStore

    path = Path(__file__).resolve().parents[1] / "assets" / "data" / "tp.json"
    store = TeleportPointStore(path)

    assert len(store.scenes["TheChasm"]) == 15
    assert store.nearest_point("TheChasm", 0, 0) is not None


def test_tp_target_resolution_matches_force_contract():
    from types import SimpleNamespace

    from bgi_touch.pathing.teleport_points import TeleportPoint
    from bgi_touch.pathing.tp import TpTask

    point = TeleportPoint(
        map_name="Teyvat",
        point_id="1",
        point_type="TeleportWaypoint",
        name="传送锚点",
        country="蒙德",
        areas=(),
        x=100,
        y=200,
        transfer_x=101,
        transfer_y=201,
    )
    task = TpTask.__new__(TpTask)
    task.map_name = "Teyvat"
    task.log = lambda *_args: None
    task._teleport_points = SimpleNamespace(nearest_point=lambda *_args: point)

    assert task._resolve_tp_target(120, 230, force=False) == (100, 200, "蒙德", point)
    assert task._resolve_tp_target(120, 230, force=True) == (120.0, 230.0, None, None)
