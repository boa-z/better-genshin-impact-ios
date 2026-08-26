from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.mark.parametrize("name", [
    "Teyvat", "TheChasm", "Enkanomiya", "SeaOfBygoneEras",
    "AncientSacredMountain", "TempleOfSpace", "MoonCanon",
])
def test_map_configs_round_trip_world_coordinates(name):
    from bgi_touch.pathing.map_locator import MapConfig, get_map_definition

    config = MapConfig.for_map(name)
    definition = get_map_definition(name)
    world = (123.25, -456.75)
    image = config.world_to_image(*world)

    assert config.image_to_world(*image) == pytest.approx(world)
    assert definition.feature_scale in (1024, 2048)


@pytest.mark.parametrize("alias, expected", [
    ("提瓦特大陆", "Teyvat"),
    ("层岩巨渊·地下矿区", "TheChasm"),
    ("渊下宫", "Enkanomiya"),
    ("旧日之海", "SeaOfBygoneEras"),
    ("远古圣山", "AncientSacredMountain"),
    ("空之神殿", "TempleOfSpace"),
    ("霜月", "MoonCanon"),
])
def test_map_name_aliases(alias, expected):
    from bgi_touch.pathing.map_locator import resolve_map_name

    assert resolve_map_name(alias) == expected


def test_map_locator_discovers_independent_map_layers(tmp_path, monkeypatch):
    import bgi_touch.pathing.map_locator as module

    base = tmp_path / "SeaOfBygoneEras"
    base.mkdir()
    for floor in ("0", "-1", "-2"):
        (base / f"SeaOfBygoneEras_{floor}_1024_SIFT.kp.bin").touch()
        (base / f"SeaOfBygoneEras_{floor}_1024_SIFT.mat.png").touch()

    stores = []

    class _Store:
        def __init__(self, keypoints, descriptors):
            self.keypoints = Path(keypoints)
            self.descriptors = Path(descriptors)
            stores.append(self)

    monkeypatch.setattr(module, "ASSETS", tmp_path)
    monkeypatch.setattr(module, "SiftFeatureStore", _Store)

    locator = module.MapLocator("旧日之海")

    assert locator.map_name == "SeaOfBygoneEras"
    assert len(locator.layers) == 3
    assert locator.layer_ids == [0, -1, -2]
    assert stores[0].keypoints.name == "SeaOfBygoneEras_0_1024_SIFT.kp.bin"
    assert locator.config.world_to_image(0, 0) == (6144, 3072)


def test_big_map_locator_uses_each_maps_native_feature_scale(tmp_path, monkeypatch):
    import bgi_touch.pathing.tp as module

    base = tmp_path / "AncientSacredMountain"
    base.mkdir()
    keypoints = base / "AncientSacredMountain_0_1024_SIFT.kp.bin"
    descriptors = base / "AncientSacredMountain_0_1024_SIFT.mat.png"
    keypoints.touch()
    descriptors.touch()
    monkeypatch.setattr(module, "ASSETS", tmp_path)
    monkeypatch.setattr(
        module, "SiftFeatureStore",
        lambda kp, mat: SimpleNamespace(keypoints=Path(kp), descriptors=Path(mat)),
    )

    locator = module.BigMapLocator("远古圣山")
    feature = locator.world_to_feature(100, -50)

    assert feature == pytest.approx((1948, 2098))
    assert locator.feature_to_world(*feature) == pytest.approx((100, -50))


def test_big_map_locator_rejects_degenerate_affine_scale():
    import bgi_touch.pathing.tp as module
    from bgi_touch.pathing.feature_store import MatchResult

    class _Sift:
        def detectAndCompute(self, _image, _mask):
            keypoints = [SimpleNamespace(pt=(float(i), float(i))) for i in range(10)]
            return keypoints, np.ones((10, 128), dtype=np.float32)

    locator = module.BigMapLocator.__new__(module.BigMapLocator)
    locator.definition = SimpleNamespace(big_map_query_resize=1.0)
    locator._sift = _Sift()
    locator.stores = [SimpleNamespace(locate=Mock(return_value=MatchResult(
        100.0, 200.0, 8, 0.001,
    )))]

    assert locator.locate_view(np.zeros((120, 200, 3), dtype=np.uint8)) is None


def test_tp_task_uses_safe_area_aware_map_close_roi(monkeypatch):
    import bgi_touch.pathing.tp as module

    monkeypatch.setattr(module, "BigMapLocator", lambda _name: SimpleNamespace())
    task = module.TpTask(SimpleNamespace(), map_name="Teyvat")

    assert task._map_close.roi == (1600, 0, 320, 140)


def test_tp_task_recovers_stale_device_channel_and_orientation():
    import bgi_touch.pathing.tp as module

    logs = []
    device = SimpleNamespace(reconnect_device=Mock())
    ctx = SimpleNamespace(device=device, sleep=Mock(), refresh_orientation=Mock())
    task = module.TpTask.__new__(module.TpTask)
    task.ctx = ctx
    task.log = logs.append

    assert task._recover_device_channel("地图视野未变化") is True
    device.reconnect_device.assert_called_once_with()
    ctx.sleep.assert_called_once_with(2000)
    ctx.refresh_orientation.assert_called_once_with()
    assert logs == ["[tp] 地图视野未变化，重建设备输入通道后重试"]


def test_pathing_executor_replaces_only_map_aware_positioners(monkeypatch):
    import bgi_touch.pathing.positioner as positioner_module
    from bgi_touch.pathing.executor import PathingExecutor

    created = []

    class _Positioner:
        def __init__(self, _ctx, map_name):
            self.map_name = map_name
            created.append(map_name)

    monkeypatch.setattr(positioner_module, "MinimapPositioner", _Positioner)
    ctx = SimpleNamespace(input=SimpleNamespace(), sleep=lambda _ms: None)
    executor = PathingExecutor(ctx, positioner=SimpleNamespace(map_name="Teyvat"))
    executor._tp_task = object()

    executor._ensure_positioner("Enkanomiya")

    assert created == ["Enkanomiya"]
    assert executor.positioner.map_name == "Enkanomiya"
    assert executor._tp_task is None

    custom = SimpleNamespace()
    executor.positioner = custom
    executor._ensure_positioner("TheChasm")
    assert executor.positioner is custom
    assert created == ["Enkanomiya"]
    assert executor._map_name == "TheChasm"


def test_switch_area_clicks_selector_and_lowest_matching_ocr_candidate():
    import bgi_touch.pathing.tp as module

    upper = SimpleNamespace(text="层岩巨渊", y=220, click=Mock())
    lower = SimpleNamespace(text="层岩巨渊·地下矿区", y=760, click=Mock())
    region = SimpleNamespace(find_multi=Mock(return_value=[upper, lower]))
    input_controller = SimpleNamespace(click_ref=Mock())
    ctx = SimpleNamespace(
        input=input_controller,
        sleep=Mock(),
        capture_region=Mock(return_value=region),
    )
    logs = []
    task = module.TpTask.__new__(module.TpTask)
    task.ctx = ctx
    task.log = logs.append
    task.map_name = "TheChasm"

    assert task._switch_area() is True
    input_controller.click_ref.assert_called_once_with(1760, 1020)
    upper.click.assert_not_called()
    lower.click.assert_called_once_with()
    assert logs == ["[tp] 切换地图区域：层岩巨渊·地下矿区"]


def test_wait_for_target_map_does_not_accept_the_previous_teyvat_country():
    import bgi_touch.pathing.tp as module

    ctx = SimpleNamespace(
        capture_bgr=Mock(return_value=object()),
        sleep=Mock(),
    )
    task = module.TpTask.__new__(module.TpTask)
    task.ctx = ctx
    task.map_name = "Teyvat"
    task.big = SimpleNamespace(locate_view=Mock(return_value=(1.0, 2.0, 3.0)))
    task.log = Mock()
    task._accept_target_view = Mock(side_effect=[False, True])

    assert task._wait_for_target_map(timeout_s=1.0, target_area="Fontaine")
    assert [item.args for item in task._accept_target_view.call_args_list] == [
        ((1.0, 2.0, 3.0), "Fontaine"),
        ((1.0, 2.0, 3.0), "Fontaine"),
    ]
