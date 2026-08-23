from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
