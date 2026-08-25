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


def test_tp_asset_can_select_nearest_statue_without_visible_icon():
    from bgi_touch.pathing.teleport_points import TeleportPointStore

    path = Path(__file__).resolve().parents[1] / "assets" / "data" / "tp.json"
    store = TeleportPointStore(path)
    statues = store.nearest_of_type("Teyvat", -874.22, 1993.73, {"Goddess"})

    assert statues
    assert statues[0].point_type == "Goddess"


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


def test_tp_json_point_types_map_to_quick_teleport_icon_types():
    from bgi_touch.pathing.tp import TpTask

    assert TpTask._map_icon_types_for_point("TeleportWaypoint") == {"TeleportWaypoint"}
    assert TpTask._map_icon_types_for_point("Goddess") == {"StatueOfTheSeven"}
    assert TpTask._map_icon_types_for_point("BlessDomain") == {"Domain"}
    assert TpTask._map_icon_types_for_point("NatlanObsidianTotemPole") == {"ObsidianTotemPole"}


def test_panel_candidate_prefers_target_type_and_area_name():
    from bgi_touch.pathing.teleport_points import TeleportPoint
    from bgi_touch.pathing.tp import TpTask, _TeleportPanelCandidate

    target = TeleportPoint(
        map_name="Teyvat",
        point_id="1",
        point_type="TeleportWaypoint",
        name="传送锚点",
        country="枫丹",
        areas=("优兰尼娅湖",),
        x=0,
        y=0,
        transfer_x=0,
        transfer_y=0,
    )
    first = _TeleportPanelCandidate(
        index=1,
        icon_types={"StatueOfTheSeven"},
        text="七天神像",
        icon_hit=None,
        text_hit=None,
        row_y=100,
    )
    second = _TeleportPanelCandidate(
        index=2,
        icon_types={"TeleportWaypoint"},
        text="优兰尼娅湖",
        icon_hit=None,
        text_hit=None,
        row_y=140,
    )

    assert TpTask._choose_panel_candidate([first, second], target) is second


def test_panel_candidate_keeps_top_row_when_tp_name_is_generic():
    from bgi_touch.pathing.teleport_points import TeleportPoint
    from bgi_touch.pathing.tp import TpTask, _TeleportPanelCandidate

    target = TeleportPoint(
        map_name="Teyvat",
        point_id="1",
        point_type="TeleportWaypoint",
        name="传送锚点",
        country="枫丹",
        areas=(),
        x=0,
        y=0,
        transfer_x=0,
        transfer_y=0,
    )
    top = _TeleportPanelCandidate(1, {"TeleportWaypoint"}, "优兰尼", None, None, 100)
    lower = _TeleportPanelCandidate(2, {"TeleportWaypoint"}, "优兰尼娅湖", None, None, 140)

    assert TpTask._choose_panel_candidate([lower, top], target) is top


def test_named_panel_candidate_rejects_combined_long_ocr_label():
    from types import SimpleNamespace

    from bgi_touch.pathing.teleport_points import TeleportPoint
    from bgi_touch.pathing.tp import TpTask

    target = TeleportPoint(
        map_name="Teyvat",
        point_id="1",
        point_type="TeleportWaypoint",
        name="优兰尼娅湖",
        country="枫丹",
        areas=(),
        x=0,
        y=0,
        transfer_x=0,
        transfer_y=0,
    )
    combined = SimpleNamespace(text="传送锚点·优兰尼娅湖", y=100)
    exact = SimpleNamespace(text="优兰尼娅湖", y=140)
    region = SimpleNamespace(find_multi=lambda *_args, **_kwargs: [combined, exact])
    task = TpTask.__new__(TpTask)

    candidate = task._find_target_text_candidate(region, target)

    assert candidate is not None
    assert candidate.text == "优兰尼娅湖"


def test_panel_candidate_click_uses_wide_mobile_row_hit_area():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bgi_touch.pathing.tp import _TeleportPanelCandidate
    from bgi_touch.vision.coordinate import ScreenTransform

    device = SimpleNamespace(tap=Mock())
    ctx = SimpleNamespace(
        device=device,
        transform=ScreenTransform(1920, 1080),
    )
    icon = SimpleNamespace(ctx=ctx, dx=1350, dy=300, dw=36, dh=36)
    text = SimpleNamespace(ctx=ctx, dx=1386, dy=304, dw=70, dh=24)
    candidate = _TeleportPanelCandidate(1, {"TeleportWaypoint"}, "优兰尼娅湖", icon, text, 300)

    candidate.click()

    device.tap.assert_called_once_with(
        1496.0,
        318.0,
        image_width=1920,
        image_height=1080,
    )


def test_panel_candidate_retries_a_tap_before_reporting_failure(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import bgi_touch.pathing.tp as module
    from bgi_touch.vision.coordinate import ScreenTransform

    candidate_text = SimpleNamespace(text="优兰尼娅湖", click=Mock())
    candidate = module._TeleportPanelCandidate(
        1, {"TeleportWaypoint"}, "优兰尼娅湖", None, candidate_text, 300,
    )

    class Region:
        def __init__(self, confirmed=False):
            self.confirmed = confirmed

        def find(self, _recognition):
            return SimpleNamespace(
                is_exist=lambda: self.confirmed,
                click=Mock(),
            )

        def find_multi(self, *_args, **_kwargs):
            return []

    regions = [Region(), Region(), Region(confirmed=True)]
    task = module.TpTask.__new__(module.TpTask)
    task.ctx = SimpleNamespace(
        capture_region=Mock(side_effect=regions),
        sleep=Mock(),
        transform=ScreenTransform(1920, 1080),
    )
    task.log = Mock()
    task._go_teleport = object()
    task._find_panel_candidates = Mock(side_effect=[[candidate], [candidate]])
    task._find_target_text_candidate = Mock(return_value=None)
    # Advance the monotonic clock enough for the bounded retry, without
    # making this offline test sleep through the mobile settle delays.
    ticks = [0.0]

    def monotonic():
        value = ticks[0]
        ticks[0] += 0.45
        return value

    monkeypatch.setattr(module.time, "monotonic", monotonic)

    assert task._find_and_tap_confirm(timeout_s=5, initial_delay_ms=0)
    assert candidate_text.click.call_count == 2


def test_absolute_map_icon_alignment_recovers_small_layer_translation():
    from pytest import approx

    from bgi_touch.pathing.tp import (
        TpTask,
        _ExpectedMapIcon,
        _ObservedMapIcon,
    )

    expected = [
        _ExpectedMapIcon(100, 100, frozenset({"TeleportWaypoint"})),
        _ExpectedMapIcon(200, 160, frozenset({"TeleportWaypoint"})),
        _ExpectedMapIcon(320, 240, frozenset({"Domain"})),
    ]
    observed = [
        _ObservedMapIcon(108, 96, {"TeleportWaypoint"}),
        _ObservedMapIcon(208, 156, {"TeleportWaypoint"}),
        _ObservedMapIcon(328, 236, {"Domain"}),
    ]

    alignment = TpTask._estimate_map_icon_alignment(expected, observed)

    assert alignment.offset_x == approx(8)
    assert alignment.offset_y == approx(-4)
    assert alignment.pair_count == 3
    assert alignment.mean_error == approx(0)


def test_absolute_map_icon_alignment_rejects_single_far_match():
    from bgi_touch.pathing.tp import (
        TpTask,
        _ExpectedMapIcon,
        _ObservedMapIcon,
    )

    expected = [_ExpectedMapIcon(100, 100, frozenset({"TeleportWaypoint"}))]
    observed = [_ObservedMapIcon(112, 100, {"TeleportWaypoint"})]

    alignment = TpTask._estimate_map_icon_alignment(expected, observed)

    # A lone 12px offset is outside the upstream single-icon trust radius.
    assert alignment.offset_x == 0
    assert alignment.offset_y == 0


def test_absolute_map_click_point_applies_safe_translation():
    from types import SimpleNamespace

    from bgi_touch.pathing.tp import (
        TpTask,
        _ExpectedMapIcon,
        _ObservedMapIcon,
    )
    from bgi_touch.vision.coordinate import ScreenTransform

    expected = [
        _ExpectedMapIcon(1000, 500, frozenset({"TeleportWaypoint"})),
        _ExpectedMapIcon(1100, 500, frozenset({"TeleportWaypoint"})),
        _ExpectedMapIcon(1200, 500, frozenset({"Domain"})),
    ]
    observed = [
        _ObservedMapIcon(1008, 496, {"TeleportWaypoint"}),
        _ObservedMapIcon(1108, 496, {"TeleportWaypoint"}),
        _ObservedMapIcon(1208, 496, {"Domain"}),
    ]
    task = TpTask.__new__(TpTask)
    task.ctx = SimpleNamespace(transform=ScreenTransform(1920, 1080))
    task.log = lambda _message: None
    task._expected_visible_map_icons = lambda _view: expected
    task._observed_visible_map_icons = lambda _region, _types: observed

    corrected = task._absolute_map_click_point(
        object(), (0, 0, 1), 1000, 500,
    )

    assert corrected == (1008, 496)
