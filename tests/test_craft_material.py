from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import pytest


def _ctx():
    return SimpleNamespace(
        input=SimpleNamespace(
            click_ref=Mock(),
            drag_ref=Mock(),
        ),
        sleep=Mock(),
        transform=SimpleNamespace(
            device_width=1920,
            device_height=1080,
            to_device=lambda x, y, anchor="auto": (x, y),
            to_ref=lambda x, y, anchor="auto": (x, y),
        ),
        device=SimpleNamespace(
            tap=Mock(),
            swipe=Mock(),
        ),
        capture_bgr=Mock(return_value=np.zeros((1080, 1920, 3), dtype=np.uint8)),
    )


def test_material_type_is_loaded_from_item_v2_csv():
    from bgi_touch.tasks.craft_material import load_material_types

    types = load_material_types()

    assert types["浓缩树脂"] == "消耗品"
    assert types["甜甜花酿鸡"] == "食物"


def test_craft_quantity_uses_slider_then_corrects_and_verifies():
    from bgi_touch.tasks.craft_material import CraftMaterialTask

    ctx = _ctx()
    task = CraftMaterialTask(
        ctx,
        "浓缩树脂",
        4,
        material_type="消耗品",
        log=Mock(),
    )
    task._read_max_quantity = Mock(return_value=10)
    task._read_current_quantity = Mock(side_effect=[3, 4])
    task._drag_slider = Mock()

    assert task._set_quantity(10**12, None) == 4

    task._drag_slider.assert_has_calls([call(0.0), call(3 / 9)])
    ctx.input.click_ref.assert_called_once_with(1614, 672)


def test_craft_rejects_quantity_larger_than_available():
    from bgi_touch.tasks.craft_material import CraftMaterialTask

    task = CraftMaterialTask(
        _ctx(),
        "浓缩树脂",
        6,
        material_type="消耗品",
        log=Mock(),
    )
    task._read_max_quantity = Mock(return_value=5)

    with pytest.raises(RuntimeError, match="材料不足"):
        task._set_quantity(10**12, None)


def test_find_and_select_material_confirms_item_v2_match_in_detail_panel():
    from bgi_touch.tasks.craft_material import CraftMaterialTask
    from bgi_touch.tasks.inventory_grid import GridCell

    ctx = _ctx()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    selected_frame = np.zeros_like(frame)
    ctx.capture_bgr.side_effect = [frame, selected_frame]
    recognizer = Mock()
    recognizer.match.return_value = SimpleNamespace(name="浓缩树脂", score=0.91)
    cell = GridCell(50, 180, 120, 145)
    task = CraftMaterialTask(
        ctx,
        "浓缩树脂",
        1,
        material_type="消耗品",
        recognizer=recognizer,
        log=Mock(),
    )
    task._grid_bounds = Mock(return_value=(45, 170, 705, 790))
    task._scan_cells = Mock(return_value=[cell])
    task._detail_matches = Mock(return_value=True)
    task._fingerprint = Mock(return_value=np.zeros((16, 24), dtype=np.uint8))

    assert task._find_and_select_material(10**12, None)

    recognizer.match.assert_called_once()
    ctx.device.tap.assert_called_once_with(
        110.0,
        252.5,
        image_width=1920,
        image_height=1080,
    )
    task._detail_matches.assert_called_once_with(selected_frame)


def test_craft_run_reports_structured_result_and_pauses_triggers():
    from bgi_touch.tasks.craft_material import CraftMaterialTask

    ctx = _ctx()
    task = CraftMaterialTask(
        ctx,
        "浓缩树脂",
        2,
        material_type="消耗品",
        log=Mock(),
    )
    task._wait_for_crafting_ui = Mock(return_value=True)
    task._select_material_type = Mock(return_value=True)
    task._find_and_select_material = Mock(return_value=True)
    task._set_quantity = Mock(return_value=2)
    task._submit = Mock(return_value=[{"name": "浓缩树脂", "quantity": 2}])
    events = []

    class Scope:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_args):
            events.append("exit")

    with patch(
        "bgi_touch.tasks.craft_material.exclusive_realtime_triggers",
        return_value=Scope(),
    ) as isolated:
        result = task.run()

    assert result["success"] is True
    assert result["materialName"] == "浓缩树脂"
    assert result["materialType"] == "消耗品"
    assert result["actualQuantity"] == 2
    assert result["crafted"] == 2
    assert result["rewards"] == [{"name": "浓缩树脂", "quantity": 2}]
    assert events == ["enter", "exit"]
    isolated.assert_called_once_with(ctx)


def test_dispatcher_passes_cancellation_to_craft_api_only_when_present():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    api = Mock()
    api.craftMaterial.return_value = {"success": True, "crafted": 1}
    dispatcher = TaskDispatcher(object())
    dispatcher._genshin_api = Mock(return_value=api)

    token = SimpleNamespace(cancelled=False)
    result = dispatcher.run_task(
        {
            "name": "CraftMaterialTask",
            "config": {
                "materialName": "浓缩树脂",
                "quantity": 1,
                "materialType": "消耗品",
            },
        },
        token,
    )

    assert result["success"] is True
    api.craftMaterial.assert_called_once()
    assert api.craftMaterial.call_args.kwargs["cancelled"] is not None
