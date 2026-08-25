from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


class _FakeScanner:
    category = SimpleNamespace(name="Food")

    def __init__(self, frame, cells):
        self.frame = frame
        self.cells_to_scan = cells
        self.opened = False
        self.closed = False
        self.tapped = []

    def open(self):
        self.opened = True
        return True

    def pages(self, cancelled=None):
        if cancelled and cancelled():
            return
        yield 1, self.frame, self.cells_to_scan

    def tap(self, cell):
        self.tapped.append(cell)

    def close(self):
        self.closed = True
        return True


def _task_context():
    return SimpleNamespace(
        input=SimpleNamespace(
            key_press=Mock(),
            release_all=Mock(),
        ),
        sleep=Mock(),
    )


def _grid_fixture():
    from bgi_touch.tasks.inventory_grid import GridCell

    frame = np.zeros((153, 125, 3), dtype=np.uint8)
    return frame, [GridCell(0, 0, 125, 153, 0, 0)]


def test_named_food_uses_itemv2_grid_and_returns_remaining_quantity():
    from bgi_touch.tasks.auto_eat import AutoEatTask

    frame, cells = _grid_fixture()
    scanner = _FakeScanner(frame, cells)
    recognizer = Mock()
    recognizer.match.return_value = SimpleNamespace(name="甜甜花酿鸡")
    confirm = Mock()
    ctx = _task_context()
    task = AutoEatTask(
        ctx,
        food_name="甜甜花酿鸡",
        scanner=scanner,
        recognizer=recognizer,
        log=lambda _message: None,
    )
    task._find_food_confirm = Mock(return_value=confirm)

    with patch(
        "bgi_touch.tasks.auto_eat.recognize_inventory_count",
        return_value=SimpleNamespace(count=3, reason=""),
    ):
        result = task.run()

    assert result == 2
    assert scanner.opened is True
    assert scanner.closed is True
    assert scanner.tapped == cells
    recognizer.match.assert_called_once()
    confirm.click.assert_called_once_with()
    ctx.input.release_all.assert_called_once_with()


def test_named_food_reports_not_found_and_always_closes_inventory():
    from bgi_touch.tasks.auto_eat import AutoEatTask

    frame, cells = _grid_fixture()
    scanner = _FakeScanner(frame, cells)
    recognizer = Mock()
    recognizer.match.return_value = SimpleNamespace(name="甜甜花")
    logs = []
    task = AutoEatTask(
        _task_context(),
        food_name="甜甜花酿鸡",
        scanner=scanner,
        recognizer=recognizer,
        log=logs.append,
    )

    assert task.run() == -1
    assert scanner.closed is True
    assert scanner.tapped == []
    assert any("未找到料理" in line for line in logs)


def test_named_food_keeps_use_result_when_quantity_ocr_fails():
    from bgi_touch.tasks.auto_eat import AutoEatTask

    frame, cells = _grid_fixture()
    scanner = _FakeScanner(frame, cells)
    recognizer = Mock()
    recognizer.match.return_value = SimpleNamespace(name="甜甜花酿鸡")
    confirm = Mock()
    logs = []
    task = AutoEatTask(
        _task_context(),
        food_name="甜甜花酿鸡",
        scanner=scanner,
        recognizer=recognizer,
        log=logs.append,
    )
    task._find_food_confirm = Mock(return_value=confirm)

    with patch(
        "bgi_touch.tasks.auto_eat.recognize_inventory_count",
        return_value=SimpleNamespace(count=-2, reason="PARSE"),
    ):
        assert task.run() == -2

    confirm.click.assert_called_once_with()
    assert any("无法识别料理数量" in line for line in logs)


def test_named_food_honors_cancellation_before_opening_inventory():
    from bgi_touch.tasks.auto_eat import AutoEatTask

    frame, cells = _grid_fixture()
    scanner = _FakeScanner(frame, cells)
    recognizer = Mock()
    task = AutoEatTask(
        _task_context(),
        food_name="甜甜花酿鸡",
        scanner=scanner,
        recognizer=recognizer,
        log=lambda _message: None,
    )

    assert task.run(cancelled=lambda: True) is False
    assert scanner.opened is False
    recognizer.match.assert_not_called()


def test_auto_eat_trigger_requires_recovery_and_handles_resurrection_icon():
    from bgi_touch.tasks.auto_eat import AutoEatTrigger

    ctx = _task_context()
    logs = []
    now = [100.0]
    region = SimpleNamespace(bgr=np.zeros((1080, 1920, 3), dtype=np.uint8))
    recovery = Mock(return_value=True)
    resurrection = Mock(return_value=True)
    trigger = AutoEatTrigger(
        ctx,
        check_interval_ms=20,
        eat_interval_s=8,
        recovery_detector=recovery,
        resurrection_detector=resurrection,
        log=logs.append,
    )

    with patch("bgi_touch.tasks.auto_eat.time.monotonic", side_effect=lambda: now[0]), \
         patch("bgi_touch.tasks.auto_eat.current_avatar_is_low_hp", return_value=True):
        trigger.on_frame(region)
        now[0] += 1
        trigger.on_frame(region)

    assert ctx.input.key_press.call_count == 2
    assert [call.args[0] for call in ctx.input.key_press.call_args_list] == ["Z", "Z"]
    recovery.assert_called_once_with(region)
    assert resurrection.call_count == 2
    assert any("复活图标" in line for line in logs)


def test_dispatcher_passes_auto_eat_grid_options():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.auto_eat.AutoEatTask") as task:
        task.return_value.run.return_value = 2
        result = TaskDispatcher(object()).run_auto_eat_task({
            "foodName": "甜甜花酿鸡",
            "maxPages": 4,
        })

    assert result == 2
    assert task.call_args.kwargs["max_pages"] == 4
