from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


class _FakeScanner:
    category = SimpleNamespace(name="Materials")

    def __init__(self, frame, cells, details):
        self.frame = frame
        self.cells_to_scan = cells
        self.details = iter(details)
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True
        return True

    def pages(self, _cancelled=None):
        yield 1, self.frame, self.cells_to_scan

    def tap(self, _cell):
        return None

    def detail_name(self, _frame=None):
        return self.current.name

    def detail_stars(self, _frame=None):
        return self.current.stars

    def close(self):
        self.closed = True
        return True

    @property
    def current(self):
        # The task reads name and stars consecutively for each selected card.
        # The iterator is advanced by the context's capture hook below.
        return self._current

    def set_current(self, value):
        self._current = value


class _FakeRecognizer:
    def __init__(self, matches):
        self.matches = iter(matches)

    def match(self, _icon):
        return next(self.matches)


def _run_context(scanner):
    # Advance the scanner detail for each click before the task's capture.
    original_tap = scanner.tap

    def tap(cell):
        original_tap(cell)
        scanner.set_current(next(scanner.details))

    scanner.tap = tap
    return SimpleNamespace(
        sleep=lambda _ms: None,
        capture_bgr=lambda: np.zeros((180, 280, 3), dtype=np.uint8),
    )


def test_normalize_grid_icon_keeps_upper_square_and_discards_quantity_strip():
    from bgi_touch.tasks.inventory_grid import normalize_grid_icon

    card = np.zeros((153, 125, 3), dtype=np.uint8)
    card[:125] = (12, 34, 56)
    card[125:] = (220, 220, 220)

    icon = normalize_grid_icon(card)

    assert icon.shape == (125, 125, 3)
    assert tuple(icon[0, 0]) == (12, 34, 56)
    assert tuple(icon[-1, -1]) == (12, 34, 56)


def test_grid_icons_accuracy_returns_name_star_and_score_report(tmp_path):
    from bgi_touch.tasks.inventory_grid import GridCell, GridIconsAccuracyTestTask

    frame = np.zeros((170, 280, 3), dtype=np.uint8)
    cells = [GridCell(0, 0, 125, 153, 0, 0), GridCell(130, 0, 125, 153, 0, 1)]
    scanner = _FakeScanner(
        frame,
        cells,
        [SimpleNamespace(name="萃凝晶", stars=1), SimpleNamespace(name="白铁块", stars=1)],
    )
    ctx = _run_context(scanner)
    recognizer = _FakeRecognizer([
        SimpleNamespace(name="萃凝晶", score=0.91, quality_level=1),
        SimpleNamespace(name="白铁块", score=0.92, quality_level=2),
    ])

    report = GridIconsAccuracyTestTask(
        ctx,
        "Materials",
        max_num_to_test=2,
        output_dir=tmp_path,
        recognizer=recognizer,
        scanner=scanner,
    ).run()

    assert report["tested"] == 2
    assert report["correct"] == 1
    assert report["nameCorrect"] == 2
    assert report["starChecked"] == 2
    assert report["starCorrect"] == 1
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["results"][0]["matched"] is True
    assert report["results"][1]["starMatch"] is False
    assert (tmp_path / "results.json").exists()
    assert scanner.opened is True
    assert scanner.closed is True


def test_grid_icons_accuracy_supports_manual_artifact_set_filter_grid(tmp_path):
    from bgi_touch.tasks.inventory_grid import GridCell, GridIconsAccuracyTestTask
    from bgi_touch.vision.coordinate import ScreenTransform

    # ArtifactSetFilter cards are wide rows.  The flower icon is cropped from
    # the left side of the selected row, not from a regular 8-column tile.
    frame = np.zeros((100, 640, 3), dtype=np.uint8)
    cells = [GridCell(0, 0, 610, 70, 0, 0)]
    scanner = _FakeScanner(
        frame,
        cells,
        [SimpleNamespace(name="守护之花", stars=-1)],
    )
    ctx = _run_context(scanner)
    ctx.transform = ScreenTransform(1920, 1080)
    recognizer = _FakeRecognizer([
        SimpleNamespace(name="守护之花", score=0.89, quality_level=5),
    ])

    report = GridIconsAccuracyTestTask(
        ctx,
        "ArtifactSetFilter",
        max_num_to_test=1,
        output_dir=tmp_path,
        recognizer=recognizer,
        scanner=scanner,
    ).run()

    assert report["category"] == "ArtifactSetFilter"
    assert report["tested"] == 1
    assert report["correct"] == 1
    assert report["starChecked"] == 0
    assert report["results"][0]["starMatch"] is None
    assert scanner.opened is False
    assert scanner.closed is True


def test_grid_icons_accuracy_missing_model_fails_before_opening_grid(tmp_path):
    from bgi_touch.tasks.inventory_grid import GridIconsAccuracyTestTask

    scanner = _FakeScanner(
        np.zeros((170, 140, 3), dtype=np.uint8), [], [],
    )
    ctx = _run_context(scanner)
    with patch(
        "bgi_touch.vision.item_recognizer.ItemIconRecognizer",
        side_effect=FileNotFoundError("missing"),
    ):
        with pytest.raises(FileNotFoundError, match="缺少 ItemV2 模型"):
            GridIconsAccuracyTestTask(
                ctx, "Materials", output_dir=tmp_path, scanner=scanner,
            ).run()

    assert scanner.opened is False


def test_grid_icons_accuracy_dispatcher_accepts_upstream_parameter_aliases():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.inventory_grid.GridIconsAccuracyTestTask") as task:
        task.return_value.run.return_value = {"tested": 1}
        result = TaskDispatcher(object()).run_task({
            "name": "GridIconsAccuracyTestTask",
            "config": {
                "gridName": "Materials",
                "maxNum": 4,
                "maxPages": 2,
                "minScore": 0.8,
                "outputDir": "/tmp/grid-accuracy",
            },
        })

    assert result == {"tested": 1}
    kwargs = task.call_args.kwargs
    assert kwargs["max_num_to_test"] == 4
    assert kwargs["max_pages"] == 2
    assert kwargs["score_threshold"] == pytest.approx(0.8)
    assert kwargs["output_dir"] == "/tmp/grid-accuracy"
