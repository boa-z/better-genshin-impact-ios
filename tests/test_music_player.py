import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


def _score_file(tmp_path: Path, **values) -> Path:
    path = tmp_path / f"{values.get('name', 'score')}.json"
    path.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")
    return path


def test_yuanqin_parser_supports_chords_tempo_and_repeated_key_gap(tmp_path: Path):
    from bgi_touch.tasks.music_player import parse_music_score

    source = _score_file(
        tmp_path,
        type="yuanqin",
        name="测试曲",
        bpm=120,
        time_signature="4/4",
        notes="(QW)[4]Q[4]Q[4]",
        instrument="风物之诗琴",
    )
    score = parse_music_score(source)

    assert score.name == "测试曲"
    assert score.duration_ms == pytest.approx(1500)
    assert [(event.time_ms, event.key, event.down) for event in score.events[:2]] == [
        (0, "Q", True), (0, "W", True),
    ]
    # The same key is released 30 ms before its immediate retrigger.
    assert any(event.key == "Q" and not event.down and event.time_ms == 470
               for event in score.events)
    assert any(event.key == "Q" and not event.down and event.time_ms == 970
               for event in score.events)


def test_music_parser_supports_midi_json_and_keyboard_json(tmp_path: Path):
    from bgi_touch.tasks.music_player import parse_music_score

    midi = parse_music_score(_score_file(
        tmp_path, name="midi", type="midi", bpm=120, ticks=480,
        notes="DQ0|UQ480|DW0|UW480",
    ))
    keyboard = parse_music_score(_score_file(
        tmp_path, name="keyboard", type="keyboard", bpm=60,
        notes="(QW)-[AS]",
    ))

    assert midi.duration_ms == pytest.approx(1000)
    assert [(event.key, event.down) for event in midi.events] == [
        ("Q", True), ("Q", False), ("W", True), ("W", False),
    ]
    assert keyboard.duration_ms == pytest.approx(3000)
    assert {event.key for event in keyboard.events if event.down} == {"Q", "W", "A", "S"}


def test_yuanqin_object_tokens_support_tempo_and_explicit_events(tmp_path: Path):
    from bgi_touch.tasks.music_player import MusicEvent, parse_music_score

    score = parse_music_score(_score_file(
        tmp_path,
        name="objects",
        type="yuanqin",
        bpm=120,
        notes=[
            {"note": "Q", "type": 60, "spl": "%"},
            {"note": "A", "type": 4, "spl": "none"},
            {"note": "W", "spl": "^"},
            {"note": "W", "spl": "&"},
        ],
    ))

    assert score.duration_ms == pytest.approx(1030)
    assert MusicEvent(1000, "W", True) in score.events
    assert MusicEvent(1000, "W", False) in score.events


def test_music_layout_and_transpose_match_bettergi_profiles():
    from bgi_touch.tasks.music_player import (
        MusicEvent,
        MusicTouchLayout,
        map_music_events,
    )

    layout = MusicTouchLayout.load()
    melodic = map_music_events(
        [MusicEvent(0, "Q", True)], source_instrument="风物之诗琴", transpose=12
    )
    percussion = map_music_events(
        [MusicEvent(0, "Q", True)], source_instrument="聚聚鼓", transpose=12
    )

    assert len(layout.points) == 21
    # Q=C5 transposed one octave folds back into the 21-key melodic range.
    assert melodic == (MusicEvent(0, "Q", True),)
    # Percussion profiles use exact mapping and therefore drop out-of-range notes.
    assert percussion == ()


def test_music_player_outputs_chords_as_one_multitouch_gesture():
    from bgi_touch.tasks.music_player import MusicEvent, MusicPlayerTask, MusicScore
    from bgi_touch.vision.coordinate import ScreenTransform

    device = SimpleNamespace(multi_touch=Mock())
    ctx = SimpleNamespace(
        device=device,
        transform=ScreenTransform(1920, 1080),
        sleep=Mock(),
        input=SimpleNamespace(release_all=Mock()),
    )
    task = MusicPlayerTask(ctx, ["unused.json"])
    score = MusicScore(
        Path("score.json"), "chord", "风物之诗琴", 120, "yuanqin",
        (
            MusicEvent(0, "Q", True), MusicEvent(0, "W", True),
            MusicEvent(100, "Q", False), MusicEvent(100, "W", False),
        ),
        100,
    )

    assert task._play_score(score)
    contacts = device.multi_touch.call_args.args[0]
    assert len(contacts) == 2
    assert device.multi_touch.call_args.kwargs["duration_ms"] == 100


def test_custom_bpm_scales_timeline_and_sequential_mode_plays_full_queue():
    from bgi_touch.tasks.music_player import MusicEvent, MusicPlayerTask, MusicScore
    from bgi_touch.vision.coordinate import ScreenTransform

    ctx = SimpleNamespace(
        device=SimpleNamespace(multi_touch=Mock()),
        transform=ScreenTransform(1920, 1080), sleep=Mock(),
        input=SimpleNamespace(release_all=Mock()),
    )
    score = MusicScore(
        Path("one.json"), "one", "风物之诗琴", 120, "yuanqin",
        (MusicEvent(0, "Q", True), MusicEvent(1200, "Q", False)), 1200,
    )
    task = MusicPlayerTask(ctx, ["unused.json"], custom_bpm=240)
    durations = []
    task._hold = lambda _keys, duration, _cancelled: durations.append(duration) or True
    # The real hold blocks for the requested duration. Advance the monotonic
    # clock accordingly so this unit test exercises the absolute timeline
    # without measuring test-runner overhead.
    with patch("bgi_touch.tasks.music_player.time.monotonic", side_effect=[0, 0, 0.6]):
        assert task._play_score(score)
    assert durations == [pytest.approx(600)]

    second = MusicScore(Path("two.json"), "two", "风物之诗琴", 120,
                        "yuanqin", (), 0)
    task = MusicPlayerTask(ctx, ["unused.json"], loop_count=2)
    task._scores = lambda: [score, second]
    played = []
    task._play_score = lambda value, _cancelled, _start: played.append(value.name) or True
    result = task.run()
    assert played == ["one", "two", "one", "two"]
    assert result == {"status": "completed", "completed": 4}


def test_music_directory_scan_skips_non_score_json(tmp_path: Path):
    from bgi_touch.tasks.music_player import MusicPlayerTask
    from bgi_touch.vision.coordinate import ScreenTransform

    (tmp_path / "settings.json").write_text('{"theme":"dark"}', encoding="utf-8")
    _score_file(tmp_path, name="valid", type="yuanqin", bpm=120, notes="Q[4]")
    messages = []
    ctx = SimpleNamespace(
        device=SimpleNamespace(multi_touch=Mock()),
        transform=ScreenTransform(1920, 1080), sleep=Mock(),
        input=SimpleNamespace(release_all=Mock()),
    )
    task = MusicPlayerTask(ctx, [tmp_path], log=messages.append)

    assert [score.name for score in task._scores()] == ["valid"]
    assert any("settings.json" in message for message in messages)


def test_music_queue_filters_match_bettergi_visible_library(tmp_path: Path):
    from bgi_touch.tasks.music_player import filter_music_scores, parse_music_score

    yuan = parse_music_score(_score_file(
        tmp_path, name="晨风", author="Alice", type="yuanqin", bpm=120,
        notes="Q[4]", instrument="风物之诗琴,镜花之琴",
    ))
    midi = parse_music_score(_score_file(
        tmp_path, name="鼓点", author="Bob", type="midi", bpm=120, ticks=480,
        notes="DQ0|UQ480", instrument="聚聚鼓",
    ))

    assert filter_music_scores([yuan, midi], search_text="alice") == [yuan]
    assert filter_music_scores([yuan, midi], format_filter="MIDI JSON") == [midi]
    assert filter_music_scores([yuan, midi], format_filter="yuanqin") == [yuan]
    assert filter_music_scores([yuan, midi], instrument_filter="镜花之琴") == [yuan]
    assert filter_music_scores([yuan, midi], instrument_filter="全部乐器") == [yuan, midi]


def test_music_sequential_queue_starts_at_selected_filtered_track():
    from bgi_touch.tasks.music_player import MusicPlayerTask, MusicScore
    from bgi_touch.vision.coordinate import ScreenTransform

    ctx = SimpleNamespace(
        device=SimpleNamespace(multi_touch=Mock()),
        transform=ScreenTransform(1920, 1080), sleep=Mock(),
        input=SimpleNamespace(release_all=Mock()),
    )
    scores = [
        MusicScore(Path(f"{name}.json"), name, "风物之诗琴", 120, "yuanqin", (), 0)
        for name in ("one", "two", "three")
    ]
    task = MusicPlayerTask(ctx, ["unused.json"], start_track="two")
    task._scores = lambda: scores
    played = []
    task._play_score = lambda score, *_args: played.append(score.name) or True

    result = task.run()

    assert played == ["two", "three"]
    assert result == {"status": "completed", "completed": 2}


def test_music_instrument_switcher_scans_gadget_names_and_equips():
    from bgi_touch.tasks.music_player import MusicInstrumentSwitcher

    replacement = SimpleNamespace(text="替换", click=Mock())
    region = SimpleNamespace(find_multi=Mock(return_value=[replacement]))
    ctx = SimpleNamespace(
        sleep=Mock(),
        capture_region=Mock(return_value=region),
        input=SimpleNamespace(key_press=Mock()),
    )
    scanner = Mock()
    scanner.open.return_value = True
    scanner.pages.return_value = [(1, object(), [object()])]
    scanner.detail_name.return_value = "风物之诗琴"

    with patch("bgi_touch.tasks.inventory_grid.InventoryGridScanner", return_value=scanner):
        assert MusicInstrumentSwitcher(ctx).switch("风物之诗琴,别名")

    replacement.click.assert_called_once_with()
    scanner.close.assert_called_once_with()
    ctx.input.key_press.assert_called_once_with("Z")


def test_music_player_dispatcher_maps_latest_bettergi_options():
    from bgi_touch.tasks.dispatcher import TaskDispatcher

    with patch("bgi_touch.tasks.music_player.MusicPlayerTask") as task:
        task.return_value.run.return_value = {"status": "completed", "completed": 1}
        result = TaskDispatcher(object()).run_music_player_task({
            "files": ["a.json", "b.json"],
            "playbackMode": "Shuffle",
            "speed": 1.5,
            "useCustomBpm": True,
            "customBpm": 88,
            "autoSwitchInstrument": True,
            "searchText": "风",
            "formatFilter": "原琴 JSON",
            "instrumentFilter": "风物之诗琴",
            "startTrack": "晨风",
            "transpose": -2,
            "loopCount": 3,
        })

    assert result["status"] == "completed"
    assert task.call_args.args[1] == ["a.json", "b.json"]
    assert task.call_args.kwargs["playback_mode"] == "Shuffle"
    assert task.call_args.kwargs["custom_bpm"] == 88
    assert task.call_args.kwargs["auto_switch_instrument"] is True
    assert task.call_args.kwargs["search_text"] == "风"
    assert task.call_args.kwargs["format_filter"] == "原琴 JSON"
    assert task.call_args.kwargs["instrument_filter"] == "风物之诗琴"
    assert task.call_args.kwargs["start_track"] == "晨风"
    assert task.call_args.kwargs["transpose"] == -2
