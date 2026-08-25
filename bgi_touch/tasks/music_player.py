"""BetterGI free-play music score parser and iOS multi-touch player."""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..engine.context import GameContext


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAYOUT = PROJECT_ROOT / "config" / "music.json"
SUPPORTED_KEYS = "QWERTYUASDFGHJZXCVBNM"
LOWEST_LATENCY_MS = 30.0
STANDARD_NOTES = dict(zip(SUPPORTED_KEYS, (
    72, 74, 76, 77, 79, 81, 83,
    60, 62, 64, 65, 67, 69, 71,
    48, 50, 52, 53, 55, 57, 59,
)))
PERCUSSION = {"绮筵之鼓", "聚聚鼓"}
FORMAT_NAMES = {
    "yuanqin": "原琴 JSON",
    "midi": "MIDI JSON",
    "keyboard": "网络键谱",
}


@dataclass(frozen=True)
class MusicEvent:
    time_ms: float
    key: str
    down: bool


@dataclass(frozen=True)
class MusicScore:
    path: Path
    name: str
    instrument: str
    bpm: float
    format: str
    events: tuple[MusicEvent, ...]
    duration_ms: float
    author: str = ""

    @property
    def format_name(self) -> str:
        return FORMAT_NAMES.get(self.format.casefold(), self.format)

    @property
    def instrument_tags(self) -> tuple[str, ...]:
        return tuple(
            value.strip() for value in self.instrument.split(",") if value.strip()
        )


def filter_music_scores(
    scores: Iterable[MusicScore],
    *,
    search_text: str = "",
    format_filter: str = "",
    instrument_filter: str = "",
) -> list[MusicScore]:
    """Freeze the playback queue from the same filters as BetterGI's music page."""
    search = str(search_text or "").strip().casefold()
    selected_format = str(format_filter or "").strip().casefold()
    selected_instrument = str(instrument_filter or "").strip().casefold()
    all_formats = {"", "全部格式", "all", "all formats"}
    all_instruments = {"", "全部乐器", "all", "all instruments"}
    output = []
    for score in scores:
        searchable = (score.name, score.author, str(score.path))
        if search and not any(search in value.casefold() for value in searchable):
            continue
        if selected_format not in all_formats and selected_format not in {
            score.format.casefold(), score.format_name.casefold(),
        }:
            continue
        if selected_instrument not in all_instruments and selected_instrument not in {
            value.casefold() for value in score.instrument_tags
        }:
            continue
        output.append(score)
    return output


@dataclass(frozen=True)
class _YuanToken:
    keys: str
    denominator: float
    special: str = "none"


@dataclass
class _KeyboardNode:
    type: str = ""
    value: str | None = None
    children: list["_KeyboardNode"] | None = None
    multiplier: float = 1.0


def _events_at(events: list[MusicEvent], when: float, keys: str, down: bool) -> None:
    for key in dict.fromkeys(str(keys).upper()):
        if key in SUPPORTED_KEYS:
            events.append(MusicEvent(when, key, down))


def _add_note(events: list[MusicEvent], start: float, duration: float, keys: str) -> None:
    if keys == "@":
        return
    _events_at(events, start, keys, True)
    _events_at(events, start + duration, keys, False)


def _normalize_events(source: Iterable[MusicEvent]) -> tuple[MusicEvent, ...]:
    events = sorted(source, key=lambda event: (event.time_ms, 0 if not event.down else 1))
    previous_down: dict[str, float] = {}
    previous_up: dict[str, int] = {}
    mutable = list(events)
    for index, event in enumerate(mutable):
        if event.down:
            up_index = previous_up.get(event.key)
            down_time = previous_down.get(event.key)
            if (up_index is not None and down_time is not None
                    and mutable[up_index].time_ms == event.time_ms):
                early = event.time_ms - LOWEST_LATENCY_MS
                if early > down_time:
                    mutable[up_index] = MusicEvent(early, event.key, False)
            previous_down[event.key] = event.time_ms
        else:
            previous_up[event.key] = index
    return tuple(sorted(mutable, key=lambda event: (event.time_ms, 0 if not event.down else 1)))


def _yuan_tokens(value: Any) -> list[_YuanToken]:
    if isinstance(value, list):
        output = []
        for item in value:
            if not isinstance(item, Mapping) or not str(item.get("note", "")).strip():
                raise ValueError("yuanqin 音符对象缺少 note 字段")
            special = str(item.get("spl", "none"))
            denominator = 0.0 if special in {"^", "&"} else float(item.get("type"))
            output.append(_YuanToken(str(item["note"]), denominator, special))
        return output
    if not isinstance(value, str):
        raise ValueError("yuanqin 曲谱的 notes 必须是字符串或音符对象数组")
    sheet = value.replace("\r", "").replace("\n", "")
    output: list[_YuanToken] = []
    index = 0
    while index < len(sheet):
        if sheet[index] == "|":
            index += 1
            continue
        if sheet[index] == "(":
            end = sheet.find(")", index + 1)
            if end < 0:
                raise ValueError("和弦缺少右括号")
            keys, index = sheet[index + 1:end], end + 1
        else:
            keys, index = sheet[index], index + 1
        if index >= len(sheet) or sheet[index] != "[":
            raise ValueError(f"音符 {keys} 缺少时值")
        end = sheet.find("]", index + 1)
        if end < 0:
            raise ValueError(f"音符 {keys} 的时值缺少右方括号")
        raw, index = sheet[index + 1:end], end + 1
        if raw in {"^", "&"}:
            output.append(_YuanToken(keys, 0, raw))
            continue
        denominator, separator, special = raw.partition("-")
        output.append(_YuanToken(keys, float(denominator), special if separator else "none"))
    return output


def _parse_yuan(value: Any, bpm: float, signature: str) -> tuple[tuple[MusicEvent, ...], float]:
    tokens = _yuan_tokens(value)
    beat_denominator = 4
    parts = signature.split("/")
    if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) > 0:
        beat_denominator = int(parts[1])
    events: list[MusicEvent] = []
    cursor = 0.0
    current_bpm = bpm
    index = 0
    tuplet = re.compile(r"^(?P<display>\d+)\.(?P<end>[36$])$")

    def duration(denominator: float) -> float:
        if current_bpm <= 0 or denominator <= 0:
            raise ValueError("BPM 和音符时值必须大于 0")
        return 60000.0 / current_bpm * beat_denominator / denominator

    while index < len(tokens):
        token = tokens[index]
        if token.special == "%":
            if token.denominator > 0:
                current_bpm = token.denominator
            index += 1
            continue
        if token.special in {"^", "&"}:
            _events_at(events, cursor, token.keys, token.special == "^")
            if token.special == "&":
                cursor += LOWEST_LATENCY_MS
            index += 1
            continue
        if tuplet.match(token.special):
            group = []
            while index < len(tokens):
                group.append(tokens[index])
                index += 1
                if group[-1].special.endswith("$"):
                    break
            if not group[-1].special.endswith("$"):
                raise ValueError("连音缺少 .$ 结束标记")
            total = duration(token.denominator)
            weights = [1.0 / float(tuplet.match(item.special).group("display")) for item in group]
            weight_sum = sum(weights)
            for item, weight in zip(group, weights):
                item_duration = total * weight / weight_sum
                _add_note(events, cursor, item_duration, item.keys)
                cursor += item_duration
            continue
        note_duration = (
            60000.0 / current_bpm / 16.0
            if token.special == "#" else duration(token.denominator)
        )
        if token.special == "*":
            note_duration *= 1.5
        if token.special in {"none", "*"}:
            ornaments = 0
            next_index = index + 1
            while next_index < len(tokens) and tokens[next_index].special == "#":
                ornaments += 1
                next_index += 1
            ornament_duration = 60000.0 / current_bpm / 16.0 * ornaments
            if ornament_duration < note_duration:
                note_duration -= ornament_duration
        _add_note(events, cursor, note_duration, token.keys)
        cursor += note_duration
        index += 1
    return _normalize_events(events), cursor


def _parse_midi_json(sheet: str, bpm: float, ticks: int) -> tuple[tuple[MusicEvent, ...], float]:
    pattern = re.compile(r"^(?P<status>[DU])(?P<keys>[A-Z@]+)(?P<ticks>\d+)$")
    parts = [value.strip() for value in sheet.split("|") if value.strip()]
    events: list[MusicEvent] = []
    cursor = 0.0
    previous_status = previous_keys = ""
    previous_event = False
    for index, raw in enumerate(parts):
        if raw.startswith("*"):
            bpm = float(raw[1:])
            if bpm <= 0:
                raise ValueError(f"无法解析 MIDI JSON 变速标记：{raw}")
            previous_event = False
            continue
        match = pattern.match(raw)
        if not match:
            raise ValueError(f"无法解析 MIDI JSON 事件：{raw}")
        status, keys = match.group("status"), match.group("keys")
        delay = int(match.group("ticks")) * 60000.0 / (bpm * ticks)
        cursor += delay
        if (status == "D" and previous_event and previous_status == "U"
                and any(key in keys for key in previous_keys)
                and delay < LOWEST_LATENCY_MS):
            cursor += LOWEST_LATENCY_MS
        elif status == "U" and keys != "@" and delay >= LOWEST_LATENCY_MS and index + 1 < len(parts):
            next_match = pattern.match(parts[index + 1])
            if (next_match and next_match.group("status") == "D"
                    and any(key in keys for key in next_match.group("keys"))):
                cursor -= LOWEST_LATENCY_MS
        if keys != "@":
            _events_at(events, cursor, keys, status == "D")
        previous_status, previous_keys, previous_event = status, keys, True
    return tuple(events), cursor


def _keyboard_nodes(text: str, index: list[int], closing: str | None = None) -> list[_KeyboardNode]:
    nodes = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    while index[0] < len(text):
        current = text[index[0]]
        if closing and current == closing:
            index[0] += 1
            return nodes
        if current in pairs:
            index[0] += 1
            node = _KeyboardNode(current, children=_keyboard_nodes(text, index, pairs[current]))
        elif current in ")]}":
            raise ValueError(f"键谱出现不匹配的括号：{current}")
        elif current == "-":
            if not nodes:
                raise ValueError("键谱延音符号前没有音符")
            nodes[-1].multiplier += 1
            index[0] += 1
            continue
        else:
            node = _KeyboardNode(value=current, children=[])
            index[0] += 1
        while index[0] < len(text) and text[index[0]] == "-":
            node.multiplier += 1
            index[0] += 1
        nodes.append(node)
    if closing:
        raise ValueError(f"键谱缺少右括号：{closing}")
    return nodes


def _unfold_keyboard(node: _KeyboardNode, start: float, duration: float,
                     output: list[tuple[str, float, float]]) -> None:
    if node.value is not None:
        key = node.value.upper()
        if key != "0" and key in SUPPORTED_KEYS:
            output.append((key, start, duration))
        return
    children = node.children or []
    offset = 0.001 if node.type == "{" else 0.0
    if node.type == "[" and children:
        unit = duration / len(children)
        for index, child in enumerate(children):
            _unfold_keyboard(child, start + index * unit, unit, output)
    else:
        for child in children:
            _unfold_keyboard(child, start + offset, duration, output)


def _parse_keyboard(sheet: str, bpm: float) -> tuple[tuple[MusicEvent, ...], float]:
    text = re.sub(r"/\(([^)]+)\)", r"{\1}", sheet)
    text = re.sub(r"/([A-Z])", r"{\1}", text)
    text = text.replace(" ", "0")
    text = re.sub(r"/\[([^]]+)\]", r"{[\1]}", text)
    text = text.replace("/", "").replace(">", "")
    nodes = _keyboard_nodes(text, [0])
    notes: list[tuple[str, float, float]] = []
    beats = 0.0
    for node in nodes:
        _unfold_keyboard(node, beats, node.multiplier, notes)
        beats += node.multiplier
    beat_ms = 60000.0 / bpm
    events = []
    for key, start, duration in notes:
        _add_note(events, start * beat_ms, duration * beat_ms, key)
    return _normalize_events(events), beats * beat_ms


def parse_music_score(path: str | Path) -> MusicScore:
    source = Path(path).expanduser().resolve()
    if source.suffix.casefold() != ".json":
        raise ValueError("iOS MusicPlayer 当前支持原琴/MIDI JSON/网络键谱 JSON")
    raw = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, Mapping):
        raise ValueError("曲谱根节点必须是对象")
    score_type = str(raw.get("type", "yuanqin")).strip().casefold()
    bpm = float(raw.get("bpm") or 120)
    if bpm <= 0:
        bpm = 120
    notes = raw.get("notes")
    if notes is None:
        raise ValueError("曲谱缺少 notes 字段")
    if score_type == "yuanqin":
        events, duration = _parse_yuan(notes, bpm, str(raw.get("time_signature", "4/4")))
    elif score_type == "midi":
        if not isinstance(notes, str):
            raise ValueError("MidiJson 曲谱的 notes 必须是字符串")
        events, duration = _parse_midi_json(notes, bpm, max(1, int(raw.get("ticks") or 480)))
    elif score_type == "keyboard":
        if not isinstance(notes, str):
            raise ValueError("Keyboard 曲谱的 notes 必须是字符串")
        events, duration = _parse_keyboard(notes, bpm)
    else:
        raise ValueError(f"不支持的 AutoYuanQin 曲谱类型：{score_type}")
    return MusicScore(
        source,
        str(raw.get("name") or source.stem),
        str(raw.get("instrument") or "风物之诗琴"),
        bpm,
        score_type,
        events,
        duration,
        str(raw.get("author") or "未知作者"),
    )


def map_music_events(events: Iterable[MusicEvent], *, source_instrument: str,
                     transpose: int = 0) -> tuple[MusicEvent, ...]:
    exact = source_instrument.split(",", 1)[0].strip() in PERCUSSION
    reverse = {note: key for key, note in STANDARD_NOTES.items()}
    output = []
    for event in events:
        note = STANDARD_NOTES.get(event.key)
        if note is None:
            continue
        target = note + int(transpose)
        if not exact:
            while target < 48:
                target += 12
            while target > 83:
                target -= 12
        key = reverse.get(target)
        if key:
            output.append(MusicEvent(event.time_ms, key, event.down))
    return _normalize_events(output)


@dataclass(frozen=True)
class MusicTouchLayout:
    points: dict[str, tuple[float, float]]
    max_contacts: int = 5

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MusicTouchLayout":
        source = Path(path or DEFAULT_LAYOUT).expanduser().resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        points = raw.get("keyPoints") if isinstance(raw, Mapping) else None
        if not isinstance(points, Mapping):
            raise ValueError(f"音乐触控布局缺少 keyPoints：{source}")
        parsed = {}
        for key, point in points.items():
            if str(key).upper() not in SUPPORTED_KEYS or not isinstance(point, Mapping):
                continue
            parsed[str(key).upper()] = (float(point["x"]), float(point["y"]))
        missing = set(SUPPORTED_KEYS) - parsed.keys()
        if missing:
            raise ValueError(f"音乐触控布局缺少按键：{''.join(sorted(missing))}")
        return cls(parsed, max(1, min(10, int(raw.get("maxContacts", 5)))))


class MusicInstrumentSwitcher:
    def __init__(self, ctx: GameContext, *, max_pages: int = 20,
                 log: Callable[[str], None] = print):
        self.ctx = ctx
        self.max_pages = max_pages
        self.log = log

    def switch(self, instrument: str, cancelled: Callable[[], bool] | None = None) -> bool:
        from .common_jobs import exclusive_realtime_triggers

        with exclusive_realtime_triggers(self.ctx):
            return self._switch_impl(instrument, cancelled)

    def _switch_impl(self, instrument: str, cancelled: Callable[[], bool] | None = None) -> bool:
        from .inventory_grid import InventoryGridScanner

        name = str(instrument).split(",", 1)[0].strip()
        if not name:
            return False
        scanner = InventoryGridScanner(
            self.ctx, "Gadget", max_pages=self.max_pages, log=self.log
        )
        if not scanner.open():
            return False
        found = False
        ready = False
        try:
            for _, _frame, cells in scanner.pages(cancelled):
                for cell in cells:
                    if cancelled and cancelled():
                        return False
                    scanner.tap(cell)
                    self.ctx.sleep(260)
                    detail_name = "".join(scanner.detail_name().split())
                    if detail_name == "".join(name.split()):
                        found = True
                        break
                if found:
                    break
            if not found:
                self.log(f"[MusicPlayer] 背包中未找到乐器：{name}")
                return False
            from ..engine.recognition import RecognitionObject

            for _ in range(6):
                hits = self.ctx.capture_region().find_multi(
                    RecognitionObject.ocr(1550, 900, 350, 160), limit=10
                )
                text = "".join(hit.text for hit in hits)
                if "卸下" in text:
                    ready = True
                    break
                replacement = next((
                    hit for hit in hits if "替换" in hit.text or "替換" in hit.text
                ), None)
                if replacement is not None:
                    replacement.click()
                    self.ctx.sleep(500)
                    ready = True
                    break
                self.ctx.sleep(300)
        finally:
            scanner.close()
        if not ready:
            self.log(f"[MusicPlayer] 无法确认 {name} 的装备按钮")
            return False
        self.ctx.sleep(1000)
        self.ctx.input.key_press("Z")
        self.ctx.sleep(2000)
        self.log(f"[MusicPlayer] 乐器已就绪：{name}")
        return True


class MusicPlayerTask:
    def __init__(
        self,
        ctx: GameContext,
        sources: Sequence[str | Path],
        *,
        layout_path: str | Path | None = None,
        playback_mode: str = "Sequential",
        speed: float = 1.0,
        custom_bpm: float | None = None,
        transpose: int = 0,
        auto_switch_instrument: bool = False,
        start_position_s: float = 0.0,
        search_text: str = "",
        format_filter: str = "",
        instrument_filter: str = "",
        start_index: int = 0,
        start_track: str | Path | None = None,
        loop_count: int = 1,
        max_instrument_pages: int = 20,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.sources = [Path(value).expanduser().resolve() for value in sources]
        self.layout = MusicTouchLayout.load(layout_path)
        mode = str(playback_mode).split(".")[-1].casefold()
        aliases = {
            "sequential": "Sequential", "顺序": "Sequential",
            "singleloop": "SingleLoop", "单曲循环": "SingleLoop",
            "shuffle": "Shuffle", "随机": "Shuffle",
        }
        if mode not in aliases:
            raise ValueError(f"不支持的播放模式：{playback_mode}")
        self.playback_mode = aliases[mode]
        self.speed = max(0.1, min(10.0, float(speed)))
        self.custom_bpm = float(custom_bpm) if custom_bpm is not None else None
        self.transpose = max(-48, min(48, int(transpose)))
        self.auto_switch_instrument = bool(auto_switch_instrument)
        self.start_position_ms = max(0.0, float(start_position_s) * 1000)
        self.search_text = str(search_text or "")
        self.format_filter = str(format_filter or "")
        self.instrument_filter = str(instrument_filter or "")
        self.start_index = max(0, int(start_index))
        self.start_track = str(start_track or "").strip()
        self.loop_count = max(1, min(1000, int(loop_count)))
        self.switcher = MusicInstrumentSwitcher(
            ctx, max_pages=max_instrument_pages, log=log
        )
        self.log = log

    def _scores(self) -> list[MusicScore]:
        files = []
        for source in self.sources:
            if source.is_dir():
                files.extend(sorted(source.rglob("*.json")))
            elif source.is_file():
                files.append(source)
            else:
                raise FileNotFoundError(f"曲谱不存在：{source}")
        scores = []
        errors = []
        for path in dict.fromkeys(files):
            try:
                scores.append(parse_music_score(path))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                errors.append(f"{path.name}: {error}")
                self.log(f"[MusicPlayer] 跳过无效曲谱 {path}: {error}")
        if not scores:
            detail = f"（{errors[0]}）" if errors else ""
            raise ValueError(f"未找到可播放的 JSON 曲谱{detail}")
        filtered = filter_music_scores(
            scores,
            search_text=self.search_text,
            format_filter=self.format_filter,
            instrument_filter=self.instrument_filter,
        )
        if not filtered:
            raise ValueError("当前搜索、格式和乐器筛选下没有可播放曲谱")
        return filtered

    def _queue_start_index(self, scores: Sequence[MusicScore]) -> int:
        if self.start_track:
            target = self.start_track.casefold()
            for index, score in enumerate(scores):
                candidates = (
                    score.name,
                    str(score.path),
                    score.path.name,
                    score.path.stem,
                )
                if any(target == value.casefold() for value in candidates):
                    return index
            raise ValueError(f"筛选后的播放队列中没有起始曲目：{self.start_track}")
        return min(self.start_index, len(scores) - 1)

    def _hold(self, keys: set[str], duration_ms: float,
              cancelled: Callable[[], bool] | None) -> bool:
        remaining = max(0.0, duration_ms)
        if not keys:
            while remaining > 0:
                if cancelled and cancelled():
                    return False
                chunk = min(250.0, remaining)
                self.ctx.sleep(chunk)
                remaining -= chunk
            return True
        selected = sorted(keys)[:self.layout.max_contacts]
        if len(keys) > len(selected):
            self.log(f"[MusicPlayer] 和弦超过 {self.layout.max_contacts} 点，已忽略多余音符")
        contacts = []
        for key in selected:
            ref_x, ref_y = self.layout.points[key]
            x, y = self.ctx.transform.to_device(ref_x, ref_y)
            contacts.append({"x1": x, "y1": y, "x2": x, "y2": y})
        while remaining > 0:
            if cancelled and cancelled():
                return False
            chunk = min(5000.0, remaining)
            self.ctx.device.multi_touch(
                contacts,
                duration_ms=max(30, round(chunk)),
                image_width=self.ctx.transform.device_width,
                image_height=self.ctx.transform.device_height,
            )
            remaining -= chunk
        return not cancelled or not cancelled()

    def _play_score(self, score: MusicScore,
                    cancelled: Callable[[], bool] | None = None,
                    start_position_ms: float = 0.0) -> bool:
        speed = (
            max(0.1, min(10.0, self.custom_bpm / score.bpm))
            if self.custom_bpm and self.custom_bpm > 0 and score.bpm > 0
            else self.speed
        )
        events = map_music_events(
            score.events, source_instrument=score.instrument, transpose=self.transpose
        )
        start = min(max(0.0, start_position_ms), score.duration_ms)
        active: set[str] = set()
        pending = []
        for event in events:
            if event.time_ms <= start:
                (active.add if event.down else active.discard)(event.key)
            else:
                pending.append(event)
        cursor = start
        index = 0
        started_at = time.monotonic()
        self.log(f"[MusicPlayer] 播放 {score.name} BPM={score.bpm:g} speed={speed:.2f}")
        while index < len(pending):
            event_time = pending[index].time_ms
            target_elapsed = (event_time - start) / speed
            elapsed = (time.monotonic() - started_at) * 1000
            if not self._hold(active, max(0.0, target_elapsed - elapsed), cancelled):
                return False
            while index < len(pending) and pending[index].time_ms == event_time:
                event = pending[index]
                (active.add if event.down else active.discard)(event.key)
                index += 1
            cursor = event_time
        target_duration = (score.duration_ms - start) / speed
        elapsed = (time.monotonic() - started_at) * 1000
        remaining = max(0.0, target_duration - elapsed)
        return self._hold(active, remaining, cancelled) if remaining > 0 else True

    def run(self, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        scores = self._scores()
        completed = 0
        active_instrument = None
        index = self._queue_start_index(scores)
        if self.playback_mode == "SingleLoop":
            target_tracks = self.loop_count
        elif self.playback_mode == "Sequential":
            target_tracks = len(scores) - index + len(scores) * (self.loop_count - 1)
        else:
            target_tracks = len(scores) * self.loop_count
        first_track = True
        try:
            while completed < target_tracks:
                if cancelled and cancelled():
                    return {"status": "cancelled", "completed": completed}
                score = scores[index]
                instrument = score.instrument.split(",", 1)[0].strip()
                if self.auto_switch_instrument and active_instrument != instrument:
                    if not self.switcher.switch(instrument, cancelled):
                        return {"status": "instrument_not_ready", "completed": completed}
                    active_instrument = instrument
                if not self._play_score(
                    score,
                    cancelled,
                    self.start_position_ms if first_track else 0.0,
                ):
                    return {"status": "cancelled", "completed": completed}
                first_track = False
                completed += 1
                if self.playback_mode == "SingleLoop":
                    continue
                if self.playback_mode == "Shuffle" and len(scores) > 1:
                    candidate = random.randrange(len(scores) - 1)
                    index = candidate + (candidate >= index)
                else:
                    index += 1
                    if index >= len(scores):
                        index = 0
            return {"status": "completed", "completed": completed}
        finally:
            self.ctx.input.release_all()
