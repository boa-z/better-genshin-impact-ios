"""Touch-compatible BetterGI AutoAlbum task.

The Windows implementation navigates the themed album with fixed reference
points and uses small templates to decide whether a song is complete.  The
same state machine works on iOS because the album is still rendered in the
same 16:9 reference layout; all clicks go through ``ScreenTransform`` and all
frames are consumed by the existing six-lane player.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from ..engine.context import GameContext
from ..engine.recognition import Mat, RecognitionObject
from .auto_music import AutoMusicGameTask


ASSETS = Path(__file__).resolve().parents[2] / "assets" / "templates" / "automusic"
@dataclass(frozen=True)
class AlbumDifficulty:
    name: str
    select_point: tuple[int, int]
    completion_roi: tuple[int, int, int, int]


DIFFICULTIES = (
    AlbumDifficulty("普通", (480, 600), (450, 430, 200, 60)),
    AlbumDifficulty("困难", (800, 600), (450, 520, 200, 60)),
    AlbumDifficulty("大师", (1150, 600), (450, 610, 200, 60)),
    AlbumDifficulty("传说", (1400, 600), (450, 690, 200, 60)),
)
_DIFFICULTY_ALIASES = {
    "normal": "普通",
    "easy": "普通",
    "hard": "困难",
    "master": "大师",
    "legend": "传说",
    "all": "所有",
}


def resolve_album_difficulties(level: str | Sequence[str] | None) -> tuple[AlbumDifficulty, ...]:
    """Normalize BetterGI's ``MusicLevel`` setting into ordered levels."""
    if level is None or str(level).strip() == "":
        values = ("传说",)
    elif isinstance(level, str):
        values = (level,)
    else:
        values = tuple(str(item) for item in level)
    normalized = []
    for value in values:
        key = str(value).strip()
        key = _DIFFICULTY_ALIASES.get(key.lower(), key)
        if key == "所有":
            return DIFFICULTIES
        if key not in {item.name for item in DIFFICULTIES}:
            supported = ", ".join(item.name for item in DIFFICULTIES) + ", 所有"
            raise ValueError(f"AutoAlbum 难度无效：{value}（支持 {supported}）")
        if key not in normalized:
            normalized.append(key)
    return tuple(item for item in DIFFICULTIES if item.name in normalized)


class AutoAlbumTask:
    """Play unfinished songs in the currently opened themed album."""

    def __init__(
        self,
        ctx: GameContext,
        *,
        music_level: str | Sequence[str] | None = "传说",
        must_canorus_level: bool = False,
        song_count: int = 13,
        track_timeout_s: float = 900.0,
        timeout_s: float = 7200.0,
        log: Callable[[str], None] = print,
    ):
        self.ctx = ctx
        self.difficulties = resolve_album_difficulties(music_level)
        self.must_canorus_level = bool(must_canorus_level)
        self.song_count = max(1, min(13, int(song_count)))
        self.track_timeout_s = max(20.0, float(track_timeout_s))
        self.timeout_s = max(30.0, float(timeout_s))
        self.log = log
        self._templates: dict[str, Mat] = {}

    def _template(self, name: str) -> Mat:
        if name not in self._templates:
            self._templates[name] = Mat.from_file(str(ASSETS / f"{name}.png"))
        return self._templates[name]

    def _find(self, region, name: str, roi=None, threshold: float = 0.70):
        ro = RecognitionObject.template_match(
            self._template(name), *(roi if roi is not None else (None,) * 4)
        )
        ro.threshold = threshold
        return region.find(ro)

    @staticmethod
    def _clean(text: str) -> str:
        return str(text).replace(" ", "").replace("\u3000", "")

    def _ocr_contains(self, region, words: tuple[str, ...], roi=(0, 0, 1920, 1080)) -> bool:
        try:
            hits = region.find_multi(RecognitionObject.ocr(*roi), limit=50)
        except Exception:
            return False
        return any(any(word in self._clean(hit.text) for word in words) for hit in hits)

    def _album_icon_visible(self, region) -> bool:
        return not self._find(region, "ui_left_top_album_icon", (0, 0, 150, 120)).is_empty()

    def _validate_album_screen(self) -> None:
        region = self.ctx.capture_region()
        if self._ocr_contains(region, ("全部",), (0, 0, 600, 220)):
            raise RuntimeError("当前在全部歌曲页面，请切换到按国家主题的专辑页面")
        if self._album_icon_visible(region):
            return
        if self._ocr_contains(region, ("专辑", "曲目"), (0, 0, 900, 300)):
            return
        raise RuntimeError("当前未处于主题专辑界面，请先打开千音雅集专辑")

    def _click_confirm(self, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            region = self.ctx.capture_region()
            try:
                confirm = self._find(region, "btn_white_confirm")
                if not confirm.is_empty():
                    confirm.click()
                    return True
            except FileNotFoundError:
                pass
            try:
                hits = region.find_multi(RecognitionObject.ocr(1100, 700, 820, 300), limit=30)
            except Exception:
                hits = []
            for hit in hits:
                if self._clean(hit.text) in {"确认", "确定", "Confirm"}:
                    hit.click()
                    return True
            self.ctx.sleep(250)
        return False

    def _song_is_complete(self, difficulty: AlbumDifficulty) -> bool:
        region = self.ctx.capture_region()
        if self.must_canorus_level:
            return not self._find(region, "music_canorus", difficulty.completion_roi).is_empty()
        return not self._find(region, "album_music_complate", (900, 320, 100, 80)).is_empty()

    def _track_list_visible(self, region) -> bool:
        return not self._find(region, "btn_list", (1200, 650, 720, 430)).is_empty()

    def _return_to_album(self, deadline: float) -> bool:
        while time.monotonic() < deadline:
            region = self.ctx.capture_region()
            if self._album_icon_visible(region):
                return True
            if self._track_list_visible(region):
                self._find(region, "btn_list", (1200, 650, 720, 430)).click()
            else:
                self.ctx.input.key_press("ESCAPE")
            self.ctx.sleep(700)
        return False

    def _play_song(self, difficulty: AlbumDifficulty, cancelled: Callable[[], bool] | None,
                   deadline: float) -> bool:
        if not self._click_confirm(timeout_s=min(6.0, max(0.5, deadline - time.monotonic()))):
            self.log("[AutoAlbum] 未找到歌曲确认按钮")
            return False
        self.ctx.input.click_ref(*difficulty.select_point)
        self.ctx.sleep(250)
        if not self._click_confirm(timeout_s=min(6.0, max(0.5, deadline - time.monotonic()))):
            self.log(f"[AutoAlbum] 未找到「{difficulty.name}」开始按钮")
            return False
        self.ctx.sleep(500)
        track_deadline = min(deadline, time.monotonic() + self.track_timeout_s)
        player = AutoMusicGameTask(
            self.ctx,
            timeout_s=max(1.0, track_deadline - time.monotonic()),
            idle_timeout_s=8.0,
            log=self.log,
        )
        return player.run(
            cancelled=cancelled,
            frame_observer=self._track_list_visible,
        )

    def run(self, cancelled: Callable[[], bool] | None = None) -> bool:
        deadline = time.monotonic() + self.timeout_s
        self._validate_album_screen()
        self.log(
            f"[AutoAlbum] 开始处理 {', '.join(item.name for item in self.difficulties)} 难度，"
            f"每个难度最多 {self.song_count} 首"
        )
        for difficulty in self.difficulties:
            for song_no in range(1, self.song_count + 1):
                if time.monotonic() >= deadline or (cancelled and cancelled()):
                    self.ctx.input.release_all()
                    return False
                if self._song_is_complete(difficulty):
                    self.log(f"[AutoAlbum] {difficulty.name} 第 {song_no} 首已完成，跳过")
                else:
                    self.log(f"[AutoAlbum] 演奏 {difficulty.name} 第 {song_no} 首")
                    if not self._play_song(difficulty, cancelled, deadline):
                        return False
                    if not self._return_to_album(deadline):
                        self.log("[AutoAlbum] 演奏结束后未回到专辑列表")
                        return False
                if song_no < self.song_count:
                    self.ctx.input.click_ref(310, 220)
                    self.ctx.sleep(800)
        self.log("[AutoAlbum] 当前专辑处理完成")
        return True
