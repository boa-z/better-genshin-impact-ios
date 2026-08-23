"""pathing JSON（bettergi-scripts-list repo/pathing，5000+ 文件）数据模型。

坐标为原神世界地图坐标（非屏幕像素），执行依赖小地图定位（见 executor.py）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


def _field(raw: dict, *names: str, default=None):
    """Read snake_case and BetterGI's camelCase spellings interchangeably."""
    for name in names:
        if name in raw:
            return raw[name]
    return default


@dataclass
class Misidentification:
    """Recovery policy attached to a BetterGI waypoint."""

    types: list[str] = field(default_factory=lambda: ["unrecognized"])
    handling_mode: str = "previousDetectedPoint"
    arrival_time: int = 0

    @classmethod
    def parse(cls, raw: dict | None) -> "Misidentification":
        raw = raw if isinstance(raw, dict) else {}
        types = _field(raw, "type", "types", default=["unrecognized"])
        if isinstance(types, str):
            types = [types]
        if not isinstance(types, list):
            types = ["unrecognized"]
        raw_arrival_time = _field(raw, "arrival_time", "arrivalTime", default=0)
        try:
            arrival_time = int(raw_arrival_time or 0)
        except (TypeError, ValueError):
            arrival_time = 0
        return cls(
            types=[str(value) for value in types if value],
            handling_mode=str(_field(
                raw, "handling_mode", "handlingMode", default="previousDetectedPoint"
            )),
            arrival_time=arrival_time,
        )


@dataclass
class Waypoint:
    id: int
    x: float
    y: float
    type: str  # path | target | teleport | orientation
    move_mode: str  # walk | run | dash | fly | jump | climb | swim
    action: str = ""
    action_params: str = ""
    misidentification: Misidentification = field(default_factory=Misidentification)
    monster_tag: str = ""
    enable_monster_loot_split: bool = False
    description: str = ""
    items: list[dict] = field(default_factory=list)

    @classmethod
    def parse(cls, raw: dict) -> "Waypoint":
        if not isinstance(raw, dict):
            raise ValueError("地图追踪路点必须是对象")
        ext = _field(raw, "point_ext_params", "pointExtParams", default={})
        ext = ext if isinstance(ext, dict) else {}
        items = _field(raw, "items", default=[])
        return cls(
            id=int(_field(raw, "id", default=0)),
            x=float(_field(raw, "x", "X", "game_x", "gameX", "GameX")),
            y=float(_field(raw, "y", "Y", "game_y", "gameY", "GameY")),
            type=str(_field(raw, "type", default="path") or "path").lower(),
            move_mode=str(_field(raw, "move_mode", "moveMode", default="walk") or "walk").lower(),
            action=str(_field(raw, "action", default="") or "").lower(),
            action_params=str(_field(raw, "action_params", "actionParams", default="") or ""),
            misidentification=Misidentification.parse(
                _field(ext, "misidentification", default={})
            ),
            monster_tag=str(_field(ext, "monster_tag", "monsterTag", default="") or ""),
            enable_monster_loot_split=bool(_field(
                ext, "enable_monster_loot_split", "enableMonsterLootSplit", default=False
            )),
            description=str(_field(ext, "description", default="") or ""),
            items=items if isinstance(items, list) else [],
        )


@dataclass
class PathingTask:
    name: str
    map_name: str
    positions: list[Waypoint]
    info: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    map_match_method: str = "SIFT"
    realtime_triggers: dict[str, bool] = field(default_factory=lambda: {"AutoPick": True})
    farming_info: dict = field(default_factory=dict)
    source_path: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "PathingTask":
        # BetterGI's bundled route set contains both plain UTF-8 and UTF-8 BOM
        # files. ``utf-8-sig`` accepts both without leaking U+FEFF into JSON.
        source = Path(path).expanduser()
        raw = json.loads(source.read_text(encoding="utf-8-sig"))
        task = cls.parse(raw)
        task.source_path = str(source.resolve())
        return task

    @classmethod
    def parse(cls, raw: dict) -> "PathingTask":
        if not isinstance(raw, dict):
            raise ValueError("地图追踪任务根节点必须是对象")
        info = raw.get("info") or {}
        if not isinstance(info, dict):
            info = {}
        config = raw.get("config") or {}
        if not isinstance(config, dict):
            config = {}
        map_name = _field(info, "map_name", "mapName", default=None)
        if not map_name:
            map_name = _field(raw, "map_name", "mapName", default="Teyvat")
        match_method = _field(
            info, "map_match_method", "mapMatchMethod", default=None
        ) or _field(config, "map_match_method", "mapMatchMethod", default="SIFT")
        triggers = _field(config, "realtime_triggers", "realtimeTriggers", default={"AutoPick": True})
        if not isinstance(triggers, dict):
            triggers = {"AutoPick": True}
        farming_info = raw.get("farming_info") or raw.get("farmingInfo") or {}
        if not isinstance(farming_info, dict):
            farming_info = {}
        return cls(
            name=str(_field(info, "name", default="") or ""),
            map_name=str(map_name or "Teyvat"),
            positions=[Waypoint.parse(p) for p in raw.get("positions", []) or []],
            info=info,
            config=config,
            map_match_method=str(match_method or "SIFT"),
            realtime_triggers={str(k): bool(v) for k, v in triggers.items()},
            farming_info=farming_info,
        )

    def validate(self) -> None:
        for waypoint in self.positions:
            if not math.isfinite(waypoint.x) or not math.isfinite(waypoint.y):
                raise ValueError(f"路点 {waypoint.id} 坐标不是有限数值")

    def summary(self) -> dict:
        from collections import Counter
        return {
            "name": self.name,
            "map": self.map_name,
            "map_match_method": self.map_match_method,
            "points": len(self.positions),
            "types": dict(Counter(p.type for p in self.positions)),
            "move_modes": dict(Counter(p.move_mode for p in self.positions)),
            "actions": dict(Counter(p.action for p in self.positions if p.action)),
            "realtime_triggers": dict(self.realtime_triggers),
            "farming_info": dict(self.farming_info),
        }
