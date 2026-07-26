"""pathing JSON（bettergi-scripts-list repo/pathing，5000+ 文件）数据模型。

坐标为原神世界地图坐标（非屏幕像素），执行依赖小地图定位（见 executor.py）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Waypoint:
    id: int
    x: float
    y: float
    type: str  # path | target | teleport | orientation
    move_mode: str  # walk | run | dash | fly | jump | climb | swim
    action: str = ""
    action_params: str = ""

    @classmethod
    def parse(cls, raw: dict) -> "Waypoint":
        return cls(
            id=int(raw.get("id", 0)),
            x=float(raw["x"]),
            y=float(raw["y"]),
            type=str(raw.get("type", "path")),
            move_mode=str(raw.get("move_mode", "walk")),
            action=str(raw.get("action") or ""),
            action_params=str(raw.get("action_params") or ""),
        )


@dataclass
class PathingTask:
    name: str
    map_name: str
    positions: list[Waypoint]
    info: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "PathingTask":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.parse(raw)

    @classmethod
    def parse(cls, raw: dict) -> "PathingTask":
        info = raw.get("info") or {}
        return cls(
            name=str(info.get("name", "")),
            map_name=str(info.get("map_name") or "Teyvat"),
            positions=[Waypoint.parse(p) for p in raw.get("positions", [])],
            info=info,
            config=raw.get("config") or {},
        )

    def summary(self) -> dict:
        from collections import Counter
        return {
            "name": self.name,
            "map": self.map_name,
            "points": len(self.positions),
            "types": dict(Counter(p.type for p in self.positions)),
            "move_modes": dict(Counter(p.move_mode for p in self.positions)),
            "actions": dict(Counter(p.action for p in self.positions if p.action)),
        }
