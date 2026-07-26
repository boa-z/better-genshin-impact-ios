"""触控布局 profile：按钮/摇杆/视角区域的归一化坐标，以及键位映射。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAYOUT = PROJECT_ROOT / "config" / "controls" / "genshin-default.json"

KEY_ALIASES = {
    " ": "SPACE",
    "SPACEBAR": "SPACE",
    "LEFTSHIFT": "LSHIFT",
    "SHIFTKEY": "SHIFT",
    "RETURN": "ENTER",
    "ESC": "ESCAPE",
}


def normalize_key(key: str) -> str:
    k = str(key).upper()
    for prefix in ("VK_", "KEY_"):
        if k.startswith(prefix):
            k = k[len(prefix):]
    return KEY_ALIASES.get(k, k)


@dataclass
class ControlLayout:
    joystick_center: tuple[float, float]
    joystick_radius_n: float
    camera_region: tuple[float, float, float, float]  # nx, ny, nw, nh
    buttons: dict[str, tuple[float, float]]
    key_map: dict[str, dict]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_LAYOUT) -> "ControlLayout":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cam = raw["camera"]["region"]
        return cls(
            joystick_center=(raw["joystick"]["center"]["nx"], raw["joystick"]["center"]["ny"]),
            joystick_radius_n=raw["joystick"]["radiusN"],
            camera_region=(cam["nx"], cam["ny"], cam["nw"], cam["nh"]),
            buttons={name: (p["nx"], p["ny"]) for name, p in raw["buttons"].items()},
            key_map={normalize_key(k): v for k, v in raw["keyMap"].items()},
        )

    def binding(self, key: str) -> dict | None:
        return self.key_map.get(normalize_key(key))
