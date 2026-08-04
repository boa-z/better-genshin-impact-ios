"""触控布局 profile：按钮/摇杆/视角区域的归一化坐标，以及键位映射。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAYOUT = PROJECT_ROOT / "config" / "controls" / "genshin-default.json"

KEY_ALIASES = {
    " ": "SPACE",
    "SPACEBAR": "SPACE",
    "LEFTSHIFT": "LSHIFT",
    "SHIFTLEFT": "LSHIFT",
    "SHIFTRIGHT": "RSHIFT",
    "SHIFTKEY": "SHIFT",
    "CONTROLLEFT": "LCTRL",
    "CONTROLRIGHT": "RCTRL",
    "RETURN": "ENTER",
    "ESC": "ESCAPE",
    "ARROWUP": "UP",
    "ARROWDOWN": "DOWN",
    "ARROWLEFT": "LEFT",
    "ARROWRIGHT": "RIGHT",
}


def normalize_key(key: str) -> str:
    raw = str(key)
    if raw == " ":
        return "SPACE"
    k = raw.strip().upper()
    for prefix in ("VK_", "KEY_"):
        if k.startswith(prefix):
            k = k[len(prefix):]
    # DeviceHub profiles use browser KeyboardEvent.code values (KeyW, Digit1,
    # ShiftLeft), while BetterGI scripts use the shorter PC key names.
    if k.startswith("KEY") and len(k) == 4:
        k = k[3:]
    elif k.startswith("DIGIT") and len(k) == 6:
        k = k[5:]
    return KEY_ALIASES.get(k, k)


def _bound_codes(bind: Any) -> list[str]:
    if isinstance(bind, str):
        return [bind]
    if isinstance(bind, (list, tuple)):
        return [value for value in bind if isinstance(value, str)]
    if isinstance(bind, Mapping):
        return [value for values in bind.values() for value in _bound_codes(values)]
    return []


def _merge_layout(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge small mode overlays without duplicating the full layout."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_layout(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class DeviceHubProfile:
    """DeviceHub Mask native v2 keymap, retaining raw browser key codes."""

    name: str
    mappings: tuple[dict[str, Any], ...]
    bundle_identifiers: tuple[str, ...] = ()
    target_resolution: tuple[int, int] | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DeviceHubProfile":
        resolution = raw.get("targetResolution")
        target = None
        if isinstance(resolution, Mapping):
            width, height = resolution.get("width"), resolution.get("height")
            if isinstance(width, int) and isinstance(height, int):
                target = (width, height)
        mappings = raw.get("mappings")
        if not isinstance(mappings, list):
            raise ValueError("DeviceHub profile 缺少 mappings 数组")
        bundles = raw.get("bundleIdentifiers")
        if not isinstance(bundles, (list, tuple)):
            bundles = []
        return cls(
            name=str(raw.get("name") or ""),
            mappings=tuple(item for item in mappings if isinstance(item, dict)),
            bundle_identifiers=tuple(
                value for value in bundles if isinstance(value, str)
            ),
            target_resolution=target,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "DeviceHubProfile":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("DeviceHub profile 根节点必须是对象")
        return cls.from_dict(raw)

    def _press_mappings(self):
        return (mapping for mapping in self.mappings if mapping.get("type") == "Press")

    def raw_key_for(self, key: str) -> str | None:
        """按 BetterGI 规范化键名查找 profile 中的原始键码。"""
        wanted = normalize_key(key)
        for mapping in self.mappings:
            for raw_key in _bound_codes(mapping.get("bind")):
                if normalize_key(raw_key) == wanted:
                    return raw_key
        return None

    def position_for(self, raw_key: str) -> tuple[float, float] | None:
        wanted = normalize_key(raw_key)
        for mapping in self._press_mappings():
            if any(normalize_key(code) == wanted for code in _bound_codes(mapping.get("bind"))):
                position = mapping.get("position")
                if isinstance(position, Mapping):
                    x, y = position.get("x"), position.get("y")
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        return float(x), float(y)
        return None

    def nearest_press_key(self, position: tuple[float, float]) -> str | None:
        """Find the profile key whose mapped touch point is nearest to a HUD point."""
        best: tuple[float, str] | None = None
        px, py = position
        for mapping in self._press_mappings():
            mapped = mapping.get("position")
            if not isinstance(mapped, Mapping):
                continue
            x, y = mapped.get("x"), mapped.get("y")
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                continue
            codes = _bound_codes(mapping.get("bind"))
            if not codes:
                continue
            distance = (float(x) - px) ** 2 + (float(y) - py) ** 2
            candidate = (distance, codes[0])
            if best is None or candidate[0] < best[0]:
                best = candidate
        return best[1] if best else None


@dataclass
class ControlLayout:
    joystick_center: tuple[float, float]
    joystick_radius_n: float
    camera_region: tuple[float, float, float, float]  # nx, ny, nw, nh
    buttons: dict[str, tuple[float, float]]
    key_map: dict[str, dict]
    devicehub_profile: DeviceHubProfile | None = None

    @classmethod
    def load(cls, path: str | Path = DEFAULT_LAYOUT,
             devicehub_profile: DeviceHubProfile | Mapping[str, Any] | None = None
             ) -> "ControlLayout":
        layout_path = Path(path)
        raw = json.loads(layout_path.read_text(encoding="utf-8"))
        base_name = raw.get("extends")
        if isinstance(base_name, str) and base_name:
            base_path = (layout_path.parent / base_name).resolve()
            if not base_path.is_relative_to(layout_path.parent.resolve()):
                raise ValueError("布局 extends 不得越出 config/controls 目录")
            base = json.loads(base_path.read_text(encoding="utf-8"))
            if not isinstance(base, dict):
                raise ValueError("布局基文件根节点必须是对象")
            raw = _merge_layout(base, {k: v for k, v in raw.items() if k != "extends"})
        cam = raw["camera"]["region"]
        if isinstance(devicehub_profile, Mapping):
            devicehub_profile = DeviceHubProfile.from_dict(devicehub_profile)
        return cls(
            joystick_center=(raw["joystick"]["center"]["nx"], raw["joystick"]["center"]["ny"]),
            joystick_radius_n=raw["joystick"]["radiusN"],
            camera_region=(cam["nx"], cam["ny"], cam["nw"], cam["nh"]),
            buttons={name: (p["nx"], p["ny"]) for name, p in raw["buttons"].items()},
            key_map={normalize_key(k): v for k, v in raw["keyMap"].items()},
            devicehub_profile=devicehub_profile,
        )

    def binding(self, key: str) -> dict | None:
        return self.key_map.get(normalize_key(key))

    def profile_key(self, key: str) -> str | None:
        """Map a canonical BetterGI key to the profile's browser key code."""
        profile = self.devicehub_profile
        if profile is None:
            return None
        binding = self.binding(key)
        if binding is not None and binding.get("profileCode"):
            raw = profile.raw_key_for(str(binding["profileCode"]))
            if raw is not None:
                return raw
        return profile.raw_key_for(key)

    def profile_key_for_button(self, name: str) -> str | None:
        profile = self.devicehub_profile
        position = self.buttons.get(name)
        if profile is None or position is None:
            return None
        # Prefer the explicit BetterGI semantic binding. Coordinate matching is
        # only a fallback for buttons such as attack that have no key binding.
        for key, binding in self.key_map.items():
            if binding.get("type") == "button" and binding.get("button") == name:
                raw = self.profile_key(key)
                if raw is not None:
                    return raw
        return profile.nearest_press_key(position)
