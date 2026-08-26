"""BetterGI pathing ``control.json5`` merge support.

The desktop loader applies a sibling ``control.json5`` before deserializing a
route. Community route packs use it for shared map/trigger defaults, per-file
overrides, and small array additions, so ignoring it changes executable route
behavior rather than only metadata.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _strip_json5_comments(text: str) -> str:
    """Remove JSON5 comments without touching comment-looking string values."""
    result: list[str] = []
    index = 0
    in_string = False
    escaped = False
    length = len(text)
    while index < length:
        char = text[index]
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < length and text[index + 1] == "/":
            index += 2
            while index < length and text[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and index + 1 < length and text[index + 1] == "*":
            index += 2
            while index + 1 < length and text[index:index + 2] != "*/":
                if text[index] in "\r\n":
                    result.append(text[index])
                index += 1
            index = min(length, index + 2)
            continue
        result.append(char)
        index += 1
    return "".join(result)


def _load_json5(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    text = _strip_json5_comments(text)
    text = _TRAILING_COMMA.sub(r"\1", text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 配置根节点必须是对象：{path}")
    return value


def _control_object(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a control file, resolving the upstream ``ref`` indirection."""
    resolved = path.resolve()
    seen = set() if seen is None else set(seen)
    if resolved in seen:
        raise ValueError(f"control.json5 ref 形成循环：{resolved}")
    seen.add(resolved)
    value = _load_json5(resolved)
    ref = value.get("ref")
    if isinstance(ref, str) and ref.strip():
        target = (resolved.parent / ref).resolve()
        if target.is_dir():
            target = target / "control.json5"
        return _control_object(target, seen)
    return value


def _token_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _merge_arrays(source: list[Any], target: list[Any]) -> list[Any]:
    """Append source items to target while preserving order and de-duplicating."""
    result: list[Any] = []
    seen: set[str] = set()
    for item in [*target, *source]:
        key = _token_key(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(copy.deepcopy(item))
    return result


def _merge_object(control: dict[str, Any], target: dict[str, Any]) -> None:
    skip: set[str] = set()

    object_cover = control.get("_obj_cover")
    if isinstance(object_cover, list):
        for name in object_cover:
            key = str(name)
            if key in control:
                target[key] = copy.deepcopy(control[key])
                skip.add(key)
        skip.add("_obj_cover")

    array_add = control.get("_arr_add")
    if isinstance(array_add, list):
        for name in array_add:
            key = str(name)
            source = control.get(key)
            if not isinstance(source, list):
                continue
            existing = target.get(key)
            target[key] = _merge_arrays(source, existing if isinstance(existing, list) else [])
            skip.add(key)
        skip.add("_arr_add")

    for key, value in control.items():
        if key in skip:
            continue
        if isinstance(value, dict):
            existing = target.get(key)
            if not isinstance(existing, dict):
                target[key] = copy.deepcopy(value)
            else:
                _merge_object(value, existing)
        else:
            # Arrays replace by default, matching Newtonsoft JObject merge
            # behavior used by BetterGI's control loader.
            target[key] = copy.deepcopy(value)


def merge_pathing_mapping(
    route: dict[str, Any], control: dict[str, Any], name: str,
) -> dict[str, Any]:
    """Apply global and file-specific covers to one parsed route mapping."""
    result = copy.deepcopy(route)
    global_cover = control.get("global_cover")
    if isinstance(global_cover, dict):
        _merge_object(global_cover, result)
    entries = control.get("json_list")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict) or str(entry.get("name", "")) != name:
                continue
            cover = entry.get("cover")
            if isinstance(cover, dict):
                _merge_object(cover, result)
            break
    return result


def load_merged_pathing_mapping(path: str | Path) -> dict[str, Any]:
    """Load a route and apply a sibling ``control.json5`` when present."""
    source = Path(path).expanduser()
    route = _load_json5(source)
    control_path = source.parent / "control.json5"
    if not control_path.is_file():
        return route
    try:
        control = _control_object(control_path)
    except Exception:
        # BetterGI treats a broken optional control file as a non-merged route
        # so one bad shared override cannot make every route in the folder
        # unusable. The route JSON itself was already parsed above.
        return route
    return merge_pathing_mapping(route, control, source.stem)


__all__ = ["load_merged_pathing_mapping", "merge_pathing_mapping"]
