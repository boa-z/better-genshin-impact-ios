"""Recognize the three inactive character labels on the mobile HUD.

The iOS party strip does not expose the PC-side slot index marker. BetterGI's
visible labels are nevertheless enough to map rows to physical slots once the
current active slot is known. An optional existing ``party_slots`` mapping
supplies the active character and is preserved while visible rows are refreshed.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .recognition import RecognitionObject


ROOT = Path(__file__).resolve().parents[2]
AVATAR_DATA_PATH = ROOT / "assets" / "data" / "combat_avatar.json"
PARTY_NAME_ROI = (1560, 120, 360, 480)
DEFAULT_ROW_Y = (0.212, 0.348, 0.457)


def _compact(text: object) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip()


@lru_cache(maxsize=1)
def _avatar_name_aliases() -> dict[str, str]:
    """Return OCR aliases → canonical names from the fixed upstream asset."""

    try:
        records = json.loads(AVATAR_DATA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    result: dict[str, str] = {}
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, Mapping):
            continue
        canonical = _compact(record.get("name"))
        if not canonical:
            continue
        aliases = record.get("alias") or ()
        if isinstance(aliases, str):
            aliases = (aliases,)
        for value in (canonical, *aliases):
            alias = _compact(value)
            if alias:
                result.setdefault(alias.casefold(), canonical)
    return result


def canonical_avatar_name(text: object) -> str | None:
    """Normalize an OCR label to BetterGI's canonical character name."""

    compact = _compact(text)
    return _avatar_name_aliases().get(compact.casefold())


def _row_centers(ctx: Any) -> tuple[float, float, float]:
    layout = getattr(ctx, "layout", None)
    buttons = getattr(layout, "buttons", {}) if layout is not None else {}
    result = []
    for index, fallback in enumerate(DEFAULT_ROW_Y, start=1):
        try:
            value = buttons.get(f"partyRow{index}", (0.96, fallback))[1]
            numeric = float(value)
            result.append(numeric * 1080 if 0 <= numeric <= 1.5 else numeric)
        except (AttributeError, IndexError, TypeError, ValueError):
            result.append(fallback * 1080)
    return tuple(result)  # type: ignore[return-value]


def _active_slot(ctx: Any) -> int:
    try:
        slot = int(getattr(getattr(ctx, "input", None), "_active_slot", 1))
    except (TypeError, ValueError):
        slot = 1
    return slot if 1 <= slot <= 4 else 1


def _existing_slots(existing: Mapping[str, int] | None) -> dict[int, str]:
    result: dict[int, str] = {}
    for raw_name, raw_slot in (existing or {}).items():
        try:
            slot = int(raw_slot)
        except (TypeError, ValueError):
            continue
        name = _compact(raw_name)
        if 1 <= slot <= 4 and name:
            result[slot] = name
    return result


def map_party_hits(
    hits: Iterable[Any],
    *,
    active_slot: int,
    row_centers: Iterable[float],
    existing: Mapping[str, int] | None = None,
    max_row_distance: float = 90.0,
) -> dict[str, int]:
    """Purely map OCR hits to physical slots; useful for frame-free tests."""

    centers = tuple(float(value) for value in row_centers)
    if len(centers) != 3:
        raise ValueError("party row centers must contain three values")
    try:
        active_slot = int(active_slot)
    except (TypeError, ValueError):
        active_slot = 1
    active_slot = active_slot if 1 <= active_slot <= 4 else 1
    slots = _existing_slots(existing)
    others = [slot for slot in (1, 2, 3, 4) if slot != active_slot]
    candidates: dict[int, tuple[float, str]] = {}
    for hit in hits:
        name = canonical_avatar_name(getattr(hit, "text", ""))
        if name is None:
            continue
        try:
            y = float(getattr(hit, "y"))
        except (AttributeError, TypeError, ValueError):
            continue
        row = min(range(3), key=lambda index: abs(y - centers[index]))
        distance = abs(y - centers[row])
        if distance > max_row_distance:
            continue
        old = candidates.get(others[row])
        if old is None or distance < old[0]:
            candidates[others[row]] = (distance, name)

    # OCR can return the same label twice (for example, once for the text and
    # once for its shadow). Keep the closest row only so one character cannot
    # occupy multiple physical slots.  The row number is a deterministic tie
    # breaker when two duplicate hits have the same distance.
    by_name: dict[str, list[tuple[float, int]]] = {}
    for slot, (distance, name) in candidates.items():
        by_name.setdefault(name, []).append((distance, slot))
    unique_candidates: dict[int, tuple[float, str]] = {}
    for name, values in by_name.items():
        distance, slot = min(values, key=lambda item: (item[0], item[1]))
        unique_candidates[slot] = (distance, name)

    # Visible labels are authoritative for their rows. Remove stale copies of
    # the same character from another slot before installing the new mapping.
    for slot, (_distance, name) in sorted(unique_candidates.items()):
        # The active character is not part of the mobile party strip. If an
        # OCR shadow happens to repeat it, keep the known active-slot mapping.
        if slots.get(active_slot) == name:
            continue
        for other_slot, other_name in list(slots.items()):
            if other_slot != slot and other_slot != active_slot and other_name == name:
                del slots[other_slot]
        slots[slot] = name
    return {
        name: slot for slot, name in sorted(slots.items())
        if name and 1 <= slot <= 4
    }


def recognize_party_slots(
    ctx: Any,
    region: Any | None = None,
    *,
    existing: Mapping[str, int] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """OCR the mobile party strip from a caller-owned frame/region."""

    try:
        region = region if region is not None else ctx.capture_region()
        hits = region.find_multi(
            RecognitionObject.ocr(*PARTY_NAME_ROI), limit=8,
        )
        result = map_party_hits(
            hits,
            active_slot=_active_slot(ctx),
            row_centers=_row_centers(ctx),
            existing=existing,
        )
        if result:
            log("[combat] HUD 队伍识别：" + ", ".join(
                f"{name}[{slot}]" for name, slot in sorted(
                    result.items(), key=lambda item: item[1]
                )
            ))
        return result
    except Exception as error:
        log(f"[combat] HUD 队伍 OCR 失败（保留已有配置）：{error}")
        return dict(existing or {})
