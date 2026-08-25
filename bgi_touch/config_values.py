"""Shared coercion helpers for BetterGI-compatible configuration values."""

from __future__ import annotations

from numbers import Real
from typing import Any


_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "是"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", "否", ""})


def as_bool(value: Any, default: bool = False) -> bool:
    """Decode JSON/API boolean variants without treating ``"false"`` as true.

    BetterGI configuration is commonly written by a JavaScript or .NET UI,
    so the same field may arrive as a JSON boolean, ``0``/``1`` or a string.
    Unknown values keep the caller's default instead of silently enabling a
    feature because Python considers every non-empty string truthy.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, Real):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    return default
