"""Shared cancellation primitives for the mobile pathing runner."""

from __future__ import annotations


class PathingCancelled(RuntimeError):
    """Signal that a pathing operation stopped at the caller's request."""

    bgi_cancelled = True
