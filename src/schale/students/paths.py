"""Deterministic shared caches with explicit path overrides."""

from pathlib import Path

from ..cache import cache_directory as cache_directory


def model_directory(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    return cache_directory() / "student-numeric"


def asset_directory(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    return cache_directory() / "assets"
