"""Shared reference JSON cache, including adoption of pre-release cache files."""

import os
from pathlib import Path

from .cache import Mode, shared_cache

CACHE_TTL_SECONDS = max(
    0, int(os.environ.get("SCHALE_CACHE_TTL_SECONDS", str(7 * 86400)))
)


def get_json(
    url: str,
    *,
    force_refresh: bool = False,
    ttl_seconds: float | None = None,
    mode: Mode | None = None,
) -> dict[str, object]:
    cache = shared_cache()
    filename = f"{url.replace('/', '_').replace(':', '_')}.json"
    legacy = cache.root / filename
    packaged = Path(__file__).with_name("cache") / filename
    if ttl_seconds is None and "SCHALE_CACHE_TTL_SECONDS" in os.environ:
        ttl_seconds = CACHE_TTL_SECONDS
    force = force_refresh or os.environ.get(
        "SCHALE_FORCE_CACHE_REFRESH", ""
    ).lower() in {"1", "true", "yes"}
    resource = cache.fetch(
        url,
        kind="json",
        mode="refresh" if force else mode,
        ttl=ttl_seconds,
        seed=legacy if legacy.exists() else packaged,
    )
    return resource.json()
