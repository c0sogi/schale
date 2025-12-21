import json
import logging
import os
import shutil
import time
from importlib import resources
from pathlib import Path
from typing import Any, Optional, cast

import requests

logger = logging.getLogger(__name__)


_DEFAULT_CACHE_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days by default
_CACHE_DIR = Path(
    os.environ.get("SCHALE_CACHE_DIR") or (Path.home() / ".schale" / "cache")
)
_PACKAGE_CACHE_DIR = Path(__file__).with_name("cache")
_FORCE_REFRESH_REQUESTED = os.environ.get("SCHALE_FORCE_CACHE_REFRESH", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _load_cache_ttl_from_env() -> int:
    raw_value = os.environ.get("SCHALE_CACHE_TTL_SECONDS")
    if raw_value is None:
        return _DEFAULT_CACHE_TTL_SECONDS
    try:
        return max(0, int(raw_value))
    except ValueError:
        logger.warning(
            "Cannot parse environment variable SCHALE_CACHE_TTL_SECONDS=%s as an integer. Using %s seconds.",
            raw_value,
            _DEFAULT_CACHE_TTL_SECONDS,
        )
        return _DEFAULT_CACHE_TTL_SECONDS


CACHE_TTL_SECONDS = _load_cache_ttl_from_env()


def _url_to_filename(url: str) -> str:
    return f"{url.replace('/', '_').replace(':', '_')}.json"


def _cache_file_path(filename: str) -> Path:
    return _CACHE_DIR / filename


def _needs_disk_refresh(path: Path, ttl_seconds: int) -> bool:
    if not path.exists():
        logger.debug("Cache file %s does not exist", path)
        return True
    if ttl_seconds <= 0:
        logger.debug("TTL is 0, forcing refresh")
        return True
    try:
        age = time.time() - path.stat().st_mtime
        logger.debug("Cache age: %d seconds, TTL: %d seconds", age, ttl_seconds)
    except OSError as exc:
        logger.debug("Failed to get cache age for %s: %s", path, exc)
        return True
    return age >= ttl_seconds


def _read_json_from_path(path: Path) -> Optional[dict[str, object]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read cache file %s: %s", path, exc)
        return None
    if isinstance(loaded, dict):
        typed_loaded: dict[Any, object] = cast(dict[Any, object], loaded)
        return {str(k): v for k, v in typed_loaded.items()}
    logger.warning("Cache file %s is not a dict", path)
    return None


def _load_seed_payload(filename: str) -> Optional[dict[str, object]]:
    local_seed = _PACKAGE_CACHE_DIR / filename
    payload = _read_json_from_path(local_seed)
    if payload is not None:
        return payload
    try:
        seed = resources.files("schale").joinpath("cache", filename)
    except ModuleNotFoundError:
        return None
    if not seed.is_file():
        return None
    with resources.as_file(seed) as local_path:
        return _read_json_from_path(local_path)


def _copy_seed_to_cache(filename: str) -> None:
    cache_path = _cache_file_path(filename)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    local_seed = _PACKAGE_CACHE_DIR / filename
    if local_seed.exists():
        try:
            shutil.copy2(local_seed, cache_path)
        except OSError as exc:
            logger.debug("Failed to copy initial cache to %s: %s", cache_path, exc)
        return

    try:
        seed = resources.files("schale").joinpath("cache", filename)
    except ModuleNotFoundError:
        return
    if not seed.is_file():
        return
    with resources.as_file(seed) as local_path:
        try:
            shutil.copy2(local_path, cache_path)
        except OSError as exc:
            logger.debug("Failed to copy initial cache to %s: %s", cache_path, exc)


def _download_and_store(url: str, destination: Path) -> bool:
    try:
        logger.debug("Downloading %s", url)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.info("Failed to refresh remote %s: %s", url, exc)
        return False

    payload = response.json()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.debug("Saving cache to %s", destination)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
    except OSError as exc:
        logger.info("Failed to save cache file %s: %s", destination, exc)
        return False
    return True


def get_json(
    url: str, *, force_refresh: bool = False, ttl_seconds: Optional[int] = None
) -> dict[str, object]:
    """Fetch JSON with packaged seeds + on-disk caching."""

    filename = _url_to_filename(url)
    cache_path = _cache_file_path(filename)
    logger.debug("Cache path: %s", cache_path)
    ttl = CACHE_TTL_SECONDS if ttl_seconds is None else max(0, ttl_seconds)
    effective_force_refresh = force_refresh or _FORCE_REFRESH_REQUESTED

    if effective_force_refresh or _needs_disk_refresh(cache_path, ttl):
        _download_and_store(url, cache_path)

    payload = _read_json_from_path(cache_path)
    if payload is not None:
        return payload

    seed_payload = _load_seed_payload(filename)
    if seed_payload is not None:
        _copy_seed_to_cache(filename)
        return seed_payload

    raise RuntimeError(f"Cannot load cache for {url}")
