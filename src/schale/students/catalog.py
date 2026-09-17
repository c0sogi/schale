"""Pin student metadata alongside each extraction for reproducibility."""

import hashlib
import json
from pathlib import Path
from typing import Callable

from ..cache import Mode, atomic_write, shared_cache
from ..data_control import get_json

URL = "https://schaledb.com/data/kr/students.min.json"


def load_catalog(path: Path) -> tuple[dict, str]:
    if path.exists():
        raw = path.read_bytes()
    else:
        data = get_json(URL)
        raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()
        if not isinstance(data, dict) or not all(
            isinstance(x, dict) and "Name" in x and "Id" in x for x in data.values()
        ):
            raise ValueError("Student catalog has an unsupported structure")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, raw)
    data = json.loads(raw)
    if (
        not isinstance(data, dict)
        or not data
        or not all("Name" in x and "Id" in x for x in data.values())
    ):
        raise ValueError("Student catalog has an unsupported structure")
    return data, hashlib.sha256(raw).hexdigest()


SKILL_KEYS = ("Ex", "Public", "Passive", "ExtraPassive")


def load_skill_assets(
    catalog: dict,
    directory: Path,
    *,
    refresh: bool = False,
    cache_mode: Mode | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Cache the actual game symbols; no learned OCR model is used."""
    directory.mkdir(parents=True, exist_ok=True)
    names = sorted(
        {
            s["Icon"]
            for row in catalog.values()
            for k in SKILL_KEYS
            if (s := row.get("Skills", {}).get(k)) and "Icon" in s
        }
    )

    def fetch(name: str) -> tuple[str, Path]:
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Unsafe asset name: {name}")
        resource = shared_cache().fetch(
            f"https://schaledb.com/images/skill/{name}.webp",
            kind="image",
            seed=directory / f"{name}.webp",
            mode="refresh" if refresh else cache_mode,
        )
        return name, resource.path

    result = {}
    for index, name in enumerate(names, 1):
        key, path = fetch(name)
        result[key] = path
        if progress is not None and (index % 20 == 0 or index == len(names)):
            progress(f"Skill reference cache: {index}/{len(names)}")
    return result
