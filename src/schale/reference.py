"""SchaleDB reference catalogs: definitions, not a player's observed collection."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .data_control import get_json
from .cache import Mode

Dataset = Literal["students", "items", "equipment", "stages", "furniture", "groups"]
Region = Literal["kr", "jp", "cn", "en"]


class ReferenceCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Dataset
    region: Region
    source: str
    sha256: str
    entries: dict[str, dict[str, Any]]

    def get(self, entity_id: int) -> dict[str, Any]:
        """Return a copy so consumers cannot silently mutate the reference snapshot."""
        if str(entity_id) not in self.entries:
            raise KeyError(f"Unknown {self.dataset} ID: {entity_id}")
        return json.loads(json.dumps(self.entries[str(entity_id)]))

    @classmethod
    def from_file(
        cls, path: Path, *, dataset: Dataset = "students", region: Region = "kr"
    ):
        entries = json.loads(path.read_bytes())
        raw = json.dumps(
            entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return cls(
            dataset=dataset,
            region=region,
            source=str(path.resolve()),
            sha256=hashlib.sha256(raw).hexdigest(),
            entries=entries,
        )

    @classmethod
    def fetch(
        cls,
        dataset: Dataset = "students",
        *,
        region: Region = "kr",
        refresh: bool = False,
        cache_mode: Mode | None = None,
    ):
        # Validate before constructing a URL, including calls from untyped clients.
        if dataset not in {
            "students",
            "items",
            "equipment",
            "stages",
            "furniture",
            "groups",
        } or region not in {"kr", "jp", "cn", "en"}:
            raise ValueError("Unsupported SchaleDB dataset or region")
        url = f"https://schaledb.com/data/{'' if dataset == 'groups' else region + '/'}{dataset}.min.json"
        entries = (
            get_json(url, force_refresh=refresh, mode=cache_mode)
            if cache_mode
            else get_json(url, force_refresh=refresh)
        )
        raw = json.dumps(
            entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return cls.model_validate(
            dict(
                dataset=dataset,
                region=region,
                source=url,
                sha256=hashlib.sha256(raw).hexdigest(),
                entries=entries,
            )
        )
