from __future__ import annotations

import time
from dataclasses import dataclass, field
from logging import getLogger
from typing import Callable, Generic, TypeVar

from schale import localization
from schale.data_control import CACHE_TTL_SECONDS, get_json
from schale.cache import shared_cache
from schale.schema.equipments import Equipment
from schale.schema.furniture import Furniture
from schale.schema.group import GroupEntry
from schale.schema.item import Item
from schale.schema.stages import Stage

logger = getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")

force_refresh: bool = False


@dataclass
class Cache(Generic[K, V]):
    update_callback: Callable[[], dict[K, V]]
    source_url: str | None = None

    _expiry: float | None = field(default=None, init=False, repr=False)
    _cache: dict[K, V] = field(default_factory=dict, init=False, repr=False)
    _generation: tuple[str, int] = field(default=("", -1), init=False, repr=False)

    @property
    def data(self) -> dict[K, V]:
        """gachagroup metadata"""

        if CACHE_TTL_SECONDS <= 0 or self._expiry is None:
            expired = True
        else:
            expired = self._expiry <= time.monotonic()

        if self.source_url and not force_refresh:
            # HTTP freshness is authoritative, including manual/offline policy.
            # Parsed objects can stay in memory until the shared store changes.
            get_json(self.source_url)
            cache = shared_cache()
            expired = self._generation != (str(cache.root), cache.generation())
        if expired or force_refresh:
            self.refresh()
        return self._cache

    def refresh(self) -> None:
        new = self.update_callback()
        self._cache.clear()
        self._cache.update(new)
        cache = shared_cache()
        self._generation = (str(cache.root), cache.generation())

        if CACHE_TTL_SECONDS <= 0:
            self._expiry = None
        else:
            self._expiry = time.monotonic() + CACHE_TTL_SECONDS


class CacheCollection:
    _groups: Cache[int, GroupEntry] = Cache(
        source_url=localization.GROUPS_URL,
        update_callback=lambda: {
            int(k): GroupEntry.model_validate(v)
            for k, v in get_json(
                localization.GROUPS_URL, force_refresh=force_refresh
            ).items()
        },
    )
    _stages: Cache[int, Stage] = Cache(
        source_url=localization.STAGES_URL,
        update_callback=lambda: {
            int(k): Stage.model_validate(v)
            for k, v in get_json(
                localization.STAGES_URL, force_refresh=force_refresh
            ).items()
        },
    )
    _items: Cache[int, Item] = Cache(
        source_url=localization.ITEM_METADATA_URL,
        update_callback=lambda: {
            int(k): Item.model_validate(v)
            for k, v in get_json(
                localization.ITEM_METADATA_URL, force_refresh=force_refresh
            ).items()
        },
    )
    _equipments: Cache[int, Equipment] = Cache(
        source_url=localization.EQUIPMENT_METADATA_URL,
        update_callback=lambda: {
            int(k): Equipment.model_validate(v)
            for k, v in get_json(
                localization.EQUIPMENT_METADATA_URL, force_refresh=force_refresh
            ).items()
        },
    )
    _furnitures: Cache[int, Furniture] = Cache(
        source_url=localization.FURNITURE_METADATA_URL,
        update_callback=lambda: {
            int(k): Furniture.model_validate(v)
            for k, v in get_json(
                localization.FURNITURE_METADATA_URL, force_refresh=force_refresh
            ).items()
        },
    )

    @property
    def groups(self) -> dict[int, GroupEntry]:
        return self._groups.data

    @property
    def stages(self) -> dict[int, Stage]:
        return self._stages.data

    @property
    def items(self) -> dict[int, Item]:
        return self._items.data

    @property
    def equipments(self) -> dict[int, Equipment]:
        return self._equipments.data

    @property
    def furnitures(self) -> dict[int, Furniture]:
        return self._furnitures.data

    def refresh_all(self) -> None:
        self._groups.refresh()
        self._stages.refresh()
        self._items.refresh()
        self._equipments.refresh()
        self._furnitures.refresh()


cache_collection = CacheCollection()
