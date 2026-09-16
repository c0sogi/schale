"""SchaleDB reference icons in the verified shared HTTP cache."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from schale.cache import CacheError, shared_cache
from schale.cache_control import cache_collection
from schale.assets import resolve_resources

if TYPE_CHECKING:
    from ._cnn import CNNClassifier

logger = logging.getLogger(__name__)
TEMPLATE_CATEGORIES = frozenset(
    {
        "Hat",
        "Gloves",
        "Shoes",
        "Bag",
        "Badge",
        "Hairpin",
        "Charm",
        "Watch",
        "Necklace",
        "Exp",
        "WeaponExpGrowthA",
        "WeaponExpGrowthB",
        "WeaponExpGrowthC",
        "WeaponExpGrowthZ",
    }
)
_ICON_BASE_URL = "https://schaledb.com/images/equipment/icon"


@dataclass(frozen=True)
class IconTemplate:
    equipment_id: int
    icon_name: str
    category: str
    tier: int
    is_blueprint: bool
    path: Path


def _get_local_icon_path(icon_name: str) -> Path | None:
    try:
        package_data = resolve_resources("inventory", verify=False) / "icons"
    except ValueError:
        return None
    for extension in (".webp", ".png"):
        path = package_data / f"{icon_name}{extension}"
        if path.is_file():
            return Path(str(path))
    return None


def _download_icon(icon_name: str, *, _force: bool = False) -> Path | None:
    if not icon_name.replace("_", "").isalnum():
        raise ValueError(f"Unsafe icon name: {icon_name}")
    seed = _get_local_icon_path(icon_name)
    suffix = seed.suffix if seed else ".webp"
    try:
        return (
            shared_cache()
            .fetch(
                f"{_ICON_BASE_URL}/{icon_name}{suffix}",
                kind="image",
                seed=seed,
                mode="refresh" if _force else None,
            )
            .path
        )
    except CacheError as error:
        logger.warning("Icon unavailable: %s", error)
        return None


@dataclass
class IconAtlas:
    _templates: dict[str, IconTemplate] = field(default_factory=dict, init=False)
    _prepared: bool = field(default=False, init=False)
    _cnn: "CNNClassifier | None" = field(default=None, init=False)
    unavailable_icons: list[str] = field(default_factory=list, init=False)

    def prepare(self, *, force_download: bool = False) -> None:
        if self._prepared and not force_download:
            return
        self._templates.clear()
        self.unavailable_icons.clear()
        for equipment_id, equipment in cache_collection.equipments.items():
            if equipment.Category not in TEMPLATE_CATEGORIES:
                continue
            path = _download_icon(equipment.Icon, _force=force_download)
            if path is None:
                self.unavailable_icons.append(equipment.Icon)
                continue
            self._templates[equipment.Icon] = IconTemplate(
                equipment_id,
                equipment.Icon,
                equipment.Category,
                equipment.Tier,
                equipment.Icon.endswith("_piece"),
                path,
            )
        self._prepared = True

    def enable_cnn(
        self, model_path: Path | None = None, mapping_path: Path | None = None
    ) -> None:
        from ._cnn import CNNClassifier

        self._cnn = CNNClassifier(model_path, mapping_path)
