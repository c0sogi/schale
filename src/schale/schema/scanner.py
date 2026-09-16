"""Pydantic models for inventory scanner results."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from schale import literal


class ScannedItem(BaseModel):
    """A single recognized equipment item from an inventory screenshot."""

    equipment_id: int
    """The equipment ID from schaledb (e.g. 1005 for Hat T6, 101005 for its blueprint)."""

    category: literal.EquipmentCategory
    """The equipment category (Exp, WeaponExpGrowth variants, or wearables like Hat, Gloves, Shoes, Bag, Badge, Hairpin, Charm, Necklace, Watch)."""

    tier: int
    """The tier number (1-10)."""

    quantity: int | None
    """Exact displayed count, or None for unreadable/abbreviated quantities."""

    quantity_status: Literal["exact", "abbreviated", "unreadable"] = "exact"
    quantity_text: str = ""
    displayed_quantity: int | None = None
    quantity_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    is_blueprint: bool
    """True if this is a blueprint (piece) item."""

    icon_name: str
    """The schaledb icon name matched (e.g. 'equipment_icon_hat_tier5')."""

    confidence: float
    """Template matching confidence score (0.0 to 1.0)."""

    grid_position: tuple[int, int]
    """(row, col) position in the detected grid, 0-indexed."""

    @model_validator(mode="after")
    def _validate_ranges(self) -> "ScannedItem":
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
        if not 0 <= self.tier <= 10:
            raise ValueError(f"tier must be between 0 and 10, got {self.tier}")
        if self.quantity is not None and self.quantity < 0:
            raise ValueError(f"quantity must be non-negative, got {self.quantity}")
        if (self.quantity_status == "exact") != (self.quantity is not None):
            raise ValueError("Only exact quantities may carry a quantity value")
        return self


class CellEvidence(BaseModel):
    grid_position: tuple[int, int]
    box: tuple[int, int, int, int]
    tier_box: tuple[int, int, int, int]
    quantity_box: tuple[int, int, int, int]
    tier_text: str
    tier_score: float
    tier_views: list[dict] = Field(default_factory=list)
    quantity_text: str
    quantity_score: float
    quantity_views: list[dict] = Field(default_factory=list)
    tier_value: int | None
    quantity_value: int | None
    quantity_status: Literal["exact", "abbreviated", "unreadable"]
    displayed_quantity: int | None
    equipment_id: int | None
    candidates: list[dict]
    issues: list[str]


class ScanResult(BaseModel):
    """Complete result from scanning an inventory screenshot."""

    items: list[ScannedItem]
    """All recognized equipment items, ordered by grid position (top-left to bottom-right)."""

    unrecognized_cells: list[tuple[int, int]]
    """Grid positions (row, col) of cells that could not be identified as equipment."""

    grid_dimensions: tuple[int, int]
    """(rows, cols) of the detected grid."""

    source_resolution: tuple[int, int]
    """(height, width) of the input image."""

    observations: list[CellEvidence] = Field(default_factory=list)
    """Every detected cell, including rejected identity and unreadable text."""
    unavailable_icons: list[str] = Field(default_factory=list)
    """Catalog references missing from this run; those items are not validated."""
