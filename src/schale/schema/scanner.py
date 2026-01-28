"""Pydantic models for inventory scanner results."""

from pydantic import BaseModel, model_validator

from schale import literal


class ScannedItem(BaseModel):
    """A single recognized equipment item from an inventory screenshot."""

    equipment_id: int
    """The equipment ID from schaledb (e.g. 1005 for Hat T6, 101005 for its blueprint)."""

    category: literal.EquipmentCategory
    """The equipment category (Exp, WeaponExpGrowth variants, or wearables like Hat, Gloves, Shoes, Bag, Badge, Hairpin, Charm, Necklace, Watch)."""

    tier: int
    """The tier number (1-10)."""

    quantity: int
    """The item count parsed from the 'x###' or 'x##K' text."""

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
        if self.quantity < 0:
            raise ValueError(f"quantity must be non-negative, got {self.quantity}")
        return self


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
