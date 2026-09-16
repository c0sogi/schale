"""Inventory recognition with catalog icon verification and shared ONNX text."""

from pathlib import Path
from typing import cast

from schale import literal
from schale.cache import cache_directory
from schale.assets import resolve_resources
from schale.schema.scanner import CellEvidence, ScannedItem, ScanResult

from ._grid import detect_grid
from ._icons import IconAtlas
from ._identity import InventoryMatcher
from ._ocr import InventoryTextReader
from ._preprocessing import ImageSource, load_image

_atlas: IconAtlas | None = None
_atlas_root: tuple[Path, Path] | None = None
_matcher: InventoryMatcher | None = None


def _get_atlas(*, force_download: bool = False) -> IconAtlas:
    global _atlas, _atlas_root, _matcher
    root = (cache_directory(), resolve_resources("inventory"))
    if _atlas is None or force_download or root != _atlas_root:
        _atlas = IconAtlas()
        _atlas.prepare(force_download=force_download)
        _atlas_root = root
        _matcher = None
    return _atlas


def scan_inventory(
    image: ImageSource,
    *,
    confidence_threshold: float = 0.65,
    force_icon_download: bool = False,
    numeric_model: Path | None = None,
) -> ScanResult:
    """Read a screenshot without Torch; retain uncertainty and all cell evidence.

    Scores are matching scores, not calibrated probabilities. ``quantity`` is
    None for failed readings and K-abbreviated displays; no count is invented.
    """
    global _matcher
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    source = load_image(image)
    cells = detect_grid(source)
    reader = InventoryTextReader(numeric_model)
    regions = [cell.extract_roi(source) for cell in cells]
    text = reader.read_cells(regions)
    atlas = _get_atlas(force_download=force_icon_download)
    if _matcher is None:
        _matcher = InventoryMatcher(atlas)
    items = []
    observations = []
    unrecognized = []
    for cell, region, reading in zip(cells, regions, text, strict=True):
        candidates = _matcher.rank(region, reading["tier"])
        best = candidates[0] if candidates else None
        margin = (
            best["score"] - candidates[1]["score"]
            if best and len(candidates) > 1
            else 1.0
        )
        accepted = bool(
            best and best["score"] >= confidence_threshold and margin >= 0.04
        )
        issues = []
        if not accepted:
            issues.append("ambiguous_icon")
        template = atlas._templates[best["icon_name"]] if best else None
        if template and template.tier > 0 and reading["tier"] != template.tier:
            accepted = False
            issues.append("unreadable_or_conflicting_tier")
        quantity = reading["quantity"]
        if quantity.status != "exact":
            issues.append(
                "abbreviated_quantity"
                if quantity.status == "abbreviated"
                else "unreadable_quantity"
            )
        position = (cell.row, cell.col)
        if accepted and template and best:
            item = ScannedItem(
                equipment_id=template.equipment_id,
                category=cast(literal.EquipmentCategory, template.category),
                tier=template.tier,
                quantity=quantity.value,
                quantity_status=quantity.status,
                quantity_text=reading["quantity_reading"].text,
                displayed_quantity=quantity.displayed_value,
                quantity_confidence=reading["quantity_reading"].score,
                is_blueprint=template.is_blueprint,
                icon_name=template.icon_name,
                confidence=best["score"],
                grid_position=position,
            )
            items.append(item)
        else:
            unrecognized.append(position)

        def absolute(box):
            x0, y0, x1, y1 = box
            return (cell.x + x0, cell.y + y0, cell.x + x1, cell.y + y1)

        observations.append(
            CellEvidence(
                grid_position=position,
                box=(cell.x, cell.y, cell.x + cell.width, cell.y + cell.height),
                tier_box=absolute(reading["tier_box"]),
                quantity_box=absolute(reading["quantity_box"]),
                tier_text=reading["tier_reading"].text,
                tier_score=reading["tier_reading"].score,
                tier_views=reading.get("tier_views", []),
                quantity_text=reading["quantity_reading"].text,
                quantity_score=reading["quantity_reading"].score,
                quantity_views=reading.get("quantity_views", []),
                tier_value=reading["tier"],
                quantity_value=quantity.value,
                quantity_status=quantity.status,
                displayed_quantity=quantity.displayed_value,
                equipment_id=template.equipment_id if accepted and template else None,
                candidates=candidates[:3],
                issues=issues,
            )
        )
    return ScanResult(
        items=items,
        unrecognized_cells=unrecognized,
        grid_dimensions=(max(c.row for c in cells) + 1, max(c.col for c in cells) + 1),
        source_resolution=source.shape[:2],
        observations=observations,
        unavailable_icons=getattr(atlas, "unavailable_icons", []),
    )


__all__ = ["ScanResult", "ScannedItem", "scan_inventory"]
