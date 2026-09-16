"""Inventory text regions and strict value parsing over the shared ONNX reader."""

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import re

import cv2
import numpy as np

from schale.students.model_bundle import validate_bundle
from schale.students.paths import model_directory
from schale.vision.text import CtcRecognizer, TextReading


@dataclass(frozen=True)
class Quantity:
    value: int | None
    displayed_value: int | None
    status: str


def parse_quantity(text: str) -> Quantity:
    text = text.strip()
    # Never drop unexpected characters or infer an exact count from an abbreviation.
    match = re.fullmatch(r"[xX×]?(0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)", text)
    if match:
        value = int(match[1].replace(",", ""))
        return Quantity(value, value, "exact")
    match = re.fullmatch(r"[xX×]?([1-9][0-9]*(?:\.[0-9]{1,3})?)[kK]", text)
    if match:
        return Quantity(None, int(Decimal(match[1]) * 1000), "abbreviated")
    return Quantity(None, None, "unreadable")


def parse_tier(text: str) -> int | None:
    match = re.fullmatch(r"[Tt](10|[1-9])", text.strip())
    return int(match[1]) if match else None


def text_boxes(cell):
    """Pixel boxes relative to a cell, keeping the tier badge outside quantity."""
    h, w = cell.shape[:2]
    y0 = int(h * 0.68)
    left = cell[y0:, : int(w * 0.44)]
    hsv = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((90, 80, 70)), np.array((135, 255, 255)))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    parts = [
        s
        for s in stats[1:]
        if s[4] >= max(3, w * h * 0.0003)
        and s[3] >= h * 0.04
        and s[4] / (s[2] * s[3]) > 0.3
    ]
    badge = bool(parts) and sum(int(s[4]) for s in parts) >= h * w * 0.002
    if badge:
        x0 = max(0, min(int(s[0]) for s in parts) - 3)
        x1 = min(w, max(int(s[0] + s[2]) for s in parts) + 3)
        top = max(y0, y0 + min(int(s[1]) for s in parts) - 3)
        bottom = min(h, y0 + max(int(s[1] + s[3]) for s in parts) + 3)
        tier_box = (x0, top, x1, bottom)
        quantity_box = (x1, int(h * 0.72), w, h)
    else:
        tier_box = (0, y0, int(w * 0.44), h)
        quantity_box = (int(w * 0.20), int(h * 0.72), w, h)
    return tier_box, quantity_box, badge


class InventoryTextReader:
    def __init__(self, directory: Path | None = None):
        directory = model_directory(directory)
        if not (directory / "model.onnx").is_file():
            raise ValueError(
                "UI text model not installed. Run schale students install-model BUNDLE, or pass numeric_model."
            )
        metadata = validate_bundle(directory)
        self.recognizer = CtcRecognizer(directory / "model.onnx", metadata["charset"])
        # Inventory acceptance is separate from the student profile's calibration.
        self.threshold = 0.65

    def read_cells(self, cells: list) -> list[dict]:
        normalized = [
            cv2.resize(cell, (195, 158), interpolation=cv2.INTER_CUBIC)
            for cell in cells
        ]
        geometry = []
        tier_patches = []
        for cell in normalized:
            boxes = text_boxes(cell)
            geometry.append(boxes)
            x0, y0, x1, y1 = boxes[0]
            tier_patches.append(cell[y0:y1, x0:x1])
            tier_patches.append(cell[int(158 * 0.70) :, : int(195 * 0.44)])
        tier_views = [
            value_confidence(view, "tT") for view in self.recognizer.read(tier_patches)
        ]
        tiers = []
        for index in range(len(cells)):
            views = tier_views[index * 2 : index * 2 + 2]
            valid = [
                v
                for v in views
                if v.score >= self.threshold and parse_tier(v.text) is not None
            ]
            if valid and all(
                parse_tier(v.text) == parse_tier(valid[0].text) for v in valid
            ):
                tiers.append(max(valid, key=lambda v: v.score))
            elif valid:
                tiers.append(TextReading("conflict", 0.0))
            else:
                tiers.append(max(views, key=lambda v: v.score))
        patches = []
        quantity_boxes = []
        for cell, tier, (tier_box, quantity_box, badge) in zip(
            normalized, tiers, geometry, strict=True
        ):
            # Blue item art is not a tier badge. Do not crop off the start of a
            # long quantity unless an actual T-label has been decoded.
            if parse_tier(tier.text) is None:
                quantity_box = (int(195 * 0.20), int(158 * 0.72), 195, 158)
            quantity_boxes.append(quantity_box)
            x0, y0, x1, y1 = quantity_box
            patch = cell[y0:y1, x0:x1]
            patches.extend([patch, localize_quantity(patch)])
        readings = self.recognizer.read(patches)
        results = []
        for index, (tier_box, quantity_box, badge) in enumerate(geometry):
            tier = tiers[index]
            views = [
                value_confidence(view, "xX×")
                for view in readings[index * 2 : index * 2 + 2]
            ]
            valid = [
                (v, parse_quantity(v.text))
                for v in views
                if v.score >= self.threshold
                and parse_quantity(v.text).status != "unreadable"
            ]
            quantity = max(views, key=lambda v: v.score)
            parsed = Quantity(None, None, "unreadable")
            if valid and all(p == valid[0][1] for _, p in valid):
                quantity, parsed = max(valid, key=lambda pair: pair[0].score)
            quantity_box = quantity_boxes[index]

            def original_box(box):
                x0, y0, x1, y1 = box
                h, w = cells[index].shape[:2]
                return (
                    round(x0 * w / 195),
                    round(y0 * h / 158),
                    round(x1 * w / 195),
                    round(y1 * h / 158),
                )

            results.append(
                dict(
                    tier=parse_tier(tier.text)
                    if badge and tier.score >= self.threshold
                    else None,
                    tier_reading=tier if badge else TextReading("", 0),
                    quantity=parsed,
                    quantity_reading=quantity,
                    badge=badge,
                    tier_box=original_box(tier_box),
                    quantity_box=original_box(quantity_box),
                    quantity_views=[dict(text=v.text, score=v.score) for v in views],
                    tier_views=[
                        dict(text=v.text, score=v.score)
                        for v in tier_views[index * 2 : index * 2 + 2]
                    ],
                )
            )
        return results


def value_confidence(reading: TextReading, prefixes: str) -> TextReading:
    # A known decorative prefix is not part of the numeric value. Keep every
    # digit, decimal separator and multiplier in the acceptance score.
    scores = reading.character_scores
    if scores and reading.text and reading.text[0] in prefixes and len(scores) > 1:
        return TextReading(reading.text, min(scores[1:]), scores)
    return reading


def localize_quantity(patch):
    """Crop dark quantity strokes without dropping leading or trailing digits."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    white = float(np.percentile(hsv[:, :, 2], 95))
    mask = ((hsv[:, :, 1] < 130) & (hsv[:, :, 2] < white * 0.7)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    h, w = mask.shape
    parts = [s for s in stats[1:] if s[3] >= h * 0.2 and s[2] < w * 0.9 and s[4] >= 4]
    if not parts:
        return patch
    x0 = max(0, min(int(s[0]) for s in parts) - 3)
    y0 = max(0, min(int(s[1]) for s in parts) - 3)
    x1 = min(w, max(int(s[0] + s[2]) for s in parts) + 3)
    y1 = min(h, max(int(s[1] + s[3]) for s in parts) + 3)
    return patch[y0:y1, x0:x1]
