"""Closed-set visual labels calibrated from explicitly annotated game crops."""

import json
from pathlib import Path

import cv2
import numpy as np

from .video import crop

BOXES = {
    "bond": (0.030, 0.776, 0.057, 0.817),
    "level": (0.025, 0.822, 0.072, 0.86),
    "ex": (0.540, 0.560, 0.592, 0.587),
    "basic": (0.620, 0.560, 0.672, 0.587),
    "passive": (0.700, 0.560, 0.752, 0.587),
    "sub": (0.785, 0.560, 0.837, 0.587),
    "weapon_level": (0.605, 0.630, 0.655, 0.662),
    "gear": (0.745, 0.850, 0.785, 0.882),
}
for _i, _x in enumerate((0.520, 0.594, 0.668), 1):
    BOXES[f"equipment{_i}"] = (_x, 0.850, _x + 0.045, 0.882)
    BOXES[f"equipment{_i}_level"] = (_x + 0.013, 0.775, _x + 0.064, 0.800)


def family(field: str) -> str:
    if field in ("ex", "basic", "passive", "sub"):
        return "skill"
    if (
        field.startswith("equipment")
        and not field.endswith("_level")
        or field == "gear"
    ):
        return "tier"
    if field == "bond":
        return "bond"
    return "level"


def label_mask(image, field: str):
    roi = crop(image, BOXES[field])
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    kind = family(field)
    if kind == "bond":
        mask = (hsv[:, :, 1] < 170) & (hsv[:, :, 2] < 160)
    elif kind == "skill":
        mask = hsv[:, :, 2] < 150
    elif kind == "tier":
        mask = (
            (hsv[:, :, 0] > 90)
            & (hsv[:, :, 0] < 125)
            & (hsv[:, :, 1] > 95)
            & (hsv[:, :, 2] < 235)
        )
    else:
        mask = (hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 200)
    mask = mask.astype(np.uint8) * 255
    if kind == "bond":
        # Thin heart-outline antialiasing is not part of the digit label.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    _, _, components, _ = cv2.connectedComponentsWithStats(mask)
    candidates = [
        c
        for c in components[1:]
        if 12 <= c[3] <= 48
        and c[4] >= (50 if kind == "bond" else 15)
        and c[2] < roi.shape[1] * 0.95
        and (kind != "bond" or c[4] / (c[2] * c[3]) > 0.25)
    ]
    if not candidates:
        return None
    x = min(c[0] for c in candidates)
    y = min(c[1] for c in candidates)
    x1 = max(c[0] + c[2] for c in candidates)
    y1 = max(c[1] + c[3] for c in candidates)
    if y1 - y > 48:
        return None
    binary = mask[y:y1, x:x1]
    # Preserve aspect ratio; stretching Lv.1 to Lv.90's width hides a real distinction.
    height = 32
    width = round(binary.shape[1] * height / binary.shape[0])
    if width > 150:
        return None
    canvas = np.zeros((40, 160), np.uint8)
    resized = cv2.resize(binary, (width, height), interpolation=cv2.INTER_AREA)
    canvas[4:36, (160 - width) // 2 : (160 - width) // 2 + width] = resized
    return canvas


def similarity(left, right) -> float:
    # A small translation allowance accounts for compression changing glyph bounds.
    padded = cv2.copyMakeBorder(left, 2, 2, 3, 3, cv2.BORDER_CONSTANT)
    return float(
        cv2.minMaxLoc(cv2.matchTemplate(padded, right, cv2.TM_CCOEFF_NORMED))[1]
    )


def digit_similarity(left, right) -> float:
    """Qualify Lv templates by each digit, not the shared prefix/background."""

    def parts(mask):
        _, _, stats, _ = cv2.connectedComponentsWithStats((mask > 100).astype(np.uint8))
        letters = sorted(
            (s for s in stats[1:] if s[3] >= 12 and s[4] >= 25),
            key=lambda s: s[0],
        )
        # Small punctuation has already been excluded. The first two glyphs
        # are L and v. A merged/unsegmentable label cannot qualify this check.
        return [mask[s[1] : s[1] + s[3], s[0] : s[0] + s[2]] for s in letters[2:]]

    a, b = parts(left), parts(right)
    if not a or len(a) != len(b):
        return 0.0
    scores = []
    for first, second in zip(a, b, strict=True):
        normalized = []
        for glyph in (first, second):
            width = round(glyph.shape[1] * 32 / glyph.shape[0])
            if width > 76:
                return 0.0
            canvas = np.zeros((40, 80), np.uint8)
            x = (80 - width) // 2
            canvas[4:36, x : x + width] = cv2.resize(glyph, (width, 32))
            normalized.append(canvas)
        scores.append(similarity(*normalized))
    return min(scores)


def clean_tier(mask):
    """Discard the thin capsule border before comparing actual T/digit shapes."""
    binary = (mask > 170).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    _, components, stats, _ = cv2.connectedComponentsWithStats(binary)
    accepted = [
        i
        for i, s in enumerate(stats)
        if i and s[3] >= 12 and s[4] >= 25 and s[4] / (s[2] * s[3]) > 0.3
    ]
    if not accepted:
        return mask
    binary = np.isin(components, accepted).astype(np.uint8) * 255
    y, x = np.where(binary)
    tight = binary[y.min() : y.max() + 1, x.min() : x.max() + 1]
    width = round(tight.shape[1] * 32 / tight.shape[0])
    if width > 160:
        return mask
    canvas = np.zeros((40, 160), np.uint8)
    canvas[4:36, (160 - width) // 2 : (160 - width) // 2 + width] = cv2.resize(
        tight, (width, 32)
    )
    return canvas


class LabelBank:
    def __init__(self, directory: Path):
        self.rows = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
        self.templates = [
            (
                row,
                cv2.imdecode(
                    np.frombuffer((directory / row["image"]).read_bytes(), np.uint8), 0
                ),
            )
            for row in self.rows
        ]
        self.templates = [
            (row, clean_tier(template) if row["family"] == "tier" else template)
            for row, template in self.templates
        ]

    def read(self, image, field: str):
        mask = label_mask(image, field)
        if mask is None:
            return None, 0.0, "no label"
        if family(field) == "tier":
            mask = clean_tier(mask)
        values: dict[str, float] = {}
        for row, template in self.templates:
            if row["family"] == family(field):
                value = str(row["value"])
                score = similarity(mask, template)
                if row["family"] == "level" and score >= 0.90:
                    score = min(score, digit_similarity(mask, template))
                values[value] = max(values.get(value, -1.0), score)
        ranked = sorted(values.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return None, 0.0, "no calibrated labels"
        value, score = ranked[0]
        margin = score - ranked[1][1] if len(ranked) > 1 else 0.0
        raw = f"template={value}; correlation={score:.4f}; margin={margin:.4f}"
        if score < 0.90 or margin < 0.045:
            return None, score, raw
        number = (5 if field == "ex" else 10) if value == "MAX" else int(value)
        if field == "ex" and number > 5 or field == "gear" and number > 2:
            return None, score, raw
        return number, score, raw
