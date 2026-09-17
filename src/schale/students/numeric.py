"""Optional SVTRv2-S inference for the fixed Korean student-information UI."""

import hashlib
import re
from pathlib import Path

import cv2
import numpy as np

from ..vision.text import CtcRecognizer, prepare as prepare

from .labels import BOXES, family
from .model_bundle import (
    PROFILE as PROFILE,
    install_bundle as install_bundle,
    validate_bundle as validate_bundle,
)
from .video import crop

DEFAULT_MODEL = Path(".schale/student-numeric")


def parse(text: str, field: str) -> int | None:
    text = text.strip().upper()
    # Yellow-only ability-badge localization may omit the tiny Lv prefix.
    # White neighboring combat statistics are excluded by that localization.
    if field.startswith("potential_") and re.fullmatch(r"[1-9][0-9]?", text):
        text = "LV." + text
    kind = family(field)
    if text == "MAX" and kind == "skill":
        return 5 if field == "ex" else 10
    pattern = (
        r"T([1-9][0-9]?)"
        if kind == "tier"
        else r"([1-9][0-9]?)"
        if field == "bond"
        else r"LV[.,]?([1-9][0-9]?)"
    )
    match = re.fullmatch(pattern, text)
    if not match:
        return None
    value = int(match[1])
    maximum = (
        25
        if field.startswith("potential_")
        else 2
        if field == "gear"
        else 10
        if kind == "tier"
        else 5
        if field == "ex"
        else 10
        if kind == "skill"
        else 100
    )
    return value if value <= maximum else None


def bond_cell(patch):
    """Keep complete numeral strokes inside the heart, excluding its dark surround."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    pink = (
        (hsv[:, :, 0] > 140)
        & (hsv[:, :, 0] < 179)
        & (hsv[:, :, 1] > 30)
        & (hsv[:, :, 2] > 170)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return patch
    heart = max(contours, key=cv2.contourArea)
    if cv2.contourArea(heart) < patch.shape[0] * patch.shape[1] * 0.15:
        return patch
    interior = np.zeros(pink.shape, np.uint8)
    cv2.drawContours(interior, [heart], -1, 255, -1)
    interior = cv2.erode(interior, np.ones((3, 3), np.uint8))
    ink = ((hsv[:, :, 2] < 150) & (hsv[:, :, 1] < 170) & (interior > 0)).astype(
        np.uint8
    )
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    keep = [
        i
        for i, part in enumerate(stats[1:], 1)
        if part[3] >= patch.shape[0] * 0.25 and part[4] >= 12
    ]
    if not keep:
        return patch
    mask = np.isin(labels, keep)
    yy, xx = np.nonzero(mask)
    cell = (
        255
        - mask[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1].astype(np.uint8) * 255
    )
    cell = cv2.copyMakeBorder(cell, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)


def localize(patch, field: str):
    """Locate the text but retain RGB texture and outlines for recognition."""
    kind = family(field)
    if field == "bond":
        return bond_cell(patch)
    if field.startswith("potential_"):
        # Detect the enclosing badge, not separate yellow glyph components:
        # a compressed last digit can otherwise disappear from the crop.
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array((90, 110, 70)), np.array((125, 255, 255)))
        contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return patch
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        yellow = (
            (hsv[:, :, 0] >= 15)
            & (hsv[:, :, 0] <= 40)
            & (hsv[:, :, 1] > 100)
            & (hsv[:, :, 2] > 130)
        )
        rows = np.flatnonzero(yellow[:, x : x + w].sum(axis=1) >= 3)
        if len(rows):
            y, h = int(rows[0]), int(rows[-1] - rows[0] + 1)
        return patch[max(0, y - 3) : y + h + 3, max(0, x - 3) : x + w + 3].copy()
    if field == "weapon_level" or field.endswith("_level"):
        return patch
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    if kind == "bond":
        mask = ((hsv[:, :, 1] < 170) & (hsv[:, :, 2] < 160)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    elif field.startswith("potential_"):
        # The yellow level text is below a separate MAX ornament. Its bounding box
        # excludes both that ornament and the neighboring combat-stat digits.
        mask = (
            (hsv[:, :, 0] >= 15)
            & (hsv[:, :, 0] <= 40)
            & (hsv[:, :, 1] > 100)
            & (hsv[:, :, 2] > 130)
        ).astype(np.uint8)
    elif kind == "skill":
        mask = (hsv[:, :, 2] < 150).astype(np.uint8)
    elif kind == "tier":
        mask = (
            (hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 125) & (hsv[:, :, 1] > 95)
        ).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    else:
        mask = ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 200)).astype(np.uint8)
    h, w = mask.shape
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    parts = [
        s
        for s in stats[1:]
        if s[3] >= h * (0.2 if field.startswith("potential_") else 0.3)
        and s[3] < h * 0.9
        and s[4] >= 15
        and s[2] < w * 0.8
        and s[4] / (s[2] * s[3]) > (0.3 if kind == "tier" else 0.15)
    ]
    if field == "level":
        parts = [s for s in parts if s[0] + s[2] < w * 0.8]
    if kind == "tier":
        # Item artwork enters this ROI from above. The tier badge itself has
        # a complete baseline and never touches the top of the registered ROI.
        parts = [s for s in parts if s[1] >= h * 0.15 and s[3] >= h * 0.4]
    if not parts:
        return patch
    pad = 6 if kind == "bond" else 3
    x0 = max(0, min(s[0] for s in parts) - pad)
    y0 = max(0, min(s[1] for s in parts) - pad)
    x1 = min(w, max(s[0] + s[2] for s in parts) + pad)
    y1 = min(h, max(s[1] + s[3] for s in parts) + pad)
    return patch[y0:y1, x0:x1].copy()


def numeric_cell(patch, field: str):
    """Keep the full two-digit cell; component crops can erase outlined zeros.

    Coordinates are in the registered 2560x1440 UI, relative to BOXES. The
    fallback is only for equipment levels, whose Lv prefix overlaps item art.
    """
    boxes = {
        "equipment1_level": (58, 5, 100, 33),
        "equipment2_level": (54, 5, 96, 33),
        "equipment3_level": (50, 5, 92, 33),
    }
    if field not in boxes:
        return None
    x0, y0, x1, y1 = boxes[field]
    if patch.shape[0] < y1 or patch.shape[1] < x1:
        return None
    return patch[y0:y1, x0:x1]


class NumericReader(CtcRecognizer):
    method = "svtrv2-s-rctc"

    def __init__(self, directory: Path):
        try:
            import onnxruntime  # noqa: F401
            import PIL  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "Install the student-ocr extra: uv sync --extra student-ocr"
            ) from error
        self.metadata = validate_bundle(directory)
        self.charset = self.metadata["charset"]
        self.threshold = self.metadata["threshold"]
        self.fingerprint = hashlib.sha256(
            (directory / "metadata.json").read_bytes()
        ).hexdigest()
        super().__init__(directory / "model.onnx", self.charset)

    def read_patches(self, patches: dict) -> dict[str, tuple[int | None, float, str]]:
        groups: dict[tuple, list] = {}
        for field, patch in patches.items():
            array = prepare(localize(patch, field))
            groups.setdefault(array.shape, []).append((field, array))
        result = {}
        primary_texts = {}
        for group in groups.values():
            inputs = np.stack([array for _, array in group])
            for (field, _), reading in zip(group, self.infer(inputs), strict=True):
                text, score = reading.text, reading.score
                primary_texts[field] = text
                number = parse(text, field) if score >= self.threshold else None
                result[field] = (
                    number,
                    score,
                    f"text={text!r}; min_char_score={score:.4f}; threshold={self.threshold}",
                )
        # A second RGB view excludes the known Lv prefix and most item art.
        # Never overwrite a valid primary reading. Retain conflicting primary
        # digit evidence, even when the prefix caused its confidence to fall.
        fallback_groups: dict[tuple, list] = {}
        for field, patch in patches.items():
            if result[field][0] is not None:
                continue
            # A tightly localized Lv.1 can lose confidence at its boundary.
            # Retry the complete registered label, but only when the primary
            # view already decoded a valid number. Both views must agree.
            cell = (
                patch
                if field == "level" and parse(primary_texts[field], field) is not None
                else numeric_cell(patch, field)
            )
            if cell is not None:
                array = prepare(cell)
                fallback_groups.setdefault(array.shape, []).append((field, array))
        for group in fallback_groups.values():
            readings = self.infer(np.stack([a for _, a in group]))
            for (field, _), reading in zip(group, readings, strict=True):
                text, score = reading.text, reading.score
                # Dots/dashes outside the number are outlines, not new digits.
                match = re.fullmatch(r"([1-9][0-9]?)[.\-]?", text)
                number = (
                    parse("Lv." + match[1], field)
                    if match and score >= self.threshold
                    else None
                )
                if field == "level":
                    number = parse(text, field) if score >= self.threshold else None
                raw = result[field][2]
                primary = re.fullmatch(
                    r"(LV[.,]?[1-9][0-9]?)[.\-$]?", primary_texts[field].upper()
                )
                if primary and number != parse(primary[1], field):
                    number = None
                if number is not None:
                    view = "full-label" if field == "level" else "numeric-cell"
                    result[field] = (
                        number,
                        score,
                        raw + f"; {view} text={text!r}; score={score:.4f}",
                    )
        return result

    def read_all(self, image) -> dict[str, tuple[int | None, float, str]]:
        return self.read_patches(
            {field: crop(image, box) for field, box in BOXES.items()}
        )
