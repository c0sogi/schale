"""Registered UI state and numeric reading with original-image evidence."""

from pathlib import Path

import cv2
import numpy as np

from .identity import IdentityMatcher, read_image
from .labels import BOXES, LabelBank
from .models import Evidence, Reading, Student
from .layout import LayoutMatcher
from .video import crop
from ..assets import resolve_resources

HEADER = (0.555, 0.180, 0.625, 0.221)
EMPTY_GEAR = (0.754, 0.790, 0.816, 0.858)
WEAPON = (0.610, 0.625, 0.856, 0.748)
POTENTIAL = {
    "potential_hp": (0.630, 0.307, 0.678, 0.357),
    "potential_attack": (0.771, 0.307, 0.817, 0.357),
    "potential_heal": (0.771, 0.362, 0.817, 0.409),
}


def dark_fraction(roi) -> float:
    return float(np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) < 130))


def tier_badge_present(roi) -> bool:
    """A tier badge remains visible even when its numeral cannot be read."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array((91, 96, 0)), np.array((124, 255, 234)))
    return int(np.count_nonzero(blue)) >= 40


def unequipped_marker(roi) -> bool:
    """Detect the filled gold notification at a slot's upper-right corner."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((15, 140, 180)), np.array((40, 255, 255)))
    _, _, parts, _ = cv2.connectedComponentsWithStats(mask)
    return any(
        12 <= p[2] <= 30 and 12 <= p[3] <= 30 and p[4] / (p[2] * p[3]) > 0.55
        for p in parts[1:]
    )


def match_patch(image, box, template) -> float:
    observed = cv2.cvtColor(crop(image, box), cv2.COLOR_BGR2GRAY)
    expected = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    observed = cv2.resize(observed, (expected.shape[1], expected.shape[0]))
    return float(cv2.matchTemplate(observed, expected, cv2.TM_CCOEFF_NORMED)[0, 0])


class Vision:
    def __init__(
        self, identity: IdentityMatcher, data: Path | None = None, numeric=None
    ):
        data = data if data is not None else resolve_resources("students")
        self.identity = identity
        self.labels = LabelBank(data)
        self.numeric = numeric
        self.layout = LayoutMatcher(data / "layout_reference.npz")
        self.header = read_image(data / "header.png")
        self.empty_gear = read_image(data / "empty_gear.png")
        self.ghost_templates = [
            read_image(data / f"ghost_{kind}.png")
            for kind in ("gloves", "shoes", "hat")
        ]
        self.potential_templates = {
            k: read_image(data / f"{k}_25.png")
            for k in ("potential_hp", "potential_attack")
        }

    def analyze(
        self,
        path: Path,
        image_name: str,
        timestamp: float,
        key: str,
        identity_hint: tuple | None = None,
    ) -> Student:
        image = read_image(path)[:, :, :3]
        h, w = image.shape[:2]
        if abs(w / h - 16 / 9) > 0.03:
            raise ValueError("The Korean profile requires a 16:9 game viewport")
        row = Student(key=key, screenshots=[image_name])
        layout = self.layout.locate(image)
        if layout is None:
            row.identity_status = "layout_unresolved"
            return row
        row.layouts[image_name] = {
            name: {k: v for k, v in info.items() if not k.endswith("_points")}
            for name, info in layout.diagnostics.items()
        }
        image = layout.normalize(image)

        def evidence(box, raw, score, method="visual-template"):
            return Evidence(
                image=image_name,
                timestamp=timestamp,
                roi=layout.source_roi(box),
                raw=raw,
                score=max(0.0, min(1.0, score)),
                method=method,
            )

        def visual(field, value, box, raw, score=0.95, inferred=False):
            row.fields[field] = Reading(
                value=value,
                status="inferred" if inferred else "observed",
                evidence=[evidence(box, raw, score, "visual-state")],
            )

        if match_patch(image, HEADER, self.header) < 0.85:
            row.identity_status = "not_student_screen"
            return row
        student_id, candidates, method = identity_hint or self.identity.identify(image)
        row.student_id = student_id
        row.name = (
            self.identity.catalog[str(student_id)]["Name"]
            if student_id
            else "미확인 학생"
        )
        row.identity_status = "observed" if student_id else "unknown"
        row.candidates = [
            {k: v for k, v in c.items() if k != "icons"} for c in candidates
        ]
        row.name_evidence = [
            evidence(
                (0.01, 0.07, 0.86, 0.77),
                method,
                candidates[0]["score"] if candidates else 0.0,
                method,
            )
        ]
        numeric = self.numeric.read_all(image) if self.numeric is not None else {}
        for field, box in BOXES.items():
            value, score, raw = (
                numeric[field]
                if self.numeric is not None
                else self.labels.read(image, field)
            )
            row.fields[field] = Reading(
                value=value,
                status="observed" if value is not None else "unknown",
                evidence=[
                    evidence(
                        box,
                        raw,
                        score,
                        self.numeric.method
                        if self.numeric is not None
                        else "visual-template",
                    )
                ],
            )
            if self.numeric is not None:
                alternate, alternate_score, alternate_raw = self.labels.read(
                    image, field
                )
                if value is None and alternate is not None:
                    row.fields[field] = Reading(
                        value=alternate,
                        status="observed",
                        evidence=[
                            *row.fields[field].evidence,
                            evidence(
                                box,
                                alternate_raw,
                                alternate_score,
                                "visual-template-fallback",
                            ),
                        ],
                    )
                elif alternate is not None and alternate != value:
                    # A systematic CTC error can repeat across every frame. A
                    # confident independent reader disagreement must survive
                    # temporal aggregation instead of becoming "confirmed".
                    row.fields[field].value = None
                    row.fields[field].status = "conflict"
                    row.fields[field].evidence.append(
                        evidence(
                            box,
                            alternate_raw,
                            alternate_score,
                            "visual-template-disagreement",
                        )
                    )

        for field, box, bounds, pitch, maximum in (
            (
                "star",
                (0.20, 0.783, 0.266, 0.827),
                ((15, 100, 130), (40, 255, 255)),
                0.011,
                5,
            ),
            (
                "weapon_star",
                (0.790, 0.700, 0.851, 0.738),
                ((85, 60, 100), (115, 255, 255)),
                0.013,
                4,
            ),
        ):
            roi = crop(image, box)
            mask = cv2.inRange(
                cv2.cvtColor(roi, cv2.COLOR_BGR2HSV),
                np.array(bounds[0]),
                np.array(bounds[1]),
            )
            _, _, components, _ = cv2.connectedComponentsWithStats(mask)
            significant = [c for c in components[1:] if c[4] > 50 and c[3] > 15]
            row.fields[field] = Reading(
                evidence=[evidence(box, "no reliable star strip", 0.0)]
            )
            if significant:
                lo = min(c[0] for c in significant)
                hi = max(c[0] + c[2] for c in significant)
                count = round((hi - lo) / (2560 * pitch))
                if 1 <= count <= maximum:
                    visual(field, count, box, f"colored star strip width={hi - lo}")
        if dark_fraction(crop(image, WEAPON)) > 0.78:
            visual("weapon_star", 0, WEAPON, "locked weapon panel", inferred=True)
            visual("weapon_level", 0, WEAPON, "locked weapon panel", inferred=True)
        for i, x in enumerate((0.520, 0.594, 0.668), 1):
            box = (x + 0.013, 0.803, x + 0.067, 0.850)
            roi = crop(image, box)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            locked = dark_fraction(roi) > 0.78
            ghost = (
                dark_fraction(roi) < 0.003
                and float(np.mean(hsv[:, :, 1] > 70)) < 0.35
                and float(np.std(hsv[:, :, 2])) < 30
            )
            badge = tier_badge_present(crop(image, BOXES[f"equipment{i}"]))
            # Notifications can also indicate an available upgrade. Only a
            # notification AND absence of the tier badge means unequipped.
            marker_x = (1540 + (i - 1) * 186) / 2560
            marker_box = (marker_x - 0.003, 0.765, marker_x + 0.014, 0.789)
            if not badge and unequipped_marker(crop(image, marker_box)):
                ghost = True
            if (
                i == 1
                and row.fields["equipment1"].value is None
                and row.fields["equipment1_level"].value is None
            ):
                ghost_box = (0.530, 0.775, 0.592, 0.873)
                observed = crop(image, ghost_box)
                for template in self.ghost_templates:
                    # Correlation alone cannot distinguish a faded placeholder from
                    # the same equipped item. Also require its absolute faded colors.
                    error = float(
                        np.mean(
                            np.abs(
                                observed.astype(np.float32)
                                - template.astype(np.float32)
                            )
                        )
                    )
                    if error < 12 and match_patch(image, ghost_box, template) > 0.94:
                        ghost = True
                        break
            if locked or ghost and row.fields[f"equipment{i}"].value is None:
                raw = (
                    "locked slot; Lv label is unlock condition"
                    if locked
                    else "pale unequipped slot"
                )
                visual(f"equipment{i}", 0, box, raw, inferred=True)
                visual(f"equipment{i}_level", 0, box, raw, inferred=True)
            elif row.fields[f"equipment{i}"].value is None:
                row.fields[f"equipment{i}_level"].value = None
                row.fields[f"equipment{i}_level"].status = "unknown"
        if row.fields["gear"].value is None:
            score = match_patch(image, EMPTY_GEAR, self.empty_gear)
            if score > 0.90:
                visual(
                    "gear", 0, EMPTY_GEAR, "EMPTY slot template", score, inferred=True
                )
            elif not tier_badge_present(crop(image, BOXES["gear"])):
                if dark_fraction(crop(image, EMPTY_GEAR)) > 0.78:
                    visual(
                        "gear", 0, EMPTY_GEAR, "locked unique gear panel", inferred=True
                    )
                elif unequipped_marker(crop(image, (0.816, 0.765, 0.832, 0.789))):
                    visual(
                        "gear",
                        0,
                        EMPTY_GEAR,
                        "unequipped unique gear; no tier badge",
                        inferred=True,
                    )
        for field, x in (("basic", 0.627), ("passive", 0.709), ("sub", 0.792)):
            box = (x, 0.455, x + 0.064, 0.588)
            if dark_fraction(crop(image, box)) > 0.75:
                visual(field, 0, box, "locked skill", inferred=True)
        for field, box in POTENTIAL.items():
            hsv = cv2.cvtColor(crop(image, box), cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((90, 110, 70)), np.array((125, 255, 255)))
            if float(np.mean(mask > 0)) < 0.025:
                visual(
                    field, 0, box, "no ability badge on visible stat row", inferred=True
                )
            else:
                row.fields[field] = Reading(
                    evidence=[evidence(box, "ability badge present; needs review", 0.0)]
                )
                if self.numeric is not None:
                    value, score, raw = self.numeric.read_patches(
                        {field: crop(image, box)}
                    )[field]
                    row.fields[field] = Reading(
                        value=value,
                        status="observed" if value is not None else "unknown",
                        evidence=[evidence(box, raw, score, self.numeric.method)],
                    )
                    if value is not None:
                        continue
                if field in self.potential_templates:
                    score = match_patch(image, box, self.potential_templates[field])
                    if score >= 0.94:
                        visual(
                            field,
                            25,
                            box,
                            "human-calibrated Lv.25 ability badge",
                            score,
                        )
        return row
