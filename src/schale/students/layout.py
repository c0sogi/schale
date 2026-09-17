"""Independent left/right UI registration using static SIFT anchors and RANSAC."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REFERENCE_SIZE = (1280, 720)


@dataclass
class Layout:
    # Canonical 1280x720 coordinates -> observed 1280x720 coordinates.
    matrices: dict
    diagnostics: dict

    def visible(self) -> bool:
        """Do not fill clipped numeric regions with replicated image borders."""
        for name, box in {
            "left": (0.025, 0.775, 0.270, 0.863),
            "right": (0.520, 0.307, 0.859, 0.888),
        }.items():
            x0, y0, x1, y1 = box
            corners = (
                np.array(
                    [
                        [x0 * 1280, y0 * 720, 1],
                        [x1 * 1280, y0 * 720, 1],
                        [x1 * 1280, y1 * 720, 1],
                        [x0 * 1280, y1 * 720, 1],
                    ]
                )
                @ self.matrices[name].T
            )
            if np.any(corners < 0) or np.any(corners > (1280, 720)):
                return False
        return True

    def source_roi(self, box):
        x0, y0, x1, y1 = box
        if x0 < 0.5 < x1:
            return (0.0, 0.0, 1.0, 1.0)
        matrix = self.matrices["left" if x1 <= 0.5 else "right"]
        corners = (
            np.array(
                [
                    [x0 * 1280, y0 * 720, 1],
                    [x1 * 1280, y0 * 720, 1],
                    [x1 * 1280, y1 * 720, 1],
                    [x0 * 1280, y1 * 720, 1],
                ]
            )
            @ matrix.T
        )
        lo, hi = corners.min(axis=0) / (1280, 720), corners.max(axis=0) / (1280, 720)
        values = np.clip([*lo, *hi], 0, 1)
        return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))

    def normalize(self, original):
        h, w = original.shape[:2]
        result = np.empty((1440, 2560, 3), np.uint8)
        for name, (start, end) in {"left": (0, 1280), "right": (1280, 2560)}.items():
            canonical_to_source = self.matrices[name].copy()
            canonical_to_source[:, :2] /= 2
            canonical_to_source[0] *= w / 1280
            canonical_to_source[1] *= h / 720
            warped = cv2.warpAffine(
                original,
                canonical_to_source,
                (2560, 1440),
                flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            result[:, start:end] = warped[:, start:end]
        return result


class LayoutMatcher:
    @classmethod
    def from_input(cls, original, layout: Layout):
        """Build transient static-UI anchors from this input, never a shipped capture."""
        matcher = cls.__new__(cls)
        matcher.reference = {}
        matcher.sift = cv2.SIFT.create(nfeatures=5000, contrastThreshold=0.025)
        matcher.matcher = cv2.BFMatcher(cv2.NORM_L2)
        canonical = cv2.resize(layout.normalize(original), REFERENCE_SIZE)
        gray = cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY)
        regions = {
            "left": [(425, 645, 622, 704)],
            "right": [
                (660, 115, 1216, 217),
                (1110, 310, 1200, 425),
                (1110, 450, 1200, 535),
                (1100, 555, 1200, 640),
            ],
        }
        for side, boxes in regions.items():
            mask = np.zeros(gray.shape, np.uint8)
            for x0, y0, x1, y1 in boxes:
                mask[y0:y1, x0:x1] = 255
            points, descriptors = matcher.sift.detectAndCompute(gray, mask)
            if descriptors is None or len(points) < 12:
                return None
            matcher.reference[side + "_points"] = np.array(
                [p.pt for p in points], np.float32
            )
            matcher.reference[side + "_descriptors"] = descriptors
        return matcher

    def __init__(self, data: Path | None = None):
        from ..assets import resolve_resources

        source = data or resolve_resources("students") / "layout_reference.npz"
        with np.load(source, allow_pickle=False) as archive:
            self.reference = {key: archive[key] for key in archive.files}
        self.sift = cv2.SIFT.create(nfeatures=5000, contrastThreshold=0.025)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def locate(self, original) -> Layout | None:
        resized = cv2.resize(original, REFERENCE_SIZE, interpolation=cv2.INTER_AREA)
        points, descriptors = self.sift.detectAndCompute(
            cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), None
        )
        if descriptors is None or len(points) < 12:
            return None
        matrices, diagnostics = {}, {}
        for name in ("left", "right"):
            reference = self.reference[name + "_points"]
            pairs = self.matcher.knnMatch(
                self.reference[name + "_descriptors"], descriptors, k=2
            )
            good = [a for a, b in pairs if a.distance < 0.72 * b.distance]
            if len(good) < 12:
                return None
            a = np.asarray([reference[m.queryIdx] for m in good], dtype=np.float32)
            b = np.asarray([points[m.trainIdx].pt for m in good], dtype=np.float32)
            matrix, inliers = cv2.estimateAffinePartial2D(
                a,
                b,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.0,
                maxIters=2000,
                confidence=0.995,
            )
            if matrix is None or inliers is None:
                return None
            keep = inliers.ravel().astype(bool)
            count = int(keep.sum())
            span = np.ptp(a[keep], axis=0)
            scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
            angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
            error = np.linalg.norm(
                a[keep] @ matrix[:, :2].T + matrix[:, 2] - b[keep], axis=1
            )
            rms = float(np.sqrt(np.mean(error**2)))
            minimum_span = (80, 20) if name == "left" else (180, 180)
            if (
                count < 12
                or count / len(good) < 0.45
                or np.any(span < minimum_span)
                or not 0.7 <= scale <= 1.3
                or abs(angle) > 3
                or rms > 1.5
                or abs(matrix[0, 2]) > 230
                or abs(matrix[1, 2]) > 130
            ):
                return None
            matrices[name] = matrix
            diagnostics[name] = {
                "matrix": matrix.tolist(),
                "scale": scale,
                "rotation_deg": angle,
                "inliers": count,
                "matches": len(good),
                "rms_px_at_720p": rms,
                "reference_points": a[keep].tolist(),
                "source_points": b[keep].tolist(),
            }
        layout = Layout(matrices, diagnostics)
        return layout if layout.visible() else None
