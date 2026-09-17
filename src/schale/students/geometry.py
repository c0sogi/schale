"""Student UI registration from geometric edges, without captured game assets."""

import cv2
import numpy as np
from itertools import product

from .layout import Layout, LayoutMatcher, REFERENCE_SIZE


class BootstrapLayoutMatcher:
    """Build static anchors from the user's input after qualifying its geometry."""

    def __init__(self):
        self.geometry = GeometryMatcher()
        self.matcher = None

    def locate(self, original):
        if self.matcher is not None:
            layout = self.matcher.locate(original)
            if layout is not None:
                for info in layout.diagnostics.values():
                    info["method"] = "input-derived-sift"
                return layout
        layout = self.geometry.locate(original)
        if layout is not None:
            self.matcher = LayoutMatcher.from_input(original, layout)
            for info in layout.diagnostics.values():
                info["method"] = "input-bootstrap-geometry"
        return layout


class GeometryMatcher:
    """Register panel separators and three footer buttons, rejecting weak fits.

    Coordinates describe the UI profile, not pixels or descriptors from a game
    capture. Scale is shared by the UI; each side has its own translation.
    """

    def locate(self, original):
        self.last_failure = "panel-edges"
        image = cv2.resize(original, REFERENCE_SIZE, interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(image, 60, 140)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 90, minLineLength=220, maxLineGap=12
        )
        if lines is None:
            return None
        lines = np.asarray(lines).reshape(-1, 4)
        horizontal = [
            line
            for line in lines
            if abs(line[1] - line[3]) < 2 and min(line[0], line[2]) > 450
        ]
        ends = [
            int(max(line[0], line[2]))
            for line in horizontal
            if abs(line[0] - line[2]) > 400
        ]
        if not ends:
            return None
        border = float(max(set(ends), key=lambda x: (ends.count(x), x)))
        rows = np.array(
            sorted(
                {
                    float(line[1])
                    for line in horizontal
                    if max(line[0], line[2]) >= border - 25
                    and abs(line[0] - line[2]) > 350
                }
            )
        )
        levels = np.array([178, 302, 309, 433, 440, 548, 651], dtype=float)
        candidates = []
        for bottom, reference_bottom in product(rows[rows > 400], (548, 651)):
            for top in rows[(rows > 70) & (rows < 300)]:
                scale = (bottom - top) / (reference_bottom - 178)
                if not 0.7 < scale < 1.3:
                    continue
                ty = top - scale * 178
                distances = np.abs((levels * scale + ty)[:, None] - rows[None, :])
                residual = distances.min(axis=1)
                supported = residual < 2.5
                # At least five separators spanning most of the panel must
                # agree. The bottom outline can blend into a bright background.
                if (
                    supported.sum() >= 5
                    and supported[0]
                    and (supported[-1] or supported[-2])
                ):
                    score = float(np.mean(residual[supported] ** 2)) + 3 * int(
                        (~supported).sum()
                    )
                    candidates.append(
                        (score, scale, ty, supported, distances.argmin(axis=1))
                    )
        if not candidates:
            self.last_failure = "separator-pattern"
            return None
        _, scale, ty, supported, matches = min(candidates, key=lambda c: c[0])
        right = np.array([[scale, 0, border - scale * 1215], [0, scale, ty]])
        reference_right = np.column_stack(
            (np.full(supported.sum(), 1215), levels[supported])
        )
        source_right = np.column_stack(
            (np.full(supported.sum(), border), rows[matches[supported]])
        )

        # Match straight geometric outlines, not icons or text. Distance to
        # actual edges allows antialiasing while penalizing missing outlines.
        distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
        closeness = np.exp(-(distance**2) / 3).astype(np.float32)
        shape = np.zeros((round(56 * scale) + 3, round(200 * scale) + 3), np.float32)
        for i in range(3):
            button = np.zeros_like(shape)
            polygon = np.array(
                [
                    [11 + i * 66, 0],
                    [66 + i * 66, 0],
                    [57 + i * 66, 53],
                    [2 + i * 66, 53],
                ],
                dtype=float,
            )
            cv2.polylines(
                button, [np.rint(polygon * scale).astype(np.int32)], True, 1.0, 1
            )
            shape = np.maximum(shape, button)
        sx = max(0, round(423 * scale + right[0, 2]) - 60)
        sy = max(0, round(646 * scale + right[1, 2]) - 30)
        search = closeness[
            sy : min(720, sy + round(56 * scale) + 60),
            sx : min(1280, sx + round(200 * scale) + 120),
        ]
        if search.shape[0] < shape.shape[0] or search.shape[1] < shape.shape[1]:
            return None
        scores = cv2.matchTemplate(search, shape, cv2.TM_CCORR) / shape.sum()
        _, agreement, _, (x, y) = cv2.minMaxLoc(scores)
        if agreement < 0.6:
            self.last_failure = f"footer-edges agreement={agreement:.3f}"
            return None
        left = np.array(
            [[scale, 0, sx + x - scale * 423], [0, scale, sy + y - scale * 646]]
        )
        yy, xx = np.nonzero(shape)
        edge_error = float(np.sqrt(np.mean(distance[sy + y + yy, sx + x + xx] ** 2)))
        reference_left, source_left = [], []
        for px, py in zip(xx[::8], yy[::8], strict=True):
            source_x, source_y = int(sx + x + px), int(sy + y + py)
            x0, y0 = max(0, source_x - 2), max(0, source_y - 2)
            candidates_y, candidates_x = np.nonzero(
                edges[y0 : source_y + 3, x0 : source_x + 3]
            )
            if not len(candidates_x):
                continue
            nearest = np.argmin(
                (candidates_x + x0 - source_x) ** 2
                + (candidates_y + y0 - source_y) ** 2
            )
            reference_left.append([float(px / scale + 423), float(py / scale + 646)])
            source_left.append(
                [int(candidates_x[nearest] + x0), int(candidates_y[nearest] + y0)]
            )
        matrices = {"left": left, "right": right}
        diagnostics = {
            "right": {
                "inliers": int(supported.sum()),
                "matches": len(levels),
                "rms_px_at_720p": float(
                    np.sqrt(
                        np.mean(
                            (
                                reference_right @ right[:, :2].T
                                + right[:, 2]
                                - source_right
                            )
                            ** 2
                        )
                    )
                ),
                "reference_points": reference_right.tolist(),
                "source_points": source_right.tolist(),
            },
            "left": {
                "inliers": int(
                    np.count_nonzero(distance[sy + y + yy, sx + x + xx] < 2)
                ),
                "matches": len(xx),
                "edge_agreement": agreement,
                "rms_px_at_720p": edge_error,
                "reference_points": reference_left,
                "source_points": source_left,
            },
        }
        for name, matrix in matrices.items():
            diagnostics[name].update(
                {
                    "matrix": matrix.tolist(),
                    "scale": float(scale),
                    "rotation_deg": 0.0,
                    "method": "geometric-edge-registration",
                }
            )
        layout = Layout(matrices, diagnostics)
        self.last_failure = "clipped-regions" if not layout.visible() else ""
        return layout if layout.visible() else None
