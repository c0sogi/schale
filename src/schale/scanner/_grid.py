"""Inventory card grid detection from repeated borders, independent of padding."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class CellRegion:
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int

    def extract_roi(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return image[self.y : self.y + self.height, self.x : self.x + self.width]


def _groups(values, tolerance):
    groups: list[list] = []
    for value in sorted(values):
        if not groups or value - float(np.median(groups[-1])) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [float(np.median(group)) for group in groups]


def detect_grid(image: NDArray[np.uint8]) -> list[CellRegion]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = (
        (gray > float(np.percentile(gray, 99)) * 0.95) & (hsv[:, :, 1] < 50)
    ).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(contour) for contour in contours]
    rects = [
        r
        for r in rects
        if 0.8 < r[2] / r[3] < 1.6
        and 0.001 * gray.size < r[2] * r[3] < 0.1 * gray.size
        and min(r[2:]) >= 20
    ]
    distinct = []
    for x, y, w, h in sorted(rects, key=lambda r: r[2] * r[3], reverse=True):
        if any(
            max(0, min(x + w, a + c) - max(x, a))
            * max(0, min(y + h, b + d) - max(y, b))
            > 0.8 * w * h
            for a, b, c, d in distinct
        ):
            continue
        distinct.append((x, y, w, h))
    if not distinct:
        raise RuntimeError("No inventory card borders detected")
    # Choose the repeated large-card geometry, not the smaller artwork inside it.
    clusters = [
        [r for r in distinct if abs(r[2] / w - 1) < 0.16 and abs(r[3] / h - 1) < 0.16]
        for _, _, w, h in distinct
    ]
    cluster = max(
        clusters,
        key=lambda group: len(group) * float(np.median([r[2] * r[3] for r in group])),
    )
    if len(cluster) < 3:
        raise RuntimeError("Insufficient repeated card borders to establish a grid")
    width = float(np.median([r[2] for r in cluster]))
    height = float(np.median([r[3] for r in cluster]))
    xs = _groups([r[0] for r in cluster], width * 0.25)
    ys = _groups([r[1] for r in cluster], height * 0.25)
    cells = []
    for row, y_float in enumerate(ys):
        for col, x_float in enumerate(xs):
            x, y, w, h = round(x_float), round(y_float), round(width), round(height)
            if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                continue
            supported = any(
                abs(a - x) < width * 0.25 and abs(b - y) < height * 0.25
                for a, b, _, _ in cluster
            )
            if not supported:
                # Selected cards have a gold rather than white outline. Complete
                # a missing grid slot only if real card interior and text remain.
                patch = image[y : y + h, x : x + w]
                patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                bottom = patch_gray[int(h * 0.75) :]
                light = float(np.percentile(gray, 99))
                if (
                    patch_gray.std() < 20
                    or np.mean(bottom > light * 0.85) < 0.3
                    or np.mean(bottom < light * 0.6) < 0.03
                ):
                    continue
            cells.append(CellRegion(row, col, x, y, w, h))
    if not cells:
        raise RuntimeError("No complete inventory cells detected")
    return cells
