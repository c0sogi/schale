"""Student identity from game skill symbols, with portrait feature verification."""

from pathlib import Path

import cv2
import numpy as np
from ..cache import shared_cache

from .catalog import SKILL_KEYS
from .video import crop

ICON_BOXES = tuple((x, 0.475, x + 0.060, 0.557) for x in (0.544, 0.627, 0.709, 0.792))


def read_image(path: Path):
    image = cv2.imdecode(
        np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED
    )
    if image is None:
        raise ValueError(f"Invalid image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def symbol(image, box):
    roi = crop(image, box)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0, 100, 35)), np.array((179, 255, 255)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if w < roi.shape[1] * 0.35 or h < roi.shape[0] * 0.45:
        return None
    interior = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(interior, [contour], -1, 255, -1)
    interior = cv2.erode(interior, np.ones((7, 7), np.uint8))
    white = ((hsv[:, :, 1] < 75) & (hsv[:, :, 2] > 170) & (interior > 0)).astype(
        np.uint8
    ) * 255
    return cv2.resize(white[y : y + h, x : x + w], (96, 96))


class IdentityMatcher:
    def __init__(self, catalog: dict, assets: dict[str, Path], cache: Path):
        self.catalog = catalog
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.names = sorted(assets)
        self.templates = []
        for name in self.names:
            alpha = read_image(assets[name])[:, :, 3]
            self.templates.append(
                [cv2.resize(alpha, (s, s)) for s in (76, 80, 84, 88, 92, 96)]
            )
        self.indices = {name: i for i, name in enumerate(self.names)}
        self.sift = cv2.SIFT.create(nfeatures=1800)
        self.portraits: dict = {}

    def rank(self, image):
        scores = []
        for box in ICON_BOXES:
            observation = symbol(image, box)
            if observation is None:
                scores.append(np.zeros(len(self.names)))
                continue
            scores.append(
                np.array(
                    [
                        max(
                            float(
                                cv2.minMaxLoc(
                                    cv2.matchTemplate(
                                        observation, t, cv2.TM_CCOEFF_NORMED
                                    )
                                )[1]
                            )
                            for t in templates
                        )
                        for templates in self.templates
                    ]
                )
            )
        ranked = []
        for student_id, row in self.catalog.items():
            values = [
                float(scores[i][self.indices[s["Icon"]]])
                if (s := row.get("Skills", {}).get(k)) and s.get("Icon") in self.indices
                else 0.0
                for i, k in enumerate(SKILL_KEYS)
            ]
            # EX receives more weight, but shared EX symbols cannot establish identity.
            score = values[0] * 0.45 + sum(values[1:]) * 0.55 / 3
            ranked.append(
                {
                    "id": int(student_id),
                    "name": row["Name"],
                    "score": score,
                    "icons": values,
                }
            )
        return sorted(ranked, key=lambda row: row["score"], reverse=True)

    def portrait_score(self, image, student_id: int):
        if student_id not in self.portraits:
            path = (
                shared_cache()
                .fetch(
                    f"https://schaledb.com/images/student/portrait/{student_id}.webp",
                    kind="image",
                    seed=self.cache / f"{student_id}.webp",
                )
                .path
            )
            portrait = read_image(path)
            portrait = cv2.resize(portrait, None, fx=0.65, fy=0.65)
            gray = cv2.cvtColor(portrait[:, :, :3], cv2.COLOR_BGR2GRAY)
            mask = portrait[:, :, 3] if portrait.shape[2] == 4 else None
            self.portraits[student_id] = self.sift.detectAndCompute(gray, mask)
        keypoints, descriptors = self.portraits[student_id]
        visible = crop(image, (0.01, 0.07, 0.51, 0.77))
        visible = cv2.resize(visible, None, fx=0.5, fy=0.5)
        target_points, target_desc = self.sift.detectAndCompute(
            cv2.cvtColor(visible, cv2.COLOR_BGR2GRAY), None
        )
        if descriptors is None or target_desc is None:
            return 0
        pairs = cv2.BFMatcher().knnMatch(descriptors, target_desc, k=2)
        matches = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
        ]
        # Repeated source textures must not all vote for one destination point.
        unique = {}
        for match in sorted(matches, key=lambda m: m.distance):
            unique.setdefault(match.trainIdx, match)
        good = list(unique.values())
        if len(good) < 8:
            return 0
        source = np.asarray(
            [keypoints[m.queryIdx].pt for m in good], dtype=np.float32
        ).reshape(-1, 1, 2)
        target = np.asarray(
            [target_points[m.trainIdx].pt for m in good], dtype=np.float32
        ).reshape(-1, 1, 2)
        transform, inliers = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=3
        )
        if transform is None or inliers is None:
            return 0
        count = int(inliers.sum())
        scale = float(np.hypot(transform[0, 0], transform[0, 1]))
        # Reject spatially concentrated coincidences even when they have many matches.
        accepted = source[inliers.ravel() > 0].reshape(-1, 2)
        area = cv2.contourArea(cv2.convexHull(accepted))
        target_area = cv2.contourArea(
            cv2.convexHull(target[inliers.ravel() > 0].reshape(-1, 2))
        )
        return (
            count
            if count / len(good) >= 0.4
            and area > 1500
            and target_area > 500
            and 0.1 < scale < 5
            else 0
        )

    def identify(self, image):
        ranked = self.rank(image)
        best = ranked[0]
        margin = best["score"] - ranked[1]["score"]
        if best["score"] >= 0.72 and margin >= 0.09 and best["icons"][0] >= 0.75:
            return best["id"], ranked[:3], "skill-combination"
        # Verify candidates from symbol ranking against the actual character art.
        candidates = [r for r in ranked[:8] if r["score"] >= best["score"] - 0.15]
        for row in candidates:
            row["portrait_inliers"] = self.portrait_score(image, row["id"])
        candidates.sort(key=lambda r: r.get("portrait_inliers", 0), reverse=True)
        if candidates and candidates[0].get("portrait_inliers", 0) >= 12:
            second = (
                candidates[1].get("portrait_inliers", 0) if len(candidates) > 1 else 0
            )
            if candidates[0]["portrait_inliers"] >= max(second * 1.6, second + 8):
                return candidates[0]["id"], candidates[:3], "portrait-ransac"
        return None, candidates[:3], "unresolved"
