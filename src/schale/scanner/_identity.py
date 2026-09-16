"""Catalog icon verification: CNN proposals, local features, and aligned color."""

import cv2
import numpy as np

from ._icons import IconAtlas


class InventoryMatcher:
    def __init__(self, atlas: IconAtlas):
        self.atlas = atlas
        atlas.enable_cnn()
        self.sift = cv2.SIFT.create(contrastThreshold=0.02)
        self.references = {}
        for name in atlas._templates:
            path = atlas._templates[name].path
            icon = cv2.imdecode(
                np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED
            )
            if icon is None:
                continue
            h, w = icon.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            mask[int(h * 0.10) : int(h * 0.72), int(w * 0.18) : int(w * 0.75)] = 255
            if icon.shape[2] == 4:
                mask[icon[:, :, 3] < 240] = 0
            bgr = icon[:, :, :3].copy()
            points, descriptors = self.sift.detectAndCompute(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), mask
            )
            self.references[name] = (bgr, mask, points, descriptors)

    def rank(self, cell, tier: int | None) -> list[dict]:
        points, descriptors = self.sift.detectAndCompute(
            cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY), None
        )
        features = {}
        if descriptors is not None and len(descriptors) >= 2:
            for name, (_, _, keypoints, reference) in self.references.items():
                if reference is None or (
                    tier is not None and self.atlas._templates[name].tier != tier
                ):
                    continue
                pairs = cv2.BFMatcher().knnMatch(reference, descriptors, k=2)
                good = [
                    p[0]
                    for p in pairs
                    if len(p) == 2 and p[0].distance < 0.7 * p[1].distance
                ]
                unique = {}
                for match in sorted(good, key=lambda m: m.distance):
                    unique.setdefault(match.trainIdx, match)
                good = list(unique.values())
                if len(good) < 4:
                    continue
                a = np.asarray(
                    [keypoints[m.queryIdx].pt for m in good], dtype=np.float32
                )
                b = np.asarray([points[m.trainIdx].pt for m in good], dtype=np.float32)
                matrix, inliers = cv2.estimateAffinePartial2D(
                    a, b, method=cv2.RANSAC, ransacReprojThreshold=2.5
                )
                if matrix is None or inliers is None:
                    continue
                count = int(inliers.sum())
                scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
                if count >= 4 and count / len(good) >= 0.6 and 0.2 < scale < 5:
                    features[name] = (count, matrix)
        assert self.atlas._cnn is not None
        cnn = self.atlas._cnn.predict(cell, top_k=min(12, self.atlas._cnn.num_classes))
        proposals = {name for name, _ in cnn}
        proposals.update(
            sorted(features, key=lambda n: features[n][0], reverse=True)[:8]
        )
        scores = []
        for name in proposals:
            if name not in self.references:
                continue
            template = self.atlas._templates[name]
            if tier is not None and template.tier != tier:
                continue
            reference, mask, _, _ = self.references[name]
            count, matrix = features.get(name, (0, None))
            appearance = self._appearance(reference, mask, cell, matrix)
            # Smooth spheres have few local features; color and the CNN provide
            # independent evidence without counting shared shape as a unique ID.
            cnn_score = dict(cnn).get(name, 0.0)
            score = 0.8 * appearance + 0.1 * min(count / 12, 1) + 0.1 * cnn_score
            if template.category in {
                "Exp",
                "WeaponExpGrowthA",
                "WeaponExpGrowthB",
                "WeaponExpGrowthC",
                "WeaponExpGrowthZ",
            }:
                score = 0.85 * appearance + 0.15 * cnn_score
            scores.append(
                dict(
                    icon_name=name,
                    score=round(score, 6),
                    appearance=round(appearance, 6),
                    inliers=count,
                    cnn_score=round(cnn_score, 6),
                )
            )
        return sorted(scores, key=lambda row: (-row["score"], row["icon_name"]))

    @staticmethod
    def _appearance(reference, mask, cell, matrix) -> float:
        def compare(source, target, valid):
            a, b = source[valid].astype(np.float32), target[valid].astype(np.float32)
            if len(a) < 30:
                return 0.0
            a0, b0 = a - a.mean(axis=0), b - b.mean(axis=0)
            denominator = float(np.linalg.norm(a0) * np.linalg.norm(b0))
            correlation = (
                float((a0 * b0).sum() / denominator) if denominator > 0 else 0.0
            )
            color = max(0.0, 1 - float(np.abs(a - b).mean()) / 100)
            return max(0.0, 0.65 * correlation + 0.35 * color)

        best = 0.0
        if matrix is not None:
            h, w = reference.shape[:2]
            aligned = cv2.warpAffine(
                cell, matrix, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
            )
            visible = cv2.warpAffine(
                np.full(cell.shape[:2], 255, np.uint8),
                matrix,
                (w, h),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            )
            valid = (mask > 0) & (visible > 0)
            best = compare(reference, aligned, valid)
        # Color template search also covers textureless icons and feature failure.
        x, y, w, h = cv2.boundingRect(mask)
        ref = reference[y : y + h, x : x + w]
        alpha = mask[y : y + h, x : x + w]
        search = cell[: int(cell.shape[0] * 0.8)]
        for factor in (0.85, 0.95, 1.05, 1.15, 1.25):
            scale = cell.shape[1] / reference.shape[1] * factor
            patch = cv2.resize(ref, None, fx=scale, fy=scale)
            if patch.shape[0] > search.shape[0] or patch.shape[1] > search.shape[1]:
                continue
            support = cv2.resize(
                alpha, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            similarity = cv2.matchTemplate(
                search, patch, cv2.TM_CCOEFF_NORMED, mask=support
            )
            similarity[~np.isfinite(similarity)] = -1
            _, _, _, location = cv2.minMaxLoc(similarity)
            xx, yy = location
            target = search[yy : yy + patch.shape[0], xx : xx + patch.shape[1]]
            best = max(best, compare(patch, target, support > 0))
        return best
