"""Reference icon downloading, caching, and template matching."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import requests
from numpy.typing import NDArray

from schale.cache_control import cache_collection

logger = logging.getLogger(__name__)

_TEMPLATE_CATEGORIES: frozenset[str] = frozenset(
    {
        "Hat", "Gloves", "Shoes", "Bag", "Badge", "Hairpin", "Charm", "Watch", "Necklace",
        "Exp", "WeaponExpGrowthA", "WeaponExpGrowthB", "WeaponExpGrowthC", "WeaponExpGrowthZ"
    }
)

_ICON_BASE_URL = "https://schaledb.com/images/equipment/icon"
_ICON_CACHE_DIR = (
    Path(os.environ.get("SCHALE_CACHE_DIR") or (Path.home() / ".schale" / "cache"))
    / "icons"
)

# Target size for template matching (width, height)
TARGET_ICON_SIZE: tuple[int, int] = (64, 64)


@dataclass(slots=True)
class IconTemplate:
    """A reference icon prepared for template matching."""

    equipment_id: int
    icon_name: str
    category: str
    tier: int
    is_blueprint: bool
    grayscale: NDArray[np.uint8]
    edges: NDArray[np.uint8]
    alpha_mask: NDArray[np.uint8] | None
    # Feature-based matching: ORB descriptors
    descriptors: NDArray[np.uint8] | None = None


def _download_icon(icon_name: str, *, force: bool = False) -> Path | None:
    """Download an equipment icon from schaledb and cache it locally.

    Args:
        icon_name: The icon identifier (e.g. 'equipment_icon_hat_tier5').
        force: Re-download even if already cached.

    Returns:
        Path to the cached icon file, or None if download failed.
    """
    _ICON_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Try .webp first, then .png as fallback
    for ext in (".webp", ".png"):
        cached_path = _ICON_CACHE_DIR / f"{icon_name}{ext}"
        if cached_path.exists() and not force:
            return cached_path

    for ext in (".webp", ".png"):
        url = f"{_ICON_BASE_URL}/{icon_name}{ext}"
        cached_path = _ICON_CACHE_DIR / f"{icon_name}{ext}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            # Validate Content-Type to avoid saving HTML error pages
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith(('image/', 'application/octet-stream')):
                logger.warning("Invalid content type for %s: %s", url, content_type)
                continue
            cached_path.write_bytes(response.content)
            logger.debug("Downloaded icon: %s", url)
            return cached_path
        except requests.RequestException:
            continue

    logger.warning("Failed to download icon: %s", icon_name)
    return None


def _load_icon_with_mask(
    path: Path,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8] | None] | None:
    """Load an icon image as grayscale and extract its alpha mask.

    WebP icons with transparency are composited over a light gray background
    (matching the game's UI) before converting to grayscale.  The raw alpha
    channel is returned separately so it can be used as a mask during
    ``cv2.matchTemplate``.

    Args:
        path: Path to the icon file.

    Returns:
        ``(grayscale, alpha_mask)`` tuple, where *alpha_mask* is ``None``
        when the source image has no alpha channel.  Returns ``None``
        entirely when the image cannot be read.
    """
    img_buffer = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.warning("Cannot decode icon: %s", path)
        return None

    alpha_mask: NDArray[np.uint8] | None = None

    # Handle alpha channel if present
    if img.shape[-1] == 4:  # BGRA
        bgr = img[..., :3]
        alpha_f = img[..., 3:4] / 255.0
        background = np.full_like(bgr, 240, dtype=np.uint8)
        composited = (bgr * alpha_f + background * (1 - alpha_f)).astype(np.uint8)
        gray: NDArray[np.uint8] = cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY)  # type: ignore[assignment]

        # Build binary mask: pixels with alpha > 0 are considered foreground
        raw_alpha: NDArray[np.uint8] = img[..., 3]  # type: ignore[assignment]
        # cv2.matchTemplate mask must be single-channel with same depth as
        # the template.  A binary 0/255 mask works well here.
        alpha_mask = np.where(raw_alpha > 0, np.uint8(255), np.uint8(0)).astype(np.uint8)  # type: ignore[assignment]

        # Crop to bounding box of opaque pixels (alpha > 128) to reduce false positives
        opaque = raw_alpha > 128
        rows = np.any(opaque, axis=1)
        cols = np.any(opaque, axis=0)

        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # Add small padding
            pad = 2
            rmin = max(0, rmin - pad)
            rmax = min(raw_alpha.shape[0], rmax + pad + 1)
            cmin = max(0, cmin - pad)
            cmax = min(raw_alpha.shape[1], cmax + pad + 1)

            gray = gray[rmin:rmax, cmin:cmax]
            alpha_mask = alpha_mask[rmin:rmax, cmin:cmax]
    elif len(img.shape) == 3:  # BGR, no alpha
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # type: ignore[assignment]
    else:  # Already grayscale
        gray = img  # type: ignore[assignment]

    return gray, alpha_mask  # type: ignore[return-value]


@dataclass
class IconAtlas:
    """Manages reference icons for template matching against inventory cells.

    Downloads equipment icons from schaledb, caches them locally,
    and provides template matching against cell icon regions.
    """

    _templates: dict[str, IconTemplate] = field(
        default_factory=lambda: dict[str, IconTemplate](), init=False
    )
    _prepared: bool = field(default=False, init=False)

    def prepare(self, *, force_download: bool = False) -> None:
        """Download all wearable equipment icons and prepare templates.

        Args:
            force_download: Re-download all icons even if cached.
        """
        if self._prepared and not force_download:
            return

        equipments = cache_collection.equipments
        count = 0
        failed = 0

        for eq_id, eq in equipments.items():
            if eq.Category not in _TEMPLATE_CATEGORIES:
                continue

            icon_name = eq.Icon
            is_blueprint = icon_name.endswith("_piece")

            path = _download_icon(icon_name, force=force_download)
            if path is None:
                failed += 1
                continue

            loaded = _load_icon_with_mask(path)
            if loaded is None:
                failed += 1
                continue

            gray, alpha_mask = loaded

            # Apply CLAHE normalization to templates to match cell processing
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_normalized: NDArray[np.uint8] = clahe.apply(gray)  # type: ignore[assignment]

            # Extract SIFT features for feature-based matching
            # SIFT is more robust than ORB for small icons
            sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.03)  # type: ignore[attr-defined]
            _kp, desc = sift.detectAndCompute(gray_normalized, mask=alpha_mask)

            # Keep templates at native resolution for masked sliding-window
            # matching.  Also compute a 64x64 edge map as a fallback.
            resized: NDArray[np.uint8] = cv2.resize(
                gray_normalized, TARGET_ICON_SIZE, interpolation=cv2.INTER_AREA
            )  # type: ignore[assignment]
            edges: NDArray[np.uint8] = cv2.Canny(resized, 50, 150)  # type: ignore[assignment]

            self._templates[icon_name] = IconTemplate(
                equipment_id=eq_id,
                icon_name=icon_name,
                category=eq.Category,
                tier=eq.Tier,
                is_blueprint=is_blueprint,
                grayscale=gray_normalized,
                edges=edges,
                alpha_mask=alpha_mask,
                descriptors=desc,
            )
            count += 1

        self._prepared = True
        logger.info("Prepared %d icon templates (%d failed)", count, failed)

    def match(
        self,
        cell_roi: NDArray[np.uint8],
        *,
        filter_blueprint: bool | None = None,
        filter_tier: int | None = None,
    ) -> tuple[IconTemplate | None, float]:
        """Match a cell region against all reference templates using single-scale search.

        With resolution normalization at the scan_inventory level, cells are now
        scaled to match template dimensions, so we only need to search a tight
        range around 100% scale (95%-105%).

        Args:
            cell_roi: BGR (or grayscale) cell region of interest --
                the **full** cell, not a pre-extracted icon sub-region.
            filter_blueprint: If set, only match blueprint (True) or
                equipment (False).
            filter_tier: If set, only match templates of this specific tier.

        Returns:
            (best_matching_template, confidence) or (None, 0.0) if no match.
        """
        if not self._templates:
            logger.warning("No icon templates loaded; call prepare() first")
            return None, 0.0

        # Convert the cell ROI to grayscale and apply histogram equalization
        # to normalize brightness/contrast differences between monitor rendering
        # and SchaleDB templates
        if len(cell_roi.shape) == 3:
            gray_roi: NDArray[np.uint8] = cv2.cvtColor(
                cell_roi, cv2.COLOR_BGR2GRAY
            )  # type: ignore[assignment]
        else:
            gray_roi = cell_roi

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # to normalize local brightness variations
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_roi = clahe.apply(gray_roi)  # type: ignore[assignment]

        roi_h, roi_w = gray_roi.shape[:2]

        # Extract icon region (top 80% of cell, almost full width)
        # This excludes tier badge and quantity text at the bottom
        # but includes enough of the icon for feature extraction
        icon_h = int(roi_h * 0.80)
        icon_w = int(roi_w * 0.95)
        margin_x = (roi_w - icon_w) // 2
        icon_roi = gray_roi[0:icon_h, margin_x:margin_x+icon_w]

        # Extract SIFT features from cell icon region
        # Use same parameters as template extraction
        sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.03)  # type: ignore[attr-defined]
        _cell_kp, cell_desc = sift.detectAndCompute(icon_roi, None)

        if cell_desc is None or len(cell_desc) < 5:
            # Not enough features detected in cell
            return None, 0.0

        # Use BFMatcher with L2 distance for SIFT descriptors (float)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        best_template: IconTemplate | None = None
        best_score: float = 0.0

        for template in self._templates.values():
            # Apply filters
            if (
                filter_blueprint is not None
                and template.is_blueprint != filter_blueprint
            ):
                continue
            if filter_tier is not None and template.tier != filter_tier:
                continue

            # Skip templates without descriptors
            if template.descriptors is None or len(template.descriptors) < 10:
                continue

            # Match descriptors using BFMatcher with k=2 for ratio test
            matches = bf.knnMatch(cell_desc, template.descriptors, k=2)

            # Apply Lowe's ratio test to filter good matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    # More strict ratio test for SIFT (better quality features)
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            # Calculate match score based on good matches
            # Need fewer matches when tier is unknown (larger search space)
            min_matches = 5 if filter_tier is None else 7
            if len(good_matches) >= min_matches:
                # Score based on:
                # 1. Number of good matches (absolute count)
                # 2. Match ratio relative to cell features
                # 3. Average distance of good matches (SIFT uses L2 distance)

                # For SIFT, good distance range is typically 0-300
                # Lower is better
                avg_distance = sum(m.distance for m in good_matches) / len(good_matches)

                # Ratio of good matches to cell features (how many cell features matched)
                match_ratio = len(good_matches) / len(cell_desc)

                # Priority 1: Adaptive distance normalization using 90th percentile
                distances = [m.distance for m in good_matches]
                distance_threshold = np.percentile(distances, 90)
                # Use adaptive threshold with minimum of 150 to prevent over-normalization
                adaptive_threshold = max(150.0, distance_threshold)
                distance_score = max(0.0, min(1.0, 1.0 - avg_distance / adaptive_threshold))

                # Priority 2: Rebalanced scoring weights
                # More weight on match ratio (quantity) vs distance quality
                score = min(1.0, 0.6 * match_ratio + 0.4 * distance_score)

                if score > best_score:
                    best_score = score
                    best_template = template

        return best_template, best_score
