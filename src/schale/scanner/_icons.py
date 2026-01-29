"""Reference icon downloading, caching, and template matching."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import requests
from numpy.typing import NDArray

from schale.cache_control import cache_collection
from schale.scanner._phash import compute_hsv_phash, hamming_distance

if TYPE_CHECKING:
    from schale.scanner._cnn import (
        CNNClassifier,  # pyright: ignore[reportAttributeAccessIssue]
    )

from importlib.resources import files

logger = logging.getLogger(__name__)

TEMPLATE_CATEGORIES: frozenset[str] = frozenset(
    {
        "Hat",
        "Gloves",
        "Shoes",
        "Bag",
        "Badge",
        "Hairpin",
        "Charm",
        "Watch",
        "Necklace",
        "Exp",
        "WeaponExpGrowthA",
        "WeaponExpGrowthB",
        "WeaponExpGrowthC",
        "WeaponExpGrowthZ",
    }
)

_ICON_BASE_URL = "https://schaledb.com/images/equipment/icon"

# Icon data is bundled in the package - no external cache needed
# To update icons, use: schale cache update

# Target size for template matching (width, height)
TARGET_ICON_SIZE: tuple[int, int] = (64, 64)


@dataclass(slots=True)
class IconTemplate:
    """A reference icon prepared for CNN-based matching."""

    equipment_id: int
    icon_name: str
    category: str
    tier: int
    is_blueprint: bool
    grayscale: np.ndarray
    edges: np.ndarray
    alpha_mask: NDArray[np.uint8] | None


def _get_package_icon_path(icon_name: str) -> Path | None:
    """Get path to bundled icon in package data.

    Args:
        icon_name: The icon identifier (e.g. 'equipment_icon_hat_tier5').

    Returns:
        Path to the bundled icon file, or None if not found.
    """
    try:
        package_data = files("schale.scanner.data") / "icons"
        for ext in (".webp", ".png"):
            bundled_file = package_data / f"{icon_name}{ext}"
            try:
                if bundled_file.is_file():
                    return Path(str(bundled_file))
            except (OSError, AttributeError):
                continue
    except (ImportError, FileNotFoundError, AttributeError):
        # Package data not available
        pass
    return None


def _download_icon(icon_name: str, *, _force: bool = False) -> Path | None:
    """Get bundled icon from package data.

    Automatic downloads are disabled. Icons are bundled with the package.
    To update icons, use: schale cache update

    Args:
        icon_name: The icon identifier (e.g. 'equipment_icon_hat_tier5').
        _force: Unused (kept for API compatibility).

    Returns:
        Path to the bundled icon file, or None if not found.
    """
    path = _get_package_icon_path(icon_name)
    if path is None:
        logger.warning(
            "Icon not found in package: %s. Run 'schale cache update' to download missing icons.",
            icon_name,
        )
    return path


def download_icon_to_package(icon_name: str, *, force: bool = False) -> Path | None:
    """Download an icon and save it to package data directory.

    This function is ONLY used by the CLI 'schale cache update' command.
    Regular scanner operation uses bundled icons only.

    Args:
        icon_name: The icon identifier (e.g. 'equipment_icon_hat_tier5').
        force: Re-download even if already exists.

    Returns:
        Path to the saved icon file, or None if download failed.
    """
    # Get the package data directory
    try:
        package_data = files("schale.scanner.data") / "icons"
        data_dir = Path(str(package_data))
    except (ImportError, FileNotFoundError, AttributeError):
        # Fallback to src layout for editable installs
        module_dir = Path(__file__).parent
        data_dir = module_dir / "data" / "icons"

    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if already exists
    if not force:
        for ext in (".webp", ".png"):
            cached_path = data_dir / f"{icon_name}{ext}"
            if cached_path.exists():
                return cached_path

    # Download from schaledb
    for ext in (".webp", ".png"):
        url = f"{_ICON_BASE_URL}/{icon_name}{ext}"
        save_path = data_dir / f"{icon_name}{ext}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            # Validate Content-Type to avoid saving HTML error pages
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith(("image/", "application/octet-stream")):
                logger.warning("Invalid content type for %s: %s", url, content_type)
                continue
            save_path.write_bytes(response.content)
            logger.debug("Downloaded icon: %s", url)
            return save_path
        except requests.RequestException as e:
            logger.debug("Failed to download %s: %s", url, e)
            continue

    return None


def _load_icon_with_mask(
    path: Path,
) -> tuple[np.ndarray, NDArray[np.uint8] | None] | None:
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
        # Match cell background better (210 instead of 240)
        background = np.full_like(bgr, 210, dtype=np.uint8)
        composited = (bgr * alpha_f + background * (1 - alpha_f)).astype(np.uint8)
        gray = cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY)

        # Build binary mask: pixels with alpha > 0 are considered foreground
        raw_alpha = img[..., 3]
        # cv2.matchTemplate mask must be single-channel with same depth as
        # the template.  A binary 0/255 mask works well here.
        alpha_mask = np.where(raw_alpha > 0, np.uint8(255), np.uint8(0)).astype(
            np.uint8
        )

        # DISABLED: Bounding box cropping causes size mismatch with cells (64x92 vs 113x136)
        # Templates must remain at 116x146 to match normalized cell size during matching
        # # Crop to bounding box of opaque pixels (alpha > 128) to reduce false positives
        # opaque = raw_alpha > 128
        # rows = np.any(opaque, axis=1)
        # cols = np.any(opaque, axis=0)
        #
        # if rows.any() and cols.any():
        #     rmin, rmax = np.where(rows)[0][[0, -1]]
        #     cmin, cmax = np.where(cols)[0][[0, -1]]
        #
        #     # Add small padding
        #     pad = 2
        #     rmin = max(0, rmin - pad)
        #     rmax = min(raw_alpha.shape[0], rmax + pad + 1)
        #     cmin = max(0, cmin - pad)
        #     cmax = min(raw_alpha.shape[1], cmax + pad + 1)
        #
        #     gray = gray[rmin:rmax, cmin:cmax]
        #     alpha_mask = alpha_mask[rmin:rmax, cmin:cmax]
    elif len(img.shape) == 3:  # BGR, no alpha
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:  # Already grayscale
        gray = img

    return gray, alpha_mask


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
    _cnn: "CNNClassifier | None" = field(default=None, init=False)
    _phashes: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: dict[str, tuple[int, int, int]](), init=False
    )
    _kaze_descriptors: dict[str, tuple[list, NDArray]] = field(
        default_factory=lambda: dict[str, tuple[list, NDArray]](), init=False
    )  # icon_name -> (keypoints, descriptors)

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
            if eq.Category not in TEMPLATE_CATEGORIES:
                continue

            icon_name = eq.Icon
            is_blueprint = icon_name.endswith("_piece")

            path = _download_icon(icon_name, _force=force_download)
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
            gray_normalized = clahe.apply(gray)

            # Keep templates at native resolution for CNN-based matching
            # Also compute a 64x64 edge map for legacy support
            resized = cv2.resize(
                gray_normalized, TARGET_ICON_SIZE, interpolation=cv2.INTER_AREA
            )
            edges = cv2.Canny(resized, 50, 150)

            self._templates[icon_name] = IconTemplate(
                equipment_id=eq_id,
                icon_name=icon_name,
                category=eq.Category,
                tier=eq.Tier,
                is_blueprint=is_blueprint,
                grayscale=gray_normalized,
                edges=edges,
                alpha_mask=alpha_mask,
            )

            # Compute and store HSV tri-hash
            # Load original BGRA icon to compute phash
            img_buffer = np.fromfile(str(path), dtype=np.uint8)
            icon_bgra = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
            if icon_bgra is not None:
                icon_bgr = icon_bgra[:, :, :3].astype(np.uint8)  # Keep BGR format to match cell ROIs
                phash = compute_hsv_phash(icon_bgr, crop=True, multi_crop=False)
                # compute_hsv_phash returns a single tuple when multi_crop=False
                assert isinstance(phash, tuple), "Expected single hash tuple"
                self._phashes[icon_name] = phash

                # Compute KAZE descriptors for feature matching
                kaze = cv2.KAZE_create()  # type: ignore[attr-defined]
                gray = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2GRAY)
                kp, desc = kaze.detectAndCompute(gray, None)
                if desc is not None and len(kp) > 5:
                    self._kaze_descriptors[icon_name] = (kp, desc)

            count += 1

        self._prepared = True
        logger.info("Prepared %d icon templates (%d failed)", count, failed)
        logger.info(f"Computed {len(self._phashes)} phashes for templates")
        logger.info(f"Computed {len(self._kaze_descriptors)} KAZE descriptors for templates")
        if self._phashes:
            sample_name = list(self._phashes.keys())[0]
            sample_hash = self._phashes[sample_name]
            logger.info(f"Sample phash for {sample_name}: H={sample_hash[0]:064b}...")

    def enable_cnn(
        self, model_path: Path | None = None, mapping_path: Path | None = None
    ) -> None:
        """Enable CNN-based matching (requires trained model).

        Args:
            model_path: Path to trained CNN model weights file.
            mapping_path: Path to index-to-icon mapping JSON file.

        Raises:
            ImportError: If CNN module is not available.
        """
        try:
            from schale.scanner._cnn import (
                CNNClassifier,  # pyright: ignore[reportAttributeAccessIssue]
            )
        except ImportError as exc:
            logger.error(
                "CNN module not available. Install required dependencies "
                "or implement schale.scanner._cnn.CNNClassifier"
            )
            raise ImportError(
                "CNN classifier requires additional dependencies. "
                "See schale/scanner/_cnn.py for implementation."
            ) from exc

        self._cnn = (
            CNNClassifier(model_path, mapping_path) if model_path else CNNClassifier()
        )
        logger.info("CNN classifier enabled")

    @property
    def phashes(self) -> dict[str, tuple[int, int, int]]:
        """Precomputed HSV tri-hashes for all templates."""
        return self._phashes

    def match_cnn(
        self,
        cell_roi: NDArray[np.uint8],
        *,
        filter_blueprint: bool | None = None,
        filter_tier: int | None = None,
        confidence_threshold: float = 0.5,
    ) -> tuple[IconTemplate | None, float]:
        """Match using CNN classifier.

        Args:
            cell_roi: BGR (or grayscale) cell region of interest.
            filter_blueprint: If set, only match blueprint (True) or
                equipment (False).
            filter_tier: If set, only match templates of this specific tier.
            confidence_threshold: Minimum CNN confidence to accept match.

        Returns:
            (best_matching_template, confidence) or (None, 0.0) if no match.
        """
        # If CNN not enabled, return no match
        if self._cnn is None:
            logger.warning("CNN not enabled, cannot perform matching")
            return None, 0.0

        # Get top-5 predictions from CNN
        try:
            predictions = self._cnn.predict(cell_roi, top_k=5)
        except Exception as exc:
            logger.error("CNN prediction failed: %s", exc)
            return None, 0.0

        # Apply tier/blueprint filters to predictions
        for icon_name, confidence in predictions:
            if icon_name not in self._templates:
                logger.debug("CNN predicted unknown icon: %s", icon_name)
                continue

            template = self._templates[icon_name]

            # Check filters
            if (
                filter_blueprint is not None
                and template.is_blueprint != filter_blueprint
            ):
                continue
            if filter_tier is not None and template.tier != filter_tier:
                continue

            # If confidence is high enough, return this match
            if confidence >= confidence_threshold:
                logger.debug(
                    "CNN matched %s with confidence %.3f",
                    icon_name,
                    confidence,
                )
                return template, confidence

        # No confident match found
        logger.debug(
            "CNN found no confident match (threshold=%.2f)",
            confidence_threshold,
        )
        return None, 0.0

    def match_template(
        self,
        cell_roi: NDArray[np.uint8],
        filter_blueprint: bool | None = None,
        filter_tier: int | None = None,
        confidence_threshold: float = 0.5,
    ) -> tuple[IconTemplate | None, float]:
        """Match cell against templates using OpenCV template matching.

        Since templates have transparent backgrounds composited to gray 210, and cells
        have varying UI backgrounds, we use simple grayscale matching without masks.
        The tier badge and quantity text in cells create localized mismatches but the
        overall icon pattern should still dominate.

        Args:
            cell_roi: BGR cell region to match.
            filter_blueprint: Filter to blueprint/regular items only.
            filter_tier: Filter to specific tier only.
            confidence_threshold: Minimum matching score (0.0-1.0).

        Returns:
            (best_match, confidence) or (None, score) if no match above threshold.
        """
        # Convert cell to grayscale
        cell_gray = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2GRAY)

        best_score = 0.0
        best_template = None

        for template in self._iter_filtered(filter_blueprint, filter_tier):
            # Get grayscale template
            template_gray = template.grayscale

            # Ensure cell is at least as large as template
            if cell_gray.shape[0] < template_gray.shape[0] or cell_gray.shape[1] < template_gray.shape[1]:
                continue

            # Template matching using normalized cross-correlation
            try:
                result = cv2.matchTemplate(
                    cell_gray,
                    template_gray,
                    cv2.TM_CCOEFF_NORMED,
                )
                _, max_val, _, _ = cv2.minMaxLoc(result)
            except cv2.error:
                # Unexpected error, skip
                continue

            if max_val > best_score:
                best_score = max_val
                best_template = template

        # Always return best match with its score for debugging
        # The caller can decide whether to use it based on threshold
        if best_template is not None:
            logger.debug(
                "Best template match: %s with score %.3f (threshold %.2f)",
                best_template.icon_name,
                best_score,
                confidence_threshold,
            )
            if best_score >= confidence_threshold:
                return (best_template, float(best_score))

        return (None, float(best_score))

    def _iter_filtered(
        self,
        filter_blueprint: bool | None = None,
        filter_tier: int | None = None,
    ):
        """Iterate over templates matching the given filters."""
        for template in self._templates.values():
            # Apply filters
            if filter_blueprint is not None and template.is_blueprint != filter_blueprint:
                continue
            if filter_tier is not None and template.tier != filter_tier:
                continue
            yield template

    def match_phash(
        self,
        cell_roi: NDArray[np.uint8],
        *,
        filter_blueprint: bool | None = None,
        filter_tier: int | None = None,
        confidence_threshold: float = 0.75,
    ) -> tuple[IconTemplate | None, float]:
        """Match cell using HSV perceptual hash with adaptive multi-crop fallback.

        This method uses DCT-based perceptual hashing in HSV color space to find
        the most similar template icon. It's robust to minor variations while
        remaining sensitive to tier color differences.

        Adaptive Multi-Crop Strategy:
            1. First attempt: Single crop ratio (0.65)
            2. If confidence < threshold: Try multiple crop ratios (0.60, 0.65, 0.70, 0.75)
            3. Select best match across all ratios
            This handles icons with varying background ratios

        Confidence Scoring:
            The confidence score is derived from the weighted Hamming distance:
            - Distance components: H (50%), S (30%), V (20%)
            - Max weighted distance: 64.0
            - Confidence formula: 1.0 - (distance / 64.0)
            - Range: [0.0, 1.0] where 1.0 is perfect match

        Filtering Logic:
            - filter_blueprint: When set, only considers templates matching the
              blueprint status (True=blueprint, False=equipment)
            - filter_tier: When set, only considers templates matching the
              specified tier number
            - Filters are applied before distance computation for efficiency

        Args:
            cell_roi: BGR cell region of interest
            filter_blueprint: If set, only match blueprint (True) or equipment (False)
            filter_tier: If set, only match templates of this specific tier
            confidence_threshold: Minimum confidence to accept match (default 0.75)

        Returns:
            (best_matching_template, confidence) or (None, 0.0) if no match
                exceeds the confidence threshold

        Example:
            >>> atlas = IconAtlas()
            >>> atlas.prepare()
            >>> cell = cv2.imread("cell.png")
            >>> template, conf = atlas.match_phash(cell, filter_tier=5)
            >>> if template:
            ...     print(f"Matched {template.icon_name} with {conf:.2%} confidence")
        """
        # First attempt: single crop ratio
        cell_hash_result = compute_hsv_phash(cell_roi, crop=True, multi_crop=False)
        # multi_crop=False returns a single tuple
        assert isinstance(cell_hash_result, tuple), "Expected single hash tuple"
        cell_hash: tuple[int, int, int] = cell_hash_result

        best_template: IconTemplate | None = None
        best_distance: float = float('inf')

        for icon_name, template_hash in self._phashes.items():
            template = self._templates.get(icon_name)
            if template is None:
                continue

            # Apply filters
            if filter_blueprint is not None and template.is_blueprint != filter_blueprint:
                continue
            if filter_tier is not None and template.tier != filter_tier:
                continue

            # Compute distance
            distance = hamming_distance(cell_hash, template_hash)

            if distance < best_distance:
                best_distance = distance
                best_template = template

        # Convert distance to confidence: 1.0 - (distance / max_distance)
        # Max weighted distance: 0.5*64 + 0.3*64 + 0.2*64 = 64
        confidence = 1.0 - (best_distance / 64.0)

        # If initial match fails or has low confidence, try multi-crop
        if confidence < confidence_threshold:
            logger.debug(
                "Initial phash match insufficient (%.3f < %.2f), trying multi-crop",
                confidence,
                confidence_threshold,
            )

            # Compute multiple hashes with different crop ratios
            cell_hashes_result = compute_hsv_phash(cell_roi, crop=True, multi_crop=True)
            # multi_crop=True returns a list of hashes
            assert isinstance(cell_hashes_result, list), "Expected list of hashes"
            cell_hashes: list[tuple[int, int, int]] = cell_hashes_result

            for icon_name, template_hash in self._phashes.items():
                template = self._templates.get(icon_name)
                if template is None:
                    continue

                # Apply filters
                if filter_blueprint is not None and template.is_blueprint != filter_blueprint:
                    continue
                if filter_tier is not None and template.tier != filter_tier:
                    continue

                # Try all crop ratios, keep best distance
                for cell_hash in cell_hashes:
                    distance = hamming_distance(cell_hash, template_hash)

                    if distance < best_distance:
                        best_distance = distance
                        best_template = template

            # Recompute confidence with best multi-crop result
            confidence = 1.0 - (best_distance / 64.0)

            if confidence >= confidence_threshold:
                logger.debug(
                    "Multi-crop improved match to %s with confidence %.3f",
                    best_template.icon_name if best_template else "None",
                    confidence,
                )

        if confidence < confidence_threshold:
            logger.debug(
                "Best phash match %s has insufficient confidence %.3f (threshold=%.2f)",
                best_template.icon_name if best_template else "None",
                confidence,
                confidence_threshold,
            )
            return None, 0.0

        logger.debug(
            "Phash matched %s with confidence %.3f (distance=%.2f)",
            best_template.icon_name if best_template else "None",
            confidence,
            best_distance,
        )
        return best_template, confidence

    def match_orb(
        self,
        cell_roi: NDArray[np.uint8],
        filter_blueprint: bool | None = None,
        filter_tier: int | None = None,
        confidence_threshold: float = 0.3,
    ) -> tuple[IconTemplate | None, float]:
        """Match cell against templates using ORB feature matching.

        ORB is rotation and scale invariant, more robust than template matching.

        Args:
            cell_roi: BGR cell region to match.
            filter_blueprint: Filter to blueprint/regular items only.
            filter_tier: Filter to specific tier only.
            confidence_threshold: Minimum matching score (0.0-1.0).

        Returns:
            (best_match, confidence) or (None, 0.0) if no match.
        """
        # Convert cell to grayscale
        cell_gray = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2GRAY)

        # Create ORB detector
        orb = cv2.ORB_create(nfeatures=500)  # type: ignore[attr-defined]

        # Detect keypoints and descriptors in cell
        cell_kp, cell_desc = orb.detectAndCompute(cell_gray, None)

        if cell_desc is None or len(cell_kp) < 10:
            return (None, 0.0)

        # BFMatcher with Hamming distance (for ORB)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        best_score = 0.0
        best_template = None

        for template in self._iter_filtered(filter_blueprint, filter_tier):
            template_gray = template.grayscale

            # Detect features in template
            template_kp, template_desc = orb.detectAndCompute(template_gray, None)

            if template_desc is None or len(template_kp) < 10:
                continue

            # Match descriptors
            try:
                matches = bf.match(cell_desc, template_desc)
            except cv2.error:
                continue

            if len(matches) < 10:
                continue

            # Calculate confidence based on match count and average distance
            match_count = len(matches)
            avg_distance = sum(m.distance for m in matches) / match_count

            # Normalize: more matches = higher score, lower distance = higher score
            # ORB distance range is typically 0-256
            score = (match_count / 100.0) * (1.0 - avg_distance / 256.0)
            score = min(1.0, score)  # Cap at 1.0

            if score > best_score:
                best_score = score
                best_template = template

        if best_score >= confidence_threshold and best_template is not None:
            return (best_template, float(best_score))

        return (None, 0.0)

    def match_kaze(
        self,
        cell_roi: NDArray[np.uint8],
        *,
        filter_blueprint: bool | None = None,
        filter_tier: int | None = None,
        confidence_threshold: float = 0.30,
    ) -> tuple[IconTemplate | None, float]:
        """Match using KAZE feature descriptors.

        KAZE (Accelerated-KAZE) is a 2D feature detection and description method
        that operates completely in a nonlinear scale space. This makes it more
        robust to noise and blur compared to SIFT while being faster.

        Matching Process:
            1. Extract KAZE keypoints and descriptors from cell ROI
            2. For each template, match descriptors using BFMatcher with L2 norm
            3. Apply Lowe's ratio test (0.7 threshold) to filter good matches
            4. Compute confidence score based on:
               - Match ratio: number of good matches / total cell features (60%)
               - Distance score: normalized average match distance (40%)

        Confidence Scoring:
            - Requires at least 7 good matches to be considered valid
            - Match ratio: proportion of cell features that matched
            - Distance score: 1.0 - (avg_distance / 150.0), capped at [0.0, 1.0]
            - Final score: min(1.0, 0.6 * match_ratio + 0.4 * distance_score)

        Args:
            cell_roi: BGR cell region of interest
            filter_blueprint: If set, only match blueprint (True) or equipment (False)
            filter_tier: If set, only match templates of this specific tier
            confidence_threshold: Minimum confidence to accept match (default 0.30)

        Returns:
            (best_matching_template, confidence) or (None, 0.0) if no match
                exceeds the confidence threshold

        Example:
            >>> atlas = IconAtlas()
            >>> atlas.prepare()
            >>> cell = cv2.imread("cell.png")
            >>> template, conf = atlas.match_kaze(cell, filter_tier=5)
            >>> if template:
            ...     print(f"Matched {template.icon_name} with {conf:.2%} confidence")
        """
        # Extract KAZE features from cell
        kaze = cv2.KAZE_create()  # type: ignore[attr-defined]
        gray = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2GRAY)
        cell_kp, cell_desc = kaze.detectAndCompute(gray, None)

        if cell_desc is None or len(cell_kp) < 5:
            logger.debug("Cell has insufficient KAZE features (%d keypoints)", len(cell_kp) if cell_kp else 0)
            return None, 0.0

        # BF Matcher with L2 distance for KAZE (float descriptors)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        best_template: IconTemplate | None = None
        best_score: float = 0.0
        best_match_count: int = 0

        for icon_name, (_, template_desc) in self._kaze_descriptors.items():
            template = self._templates.get(icon_name)
            if template is None:
                continue

            # Apply filters
            if filter_blueprint is not None and template.is_blueprint != filter_blueprint:
                continue
            if filter_tier is not None and template.tier != filter_tier:
                continue

            # Match descriptors with ratio test
            matches = bf.knnMatch(cell_desc, template_desc, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            # Calculate confidence based on match quality
            if len(good_matches) >= 7:  # Need at least 7 good matches
                # Normalize by number of cell features
                match_ratio = len(good_matches) / len(cell_desc)
                # Normalize by average distance
                avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
                distance_score = max(0.0, 1.0 - avg_distance / 150.0)  # Assume max distance ~150

                # Combined score
                score = min(1.0, 0.6 * match_ratio + 0.4 * distance_score)

                if score > best_score:
                    best_score = score
                    best_template = template
                    best_match_count = len(good_matches)

        if best_score < confidence_threshold:
            logger.debug(
                "Best KAZE match %s has insufficient confidence %.3f (threshold=%.2f)",
                best_template.icon_name if best_template else "None",
                best_score,
                confidence_threshold,
            )
            return None, 0.0

        logger.debug(
            "KAZE matched %s with confidence %.3f (%d good matches)",
            best_template.icon_name if best_template else "None",
            best_score,
            best_match_count,
        )
        return best_template, best_score
