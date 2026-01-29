"""HSV tri-hash perceptual hashing for robust icon matching.

This module implements DCT-based perceptual hashing across HSV color channels
to enable robust icon matching that is resilient to minor visual variations
while remaining sensitive to tier color differences.
"""

import logging
from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def center_crop(image: NDArray[np.uint8], crop_ratio: float = 0.65) -> NDArray[np.uint8]:
    """Extract central region of image to focus on icon, removing background.

    Args:
        image: Input BGR/grayscale image
        crop_ratio: Ratio of image to keep (0.7 = keep center 70%)

    Returns:
        Center-cropped image
    """
    h, w = image.shape[:2]
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    y1, x1 = (h - crop_h) // 2, (w - crop_w) // 2
    return image[y1:y1+crop_h, x1:x1+crop_w]


def _compute_channel_phash(channel: NDArray[np.uint8], hash_size: int = 8) -> int:
    """Compute DCT-based perceptual hash for a single channel.

    Args:
        channel: Single-channel image (H, S, or V channel)
        hash_size: Size of the hash (default 8 for 64-bit hash)

    Returns:
        64-bit integer representing the perceptual hash

    Note:
        The hash is computed by:
        1. Resizing to (hash_size+1) x (hash_size+1)
        2. Applying DCT (Discrete Cosine Transform)
        3. Extracting top-left hash_size x hash_size (low frequencies)
        4. Computing median of DCT coefficients
        5. Creating binary hash: 1 if coefficient > median, 0 otherwise
    """
    # Resize to hash_size+1 for DCT
    # +1 because we'll crop to hash_size x hash_size after DCT
    resized = cv2.resize(
        channel,
        (hash_size + 1, hash_size + 1),
        interpolation=cv2.INTER_AREA
    )

    # Apply DCT (Discrete Cosine Transform)
    # DCT concentrates image information in low frequencies
    dct = cv2.dct(resized.astype(np.float32))

    # Extract top-left hash_size x hash_size (low frequencies)
    # Low frequencies capture the overall structure, ignoring fine details
    dct_low = dct[:hash_size, :hash_size]

    # Compute median of DCT coefficients
    median = np.median(dct_low)

    # Binary hash: 1 if > median, 0 otherwise
    hash_bits = (dct_low > median).flatten()

    # Convert binary array to 64-bit integer
    hash_int = 0
    for i, bit in enumerate(hash_bits):
        if bit:
            hash_int |= (1 << i)

    logger.debug(
        f"Channel hash computed: median={median:.2f}, "
        f"bits_set={np.sum(hash_bits)}/64"
    )

    return hash_int


def compute_hsv_phash(
    image: NDArray[np.uint8],
    crop: bool = True,
    multi_crop: bool = False
) -> Tuple[int, int, int] | list[Tuple[int, int, int]]:
    """Compute HSV tri-hash perceptual hash with optional multi-crop mode.

    Args:
        image: BGR image (OpenCV format)
        crop: If True, apply center cropping
        multi_crop: If True, return hashes for multiple crop ratios

    Returns:
        Single hash tuple or list of hashes if multi_crop=True

    Note:
        - Each channel gets an independent 64-bit perceptual hash
        - Hue channel captures tier color information
        - Saturation channel captures color intensity
        - Value channel captures brightness patterns
        - Center cropping removes background/border interference
        - Multi-crop mode tries multiple ratios for difficult cases

    Example:
        >>> icon = cv2.imread("icon.png")
        >>> h_hash, s_hash, v_hash = compute_hsv_phash(icon, crop=True)
        >>> print(f"H: {h_hash:016x}, S: {s_hash:016x}, V: {v_hash:016x}")

        >>> # Try multiple crop ratios
        >>> hashes = compute_hsv_phash(icon, crop=True, multi_crop=True)
        >>> for h, s, v in hashes:
        ...     print(f"H: {h:016x}, S: {s:016x}, V: {v:016x}")
    """
    if image is None or image.size == 0:
        raise ValueError("Image is empty or None")

    if not crop:
        # No cropping - use full image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        h_hash = _compute_channel_phash(h_channel.astype(np.uint8))
        s_hash = _compute_channel_phash(s_channel.astype(np.uint8))
        v_hash = _compute_channel_phash(v_channel.astype(np.uint8))

        logger.debug(
            f"HSV tri-hash computed (no crop): "
            f"H={h_hash:016x}, S={s_hash:016x}, V={v_hash:016x}"
        )
        return (h_hash, s_hash, v_hash)

    if multi_crop:
        # Try multiple crop ratios for better matching
        ratios = [0.60, 0.65, 0.70, 0.75]
        hashes = []

        for ratio in ratios:
            cropped = center_crop(image, crop_ratio=ratio)
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            h_channel, s_channel, v_channel = cv2.split(hsv)

            h_hash = _compute_channel_phash(h_channel.astype(np.uint8))
            s_hash = _compute_channel_phash(s_channel.astype(np.uint8))
            v_hash = _compute_channel_phash(v_channel.astype(np.uint8))

            hashes.append((h_hash, s_hash, v_hash))

            logger.debug(
                f"HSV tri-hash computed (crop_ratio={ratio:.2f}): "
                f"H={h_hash:016x}, S={s_hash:016x}, V={v_hash:016x}"
            )

        return hashes
    else:
        # Single crop with default ratio
        cropped = center_crop(image, crop_ratio=0.65)
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)

        h_hash = _compute_channel_phash(h_channel.astype(np.uint8))
        s_hash = _compute_channel_phash(s_channel.astype(np.uint8))
        v_hash = _compute_channel_phash(v_channel.astype(np.uint8))

        logger.debug(
            f"HSV tri-hash computed (crop_ratio=0.65): "
            f"H={h_hash:016x}, S={s_hash:016x}, V={v_hash:016x}"
        )

        return (h_hash, s_hash, v_hash)


def hamming_distance(
    hash1: Tuple[int, int, int],
    hash2: Tuple[int, int, int]
) -> float:
    """Compute weighted hamming distance between two HSV tri-hashes.

    Args:
        hash1: First hash tuple (h_hash, s_hash, v_hash)
        hash2: Second hash tuple (h_hash, s_hash, v_hash)

    Returns:
        Weighted hamming distance in range [0, 64]
        - 0 means identical hashes
        - 64 means maximum difference

    Note:
        Weighted combination:
        - Hue: 50% (most important for tier colors)
        - Saturation: 30% (secondary color information)
        - Value: 20% (brightness, least important)

        Raw hamming range per channel is [0, 64]
        Weighted total range is [0, 0.5*64 + 0.3*64 + 0.2*64] = [0, 64]

    Example:
        >>> hash1 = compute_hsv_phash(icon1)
        >>> hash2 = compute_hsv_phash(icon2)
        >>> dist = hamming_distance(hash1, hash2)
        >>> if dist < 10:
        ...     print("Very similar icons")
        >>> elif dist < 20:
        ...     print("Similar icons")
        >>> else:
        ...     print("Different icons")
    """
    # Compute hamming distance for each channel
    # XOR gives 1 where bits differ, count '1's in binary representation
    h_dist = bin(hash1[0] ^ hash2[0]).count('1')
    s_dist = bin(hash1[1] ^ hash2[1]).count('1')
    v_dist = bin(hash1[2] ^ hash2[2]).count('1')

    # Weighted combination
    # Hue is most important for distinguishing tier colors
    weighted_dist = 0.5 * h_dist + 0.3 * s_dist + 0.2 * v_dist

    logger.debug(
        f"Hamming distance: H={h_dist}, S={s_dist}, V={v_dist}, "
        f"weighted={weighted_dist:.2f}"
    )

    return weighted_dist


def is_similar(
    hash1: Tuple[int, int, int],
    hash2: Tuple[int, int, int],
    threshold: float = 15.0
) -> bool:
    """Check if two hashes represent similar images.

    Args:
        hash1: First hash tuple
        hash2: Second hash tuple
        threshold: Maximum distance to consider similar (default 15.0)

    Returns:
        True if hamming distance <= threshold, False otherwise

    Note:
        Suggested thresholds:
        - 10: Very strict (near-identical)
        - 15: Balanced (default)
        - 20: Lenient (allows more variation)
    """
    dist = hamming_distance(hash1, hash2)
    return dist <= threshold
