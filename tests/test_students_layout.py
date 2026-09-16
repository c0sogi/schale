"""Registration geometry, original-image evidence, and rejection contracts."""

import cv2
import numpy as np
import pytest

from schale.students.layout import Layout, LayoutMatcher


def test_blank_screen_is_not_registered(student_references):
    assert (
        LayoutMatcher(student_references / "layout_reference.npz").locate(
            np.full((720, 1280, 3), 255, np.uint8)
        )
        is None
    )


def test_clipped_numeric_regions_are_rejected():
    layout = Layout(
        {
            key: np.array([[1.0, 0.0, -80.0], [0.0, 1.0, 0.0]])
            for key in ("left", "right")
        },
        {},
    )
    assert not layout.visible()


def test_original_roi_uses_each_panel_transform():
    layout = Layout(
        {
            "left": np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]]),
            "right": np.array([[1.1, 0.0, -20.0], [0.0, 1.1, -30.0]]),
        },
        {},
    )
    assert layout.source_roi((0.1, 0.2, 0.3, 0.4)) == pytest.approx(
        (0.1 + 10 / 1280, 0.2 + 20 / 720, 0.3 + 10 / 1280, 0.4 + 20 / 720)
    )
    assert layout.source_roi((0.6, 0.2, 0.8, 0.4)) == pytest.approx(
        (0.66 - 20 / 1280, 0.22 - 30 / 720, 0.88 - 20 / 1280, 0.44 - 30 / 720)
    )


def test_identity_normalization_preserves_original_pixels():
    original = np.random.default_rng(1).integers(0, 256, (1440, 2560, 3), np.uint8)
    layout = Layout(
        {
            key: np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            for key in ("left", "right")
        },
        {},
    )
    assert np.array_equal(layout.normalize(original), original)


@pytest.mark.parametrize(
    "scale,dx,dy", [(0.85, 45, 30), (1.12, -10, -25), (1.0, 13, -8)]
)
def test_recovers_known_feature_correspondence_transform(
    monkeypatch, scale, dx, dy, student_references
):
    matcher = LayoutMatcher(student_references / "layout_reference.npz")
    expected = np.array([[scale, 0, dx], [0, scale, dy]])
    reference_points = np.concatenate(
        [matcher.reference[k + "_points"] for k in ("left", "right")]
    )
    descriptors = np.concatenate(
        [matcher.reference[k + "_descriptors"] for k in ("left", "right")]
    )
    target = reference_points * scale + (dx, dy)

    class Features:
        def detectAndCompute(self, image, mask):
            return [cv2.KeyPoint(float(x), float(y), 1) for x, y in target], descriptors

    monkeypatch.setattr(matcher, "sift", Features())
    layout = matcher.locate(np.zeros((720, 1280, 3), np.uint8))
    assert layout is not None
    for matrix in layout.matrices.values():
        assert matrix == pytest.approx(expected, abs=1e-3)
