"""Generated contract fixtures contain no game images or learned model weights."""

import json

import cv2
import numpy as np
import pytest


@pytest.fixture
def student_references(tmp_path):
    directory = tmp_path / "references"
    directory.mkdir()
    rng = np.random.default_rng(42)
    for name in (
        "header",
        "empty_gear",
        "ghost_gloves",
        "ghost_hat",
        "ghost_shoes",
        "potential_hp_25",
        "potential_attack_25",
        "level_fixture",
    ):
        shape = (141, 159, 3) if name.startswith("ghost_") else (40, 160, 3)
        patch = rng.integers(0, 256, shape, dtype=np.uint8)
        assert cv2.imwrite(str(directory / f"{name}.png"), patch)
    (directory / "labels.json").write_text(
        json.dumps(
            [
                {"family": "level", "value": 1, "image": "level_fixture.png"},
            ]
        ),
        encoding="utf-8",
    )
    reference = {}
    for panel, (x0, x1) in {"left": (25, 580), "right": (660, 1080)}.items():
        reference[panel + "_points"] = rng.uniform(
            (x0, 100), (x1, 610), (50, 2)
        ).astype(np.float32)
        reference[panel + "_descriptors"] = rng.uniform(0, 255, (50, 128)).astype(
            np.float32
        )
    np.savez(directory / "layout_reference.npz", **reference)
    return directory
