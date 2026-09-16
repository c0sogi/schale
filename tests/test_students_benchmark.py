"""Benchmark accounting, leakage guards, and production-preprocessing parity."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from schale.students.labels import BOXES, clean_tier, family, label_mask
from schale.students.video import crop

torch = pytest.importorskip(
    "torch", reason="Install the training group for CNN benchmarks"
)

from schale.students.benchmark import (  # noqa: E402 - requires optional Torch first
    CONDITIONS,
    Encoder,
    MaskReader,
    calibrate,
    contrastive_loss,
    metrics,
    perturb,
    validate_dataset,
)


def test_masks_match_production():
    rng = np.random.default_rng(5)
    image = rng.integers(0, 255, (1440, 2560, 3), dtype=np.uint8)
    reader = MaskReader()
    for field, box in BOXES.items():
        expected = label_mask(image, field)
        if expected is not None and family(field) == "tier":
            expected = clean_tier(expected)
        actual = reader.read(crop(image, box), field)
        if expected is None:
            assert actual is None
        else:
            assert actual is not None and np.array_equal(actual, expected)


def test_stress_conditions_keep_dimensions_and_are_repeatable():
    image = np.random.default_rng(1).integers(0, 255, (44, 101, 3), dtype=np.uint8)
    for condition in CONDITIONS:
        a = perturb(image, condition)
        assert a.shape == image.shape and a.dtype == np.uint8
        assert np.array_equal(a, perturb(image, condition))
    with pytest.raises(ValueError):
        perturb(image, "typo")


def test_metric_denominators_and_no_false_success_on_abstention():
    rows = [
        dict(value=v, prediction=p, score=s, margin=0.2)
        for v, p, s in [
            (1, "1", 0.99),
            (2, "3", 0.99),
            (4, "4", 0.2),
            (None, "8", 0.99),
            (None, None, 0),
        ]
    ]
    m = metrics(rows, (0.9, 0.1))
    assert (m["numeric_n"], m["correct"], m["wrong"], m["abstained"]) == (3, 1, 1, 1)
    assert m["accepted_accuracy"] == 0.5
    assert m["control_n"] == 2 and m["control_false_accepts"] == 1
    none = metrics(rows, (1.01, 1))
    assert none["correct_rate"] == 0 and none["accepted_accuracy"] is None


def test_calibration_must_account_for_empty_controls():
    rows = [
        dict(value=1, prediction="1", score=0.95, margin=0.2),
        dict(value=None, prediction="1", score=0.99, margin=0.3),
    ]
    threshold = calibrate(rows)
    assert metrics(rows, threshold)["control_false_accepts"] == 0
    assert metrics(rows, threshold)["coverage"] == 0


def test_group_split_and_content_integrity(tmp_path: Path):
    image = tmp_path / "a.png"
    image.write_bytes(b"test")
    digest = hashlib.sha256(b"test").hexdigest()
    row = dict(
        key="a",
        visit=2,
        split="calibration",
        image="a.png",
        image_sha256=digest,
        value=1,
    )
    data: dict = dict(reference_student_visits_excluded=[1], rows=[row])
    refs = [dict(source="student_001_time.png")]
    validate_dataset(tmp_path, data, refs)
    data["rows"].append({**row, "key": "b", "split": "test"})
    with pytest.raises(ValueError, match="leakage"):
        validate_dataset(tmp_path, data, refs)
    data["rows"] = [row]
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="modified"):
        validate_dataset(tmp_path, data, refs)


def test_embedding_and_contrastive_gradient():
    torch.set_num_threads(2)
    model = Encoder(1)
    z = model(torch.rand(4, 1, 40, 160))
    assert z.shape == (4, 128)
    assert torch.allclose(z.norm(dim=1), torch.ones(4), atol=1e-5)
    loss = contrastive_loss(z, torch.tensor([0, 0, 1, 1]))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
