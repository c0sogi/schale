"""Production CTC contracts; no downloaded model or Torch needed for unit tests."""

import hashlib
import json
from typing import cast

import numpy as np
import pytest

from schale.students.numeric import (
    PROFILE,
    install_bundle,
    localize,
    numeric_cell,
    parse,
    prepare,
    validate_bundle,
)


@pytest.mark.parametrize(
    "text,field,expected",
    [
        ("Lv.54", "level", 54),
        ("Lv,54", "level", 54),
        ("Lv70", "equipment2_level", 70),
        ("MAX", "ex", 5),
        ("MAX", "sub", 10),
        ("T10", "equipment1", 10),
        ("Lv.1", "sub", 1),
        ("50", "bond", 50),
        ("T3", "gear", None),
        ("1Lv.14", "basic", None),
        ("Lv.1.", "passive", None),
        ("T60", "equipment1", None),
        ("Lv.6", "ex", None),
        ("MAX", "level", None),
        ("Lv.25", "potential_heal", 25),
        ("25", "potential_heal", 25),
        ("26", "potential_hp", None),
        ("25", "level", None),
        ("Lv.26", "potential_hp", None),
        ("MAX", "potential_attack", None),
        ("Lv.S4", "level", None),
        ("Lv,l", "level", None),
        ("Lv,54-", "level", None),
        ("54|", "bond", None),
        ("", "gear", None),
    ],
)
def test_full_string_parse(text, field, expected):
    assert parse(text, field) == expected


def test_preserve_equipment_color_and_rgb_normalization():
    patch = np.zeros((36, 130, 3), np.uint8)
    patch[:, :, 2] = 255
    assert np.array_equal(localize(patch, "equipment1_level"), patch)
    tensor = prepare(patch)
    assert tensor.shape == (3, 32, 128)
    assert float(tensor[0].mean()) == 1
    assert float(tensor[2].mean()) == -1


def test_numeric_cell_preserves_complete_digit_cell_and_rgb():
    patch = np.arange(36 * 131 * 3, dtype=np.uint8).reshape(36, 131, 3)
    cell = numeric_cell(patch, "equipment1_level")
    assert cell is not None
    assert np.array_equal(cell, patch[5:33, 58:100])
    assert numeric_cell(patch, "level") is None
    assert numeric_cell(patch[:20], "equipment1_level") is None


@pytest.mark.parametrize(
    "primary,secondary,field,expected",
    [
        ("Lv.70", "7", "equipment1_level", None),
        ("Lvl", "1", "equipment1_level", 1),
        ("Lv.1", "11", "equipment1_level", None),
        ("Lv70-", "70", "equipment1_level", 70),
        ("Lv.1", "Lv1", "level", 1),
        ("Lv.1", "Lv11", "level", None),
    ],
)
def test_numeric_cell_does_not_discard_conflicting_primary_digits(
    primary, secondary, field, expected
):
    from schale.students.numeric import NumericReader
    from unittest.mock import Mock

    reader = NumericReader.__new__(NumericReader)
    reader.charset = [""] + sorted(set(primary + secondary))
    reader.threshold = 0.436

    def tensor(text, confidence):
        values = np.zeros((1, len(text) * 2, len(reader.charset)), np.float32)
        for i, character in enumerate(text):
            values[0, 2 * i, reader.charset.index(character)] = confidence
            values[0, 2 * i + 1, 0] = 1
        return values

    reader.session = Mock()
    reader.session.run.side_effect = [[tensor(primary, 0.3)], [tensor(secondary, 0.9)]]
    result = reader.read_patches({field: np.zeros((54, 131, 3), np.uint8)})
    assert result[field][0] == expected


def test_tier_localization_excludes_blue_item_art_touching_top():
    patch = np.full((46, 103, 3), 245, np.uint8)
    patch[0:24, 65:103] = (180, 90, 10)
    patch[13:36, 26:55] = (180, 90, 10)
    result = localize(patch, "gear")
    assert result.shape[1] < 40
    assert result.shape[0] < 35


def test_bond_localization_removes_heart_background_not_digit():
    patch = np.full((59, 69, 3), (190, 150, 245), np.uint8)
    patch[15:38, 25:40] = (70, 50, 40)
    result = localize(patch, "bond")
    assert result.shape[0] < 45
    assert result.shape[1] < 40


def test_slot_state_requires_filled_marker_and_detects_unread_tier():
    import cv2
    from schale.students.vision import tier_badge_present, unequipped_marker

    patch = np.full((36, 44, 3), 245, np.uint8)
    assert not unequipped_marker(patch)
    cv2.circle(patch, (22, 18), 10, (0, 190, 250), -1)
    assert unequipped_marker(patch)
    assert not tier_badge_present(patch)
    patch[20:30, 5:15] = (180, 90, 10)
    assert tier_badge_present(patch)


def test_level_template_cannot_win_on_shared_lv_prefix():
    import cv2
    from schale.students.labels import digit_similarity, similarity

    actual = np.zeros((40, 160), np.uint8)
    wrong = np.zeros_like(actual)
    cv2.putText(actual, "Lv.20", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
    cv2.putText(wrong, "Lv.26", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
    assert similarity(actual, wrong) > 0.90
    assert digit_similarity(actual, wrong) < 0.90
    assert digit_similarity(actual, actual) > 0.99


def test_bundle_install_and_corruption(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"fixture")
    metadata = {
        "schema_version": 1,
        "profile": PROFILE,
        "charset": ["", "1"],
        "threshold": 0.436,
        "model_sha256": hashlib.sha256(b"fixture").hexdigest(),
    }
    (source / "metadata.json").write_text(json.dumps(metadata))
    target = tmp_path / "installed"
    install_bundle(source, target)
    assert validate_bundle(target) == metadata
    with pytest.raises(ValueError, match="not empty"):
        install_bundle(source, target)
    (target / "model.onnx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        validate_bundle(target)


def test_bad_profile_is_rejected_before_model_loading(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"schema_version": 1, "profile": "wrong-ui"})
    )
    with pytest.raises(ValueError, match="profile"):
        validate_bundle(tmp_path)


def test_potential_localization_excludes_neighboring_white_text():
    patch = np.full((72, 120, 3), (150, 85, 55), np.uint8)
    patch[3:17, 60:110] = 255
    patch[35:55, 12:26] = (0, 230, 240)
    patch[35:55, 36:50] = (0, 230, 240)
    localized = localize(patch, "potential_heal")
    assert localized.shape[0] == 26
    assert not np.any(np.all(localized == 255, axis=2))


def test_confident_reader_disagreement_survives_temporal_consensus(
    monkeypatch, tmp_path, student_references
):
    from schale.students.extract import consensus
    from schale.students.labels import BOXES
    from schale.students import vision as module

    class Identity:
        catalog = {"10001": {"Name": "test student"}}

    class Numbers:
        method = "svtrv2-s-rctc"

        def read_all(self, image):
            return {
                field: (1, 0.86, "CTC 1") if field == "bond" else (None, 0.0, "no text")
                for field in BOXES
            }

    reader = module.Vision(
        cast(module.IdentityMatcher, Identity()),
        data=student_references,
        numeric=Numbers(),
    )
    from schale.students.layout import Layout

    monkeypatch.setattr(
        reader.layout,
        "locate",
        lambda image: Layout(
            {
                k: np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                for k in ("left", "right")
            },
            {},
        ),
    )
    monkeypatch.setattr(
        module, "read_image", lambda path: np.full((1440, 2560, 3), 255, np.uint8)
    )
    monkeypatch.setattr(
        module,
        "match_patch",
        lambda image, box, template: 1.0 if box == module.HEADER else 0.0,
    )
    monkeypatch.setattr(
        reader.labels,
        "read",
        lambda image, field: (7, 0.99, "NCC 7")
        if field == "bond"
        else (None, 0, "blank"),
    )
    rows = [
        reader.analyze(
            tmp_path / name, name, i / 30, "test", identity_hint=(10001, [], "fixture")
        )
        for i, name in enumerate(("a.png", "b.png"))
    ]
    result = consensus(rows, "test")
    assert result.fields["bond"].value is None
    assert result.fields["bond"].status == "conflict"
    assert {e.method for e in result.fields["bond"].evidence} == {
        "svtrv2-s-rctc",
        "visual-template-disagreement",
    }
