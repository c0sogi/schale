"""Torch-free inventory text, uncertainty and entry-point contracts."""

import sys
import subprocess
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from typer.testing import CliRunner

from schale.cli import app
from schale.scanner._grid import CellRegion
from schale.scanner._ocr import Quantity, parse_quantity, parse_tier
from schale.schema.scanner import ScannedItem
from schale.vision.text import CtcRecognizer, TextReading, prepare


def test_numeric_prefix_score_never_drops_digits_or_k_multiplier():
    from schale.scanner._ocr import value_confidence

    assert value_confidence(TextReading("x3", 0.4, (0.4, 0.95)), "xX").score == 0.95
    assert (
        value_confidence(TextReading("x13K", 0.3, (0.4, 0.95, 0.3, 0.9)), "xX").score
        == 0.3
    )
    assert (
        value_confidence(TextReading("x13K", 0.2, (0.4, 0.95, 0.9, 0.2)), "xX").score
        == 0.2
    )


def test_grid_is_padding_and_brightness_invariant_for_repeated_cards():
    import cv2
    from schale.scanner._grid import detect_grid

    image = np.full((350, 440, 3), 150, np.uint8)
    for row in range(3):
        for col in range(3):
            x, y = 25 + col * 130, 20 + row * 105
            cv2.rectangle(image, (x, y), (x + 110, y + 90), (255, 255, 255), -1)
            cv2.circle(image, (x + 55, y + 37), 20, (40, 60, 160), -1)
            cv2.putText(
                image,
                "x123",
                (x + 40, y + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (35, 35, 35),
                2,
            )
    original = detect_grid(image)
    assert len(original) == 9
    changed = cv2.copyMakeBorder(
        (image * 0.75).astype(np.uint8), 40, 20, 60, 20, cv2.BORDER_CONSTANT
    )
    transformed = detect_grid(changed.astype(np.uint8))
    assert [(c.row, c.col) for c in transformed] == [(c.row, c.col) for c in original]
    assert all(
        abs(b.x - a.x - 60) <= 1 and abs(b.y - a.y - 40) <= 1
        for a, b in zip(original, transformed, strict=True)
    )


def test_blank_frame_cannot_become_an_inventory():
    from schale.scanner._grid import detect_grid

    with pytest.raises(RuntimeError, match="card"):
        detect_grid(np.full((400, 600, 3), 200, np.uint8))


@pytest.mark.parametrize(
    "text,value,display,status",
    [
        ("x104", 104, 104, "exact"),
        ("x0", 0, 0, "exact"),
        ("1,234", 1234, 1234, "exact"),
        ("×87", 87, 87, "exact"),
        ("x13K", None, 13000, "abbreviated"),
        ("x1.2k", None, 1200, "abbreviated"),
        ("", None, None, "unreadable"),
        ("0x104", None, None, "unreadable"),
        ("x1O4", None, None, "unreadable"),
        ("x1,23", None, None, "unreadable"),
        ("-3", None, None, "unreadable"),
        ("x1.5", None, None, "unreadable"),
        ("2x40", None, None, "unreadable"),
    ],
)
def test_quantity_keeps_precision_and_rejects_garbage(text, value, display, status):
    assert parse_quantity(text) == Quantity(value, display, status)


@pytest.mark.parametrize(
    "text,value",
    [("T10", 10), ("T1", 1), ("T0", None), ("T11", None), ("13K", None), ("0T3", None)],
)
def test_tier_parse(text, value):
    assert parse_tier(text) == value


def test_shared_ctc_decoder_keeps_repeated_characters_separated_by_blank():
    model = CtcRecognizer.__new__(CtcRecognizer)
    model.charset = ["", "1", "x"]
    probabilities = np.zeros((1, 6, 3), np.float32)
    for index, character in enumerate([2, 2, 1, 0, 1, 1]):
        probabilities[0, index, character] = 0.9
    model.session = Mock()
    model.session.run.return_value = [probabilities]
    reading = model.infer(np.zeros((1, 3, 32, 128)))[0]
    assert reading.text == "x11" and reading.score == pytest.approx(0.9)


def test_empty_crop_is_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        prepare(np.zeros((0, 40, 3), np.uint8))


def test_two_conflicting_quantity_views_are_not_accepted():
    from schale.scanner._ocr import InventoryTextReader

    reader = InventoryTextReader.__new__(InventoryTextReader)
    reader.threshold = 0.65
    reader.recognizer = Mock()
    reader.recognizer.read.side_effect = [
        [TextReading("T3", 0.9), TextReading("T3", 0.9)],
        [TextReading("x13", 0.95), TextReading("x18", 0.99)],
    ]
    result = reader.read_cells([np.full((158, 195, 3), 245, np.uint8)])[0]
    assert result["quantity"] == Quantity(None, None, "unreadable")
    assert len(result["quantity_views"]) == 2


def test_scanner_import_never_loads_torch_or_easyocr():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import schale.scanner; assert not {'torch','easyocr'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_inventory_entrypoint_retains_failed_quantity_and_honors_identity_threshold(
    monkeypatch,
):
    import schale.scanner as module
    from types import SimpleNamespace

    cell = CellRegion(0, 0, 0, 0, 120, 100)
    monkeypatch.setattr(module, "detect_grid", lambda image: [cell])
    reading = dict(
        tier=3,
        tier_reading=TextReading("T3", 0.9),
        quantity=Quantity(None, None, "unreadable"),
        quantity_reading=TextReading("garbage", 0.9),
        tier_box=(0, 70, 40, 100),
        quantity_box=(40, 70, 120, 100),
    )
    monkeypatch.setattr(
        module,
        "InventoryTextReader",
        lambda directory: SimpleNamespace(read_cells=lambda cells: [reading]),
    )
    template = SimpleNamespace(
        equipment_id=1, category="Watch", tier=3, is_blueprint=True, icon_name="watch"
    )
    atlas = SimpleNamespace(_templates={"watch": template})
    monkeypatch.setattr(module, "_get_atlas", lambda **kwargs: atlas)
    matcher = Mock()
    matcher.rank.return_value = [{"icon_name": "watch", "score": 0.8}]
    monkeypatch.setattr(module, "_matcher", matcher)
    result = module.scan_inventory(np.zeros((100, 120, 3), np.uint8))
    assert result.items[0].quantity is None
    assert result.items[0].quantity_status == "unreadable"
    assert result.observations[0].issues == ["unreadable_quantity"]
    rejected = module.scan_inventory(
        np.zeros((100, 120, 3), np.uint8), confidence_threshold=0.9
    )
    assert not rejected.items and rejected.unrecognized_cells == [(0, 0)]
    assert rejected.observations[0].quantity_text == "garbage"


def test_abbreviated_quantity_cannot_be_reported_as_exact():
    with pytest.raises(ValueError, match="Only exact"):
        ScannedItem(
            equipment_id=1,
            category="Exp",
            tier=0,
            quantity=13000,
            quantity_status="abbreviated",
            is_blueprint=False,
            icon_name="exp",
            confidence=0.9,
            grid_position=(0, 0),
        )


def test_cli_does_not_sum_abbreviations(tmp_path: Path, monkeypatch):
    import schale.scanner as module
    from schale.schema.scanner import ScanResult, CellEvidence

    item = ScannedItem(
        equipment_id=1,
        category="Exp",
        tier=0,
        quantity=None,
        quantity_status="abbreviated",
        quantity_text="x13K",
        displayed_quantity=13000,
        is_blueprint=False,
        icon_name="exp",
        confidence=0.9,
        grid_position=(0, 0),
    )
    observation = CellEvidence(
        grid_position=(0, 0),
        box=(0, 0, 1, 1),
        tier_box=(0, 0, 1, 1),
        quantity_box=(0, 0, 1, 1),
        tier_text="",
        tier_score=0,
        quantity_text="x13K",
        quantity_score=0.9,
        tier_value=None,
        quantity_value=None,
        quantity_status="abbreviated",
        displayed_quantity=13000,
        equipment_id=1,
        candidates=[],
        issues=["abbreviated_quantity"],
    )
    monkeypatch.setattr(
        module,
        "scan_inventory",
        lambda *args, **kwargs: ScanResult(
            items=[item],
            observations=[observation],
            unrecognized_cells=[],
            grid_dimensions=(1, 1),
            source_resolution=(1, 1),
        ),
    )
    source = tmp_path / "image.png"
    source.write_bytes(b"fixture")
    output = tmp_path / "results.json"
    result = CliRunner().invoke(app, ["scan", str(source), "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert "0 exact-count subtotal" in result.output and "abbreviated" in result.output
    assert '"quantity": null' in output.read_text()
