"""Input normalization, content-addressing and independent-image evidence."""

import json
import os

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from schale import ImageBatch, ImageInput
from schale.cli import app
from schale.inputs import prepare_images, resolve_input


def png(value=40):
    ok, data = cv2.imencode(".png", np.full((36, 64, 3), value, dtype=np.uint8))
    assert ok
    return data.tobytes()


def test_directory_list_bytes_manifest_single_and_duplicate_inputs(tmp_path):
    a, b = tmp_path / "image2.png", tmp_path / "image10.png"
    a.write_bytes(png(20))
    b.write_bytes(png(80))
    directory = resolve_input(tmp_path)
    assert [i.name for i in directory.images] == ["image2.png", "image10.png"]
    assert directory.fingerprint == resolve_input([a, b]).fingerprint
    assert resolve_input(a).fingerprint == resolve_input(a.read_bytes()).fingerprint
    assert resolve_input([a, b]).fingerprint != resolve_input([b, a]).fingerprint
    manifest = tmp_path / "images.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "schale.images",
                "schema_version": 1,
                "images": [{"path": a.name}, {"path": b.name}],
            }
        )
    )
    assert resolve_input(manifest).fingerprint == directory.fingerprint
    source = resolve_input(
        ImageBatch([a, ImageInput(png(80), timestamp=2.5), a.read_bytes()])
    )
    plan = prepare_images(source, tmp_path / "out")
    assert plan["frame_count"] == 2
    assert len(plan["discarded"]) == 1
    assert plan["segments"][0]["frames"][0]["timestamp_known"] is False
    assert plan["segments"][1]["frames"][0]["timestamp"] == 2.5
    assert plan["timeline_unit"] == "image_index"


def test_renaming_does_not_change_content_identity_and_same_stat_changes_detected(
    tmp_path,
):
    path = tmp_path / "one.png"
    path.write_bytes(png(50))
    original = resolve_input(path).fingerprint
    renamed = tmp_path / "two.png"
    path.rename(renamed)
    assert resolve_input(renamed).fingerprint == original
    before = renamed.stat()
    raw = bytearray(renamed.read_bytes())
    raw[-20] ^= 1
    renamed.write_bytes(raw)
    os.utime(renamed, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert renamed.stat().st_size == before.st_size
    assert resolve_input(renamed).fingerprint != original


def test_duplicate_pixels_with_different_encodings_cannot_confirm(tmp_path):
    image = np.full((36, 64, 3), 40, np.uint8)
    _, first = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    _, second = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    assert first.tobytes() != second.tobytes()
    plan = prepare_images(resolve_input([first.tobytes(), second.tobytes()]), tmp_path)
    assert plan["frame_count"] == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "2"])
def test_bad_timestamps_rejected(value):
    with pytest.raises(ValueError):
        resolve_input(ImageInput(png(), timestamp=value))


def test_empty_bad_changed_input_rejected(tmp_path):
    with pytest.raises(ValueError, match="No screenshots"):
        resolve_input([])
    with pytest.raises(ValueError, match="Invalid encoded"):
        prepare_images(resolve_input(b"not image"), tmp_path / "bad")
    path = tmp_path / "a.png"
    path.write_bytes(png())
    resolved = resolve_input(path)
    path.write_bytes(png(70))
    with pytest.raises(ValueError, match="Input changed"):
        prepare_images(resolved, tmp_path / "changed")


def test_same_basename_images_do_not_overwrite(tmp_path):
    paths = []
    for i in range(2):
        path = tmp_path / str(i) / "same.png"
        path.parent.mkdir()
        path.write_bytes(png(30 + i))
        paths.append(path)
    plan = prepare_images(resolve_input(paths), tmp_path / "output")
    assert len({s["frames"][0]["image"] for s in plan["segments"]}) == 2


def test_cli_multiple_images_reaches_public_extractor(tmp_path, monkeypatch):
    import schale.students.extract as module
    import schale.setup as setup
    from schale.students.models import Extraction

    paths = [tmp_path / "a.png", tmp_path / "b.png"]
    for path in paths:
        path.write_bytes(png())
    calls = []

    def extract(source, output, **kwargs):
        calls.append(source)
        return Extraction(
            source="test",
            catalog_sha256="a",
            catalog_url="b",
            diagnostics={"field_statuses": {}},
        )

    monkeypatch.setattr(module, "extract", extract)
    monkeypatch.setattr(setup, "ensure_student_runtime", lambda *args, **kwargs: None)
    result = CliRunner().invoke(
        app,
        ["students", "extract", *map(str, paths), "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert calls == [paths]


def test_image_inspector_uses_saved_evidence_without_video(tmp_path, monkeypatch):
    import hashlib
    from schale.students.inspector import build_inspector
    from schale.students.models import Extraction, Student

    monkeypatch.setenv("SCHALE_CACHE_DIR", str(tmp_path / "empty-cache"))
    extraction = tmp_path / "extraction"
    plan = prepare_images(resolve_input([png(30), png(60)]), extraction)
    (extraction / "segments.json").write_text(json.dumps(plan))
    observation_dir = extraction / "observations"
    observation_dir.mkdir()
    rows = []
    for segment in plan["segments"]:
        frame = segment["frames"][0]
        path = extraction / frame["image"]
        row = Student(key=segment["key"], screenshots=[frame["image"]])
        raw = path.read_bytes()
        (observation_dir / f"{path.stem}.json").write_text(
            json.dumps(
                {
                    "image_sha256": hashlib.sha256(raw).hexdigest(),
                    "stamp": hashlib.sha256(raw + b"fixture-revision").hexdigest(),
                    "observation": row.model_dump(),
                }
            )
        )
        rows.append(row)
    result = Extraction(
        source="fixture images",
        catalog_sha256="fixture",
        catalog_url="fixture",
        students=rows,
        diagnostics={"revision": "fixture-revision"},
    )
    (extraction / "results.json").write_text(result.model_dump_json())
    output = tmp_path / "inspector"
    report = build_inspector(extraction, output)
    assert report["selected_frames"] == 2
    assert report["image_order_and_evidence_hashes_verified"] is True
    assert report["selection_exact_replay"] is False
    assert (output / "preview.png").is_file()
    data = json.loads((output / "data.json").read_text())
    assert data["plan"]["timeline_unit"] == "image_index"
    assert data["motion_boxes"] == []
