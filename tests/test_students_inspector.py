"""Supervisor decisions must expose, never change, recognition outputs."""

import hashlib
import json

import cv2
import numpy as np
import pytest

from schale.students.inspector import (
    build_inspector,
    contained,
    model_views,
    polygon,
    replay_video,
)
from schale.students.layout import Layout
from schale.students.numeric import localize, prepare
from schale.students.video import crop, selection_plan
from schale.students.labels import BOXES


def legacy_selection(signatures, fps=30):
    # Frozen pre-refactor algorithm, to detect accidental segmentation changes.
    motion = np.r_[0, np.mean(np.abs(np.diff(signatures, axis=0)), axis=1)]
    median = float(np.median(motion))
    mad = float(np.median(np.abs(motion - median)))
    threshold = max(1.5, min(5.0, median + 8 * max(mad, 0.05)))
    groups = []
    for i in np.flatnonzero(motion > threshold):
        if groups and i == groups[-1][-1] + 1:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    cuts = sorted(set([0] + [g[-1] for g in groups] + [len(signatures)]))
    kept, discarded = [], []
    for left, right in zip(cuts[:-1], cuts[1:]):
        valid = [
            i
            for i in range(left + 1, right - 1)
            if motion[i] < threshold * 0.35 and motion[i + 1] < threshold * 0.35
        ]
        if valid:
            kept.append(
                (
                    left / fps,
                    right / fps,
                    sorted(
                        set(valid[round(q * (len(valid) - 1))] for q in (0.2, 0.5, 0.8))
                    ),
                )
            )
        else:
            discarded.append((left / fps, right / fps))
    return kept, discarded, threshold


@pytest.mark.parametrize("seed", range(8))
def test_selection_matches_frozen_algorithm(seed):
    rng = np.random.default_rng(seed)
    signatures = np.repeat(
        rng.uniform(0, 255, (30, 10)), rng.integers(1, 30, 30), axis=0
    )
    signatures += rng.normal(0, 0.1, signatures.shape)
    expected, discarded, threshold = legacy_selection(signatures)
    got, rejected, actual_threshold, _ = selection_plan(signatures)
    assert [
        (s["start"], s["end"], [f["sample_index"] for f in s["frames"]]) for s in got
    ] == expected
    assert [(s["start"], s["end"]) for s in rejected] == discarded
    assert actual_threshold == threshold


def test_short_recording_rejected():
    with pytest.raises(ValueError, match="too short"):
        selection_plan(np.zeros((1, 20)))


def test_evidence_cannot_escape_root(tmp_path):
    assert contained(tmp_path, "frames/a.png") == tmp_path / "frames/a.png"
    with pytest.raises(ValueError, match="escapes"):
        contained(tmp_path, "../outside.png")


def test_exact_polygon_retains_rotation():
    matrix = np.array([[1, -0.01, 5], [0.01, 1, -3]])
    layout = Layout({"left": matrix, "right": matrix}, {})
    box = (0.6, 0.2, 0.7, 0.3)
    actual = np.asarray(polygon(box, layout))
    assert actual[0, 1] != actual[1, 1]
    assert layout.source_roi(box) == pytest.approx([*actual.min(0), *actual.max(0)])


def test_model_view_is_actual_preprocessing_pixels(tmp_path):
    image = np.random.default_rng(5).integers(0, 256, (1440, 2560, 3), dtype=np.uint8)
    fields = {
        "level": {"value": 1, "evidence": [{"method": "svtrv2-s-rctc", "raw": "Lv.1"}]}
    }
    before = json.dumps(fields)
    target = tmp_path / "patches.png"
    boxes = model_views(image, fields, target)
    sprite = cv2.imdecode(
        np.frombuffer(target.read_bytes(), np.uint8), cv2.IMREAD_COLOR
    )
    assert sprite is not None
    x, y, w, h = boxes["level"]["model_input"]
    tensor = prepare(localize(crop(image, BOXES["level"]), "level"))
    expected = np.rint((tensor.transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)[
        :, :, ::-1
    ]
    assert np.array_equal(sprite[y : y + h, x : x + w], expected)
    assert json.dumps(fields) == before


def test_state_readings_are_not_fabricated_as_ctc(tmp_path):
    assert (
        model_views(
            np.zeros((1440, 2560, 3), np.uint8),
            {"level": {"evidence": [{"method": "visual-state", "raw": "locked"}]}},
            tmp_path / "none.png",
        )
        == {}
    )
    assert not (tmp_path / "none.png").exists()


def test_replay_rejects_changed_video_before_decoding(tmp_path):
    video = tmp_path / "fake.mp4"
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        replay_video({"source_fingerprint": "wrong"}, video, tmp_path, print)


def test_existing_report_not_overwritten(tmp_path):
    (tmp_path / "keep.txt").write_text("user data")
    with pytest.raises(ValueError, match="not empty"):
        build_inspector(tmp_path, tmp_path)
    assert (tmp_path / "keep.txt").read_text() == "user data"


def test_replay_requires_exact_saved_decisions(tmp_path, monkeypatch):
    import schale.students.inspector as inspector

    video = tmp_path / "fake.mp4"
    video.write_bytes(b"mock decoded video")
    (tmp_path / "timeline").mkdir()
    monkeypatch.setattr(
        inspector,
        "frames",
        lambda *args: iter([np.zeros((360, 640, 3), np.uint8) for _ in range(8)]),
    )
    plan = {
        "source_fingerprint": hashlib.sha256(video.read_bytes()).hexdigest(),
        "fps": 30,
        "decoder": [],
        "segments": [],
        "discarded": [],
        "threshold": 1.5,
    }
    with pytest.raises(ValueError, match="differs"):
        replay_video(plan, video, tmp_path, lambda _: None)
