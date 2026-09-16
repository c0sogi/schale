"""Visual extraction contracts, independent of OCR and network availability."""

import base64
import json
import subprocess
import sys
from typing import Literal

import cv2
import numpy as np
import pytest

from schale.students.export import (
    KEYS,
    corrections,
    decode_collection,
    export_collection,
)
from schale.students.extract import consensus, source_fingerprint
from schale.students.identity import symbol
from schale.students.labels import LabelBank
from schale.students.models import Evidence, Extraction, Reading, Student
from schale.students.video import frames, prepare_video


def observation(
    value: int | None, image: str, status: Literal["observed", "unknown"] = "observed"
) -> Student:
    return Student(
        key="a",
        student_id=10001,
        name="에이미",
        identity_status="observed",
        screenshots=[image],
        fields={
            "level": Reading(
                value=value,
                status=status,
                evidence=[
                    Evidence(
                        image=image,
                        timestamp=0.0,
                        roi=(0, 0, 1, 1),
                        raw="test",
                        score=0.95,
                    )
                ],
            )
        },
    )


def complete() -> Extraction:
    values = dict(
        star=5,
        level=90,
        bond=21,
        ex=5,
        basic=10,
        passive=10,
        sub=10,
        equipment1=10,
        equipment2=10,
        equipment3=10,
        gear=2,
        weapon_star=2,
        weapon_level=40,
        potential_hp=25,
        potential_attack=24,
        potential_heal=23,
    )
    row = Student(
        key="a",
        student_id=10001,
        name="에이미",
        identity_status="confirmed",
        fields={k: Reading(value=v, status="confirmed") for k, v in values.items()},
    )
    return Extraction(
        source="test", catalog_sha256="test", catalog_url="test", students=[row]
    )


def test_two_frames_confirm_but_one_frame_does_not():
    assert (
        consensus([observation(90, "a.png")], "a").fields["level"].status == "observed"
    )
    assert (
        consensus([observation(90, "a.png"), observation(90, "b.png")], "a")
        .fields["level"]
        .status
        == "confirmed"
    )
    assert (
        consensus([observation(90, "a.png"), observation(90, "a.png")], "a")
        .fields["level"]
        .status
        == "observed"
    )


def test_disagreement_is_not_majority_voted_away():
    result = consensus(
        [observation(90, "a"), observation(90, "b"), observation(80, "c")], "a"
    )
    assert result.fields["level"].value is None
    assert result.fields["level"].status == "conflict"


def test_repeated_same_image_does_not_confirm_identity():
    assert (
        consensus(
            [observation(90, "same"), observation(90, "same")], "a"
        ).identity_status
        == "observed"
    )


def test_unidentified_frame_does_not_contaminate_identified_values():
    identified = observation(90, "a")
    unidentified = observation(80, "b")
    unidentified.student_id = None
    result = consensus([identified, unidentified], "a")
    assert result.fields["level"].value == 90
    assert result.fields["level"].status == "observed"


def test_review_queue_includes_single_observations_and_unknowns():
    from schale.students.audit import review_queue

    result = complete()
    result.students[0].fields["gear"] = Reading()
    result.students[0].fields["level"].status = "observed"
    queue = review_queue(result)
    assert queue["students_needing_review"] == 1
    assert set(queue["items"][0]["fields"]) == {"gear", "level"}


def test_fingerprint_detects_replaced_video(tmp_path):
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"old video")
    original = source_fingerprint(source)
    source.write_bytes(b"new video")
    assert source_fingerprint(source) != original


def test_unknown_is_not_zero_and_identity_conflict_is_kept():
    a, b = observation(None, "a", "unknown"), observation(None, "b", "unknown")
    b.student_id = 10002
    merged = consensus([a, b], "a")
    assert merged.student_id is None and merged.identity_status == "conflict"
    assert merged.fields["level"].value is None


def test_site_export_field_mapping_and_base64_roundtrip():
    text, report = export_collection(complete())
    row = decode_collection(text)["10001"]
    assert row["pm"] == 25 and row["pa"] == 24 and row["ph"] == 23
    assert row["e1"] == 10 and row["ws"] == 2
    assert row["lock"] is False
    assert report["updated"] == [10001]
    assert set(KEYS.values()) <= row.keys()


def test_unknown_new_student_cannot_silently_export_defaults():
    result = complete()
    result.students[0].fields["gear"] = Reading()
    with pytest.raises(ValueError, match="No complete"):
        export_collection(result)


def test_baseline_preserves_unseen_students_unknown_fields_and_lock():
    result = complete()
    result.students[0].fields["gear"] = Reading()
    baseline = {"10001": {"e4": 1, "lock": True}, "99999": {"sentinel": "preserved"}}
    text, _ = export_collection(result, baseline)
    decoded = decode_collection(text)
    assert decoded["10001"]["e4"] == 1 and decoded["10001"]["lock"] is True
    assert decoded["99999"] == baseline["99999"]
    assert baseline["10001"] == {"e4": 1, "lock": True}


def test_export_observations_require_explicit_opt_in():
    result = complete()
    result.students[0].identity_status = "observed"
    with pytest.raises(ValueError):
        export_collection(result)
    export_collection(result, allow_observed=True)


def test_locked_skill_uses_site_minimum_but_preserves_source():
    result = complete()
    result.students[0].fields["sub"].value = 0
    text, _ = export_collection(result)
    assert decode_collection(text)["10001"]["s4"] == 1
    assert result.students[0].fields["sub"].value == 0


@pytest.mark.parametrize(
    "text",
    [
        "bad",
        "e30=\ntrash",
        base64.b64encode(b"[]").decode(),
        base64.b64encode(b'{"1":{"l":true}}').decode(),
    ],
)
def test_invalid_export_is_rejected(text):
    with pytest.raises(ValueError):
        decode_collection(text)


def test_corrections_are_validated_and_do_not_mutate_input(tmp_path):
    path = tmp_path / "fix.json"
    path.write_text(
        json.dumps({"schema_version": 1, "students": {"a": {"fields": {"level": 91}}}})
    )
    original = complete()
    fixed = corrections(original, path, {})
    assert fixed.students[0].fields["level"].value == 91
    assert original.students[0].fields["level"].value == 90
    path.write_text(
        json.dumps({"schema_version": 1, "students": {"a": {"fields": {"gear": 99}}}})
    )
    with pytest.raises(ValueError):
        corrections(original, path, {})


def test_blank_screen_has_no_icon_or_label(student_references):
    image = np.full((1440, 2560, 3), 255, np.uint8)
    assert symbol(image, (0.544, 0.475, 0.604, 0.557)) is None
    assert LabelBank(student_references).read(image, "level")[0] is None


def test_no_ocr_dependency_is_imported():
    code = "import sys; import schale.students.vision; assert 'easyocr' not in sys.modules; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_raw_decoder_rejects_truncated_frames():
    command = [sys.executable, "-c", "import sys;sys.stdout.buffer.write(b'abc')"]
    with pytest.raises(RuntimeError, match="Truncated"):
        list(frames(command, 2, 2))


def test_video_selection_keeps_fast_stable_visits(tmp_path):
    import shutil

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is an external optional video dependency")
    video = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter.fourcc(*"MJPG"), 30, (640, 360)
    )
    if not writer.isOpened():
        pytest.skip("MJPG encoder unavailable")
    # Four 0.3-second screens; uniform colors give exact independent expectations.
    for value in (30, 90, 150, 220):
        for _ in range(9):
            writer.write(np.full((360, 640, 3), value, np.uint8))
    writer.release()
    output = tmp_path / "out"
    plan = prepare_video(video, output)
    assert len(plan["segments"]) == 4
    for segment, expected in zip(plan["segments"], (30, 90, 150, 220), strict=True):
        assert len(segment["frames"]) >= 2
        for frame in segment["frames"]:
            image = cv2.imread(str(output / frame["image"]))
            assert image is not None
            assert abs(float(image.mean()) - expected) < 3
