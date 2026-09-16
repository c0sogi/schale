"""Evidence-preserving extraction and temporal consensus."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable

from .catalog import URL, load_catalog, load_skill_assets
from .identity import IdentityMatcher
from .models import Extraction, Reading, Student
from .paths import asset_directory, model_directory
from .video import prepare_video
from .vision import Vision


from ..inputs import InputSource, prepare_images, resolve_input
from ..assets import resolve_resources


def source_fingerprint(source: InputSource) -> str:
    return resolve_input(source).fingerprint


def consensus(rows: list[Student], key: str) -> Student:
    result = Student(key=key)
    ids = {row.student_id for row in rows if row.student_id is not None}
    result.screenshots = list(dict.fromkeys(p for row in rows for p in row.screenshots))
    result.visits = [visit for row in rows for visit in row.visits]
    result.name_evidence = [e for row in rows for e in row.name_evidence]
    result.layouts = {
        image: info for row in rows for image, info in row.layouts.items()
    }
    result.candidates = next((row.candidates for row in rows if row.candidates), [])
    if len(ids) == 1:
        result.student_id = ids.pop()
        result.name = next(
            row.name for row in rows if row.student_id == result.student_id
        )
        count = len(
            {
                image
                for row in rows
                if row.student_id == result.student_id
                for image in row.screenshots
            }
        )
        result.identity_status = "confirmed" if count >= 2 else "observed"
    elif len(ids) > 1:
        result.name = "학생 식별 충돌"
        result.identity_status = "conflict"
    else:
        result.name = "미확인 학생"
    for field in sorted({k for row in rows for k in row.fields}):
        # An unidentified frame cannot contribute values to an identified student.
        # Keep all evidence when identity itself is unresolved or conflicting.
        readings = [
            row.fields[field]
            for row in rows
            if field in row.fields
            and (result.student_id is None or row.student_id == result.student_id)
        ]
        evidence = [e for reading in readings for e in reading.evidence]
        values = {reading.value for reading in readings if reading.value is not None}
        if len(values) > 1 or any(r.status == "conflict" for r in readings):
            result.fields[field] = Reading(status="conflict", evidence=evidence)
        elif values:
            value = values.pop()
            supporters = [r for r in readings if r.value == value]
            images = {e.image for r in supporters for e in r.evidence}
            status = (
                "confirmed"
                if len(images) >= 2
                else "inferred"
                if all(r.status == "inferred" for r in supporters)
                else "observed"
            )
            result.fields[field] = Reading(
                value=value, status=status, evidence=evidence
            )
        else:
            result.fields[field] = Reading(evidence=evidence)
    return result


def extract(
    source: InputSource,
    output: Path,
    *,
    asset_cache: Path | None = None,
    resume: bool = False,
    numeric_model: Path | None = None,
    reader: str = "ctc",
    progress: Callable[[str], None] = print,
) -> Extraction:
    resolved = resolve_input(source)
    source_hash = resolved.fingerprint
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError(
            "Output directory is not empty; choose a new directory or use --resume"
        )
    cache = asset_directory(asset_cache)
    if reader not in {"auto", "ctc", "template"}:
        raise ValueError("Reader must be auto, ctc, or template")
    from .numeric import NumericReader

    model_path = model_directory(numeric_model)
    numeric = None
    if reader != "template" and (numeric_model is not None or model_path.exists()):
        numeric = NumericReader(model_path)
    elif reader == "ctc":
        raise ValueError(
            f"CTC model not installed at {model_path}; run 'schale students install-model MODEL.zip' or pass --numeric-model"
        )
    references = resolve_resources("students")
    progress(
        f"Numeric reader: {numeric.method if numeric is not None else 'NCC templates (no CTC model selected)'}"
    )
    output.mkdir(parents=True, exist_ok=True)
    plan_file = output / "segments.json"
    if resume and plan_file.exists():
        previous_plan = json.loads(plan_file.read_text(encoding="utf-8"))
        if previous_plan.get("source_fingerprint") != source_hash:
            raise ValueError(
                "Input changed or the saved plan predates fingerprinting; choose a new output directory"
            )
    if resume and plan_file.exists():
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    elif resolved.kind == "images":
        plan = prepare_images(resolved, output)
    else:
        progress("Finding stable intervals and decoding selected video frames...")
        assert resolved.video is not None
        plan = prepare_video(resolved.video, output)
        plan["input_kind"] = "video"
    plan["source_fingerprint"] = source_hash
    plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    catalog, digest = load_catalog(output / "students.json")
    progress("Loading game-symbol references...")
    assets = load_skill_assets(catalog, cache / "skills")
    vision = Vision(
        IdentityMatcher(catalog, assets, cache / "portraits"),
        data=references,
        numeric=numeric,
    )
    observations = output / "observations"
    observations.mkdir(exist_ok=True)
    # Cache validity follows code, templates, catalog, and image bytes, not filenames.
    module = Path(__file__).parent
    code = b"".join(
        (module / name).read_bytes()
        for name in (
            "catalog.py",
            "identity.py",
            "labels.py",
            "models.py",
            "vision.py",
            "numeric.py",
            "model_bundle.py",
            "layout.py",
        )
    )
    templates = b"".join(
        p.read_bytes() for p in sorted(references.iterdir()) if p.is_file()
    )
    revision = hashlib.sha256(
        code
        + templates
        + digest.encode()
        + (numeric.fingerprint.encode() if numeric is not None else b"template")
    ).hexdigest()
    identity_revision = hashlib.sha256(
        (module / "identity.py").read_bytes()
        + (module / "layout.py").read_bytes()
        + (references / "layout_reference.npz").read_bytes()
        + digest.encode()
    ).hexdigest()
    visits = []
    rejected = []
    frame_quality: dict[str, int] = {}
    for index, segment in enumerate(plan["segments"], 1):
        rows = []
        for frame in segment["frames"]:
            path = output / frame["image"]
            stamp = hashlib.sha256(path.read_bytes() + revision.encode()).hexdigest()
            saved = observations / f"{path.stem}.json"
            cached = (
                json.loads(saved.read_text(encoding="utf-8"))
                if resume and saved.exists()
                else None
            )
            if cached and cached.get("stamp") == stamp:
                row = Student.model_validate(cached["observation"])
            else:
                hint = None
                if (
                    cached
                    and cached.get("identity_revision") == identity_revision
                    and cached.get("image_sha256")
                    == hashlib.sha256(path.read_bytes()).hexdigest()
                ):
                    previous = Student.model_validate(cached["observation"])
                    if previous.student_id is not None and previous.name_evidence:
                        hint = (
                            previous.student_id,
                            previous.candidates,
                            previous.name_evidence[0].method,
                        )
                row = vision.analyze(
                    path,
                    frame["image"],
                    frame["timestamp"],
                    segment["key"],
                    identity_hint=hint,
                )
                saved.write_text(
                    json.dumps(
                        {
                            "stamp": stamp,
                            "identity_revision": identity_revision,
                            "image_sha256": hashlib.sha256(
                                path.read_bytes()
                            ).hexdigest(),
                            "observation": row.model_dump(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if row.identity_status in {"not_student_screen", "layout_unresolved"}:
                rejected.append({**frame, "reason": row.identity_status})
            else:
                rows.append(row)
                frame_quality[frame["image"]] = sum(
                    r.value is not None for r in row.fields.values()
                )
        if rows:
            visit = consensus(rows, segment["key"])
            visit.visits = [(segment["start"], segment["end"])]
            visits.append(visit)
            progress(
                f"{index}/{len(plan['segments'])}: {visit.name} [{visit.identity_status}]"
            )
    groups: dict[str, list[Student]] = {}
    for visit in visits:
        key = (
            f"student_{visit.student_id}" if visit.student_id is not None else visit.key
        )
        groups.setdefault(key, []).append(visit)
    students = [
        consensus(group, key)
        if len(group) > 1
        else group[0].model_copy(update={"key": key})
        for key, group in groups.items()
    ]
    counts = Counter(r.status for student in students for r in student.fields.values())
    if not students:
        raise ValueError(
            "No student-information screens were accepted; inspect segments.json and the selected frames"
        )
    (output / "screenshots").mkdir(exist_ok=True)
    representatives = {}
    for student in students:
        student.screenshots.sort(
            key=lambda name: frame_quality.get(name, 0), reverse=True
        )
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", student.name).rstrip(". ")
        representative = (
            f"screenshots/{student.student_id or student.key}_{safe_name}.png"
        )
        # Re-encode screenshot inputs too, so the .png extension remains truthful.
        from .identity import read_image
        import cv2

        image = read_image(output / student.screenshots[0])
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise RuntimeError("Could not encode representative screenshot")
        (output / representative).write_bytes(encoded.tobytes())
        representatives[student.key] = representative
    result = Extraction(
        source=resolved.label,
        catalog_sha256=digest,
        catalog_url=URL,
        students=students,
        diagnostics={
            "engine": "registered-visual-svtrv2-v2"
            if numeric is not None
            else "visual-template-v1",
            "ocr_used": numeric is not None,
            "numeric_model": numeric.metadata if numeric is not None else None,
            "revision": revision,
            "input_kind": resolved.kind,
            "source_fingerprint": source_hash,
            "segments": len(plan["segments"]),
            "discarded_intervals": plan.get("discarded", []),
            "rejected_frames": rejected,
            "field_statuses": dict(counts),
            "identified_students": sum(s.student_id is not None for s in students),
            "visit_identities": [
                {
                    "key": v.key,
                    "student_id": v.student_id,
                    "status": v.identity_status,
                    "visits": v.visits,
                }
                for v in visits
            ],
            "confirmation_semantics": "Repeated consistent frames, not statistical independence or an accuracy guarantee; conflicts are never majority-voted away.",
            "representative_screenshots": representatives,
        },
    )
    (output / "results.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    from .review import write_review
    from .audit import review_queue

    (output / "review_queue.json").write_text(
        json.dumps(review_queue(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_review(result, output)
    return result
