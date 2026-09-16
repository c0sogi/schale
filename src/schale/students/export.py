"""SchaleDB collection serialization; incomplete new records never get defaults."""

import json
from pathlib import Path

from .models import Extraction, Reading
from ..account import GROWTH_BOUNDS as BOUNDS
from ..adapters.extraction import from_extraction
from ..adapters.schaledb import (
    KEYS as KEYS,
    decode_collection as decode_collection,
    to_schaledb,
)


def corrections(result: Extraction, path: Path, catalog: dict) -> Extraction:
    result = result.model_copy(deep=True)
    content = json.loads(path.read_text(encoding="utf-8"))
    if content.get("schema_version") != 1:
        raise ValueError("Unsupported corrections format")
    index = {student.key: student for student in result.students}
    for key, change in content["students"].items():
        if key not in index:
            raise ValueError(f"Unknown student key: {key}")
        student = index[key]
        if "student_id" in change:
            value = change["student_id"]
            if value is not None and (
                type(value) is not int or str(value) not in catalog
            ):
                raise ValueError(f"Unknown SchaleDB ID: {value}")
            student.student_id = value
            student.identity_status = "corrected" if value is not None else "unknown"
            student.name = (
                catalog[str(value)]["Name"] if value is not None else "미확인 학생"
            )
        for field, value in change.get("fields", {}).items():
            if (
                field not in BOUNDS
                or value is not None
                and (
                    type(value) is not int
                    or not BOUNDS[field][0] <= value <= BOUNDS[field][1]
                )
            ):
                raise ValueError(f"Invalid correction: {field}={value}")
            old = student.fields.get(field, Reading())
            student.fields[field] = Reading(
                value=value,
                status="corrected" if value is not None else "unknown",
                evidence=old.evidence,
            )
    ids = [s.student_id for s in result.students if s.student_id is not None]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Corrections produce duplicate student IDs; merge/review the conflicting records first"
        )
    return result


def export_collection(
    result: Extraction, baseline: dict | None = None, *, allow_observed: bool = False
) -> tuple[str, dict]:
    return to_schaledb(from_extraction(result), baseline, allow_observed=allow_observed)
