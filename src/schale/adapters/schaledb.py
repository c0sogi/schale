"""SchaleDB collection wire format isolated from the common account schema."""

import base64
import hashlib
import json

from ..account import (
    AccountSnapshot,
    GROWTH_BOUNDS,
    GrowthValue,
    SourceReference,
    StudentState,
)
from ..reference import ReferenceCatalog

KEYS = {
    "star": "s",
    "level": "l",
    "bond": "b",
    "ex": "s1",
    "basic": "s2",
    "passive": "s3",
    "sub": "s4",
    "equipment1": "e1",
    "equipment2": "e2",
    "equipment3": "e3",
    "gear": "e4",
    "weapon_star": "ws",
    "weapon_level": "wl",
    "potential_hp": "pm",
    "potential_attack": "pa",
    "potential_heal": "ph",
}


def decode_collection(text: str) -> dict:
    try:
        data = json.loads(base64.b64decode(text.strip(), validate=True))
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("Invalid SchaleDB base64 JSON export") from error
    if not isinstance(data, dict) or not all(
        str(k).isdigit() and isinstance(v, dict) for k, v in data.items()
    ):
        raise ValueError("Unsupported SchaleDB collection structure")
    for fields in data.values():
        for key, value in fields.items():
            if key in KEYS.values() and (
                type(value) is not int or value < 0 or value > 100
            ):
                raise ValueError("Invalid numeric value in baseline collection")
    return data


def from_schaledb(
    text: str,
    *,
    source: str = "schaledb-export",
    catalog: ReferenceCatalog | None = None,
) -> AccountSnapshot:
    if catalog is not None and catalog.dataset != "students":
        raise ValueError("A student reference catalog is required")
    data = decode_collection(text)
    reference = SourceReference(
        kind="schaledb-collection",
        uri=source,
        details={"sha256": hashlib.sha256(text.strip().encode()).hexdigest()},
    )
    students = []
    for sid, record in data.items():
        name = str(catalog.get(int(sid)).get("Name", "")) if catalog else ""
        students.append(
            StudentState(
                key=f"student_{int(sid)}",
                student_id=int(sid),
                name=name,
                identity_status="imported",
                fields={
                    field: GrowthValue(
                        value=record[key], status="imported", sources=[reference]
                    )
                    for field, key in KEYS.items()
                    if key in record
                },
                sources=[reference],
                extensions={
                    "schaledb": {
                        k: v for k, v in record.items() if k not in KEYS.values()
                    }
                },
            )
        )
    sources = [reference]
    if catalog:
        sources.append(
            SourceReference(
                kind="reference-catalog",
                uri=catalog.source,
                details={"sha256": catalog.sha256, "dataset": catalog.dataset},
            )
        )
    return AccountSnapshot(students=students, sources=sources)


def to_schaledb(
    snapshot: AccountSnapshot,
    baseline: dict | None = None,
    *,
    allow_observed: bool = False,
) -> tuple[str, dict]:
    collection = json.loads(json.dumps(baseline or {}))
    accepted = {"confirmed", "corrected", "imported"}
    if allow_observed:
        accepted.update(("observed", "inferred"))
    skipped, updated = [], []
    for student in snapshot.students:
        if student.student_id is None or student.identity_status not in accepted:
            skipped.append({"key": student.key, "reason": "identity not confirmed"})
            continue
        record = dict(collection.get(str(student.student_id), {}))
        extras = student.extensions.get("schaledb", {})
        if not isinstance(extras, dict):
            raise ValueError("SchaleDB extension must be an object")
        for key, value in extras.items():
            if key not in KEYS.values():
                record.setdefault(key, value)
        for field, key in KEYS.items():
            reading = student.fields.get(field)
            if (
                reading is None
                or reading.value is None
                or reading.status not in accepted
            ):
                continue
            if not GROWTH_BOUNDS[field][0] <= reading.value <= GROWTH_BOUNDS[field][1]:
                raise ValueError(f"Out-of-range extracted value: {student.key}/{field}")
            value = reading.value
            if field in ("basic", "passive", "sub") and value == 0:
                value = 1
            record[key] = value
        missing = sorted(set(KEYS.values()) - record.keys())
        if missing:
            skipped.append(
                {"key": student.key, "reason": "incomplete record", "missing": missing}
            )
            continue
        record.setdefault("lock", False)
        collection[str(student.student_id)] = record
        updated.append(student.student_id)
    if not updated:
        raise ValueError(
            "No complete confirmed records to export. Review missing fields or supply a baseline export."
        )
    raw = json.dumps(collection, separators=(",", ":"), ensure_ascii=True)
    return base64.b64encode(raw.encode()).decode(), {
        "updated": updated,
        "skipped": skipped,
        "total": len(collection),
        "baseline_used": baseline is not None,
        "warning": "SchaleDB import replaces the collection; use --base to preserve unscanned students.",
    }
