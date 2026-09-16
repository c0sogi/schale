"""Composable collection operations with conservative, explicit update policies."""

from .account import AccountSnapshot, GrowthValue


def compare(before: AccountSnapshot, after: AccountSnapshot) -> dict:
    old = {s.student_id: s for s in before.students if s.student_id is not None}
    new = {s.student_id: s for s in after.students if s.student_id is not None}
    changes, identities = [], []
    for sid in sorted(old.keys() & new.keys()):
        if old[sid].identity_status != new[sid].identity_status:
            identities.append(
                {
                    "student_id": sid,
                    "before": old[sid].identity_status,
                    "after": new[sid].identity_status,
                }
            )
        for field in sorted(old[sid].fields.keys() | new[sid].fields.keys()):
            a, b = (
                old[sid].fields.get(field, GrowthValue()),
                new[sid].fields.get(field, GrowthValue()),
            )
            if (a.value, a.status) != (b.value, b.status):
                changes.append(
                    {
                        "student_id": sid,
                        "field": field,
                        "before": {"value": a.value, "status": a.status},
                        "after": {"value": b.value, "status": b.status},
                    }
                )
    return {
        "schema_version": 1,
        "kind": "schale.account-diff",
        "added_ids": sorted(new.keys() - old.keys()),
        "missing_ids": sorted(old.keys() - new.keys()),
        "changes": changes,
        "identity_changes": identities,
        "unresolved_before": [s.key for s in before.students if s.student_id is None],
        "unresolved_after": [s.key for s in after.students if s.student_id is None],
        "note": "Missing IDs mean not present in this snapshot, not lost ownership.",
    }


def merge(
    base: AccountSnapshot, incoming: AccountSnapshot, *, allow_observed: bool = False
) -> tuple[AccountSnapshot, dict]:
    """Apply accepted incoming values, preserve old values when incoming is uncertain."""
    result = base.model_copy(deep=True)
    accepted = {"confirmed", "corrected", "imported"}
    if allow_observed:
        accepted.update({"observed", "inferred"})
    index = {s.student_id: s for s in result.students if s.student_id is not None}
    applied, held = [], []
    for student in incoming.students:
        sid = student.student_id
        if sid is None or student.identity_status not in accepted:
            held.append({"key": student.key, "reason": "identity not accepted"})
            continue
        target = index.get(sid)
        if target is None:
            target = student.model_copy(deep=True)
            target.key = f"student_{sid}"
            if target.key in {s.key for s in result.students}:
                raise ValueError(f"Student key collision: {target.key}")
            target.fields = {}
            target.sources = []
            result.students.append(target)
            index[sid] = target
        if not (
            target.identity_status in {"confirmed", "corrected"}
            and student.identity_status in {"observed", "inferred"}
        ):
            target.identity_status = student.identity_status
        if student.name:
            target.name = student.name
        for name, reading in student.fields.items():
            if reading.value is None or reading.status not in accepted:
                held.append(
                    {"student_id": sid, "field": name, "reason": reading.status}
                )
                continue
            old = target.fields.get(name)
            new = reading.model_copy(deep=True)
            if old is not None and old.value == new.value:
                new.sources = [*old.sources, *new.sources]
                if old.status in {"confirmed", "corrected"} and new.status in {
                    "observed",
                    "inferred",
                }:
                    new.status = old.status
            target.fields[name] = new
            applied.append(
                {
                    "student_id": sid,
                    "field": name,
                    "before": old.value if old else None,
                    "after": new.value,
                }
            )
        target.sources.extend(student.sources)
        # Existing website preferences belong to the base; imported preferences
        # fill absent keys only. They are not inferred from game pixels.
        for namespace, attributes in student.extensions.items():
            if namespace not in target.extensions:
                target.extensions[namespace] = attributes
            elif isinstance(attributes, dict) and isinstance(
                target.extensions[namespace], dict
            ):
                for key, value in attributes.items():
                    target.extensions[namespace].setdefault(key, value)
    result.sources.extend(incoming.sources)
    result = AccountSnapshot.model_validate(result.model_dump())
    return result, {
        "applied": applied,
        "held": held,
        "students": len(result.students),
        "policy": "Incoming accepted observations update values; no deletion and no unknown-to-zero conversion.",
    }
