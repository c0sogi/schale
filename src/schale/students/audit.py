"""Coverage diagnostics and an explicit queue of records needing review."""

from collections import Counter

from .models import Extraction


def review_queue(result: Extraction) -> dict:
    items = []
    for student in result.students:
        fields = {
            name: {"status": reading.status, "value": reading.value}
            for name, reading in student.fields.items()
            if reading.status not in {"confirmed", "corrected"}
        }
        if fields or student.identity_status not in {"confirmed", "corrected"}:
            items.append(
                {
                    "key": student.key,
                    "student_id": student.student_id,
                    "name": student.name,
                    "identity_status": student.identity_status,
                    "fields": fields,
                    "screenshots": student.screenshots,
                }
            )
    return {
        "schema_version": 1,
        "meaning": "Needs review before strict export. This queue is not a ground-truth accuracy assessment.",
        "students_total": len(result.students),
        "students_needing_review": len(items),
        "field_statuses": dict(
            Counter(r.status for s in result.students for r in s.fields.values())
        ),
        "discarded_intervals": result.diagnostics.get("discarded_intervals", []),
        "rejected_frames": result.diagnostics.get("rejected_frames", []),
        "items": items,
    }
