"""Visual extraction is one input adapter, not the common state representation."""

from ..account import AccountSnapshot, GrowthValue, SourceReference, StudentState
from ..students.models import Extraction


def from_extraction(result: Extraction) -> AccountSnapshot:
    students = []
    for student in result.students:
        students.append(
            StudentState.model_validate(
                {
                    "key": student.key,
                    "student_id": student.student_id,
                    "name": student.name,
                    "identity_status": student.identity_status,
                    "fields": {
                        name: GrowthValue.model_validate(
                            {
                                "value": value.value,
                                "status": value.status,
                                "sources": [
                                    SourceReference(
                                        kind="visual-observation",
                                        uri=e.image,
                                        details=e.model_dump(exclude={"image"}),
                                    )
                                    for e in value.evidence
                                ],
                            }
                        )
                        for name, value in student.fields.items()
                    },
                    "sources": [
                        SourceReference(
                            kind="extraction",
                            uri=result.source,
                            details={"key": student.key, "visits": student.visits},
                        )
                    ],
                }
            )
        )
    return AccountSnapshot(
        students=students,
        sources=[
            SourceReference(
                kind="reference-catalog",
                uri=result.catalog_url,
                details={"dataset": "students", "sha256": result.catalog_sha256},
            ),
            SourceReference(
                kind="extraction",
                uri=result.source,
                details={"revision": result.diagnostics.get("revision")},
            ),
        ],
    )
