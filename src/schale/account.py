"""Source-independent player state. Game definitions live in reference catalogs."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

GROWTH_BOUNDS = {
    "star": (1, 5),
    "level": (1, 100),
    "bond": (1, 100),
    "ex": (1, 5),
    "basic": (0, 10),
    "passive": (0, 10),
    "sub": (0, 10),
    "weapon_star": (0, 4),
    "weapon_level": (0, 100),
    "gear": (0, 2),
    **{f"equipment{i}": (0, 20) for i in range(1, 4)},
    **{f"equipment{i}_level": (0, 100) for i in range(1, 4)},
    **{f"potential_{s}": (0, 25) for s in ("hp", "attack", "heal")},
}
Status = Literal[
    "unknown", "observed", "inferred", "confirmed", "corrected", "conflict", "imported"
]


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    uri: str
    details: dict[str, Any] = Field(default_factory=dict)


class GrowthValue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: StrictInt | None = None
    status: Status = "unknown"
    sources: list[SourceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent_status(self):
        if self.status in {"unknown", "conflict"} and self.value is not None:
            raise ValueError("Unknown/conflicting values must remain null")
        if self.value is None and self.status not in {"unknown", "conflict"}:
            raise ValueError("A known status requires a value")
        return self


class StudentState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    student_id: StrictInt | None = Field(default=None, gt=0)
    name: str = ""
    identity_status: Status = "unknown"
    fields: dict[str, GrowthValue] = Field(default_factory=dict)
    sources: list[SourceReference] = Field(default_factory=list)
    # Adapter-owned attributes such as a website's lock flag, not game state.
    extensions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_fields(self):
        for name, reading in self.fields.items():
            if name not in GROWTH_BOUNDS:
                raise ValueError(f"Unsupported growth field: {name}")
            lo, hi = GROWTH_BOUNDS[name]
            if reading.value is not None and not lo <= reading.value <= hi:
                raise ValueError(f"Out-of-range growth field: {name}={reading.value}")
        if self.student_id is None and self.identity_status not in {
            "unknown",
            "conflict",
        }:
            raise ValueError("An identified student needs a SchaleDB ID")
        return self


class AccountSnapshot(BaseModel):
    """Portable schema v1; missing observations do not imply missing ownership."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    kind: Literal["schale.account"] = "schale.account"
    sources: list[SourceReference] = Field(default_factory=list)
    students: list[StudentState] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_students(self):
        ids = [s.student_id for s in self.students if s.student_id is not None]
        keys = [s.key for s in self.students]
        if len(ids) != len(set(ids)) or len(keys) != len(set(keys)):
            raise ValueError(
                "Duplicate student IDs or keys; resolve identity before merging"
            )
        return self
