"""Observations keep uncertain readings distinct from game state zeroes."""

from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    image: str
    timestamp: float
    roi: tuple[float, float, float, float]
    raw: str
    score: float = Field(ge=0, le=1)
    method: str = "visual-template"


class Reading(BaseModel):
    value: int | None = None
    status: Literal[
        "observed", "confirmed", "inferred", "unknown", "conflict", "corrected"
    ] = "unknown"
    evidence: list[Evidence] = Field(default_factory=list)


class Student(BaseModel):
    key: str
    student_id: int | None = None
    name: str = ""
    identity_status: str = "unknown"
    name_evidence: list[Evidence] = Field(default_factory=list)
    candidates: list[dict[str, str | int | float]] = Field(default_factory=list)
    fields: dict[str, Reading] = Field(default_factory=dict)
    screenshots: list[str] = Field(default_factory=list)
    visits: list[tuple[float, float]] = Field(default_factory=list)
    layouts: dict[str, dict] = Field(default_factory=dict)


class Extraction(BaseModel):
    schema_version: int = 1
    source: str
    catalog_sha256: str
    catalog_url: str
    students: list[Student] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)
