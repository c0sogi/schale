"""Default and student environments must not pull in training frameworks."""

from pathlib import Path
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]


def dependency_names(group: str | None = None, extra: str | None = None) -> set[str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = lock["package"]
    project = next(p for p in packages if p["name"] == "schale")
    pending = list(project["dependencies"])
    if group:
        pending.extend(project["dev-dependencies"][group])
    if extra:
        pending.extend(project["optional-dependencies"][extra])
    visited: set[tuple[str, tuple[str, ...]]] = set()
    # Follow all platform variants conservatively, including transitive extras.
    while pending:
        dependency = pending.pop()
        key = (dependency["name"], tuple(dependency.get("extra", [])))
        if key in visited:
            continue
        visited.add(key)
        variants = [p for p in packages if p["name"] == key[0]]
        assert variants, f"Missing locked dependency: {key[0]}"
        for package in variants:
            pending.extend(package.get("dependencies", []))
            for selected in key[1]:
                pending.extend(package["optional-dependencies"][selected])
    return {name for name, _ in visited}


@pytest.mark.parametrize(
    "group,extra",
    [
        (None, None),
        (None, "vision"),
        (None, "student-ocr"),
        (None, "scanner"),
        ("dev", None),
        ("test", None),
    ],
)
def test_lightweight_dependency_closure(group, extra):
    names = dependency_names(group, extra)
    assert not names & {
        "torch",
        "torchvision",
        "easyocr",
        "albumentations",
        "onnxscript",
        "triton",
    }
    assert not any(name.startswith(("nvidia-", "cuda-")) for name in names)


def test_only_explicit_training_needs_torch():
    assert {"torch", "torchvision", "albumentations", "onnxscript"} <= dependency_names(
        "training"
    )
    assert "easyocr" not in (ROOT / "uv.lock").read_text(encoding="utf-8")
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["uv"].get("default-groups", ["dev"]) == ["dev"]
