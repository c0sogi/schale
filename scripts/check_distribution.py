"""Check built artifacts for runtime resources and accidental private data."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from zipfile import ZipFile


def check(path: Path) -> dict:
    if path.suffix == ".whl":
        with ZipFile(path) as archive:
            names = archive.namelist()
            assert archive.testzip() is None
        required = (
            "schale/account.py",
            "schale/toolkit_cli.py",
            "schale/adapters/schaledb.py",
            "schale/assets.py",
            "schale/students/inspector.html",
            "schale/py.typed",
            "schale/vision/text.py",
            "schale/scanner/_identity.py",
        )
        assert all(name in names for name in required), (
            "Missing public API or runtime resource"
        )
        assert any(name.endswith("/licenses/LICENSE") for name in names)
        assert any(name.endswith("/licenses/THIRD_PARTY_NOTICES.md") for name in names)
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            names = archive.getnames()
        required = (
            "/README.md",
            "/CONTRIBUTING.md",
            "/tests/test_toolkit.py",
            "/docs/architecture.md",
            "/docs/students.md",
            "/scripts/check_distribution.py",
        )
        assert all(any(name.endswith(suffix) for name in names) for suffix in required)
    else:
        raise ValueError(f"Expected a wheel or sdist: {path}")
    for name in names:
        parts = set(name.split("/"))
        assert not parts & {
            ".schale",
            ".venv",
            "__pycache__",
            "outputs",
            "work",
            "observations",
            "frames",
            "screenshots",
        }, name
        assert not name.endswith(
            (
                ".mp4",
                ".mkv",
                ".pyc",
                "/results.json",
                "/ground_truth.json",
                "/model.onnx",
                ".onnx",
                ".onnx.data",
                ".png",
                ".webp",
                ".npz",
            )
        ), name
        assert not any(
            part in name
            for part in (
                "schale/students/data/",
                "schale/scanner/data/",
                "schale/scanner/models/",
            )
        ), name
    return {
        "path": str(path.resolve()),
        "entries": len(names),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([check(path) for path in args.artifacts], indent=2))


if __name__ == "__main__":
    main()
