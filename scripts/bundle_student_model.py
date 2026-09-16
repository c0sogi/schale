"""Build a flat, validated model ZIP separate from the Python wheel."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from schale.students.model_bundle import validate_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    validate_bundle(args.source)
    names = ("model.onnx", "metadata.json", "LICENSE-OpenOCR", "NOTICE.md")
    for name in names:
        if not (args.source / name).is_file():
            raise ValueError(f"Required distribution file missing: {name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "x", compression=ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(args.source / name, name)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
