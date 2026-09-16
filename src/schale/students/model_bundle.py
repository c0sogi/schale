"""Model installation needs only the standard library, not the inference runtime."""

import hashlib
import json
import math
import re
import shutil
import stat
import tempfile
from pathlib import Path
from zipfile import BadZipFile, ZipFile

PROFILE = "bluearchive-ko-16x9-v1"
BUNDLE_FILES = ("model.onnx", "metadata.json", "LICENSE-OpenOCR", "NOTICE.md")


def validate_bundle(directory: Path) -> dict:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != 1
        or metadata.get("profile") != PROFILE
    ):
        raise ValueError("Unsupported student numeric model profile")
    expected = metadata.get("model_sha256")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("Invalid model checksum in metadata")
    with (directory / "model.onnx").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise ValueError("Student numeric model checksum mismatch")
    charset = metadata.get("charset")
    if (
        not isinstance(charset, list)
        or not charset
        or charset[0] != ""
        or not all(isinstance(c, str) for c in charset)
    ):
        raise ValueError("Invalid CTC character dictionary")
    threshold = metadata.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid numeric acceptance threshold")
    return metadata


def install_bundle(source: Path, destination: Path) -> None:
    """Validate in staging before publishing; never overwrite an installed bundle."""
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ValueError("Model destination is not empty; choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".student-model-", dir=destination.parent
    ) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        if source.is_dir():
            for name in BUNDLE_FILES:
                path = source / name
                if path.is_file():
                    shutil.copyfile(path, stage / name)
        else:
            try:
                with ZipFile(source) as archive:
                    members = archive.infolist()
                    if (
                        len({m.filename for m in members}) != len(members)
                        or any(
                            m.filename not in BUNDLE_FILES
                            or stat.S_ISLNK(m.external_attr >> 16)
                            for m in members
                        )
                        or sum(m.file_size for m in members) > 256 * 1024 * 1024
                    ):
                        raise ValueError(
                            "Model ZIP must contain only flat bundle files (maximum 256 MiB)"
                        )
                    for member in members:
                        with (
                            archive.open(member) as incoming,
                            (stage / member.filename).open("wb") as outgoing,
                        ):
                            shutil.copyfileobj(incoming, outgoing)
            except BadZipFile as error:
                raise ValueError(
                    "Model source must be a bundle directory or ZIP"
                ) from error
        validate_bundle(stage)
        if destination.exists():
            # Only an empty destination is accepted; rmdir cannot remove contents.
            destination.rmdir()
        stage.rename(destination)
