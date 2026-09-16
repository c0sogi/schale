"""One private setup file, automatically reused on first student extraction."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

from filelock import FileLock

from .assets import install_resources, pack_resources, resolve_resources
from .cache import cache_directory
from .students.model_bundle import BUNDLE_FILES, install_bundle, validate_bundle
from .students.paths import model_directory
from .students.runtime import runtime_report

SETUP_NAME = "schale-setup.zip"
SETUP_FILES = {"setup.json", "references.zip", "numeric.zip"}


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_setup(output: Path) -> None:
    """Export only vision resources and model, never account files or HTTP caches."""
    resources = resolve_resources("students").parent
    model = model_directory()
    validate_bundle(model)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="schale-setup-export-") as temporary:
        stage = Path(temporary)
        pack_resources(resources, stage / "references.zip")
        with ZipFile(stage / "numeric.zip", "w", ZIP_DEFLATED) as archive:
            for name in BUNDLE_FILES:
                archive.write(model / name, name)
        manifest = {
            "kind": "schale.setup",
            "schema_version": 1,
            "files": {
                name: _digest(stage / name)
                for name in ("references.zip", "numeric.zip")
            },
        }
        with output.open("xb") as stream, ZipFile(stream, "w", ZIP_DEFLATED) as archive:
            archive.writestr("setup.json", json.dumps(manifest, indent=2))
            for name in manifest["files"]:
                archive.write(stage / name, name)


def install_setup(source: Path) -> None:
    """Validate both nested bundles before changing the user's setup."""
    root = cache_directory()
    root.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(root / "setup.lock")),
        tempfile.TemporaryDirectory(prefix=".setup-", dir=root) as temporary,
    ):
        stage = Path(temporary)
        try:
            with ZipFile(source) as archive:
                members = archive.infolist()
                if (
                    len(members) != 3
                    or {m.filename for m in members} != SETUP_FILES
                    or any(
                        m.orig_filename != m.filename
                        or stat.S_ISLNK(m.external_attr >> 16)
                        for m in members
                    )
                    or sum(m.file_size for m in members) > 800 * 1024 * 1024
                ):
                    raise ValueError(
                        "Invalid setup ZIP; expected setup.json, references.zip and numeric.zip"
                    )
                for member in members:
                    with (
                        archive.open(member) as incoming,
                        (stage / member.filename).open("xb") as outgoing,
                    ):
                        shutil.copyfileobj(incoming, outgoing)
        except BadZipFile as error:
            raise ValueError("Invalid setup ZIP") from error
        manifest = json.loads((stage / "setup.json").read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("kind") != "schale.setup"
            or manifest.get("schema_version") != 1
        ):
            raise ValueError("Unsupported setup bundle")
        expected = {
            name: _digest(stage / name) for name in ("references.zip", "numeric.zip")
        }
        if manifest.get("files") != expected:
            raise ValueError("Setup bundle checksum mismatch")
        staged_resources = install_resources(
            stage / "references.zip", root=stage / "verified-resources"
        )
        if not (staged_resources / "students").is_dir():
            raise ValueError("Setup bundle has no student references")
        staged_model = stage / "verified-model"
        install_bundle(stage / "numeric.zip", staged_model)
        destination = model_directory()
        try:
            validate_bundle(destination)
            model_valid = True
        except (OSError, ValueError):
            model_valid = False
        if not model_valid:
            if destination.exists():
                # Preserve a broken installation for recovery, rather than deleting it.
                backup_root = root / "setup-backups"
                backup_root.mkdir(exist_ok=True)
                backup = (
                    Path(tempfile.mkdtemp(prefix="numeric-", dir=backup_root))
                    / "student-numeric"
                )
                destination.rename(backup)
            staged_model.rename(destination)
        try:
            resolve_resources("students")
        except ValueError:
            install_resources(stage / "references.zip")


def find_setup(sources: list[Path] | None = None) -> Path | None:
    configured = os.environ.get("SCHALE_SETUP_BUNDLE")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not candidate.is_file():
            raise ValueError(f"SCHALE_SETUP_BUNDLE does not exist: {candidate}")
        return candidate
    folders = [Path.cwd()]
    folders.extend(path.resolve().parent for path in sources or [])
    folders.append(Path.home() / "Downloads")
    for folder in dict.fromkeys(folders):
        candidate = folder / SETUP_NAME
        if candidate.is_file():
            return candidate
    return None


def prepare_local_setup(sources: list[Path] | None = None) -> bool:
    report = runtime_report()
    if report["model"]["valid"] and report["references"]["valid"]:
        return True
    bundle = find_setup(sources)
    if bundle is None:
        return False
    print(f"Preparing local vision resources from {bundle} ...", flush=True)
    install_setup(bundle)
    return True


def ensure_student_runtime(
    sources: list[Path], *, reader: str, numeric_model: Path | None
) -> None:
    if reader not in {"ctc", "auto", "template"}:
        raise ValueError("Reader must be auto, ctc, or template")
    report = runtime_report(numeric_model)
    if not report["references"]["valid"] or (
        reader == "ctc" and not report["model"]["valid"] and numeric_model is None
    ):
        prepare_local_setup(sources)
        report = runtime_report(numeric_model)
    missing = []
    if not report["references"]["valid"]:
        missing.append("student UI references")
    if reader == "ctc" and not report["model"]["valid"]:
        missing.append(f"numeric model ({report['model']['path']})")
    image_suffixes = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".json",
    }
    video = any(
        path.is_file() and path.suffix.lower() not in image_suffixes for path in sources
    )
    tools = (
        [name for name, path in report["tools"].items() if not path] if video else []
    )
    if missing or tools:
        lines = ["First-run setup is incomplete (all missing requirements are listed):"]
        if missing:
            lines.append("  Resources: " + ", ".join(missing))
            lines.append(
                "  This package does not distribute the private game references."
            )
            lines.append(
                f"  Put your existing {SETUP_NAME} in {Path.home() / 'Downloads'}; the SAME extract command installs it automatically."
            )
            lines.append(
                "  A prepared PC can create it with: schale setup --export schale-setup.zip"
            )
            if numeric_model is not None:
                lines.append(
                    "  The explicit --numeric-model path must point to a valid model bundle."
                )
        if tools:
            lines.append("  Video tools missing from PATH: " + ", ".join(tools))
            lines.append(
                "  FFmpeg download: https://ffmpeg.org/download.html (both ffmpeg and ffprobe are required)"
            )
        raise ValueError("\n".join(lines))


def run_with_vision_if_needed(arguments: list[str]) -> int | None:
    """Use uv's isolated tool environment, never pip-install into the caller's project."""
    required = ("cv2", "numpy", "PIL", "onnxruntime")
    if all(importlib.util.find_spec(name) is not None for name in required):
        return None
    uv = shutil.which("uv")
    if uv is None or os.environ.get("SCHALE_VISION_CHILD"):
        raise ValueError(
            "Vision runtime unavailable. Install with: uv tool install --reinstall 'schale[student-ocr]'"
        )
    from importlib.metadata import version

    print(
        "Preparing the isolated vision runtime with uv (first run only) ...", flush=True
    )
    environment = {**os.environ, "SCHALE_VISION_CHILD": "1"}
    return subprocess.run(
        [
            uv,
            "tool",
            "run",
            "--from",
            f"schale[student-ocr]=={version('schale')}",
            "schale",
            *arguments,
        ],
        env=environment,
        check=False,
    ).returncode
