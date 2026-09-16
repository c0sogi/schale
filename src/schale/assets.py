"""Local, checksum-verified vision resources; no bundled game art or downloads."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

from filelock import FileLock

from .cache import cache_directory

PROFILE = "bluearchive-ko-16x9-v1"
MAX_BYTES = 512 * 1024 * 1024
REQUIRED = {
    "students": {
        "labels.json",
        "header.png",
        "empty_gear.png",
        "ghost_gloves.png",
        "ghost_shoes.png",
        "ghost_hat.png",
        "potential_hp_25.png",
        "potential_attack_25.png",
        "layout_reference.npz",
    },
    "inventory": {"models/equipment_classifier.onnx", "models/class_mapping.json"},
}


def resource_directory() -> Path:
    return cache_directory() / "vision-resources"


def _safe_name(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return bool(
        parts
        and re.fullmatch(r"[A-Za-z0-9_./-]+", name)
        and not name.startswith("/")
        and "//" not in name
        and all(p not in {".", ".."} and not p.endswith(".") for p in name.split("/"))
        and all(
            p.split(".")[0].upper()
            not in {
                "CON",
                "PRN",
                "AUX",
                "NUL",
                *[f"COM{i}" for i in range(10)],
                *[f"LPT{i}" for i in range(10)],
            }
            for p in parts
        )
    )


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inventory(directory: Path) -> dict[str, str]:
    files = {}
    size = 0
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Resource bundles cannot contain symbolic links")
        if not path.is_file() or path == directory / "manifest.json":
            continue
        name = path.relative_to(directory).as_posix()
        if not _safe_name(name) or name.split("/")[0] not in REQUIRED:
            raise ValueError(f"Invalid resource path: {name}")
        if path.suffix not in {".png", ".webp", ".json", ".npz", ".onnx", ".data"}:
            raise ValueError(f"Unsupported resource file: {name}")
        size += path.stat().st_size
        files[name] = _hash(path)
    if not files or len(files) > 4096 or size > MAX_BYTES:
        raise ValueError("Resource bundle must contain 1..4096 files, at most 512 MiB")
    if len({name.casefold() for name in files}) != len(files):
        raise ValueError("Case-colliding resource paths")
    for component in {name.split("/")[0] for name in files}:
        missing = {f"{component}/{n}" for n in REQUIRED[component]} - files.keys()
        if missing:
            raise ValueError(f"Missing {component} resources: {sorted(missing)}")
    if "students/labels.json" in files:
        labels = json.loads(
            (directory / "students/labels.json").read_text(encoding="utf-8")
        )
        if not isinstance(labels, list):
            raise ValueError("Student labels must be a list")
        for label in labels:
            name = label.get("image") if isinstance(label, dict) else None
            if (
                not isinstance(name, str)
                or not _safe_name(name)
                or f"students/{name}" not in files
            ):
                raise ValueError("Student label references a missing or unsafe image")
    return files


def validate_resources(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("profile") != PROFILE
    ):
        raise ValueError("Unsupported vision resource profile")
    if manifest.get("files") != _inventory(directory):
        raise ValueError("Vision resource checksum or file inventory mismatch")
    return manifest


def pack_resources(source: Path, output: Path) -> None:
    """Package a user's own reference directory; never infer redistribution rights."""
    source, output = source.resolve(), output.resolve()
    files = _inventory(source)
    manifest = {"schema_version": 1, "profile": PROFILE, "files": files}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: preserve any existing bundle.
    with output.open("xb") as stream, ZipFile(stream, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json", json.dumps(manifest, sort_keys=True, indent=2)
        )
        for name in files:
            archive.write(source / name, name)


def install_resources(source: Path) -> Path:
    """Install immutable content-addressed resources and atomically select them."""
    root = resource_directory()
    root.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(root / "install.lock")),
        tempfile.TemporaryDirectory(prefix=".install-", dir=root) as temporary,
    ):
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        try:
            with ZipFile(source) as archive:
                members = archive.infolist()
                if (
                    not members
                    or len(members) > 4097
                    or len({m.filename.casefold() for m in members}) != len(members)
                    or sum(m.file_size for m in members) > MAX_BYTES + 1024 * 1024
                    or any(
                        m.orig_filename != m.filename
                        or not _safe_name(m.filename)
                        or m.is_dir()
                        or stat.S_ISLNK(m.external_attr >> 16)
                        for m in members
                    )
                ):
                    raise ValueError("Unsafe or oversized resource ZIP")
                for member in members:
                    path = stage / member.filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as incoming, path.open("xb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
        except BadZipFile as error:
            raise ValueError("Resources must be a valid ZIP bundle") from error
        manifest = validate_resources(stage)
        digest = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest()
        target = root / "bundles" / digest
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            validate_resources(target)
        else:
            stage.rename(target)
        pointer = Path(temporary) / "current.json"
        pointer.write_text(json.dumps({"bundle": digest}), encoding="utf-8")
        os.replace(pointer, root / "current.json")
    return target


def resolve_resources(component: str, *, verify: bool = True) -> Path:
    if component not in REQUIRED:
        raise ValueError(f"Unknown resource component: {component}")
    root = resource_directory()
    try:
        selection = json.loads((root / "current.json").read_text(encoding="utf-8"))
        digest = selection.get("bundle") if isinstance(selection, dict) else None
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Invalid resource selection")
        directory = root / "bundles" / digest
        if verify:
            validate_resources(directory)
        if not (directory / component).is_dir():
            raise ValueError(f"Bundle has no {component} resources")
        return directory / component
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Vision resources unavailable for {component}: {error}. "
            "Install your trusted local bundle with 'schale assets install BUNDLE.zip'."
        ) from error


def resource_report() -> dict:
    report: dict = {"cache": str(resource_directory()), "components": {}}
    for component in REQUIRED:
        try:
            path = resolve_resources(component)
            report["components"][component] = {"valid": True, "path": str(path)}
        except ValueError as error:
            report["components"][component] = {"valid": False, "error": str(error)}
    return report
