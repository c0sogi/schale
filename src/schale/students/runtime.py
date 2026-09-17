"""Read-only diagnostics usable even before optional vision dependencies exist."""

import importlib.util
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .model_bundle import validate_bundle
from .paths import asset_directory, cache_directory, model_directory
from ..assets import resource_report


def runtime_report(model: Path | None = None) -> dict:
    packages = {}
    for distribution, module in (
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("pillow", "PIL"),
        ("onnxruntime", "onnxruntime"),
        ("av", "av"),
    ):
        try:
            installed = (
                version(distribution) if importlib.util.find_spec(module) else None
            )
        except PackageNotFoundError:
            installed = None
        packages[distribution] = installed
    model_path = model_directory(model)
    try:
        model_info = validate_bundle(model_path)
        model_status = {"valid": True, "sha256": model_info["model_sha256"]}
    except (OSError, ValueError) as error:
        model_status = {"valid": False, "error": str(error)}
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    references = resource_report()["components"]["students"]
    return {
        "python": sys.version.split()[0],
        "cache": str(cache_directory()),
        "model": {"path": str(model_path), **model_status},
        "assets": str(asset_directory()),
        "references": references,
        "registration": "sift" if references["valid"] else "geometric-edge-registration",
        "packages": packages,
        "tools": tools,
        "ready": all(packages.values())
        and model_status["valid"],
    }
