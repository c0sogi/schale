"""Install the pinned public numeric runtime once, with publisher-pinned integrity."""

import hashlib
import os
from pathlib import Path
import tempfile

from filelock import FileLock
import requests

from ..cache import shared_cache
from .model_bundle import install_bundle, validate_bundle
from .paths import model_directory

MODEL_URL = (
    "https://github.com/c0sogi/schale/releases/download/numeric-runtime-v1/"
    "student-numeric-runtime-v1.zip"
)
MODEL_SHA256 = "f8233f8e60d77d4f2ad357206b0e087eedc8140e515c14f6a3239c2407f3fa5b"
MODEL_BYTES = 41769025


def ensure_numeric_model() -> Path:
    """Reuse valid local models; never contact the server on a cache hit."""
    destination = model_directory()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination.parent / "student-numeric.lock")):
        try:
            validate_bundle(destination)
            return destination
        except (OSError, ValueError):
            pass
        offline = os.environ.get("SCHALE_OFFLINE", "").lower() in {"1", "true", "yes"}
        if offline or shared_cache().settings.mode in {"offline", "manual"}:
            raise ValueError(
                "Numeric model is not cached and automatic downloads are disabled. "
                "Run extraction once with online/automatic cache access."
            )
        print(
            "Downloading numeric recognition model (39.8 MiB, first run only) ...",
            flush=True,
        )
        with tempfile.TemporaryDirectory(
            prefix=".numeric-download-", dir=destination.parent
        ) as temporary:
            stage = Path(temporary)
            archive = stage / "model.zip"
            digest = hashlib.sha256()
            size = 0
            try:
                with requests.get(MODEL_URL, stream=True, timeout=(10, 60)) as response:
                    response.raise_for_status()
                    with archive.open("xb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            size += len(chunk)
                            if size > MODEL_BYTES:
                                raise ValueError(
                                    "Numeric model download exceeds its published size"
                                )
                            digest.update(chunk)
                            stream.write(chunk)
            except requests.RequestException as error:
                raise RuntimeError(
                    "Numeric model download failed; check network access and retry the same extract command."
                ) from error
            if size != MODEL_BYTES or digest.hexdigest() != MODEL_SHA256:
                raise ValueError(
                    "Numeric model download failed the published size/SHA-256 check"
                )
            verified = stage / "verified"
            install_bundle(archive, verified)
            if destination.exists():
                backup_root = destination.parent / "setup-backups"
                backup_root.mkdir(exist_ok=True)
                backup = (
                    Path(tempfile.mkdtemp(prefix="numeric-", dir=backup_root))
                    / destination.name
                )
                destination.rename(backup)
            verified.rename(destination)
        return destination
