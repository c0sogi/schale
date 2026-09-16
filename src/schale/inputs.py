"""Flexible local inputs. Importing input contracts does not load a vision library."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class VideoInput:
    path: Path | str


@dataclass(frozen=True)
class ImageInput:
    """Encoded image bytes or a path; timestamps are optional, never invented."""

    data: bytes | Path | str
    name: str = ""
    timestamp: float | None = None

    def read(self) -> bytes:
        return (
            self.data if isinstance(self.data, bytes) else Path(self.data).read_bytes()
        )


@dataclass(frozen=True)
class ImageBatch:
    images: Sequence[ImageInput | Path | str | bytes]
    name: str = "images"


InputSource = (
    VideoInput
    | ImageInput
    | ImageBatch
    | Path
    | str
    | bytes
    | Sequence[ImageInput | Path | str | bytes]
)


@dataclass(frozen=True)
class ResolvedInput:
    kind: str
    label: str
    fingerprint: str
    video: Path | None = None
    images: tuple[ImageInput, ...] = ()
    digests: tuple[str, ...] = ()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def natural_key(path: Path) -> list:
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


def resolve_input(source: InputSource) -> ResolvedInput:
    label = "images"
    if isinstance(source, VideoInput):
        path = Path(source.path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Video does not exist: {path}")
        return ResolvedInput("video", str(path), file_digest(path), video=path)
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        label = str(path)
        if path.is_dir():
            files = [
                p
                for p in path.iterdir()
                if p.is_file()
                and p.suffix.lower() in IMAGE_SUFFIXES
                and p.stem != "contact_sheet"
            ]
            # Natural order: capture2 precedes capture10. Explicit lists retain caller order.
            source = ImageBatch(
                sorted(
                    files,
                    key=natural_key,
                ),
                label,
            )
        elif path.suffix.lower() == ".json":
            manifest = json.loads(path.read_text(encoding="utf-8-sig"))
            if (
                not isinstance(manifest, dict)
                or manifest.get("kind") != "schale.images"
                or manifest.get("schema_version") != 1
            ):
                raise ValueError("Expected a schale.images schema_version=1 manifest")
            if not isinstance(manifest.get("images"), list):
                raise ValueError("Manifest images must be a list")
            entries = []
            for row in manifest["images"]:
                if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                    raise ValueError("Each manifest image requires a path")
                entries.append(
                    ImageInput(
                        (path.parent / row["path"]).resolve(),
                        str(row.get("name", "")),
                        row.get("timestamp"),
                    )
                )
            source = ImageBatch(entries, label)
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            source = ImageInput(path)
        elif path.is_file():
            return resolve_input(VideoInput(path))
        else:
            raise ValueError(f"Input does not exist: {path}")
    if isinstance(source, (bytes, ImageInput)):
        source = ImageBatch([source], label)
    if isinstance(source, ImageBatch):
        label, values = source.name, source.images
    elif isinstance(source, Sequence):
        values = source
    else:
        raise ValueError(
            "Expected a video, image, directory, image list, bytes, or image manifest"
        )
    if not values:
        raise ValueError("No screenshots in the input")
    images, hashes = [], []
    for value in values:
        image = value if isinstance(value, ImageInput) else ImageInput(value)
        if image.timestamp is not None and (
            isinstance(image.timestamp, bool)
            or not isinstance(image.timestamp, (int, float))
            or not math.isfinite(image.timestamp)
            or image.timestamp < 0
        ):
            raise ValueError(
                "Image timestamp must be finite non-negative seconds or null"
            )
        if not isinstance(image.data, bytes):
            path = Path(image.data).expanduser().resolve()
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Expected an image file: {path}")
            image = ImageInput(path, image.name or path.name, image.timestamp)
            digest = file_digest(path)
        else:
            if not image.data:
                raise ValueError("Empty encoded image")
            digest = hashlib.sha256(image.data).hexdigest()
        images.append(image)
        hashes.append(digest)
    stamp = json.dumps(
        [
            (digest, image.timestamp)
            for digest, image in zip(hashes, images, strict=True)
        ],
        separators=(",", ":"),
    )
    return ResolvedInput(
        "images",
        label,
        hashlib.sha256(stamp.encode()).hexdigest(),
        images=tuple(images),
        digests=tuple(hashes),
    )


def prepare_images(source: ResolvedInput, output: Path) -> dict:
    """Decode each accepted input once, deduplicate pixels, persist honest PNGs."""
    import cv2
    import numpy as np

    from .cache import atomic_write

    segments, discarded = [], []
    seen = set()
    for i, (item, expected) in enumerate(
        zip(source.images, source.digests, strict=True)
    ):
        raw = item.read()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Input changed while preparing image {i}")
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Invalid encoded image at index {i}: {item.name}")
        digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        if digest in seen:
            discarded.append(
                {
                    "input_index": i,
                    "reason": "duplicate image pixels",
                    "name": item.name,
                }
            )
            continue
        seen.add(digest)
        name = f"frames/image_{len(segments):06}.png"
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError(f"Could not encode input image {i}")
        atomic_write(output / name, encoded.tobytes())
        # Legacy evidence models use a numeric timestamp. Explicit metadata
        # distinguishes unknown time (placeholder 0) from a measured video time.
        timestamp = item.timestamp if item.timestamp is not None else 0.0
        index = len(segments)
        segments.append(
            {
                "key": f"segment_{index + 1:03}",
                "start": timestamp,
                "end": timestamp,
                "frames": [
                    {
                        "image": name,
                        "timestamp": timestamp,
                        "timestamp_known": item.timestamp is not None,
                        "sample_index": index,
                        "input_index": i,
                        "input_name": item.name,
                        "input_sha256": expected,
                        "pixel_sha256": digest,
                    }
                ],
            }
        )
    return {
        "source": source.label,
        "source_fingerprint": source.fingerprint,
        "input_kind": "images",
        "segments": segments,
        "discarded": discarded,
        "frame_count": len(segments),
        "timeline_unit": "image_index",
        "selection_method": "all unique decoded images in caller order",
    }
