"""Safe automatic CLI destinations without overwriting previous extractions."""

from pathlib import Path


def select_output(sources: list[Path], requested: Path | None, resume: bool) -> Path:
    if resume:
        if requested is None:
            raise ValueError("--resume requires -o pointing to the existing extraction")
        previous = requested.expanduser().resolve()
        if not (previous / "segments.json").is_file():
            raise ValueError(
                "--resume requires an existing extraction with segments.json; do not point it at your home directory"
            )
        return previous
    first = sources[0].expanduser().resolve()
    name = f"{first.stem}-schale" if len(sources) == 1 else "students-schale"
    if requested is None:
        parent = first.parent
        candidate = parent / name
    else:
        candidate = requested.expanduser().resolve()
        if candidate.is_file():
            raise ValueError(f"Output is a file, not a directory: {candidate}")
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
        # Treat an occupied directory as a parent, never as something to clear.
        parent = candidate
        candidate = parent / name
    base = candidate
    index = 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}-{index}")
        index += 1
    return candidate
