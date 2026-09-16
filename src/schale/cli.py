"""Command-line interface for schale package."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from .toolkit_cli import collection_app, data_app, rewards_app
from .cache_cli import cache_app
from .assets_cli import assets_app

app = typer.Typer(help="SchaleDB-compatible Blue Archive automation toolkit")
app.add_typer(data_app, name="data")
app.add_typer(collection_app, name="collection")
app.add_typer(rewards_app, name="rewards")
app.add_typer(cache_app, name="cache")
app.add_typer(assets_app, name="assets")
students_app = typer.Typer(
    help="Extract student growth information with visual evidence"
)
app.add_typer(students_app, name="students")


@students_app.command("extract")
def students_extract(
    source: Annotated[
        list[Path],
        typer.Argument(
            exists=True,
            help="Video, image, directory, image manifest, or multiple images",
        ),
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="New output directory")
    ],
    asset_cache: Annotated[
        Path | None, typer.Option(help="Reusable SchaleDB image cache")
    ] = None,
    resume: Annotated[
        bool, typer.Option(help="Resume a matching extraction directory")
    ] = False,
    reader: Annotated[
        str, typer.Option(help="ctc (default), template, or explicit auto fallback")
    ] = "ctc",
    numeric_model: Annotated[
        Path | None, typer.Option(help="Verified ONNX student model bundle directory")
    ] = None,
) -> None:
    """Extract screenshots and fields; open review.html to inspect uncertain values."""
    try:
        from schale.students.extract import extract

        result = extract(
            source[0] if len(source) == 1 else source,
            output,
            asset_cache=asset_cache,
            resume=resume,
            reader=reader,
            numeric_model=numeric_model,
            progress=lambda message: print(message, flush=True),
        )
    except ImportError as error:
        typer.echo(
            "Student extraction dependencies missing; install 'schale[student-ocr]' (or 'schale[vision]' for --reader template).",
            err=True,
        )
        raise typer.Exit(1) from error
    except (ValueError, RuntimeError, OSError) as error:
        typer.echo(f"Extraction failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(
        f"Students: {len(result.students)}; {result.diagnostics['field_statuses']}"
    )
    typer.echo(f"Review: {output.resolve() / 'review.html'}")


@students_app.command("inspect")
def students_inspect(
    extraction: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    video: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Build a read-only timeline/ROI/registration inspector from saved extraction."""
    try:
        from schale.students.inspector import build_inspector

        report = build_inspector(
            extraction,
            output,
            video=video,
            progress=lambda message: print(message, flush=True),
        )
    except ImportError as error:
        typer.echo(
            "Inspector dependencies missing; install 'schale[student-ocr]'.", err=True
        )
        raise typer.Exit(1) from error
    except (ValueError, RuntimeError, OSError) as error:
        typer.echo(f"Inspector failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Inspected {report['selected_frames']} selected frames")
    typer.echo(f"Inspector: {output.resolve() / 'index.html'}")


@students_app.command("install-model")
def students_install_model(
    source: Annotated[
        Path, typer.Argument(exists=True, help="Model bundle directory or ZIP")
    ],
    destination: Annotated[
        Path | None, typer.Option(help="Defaults to SCHALE_CACHE_DIR/student-numeric")
    ] = None,
) -> None:
    """Install a verified model bundle once; subsequent extract commands auto-select it."""
    from schale.students.model_bundle import install_bundle
    from schale.students.paths import cache_directory

    destination = destination or cache_directory() / "student-numeric"
    try:
        install_bundle(source, destination)
    except (ValueError, OSError) as error:
        typer.echo(f"Model installation failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Installed: {destination.resolve()}")


@students_app.command("doctor")
def students_doctor(
    numeric_model: Annotated[
        Path | None, typer.Option(help="Check a specific model bundle")
    ] = None,
) -> None:
    """Check cache paths, model checksum, optional packages and FFmpeg; no downloads."""
    import json
    from schale.students.runtime import runtime_report

    report = runtime_report(numeric_model)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ready"]:
        raise typer.Exit(1)


@students_app.command("export")
def students_export(
    results: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    correction_file: Annotated[
        Path | None, typer.Option("--corrections", exists=True, dir_okay=False)
    ] = None,
    base: Annotated[
        Path | None, typer.Option("--base", exists=True, dir_okay=False)
    ] = None,
    allow_observed: Annotated[
        bool, typer.Option(help="Explicitly accept single-frame observations")
    ] = False,
) -> None:
    """Export reviewed records. SchaleDB imports replace the whole collection: prefer --base."""
    import json
    from schale.students.export import corrections, decode_collection, export_collection
    from schale.students.models import Extraction

    try:
        if (
            output.exists()
            or output.with_suffix(output.suffix + ".report.json").exists()
        ):
            raise ValueError("Export output already exists; choose a new filename")
        result = Extraction.model_validate_json(results.read_text(encoding="utf-8"))
        if correction_file:
            catalog = json.loads(
                (results.parent / "students.json").read_text(encoding="utf-8")
            )
            result = corrections(result, correction_file, catalog)
        baseline = decode_collection(base.read_text(encoding="utf-8")) if base else None
        text, report = export_collection(
            result, baseline, allow_observed=allow_observed
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="ascii")
        output.with_suffix(output.suffix + ".report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except (ValueError, OSError) as error:
        typer.echo(f"Export failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(
        f"Exported {len(report['updated'])} students; skipped {len(report['skipped'])}."
    )
    typer.echo(report["warning"])


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        from importlib.metadata import version

        print(f"schale version {version('schale')}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=version_callback, is_eager=True),
    ] = None,
) -> None:
    """SchaleDB-compatible Blue Archive automation toolkit."""
    pass


@app.command()
def scan(
    screenshot: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    confidence: Annotated[
        float,
        typer.Option("--confidence", help="Minimum confidence threshold (0.0-1.0)"),
    ] = 0.65,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Save results to JSON file"),
    ] = None,
    numeric_model: Annotated[
        Path | None,
        typer.Option("--numeric-model", help="Installed UI text model directory"),
    ] = None,
) -> None:
    """Scan inventory using ONNX text and catalog icon verification.

    Examples:
        schale scan screenshot.png
        schale scan screenshot.png --confidence 0.6
        schale scan screenshot.png --output results.json
    """
    try:
        from schale.scanner import scan_inventory
    except ImportError as e:
        print(
            "Error: Scanner dependencies not installed. Run: uv add schale[scanner]",
            file=sys.stderr,
        )
        print(f"Details: {e}", file=sys.stderr)
        raise typer.Exit(code=1)

    print(f"Scanning: {screenshot}")
    print(f"Confidence threshold: {confidence}")
    print()

    try:
        result = scan_inventory(
            str(screenshot),
            confidence_threshold=confidence,
            numeric_model=numeric_model,
        )
    except Exception as e:
        print(f"Error during scanning: {e}", file=sys.stderr)
        raise typer.Exit(code=1)

    # Display results
    total_cells = len(result.observations)
    print(
        f"Grid: {result.grid_dimensions[0]}×{result.grid_dimensions[1]} ({total_cells} cells)"
    )
    print(f"Recognized: {len(result.items)} items")
    print(f"Unrecognized: {len(result.unrecognized_cells)} cells")
    print(
        f"Identity coverage: {len(result.items) / total_cells * 100:.1f}% (not accuracy)"
    )
    print(
        f"Exact quantities: {sum(item.quantity_status == 'exact' for item in result.items)}"
    )
    print()

    # Group items by category and display with details
    from collections import defaultdict

    categories: dict[str, list] = defaultdict(list)
    for item in result.items:
        categories[item.category].append(item)

    if categories:
        print("Items by category:")
        print()
        for cat in sorted(categories.keys()):
            items = categories[cat]
            total_qty = sum(
                item.quantity for item in items if item.quantity is not None
            )
            print(f"{cat} ({len(items)} types, {total_qty} exact-count subtotal):")

            # Sort by tier then position
            items_sorted = sorted(items, key=lambda x: (x.tier, x.grid_position))
            for item in items_sorted:
                pos = f"({item.grid_position[0]},{item.grid_position[1]})"
                blueprint_mark = " [blueprint]" if item.is_blueprint else ""
                tier_display = f"T{item.tier}" if item.tier > 0 else "EXP"
                count = (
                    str(item.quantity)
                    if item.quantity is not None
                    else f"? [{item.quantity_text}; {item.quantity_status}]"
                )
                print(f"  {pos} {tier_display} x{count}{blueprint_mark}")
            print()

    # Save to JSON if requested
    if output:
        output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print()
        print(f"Results saved to: {output}")


if __name__ == "__main__":
    app()
