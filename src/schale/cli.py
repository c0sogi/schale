"""Command-line interface for schale package."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Blue Archive data parser and inventory scanner")
cache_app = typer.Typer(help="Manage cached data (icons, models)")
app.add_typer(cache_app, name="cache")


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        print("schale version 0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=version_callback, is_eager=True),
    ] = None,
) -> None:
    """Blue Archive data parser and inventory scanner."""
    pass


@cache_app.command("update")
def cache_update(
    force: Annotated[
        bool,
        typer.Option("--force", help="Force re-download even if cached files exist"),
    ] = False,
) -> None:
    """Update cached equipment icons from SchaleDB.

    Downloads all equipment icons and caches them in the package data directory.
    This is the ONLY way to update cached icons - automatic downloads are disabled.

    Examples:
        schale cache update           # Update missing icons
        schale cache update --force   # Re-download all icons
    """
    try:
        from schale.cache_control import cache_collection
        from schale.scanner._icons import (  # pyright: ignore[reportPrivateUsage]
            TEMPLATE_CATEGORIES,
            download_icon_to_package,
        )
    except ImportError as e:
        print(
            "Error: Scanner dependencies not installed. Run: uv add schale[scanner]",
            file=sys.stderr,
        )
        print(f"Details: {e}", file=sys.stderr)
        raise typer.Exit(code=1)

    print("Updating equipment icon cache...")

    equipments = cache_collection.equipments
    count = 0
    failed = 0
    skipped = 0

    for _, eq in equipments.items():
        if eq.Category not in TEMPLATE_CATEGORIES:
            continue

        icon_name = eq.Icon

        try:
            path = download_icon_to_package(icon_name, force=force)
            if path is None:
                failed += 1
                print(f"  ✗ Failed: {icon_name}", file=sys.stderr)
            elif force or not path.exists():
                count += 1
                print(f"  ✓ Downloaded: {icon_name}")
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ Error downloading {icon_name}: {e}", file=sys.stderr)

    print()
    print("Summary:")
    print(f"  Downloaded: {count}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")

    if failed > 0:
        raise typer.Exit(code=1)


@cache_app.command("info")
def cache_info() -> None:
    """Show information about cached data."""
    from importlib.resources import files

    try:
        # Check bundled icons
        icons_dir = files("schale.scanner.data") / "icons"
        icons = list(icons_dir.iterdir())
        print(f"Bundled icons: {len(icons)}")

        # Check models
        models_dir = files("schale.scanner.models")
        models = [
            m.name
            for m in models_dir.iterdir()
            if m.is_file() and m.name != "__init__.py"
        ]
        print(f"Models: {', '.join(models)}")

        # Check total size
        total_size = sum(
            Path(str(icon)).stat().st_size if Path(str(icon)).exists() else 0
            for icon in icons
        )
        print(f"Total icon size: {total_size / (1024 * 1024):.1f} MB")

    except Exception as e:
        print(f"Error reading cache: {e}", file=sys.stderr)
        raise typer.Exit(code=1)


@app.command()
def scan(
    screenshot: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    confidence: Annotated[
        float,
        typer.Option("--confidence", help="Minimum confidence threshold (0.0-1.0)"),
    ] = 0.15,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Save results to JSON file"),
    ] = None,
) -> None:
    """Scan a Blue Archive inventory screenshot using CNN classifier.

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
        )
    except Exception as e:
        print(f"Error during scanning: {e}", file=sys.stderr)
        raise typer.Exit(code=1)

    # Display results
    total_cells = result.grid_dimensions[0] * result.grid_dimensions[1]
    print(
        f"Grid: {result.grid_dimensions[0]}×{result.grid_dimensions[1]} ({total_cells} cells)"
    )
    print(f"Recognized: {len(result.items)} items")
    print(f"Unrecognized: {len(result.unrecognized_cells)} cells")
    print(f"Recognition rate: {len(result.items) / total_cells * 100:.1f}%")
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
            total_qty = sum(item.quantity for item in items)
            print(f"{cat} ({len(items)} types, {total_qty} total):")

            # Sort by tier then position
            items_sorted = sorted(items, key=lambda x: (x.tier, x.grid_position))
            for item in items_sorted:
                pos = f"({item.grid_position[0]},{item.grid_position[1]})"
                blueprint_mark = " [blueprint]" if item.is_blueprint else ""
                tier_display = f"T{item.tier}" if item.tier > 0 else "EXP"
                print(f"  {pos} {tier_display} x{item.quantity}{blueprint_mark}")
            print()

    # Save to JSON if requested
    if output:
        output.write_text(result.model_dump_json(indent=2))
        print()
        print(f"Results saved to: {output}")


if __name__ == "__main__":
    app()
