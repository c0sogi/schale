"""Manage private local vision resources without importing vision runtimes."""

import json
from pathlib import Path
from typing import Annotated

import typer

from .assets import install_resources, pack_resources, resource_report

assets_app = typer.Typer(
    help="Install and verify your local vision resources; no downloads"
)


@assets_app.command("pack")
def pack(
    source: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Build a ZIP from your reference directory; this grants no distribution rights."""
    try:
        pack_resources(source, output)
    except (OSError, ValueError) as error:
        typer.echo(f"Resource packaging failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Created: {output.resolve()}")


@assets_app.command("install")
def install(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Verify and activate a trusted local ZIP. Previous versions are retained."""
    try:
        installed = install_resources(source)
    except (OSError, ValueError) as error:
        typer.echo(f"Resource installation failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Installed: {installed}")


@assets_app.command("info")
def info() -> None:
    """Check all installed reference files against their SHA-256 manifest."""
    report = resource_report()
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(row["valid"] for row in report["components"].values()):
        raise typer.Exit(1)
