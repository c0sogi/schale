"""Data and collection tools; importing this module needs no vision runtime."""

import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from pydantic import TypeAdapter
from requests import RequestException

from .account import AccountSnapshot
from .adapters.extraction import from_extraction
from .adapters.schaledb import decode_collection, from_schaledb, to_schaledb
from .collections import compare, merge
from .literal import RewardCondition, Server
from .reference import Dataset, ReferenceCatalog, Region
from .stage_rewards import StageRewards
from .students.models import Extraction

data_app = typer.Typer(help="Fetch SchaleDB reference definitions")
collection_app = typer.Typer(
    help="Import, compare, merge and export portable account state"
)
rewards_app = typer.Typer(help="Calculate stage reward expectations from SchaleDB")
Input = Annotated[Path, typer.Argument(exists=True, dir_okay=False)]
Output = Annotated[
    Path, typer.Option("--output", "-o", help="New output file; never overwrite")
]


@contextmanager
def _errors():
    try:
        yield
    except (ValueError, OSError, KeyError, RuntimeError, RequestException) as error:
        typer.echo(f"Failed: {error}", err=True)
        raise typer.Exit(1) from error


def _new_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, indent=2)
        )
        stream.write("\n")
    typer.echo(str(path.resolve()))


def _account(path: Path) -> AccountSnapshot:
    return AccountSnapshot.model_validate_json(path.read_text(encoding="utf-8-sig"))


@data_app.command("fetch")
def fetch_data(
    dataset: Annotated[
        str,
        typer.Argument(help="students, items, equipment, stages, furniture, groups"),
    ],
    output: Output,
    region: str = "kr",
    refresh: bool = False,
) -> None:
    """Save raw ID-keyed definitions and a .meta.json provenance sidecar."""
    with _errors():
        meta = output.with_suffix(".meta.json")
        if meta == output:
            raise ValueError("Use an output filename other than *.meta.json")
        _new_outputs(output, meta)
        catalog = ReferenceCatalog.fetch(
            TypeAdapter(Dataset).validate_python(dataset),
            region=TypeAdapter(Region).validate_python(region),
            refresh=refresh,
        )
        _write(output, catalog.entries)
        _write(meta, catalog.model_dump(exclude={"entries"}))


@collection_app.command("from-extraction")
def convert_extraction(source: Input, output: Output) -> None:
    """Convert results.json, retaining all 19 fields and their visual evidence."""
    with _errors():
        _new_outputs(output)
        result = Extraction.model_validate_json(source.read_text(encoding="utf-8-sig"))
        _write(output, from_extraction(result).model_dump())


@collection_app.command("import")
def import_collection(
    source: Input,
    output: Output,
    catalog: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Import SchaleDB base64 export; absent fields remain absent, never zero."""
    with _errors():
        _new_outputs(output)
        result = from_schaledb(
            source.read_text(encoding="utf-8-sig"),
            source=str(source.resolve()),
            catalog=ReferenceCatalog.from_file(catalog) if catalog else None,
        )
        _write(output, result.model_dump())


@collection_app.command("diff")
def diff_collection(before: Input, after: Input, output: Output) -> None:
    """Report value/status changes and IDs missing from the new snapshot."""
    with _errors():
        _new_outputs(output)
        _write(output, compare(_account(before), _account(after)))


@collection_app.command("merge")
def merge_collection(
    base: Input, incoming: Input, output: Output, allow_observed: bool = False
) -> None:
    """Apply accepted incoming values without deleting unseen students."""
    with _errors():
        report = output.with_suffix(".report.json")
        if report == output:
            raise ValueError("Use an output filename other than *.report.json")
        _new_outputs(output, report)
        result, details = merge(
            _account(base), _account(incoming), allow_observed=allow_observed
        )
        _write(output, result.model_dump())
        _write(report, details)


@collection_app.command("export")
def export_collection(
    source: Input,
    output: Output,
    base: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    allow_observed: bool = False,
) -> None:
    """Produce a SchaleDB import string and an export report; never upload it."""
    with _errors():
        report = output.with_suffix(".report.json")
        if report == output:
            raise ValueError("Use an output filename other than *.report.json")
        _new_outputs(output, report)
        baseline = (
            decode_collection(base.read_text(encoding="utf-8-sig")) if base else None
        )
        encoded, details = to_schaledb(
            _account(source), baseline, allow_observed=allow_observed
        )
        _write(output, encoded)
        _write(report, details)


@rewards_app.command("calculate")
def calculate_rewards(
    stage: Annotated[
        list[int], typer.Option("--stage", help="Stage ID; repeat for multiple stages")
    ],
    output: Output,
    server: str = "Global",
    condition: Annotated[
        list[str] | None,
        typer.Option("--condition", help="Optional reward condition; repeat as needed"),
    ] = None,
) -> None:
    """Calculate one-run expectations (not guaranteed drops or a farming plan)."""
    with _errors():
        _new_outputs(output)
        selected_server = TypeAdapter(Server).validate_python(server)
        conditions = TypeAdapter(list[RewardCondition]).validate_python(condition or [])
        if any(sid <= 0 for sid in stage):
            raise ValueError("Stage IDs must be positive")
        results = StageRewards.from_stages(
            stage, server=selected_server, reward_conditions=conditions
        )
        _write(
            output,
            {
                "schema_version": 1,
                "server": selected_server,
                "conditions": conditions,
                "stages": [
                    {
                        "stage_id": sid,
                        "entry_cost": summary.stage.root.EntryCost,
                        "rewards": [asdict(reward) for reward in summary.rewards],
                    }
                    for sid, summary in zip(stage, results, strict=True)
                ],
            },
        )
