"""Inspectable, incremental cache management; no vision dependencies required."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer

from .cache import CacheError, shared_cache
from .reference import ReferenceCatalog, Region
from pydantic import TypeAdapter

cache_app = typer.Typer(
    help="Incremental shared reference cache: policy, updates, history and integrity"
)


def echo(value):
    def dates(item):
        if isinstance(item, list):
            return [dates(row) for row in item]
        if isinstance(item, dict):
            result = {key: dates(v) for key, v in item.items()}
            for key in (
                "at",
                "first_seen",
                "fetched_at",
                "checked_at",
                "accessed_at",
                "expires_at",
                "retry_at",
            ):
                if isinstance(item.get(key), (int, float)) and item[key] > 0:
                    result[key + "_utc"] = datetime.fromtimestamp(
                        item[key], timezone.utc
                    ).isoformat()
            return result
        return item

    typer.echo(json.dumps(dates(value), ensure_ascii=False, indent=2))


@cache_app.command("info")
def info() -> None:
    cache = shared_cache()
    rows = cache.entries()
    echo(
        {
            "root": str(cache.root),
            "policy": cache.settings.model_dump(),
            "resources": len(rows),
            "logical_bytes": sum(row["size"] for row in rows),
            "entries": rows,
        }
    )


@cache_app.command("history")
def history(url: str | None = None, limit: int = 100) -> None:
    """Newest checks, changes, adoptions and failures first; timestamps are UTC epoch seconds."""
    echo(shared_cache().history(url, limit=limit))


@cache_app.command("verify")
def verify() -> None:
    """Verify current reference bodies and an installed numeric model; no network."""
    from .students.model_bundle import validate_bundle
    from .students.paths import model_directory

    cache = shared_cache()
    report = cache.verify()
    path = model_directory()
    report["model"] = "not-installed"
    if path.exists():
        try:
            validate_bundle(path)
            report["model"] = "valid"
        except (ValueError, OSError) as error:
            report["model"] = str(error)
            report["ok"] = False
    echo(report)
    if not report["ok"]:
        raise typer.Exit(1)


@cache_app.command("policy")
def policy(
    mode: str | None = None,
    json_ttl: float | None = None,
    image_ttl: float | None = None,
    min_request_interval: float | None = None,
) -> None:
    """Show/set auto, manual or offline policy and TTLs in seconds."""
    cache = shared_cache()
    changes = {
        name: value
        for name, value in locals().copy().items()
        if name in {"mode", "json_ttl", "image_ttl", "min_request_interval"}
        and value is not None
    }
    try:
        echo(
            cache.configure(**changes).model_dump()
            if changes
            else cache.settings.model_dump()
        )
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error


@cache_app.command("update")
def update(
    scope: Annotated[
        str, typer.Option(help="used (default), data, students, equipment")
    ] = "used",
    refresh: Annotated[
        bool,
        typer.Option(
            "--refresh",
            "--force",
            help="Revalidate even fresh entries; still conditional and paced",
        ),
    ] = False,
    region: str = "kr",
) -> None:
    """Only used/due resources by default. Explicit scopes populate missing references."""
    cache = shared_cache()
    cache.stats.clear()
    if scope not in {"used", "data", "students", "equipment"}:
        raise typer.BadParameter("scope must be used, data, students or equipment")
    try:
        region_value = TypeAdapter(Region).validate_python(region)
        if scope == "used":
            report = cache.update(refresh=refresh)
        else:
            names = (
                ("students", "items", "equipment", "stages", "furniture", "groups")
                if scope == "data"
                else ("students" if scope == "students" else "equipment",)
            )
            resources = []
            for name in names:
                from .reference import Dataset

                catalog = ReferenceCatalog.fetch(
                    TypeAdapter(Dataset).validate_python(name),
                    region=region_value,
                    refresh=refresh,
                    cache_mode="auto",
                )
                resources.append({"dataset": name, "entries": len(catalog.entries)})
                if scope == "students":
                    from .students.catalog import load_skill_assets
                    from .students.paths import asset_directory

                    resources.append(
                        {
                            "skill_assets": len(
                                load_skill_assets(
                                    catalog.entries,
                                    asset_directory() / "skills",
                                    refresh=refresh,
                                    cache_mode="auto",
                                )
                            )
                        }
                    )
                elif scope == "equipment":
                    icons = sorted(
                        {
                            str(row["Icon"])
                            for row in catalog.entries.values()
                            if row.get("Icon")
                        }
                    )
                    for icon in icons:
                        if not icon.replace("_", "").isalnum():
                            raise ValueError(f"Unsafe icon name: {icon}")
                        seed = (
                            Path(__file__).parent
                            / "scanner/data/icons"
                            / f"{icon}.webp"
                        )
                        cache.fetch(
                            f"https://schaledb.com/images/equipment/icon/{icon}.webp",
                            kind="image",
                            seed=seed,
                            mode="refresh" if refresh else "auto",
                        )
                    resources.append({"equipment_icons": len(icons)})
            report = {"resources": resources, "stats": dict(cache.stats)}
        echo(report)
        if (
            any(row.get("error") for row in report["resources"])
            or cache.stats.get("stale-error")
            or cache.stats.get("stale-backoff")
        ):
            raise typer.Exit(1)
    except (CacheError, ValueError, OSError) as error:
        typer.echo(f"Cache update failed: {error}", err=True)
        raise typer.Exit(1) from error
