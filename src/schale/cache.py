"""Verified, conditional HTTP cache shared by every SchaleDB consumer.

The cache-wide lock serializes requests across processes: recheck after acquiring
it, reuse a connection, and pace requests instead of flooding the origin.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import tempfile
import time
from contextlib import closing
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import urlsplit, urlunsplit

import requests
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field

Mode = Literal["auto", "manual", "offline", "refresh"]
Kind = Literal["json", "image"]
logger = logging.getLogger(__name__)


def cache_directory() -> Path:
    return (
        Path(os.environ.get("SCHALE_CACHE_DIR") or Path.home() / ".schale/cache")
        .expanduser()
        .resolve()
    )


class CacheSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mode: Literal["auto", "manual", "offline"] = "auto"
    json_ttl: float = Field(default=7 * 86400, ge=0)
    image_ttl: float = Field(default=30 * 86400, ge=0)
    min_request_interval: float = Field(default=0.5, ge=0)
    retry_delay: float = Field(default=60, ge=1)


class CacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class CachedResource:
    url: str
    path: Path
    sha256: str
    status: str
    checked_at: float | None
    error: str | None = None

    def json(self) -> dict:
        return json.loads(self.path.read_bytes())


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_payload(data: bytes, kind: Kind) -> None:
    if not data or len(data) > 64 * 1024 * 1024:
        raise ValueError("Empty or oversized reference payload")
    if kind == "json":
        if not isinstance(json.loads(data), dict):
            raise ValueError("Reference JSON must be an object")
    elif not (
        (
            data[:4] == b"RIFF"
            and data[8:12] == b"WEBP"
            and int.from_bytes(data[4:8], "little") + 8 == len(data)
        )
        or (data.startswith(b"\x89PNG\r\n\x1a\n") and data.endswith(b"IEND\xaeB`\x82"))
        or (data.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9"))
    ):
        raise ValueError("Invalid or truncated PNG/JPEG/WebP payload")


def normalized_url(url: str) -> str:
    parts = urlsplit(url)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
    ):
        raise ValueError("Cache requires a public HTTP(S) URL without credentials")
    # Query order may be meaningful; do not rewrite or remove it.
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, "")
    )


class HttpCache:
    def __init__(
        self,
        root: Path | None = None,
        *,
        settings: CacheSettings | None = None,
        session: requests.Session | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.root = (root or cache_directory()).resolve()
        self.directory = self.root / "http-v1"
        self._settings = settings
        self.session = session or requests.Session()
        self.clock, self.sleep = clock, sleep
        self.stats: dict[str, int] = {}

    @property
    def settings(self) -> CacheSettings:
        path = self.root / "cache-policy.json"
        if self._settings is not None:
            return self._settings
        return (
            CacheSettings.model_validate_json(path.read_bytes())
            if path.exists()
            else CacheSettings()
        )

    def configure(self, **changes) -> CacheSettings:
        settings = CacheSettings.model_validate(
            {**self.settings.model_dump(), **changes}
        )
        atomic_write(
            self.root / "cache-policy.json", settings.model_dump_json(indent=2).encode()
        )
        self._settings = None
        return settings

    def _connect(self) -> sqlite3.Connection:
        self.directory.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.directory / "index.sqlite3", timeout=120, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS entries (
                url TEXT PRIMARY KEY, kind TEXT NOT NULL, sha256 TEXT, size INTEGER DEFAULT 0,
                etag TEXT, modified TEXT, first_seen REAL, fetched_at REAL, checked_at REAL,
                accessed_at REAL, expires_at REAL DEFAULT 0, retry_at REAL DEFAULT 0,
                failures INTEGER DEFAULT 0, error TEXT);
            CREATE TABLE IF NOT EXISTS versions (
                url TEXT, sha256 TEXT, first_seen REAL, last_seen REAL, size INTEGER,
                PRIMARY KEY (url, sha256));
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, url TEXT, at REAL, status TEXT, sha256 TEXT,
                downloaded_bytes INTEGER DEFAULT 0, error TEXT);
            CREATE TABLE IF NOT EXISTS control (key TEXT PRIMARY KEY, value REAL);
        """)
        return connection

    def _blob(self, digest: str) -> Path:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise CacheError("Invalid digest in cache index")
        return self.directory / "objects" / digest[:2] / digest

    def _valid(self, row) -> bool:
        if row is None or not row["sha256"]:
            return False
        try:
            path = self._blob(row["sha256"])
            with path.open("rb") as stream:
                return (
                    hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]
                )
        except OSError:
            return False

    def _event(self, db, url, status, digest=None, size=0, error=None):
        db.execute(
            "INSERT INTO events(url,at,status,sha256,downloaded_bytes,error) VALUES(?,?,?,?,?,?)",
            (url, self.clock(), status, digest, size, error),
        )

    def _result(self, db, row, status: str) -> CachedResource:
        self.stats[status] = self.stats.get(status, 0) + 1
        db.execute(
            "UPDATE entries SET accessed_at=? WHERE url=?", (self.clock(), row["url"])
        )
        return CachedResource(
            row["url"],
            self._blob(row["sha256"]),
            row["sha256"],
            status,
            row["checked_at"],
            row["error"],
        )

    def _store(
        self,
        db,
        url,
        kind,
        data,
        now,
        expires,
        etag=None,
        modified=None,
        status="downloaded",
    ):
        digest = hashlib.sha256(data).hexdigest()
        path = self._blob(digest)
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            atomic_write(path, data)
        db.execute(
            """INSERT INTO entries(url,kind,sha256,size,etag,modified,first_seen,fetched_at,checked_at,expires_at)
                      VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(url) DO UPDATE SET
                      kind=excluded.kind,sha256=excluded.sha256,size=excluded.size,etag=excluded.etag,
                      modified=excluded.modified,fetched_at=excluded.fetched_at,checked_at=excluded.checked_at,
                      expires_at=excluded.expires_at,retry_at=0,failures=0,error=NULL""",
            (
                url,
                kind,
                digest,
                len(data),
                etag,
                modified,
                now,
                now,
                None if status == "adopted" else now,
                expires,
            ),
        )
        db.execute(
            """INSERT INTO versions VALUES(?,?,?,?,?) ON CONFLICT(url,sha256)
                      DO UPDATE SET last_seen=excluded.last_seen""",
            (url, digest, now, now, len(data)),
        )
        self._event(db, url, status, digest, 0 if status == "adopted" else len(data))

    def fetch(
        self,
        url: str,
        *,
        kind: Kind = "json",
        mode: Mode | None = None,
        ttl: float | None = None,
        seed: Path | None = None,
    ) -> CachedResource:
        url = normalized_url(url)
        settings = self.settings
        selected = mode or settings.mode
        if settings.mode == "offline" or os.environ.get(
            "SCHALE_OFFLINE", ""
        ).lower() in {"1", "true", "yes"}:
            selected = "offline"
        if selected not in {"auto", "manual", "offline", "refresh"} or kind not in {
            "json",
            "image",
        }:
            raise ValueError("Invalid cache mode or resource kind")
        lifetime = (
            ttl
            if ttl is not None
            else settings.json_ttl
            if kind == "json"
            else settings.image_ttl
        )
        if lifetime < 0:
            raise ValueError("Cache TTL cannot be negative")
        started = self.clock()
        self.directory.mkdir(parents=True, exist_ok=True)
        with (
            FileLock(str(self.directory / "requests.lock"), timeout=120),
            closing(self._connect()) as db,
        ):
            now = self.clock()
            row = db.execute("SELECT * FROM entries WHERE url=?", (url,)).fetchone()
            valid = self._valid(row)
            # Existing cache files are adopted once. Their hash establishes a new
            # integrity baseline, not proof of the original server's signature.
            if row is None and seed is not None and seed.is_file():
                try:
                    data = seed.read_bytes()
                    validate_payload(data, kind)
                except (ValueError, OSError):
                    pass
                else:
                    self._store(
                        db,
                        url,
                        kind,
                        data,
                        now,
                        seed.stat().st_mtime + lifetime,
                        status="adopted",
                    )
                    row = db.execute(
                        "SELECT * FROM entries WHERE url=?", (url,)
                    ).fetchone()
                    valid = True
            if valid and (
                selected in {"manual", "offline"}
                or (selected == "auto" and now < row["expires_at"])
                or (row["checked_at"] is not None and row["checked_at"] >= started)
            ):
                return self._result(
                    db, row, "hit" if selected != "offline" else "offline-hit"
                )
            if selected == "offline":
                raise CacheError(f"Offline cache missing or corrupt: {url}")
            host = urlsplit(url).netloc
            blocked = db.execute(
                "SELECT value FROM control WHERE key=?", ("blocked:" + host,)
            ).fetchone()
            retry_at = max(row["retry_at"] if row else 0, blocked[0] if blocked else 0)
            if retry_at > now:
                if valid:
                    return self._result(db, row, "stale-backoff")
                raise CacheError(f"Retry deferred until {retry_at:.0f}: {url}")
            headers = {"User-Agent": "schale/0.1.0 (+https://github.com/c0sogi/schale)"}
            if valid and row["etag"]:
                headers["If-None-Match"] = row["etag"]
            if valid and row["modified"]:
                headers["If-Modified-Since"] = row["modified"]
            last = db.execute(
                "SELECT value FROM control WHERE key='last_request'"
            ).fetchone()
            delay = max(
                0, (last[0] if last else 0) + settings.min_request_interval - now
            )
            if delay:
                self.sleep(delay)
            db.execute(
                "INSERT OR REPLACE INTO control VALUES('last_request',?)",
                (self.clock(),),
            )
            self.stats["requests"] = self.stats.get("requests", 0) + 1
            response = None
            try:
                response = self.session.get(url, headers=headers, timeout=(10, 30))
                now = self.clock()
                if response.status_code == 304:
                    if not valid or not (
                        {"If-None-Match", "If-Modified-Since"} & headers.keys()
                    ):
                        raise ValueError(
                            "Server returned 304 without a valid local body"
                        )
                    db.execute(
                        """UPDATE entries SET checked_at=?,expires_at=?,retry_at=0,failures=0,error=NULL,
                                  etag=COALESCE(?,etag),modified=COALESCE(?,modified) WHERE url=?""",
                        (
                            now,
                            now + lifetime,
                            response.headers.get("ETag"),
                            response.headers.get("Last-Modified"),
                            url,
                        ),
                    )
                    self._event(db, url, "not-modified", row["sha256"])
                    status = "not-modified"
                else:
                    response.raise_for_status()
                    if response.status_code != 200:
                        raise ValueError(
                            f"Expected HTTP 200 or 304, got {response.status_code}"
                        )
                    data = response.content
                    validate_payload(data, kind)
                    status = (
                        "unchanged"
                        if valid and hashlib.sha256(data).hexdigest() == row["sha256"]
                        else "downloaded"
                    )
                    self._store(
                        db,
                        url,
                        kind,
                        data,
                        now,
                        now + lifetime,
                        response.headers.get("ETag"),
                        response.headers.get("Last-Modified"),
                        status,
                    )
            except (requests.RequestException, ValueError) as error:
                now = self.clock()
                failures = (row["failures"] if row else 0) + 1
                delay = min(86400, settings.retry_delay * 2 ** min(failures - 1, 16))
                if response is not None and response.status_code in {404, 410}:
                    delay = 86400
                if response is not None and response.status_code in {429, 503}:
                    retry = response.headers.get("Retry-After", "")
                    try:
                        delay = max(delay, float(retry))
                    except ValueError:
                        try:
                            delay = max(
                                delay, parsedate_to_datetime(retry).timestamp() - now
                            )
                        except (ValueError, TypeError, OverflowError):
                            pass
                    db.execute(
                        "INSERT OR REPLACE INTO control VALUES(?,?)",
                        ("blocked:" + host, now + delay),
                    )
                db.execute(
                    """INSERT INTO entries(url,kind,first_seen,retry_at,failures,error) VALUES(?,?,?,?,?,?)
                              ON CONFLICT(url) DO UPDATE SET retry_at=excluded.retry_at,failures=excluded.failures,error=excluded.error""",
                    (url, kind, now, now + delay, failures, str(error)),
                )
                self._event(db, url, "failed", error=str(error))
                if valid:
                    logger.warning("Using verified stale cache for %s: %s", url, error)
                    row = db.execute(
                        "SELECT * FROM entries WHERE url=?", (url,)
                    ).fetchone()
                    return self._result(db, row, "stale-error")
                raise CacheError(f"Cannot fetch {url}: {error}") from error
            finally:
                if response is not None:
                    response.close()
            row = db.execute("SELECT * FROM entries WHERE url=?", (url,)).fetchone()
            return self._result(db, row, status)

    def entries(self) -> list[dict]:
        if not (self.directory / "index.sqlite3").exists():
            return []
        with closing(self._connect()) as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM entries ORDER BY COALESCE(checked_at,first_seen) DESC,url"
                )
            ]

    def generation(self) -> int:
        if not (self.directory / "index.sqlite3").exists():
            return 0
        with closing(self._connect()) as db:
            return db.execute("SELECT COALESCE(MAX(id),0) FROM events").fetchone()[0]

    def history(self, url: str | None = None, *, limit: int = 100) -> list[dict]:
        if not (self.directory / "index.sqlite3").exists():
            return []
        with closing(self._connect()) as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM events WHERE (? IS NULL OR url=?) ORDER BY id DESC LIMIT ?",
                    (
                        url,
                        normalized_url(url) if url else None,
                        max(1, min(limit, 10000)),
                    ),
                )
            ]

    def verify(self) -> dict:
        rows = self.entries()
        bad = [row["url"] for row in rows if row["sha256"] and not self._valid(row)]
        return {
            "checked": sum(bool(row["sha256"]) for row in rows),
            "corrupt": bad,
            "ok": not bad,
        }

    def update(self, *, refresh: bool = False) -> dict:
        results = []
        for row in self.entries():
            try:
                item = self.fetch(
                    row["url"], kind=row["kind"], mode="refresh" if refresh else "auto"
                )
                results.append(
                    {"url": item.url, "status": item.status, "error": item.error}
                )
            except CacheError as error:
                results.append(
                    {"url": row["url"], "status": "failed", "error": str(error)}
                )
        return {"resources": results, "stats": dict(self.stats)}


@lru_cache(maxsize=8)
def _shared(root: Path) -> HttpCache:
    return HttpCache(root)


def shared_cache() -> HttpCache:
    return _shared(cache_directory())
