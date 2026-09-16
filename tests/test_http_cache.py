"""Traffic contracts measured with fake transports and a local HTTP origin."""

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import pytest
import requests
from typer.testing import CliRunner

from schale.cache import CacheError, CacheSettings, HttpCache, shared_cache
from schale.cli import app

URL = "https://schaledb.com/data/kr/students.min.json"


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch):
    # These tests use fake transports or a loopback origin, not SchaleDB.
    monkeypatch.delenv("SCHALE_OFFLINE", raising=False)


class Clock:
    now = 100000.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


def response(
    status: int = 200,
    body: bytes = b'{"1":{"Id":1}}',
    headers: dict[str, str] | None = None,
    **extra_headers: str,
):
    result = requests.Response()
    result.status_code, result._content = status, body
    setattr(result, "_content_consumed", True)
    result.headers.update(headers or {})
    result.headers.update(extra_headers)
    return result


class Transport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def cache(tmp_path, responses, **settings):
    clock = Clock()
    transport = Transport(responses)
    instance = HttpCache(
        tmp_path,
        session=cast(requests.Session, transport),
        clock=clock,
        sleep=clock.sleep,
        settings=CacheSettings(json_ttl=10, **settings),
    )
    return instance, transport, clock


def test_fresh_hit_makes_zero_network_and_expired_uses_304(tmp_path):
    c, transport, clock = cache(
        tmp_path,
        [
            response(
                ETag='"v1"', headers={"Last-Modified": "Tue, 15 Sep 2026 00:00:00 GMT"}
            ),
            response(304),
        ],
    )
    first = c.fetch(URL)
    original_mtime = first.path.stat().st_mtime_ns
    clock.now += 1
    assert c.fetch(URL).status == "hit" and len(transport.calls) == 1
    clock.now += 11
    second = c.fetch(URL)
    assert second.status == "not-modified" and second.path == first.path
    assert first.path.stat().st_mtime_ns == original_mtime
    assert transport.calls[1][1]["headers"]["If-None-Match"] == '"v1"'
    assert "If-Modified-Since" in transport.calls[1][1]["headers"]
    assert c.history()[0]["downloaded_bytes"] == 0
    assert c.entries()[0]["fetched_at"] < c.entries()[0]["checked_at"]


def test_changed_versions_retained_and_identical_bytes_deduplicated(tmp_path):
    c, transport, clock = cache(
        tmp_path,
        [
            response(),
            response(body=b'{"1":{"Id":2}}'),
            response(body=b'{"1":{"Id":2}}'),
        ],
    )
    a = c.fetch(URL)
    clock.now += 11
    b = c.fetch(URL)
    clock.now += 1
    other = c.fetch(URL.replace("students", "items"))
    assert a.path != b.path and a.path.exists()
    assert b.path == other.path
    assert len(list((c.directory / "objects").glob("*/*"))) == 2
    assert len(transport.calls) == 3 and c.verify()["ok"]


def test_server_without_validators_gets_no_head_and_one_due_get(tmp_path):
    c, transport, clock = cache(tmp_path, [response(), response()])
    first = c.fetch(URL)
    clock.now += 11
    second = c.fetch(URL)
    assert second.status == "unchanged" and first.path == second.path
    assert len(transport.calls) == 2
    assert "If-None-Match" not in transport.calls[1][1]["headers"]


def test_corruption_not_served_offline_and_recovery_is_unconditional(tmp_path):
    c, transport, clock = cache(
        tmp_path, [response(ETag='"v1"'), response(ETag='"v1"')]
    )
    first = c.fetch(URL)
    first.path.write_bytes(b"corrupt")
    assert c.verify()["corrupt"] == [URL]
    with pytest.raises(CacheError, match="Offline"):
        c.fetch(URL, mode="offline")
    clock.now += 1
    fixed = c.fetch(URL)
    assert fixed.json() == {"1": {"Id": 1}}
    assert "If-None-Match" not in transport.calls[-1][1]["headers"]


def test_failure_backoff_returns_only_verified_stale(tmp_path):
    c, transport, clock = cache(tmp_path, [response(), requests.Timeout("test")])
    original = c.fetch(URL)
    clock.now += 11
    assert c.fetch(URL).status == "stale-error"
    assert c.fetch(URL, mode="refresh").status == "stale-backoff"
    assert len(transport.calls) == 2
    original.path.write_bytes(b"broken")
    with pytest.raises(CacheError, match="Retry deferred"):
        c.fetch(URL)


@pytest.mark.parametrize("status", [404, 429, 503])
def test_negative_cache_and_host_retry_after(tmp_path, status):
    c, transport, clock = cache(
        tmp_path, [response(status, headers={"Retry-After": "120"})]
    )
    with pytest.raises(CacheError):
        c.fetch(URL)
    clock.now += 1
    with pytest.raises(CacheError, match="Retry deferred"):
        c.fetch(URL)
    if status != 404:
        with pytest.raises(CacheError, match="Retry deferred"):
            c.fetch(URL.replace("students", "items"))
    assert len(transport.calls) == 1


def test_bad_200_never_replaces_good_body(tmp_path):
    c, _, clock = cache(tmp_path, [response(), response(body=b"<html>error</html>")])
    first = c.fetch(URL)
    clock.now += 11
    failed = c.fetch(URL)
    assert failed.status == "stale-error" and failed.path == first.path
    assert c.verify()["ok"]


def test_seed_adoption_has_no_network_and_preserves_original(tmp_path):
    c, transport, _ = cache(tmp_path, [])
    seed = tmp_path / "old.json"
    seed.write_bytes(b'{"1":{}}')
    actual = c.fetch(URL, seed=seed)
    assert actual.path != seed and seed.read_bytes() == actual.path.read_bytes()
    assert not transport.calls
    assert c.history()[0]["status"] == "adopted"
    assert c.entries()[0]["checked_at"] is None


def test_manual_policy_and_explicit_refresh(tmp_path):
    c, transport, clock = cache(
        tmp_path, [response(ETag='"manual"'), response(304)], mode="manual"
    )
    c.fetch(URL)
    clock.now += 100
    assert c.fetch(URL).status == "hit"
    assert len(transport.calls) == 1
    assert c.fetch(URL, mode="refresh").status == "not-modified"


def test_global_offline_overrides_force_and_does_not_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHALE_OFFLINE", "1")
    c, transport, _ = cache(tmp_path, [])
    with pytest.raises(CacheError, match="Offline"):
        c.fetch(URL, mode="refresh")
    assert not transport.calls


def test_persistent_offline_policy_also_overrides_refresh(tmp_path):
    c, transport, _ = cache(tmp_path, [], mode="offline")
    with pytest.raises(CacheError, match="Offline"):
        c.fetch(URL, mode="refresh")
    assert not transport.calls


def test_unconditional_304_is_not_a_valid_revalidation(tmp_path):
    c, transport, clock = cache(tmp_path, [response(), response(304)])
    c.fetch(URL)
    clock.now += 11
    assert c.fetch(URL).status == "stale-error"
    assert len(transport.calls) == 2


def test_catalog_pin_preserves_reference_order(tmp_path, monkeypatch):
    from schale.students.catalog import load_catalog
    import schale.data_control as module

    c, _, _ = cache(
        tmp_path, [response(body=b'{"2":{"Id":2,"Name":"B"},"1":{"Id":1,"Name":"A"}}')]
    )
    monkeypatch.setattr(module, "shared_cache", lambda: c)
    rows, _ = load_catalog(tmp_path / "students.json")
    assert list(rows) == ["2", "1"]


def test_pacing_between_distinct_resources(tmp_path):
    c, transport, clock = cache(
        tmp_path, [response(), response()], min_request_interval=0.5
    )
    c.fetch(URL)
    start = clock.now
    c.fetch(URL.replace("students", "items"))
    assert clock.now - start == 0.5 and len(transport.calls) == 2


def test_incremental_update_only_checks_due_used_resources(tmp_path):
    c, transport, clock = cache(
        tmp_path, [response(ETag='"a"'), response(ETag='"b"'), response(304)]
    )
    c.fetch(URL, ttl=10)
    c.fetch(URL.replace("students", "items"), ttl=100)
    clock.now += 11
    report = c.update()
    assert len(transport.calls) == 3
    assert sorted(r["status"] for r in report["resources"]) == ["hit", "not-modified"]


def test_cli_policy_info_verify_need_no_vision_or_network(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHALE_CACHE_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(app, ["cache", "info"])
    assert result.exit_code == 0 and not tmp_path.joinpath("http-v1").exists()
    result = runner.invoke(
        app, ["cache", "policy", "--mode", "manual", "--json-ttl", "120"]
    )
    assert result.exit_code == 0, result.output
    assert shared_cache().settings.mode == "manual"
    assert shared_cache().settings.json_ttl == 120
    assert runner.invoke(app, ["cache", "verify"]).exit_code == 0
    assert runner.invoke(app, ["cache", "policy", "--json-ttl", "-1"]).exit_code == 1


def test_concurrent_processes_download_same_resource_once(tmp_path):
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            data = b'{"1":{"Id":1}}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("ETag", '"shared"')
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/catalog"
    code = "from pathlib import Path; from schale.cache import HttpCache; import sys; print(HttpCache(Path(sys.argv[1])).fetch(sys.argv[2]).sha256)"

    def run(_):
        return subprocess.run(
            [sys.executable, "-c", code, str(tmp_path), url],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            digests = list(pool.map(run, range(4)))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(set(digests)) == 1 and hits == ["/catalog"]


def test_extraction_catalog_reuses_reference_cache_across_outputs(
    tmp_path, monkeypatch
):
    from schale.students.catalog import load_catalog
    import schale.data_control as data_control

    c, transport, _ = cache(tmp_path, [response(body=b'{"1":{"Id":1,"Name":"Test"}}')])
    monkeypatch.setattr(data_control, "shared_cache", lambda: c)
    one = load_catalog(tmp_path / "one/students.json")
    two = load_catalog(tmp_path / "two/students.json")
    assert one == two and len(transport.calls) == 1


def test_scope_update_overrides_manual_but_not_offline(tmp_path, monkeypatch):
    import schale.cache_cli as cli

    calls = []

    def fetch(dataset, **kwargs):
        calls.append(kwargs)
        from types import SimpleNamespace

        return SimpleNamespace(entries={})

    monkeypatch.setenv("SCHALE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cli.ReferenceCatalog, "fetch", fetch)
    result = CliRunner().invoke(app, ["cache", "update", "--scope", "data"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 6 and all(c["cache_mode"] == "auto" for c in calls)
