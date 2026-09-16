"""Public toolkit contracts: reusable without vision dependencies or a network."""

import base64
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from schale import (
    AccountSnapshot,
    GrowthValue,
    ReferenceCatalog,
    SourceReference,
    StudentState,
)
from schale.account import GROWTH_BOUNDS
from schale.adapters.extraction import from_extraction
from schale.adapters.schaledb import KEYS, decode_collection, from_schaledb, to_schaledb
from schale.cli import app
from schale.collections import compare, merge
from schale.students.models import Extraction, Reading, Student


def encode(value):
    return base64.b64encode(json.dumps(value).encode()).decode()


def site_row():
    return {
        **{key: GROWTH_BOUNDS[name][0] for name, key in KEYS.items()},
        "lock": True,
        "future": {"preserve": 2},
    }


def account(level: int | None = 90, status="confirmed", sid=10001):
    return AccountSnapshot.model_validate(
        {
            "students": [
                {
                    "key": str(sid),
                    "student_id": sid,
                    "identity_status": "confirmed",
                    "fields": {"level": {"value": level, "status": status}},
                }
            ]
        }
    )


def test_toolkit_import_has_no_vision_or_network_side_effects():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, requests; "
            "requests.get=lambda *a,**k: (_ for _ in ()).throw(AssertionError('network')); "
            "import schale.cli, schale.collections, schale.adapters.schaledb; "
            "assert not {'cv2','numpy','onnxruntime','torch','easyocr'} & sys.modules.keys()",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "value,status",
    [(True, "confirmed"), (None, "confirmed"), (90, "unknown"), (90, "conflict")],
)
def test_incoherent_or_boolean_values_rejected(value, status):
    with pytest.raises(ValueError):
        GrowthValue.model_validate({"value": value, "status": status})


@pytest.mark.parametrize(
    "field,value", [("level", 0), ("star", 6), ("unknown_field", 1)]
)
def test_field_contracts(field, value):
    with pytest.raises(ValueError):
        StudentState(
            key="s", fields={field: GrowthValue(value=value, status="confirmed")}
        )


def test_duplicate_and_unidentified_known_identity_rejected():
    student = account().students[0]
    with pytest.raises(ValueError, match="Duplicate"):
        AccountSnapshot(students=[student, student.model_copy()])
    with pytest.raises(ValueError, match="SchaleDB ID"):
        StudentState(key="unknown", identity_status="confirmed")


def test_site_roundtrip_preserves_future_attributes_and_reports_locked_conversion():
    row = site_row()
    snapshot = from_schaledb(encode({"10001": row}))
    assert snapshot.students[0].identity_status == "imported"
    assert "equipment1_level" not in snapshot.students[0].fields
    exported, report = to_schaledb(snapshot)
    expected = {**row, "s2": 1, "s3": 1, "s4": 1}
    assert decode_collection(exported) == {"10001": expected}
    assert report["updated"] == [10001]
    assert snapshot.students[0].fields["sub"].value == 0


def test_partial_import_is_not_filled_and_cannot_be_exported_without_baseline():
    snapshot = from_schaledb(encode({"10001": {"l": 90}}))
    assert set(snapshot.students[0].fields) == {"level"}
    with pytest.raises(ValueError, match="No complete"):
        to_schaledb(snapshot)
    baseline = {"10001": site_row(), "10002": site_row()}
    exported, _ = to_schaledb(snapshot, baseline)
    assert decode_collection(exported)["10001"]["l"] == 90
    assert decode_collection(exported)["10002"] == baseline["10002"]
    assert baseline["10001"]["l"] == 1


@pytest.mark.parametrize(
    "payload",
    [{"10001": {"s": 6}}, {"01": {}, "1": {}}, {"0": {}}, {"10001": {"l": True}}],
)
def test_bad_site_data_is_not_imported(payload):
    with pytest.raises(ValueError):
        from_schaledb(encode(payload))


def test_visual_adapter_retains_all_fields_and_unknowns():
    result = Extraction(
        source="video.mp4",
        catalog_url="catalog",
        catalog_sha256="digest",
        students=[
            Student(
                key="s",
                student_id=10001,
                identity_status="confirmed",
                fields={
                    name: Reading(value=lo, status="confirmed")
                    for name, (lo, _) in GROWTH_BOUNDS.items()
                },
            ),
            Student(key="unknown", fields={"level": Reading()}),
        ],
    )
    converted = from_extraction(result)
    assert len(converted.students[0].fields) == 19
    assert converted.students[1].student_id is None
    assert converted.students[1].fields["level"].value is None
    assert converted.sources[0].details["sha256"] == "digest"


def test_merge_preserves_unseen_uncertain_and_input_objects():
    base = account()
    base.students.append(account(sid=10002).students[0])
    incoming = account(None, "unknown")
    saved = base.model_dump()
    merged, report = merge(base, incoming)
    assert merged.model_dump()["students"] == saved["students"]
    assert base.model_dump() == saved
    assert len(report["held"]) == 1 and not report["applied"]
    assert incoming.students[0].fields["level"].value is None


def test_merge_observed_requires_explicit_opt_in():
    original = account()
    incoming = account(91, "observed")
    assert merge(original, incoming)[0].students[0].fields["level"].value == 90
    assert (
        merge(original, incoming, allow_observed=True)[0]
        .students[0]
        .fields["level"]
        .value
        == 91
    )


def test_merge_accepted_update_and_extension_preferences():
    base, incoming = account(), account(91, "imported")
    base.students[0].extensions = {"schaledb": {"lock": True}}
    incoming.students[0].extensions = {"schaledb": {"lock": False, "future": 3}}
    result, report = merge(base, incoming)
    assert result.students[0].fields["level"].value == 91
    assert result.students[0].extensions["schaledb"] == {"lock": True, "future": 3}
    assert report["applied"] == [
        {"student_id": 10001, "field": "level", "before": 90, "after": 91}
    ]
    assert base.students[0].fields["level"].value == 90


def test_merge_new_student_provenance_not_duplicated_and_unresolved_held():
    incoming = account()
    incoming.students[0].sources = [SourceReference(kind="test", uri="source")]
    incoming.students.append(StudentState(key="unknown"))
    result, report = merge(AccountSnapshot(), incoming)
    assert len(result.students) == len(result.students[0].sources) == 1
    assert report["held"] == [{"key": "unknown", "reason": "identity not accepted"}]


def test_compare_includes_status_and_absence_not_deletion():
    before, after = account(), account(90, "observed")
    before.students.append(account(sid=10002).students[0])
    report = compare(before, after)
    assert report["missing_ids"] == [10002]
    assert "deleted_ids" not in report
    assert report["changes"][0]["after"] == {"value": 90, "status": "observed"}


def test_reference_content_hash_stable_across_fetch_and_file(tmp_path, monkeypatch):
    import schale.reference as module

    calls = []
    payload = {"10001": {"Name": "에이미", "Id": 10001}}

    def get_json(url, **kwargs):
        calls.append((url, kwargs))
        return payload

    monkeypatch.setattr(module, "get_json", get_json)
    catalog = ReferenceCatalog.fetch("students", refresh=True)
    path = tmp_path / "students.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert ReferenceCatalog.from_file(path).sha256 == catalog.sha256
    assert calls == [
        ("https://schaledb.com/data/kr/students.min.json", {"force_refresh": True})
    ]
    catalog.get(10001)["Name"] = "changed"
    assert catalog.get(10001)["Name"] == "에이미"
    assert (
        from_schaledb(encode({"10001": {"l": 90}}), catalog=catalog).students[0].name
        == "에이미"
    )
    with pytest.raises(KeyError):
        catalog.get(99999)


def test_cli_end_to_end_and_no_overwrite(tmp_path):
    runner = CliRunner()
    imported, incoming, merged, delta, exported = [
        tmp_path / name
        for name in (
            "base.json",
            "incoming.json",
            "merged.json",
            "diff.json",
            "export.txt",
        )
    ]
    site = tmp_path / "site.txt"
    site.write_text(encode({"10001": site_row(), "10002": site_row()}))
    incoming.write_text(account(91).model_dump_json())
    commands = [
        ["import", str(site), "-o", str(imported)],
        ["merge", str(imported), str(incoming), "-o", str(merged)],
        ["diff", str(imported), str(merged), "-o", str(delta)],
        ["export", str(merged), "-o", str(exported)],
    ]
    for command in commands:
        result = runner.invoke(app, ["collection", *command])
        assert result.exit_code == 0, result.output
    wire = decode_collection(exported.read_text())
    assert len(wire) == 2 and wire["10001"]["l"] == 91 and wire["10001"]["lock"] is True
    assert json.loads(delta.read_text())["changes"][0]["field"] == "level"
    saved = exported.read_bytes()
    result = runner.invoke(app, ["collection", *commands[-1]])
    assert result.exit_code == 1 and "already exists" in result.output
    assert exported.read_bytes() == saved


def test_cli_refuses_existing_report_before_creating_output(tmp_path):
    source, target = tmp_path / "source.json", tmp_path / "target.txt"
    source.write_text(account().model_dump_json())
    target.with_suffix(".report.json").write_text("keep")
    result = CliRunner().invoke(
        app, ["collection", "export", str(source), "-o", str(target)]
    )
    assert result.exit_code == 1 and not target.exists()


def test_cli_data_and_rewards_use_actual_public_services(tmp_path, monkeypatch):
    import schale.reference as reference
    import schale.toolkit_cli as cli
    from schale import RewardExpectation

    monkeypatch.setattr(reference, "get_json", lambda *a, **k: {"1": {"Id": 1}})
    data = tmp_path / "students.json"
    result = CliRunner().invoke(app, ["data", "fetch", "students", "-o", str(data)])
    assert result.exit_code == 0, result.output
    assert json.loads(data.read_text()) == {"1": {"Id": 1}}
    assert data.with_suffix(".meta.json").exists()
    calls = []

    def calculate(stage_ids, **kwargs):
        calls.append((stage_ids, kwargs))
        return [
            SimpleNamespace(
                stage=SimpleNamespace(root=SimpleNamespace(EntryCost=[[5, 10]])),
                rewards=[
                    RewardExpectation(
                        type="Item",
                        id=1,
                        expected_amount=0.5,
                        chance=0.5,
                        reward_type=None,
                    )
                ],
            )
        ]

    monkeypatch.setattr(cli.StageRewards, "from_stages", calculate)
    output = tmp_path / "rewards.json"
    result = CliRunner().invoke(
        app, ["rewards", "calculate", "--stage", "1011101", "-o", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert calls == [([1011101], {"server": "Global", "reward_conditions": []})]
    assert (
        json.loads(output.read_text())["stages"][0]["rewards"][0]["expected_amount"]
        == 0.5
    )


def test_cli_data_failure_is_actionable(tmp_path, monkeypatch):
    import schale.reference as reference

    def fail(*args, **kwargs):
        raise RuntimeError("Cannot load cache for test")

    monkeypatch.setattr(reference, "get_json", fail)
    output = tmp_path / "students.json"
    result = CliRunner().invoke(app, ["data", "fetch", "students", "-o", str(output)])
    assert result.exit_code == 1 and "Cannot load cache" in result.output
    assert not output.exists()
