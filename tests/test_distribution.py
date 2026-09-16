"""Installation/cache contracts independent of downloaded game or model data."""

import hashlib
import json
import stat
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest
from typer.testing import CliRunner

from schale.cli import app
from schale.students.model_bundle import PROFILE, install_bundle, validate_bundle
from schale.students.paths import asset_directory, cache_directory, model_directory


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"test fixture, not an inference model")
    metadata = {
        "schema_version": 1,
        "profile": PROFILE,
        "charset": ["", "1"],
        "threshold": 0.436,
        "model_sha256": hashlib.sha256(
            (source / "model.onnx").read_bytes()
        ).hexdigest(),
    }
    (source / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (source / "LICENSE-OpenOCR").write_text("license fixture", encoding="utf-8")
    (source / "NOTICE.md").write_text("notice fixture", encoding="utf-8")
    return source


def test_zip_install_retains_provenance(bundle, tmp_path):
    archive = tmp_path / "model.zip"
    with ZipFile(archive, "w") as z:
        for path in bundle.iterdir():
            z.write(path, path.name)
    destination = tmp_path / "cache" / "model"
    install_bundle(archive, destination)
    assert validate_bundle(destination) == validate_bundle(bundle)
    assert (destination / "NOTICE.md").read_text() == "notice fixture"
    assert (destination / "LICENSE-OpenOCR").read_text() == "license fixture"
    assert not list(destination.parent.glob(".student-model-*"))


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "nested/model.onnx", "unexpected.py"]
)
def test_zip_paths_are_rejected(tmp_path, name):
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as z:
        z.writestr(name, b"bad")
    with pytest.raises(ValueError, match="flat bundle"):
        install_bundle(archive, tmp_path / "cache" / "model")
    assert not (tmp_path / "cache" / "model").exists()


def test_zip_symlink_is_rejected(tmp_path):
    archive = tmp_path / "unsafe.zip"
    link = ZipInfo("model.onnx")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(archive, "w") as z:
        z.writestr(link, "../../outside")
    with pytest.raises(ValueError, match="flat bundle"):
        install_bundle(archive, tmp_path / "cache" / "model")


def test_corrupt_install_leaves_no_partial_model(bundle, tmp_path):
    (bundle / "model.onnx").write_bytes(b"corrupt")
    destination = tmp_path / "cache" / "model"
    with pytest.raises(ValueError, match="checksum"):
        install_bundle(bundle, destination)
    assert not destination.exists()
    assert not list(destination.parent.iterdir())


def test_populated_target_untouched(bundle, tmp_path):
    target = tmp_path / "installed"
    target.mkdir()
    (target / "important.txt").write_text("keep")
    with pytest.raises(ValueError, match="not empty"):
        install_bundle(bundle, target)
    assert (target / "important.txt").read_text() == "keep"


@pytest.mark.parametrize("threshold", [True, float("nan"), -1, 2])
def test_bad_threshold_is_not_accepted(bundle, threshold):
    path = bundle / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["threshold"] = threshold
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="threshold"):
        validate_bundle(bundle)


def test_cache_override_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHALE_CACHE_DIR", str(tmp_path / "shared"))
    for folder in ("one", "two"):
        directory = tmp_path / folder
        directory.mkdir()
        monkeypatch.chdir(directory)
        assert cache_directory() == tmp_path / "shared"
        assert model_directory() == tmp_path / "shared" / "student-numeric"
        assert asset_directory() == tmp_path / "shared" / "assets"
    assert model_directory(tmp_path / "explicit") == tmp_path / "explicit"
    assert not (tmp_path / "shared").exists()


def test_working_directory_does_not_change_default_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("SCHALE_CACHE_DIR", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "user"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".schale/student-numeric").mkdir(parents=True)
    (tmp_path / ".schale/assets").mkdir()
    assert model_directory() == tmp_path / "user/.schale/cache/student-numeric"
    assert asset_directory() == tmp_path / "user/.schale/cache/assets"
    monkeypatch.setenv("SCHALE_CACHE_DIR", str(tmp_path / "shared"))
    assert model_directory() == tmp_path / "shared/student-numeric"
    assert asset_directory() == tmp_path / "shared/assets"


def test_missing_model_does_not_download_or_create_output(tmp_path, monkeypatch):
    import schale.students.extract as module

    monkeypatch.setenv("SCHALE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        module, "load_catalog", lambda *args: pytest.fail("should not download")
    )
    source = tmp_path / "video.mp4"
    source.write_bytes(b"no decoding needed")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="CTC model not installed"):
        module.extract(source, output)
    assert not output.exists()


def test_install_cli_uses_shared_cache(bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("SCHALE_CACHE_DIR", str(tmp_path / "shared"))
    result = CliRunner().invoke(app, ["students", "install-model", str(bundle)])
    assert result.exit_code == 0, result.output
    assert validate_bundle(tmp_path / "shared/student-numeric") == validate_bundle(
        bundle
    )


def test_cli_version_matches_metadata():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert version("schale") in result.output


def test_doctor_reports_failure_instead_of_success(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHALE_CACHE_DIR", str(tmp_path / "empty-cache"))
    result = CliRunner().invoke(app, ["students", "doctor"])
    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["ready"] is False and report["model"]["valid"] is False
    assert not (tmp_path / "empty-cache").exists()


def test_model_installer_import_is_lightweight():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import schale.students.model_bundle; "
            "assert not {'cv2','numpy','onnxruntime','torch','easyocr'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_current_and_legacy_stage_armor_values():
    from schale.schema.stages import BaseStage

    row = {
        "Id": 1,
        "Category": "Campaign",
        "EntryCost": [],
        "StarCondition": [0, 0],
        "Rewards": [],
        "Terrain": "Street",
        "Level": 1,
        "Formations": [],
    }
    for armor in ([1, 2], ["LightArmor", "ElasticArmor"]):
        parsed = BaseStage.model_validate({**row, "ArmorTypes": armor})
        assert parsed.ArmorTypes == armor
    with pytest.raises(ValueError):
        BaseStage.model_validate({**row, "ArmorTypes": ["not-an-armor-type"]})


def test_immediate_exchange_consumable_does_not_invent_consume_type():
    from schale.schema.item import ConsumableItem

    row = {
        "Id": 150049,
        "IsReleased": [True, True, False],
        "Category": "Consumable",
        "Rarity": "SSR",
        "Quality": 4,
        "Tags": [],
        "Craftable": [False] * 3,
        "StageDrop": [False] * 3,
        "Shop": [False] * 3,
        "Icon": "ticket",
        "Name": "Ticket",
        "Desc": "",
    }
    assert ConsumableItem.model_validate(row).ConsumeType is None
    assert (
        ConsumableItem.model_validate({**row, "ConsumeType": "Choice"}).ConsumeType
        == "Choice"
    )


@pytest.mark.parametrize(
    "override,expected",
    [
        ({"Formations": []}, [5]),
        ({"Rewards": []}, []),
        ({"Rewards": [{"Type": "Item", "Id": 7, "Amount": 1}]}, [7]),
    ],
)
def test_formation_only_override_inherits_rewards_but_empty_override_does_not(
    monkeypatch, override, expected
):
    from schale import StageRewards
    import schale.stage_rewards as module
    from schale.schema.stages import Stage

    stage = Stage.model_validate(
        {
            "Id": 1,
            "Category": "Campaign",
            "EntryCost": [],
            "StarCondition": [0, 0],
            "Rewards": [{"Type": "Item", "Id": 5, "Amount": 1}],
            "Terrain": "Street",
            "Level": 1,
            "ArmorTypes": ["LightArmor"],
            "Formations": [],
            "Name": "Test",
            "Area": 1,
            "Stage": 1,
            "Difficulty": 0,
            "ChallengeCondition": [],
            "ServerData": {"Cn": override},
        }
    )

    class Cache:
        stages = {1: stage}

    monkeypatch.setattr(module, "cache_collection", Cache())
    result = StageRewards.from_stages([1], server="Cn")
    assert [r.id for r in result[0].rewards] == expected
