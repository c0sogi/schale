"""First-use behavior from an empty user cache, not a developer's warm cache."""

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from typer.testing import CliRunner

from schale.assets import install_resources, pack_resources, resolve_resources
from schale.cli import app
from schale.setup import export_setup, install_setup, run_with_vision_if_needed
from schale.students.model_bundle import validate_bundle
from schale.students.models import Extraction
from schale.students.output import select_output
from schale.students.paths import model_directory


@pytest.fixture
def private_setup(tmp_path, monkeypatch, student_references):
    source_cache = tmp_path / "prepared-cache"
    monkeypatch.setenv("SCHALE_CACHE_DIR", str(source_cache))
    source = tmp_path / "bundle-source"
    shutil.copytree(student_references, source / "students")
    refs = tmp_path / "references.zip"
    pack_resources(source, refs)
    install_resources(refs)
    model = model_directory()
    model.mkdir()
    data = b"unit test only, not an inference model"
    (model / "model.onnx").write_bytes(data)
    (model / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "bluearchive-ko-16x9-v1",
                "model_sha256": hashlib.sha256(data).hexdigest(),
                "charset": ["", "1"],
                "threshold": 0.5,
            }
        )
    )
    for name in ("NOTICE.md", "LICENSE-OpenOCR"):
        (model / name).write_text("Synthetic test fixture, no third-party model")
    portable = tmp_path / "schale-setup.zip"
    export_setup(portable)
    monkeypatch.setenv("SCHALE_CACHE_DIR", str(tmp_path / "empty-user-cache"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "new-user")
    monkeypatch.chdir(
        tmp_path / "new-user" if (tmp_path / "new-user").exists() else tmp_path
    )
    return portable


def test_single_extract_command_bootstraps_and_allocates_output(
    tmp_path, monkeypatch, private_setup
):
    import schale.students.extract as module

    downloads = Path.home() / "Downloads"
    downloads.mkdir(parents=True)
    shutil.move(str(private_setup), downloads / private_setup.name)
    source = tmp_path / "capture.png"
    source.write_bytes(b"input checked by stub; actual inference is tested separately")
    calls = []

    def extract(source, output, **kwargs):
        validate_bundle(model_directory())
        assert resolve_resources("students").is_dir()
        assert not output.exists()
        output.mkdir()
        (output / "review.html").write_text("test output")
        calls.append(output)
        return Extraction(
            source=str(source),
            catalog_sha256="test",
            catalog_url="test",
            diagnostics={"field_statuses": {}},
        )

    monkeypatch.setattr(module, "extract", extract)
    runner = CliRunner()
    for _ in range(2):
        result = runner.invoke(app, ["students", "extract", str(source)])
        assert result.exit_code == 0, result.output
    assert calls == [tmp_path / "capture-schale", tmp_path / "capture-schale-2"]
    assert validate_bundle(model_directory())["threshold"] == 0.5


def test_output_dot_preserves_home_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "keep.txt"
    marker.write_text("unchanged")
    source = tmp_path / "video.mp4"
    output = select_output([source], Path("."), False)
    assert output == tmp_path / "video-schale"
    assert marker.read_text() == "unchanged"
    assert not output.exists()
    with pytest.raises(ValueError, match="segments.json"):
        select_output([source], Path("."), True)


def test_missing_resources_reported_together_without_creating_output(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCHALE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "new-user")
    source = tmp_path / "capture.png"
    source.write_bytes(b"fixture")
    result = CliRunner().invoke(app, ["students", "extract", str(source)])
    assert result.exit_code == 1
    assert "student UI references" in result.output
    assert "numeric model" in result.output
    assert "schale-setup.zip" in result.output
    assert "MODEL.zip" not in result.output
    assert not (tmp_path / "capture-schale").exists()


def test_setup_validates_both_parts_before_install_and_repairs_broken_model(
    tmp_path, private_setup
):
    corrupt = tmp_path / "broken.zip"
    with ZipFile(private_setup) as source, ZipFile(corrupt, "w") as target:
        for name in source.namelist():
            target.writestr(
                name, b"broken" if name == "references.zip" else source.read(name)
            )
    with pytest.raises(ValueError, match="checksum"):
        install_setup(corrupt)
    assert not model_directory().exists()
    model_directory().mkdir()
    (model_directory() / "model.onnx").write_bytes(b"old broken file")
    install_setup(private_setup)
    validate_bundle(model_directory())
    backups = list((model_directory().parent / "setup-backups").rglob("model.onnx"))
    assert len(backups) == 1 and backups[0].read_bytes() == b"old broken file"
    install_setup(private_setup)
    assert (
        len(list((model_directory().parent / "setup-backups").rglob("model.onnx"))) == 1
    )


def test_install_model_without_argument_uses_prepared_file(private_setup):
    result = CliRunner().invoke(app, ["students", "install-model"])
    assert result.exit_code == 0, result.output
    validate_bundle(model_directory())


def test_missing_optional_dependencies_use_isolated_uv_not_parent_install(monkeypatch):
    import schale.setup as module

    monkeypatch.delenv("SCHALE_VISION_CHILD", raising=False)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _: None)
    monkeypatch.setattr(module.shutil, "which", lambda _: "uv")
    captured = []

    def run(command, **kwargs):
        captured.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    args = ["students", "extract", "한글 영상.mp4"]
    assert run_with_vision_if_needed(args) == 0
    assert captured[0][0][1:4] == ["tool", "run", "--from"]
    assert captured[0][0][-3:] == args
    assert captured[0][1]["env"]["SCHALE_VISION_CHILD"] == "1"
    monkeypatch.setenv("SCHALE_VISION_CHILD", "1")
    with pytest.raises(ValueError, match="runtime unavailable"):
        run_with_vision_if_needed(args)


def test_setup_does_not_include_account_or_http_cache(private_setup):
    with ZipFile(private_setup) as archive:
        assert set(archive.namelist()) == {
            "setup.json",
            "numeric.zip",
            "references.zip",
        }


def test_output_file_is_rejected(tmp_path):
    file = tmp_path / "keep.txt"
    file.write_text("keep")
    with pytest.raises(ValueError, match="Output is a file"):
        select_output([tmp_path / "video.mp4"], file, False)
    assert file.read_text() == "keep"
