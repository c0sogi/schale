"""Resource lifecycle and missing-resource behavior, using generated fixtures only."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile, ZipInfo
import stat

import pytest
from typer.testing import CliRunner

from schale.assets import (
    install_resources,
    pack_resources,
    resolve_resources,
    resource_directory,
)
from schale.cli import app


@pytest.fixture
def resource_source(tmp_path, student_references):
    source = tmp_path / "source"
    shutil.copytree(student_references, source / "students")
    models = source / "inventory/models"
    models.mkdir(parents=True)
    (models / "equipment_classifier.onnx").write_bytes(b"fixture, not for inference")
    (models / "class_mapping.json").write_text("{}", encoding="utf-8")
    return source


def test_resource_install_is_verified_idempotent_and_keeps_previous(
    tmp_path, monkeypatch, resource_source
):
    monkeypatch.setenv("SCHALE_CACHE_DIR", str(tmp_path / "cache"))
    first = tmp_path / "one.zip"
    pack_resources(resource_source, first)
    installed = install_resources(first)
    assert install_resources(first) == installed
    assert resolve_resources("students") == installed / "students"
    (resource_source / "inventory/models/class_mapping.json").write_text(
        '{"version":2}'
    )
    second = tmp_path / "two.zip"
    pack_resources(resource_source, second)
    updated = install_resources(second)
    assert installed != updated and installed.is_dir()
    assert resolve_resources("inventory") == updated / "inventory"
    (updated / "students/header.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        resolve_resources("students")
    install_resources(first)
    assert resolve_resources("students") == installed / "students"


@pytest.mark.parametrize(
    "name",
    [
        "../escaped.png",
        "/absolute.png",
        "students/../../escape",
        "C:/escape",
        "students\\escape",
        "students/CON.png",
    ],
)
def test_resource_zip_cannot_escape(tmp_path, monkeypatch, name):
    monkeypatch.setenv("SCHALE_CACHE_DIR", str(tmp_path / "cache"))
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as bundle:
        info = ZipInfo("placeholder")
        info.filename = name
        bundle.writestr(info, b"bad")
    with pytest.raises(ValueError, match="Unsafe"):
        install_resources(archive)
    assert not (resource_directory() / "current.json").exists()


def test_symlink_and_case_collision_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHALE_CACHE_DIR", str(tmp_path / "cache"))
    for mode in ("link", "collision"):
        archive = tmp_path / f"{mode}.zip"
        with ZipFile(archive, "w") as bundle:
            if mode == "link":
                info = ZipInfo("students/link.png")
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                bundle.writestr(info, "../target")
            else:
                bundle.writestr("students/X.png", b"a")
                bundle.writestr("students/x.png", b"b")
        with pytest.raises(ValueError, match="Unsafe"):
            install_resources(archive)


def test_failed_update_preserves_active_resources(
    tmp_path, monkeypatch, resource_source
):
    monkeypatch.setenv("SCHALE_CACHE_DIR", str(tmp_path / "cache"))
    archive = tmp_path / "valid.zip"
    pack_resources(resource_source, archive)
    installed = install_resources(archive)
    broken = tmp_path / "broken.zip"
    with ZipFile(archive) as source, ZipFile(broken, "w") as target:
        for name in source.namelist():
            target.writestr(
                name, b"bad" if name == "students/header.png" else source.read(name)
            )
    with pytest.raises(ValueError, match="checksum"):
        install_resources(broken)
    assert resolve_resources("students") == installed / "students"


def test_pack_refuses_missing_references_and_overwrite(tmp_path, resource_source):
    archive = tmp_path / "valid.zip"
    pack_resources(resource_source, archive)
    with pytest.raises(FileExistsError):
        pack_resources(resource_source, archive)
    (resource_source / "students/header.png").unlink()
    with pytest.raises(ValueError, match="Missing"):
        pack_resources(resource_source, tmp_path / "missing.zip")


def test_no_resources_no_network_no_output(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHALE_CACHE_DIR", str(tmp_path / "empty"))
    from schale.students.extract import extract
    import cv2
    import numpy as np
    import requests

    monkeypatch.setattr(
        requests.Session, "get", lambda *a, **k: pytest.fail("Unexpected HTTP request")
    )
    source = tmp_path / "image.png"
    assert cv2.imwrite(str(source), np.zeros((720, 1280, 3), np.uint8))
    with pytest.raises(ValueError, match="schale assets install"):
        extract(source, tmp_path / "result", reader="template")
    assert not (tmp_path / "result").exists()
    result = CliRunner().invoke(app, ["students", "doctor"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["references"]["valid"] is False


def test_cli_pack_install_and_lightweight_import(
    tmp_path, monkeypatch, resource_source
):
    monkeypatch.setenv("SCHALE_CACHE_DIR", str(tmp_path / "cache"))
    runner = CliRunner()
    bundle = tmp_path / "cli.zip"
    for args in (
        ["assets", "pack", str(resource_source), "-o", str(bundle)],
        ["assets", "install", str(bundle)],
        ["assets", "info"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import schale.assets_cli; assert not {'cv2','numpy','torch','onnxruntime'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_source_has_no_bundled_game_images_or_models():
    source = Path(__file__).resolve().parents[1] / "src/schale"
    assert not [
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.suffix in {".png", ".webp", ".npz", ".onnx", ".data", ".pt", ".pth"}
    ]
