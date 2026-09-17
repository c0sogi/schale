"""Public first-use contracts: no developer cache, game captures or manual ZIP."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from zipfile import ZipFile

import av
import cv2
import numpy as np
import pytest

from schale.students import download
from schale.students.geometry import BootstrapLayoutMatcher, GeometryMatcher
from schale.students.model_bundle import validate_bundle
from schale.students.video import prepare_video


@pytest.fixture
def public_model(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHALE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SCHALE_OFFLINE", "0")
    monkeypatch.delenv("SCHALE_SETUP_BUNDLE", raising=False)
    payload = b"synthetic model, never used for inference"
    archive = tmp_path / "public.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("model.onnx", payload)
        output.writestr(
            "metadata.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "profile": "bluearchive-ko-16x9-v1",
                    "model_sha256": hashlib.sha256(payload).hexdigest(),
                    "charset": ["", "1"],
                    "threshold": 0.5,
                }
            ),
        )
        output.writestr("LICENSE-OpenOCR", "synthetic fixture")
        output.writestr("NOTICE.md", "synthetic fixture")
    data = archive.read_bytes()
    monkeypatch.setattr(download, "MODEL_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(download, "MODEL_BYTES", len(data))
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield data[:100]
            yield data[100:]

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(download.requests, "get", get)
    return calls


def test_automatic_public_model_download_is_verified_and_shared(public_model):
    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(lambda _: download.ensure_numeric_model(), range(3)))
    assert paths[0] == paths[1] == paths[2]
    assert len(public_model) == 1
    assert public_model[0][0].startswith("https://github.com/")
    assert validate_bundle(paths[0])["threshold"] == 0.5
    assert not (paths[0].parent / "vision-resources").exists()


def test_download_corruption_never_replaces_existing_data(public_model, monkeypatch):
    path = download.model_directory()
    path.mkdir(parents=True)
    marker = path / "original.txt"
    marker.write_text("preserve")
    monkeypatch.setattr(download, "MODEL_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        download.ensure_numeric_model()
    assert marker.read_text() == "preserve"
    assert not (path / "model.onnx").exists()


def test_offline_cache_hit_has_no_network(public_model, monkeypatch):
    path = download.ensure_numeric_model()
    monkeypatch.setenv("SCHALE_OFFLINE", "1")
    assert download.ensure_numeric_model() == path
    assert len(public_model) == 1


def profile_image():
    image = np.full((720, 1280, 3), 50, dtype=np.uint8)
    for y in (178, 302, 309, 433, 440, 548, 651):
        cv2.line(image, (659, y), (1215, y), (240, 240, 240), 1)
    for i in range(3):
        polygon = np.array(
            [
                [433 + i * 66, 647],
                [488 + i * 66, 647],
                [479 + i * 66, 700],
                [424 + i * 66, 700],
            ]
        )
        cv2.fillPoly(image, [polygon], (240, 240, 240))
    return image


@pytest.mark.parametrize("scale,tx,ty", [(1.0, 0, 0), (0.85, 120, 20), (0.95, 20, 0)])
def test_geometric_profile_without_any_resource_files(scale, tx, ty):
    matrix = np.array([[scale, 0, tx], [0, scale, ty]], dtype=float)
    image = cv2.warpAffine(profile_image(), matrix, (1280, 720))
    layout = GeometryMatcher().locate(image)
    assert layout is not None
    for side, points in [
        ("left", [[40, 580, 1], [350, 615, 1]]),
        ("right", [[680, 230, 1], [1080, 640, 1]]),
    ]:
        error = np.array(points) @ (layout.matrices[side] - matrix).T
        assert np.linalg.norm(error, axis=1).max() < 4


def test_geometry_rejects_unrelated_or_incomplete_screen():
    assert GeometryMatcher().locate(np.zeros((720, 1280, 3), np.uint8)) is None
    image = profile_image()
    image[640:, :850] = 50
    assert GeometryMatcher().locate(image) is None


def test_bootstrap_builds_features_from_input_not_installed_resources(monkeypatch):
    image = profile_image()
    rng = np.random.default_rng(42)
    for x, y in [
        (434, 677),
        (500, 677),
        (566, 677),
        (710, 202),
        (1130, 360),
        (1130, 495),
        (1130, 610),
    ]:
        image[y - 28 : y, x : x + 35] = rng.integers(
            0, 220, (28, 35, 3), dtype=np.uint8
        )
    matcher = BootstrapLayoutMatcher()
    first = matcher.locate(image)
    assert first is not None and matcher.matcher is not None
    monkeypatch.setattr(
        matcher.geometry, "locate", lambda _: pytest.fail("must reuse input features")
    )
    second = matcher.locate(image)
    assert second is not None
    assert all(
        info["method"] == "input-derived-sift" for info in second.diagnostics.values()
    )


def test_bond_cell_retains_digit_and_excludes_dark_heart_surround():
    from schale.students import numeric

    patch = np.full((60, 70, 3), (90, 60, 40), np.uint8)
    cv2.circle(patch, (35, 30), 25, (190, 180, 255), -1)
    cv2.putText(patch, "7", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    digit_pixels = int(np.count_nonzero(np.all(patch == (50, 50, 50), axis=2)))
    cell = numeric.bond_cell(patch)
    assert np.count_nonzero(cell[:, :, 0] == 0) == digit_pixels
    assert np.all(cell[:5] == 255) and np.all(cell[-5:] == 255)
    assert cell.shape[0] < patch.shape[0]


def test_video_decodes_without_ffmpeg_or_ffprobe(tmp_path, monkeypatch):
    import schale.students.video as module

    path = tmp_path / "recording.mkv"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=30)
        stream.width, stream.height = 640, 360
        stream.pix_fmt = "bgr0"
        for index in range(60):
            image = np.full((360, 640, 3), 40 if index < 30 else 180, np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    result = prepare_video(path, tmp_path / "result")
    assert result["decoder"][0] == "pyav"
    assert len(result["segments"]) == 2
    assert result["alignment_max_mae"] == 0
    assert result["frame_count"] >= 4
