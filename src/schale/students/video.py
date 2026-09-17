"""Two-pass sequential decoding; sampled frame indices match in both passes."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def crop(image, box):
    h, w = image.shape[:2]
    x, y, x1, y1 = box
    return image[round(y * h) : round(y1 * h), round(x * w) : round(x1 * w)]


def signature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # The name band and opaque panel avoid Live2D and most animated backgrounds.
    return np.concatenate(
        [
            cv2.resize(
                crop(gray, (0.025, 0.773, 0.27, 0.858)),
                (100, 24),
                interpolation=cv2.INTER_AREA,
            ).ravel(),
            cv2.resize(
                crop(gray, (0.535, 0.31, 0.852, 0.895)),
                (80, 100),
                interpolation=cv2.INTER_AREA,
            ).ravel(),
        ]
    ).astype(np.float32)


def probe(path: Path):
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} must be installed on PATH")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(
            f"Cannot inspect video: {result.stderr.decode(errors='replace')[-1000:]}"
        )
    data = json.loads(result.stdout)
    stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if stream is None:
        raise ValueError("Input contains no video stream")
    return {
        "width": stream["width"],
        "height": stream["height"],
        "codec": stream["codec_name"],
        "duration": float(data["format"]["duration"]),
    }


def choose_decoder(path: Path, codec: str):
    options = (
        [["-c:v", "av1_cuvid"], ["-c:v", "libdav1d"], []] if codec == "av1" else [[]]
    )
    for decoder in options:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "fatal",
                *decoder,
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-vf",
                "scale=64:36",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-",
            ],
            capture_output=True,
        )
        if result.returncode == 0 and len(result.stdout) == 64 * 36 * 3:
            return decoder
    raise RuntimeError("No available decoder could read the first frame")


def frames(command, width, height):
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        assert process.stdout is not None
        try:
            while True:
                raw = process.stdout.read(width * height * 3)
                if not raw:
                    break
                if len(raw) != width * height * 3:
                    raise RuntimeError("Truncated decoded frame")
                yield np.frombuffer(raw, np.uint8).reshape(height, width, 3)
            if process.wait() != 0:
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors="replace")[-1200:])
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
            process.wait()


def selection_plan(signatures, fps: int = 30):
    """Return the exact temporal decisions, independently of image export."""
    if len(signatures) < 2:
        raise ValueError("Recording is too short")
    motion = np.r_[0, np.mean(np.abs(np.diff(signatures, axis=0)), axis=1)]
    median = float(np.median(motion))
    mad = float(np.median(np.abs(motion - median)))
    threshold = max(1.5, min(5.0, median + 8 * max(mad, 0.05)))
    changed = np.flatnonzero(motion > threshold)
    groups = []
    for i in changed:
        if groups and i == groups[-1][-1] + 1:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    cuts = sorted(set([0] + [g[-1] for g in groups] + [len(signatures)]))
    segments, discarded = [], []
    for left, right in zip(cuts[:-1], cuts[1:]):
        valid = [
            i
            for i in range(left + 1, right - 1)
            if motion[i] < threshold * 0.35 and motion[i + 1] < threshold * 0.35
        ]
        if not valid:
            discarded.append(
                {
                    "start": left / fps,
                    "end": right / fps,
                    "reason": "no stable interior samples",
                }
            )
            continue
        chosen = sorted(
            set(valid[round(q * (len(valid) - 1))] for q in (0.2, 0.5, 0.8))
        )
        key = f"segment_{len(segments) + 1:03}"
        segments.append(
            {
                "key": key,
                "start": left / fps,
                "end": right / fps,
                "frames": [
                    {
                        "image": f"frames/{key}_{i:06}.png",
                        "timestamp": i / fps,
                        "sample_index": i,
                    }
                    for i in chosen
                ],
            }
        )
    return segments, discarded, threshold, motion


def prepare_video(video: Path, output: Path, fps: int = 30):
    if not all(shutil.which(name) for name in ("ffmpeg", "ffprobe")):
        from .video_native import prepare_native

        return prepare_native(video, output, fps)
    info = probe(video)
    if abs(info["width"] / info["height"] - 16 / 9) > 0.03:
        raise ValueError(
            "This profile supports 16:9 game recordings; crop game viewport before extraction"
        )
    decoder = choose_decoder(video, info["codec"])
    prefix = [
        "ffmpeg",
        "-v",
        "fatal",
        *decoder,
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-an",
    ]
    small = prefix + [
        "-vf",
        f"fps={fps},scale=640:360",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    signatures = np.stack([signature(f) for f in frames(small, 640, 360)])
    segments, discarded, threshold, _ = selection_plan(signatures, fps)
    target_map = {f["sample_index"]: f["image"] for s in segments for f in s["frames"]}
    if not segments:
        raise ValueError("No stable screen intervals found")
    (output / "frames").mkdir(parents=True, exist_ok=True)
    # select uses the very same FPS filter grid as analysis. Sequential decoding
    # avoids assumptions about keyframe seeking and VFR frame-number arithmetic.
    select = "+".join(f"eq(n\\,{i})" for i in sorted(target_map))
    command = prefix + [
        "-vf",
        f"fps={fps},select='{select}'",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    count = 0
    alignment_errors = []
    for i, frame in zip(
        sorted(target_map), frames(command, info["width"], info["height"]), strict=True
    ):
        difference = float(
            np.mean(np.abs(signature(cv2.resize(frame, (640, 360))) - signatures[i]))
        )
        alignment_errors.append(difference)
        # FFmpeg and OpenCV use different scaling kernels. The motion threshold
        # measures temporal differences within one decoder, not cross-kernel error.
        if difference > 10:
            raise RuntimeError(
                f"Analysis/extraction frame mismatch at sample {i}: {difference:.2f}"
            )
        ok, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        if not ok:
            raise RuntimeError("PNG encoding failed")
        (output / target_map[i]).write_bytes(encoded.tobytes())
        count += 1
    plan = {
        "source": str(video.resolve()),
        "video": info,
        "fps": fps,
        "decoder": decoder,
        "segments": segments,
        "discarded": discarded,
        "threshold": threshold,
        "frame_count": count,
        "alignment_max_mae": max(alignment_errors),
    }
    (output / "segments.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan
