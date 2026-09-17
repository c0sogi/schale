"""Two-pass timestamp sampling using the packaged CPU decoder, without PATH tools."""

import json
from pathlib import Path

import av
import cv2
import numpy as np


def probe_native(path: Path) -> dict:
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("Input contains no video stream")
        stream = container.streams.video[0]
        return {
            "width": stream.width,
            "height": stream.height,
            "codec": stream.codec_context.name,
            "duration": float(container.duration / av.time_base)
            if container.duration
            else None,
        }


def sampled_frames(path: Path, fps: int):
    """Select the displayed frame for each grid point using presentation time.

    Both passes use this generator. No average-FPS frame-number conversion or
    approximate keyframe seeking is involved, including for VFR recordings.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        previous = None
        origin = None
        index = 0
        for frame in container.decode(stream):
            if frame.time is None:
                raise ValueError("Video frame has no presentation timestamp")
            if origin is None:
                origin = frame.time
            timestamp = frame.time - origin
            if previous is not None:
                while index / fps < timestamp - 1e-8:
                    yield index, previous
                    index += 1
            previous = frame
        if previous is not None:
            end = float(previous.time or 0) - (origin or 0)
            duration = (
                float(previous.duration * previous.time_base)
                if previous.time_base
                else 0
            )
            while index / fps < end + max(duration, 1 / fps) - 1e-8:
                yield index, previous
                index += 1


def prepare_native(video: Path, output: Path, fps: int = 30):
    from .video import selection_plan, signature

    info = probe_native(video)
    if abs(info["width"] / info["height"] - 16 / 9) > 0.03:
        raise ValueError(
            "This profile supports 16:9 game recordings; crop game viewport before extraction"
        )
    samples = [
        signature(cv2.resize(frame.to_ndarray(format="bgr24"), (640, 360)))
        for _, frame in sampled_frames(video, fps)
    ]
    if len(samples) < 2:
        raise ValueError("Recording is too short")
    signatures = np.stack(samples)
    segments, discarded, threshold, _ = selection_plan(signatures, fps)
    if not segments:
        raise ValueError("No stable screen intervals found")
    targets = {f["sample_index"]: f["image"] for s in segments for f in s["frames"]}
    (output / "frames").mkdir(parents=True, exist_ok=True)
    errors = []
    for index, frame in sampled_frames(video, fps):
        if index not in targets:
            continue
        image = frame.to_ndarray(format="bgr24")
        difference = float(
            np.mean(
                np.abs(signature(cv2.resize(image, (640, 360))) - signatures[index])
            )
        )
        if difference > 1e-6:
            raise RuntimeError(
                f"Analysis/extraction frame mismatch at sample {index}: {difference:.2f}"
            )
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        if not ok:
            raise RuntimeError("PNG encoding failed")
        (output / targets[index]).write_bytes(encoded.tobytes())
        errors.append(difference)
        if len(errors) == len(targets):
            break
    if len(errors) != len(targets):
        raise RuntimeError("Video ended before all selected frames were decoded")
    plan = {
        "source": str(video.resolve()),
        "video": info,
        "fps": fps,
        "decoder": ["pyav", av.__version__],
        "segments": segments,
        "discarded": discarded,
        "threshold": threshold,
        "frame_count": len(errors),
        "alignment_max_mae": max(errors),
    }
    (output / "segments.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan
