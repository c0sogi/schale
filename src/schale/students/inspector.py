"""Read-only replay of selection, geometry and recorded recognition evidence."""

import hashlib
import json
import os
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import cv2
import numpy as np

from .identity import ICON_BOXES, read_image
from .labels import BOXES
from .layout import Layout, LayoutMatcher
from .numeric import localize, numeric_cell, prepare
from .video import crop, frames, selection_plan, signature
from .vision import HEADER, POTENTIAL

MOTION_BOXES = ((0.025, 0.773, 0.27, 0.858), (0.535, 0.31, 0.852, 0.895))


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Evidence path escapes extraction directory: {relative}")
    return path


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_image(path: Path, image) -> None:
    params = [cv2.IMWRITE_JPEG_QUALITY, 90] if path.suffix == ".jpg" else []
    ok, encoded = cv2.imencode(path.suffix, image, params)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    path.write_bytes(encoded.tobytes())


def polygon(box, layout: Layout) -> list:
    """Exact transformed quadrilateral, not its axis-aligned evidence envelope."""
    x0, y0, x1, y1 = box
    matrix = layout.matrices["left" if x1 <= 0.5 else "right"]
    points = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) * (1280, 720)
    return ((points @ matrix[:, :2].T + matrix[:, 2]) / (1280, 720)).tolist()


def model_views(normalized, fields: dict, destination: Path) -> dict:
    """Rebuild preprocessing only; never run a recognizer or replace recorded values."""
    patches = []
    for field, reading in fields.items():
        if not any(e["method"] == "svtrv2-s-rctc" for e in reading["evidence"]):
            continue
        box = BOXES.get(field, POTENTIAL.get(field))
        if box is None:
            continue
        patch = crop(normalized, box)
        localized = localize(patch, field)
        tensor = prepare(localized)
        pixels = np.rint((tensor.transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
        views = {
            "registered": patch,
            "localized": localized,
            "model_input": cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
        }
        raw = " ".join(e["raw"] for e in reading["evidence"])
        if "numeric-cell text=" in raw:
            cell = numeric_cell(patch, field)
            if cell is not None:
                views["fallback"] = cell
        elif "full-label text=" in raw:
            views["fallback"] = patch
        if "fallback" in views:
            fallback_tensor = prepare(views["fallback"])
            fallback_rgb = np.rint(
                (fallback_tensor.transpose(1, 2, 0) + 1) * 127.5
            ).astype(np.uint8)
            views["fallback_input"] = cv2.cvtColor(fallback_rgb, cv2.COLOR_RGB2BGR)
        for name, image in views.items():
            patches.append((field, name, image))
    if not patches:
        return {}
    width = max(p.shape[1] for _, _, p in patches)
    height = sum(p.shape[0] + 2 for _, _, p in patches)
    sheet = np.zeros((height, width, 3), np.uint8)
    result = {}
    y = 0
    for field, name, image in patches:
        h, w = image.shape[:2]
        sheet[y : y + h, :w] = image
        result.setdefault(field, {})[name] = [0, y, w, h]
        y += h + 2
    save_image(destination, sheet)
    return result


def replay_video(plan: dict, source: Path, output: Path, progress: Callable) -> tuple:
    if not source.is_file():
        raise ValueError(
            "Original video is required for the complete selection timeline"
        )
    if file_hash(source) != plan.get("source_fingerprint"):
        raise ValueError("Video fingerprint differs from the extraction")
    fps = plan["fps"]
    command = [
        "ffmpeg",
        "-v",
        "fatal",
        *plan["decoder"],
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"fps={fps},scale=640:360",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    signatures = []
    if plan["decoder"] and plan["decoder"][0] == "pyav":
        from .video_native import sampled_frames

        images = (
            cv2.resize(frame.to_ndarray(format="bgr24"), (640, 360))
            for _, frame in sampled_frames(source, fps)
        )
    else:
        images = frames(command, 640, 360)
    for index, image in enumerate(images):
        signatures.append(signature(image))
        save_image(output / "timeline" / f"{index:06}.jpg", image)
        if index % 300 == 0:
            progress(f"Timeline: {index + 1} analysis samples decoded")
    segments, discarded, threshold, motion = selection_plan(np.stack(signatures), fps)
    if (
        segments != plan["segments"]
        or discarded != plan["discarded"]
        or abs(threshold - plan["threshold"]) > 1e-9
    ):
        raise ValueError("Replayed frame selection differs from the saved plan")
    return motion, np.stack(signatures)


def build_inspector(
    extraction: Path,
    output: Path,
    *,
    video: Path | None = None,
    progress: Callable[[str], None] = print,
) -> dict:
    extraction, output = extraction.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Inspector output is not empty; choose a new directory")
    plan = json.loads((extraction / "segments.json").read_text(encoding="utf-8"))
    result = json.loads((extraction / "results.json").read_text(encoding="utf-8"))
    image_input = plan.get("input_kind") == "images"
    if "fps" not in plan and not image_input:
        raise ValueError("This inspector needs a video extraction with a temporal plan")
    source = video or Path(plan["source"])
    for name in ("timeline", "normalized", "patches"):
        (output / name).mkdir(parents=True, exist_ok=True)
    if image_input:
        samples = []
        for segment in plan["segments"]:
            for frame in segment["frames"]:
                original = read_image(contained(extraction, frame["image"]))[:, :, :3]
                samples.append(signature(cv2.resize(original, (640, 360))))
                save_image(
                    output / "timeline" / f"{frame['sample_index']:06}.jpg", original
                )
        signatures = np.stack(samples)
        motion = np.zeros(len(samples))
    else:
        motion, signatures = replay_video(plan, source, output, progress)
    matcher = None
    final_by_image = {
        e["image"]: student
        for student in result["students"]
        for reading in student["fields"].values()
        for e in reading["evidence"]
    }
    records = []
    max_geometry_error = 0.0
    geometry_missing = 0
    for segment in plan["segments"]:
        for frame in segment["frames"]:
            image_name = frame["image"]
            path = contained(extraction, image_name)
            cached = json.loads(
                (extraction / "observations" / f"{path.stem}.json").read_text(
                    encoding="utf-8"
                )
            )
            raw_image = path.read_bytes()
            if hashlib.sha256(raw_image).hexdigest() != cached["image_sha256"]:
                raise ValueError(f"Source frame changed: {image_name}")
            revision = result["diagnostics"]["revision"]
            if (
                hashlib.sha256(raw_image + revision.encode()).hexdigest()
                != cached["stamp"]
            ):
                raise ValueError(f"Observation revision differs: {image_name}")
            row = cached["observation"]
            original = read_image(path)[:, :, :3]
            layout_info = row["layouts"].get(image_name, {})
            record = {
                **frame,
                "segment": segment["key"],
                "start": segment["start"],
                "end": segment["end"],
                "observation": row,
                "source_url": quote(os.path.relpath(path, output).replace("\\", "/")),
                "layouts": layout_info,
                "anchors": {},
                "regions": [],
                "views": {},
                "final_key": final_by_image.get(image_name, {}).get("key"),
            }
            alignment_mae = float(
                np.mean(
                    np.abs(
                        signature(cv2.resize(original, (640, 360)))
                        - signatures[frame["sample_index"]]
                    )
                )
            )
            if alignment_mae > 10:
                raise ValueError(f"Timeline/source frame mismatch: {image_name}")
            record["timeline_alignment_mae"] = alignment_mae
            if layout_info:
                if matcher is None:
                    if layout_info["right"].get("method", "").startswith("input-"):
                        from .geometry import BootstrapLayoutMatcher

                        matcher = BootstrapLayoutMatcher()
                    elif (
                        layout_info["right"].get("method")
                        == "geometric-edge-registration"
                    ):
                        from .geometry import GeometryMatcher

                        matcher = GeometryMatcher()
                    else:
                        matcher = LayoutMatcher()
                layout = Layout(
                    {k: np.array(v["matrix"]) for k, v in layout_info.items()},
                    layout_info,
                )
                normalized = layout.normalize(original)
                save_image(
                    output / "normalized" / f"{path.stem}.jpg",
                    cv2.resize(normalized, (1280, 720), interpolation=cv2.INTER_AREA),
                )
                record["normalized_url"] = f"normalized/{path.stem}.jpg"
                record["views"] = model_views(
                    normalized, row["fields"], output / "patches" / f"{path.stem}.png"
                )
                record["patch_url"] = (
                    f"patches/{path.stem}.png" if record["views"] else None
                )
                boxes = {
                    **BOXES,
                    **POTENTIAL,
                    "star": (0.20, 0.783, 0.266, 0.827),
                    "weapon_star": (0.790, 0.700, 0.851, 0.738),
                }
                for field, box in boxes.items():
                    record["regions"].append(
                        {
                            "name": field,
                            "kind": "field",
                            "box": box,
                            "polygon": polygon(box, layout),
                        }
                    )
                for name, box in [
                    ("header", HEADER),
                    ("portrait", (0.01, 0.07, 0.50, 0.77)),
                    ("portrait_edge", (0.50, 0.07, 0.51, 0.77)),
                    *[(f"skill_icon_{i + 1}", b) for i, b in enumerate(ICON_BOXES)],
                ]:
                    record["regions"].append(
                        {
                            "name": name,
                            "kind": "identity",
                            "box": box,
                            "polygon": polygon(box, layout),
                        }
                    )
                replayed = matcher.locate(original)
                if replayed is not None:
                    for side, info in replayed.diagnostics.items():
                        reference = np.asarray(
                            info.get("reference_points", [[25, 550], [620, 700]])
                        )
                        a, b = layout.matrices[side], replayed.matrices[side]
                        delta = float(
                            np.max(
                                np.linalg.norm(
                                    reference @ (a[:, :2] - b[:, :2]).T
                                    + a[:, 2]
                                    - b[:, 2],
                                    axis=1,
                                )
                            )
                        )
                        max_geometry_error = max(max_geometry_error, delta)
                        record["anchors"][side] = {
                            **info,
                            "saved_transform_max_delta_px": delta,
                        }
                else:
                    geometry_missing += 1
            records.append(record)
            if len(records) % 25 == 0:
                progress(f"Evidence overlays: {len(records)}/{plan['frame_count']}")
    fields = {f for r in records for f in r["observation"]["fields"]}
    final = {
        s["key"]: {
            "name": s["name"],
            "student_id": s["student_id"],
            "identity_status": s["identity_status"],
            "fields": {
                k: {"value": v["value"], "status": v["status"]}
                for k, v in s["fields"].items()
            },
        }
        for s in result["students"]
    }
    report = {
        "plan": plan,
        "motion": motion.tolist(),
        "frames": records,
        "final": final,
        "motion_boxes": [] if image_input else MOTION_BOXES,
        "field_names": sorted(fields),
        "provenance": {
            "source_sha256": plan["source_fingerprint"],
            "recognition_revision": result["diagnostics"]["revision"],
            "selection_exact_replay": not image_input,
            "input_kind": "images" if image_input else "video",
            "image_order_and_evidence_hashes_verified": image_input,
            "replayed_geometry_max_delta_px_at_720p": max_geometry_error,
            "geometry_replay_missing": geometry_missing,
            "recognition_reexecuted": False,
            "inspector_code_sha256": file_hash(Path(__file__)),
            "preprocessing_code_sha256": file_hash(
                Path(__file__).with_name("numeric.py")
            ),
            "layout_code_sha256": file_hash(Path(__file__).with_name("layout.py")),
        },
    }
    serialized = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    (output / "data.json").write_text(serialized, encoding="utf-8")
    (output / "data.js").write_text(
        "window.REPORT=" + serialized + ";", encoding="utf-8"
    )
    template = Path(__file__).with_name("inspector.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(template, encoding="utf-8")
    draw_preview(report, extraction, output / "preview.png")
    summary = {
        "analysis_samples": len(motion),
        "selected_frames": len(records),
        "segments": len(plan["segments"]),
        "discarded_intervals": len(plan["discarded"]),
        "fields": sum(len(r["observation"]["fields"]) for r in records),
        **report["provenance"],
    }
    (output / "verification.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# 학생 인식 감독 화면\n\n`index.html`을 브라우저에서 여세요. 서버나 인터넷은 필요 없습니다.\n\n"
        "시간축은 원본 전체의 30fps 분석 격자이며 원본 60fps의 모든 프레임이 아닙니다. "
        "프레임 슬라이더·방향키로 이동하고, 선택 프레임 모드·학생 검색·필드 클릭을 사용하세요. "
        "선택되지 않은 프레임에는 인식 결과를 덧씌우지 않습니다.\n\n"
        "초록=숫자 관측, 주황=상태 기반 추론, 빨강=미확인/충돌, 보라=학생 식별/화면 검사, "
        "하늘색=정합 특징점입니다. 색은 정답 여부나 확률을 뜻하지 않습니다.\n\n"
        "프레임 선택은 원본 SHA-256 확인 후 재실행하여 저장 구간/선정 시점과 정확히 대조했습니다. "
        "판독 결과와 변환행렬은 저장 기록이며 숫자·학생 인식을 다시 실행하지 않았습니다. "
        "특징점과 전처리 이미지만 재계산했습니다. 각 프레임의 정합 오차를 표시합니다. "
        "초상화 식별의 개별 대응점, 탈락한 SIFT 매치, 중간 상태 마스크는 기존 기록에 없어 표시하지 않습니다.\n\n"
        "원본 PNG는 기존 추출 폴더의 frames/를 참조합니다. 이 폴더만 따로 이동하지 마세요. "
        "정렬 전체화면은 JPEG 미리보기이며 모델 입력/정렬 ROI는 무손실 PNG입니다.\n",
        encoding="utf-8",
    )
    if image_input:
        (output / "README.md").write_text(
            "# 이미지 묶음 인식 감독\n\nindex.html을 브라우저에서 여세요. 가로축은 시간이나 FPS가 아닌 입력 이미지 순서입니다. "
            "디코딩 픽셀이 같은 중복은 추출 단계에서 제외합니다. 명시적으로 제공한 촬영 시각만 표시합니다.\n\n"
            "저장한 근거 이미지의 SHA-256 및 관측 revision을 확인했습니다. 숫자·학생 인식은 재실행하지 않았으며 "
            "정합 특징점·전처리 이미지만 재계산합니다. ROI와 값은 저장 관측입니다. "
            "근거 원본은 추출 폴더의 frames/를 참조하므로 상대 위치를 유지하세요.\n",
            encoding="utf-8",
        )
    return summary


def draw_preview(report: dict, extraction: Path, destination: Path) -> None:
    """A portable real-frame overlay preview, independent of browser rendering."""
    selected = next(
        (
            r
            for r in report["frames"]
            if any(v["value"] is None for v in r["observation"]["fields"].values())
        ),
        report["frames"][0],
    )
    image = cv2.resize(
        read_image(extraction / selected["image"])[:, :, :3], (1600, 900)
    )
    canvas = np.full((1120, 1600, 3), (30, 22, 17), np.uint8)
    canvas[70:970] = image
    for region in selected["regions"]:
        if region["kind"] != "field":
            continue
        field = region["name"]
        value = selected["observation"]["fields"].get(field, {})
        color = (
            (90, 80, 255)
            if value.get("value") is None
            else (80, 185, 255)
            if value.get("status") == "inferred"
            else (100, 230, 90)
        )
        pts = np.rint(np.asarray(region["polygon"]) * (1600, 900) + (0, 70)).astype(
            np.int32
        )
        cv2.polylines(canvas, [pts], True, color, 2)
        x, y = pts[0]
        short = (
            field.replace("equipment", "E")
            .replace("potential_", "P_")
            .replace("weapon_", "W_")
            .replace("_level", "Lv")
        )
        label = f"{short}={value.get('value')}"
        cv2.putText(
            canvas,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )
    for info in selected["anchors"].values():
        for point in info.get("source_points", []):
            p = np.rint(np.asarray(point) * 1.25 + (0, 70)).astype(int)
            cv2.circle(canvas, tuple(p), 2, (255, 220, 80), -1)
    uncertain = [
        k for k, v in selected["observation"]["fields"].items() if v["value"] is None
    ]
    final = report["final"].get(selected["final_key"], {}).get("fields", {})
    detail = (
        "; ".join(
            f"{k}: frame abstains / final={final.get(k, {}).get('value')}"
            for k in uncertain
        )
        if uncertain
        else "All recorded fields in this frame have values (not an accuracy guarantee)."
    )
    position = (
        f"image #{selected['sample_index'] + 1}"
        if report["plan"].get("input_kind") == "images"
        else f"t={selected['timestamp']:.3f}s"
    )
    lines = [
        f"RECOGNITION INSPECTOR | {selected['segment']} | {position}",
        "Real source frame + saved ROI decisions + replayed SIFT inliers",
        detail,
        "Open index.html: timeline / source vs normalized / field crops / evidence / consensus",
    ]
    for y, line in zip((40, 1005, 1043, 1080), lines, strict=True):
        cv2.putText(
            canvas,
            line,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    save_image(destination, canvas)
