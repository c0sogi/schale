"""Reproducible, grouped numeric-patch benchmark; never changes extraction results.

Run with ``uv run --group training python -m schale.students.benchmark --help``. Torch is imported
only by this experimental command, not by the production students extractor.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .identity import read_image
from .labels import BOXES, LabelBank, clean_tier, family, label_mask, similarity
from .video import crop

from ..assets import resolve_resources

CONDITIONS = (
    "clean",
    "jpeg60",
    "jpeg30",
    "scale75",
    "scale50",
    "dark75",
    "bright120",
    "blur06",
    "blur12",
    "shift2",
    "shift4",
    "combined",
)
CALIBRATION_CONDITIONS = {"clean", "jpeg60", "scale75", "dark75", "shift2"}


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resize_roundtrip(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    small = cv2.resize(
        image,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoder failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoder failed")
    return decoded


def translate(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.warpAffine(
        image,
        np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32),
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def perturb(image: np.ndarray, condition: str) -> np.ndarray:
    if condition == "clean":
        return image.copy()
    if condition.startswith("jpeg"):
        return jpeg(image, int(condition[4:]))
    if condition.startswith("scale"):
        return resize_roundtrip(image, int(condition[5:]) / 100)
    if condition in ("dark75", "bright120"):
        return np.clip(
            image.astype(np.float32) * (0.75 if condition == "dark75" else 1.2), 0, 255
        ).astype(np.uint8)
    if condition.startswith("blur"):
        sigma = int(condition[4:]) / 10
        return cv2.GaussianBlur(image, (0, 0), sigma)
    if condition.startswith("shift"):
        dx = int(condition[5:])
        return translate(image, dx, dx / 2)
    if condition == "combined":
        return translate(jpeg(resize_roundtrip(image, 0.5), 40), 2, 1)
    raise ValueError(f"Unknown perturbation: {condition}")


class MaskReader:
    """Reuse the exact production preprocessor without modifying its source."""

    def __init__(self):
        self.canvas = np.zeros((1440, 2560, 3), np.uint8)

    def read(self, patch: np.ndarray, field: str) -> np.ndarray | None:
        box = BOXES[field]
        target = crop(self.canvas, box)
        if target.shape != patch.shape:
            raise ValueError(f"Patch dimensions do not match the recorded ROI: {field}")
        target[:] = patch
        mask = label_mask(self.canvas, field)
        return (
            clean_tier(mask) if mask is not None and family(field) == "tier" else mask
        )


def raw_input(patch: np.ndarray) -> np.ndarray:
    # Raw RGB/BGR is a separate ablation, not a like-for-like matcher comparison.
    return cv2.resize(patch, (160, 40), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)


class Encoder(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        layers: list[nn.Module] = []
        for width in (16, 32, 64, 64):
            layers.extend(
                [
                    nn.Conv2d(channels, width, 3, stride=2, padding=1),
                    nn.BatchNorm2d(width),
                    nn.ReLU(),
                ]
            )
            channels = width
        self.features = nn.Sequential(
            *layers, nn.Flatten(), nn.Linear(64 * 3 * 10, 128)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.features(x), dim=1)


def contrastive_loss(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = features @ features.T / 0.1
    diagonal = torch.eye(len(features), dtype=torch.bool, device=features.device)
    logits = logits.masked_fill(diagonal, -1e4)
    positives = (labels[:, None] == labels[None, :]) & ~diagonal
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -((log_prob * positives).sum(1) / positives.sum(1).clamp_min(1)).mean()


def augmented(image: np.ndarray, rng: np.random.Generator, masked: bool) -> np.ndarray:
    if masked:
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D(
            (w / 2, h / 2), rng.uniform(-1, 1), rng.uniform(0.88, 1.12)
        )
        matrix[:, 2] += rng.uniform(-2, 2, size=2)
        result = cv2.warpAffine(image, matrix, (w, h))
        if rng.random() < 0.5:
            result = cv2.GaussianBlur(result, (3, 3), rng.uniform(0.2, 0.7))
        return result
    result = translate(image, rng.uniform(-2, 2), rng.uniform(-1.5, 1.5))
    result = resize_roundtrip(result, rng.uniform(0.6, 1))
    result = np.clip(
        result.astype(np.float32) * rng.uniform(0.75, 1.25), 0, 255
    ).astype(np.uint8)
    if rng.random() < 0.6:
        result = jpeg(result, int(rng.integers(40, 96)))
    if rng.random() < 0.3:
        result = cv2.GaussianBlur(result, (3, 3), rng.uniform(0.2, 0.8))
    return result


def training_views(
    references: list[dict], mode: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(1234)
    classes = sorted({r["class"] for r in references})
    images, targets = [], []
    for r in references:
        for i in range(32):
            if mode == "mask":
                patch = r["mask"] if i == 0 else augmented(r["mask"], rng, True)
                images.append(patch[None, :, :])
            else:
                patch = r["raw"] if i == 0 else augmented(r["raw"], rng, False)
                images.append(raw_input(patch))
            targets.append(classes.index(r["class"]))
    return np.stack(images), np.array(targets), classes


def train(
    references: list[dict], mode: str, seed: int, steps: int, device: str, output: Path
) -> Encoder:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    images, labels, classes = training_views(references, mode)
    x = torch.from_numpy(images).to(device=device, dtype=torch.float32) / 255
    y = torch.from_numpy(labels).to(device=device, dtype=torch.long)
    groups = [torch.where(y == c)[0] for c in range(len(classes))]
    model = Encoder(1 if mode == "mask" else 3).to(device)
    classifier = nn.Linear(128, len(classes), bias=False).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=0.002,
        weight_decay=0.0001,
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    started = time.perf_counter()
    trace = []
    for step in range(steps):
        chosen = torch.randperm(len(classes), device=device)[:32]
        indices = torch.cat(
            [
                g[torch.randint(len(g), (2,), device=device)]
                for c in chosen
                for g in [groups[int(c)]]
            ]
        )
        features = model(x[indices])
        logits = F.linear(features, F.normalize(classifier.weight, dim=1)) * 12
        loss = contrastive_loss(features, y[indices]) + 0.25 * F.cross_entropy(
            logits, y[indices]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 100 == 0 or step == steps - 1:
            trace.append({"step": step + 1, "loss": float(loss.detach())})
            print(
                f"{mode} seed={seed} step={step + 1}/{steps} loss={float(loss.detach()):.4f}",
                flush=True,
            )
    model.eval()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mode": mode,
            "seed": seed,
            "steps": steps,
            "classes": classes,
            "trace": trace,
            "training_seconds": time.perf_counter() - started,
        },
        output,
    )
    return model


def ranked(values: dict[str, float], field: str) -> dict:
    order = sorted(values.items(), key=lambda item: item[1], reverse=True)
    if not order:
        return {"prediction": None, "score": -1.0, "margin": 0.0}
    label, score = order[0]
    if (
        field == "gear"
        and int(label) > 2
        or field == "ex"
        and label != "MAX"
        and int(label) > 5
    ):
        label = None
    return {
        "prediction": label,
        "score": float(score),
        "margin": float(score - order[1][1]) if len(order) > 1 else 0.0,
    }


def ncc_prediction(mask: np.ndarray | None, field: str, references: list[dict]) -> dict:
    values: dict[str, float] = {}
    if mask is not None:
        for r in references:
            if r["family"] == family(field):
                value = str(r["value"])
                values[value] = max(
                    values.get(value, -1.0), similarity(mask, r["mask"])
                )
    return ranked(values, field)


def is_accepted(row: dict, threshold: tuple[float, float]) -> bool:
    return (
        row["prediction"] is not None
        and row["score"] >= threshold[0]
        and row["margin"] >= threshold[1]
    )


def metrics(rows: list[dict], threshold: tuple[float, float]) -> dict:
    numeric = [r for r in rows if r["value"] is not None]
    controls = [r for r in rows if r["value"] is None]
    accepted = [r for r in numeric if is_accepted(r, threshold)]
    correct = sum(r["prediction"] == str(r["value"]) for r in accepted)
    wrong = len(accepted) - correct
    negative_false = sum(is_accepted(r, threshold) for r in controls)
    top1 = sum(r["prediction"] == str(r["value"]) for r in numeric)
    return {
        "numeric_n": len(numeric),
        "correct": correct,
        "wrong": wrong,
        "abstained": len(numeric) - len(accepted),
        "top1_correct": top1,
        "correct_rate": correct / len(numeric) if numeric else None,
        "wrong_rate": wrong / len(numeric) if numeric else None,
        "coverage": len(accepted) / len(numeric) if numeric else None,
        "accepted_accuracy": correct / len(accepted) if accepted else None,
        "top1_accuracy": top1 / len(numeric) if numeric else None,
        "control_n": len(controls),
        "control_false_accepts": negative_false,
    }


def calibrate(rows: list[dict]) -> tuple[float, float]:
    # The nominal 1% error cap is empirical on grouped calibration data, not a
    # statistical guarantee. Include empty/locked controls in erroneous accepts.
    best = (0, 0, 1.01, 1.0)
    for score in np.arange(0.5, 1.001, 0.01):
        for margin in np.arange(0, 0.301, 0.01):
            m = metrics(rows, (float(score), float(margin)))
            mistakes = m["wrong"] + m["control_false_accepts"]
            total = m["correct"] + mistakes
            if mistakes > 0.01 * total:
                continue
            candidate = (m["correct"], -mistakes, float(score), float(margin))
            if candidate > best:
                best = candidate
    return best[2], best[3]


def prepare_references(source: Path, output: Path) -> list[dict]:
    directory = output / "references"
    directory.mkdir(exist_ok=True)
    bank = LabelBank(resolve_resources("students"))
    refs = []
    for i, (row, mask) in enumerate(bank.templates):
        if mask is None:
            raise ValueError(f"Unreadable reference: {row['image']}")
        image = read_image(source / row["source"])
        raw = crop(image, BOXES[row["field"]]).copy()
        for name, patch in ((f"{i:03}_raw.png", raw), (f"{i:03}_mask.png", mask)):
            ok, data = cv2.imencode(".png", patch)
            if not ok:
                raise RuntimeError("Reference encoding failed")
            (directory / name).write_bytes(data.tobytes())
        refs.append(
            {
                **row,
                "class": f"{row['family']}:{row['value']}",
                "raw": raw,
                "mask": mask,
                "raw_path": f"references/{i:03}_raw.png",
                "mask_path": f"references/{i:03}_mask.png",
                "source_sha256": checksum(source / row["source"]),
            }
        )
    (output / "references.json").write_text(
        json.dumps(
            [{k: v for k, v in r.items() if k not in ("raw", "mask")} for r in refs],
            indent=2,
        ),
        encoding="utf-8",
    )
    return refs


def validate_dataset(dataset: Path, annotations: dict, references: list[dict]) -> None:
    split_groups: dict[str, set] = {"calibration": set(), "test": set()}
    ids = set()
    excluded = set(annotations["reference_student_visits_excluded"])
    for r in annotations["rows"]:
        if r["value"] == "UNANNOTATED":
            raise ValueError("Unannotated test data")
        if r["key"] in ids or checksum(dataset / r["image"]) != r["image_sha256"]:
            raise ValueError("Duplicate ID or modified dataset image")
        ids.add(r["key"])
        split_groups[r["split"]].add(r["visit"])
    if split_groups["calibration"] & split_groups["test"]:
        raise ValueError("Student leakage between calibration and test")
    if excluded & (split_groups["calibration"] | split_groups["test"]):
        raise ValueError("Reference student appears in evaluation")
    actual = {int(r["source"].split("_")[1]) for r in references}
    mapped = {n if n < 67 else n + 1 for n in actual}
    if excluded != mapped:
        raise ValueError("Reference provenance changed since group split")


def build_queries(
    dataset: Path, annotations: dict
) -> tuple[list[dict], list[np.ndarray | None], np.ndarray]:
    reader = MaskReader()
    queries, masks, raw = [], [], []
    for row in annotations["rows"]:
        patch = read_image(dataset / row["image"])
        for condition in CONDITIONS:
            if (
                row["split"] == "calibration"
                and condition not in CALIBRATION_CONDITIONS
            ):
                continue
            changed = perturb(patch, condition)
            queries.append(
                {**row, "family": family(row["field"]), "condition": condition}
            )
            masks.append(reader.read(changed, row["field"]))
            raw.append(raw_input(changed))
    return queries, masks, np.stack(raw)


def evaluate_cnn(
    model: Encoder,
    mode: str,
    queries: list[dict],
    masks: list,
    raw: np.ndarray,
    refs: list[dict],
    device: str,
) -> list[dict]:
    gallery = np.stack(
        [r["mask"][None, :, :] if mode == "mask" else raw_input(r["raw"]) for r in refs]
    )
    inputs = (
        np.stack(
            [
                m[None, :, :] if m is not None else np.zeros((1, 40, 160), np.uint8)
                for m in masks
            ]
        )
        if mode == "mask"
        else raw
    )
    with torch.inference_mode():
        ref_features = model(
            torch.from_numpy(gallery).to(device=device, dtype=torch.float32) / 255
        )
        scores = np.concatenate(
            [
                (
                    model(
                        torch.from_numpy(inputs[i : i + 128]).to(
                            device=device, dtype=torch.float32
                        )
                        / 255
                    )
                    @ ref_features.T
                )
                .cpu()
                .numpy()
                for i in range(0, len(inputs), 128)
            ]
        )
    rows = []
    for i, query in enumerate(queries):
        values: dict[str, float] = {}
        if mode != "mask" or masks[i] is not None:
            for j, r in enumerate(refs):
                if r["family"] == query["family"]:
                    key = str(r["value"])
                    values[key] = max(values.get(key, -1.0), float(scores[i, j]))
        rows.append({**query, **ranked(values, query["field"])})
    return rows


def summarize(rows: list[dict], threshold: tuple[float, float]) -> dict:
    test = [r for r in rows if r["split"] == "test"]
    return {
        "threshold": threshold,
        "calibration": metrics(
            [r for r in rows if r["split"] == "calibration"], threshold
        ),
        "conditions": {
            c: metrics([r for r in test if r["condition"] == c], threshold)
            for c in CONDITIONS
        },
        "clean_families": {
            f: metrics(
                [r for r in test if r["condition"] == "clean" and r["family"] == f],
                threshold,
            )
            for f in ("level", "bond", "tier", "skill")
        },
        "clean_equipment_levels": metrics(
            [
                r
                for r in test
                if r["condition"] == "clean"
                and r["field"].startswith("equipment")
                and r["field"].endswith("_level")
            ],
            threshold,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty; choose a new run directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    refs = prepare_references(args.reference_source, args.output)
    annotations = json.loads(
        (args.dataset / "annotations.json").read_text(encoding="utf-8")
    )
    validate_dataset(args.dataset, annotations, refs)
    config = {
        "steps": args.steps,
        "seeds": args.seeds,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "code_sha256": checksum(Path(__file__)),
        "labels_sha256": checksum(resolve_resources("students") / "labels.json"),
        "annotations_sha256": checksum(args.dataset / "annotations.json"),
        "conditions": CONDITIONS,
        "calibration_conditions": sorted(CALIBRATION_CONDITIONS),
        "reference_n": len(refs),
        "class_n": len({r["class"] for r in refs}),
        "architecture": "4 stride-2 conv blocks (16,32,64,64) + 128D L2 embedding; supervised contrastive + 0.25 cosine CE; nearest reference cosine at inference",
        "no_test_tuning": True,
    }
    (args.output / "protocol.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(
        "Protocol frozen. Training uses only the old reference crops; no evaluation gradients.",
        flush=True,
    )
    queries, masks, raw = build_queries(args.dataset, annotations)
    ncc_rows = [
        {**q, **ncc_prediction(m, q["field"], refs)} for q, m in zip(queries, masks)
    ]
    ncc_threshold = calibrate([r for r in ncc_rows if r["split"] == "calibration"])
    summary: dict[str, Any] = {
        "ncc_original": summarize(ncc_rows, (0.90, 0.045)),
        "ncc_calibrated": summarize(ncc_rows, ncc_threshold),
    }
    (args.output / "ncc_predictions.json").write_text(
        json.dumps(ncc_rows, ensure_ascii=False), encoding="utf-8"
    )
    # All model settings are fixed before any final-test model scores are read.
    for mode in ("mask", "raw"):
        for seed in args.seeds:
            name = f"cnn_{mode}_{seed}"
            model = train(
                refs, mode, seed, args.steps, device, args.output / f"{name}.pt"
            )
            predictions = evaluate_cnn(model, mode, queries, masks, raw, refs, device)
            threshold = calibrate(
                [r for r in predictions if r["split"] == "calibration"]
            )
            summary[name] = summarize(predictions, threshold)
            (args.output / f"{name}_predictions.json").write_text(
                json.dumps(predictions, ensure_ascii=False), encoding="utf-8"
            )
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(name, "completed; calibration threshold", threshold, flush=True)
            del model
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Complete:", args.output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
