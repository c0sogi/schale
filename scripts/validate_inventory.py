"""Compare inventory recognition with manual labels and render auditable overlays.

Derived image perturbations measure development robustness, not independent
recording generalization. Label format is described in docs/inventory.md.
"""

import argparse
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import time
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from schale.cache import shared_cache
from schale.scanner import scan_inventory
from schale.scanner._preprocessing import load_image
from schale.schema.scanner import ScanResult


def variants(image, stress):
    yield "original", image
    if not stress:
        return
    for factor in (0.75, 0.5, 1.25):
        yield (
            f"scale{int(factor * 100)}",
            cv2.resize(
                image,
                None,
                fx=factor,
                fy=factor,
                interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC,
            ),
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 60])
    assert ok
    yield "jpeg60", cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    yield "dark75", np.clip(image.astype(np.float32) * 0.75, 0, 255).astype(np.uint8)
    yield (
        "translated",
        cv2.copyMakeBorder(
            image, 80, 40, 120, 50, cv2.BORDER_CONSTANT, value=(35, 35, 35)
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--numeric-model", type=Path, required=True)
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    assert (
        hashlib.sha256(args.image.read_bytes()).hexdigest() == labels["image_sha256"]
    ), "Image does not match labels"
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    args.output.mkdir(parents=True)
    (args.output / "ground-truth.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cache = shared_cache()
    cache.stats.clear()
    reports = []
    details = []
    for name, image in variants(load_image(args.image), args.stress):
        start = time.perf_counter()
        assert image is not None
        image = cast(NDArray[np.uint8], image)
        case_error = None
        try:
            result = scan_inventory(image, numeric_model=args.numeric_model)
        except RuntimeError as error:
            case_error = str(error)
            result = ScanResult(
                items=[],
                observations=[],
                unrecognized_cells=[],
                grid_dimensions=(0, 0),
                source_resolution=image.shape[:2],
            )
        (args.output / f"{name}.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        by_position = {cell.grid_position: cell for cell in result.observations}
        items = {item.grid_position: item for item in result.items}
        rows = []
        overlay = image.copy()
        for r, c, icon, tier, quantity, display, status in labels["cells"]:
            cell = by_position.get((r, c))
            item = items.get((r, c))
            correct_id = bool(item and item.icon_name == icon)
            correct_tier = bool(cell and cell.tier_value == tier)
            correct_quantity = bool(
                cell
                and cell.quantity_value == quantity
                and cell.displayed_quantity == display
                and cell.quantity_status == status
            )
            rows.append(
                dict(
                    position=[r, c],
                    identity_correct=correct_id,
                    tier_correct=correct_tier,
                    quantity_correct=correct_quantity,
                    expected_icon=icon,
                    actual_icon=item.icon_name if item else None,
                    expected_tier=tier,
                    actual_tier=cell.tier_value if cell else None,
                    expected_quantity=quantity,
                    expected_display=display,
                    actual_quantity=cell.quantity_value if cell else None,
                    raw_quantity=cell.quantity_text if cell else None,
                    issues=cell.issues if cell else ["missing_cell"],
                )
            )
            if cell:
                color = (
                    (30, 175, 30)
                    if correct_id and correct_tier and correct_quantity
                    else (0, 0, 230)
                )
                x0, y0, x1, y1 = cell.box
                cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
                for box, col in (
                    (cell.tier_box, (255, 160, 0)),
                    (cell.quantity_box, (255, 0, 255)),
                ):
                    xx0, yy0, xx1, yy1 = box
                    cv2.rectangle(overlay, (xx0, yy0), (xx1, yy1), col, 1)
                cv2.putText(
                    overlay,
                    f"{r},{c}: {cell.quantity_text}",
                    (x0, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        cv2.imwrite(str(args.output / f"{name}.png"), overlay)
        expected_positions = {(r[0], r[1]) for r in labels["cells"]}
        report = dict(
            error=case_error,
            condition=name,
            expected_cells=len(rows),
            detected_cells=len(by_position),
            missing_cells=len(expected_positions - set(by_position)),
            extra_cells=len(set(by_position) - expected_positions),
            identity_correct=sum(r["identity_correct"] for r in rows),
            identity_wrong=sum(
                r["actual_icon"] is not None and not r["identity_correct"] for r in rows
            ),
            identity_abstained=sum(r["actual_icon"] is None for r in rows),
            tier_correct=sum(r["tier_correct"] for r in rows),
            quantity_correct=sum(r["quantity_correct"] for r in rows),
            quantity_wrong=sum(
                r["actual_quantity"] is not None and not r["quantity_correct"]
                for r in rows
            ),
            exact_quantity_correct=sum(
                r["quantity_correct"] and r["expected_quantity"] is not None
                for r in rows
            ),
            seconds=round(time.perf_counter() - start, 3),
            rows=rows,
        )
        reports.append(report)
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
        detail_rows = "".join(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row[k]))}</td>"
                for k in (
                    "position",
                    "expected_icon",
                    "actual_icon",
                    "expected_tier",
                    "actual_tier",
                    "expected_quantity",
                    "raw_quantity",
                    "issues",
                )
            )
            + "</tr>"
            for row in rows
        )
        details.append(
            f'<h2>{name}</h2><p>아이콘 {report["identity_correct"]}/25 · 티어/배지 없음 {report["tier_correct"]}/25 · 수량 표기 {report["quantity_correct"]}/25</p><img src="{name}.png"><details><summary>칸별 대조</summary><table><tr><th>칸</th><th>정답 아이콘</th><th>인식 아이콘</th><th>정답 T</th><th>판독 T</th><th>정확 수량</th><th>읽은 문자열</th><th>상태</th></tr>{detail_rows}</table></details>'
        )
    summary = dict(
        schema_version=1,
        source_sha256=labels["image_sha256"],
        torch_installed=importlib.util.find_spec("torch") is not None,
        easyocr_installed=importlib.util.find_spec("easyocr") is not None,
        http_requests=cache.stats.get("requests", 0),
        cases=reports,
        limitations=[
            "One development screenshot; not independent holdout recordings.",
            "21 visible tier badges and 4 no-badge controls per image.",
            "24 exact counts; x13K tests abbreviation recognition, not hidden exact quantity.",
            "Perturbed images are correlated derivatives of the same screenshot.",
        ],
    )
    (args.output / "validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    page = (
        '<!doctype html><meta charset="utf-8"><title>인벤토리 ONNX 검증</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;background:#151922;color:#eef}img{max-width:100%}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #567;padding:6px}details{overflow:auto}</style><h1>인벤토리 ONNX 검증</h1><p>개발용 원본 1장과 파생 변형입니다. 독립 촬영 일반화 검증이 아닙니다. 녹색: 정답 일치, 빨강: 오류/보류, 파랑: 티어 영역, 자홍: 수량 영역.</p><p>수량 x13K는 축약 표기이며 정확 수량은 null입니다. 티어 분모 25는 배지 21개와 배지 없음 4개를 포함합니다.</p>'
        + "".join(details)
    )
    (args.output / "review.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
