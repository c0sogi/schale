# 인벤토리 인식

`scanner` extra는 OpenCV·NumPy·Pillow·ONNX Runtime으로 동작합니다. Torch/EasyOCR는 설치하지 않습니다. 문자 모델은 학생 인식과 공유하며 신뢰할 수 있는 번들을 별도 설치합니다.

```powershell
uv tool install "schale[scanner]==0.1.0"
schale assets install ".\my-vision-resources.zip"
schale students install-model ".\schale-student-numeric-v1.zip"
schale scan inventory.png --output inventory.json
```

장비 모델·참조 이미지는 패키지에 포함하지 않습니다. 사용자가 준비한 `inventory`
구성요소가 필요하며 [로컬 자원 구성](assets.md)을 참고하세요.

```python
from pathlib import Path
from schale.scanner import scan_inventory

result = scan_inventory(Path("inventory.png"))
for item in result.items:
    print(item.equipment_id, item.tier, item.quantity, item.quantity_status)
```

`numeric_model=Path(...)` 또는 CLI `--numeric-model`로 모델 경로를 지정할 수 있습니다. 입력은 파일 경로 또는 BGR uint8 배열입니다. 한 이미지의 결과이며 학생 영상의 다중 관측 확정 상태를 부여하지 않습니다.

## 판독과 근거

1. 반복되는 밝은 카드 테두리로 셀 크기·행·열을 찾습니다. 선택된 카드의 금색 테두리는 이웃 격자와 실제 내부 픽셀의 존재를 함께 확인합니다. 잘린 하단 행은 제외합니다.
2. 셀별로 문자 판독 크기를 정규화합니다. 파란 티어 글자를 찾고, 실제 T 문자열이 판독될 때만 수량과 영역을 분리합니다.
3. 공통 ONNX CTC 엔진으로 티어와 수량을 읽습니다. 수량은 전체 영역과 글자 영역의 판독을 비교하며 수용 가능한 값이 충돌하면 보류합니다. 두 영역은 같은 이미지이므로 독립 관측이 아닙니다.
4. ONNX 아이콘 분류기로 후보를 만들고 카탈로그 참조의 SIFT 대응점·RANSAC 변환·정렬 색상/모양을 확인합니다. 티어와 후보 간 점수 차이를 함께 검사합니다.
5. `observations`에 모든 검출 칸의 원본 좌표 ROI, 원문, 점수, 상위 후보, 보류 이유를 기록합니다. 신원이 보류된 칸의 문자 판독도 버리지 않습니다.

점수는 보정된 정답 확률이 아닙니다. 기본 아이콘 점수 0.65, 차순위와의 차이 0.04, 문자 점수 0.65는 개발용 수용 기준이며 독립 촬영 자료에 대한 추가 검증이 필요합니다. 문자 점수는 인식된 장식 접두어 `x`/`T`를 제외한 숫자·소수점·배율 문자의 최솟값입니다. 아이콘 분류기의 클래스 밖 자료는 참조 특징 후보가 제공될 때만 검증할 수 있습니다. `unavailable_icons`는 해당 실행에서 확보하지 못한 카탈로그 참조 목록입니다. 모든 종류의 아이템을 지원한다고 보장하지 않습니다.

## 수량 계약

| 표시 | `quantity` | `displayed_quantity` | 상태 |
|---|---:|---:|---|
| x104 | 104 | 104 | exact |
| x13K | null | 13000 | abbreviated |
| 판독 실패 | null | null | unreadable |

실패를 1이나 0으로 바꾸지 않습니다. `13K`에서 화면에 숨겨진 정확한 수량이나 반올림 구간은 추정하지 않습니다. `items`는 아이콘 신원이 수용된 칸만 포함하며 `unrecognized_cells`는 신원 보류 위치입니다. 수량이 미확정인 아이템도 있으므로 `items` 길이를 완전한 인식 건수로 사용하지 마세요.

## 정확도 대조

```powershell
uv run python scripts/validate_inventory.py inventory.png labels.json report --numeric-model path/to/model --stress
```

보고서는 칸별 대조 JSON과 ROI 오버레이 HTML을 생성합니다. 정답 JSON에는 `schema_version: 1`, 이미지 바이트의 `image_sha256`, `cells` 배열을 둡니다. 각 행은 `[row, column, icon_name, visible_tier, exact_quantity, displayed_quantity, quantity_status]`입니다. 배지가 없는 성장 재료의 `visible_tier`는 null입니다. 정답은 모델 출력으로 만들지 말고 원본을 따로 확인하여 작성하세요.

원본과 축소·확대·JPEG·밝기·위치 이동은 구분하여 보고합니다. 같은 이미지에서 만든 변형의 성공은 별도 촬영이나 다른 아이템 구성에 대한 일반화 증거가 아닙니다. 정답 데이터와 사용자 이미지는 패키지에 포함하지 않습니다.
