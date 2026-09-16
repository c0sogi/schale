# 영상·이미지 입력 계약

`schale.students.extract.extract(source, output, ...)`는 입력을 정규화한 뒤 공통 프레임 분석·학생 병합·근거 검토·내보내기를 수행합니다. 출력의 `results.json`은 입력 종류와 무관하게 같은 Extraction 스키마입니다.

```python
from pathlib import Path
from schale import VideoInput, ImageInput, ImageBatch
from schale.students.extract import extract

extract(VideoInput("recording.mp4"), Path("out-video"))
extract(Path("single.png"), Path("out-single"))
extract(Path("screenshots"), Path("out-directory"))
extract([Path("a.png"), Path("another/b.webp")], Path("out-list"))

batch = ImageBatch([
    ImageInput(Path("a.png").read_bytes(), name="capture-a", timestamp=1.2),
    ImageInput(Path("b.png"), name="capture-b"),
])
extract(batch, Path("out-memory"))
```

`bytes`는 PNG/JPEG/WebP 등의 **인코딩된 이미지 파일 바이트**입니다. NumPy/PIL 객체나 raw RGB 버퍼를 직접 받는 것은 아닙니다. PNG 등으로 인코딩해 전달하세요. 영상/이미지 혼합 리스트, 원격 URL, ZIP은 현재 입력 계약에 없습니다.

## CLI와 매니페스트

```powershell
schale students extract recording.mp4 --output out-video
schale students extract screenshots --output out-directory
schale students extract a.png b.webp another/c.jpg --output out-list
schale students extract images.json --output out-manifest
schale students inspect out-list --output out-list/inspector
```

`images.json`의 상대 경로는 매니페스트 파일 위치 기준입니다.

```json
{
  "schema_version": 1,
  "kind": "schale.images",
  "images": [
    {"path": "captures/a.png", "name": "capture-a", "timestamp": 1.2},
    {"path": "captures/b.png"}
  ]
}
```

폴더는 재귀 탐색하지 않으며 PNG/JPG/JPEG/WebP/BMP만 파일명 자연 정렬합니다 (`image2`가 `image10`보다 먼저). 기존 연락용 합성 이미지 `contact_sheet`는 제외합니다. 목록·매니페스트는 호출자가 준 순서를 유지하며 파일명 중복도 충돌하지 않는 내부 이름으로 저장합니다.

## 근거와 재개

이미지들은 실제 디코딩 픽셀을 해시해 중복 제거합니다. 다른 파일명이나 PNG 압축률로 저장한 같은 화면을 반복 입력해도 `confirmed` 근거가 늘지 않습니다. 손실 압축·리사이즈로 픽셀이 달라진 유사 복제까지 동일 캡처로 판별하는 것은 아닙니다. 확인 상태는 여전히 정답 확률을 뜻하지 않습니다.

입력 지문은 파일 내용·순서·제공한 촬영 시각에 기반합니다. 이름과 mtime만으로 캐시를 믿지 않습니다. 동일 내용 파일을 옮기거나 이름만 바꾼 경우 재개할 수 있지만 내용/순서/시각이 바뀌면 기존 출력으로 재개하지 않습니다. 이미지 저장 직전에도 내용 변경을 확인합니다.

모든 이미지 입력은 디코딩한 PNG를 근거로 저장합니다. JPEG를 PNG로 바꾸어도 기존 손실이 복원되는 것은 아닙니다. 촬영 시각이 없으면 `timestamp_known=false`를 남깁니다. 기존 수치 timestamp 필드에는 0이 들어가지만 이를 실제 촬영 시각으로 해석하지 않습니다. 감독 화면은 이미지 순서축을 사용하고 제공된 촬영 시각만 따로 표시합니다. 영상에서는 기존 30fps 분석 격자·안정 구간 선별을 유지합니다.

단일 화면만 있는 학생은 기본적으로 다중 관측 확정이 되지 않습니다. SchaleDB 엄격 내보내기는 미확정 항목을 보류합니다. 검토 후 교정하거나 위험을 이해한 경우에만 `--allow-observed`를 명시하세요.
