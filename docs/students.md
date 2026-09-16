# 학생 정보 영상 추출과 알고리즘 감독

한국어 Blue Archive PC판의 **16:9 기본 정보 화면**을 넘기는 영상에서 학생별 스크린샷과 육성정보를 추출하는 로컬 도구입니다. 숫자는 SVTRv2-S 문자열 인식 또는 기존 템플릿 방식으로 읽고, 학생 식별·잠김·성급은 UI 시각 정보로 판별합니다. 실행용 모델은 ONNX이며 EasyOCR·PyTorch 없이 사용할 수 있습니다.

## 설치와 실행

Python 3.12+, `uv`, 영상용 `ffmpeg`/`ffprobe`가 필요합니다. 기준 데이터와 스킬 이미지는 공유 HTTP 캐시를 재사용하고, 없는 항목만 받습니다. 초상화는 식별이 모호할 때 후보만 필요에 따라 받습니다. 자동 재검증·오프라인 정책은 [캐시 안내](cache.md)를 따릅니다.

학생 UI 참조와 문자 모델은 패키지에 포함하지 않습니다. [로컬 자원 구성](assets.md)에
따라 자신의 참조 번들과 문자 모델을 준비한 다음 설치합니다.

```powershell
uv tool install "schale[student-ocr]==0.1.0"
schale assets install ".\my-vision-resources.zip"
schale students install-model ".\schale-student-numeric-v1.zip"
schale students doctor
schale students extract "C:\path\recording.mp4" --output "C:\path\result"
schale students inspect "C:\path\result" --output "C:\path\result\inspector"
```

`doctor`는 의존성, FFmpeg 경로, 모델·참조 파일 해시, 캐시 경로를 검사하며 준비가 안 되어 있으면 종료 코드 1을 반환합니다. FFmpeg/FFprobe는 OS에 맞게 별도로 설치하고 PATH에 추가하세요. Windows + Python 3.12에서 실제 영상 처리까지 검증하며, 다른 OS의 실제 영상 처리는 별도 검증이 필요합니다.

Python 프로젝트에서는 `uv add "schale[student-ocr]==0.1.0"`로 설치합니다. 소스 저장소에서는 `uv run --no-dev --extra student-ocr schale ...`을 사용하면 학습 패키지를 설치하지 않습니다. 기본 설치는 데이터·계정·보상·자원 관리 기능을 제공하며, 학생 비전은 `student-ocr`, 템플릿 전용 실행은 `vision`, 인벤토리 스캐너는 `scanner` extra를 사용합니다.

모델·학생 이미지의 기본 캐시는 사용자 홈의 `.schale/cache/`입니다. 기존 데이터 조회의 `SCHALE_CACHE_DIR` 환경변수로 함께 변경할 수 있습니다. 학생 DB 스냅샷은 재현성을 위해 각 추출 결과에도 저장합니다.

```text
<SCHALE_CACHE_DIR 또는 사용자 홈/.schale/cache>/
├─ student-numeric/       # ONNX 모델·메타데이터·출처 문서
├─ vision-resources/      # 별도 설치하는 UI 참조·장비 모델과 활성 버전
├─ http-v1/              # 체크섬 기반 JSON/이미지 객체·SQLite 이력
└─ assets/
   ├─ skills/
   └─ portraits/
```

`--numeric-model "C:\path\model"`로 모델 경로를 지정할 수 있습니다. `--asset-cache`는 이미지 seed 경로이며 새 다운로드는 공유 HTTP 캐시에 저장됩니다. 기본 경로는 작업 디렉터리와 무관하게 사용자 캐시로 결정됩니다. `--resume`은 입력 지문이 같은 실행을 재개합니다. 다른 실행은 새 출력 폴더를 지정하세요. 단일 화면 판독은 기본적으로 다중 프레임 확인 상태가 되지 않습니다.

모델 번들은 `model.onnx`, `metadata.json`, `LICENSE-OpenOCR`, `NOTICE.md`로 구성하며 폴더 또는 평평한 ZIP 구조를 받습니다. `install-model`은 임시 경로에서 SHA-256과 UI 프로파일을 검증한 뒤 설치하며 기존 모델을 덮어쓰지 않습니다. 신뢰할 수 있는 출처의 번들만 설치하세요. 메타데이터와의 해시 일치는 파일 손상 검사이지 배포자의 서명 검증은 아닙니다.

**기본 판독기는 `ctc`입니다.** 모델이 없으면 설치 방법을 안내하고 중단합니다. 모델 없는 템플릿 실행은 `--reader template`, 이전의 자동 선택 동작은 명시적인 `--reader auto`로 사용할 수 있습니다. 손상된 모델은 템플릿으로 대체하지 않습니다.

Python API는 `from schale.students.extract import extract`, 감독 보고서는 `from schale.students.inspector import build_inspector`로 사용하며 경로 인수는 `pathlib.Path`입니다. 원본 패키지의 `StageRewards` 등 기존 공개 API도 유지합니다.

출력:

- `frames/`: 안정 구간마다 선택한 원본 해상도 이미지
- `screenshots/`: 학생 ID와 이름으로 저장한 학생당 대표 스크린샷
- `segments.json`: 구간, 프레임 시각, 디코더, 입력 지문, 제외 구간
- `students.json`: 실행 시점의 학생 DB 스냅샷
- `observations/`: 프레임별 값·근거 영역·매칭 점수
- `results.json`: 중복 학생 병합 및 필드별 일치/충돌 결과
- `review.html`: 브라우저로 여는 검토·수정 화면. 근거를 펼치면 해당 영역을 확대해서 볼 수 있습니다.

## 알고리즘 감독: 시간축·UI 정합·판독 영역

추출 후 `inspect`로 읽기 전용 감독 화면을 만들 수 있습니다. 원본 영상과 기존 `frames/`, `observations/`, `segments.json`, `results.json`이 필요합니다. 원본·판독 결과·내보내기는 변경하지 않습니다.

```powershell
schale students inspect "C:\path\result" --output "C:\path\result\inspector"
# 영상을 이동했다면 같은 내용의 파일을 지정 (SHA-256 대조)
schale students inspect "C:\path\result" --output "C:\path\inspector-new" --video "D:\recording.mp4"
```

`inspector/index.html`을 브라우저에서 열면 다음을 확인할 수 있습니다. 서버·인터넷·인식 모델 재실행은 필요 없습니다.

- 원본 전체의 30fps 분석 시간축, 전환량·임계값·제외 구간·선택/미선택 이유. 원본이 60fps여도 시간축은 실제 분석에 쓴 30fps 격자입니다.
- 원본/정렬 화면 전환, 판독 ROI·식별 영역·전환 감지 영역·보정 전 ROI·정합 inlier 레이어.
- 필드 클릭 시 원본 확대, 정렬 RGB, 글자 위치 제한 결과, 모델 입력 이미지, 저장된 원문/점수/판독 방법.
- 단일 프레임의 판독값과 학생별 최종 합의값, 같은 학생의 다른 시점 관측, 미확인/충돌 프레임 필터.

프레임 선택은 원본 지문 확인 후 같은 선택 로직으로 재현하여 저장 계획과 정확히 대조합니다. 값·원문·점수·변환행렬은 저장 기록입니다. 당시 저장하지 않은 SIFT inlier 좌표와 전처리 이미지는 재계산하며, 저장 변환과의 오차 및 사용 코드 해시를 별도로 기록합니다. 초상화 식별의 개별 대응점, 탈락 매치, 중간 상태 마스크는 이 보고서에 없습니다. 정렬 전체화면은 JPEG 미리보기이고 판독 ROI/모델 입력은 무손실 PNG입니다. 원본 PNG를 상대 경로로 참조하므로 추출 폴더와 감독 폴더의 상대 위치를 유지하세요.

## 검토 후 SchaleDB로 내보내기

`review.html`에서 잘못 읽힌 값이나 빈칸을 수정하고 `corrections.json`을 저장합니다. 빈칸은 0과 다릅니다. 학생 ID 수정 시에는 출력의 DB 스냅샷에 있는 ID를 사용하세요.

```powershell
schale students export "C:\path\result\results.json" `
  --corrections "C:\path\corrections.json" `
  --base "C:\path\existing-schaledb-export.txt" `
  --output "C:\path\reviewed-export.txt"
```

`--base`와 `--corrections`는 선택 사항입니다. **SchaleDB 가져오기는 컬렉션 전체를 교체하므로 기존 데이터가 있으면 먼저 사이트에서 내보내기를 받아 `--base`로 병합하세요.** 도구는 사이트에 자동 업로드하지 않습니다. 새 학생은 필요한 모든 값이 확인된 경우만 내보내며, 제외 사유는 옆의 `.report.json`에 남습니다. 기존 학생의 미확인 필드는 기준 데이터 값을 보존합니다.

기본 내보내기는 `confirmed`/`corrected`만 사용합니다. 단일 관측도 수용하려면 명시적으로 `--allow-observed`를 지정해야 합니다. 사이트 포맷은 장비 **티어**를 저장하며 장비 자체 레벨은 담지 못합니다. 게임의 잠긴 스킬 값 0은 사이트의 최소 스킬 레벨 1로 변환하고 원본 JSON에는 0과 근거를 보존합니다. 능력 개방 키는 체력 `pm`, 공격 `pa`, 치유 `ph`입니다. `lock`은 게임 잠금이 아닌 사이트 설정입니다.

## 인식 범위와 제한

1. 불투명 정보 패널과 이름 띠의 변화로 화면 전환을 분리하고, 안정된 내부 프레임을 최대 3장 선택합니다.
2. 학생별 그림·숫자를 제외한 고정 UI의 SIFT 특징점을 매칭합니다. 좌측 정보 영역과 우측 패널 각각 RANSAC으로 위치·균일 배율·소각도 회전을 추정하고, 대응점 수·분포·잔차를 검사한 뒤 기준 UI로 정렬합니다. 프레임마다 재검출하며 추적 캐시는 사용하지 않습니다. 정렬 실패는 `layout_unresolved`로 기록해 숫자 판독과 구별합니다.
3. 정렬된 화면에서 게임 스킬 원본의 흰 심벌을 네 아이콘과 비교합니다. 공용 아이콘 때문에 후보가 겹치면 초상화 SIFT 대응점과 RANSAC 기하 검증을 사용합니다.
4. CTC 모드에서는 색상을 보존한 글자 영역을 SVTRv2-S로 읽습니다. 인연 숫자는 하트 배경을 분리하고, 티어는 슬롯 위쪽에서 들어오는 아이템 그림을 제외합니다. 장비 레벨 문자열이 불확실할 때는 기준 UI의 두 자리 숫자 칸 전체를 다시 읽습니다. 이 보조 판독은 기존 숫자와 충돌하면 채택하지 않습니다. `Lv` 뒤 구두점과 숫자를 구별하되, 숫자의 문법·범위·최소 글자 신뢰도 기준은 유지합니다. CTC가 보류한 경우 기존 고신뢰 숫자 템플릿도 사용합니다.
5. 성급은 활성 별의 색/배치를, 잠김은 어두운 슬롯을 판별합니다. 미장착 장비·애용품은 티어 배지가 없고 금색 알림 표시가 있는지 함께 검사합니다. 알림만으로 미장착 처리하지 않으므로 장착 중인 업그레이드 가능 장비와 구별됩니다. 능력 개방은 배지의 유무를 확인하고 숫자를 읽습니다.
6. 여러 프레임에서 다른 값이 나오면 다수결로 지우지 않고 `conflict`로 남깁니다. `confirmed`는 여러 관측이 일치했다는 뜻이며, 정확도 보증이나 확률 점수가 아닙니다.

`observations/`와 학생별 `layouts`에는 원본 프레임별 좌·우 변환과 정렬 잔차가 저장됩니다. 변환 좌표계는 기준 1280×720 → 관측 1280×720이며, 검토 화면의 모든 근거 ROI는 원본 이미지의 정규화 좌표로 역매핑됩니다. 정렬 기준 특징점은 패키지의 `data/layout_reference.npz`에 포함되므로 별도 영상 경로가 실행 시 필요하지 않습니다.

CTC와 고신뢰 템플릿이 다른 숫자를 읽어도 `conflict`로 보류합니다. 레벨 템플릿은 공통 `Lv` 부분뿐 아니라 개별 숫자도 일치해야 고신뢰로 인정합니다. 한 모델의 반복적인 오독을 여러 프레임 일치만으로 확정하지 않기 위한 검사입니다. `review_queue.json`에는 미확인·충돌·단일 관측을 포함한 검토 목록이 저장됩니다.

한국어 학생 정보 화면 전용 추출기입니다. 고정 UI 기준점이 보이는 16:9 녹화에서 위치·균일 배율 차이를 보정합니다. 레터박스, 큰 팝업, 패널 내부의 재배치, 다른 언어는 검증되지 않았습니다. 기하 검사에서 허용하는 배율 범위 0.7–1.3은 정확도 보장 범위가 아닙니다. 전환 검출은 여전히 넓은 고정 정규화 영역을 사용하므로 큰 UI 이동 영상의 누락 검증은 별도 필요합니다. 숫자 템플릿은 모든 가능한 숫자를 포함한 범용 숫자 분류기가 아닙니다. 매우 빠르게 지나가 안정 프레임이 없는 구간은 제외 사유를 남깁니다.

보정 템플릿은 `src/schale/students/data/labels.json`에 사람이 읽은 값·원본 화면·필드 출처를 기록합니다. 템플릿을 바꾸면 캐시가 무효화됩니다. 게임/DB 이미지의 권리는 원 권리자에게 있습니다.

학생과 인벤토리 추출은 `schale.vision.text`의 ONNX 전처리·CTC 디코더를 공유합니다. 화면별 영역 추출·문법·수용 기준은 각 모듈에서 관리합니다. 인벤토리 수량은 긴 숫자와 `K` 축약을 별도로 처리합니다.

## 검증

패키지 빌드·배포 체크리스트는 [배포 안내](releasing.md)를 참고하세요. 자체 코드는 MIT이며 제3자 자료는 [별도 고지](../THIRD_PARTY_NOTICES.md)를 따릅니다.

2026-09-16 전수 대조: `0915 (2).mp4`의 198개 방문·485프레임에서 188명을 추출했고, 최종 **3,572필드 모두 정답·확정**, 수동 보정 0, 엄격 내보내기 188명·제외 0을 확인했습니다. 원본 정답은 별도로 전사해 평가에만 사용했습니다. 프레임 단위 판독은 9,214건 정답·1건 보류·오독 0이며, 보류된 값은 다른 프레임에서 확인됐습니다. 기존 영상 560개 검증 항목과 720p·이동·90% 축소·JPEG 재압축 합성 검사 608필드도 모두 통과했습니다. 이 수치는 검증한 입력에 대한 결과이며 모든 향후 촬영 조건의 정확도 보장은 아닙니다.

```powershell
uv run pytest tests/test_students.py tests/test_students_benchmark.py tests/test_students_numeric.py tests/test_students_layout.py -q
uv run pyright src/schale/students src/schale/cli.py tests/test_students.py tests/test_students_numeric.py tests/test_students_layout.py
```

SchaleDB 연동 포맷은 2026-09-15의 [배포 코드](https://schaledb.com/assets/index-8602bb4e.js)에서 확인했습니다. 사이트 변경 시 내보내기 호환성을 다시 확인해야 합니다.

## 숫자 판독 비교 실험

`students.benchmark`는 기존 추출기를 바꾸지 않는 별도 실험입니다. 선택형 `training` 그룹의 PyTorch를 사용해 NCC, 동일 마스크 CNN, 원본 색상 ROI CNN을 비교합니다. 기본 개발 환경에는 PyTorch가 없습니다. CNN은 128차원 특징 임베딩을 학습하고 기준 이미지와 코사인 유사도로 매칭합니다. 기본 실행은 각 CNN을 3개 seed로 학습합니다.

```powershell
uv run --group training python -m schale.students.benchmark --dataset "C:\path\numeric_benchmark" --reference-source "C:\path\old_screenshots" --output "C:\path\benchmark_run"
```

데이터셋에는 사람이 확인한 `annotations.json`과 `patches/`가 필요합니다. 각 주석은 `key`, `visit`, `field`, `split`(calibration/test), `image`, `image_sha256`, `value`를 포함합니다. 활성 숫자가 없는 잠김·빈 슬롯은 `null`이며 게임 값 0과 구별합니다. `reference_student_visits_excluded`는 기준 이미지의 학생을 제외한 그룹 분할을 검증합니다. 현재 분할 검증은 제공된 두 녹화의 방문 순서 대응(새 영상의 67번째 학생 추가)에 맞춰져 있습니다.

학습·기준 이미지는 기존 `students/data/labels.json`의 수동 주석만 사용합니다. 보정 집합에서 거부 임계값을 선택한 뒤 시험 집합에서 정답 수용·오판독·보류·비활성 영역 오수용을 각각 기록합니다. JPEG 재압축, 축소/복원, 밝기, 블러, 위치 이동은 ROI 단위 스트레스 테스트이며 실제 재촬영 일반화 검증은 아닙니다. 모든 프레임을 무작위로 나누면 같은 학생 화면이 학습과 시험에 겹칠 수 있으므로 학생 단위 분할이 필수입니다.

출력에는 실험 설정과 입력/코드 해시, 기준 이미지, 모델 가중치, 각 사례의 후보·점수·차순위 차이, 조건별 수치가 포함됩니다. 보정 집합에서 관측 오류가 적더라도 시험 오류율을 보장하지 않으며, 이 결과만으로 자동 확정 기준을 완화하지 마세요.

```powershell
uv run --group training pytest tests/test_students.py tests/test_students_benchmark.py -q
```

## 이미지 입력

단일 이미지·폴더·목록·메모리 bytes·매니페스트도 같은 추출기를 사용합니다. [입력 계약](inputs.md)의 예시를 참고하세요. `students inspect`도 이미지 결과를 지원하며 시간/FPS 대신 이미지 순서축과 시각 제공 여부를 표시합니다. 위의 30fps 전환/안정 구간 설명은 영상 입력에만 해당합니다.
