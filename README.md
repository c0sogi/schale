# Schale

**SchaleDB 데이터와 호환되는 Blue Archive 자동화 툴킷.**

게임의 기준 데이터와 사용자의 계정 상태를 연결하고, 반복적인 정보 수집·비교·갱신·계산을 재사용 가능한 Python API와 CLI로 제공합니다. 영상 인식은 여러 입력 방식 중 하나입니다. 새로운 도구도 같은 학생 ID와 공통 데이터 모델을 사용해 기존 도구와 조합할 수 있도록 확장합니다.

## 설계 원칙

- **SchaleDB 호환성:** 기준 ID를 재사용하고, 사이트 저장 형식은 어댑터에서 처리합니다.
- **관측과 사실의 구분:** 값, 확인 상태, 출처를 함께 보관합니다. 미확인은 0이 아니며, 화면에 없었다고 미보유로 판단하지 않습니다.
- **조합 가능한 도구:** 데이터 조회, 계정 처리, 보상 계산, 비전 추출을 독립적으로 실행하거나 Python에서 연결합니다.
- **가벼운 코어:** 일반 데이터 자동화에는 OpenCV·ONNX Runtime·PyTorch가 필요하지 않습니다.
- **검토 가능한 자동화:** 원본을 보존하고 병합·내보내기의 적용 및 보류 이유를 남깁니다. 사이트 업로드나 게임 조작은 자동 실행하지 않습니다.

## 현재 기능

| 기능 | CLI | 결과 |
|---|---|---|
| SchaleDB 기준 데이터 조회 | `data fetch` | 정의 JSON + 출처·내용 해시 |
| 사이트 육성정보 가져오기 | `collection import` | 공통 `AccountSnapshot` |
| 영상 추출 결과 연결 | `collection from-extraction` | 19개 육성 필드·상태·시각 근거 |
| 계정 비교·안전한 갱신 | `collection diff`, `collection merge` | 변경 내역, 병합 결과·보류 보고서 |
| SchaleDB 호환 내보내기 | `collection export` | 사이트 가져오기 문자열 + 보고서 |
| 스테이지 보상 기대값 | `rewards calculate` | 서버·조건별 1회 보상 기대값 |
| 학생 정보 영상·스크린샷 분석 | `students extract`, `students inspect` | 학생별 이미지·판독·감독 화면 |
| 인벤토리 스캔 | `scan` | 아이템 ID·티어·수량 표기·칸별 판독 근거 |

소재 소요량 계산, 육성 계획 최적화, 새로운 화면/게임 조작 자동화는 향후 확장 대상이며 0.1.0의 구현 기능에 포함하지 않습니다.

## 설치

Python 3.12 이상이 필요합니다.

```powershell
# CLI: 데이터·계정·보상 도구
uv tool install "schale==0.1.0"
# Python 프로젝트에 추가할 때
uv add "schale==0.1.0"
```

학생 비전도 사용하려면 위 CLI 설치 대신 다음으로 설치합니다. 기본 설치가 이미 있다면 `uv tool install --reinstall`로 extras를 추가하세요.

```powershell
uv tool install "schale[student-ocr,scanner]==0.1.0"
schale assets install ".\my-vision-resources.zip"
schale students install-model ".\schale-student-numeric-v1.zip"
schale students doctor
```

영상 처리에는 별도 설치한 `ffmpeg`와 `ffprobe`가 PATH에 있어야 합니다. 학생 인식은 `student-ocr`, 인벤토리는 `scanner`, 템플릿 전용은 `vision` extra입니다. 학생·인벤토리 문자 인식은 같은 ONNX 엔진과 별도 설치한 문자 모델을 공유합니다.

**PyPI 패키지는 코드만 제공합니다.** 게임 이미지·UI 참조·장비 모델은 포함하지 않으며 자동으로 내려받지 않습니다. 비전 기능에는 사용자가 별도로 준비한 신뢰할 수 있는 로컬 참조 번들과 문자 모델이 필요합니다. 현재 공개 배포하는 게임 참조 번들은 없습니다. 자원 준비 전에도 데이터·계정·보상 기능은 사용할 수 있습니다. [로컬 비전 자원 구성](docs/assets.md)을 확인하세요.

모든 실행용 extras와 기본 개발·테스트 환경은 Torch/EasyOCR를 설치하지 않습니다. 학습·CNN 비교 실험만 `uv run --group training ...`으로 선택하며 CPU Torch를 사용합니다.

## 계정 워크플로

사이트의 기존 컬렉션을 보존하면서 새 영상의 확인된 값만 반영하는 예입니다. 출력 경로는 새 파일/폴더여야 합니다.

```powershell
schale data fetch students --region kr --output students.json
schale collection import existing-export.txt --catalog students.json --output account.json
schale students extract recording.mp4 --output recording-result
schale collection from-extraction recording-result/results.json --output observed.json
schale collection diff account.json observed.json --output differences.json
schale collection merge account.json observed.json --output merged.json
schale collection export merged.json --base existing-export.txt --output schaledb.txt
```

비전 없이 사이트 내보내기 두 개를 각각 가져와 비교·병합할 수도 있습니다. 기본 병합은 `confirmed`, `corrected`, `imported` 값만 수용합니다. 미확인 신원/필드는 보고서에 보류하며 원본 관측 파일에 남습니다. 관측 시각으로 자동 정렬하지 않으므로 **incoming은 사용자가 최신 또는 우선할 데이터로 선택**해야 합니다.

SchaleDB 가져오기는 컬렉션 전체를 교체합니다. 기존 사이트 내보내기를 보관하고 `--base`로 전달하세요. 도구는 사이트 계정을 변경하지 않습니다. 장비 자체 레벨은 공통 모델에 남지만 사이트 포맷에 없어 내보내지 못합니다. 잠긴 일반 스킬의 0은 사이트 최소값 1로 변환됩니다.

```powershell
schale rewards calculate --stage 1011101 --server Global --output rewards.json
schale rewards calculate --stage 1011101 --condition FirstClear --output first-clear.json
```

기본은 조건 없는 보상이며 `--condition`은 해당 조건의 보상을 추가합니다. 기대값은 확정 드롭 수량이나 최적 파밍 계획이 아닙니다.

## 유연한 입력

```python
from pathlib import Path
from schale import ImageBatch, ImageInput
from schale.students.extract import extract

extract(Path("recording.mp4"), Path("out-video"))
extract(Path("screenshots"), Path("out-folder"))
extract([Path("a.png"), Path("b.webp")], Path("out-list"))
extract(ImageBatch([ImageInput(Path("a.png").read_bytes(), name="capture")]), Path("out-memory"))
```

단일 이미지, 디렉터리, 명시적 이미지 목록, 인코딩된 이미지 bytes, JSON 매니페스트를 받습니다. 동일 픽셀은 중복 제거하여 확인 근거를 부풀리지 않습니다. [입력 계약과 예시](docs/inputs.md)를 참고하세요.

## Python API

```python
from pathlib import Path
from schale import AccountSnapshot, ReferenceCatalog, StageRewards
from schale.adapters.schaledb import from_schaledb, to_schaledb
from schale.collections import compare, merge

catalog = ReferenceCatalog.fetch("students", region="kr")
base = from_schaledb(Path("existing-export.txt").read_text(), catalog=catalog)
incoming = AccountSnapshot.model_validate_json(Path("observed.json").read_text())
changes = compare(base, incoming)
updated, merge_report = merge(base, incoming)
encoded, export_report = to_schaledb(updated)
rewards = StageRewards.from_stages([1011101], server="Global")
```

학생 추출은 `schale.students.extract.extract`, 보상 계산은 `StageRewards`, 계정 교환은 `schale.adapters`에서 제공합니다. 학생 추출 결과는 `Extraction/results.json`과 원본 판독 근거로 구성됩니다.

## 캐시와 지원 범위

공유 캐시는 사용자 홈의 `.schale/cache`이며 `SCHALE_CACHE_DIR`로 변경합니다. 모든 SchaleDB JSON·이미지는 조건부 HTTP 캐시를 사용합니다. 기본 자동 모드는 사용 시 만료된 항목만 갱신하며, 유효 기간 안에는 HTTP 요청을 하지 않습니다. JSON 7일·이미지 30일 TTL, ETag/Last-Modified 재검증, SHA-256 무결성 확인, 콘텐츠 중복 제거, 프로세스 간 요청 합치기, 요청 간격 제한과 오류 재시도 유예를 제공합니다. 사용자 영상/계정은 캐시에 넣지 않습니다.

```powershell
schale cache info
schale cache update                 # 이미 사용한 리소스 중 만료/누락만
schale cache update --refresh       # 유효한 항목도 조건부 재검증
schale cache history
schale cache verify
schale cache policy --mode manual   # 자동 재검증 끄기 (최초 누락은 받음)
schale cache policy --mode offline  # 최초 누락도 받지 않음
schale cache policy --mode auto
```

캐시는 파일 단위 증분입니다. 서버가 변경을 알리면 해당 JSON/이미지를 다시 받으며, JSON 내부 개별 학생만 전송받는 delta API는 사용하지 않습니다. 실패 시 검증된 이전 버전만 사용하고 오류/재시도 시각을 기록합니다. 숫자 모델은 신뢰할 수 있는 로컬 번들을 별도 설치하며 자동 원격 교체하지 않습니다. [캐시 정책과 트래픽](docs/cache.md)을 참고하세요.

학생 비전은 한국어 16:9 정보 화면용입니다. 위치·균일 배율 차이를 보정하지만 모든 언어·레이아웃·촬영 조건을 지원하지 않습니다. `confirmed`는 다중 관측 일치이며 정답 보증이 아닙니다. Windows에서 배포본의 실제 영상 실행을 검증합니다. 다른 OS는 별도 실행 검증이 필요합니다.

## 문서

- [구조와 확장 계약](docs/architecture.md)
- [캐시 정책·증분 업데이트](docs/cache.md), [유연한 입력](docs/inputs.md)
- [로컬 비전 자원 설치·검증·구성](docs/assets.md)
- [학생 추출·검토·알고리즘 감독](docs/students.md)
- [인벤토리 판독·정확도 검증](docs/inventory.md)
- [개발 및 테스트](CONTRIBUTING.md)
- [배포 절차](docs/releasing.md), [변경 이력](CHANGELOG.md)

자체 코드는 MIT입니다. SchaleDB·Blue Archive의 공식 프로젝트가 아니며 모델·게임 이미지·상표의 권리는 원 권리자에게 있습니다. [제3자 고지](THIRD_PARTY_NOTICES.md)를 확인하세요.
