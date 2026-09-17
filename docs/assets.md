# 비전 자원과 자동 준비

Schale의 Git 소스·PyPI wheel·sdist에는 게임 이미지, UI 참조, 특징 기술자,
학습 모델을 넣지 않습니다. 학생 추출은 공개 ONNX 숫자 모델을 자동 설치하고,
UI 캡처 대신 기하 구조를 검출하므로 빈 캐시에서도 실행할 수 있습니다.
인벤토리의 장비 모델 및 템플릿 전용 학생 판독에는 별도 로컬 자원이 필요합니다.
데이터·계정·보상 도구는 비전 자원이 필요하지 않습니다.

## 설치와 점검

0.1.2부터 학생 추출의 기본 사용법은 다음 한 명령입니다. 최초 실행에는
패키지 저장소·GitHub Releases·SchaleDB에 연결할 수 있어야 합니다.

```powershell
schale students extract recording.mp4
```

숫자 모델은 약 40MiB이며 고정된 SHA-256과 크기를 검증하고 원자적으로 설치합니다.
동시 실행은 설치 잠금을 공유하고, 유효한 캐시는 네트워크 요청 없이 재사용합니다.
손상된 설치는 검증된 새 모델을 받은 뒤 백업하고 교체합니다. 공개 모델에는
OpenOCR 라이선스·출처가 포함되며 게임 이미지·계정 데이터는 없습니다.
FFmpeg가 PATH에 없으면 PyAV의 CPU 디코더를 사용합니다.

## 선택 사항: 기존 개인 자원 이동

기존 템플릿·장비 모델이 있다면 개인 준비 파일로 이동할 수도 있습니다.
학생 추출의 첫 실행에 필요한 절차는 아닙니다.

```powershell
# 준비된 PC에서 한 번:
schale setup --export schale-setup.zip
# 새 PC의 Downloads에 이 파일을 복사한 뒤:
schale students extract recording.mp4
```

추출 명령은 현재 폴더, 입력 파일의 부모 폴더, 사용자 Downloads 순으로
`schale-setup.zip`을 찾아 설치하고 같은 실행에서 추출을 계속합니다.
`SCHALE_SETUP_BUNDLE`로 준비 파일을 직접 지정할 수도 있습니다.
Python 비전 런타임이 없으면 uv의 별도 실행 환경을 자동 사용합니다.

개인 준비 파일에는 계정 정보·영상·추출 결과·HTTP 캐시가 포함되지 않습니다.
기존 유효한 모델·참조는 보존하며 손상된 문자 모델은 백업 후 복구합니다.
SHA-256은 무결성 검사용이므로 신뢰하는 준비 파일만 사용하세요.
설치된 자원이 유효하면 준비 파일을 다시 읽지 않습니다.

각 자원을 별도로 관리할 때만 다음 명령을 사용합니다.

```powershell
uv tool install "schale[student-ocr,scanner]"
schale assets install my-vision-resources.zip
schale assets info
schale students install-model schale-student-numeric-v1.zip
schale students doctor
```

게임 참조·장비 모델은 첫 번째 ZIP, 공통 문자 모델은 두 번째 ZIP입니다.
신뢰하는 출처와 사용 권한을 확인한 자료만 설치하세요. 체크섬은 손상 검사용이며
제공자의 신원이나 재배포 허락을 보증하지 않습니다. 이 명령은 다운로드하지 않습니다.

자원은 `SCHALE_CACHE_DIR/vision-resources`에 설치하며, 환경 변수가 없으면
`~/.schale/cache/vision-resources`를 사용합니다. 내용 해시별 `bundles/<hash>`와
현재 선택을 담은 `current.json`으로 관리합니다. 같은 번들의 재설치는 중복 저장하지
않고, 변경된 번들은 별도로 저장한 뒤 현재 선택을 원자적으로 바꿉니다. 검증 실패 시
이전 선택을 유지합니다. 이전 ZIP을 다시 설치하면 해당 버전으로 돌아갑니다.
설치 작업은 프로세스 잠금으로 직렬화하며 기존 파일을 덮어쓰지 않습니다.

`assets info`는 모든 파일의 SHA-256을 검사합니다. 학생/인벤토리 실행 시에도 필요한
번들을 검증하고, 없거나 손상됐으면 설치 안내와 함께 실패합니다. 조용히 낮은 정확도의
다른 알고리즘으로 전환하지 않습니다. HTTP 캐시의 `cache update`는 이 로컬 번들을
수정하거나 자동 교체하지 않습니다.

## 자신의 자원 번들 만들기

이미 준비한 참조 데이터나 직접 보정·학습한 결과를 아래 구조에 배치합니다.
`assets pack`은 포장과 무결성 검사를 수행하며 이미지에서 참조를 자동 생성하거나
모델을 학습하는 명령은 아닙니다. 다른 레이아웃의 자료는 별도 보정과 검증이 필요합니다.

```text
my-resources/
  students/
    labels.json
    layout_reference.npz
    header.png
    empty_gear.png
    ghost_gloves.png
    ghost_shoes.png
    ghost_hat.png
    potential_hp_25.png
    potential_attack_25.png
    <labels.json에서 지정한 마스크 PNG들>
  inventory/
    models/
      equipment_classifier.onnx
      equipment_classifier.onnx.data  (ONNX가 외부 가중치를 사용하는 경우)
      class_mapping.json
    icons/                            (선택: 로컬 참조 이미지)
      equipment_icon_*.webp
```

```powershell
schale assets pack my-resources --output my-vision-resources.zip
schale assets install my-vision-resources.zip
```

최소 하나의 완전한 구성요소(`students` 또는 `inventory`)가 필요합니다. 설치는 번들
전체를 선택하므로 둘 다 사용하려면 둘을 함께 포장하세요. 파일 목록과 SHA-256을 담은
`manifest.json`은 명령이 생성합니다. 학생 프로필은 한국어 16:9 화면용이며 번들 프로필
식별자는 `bluearchive-ko-16x9-v1`입니다.

- `labels.json`: `family` (`level`, `tier`, `skill`, `bond`), `value` (정수 또는 `MAX`),
  `image` (같은 디렉터리 기준 마스크 파일명)를 담은 배열입니다. 마스크는
  `schale.students.labels.label_mask`의 출력과 같은 40×160 형태입니다.
- `layout_reference.npz`: 1280×720 기준 좌표의 `left_points`, `right_points` (N×2)와
  대응하는 OpenCV SIFT `left_descriptors`, `right_descriptors` (N×128)입니다.
  기준 특징은 움직이지 않는 UI에서 수집해야 합니다. 원본 그림은 포함하지 않아도 됩니다.
- UI PNG는 `schale.students.vision`의 해당 ROI와 같은 의미의 BGR 참조입니다.
- 장비 모델 입력은 RGB 정규화 float32 N×3×116×146, 출력은 클래스별 logits입니다.
  `class_mapping.json`은 `num_classes`, `class_to_name`, `name_to_class`, `categories`를
  포함하며 학습한 모델의 순서와 일치해야 합니다. 학습·ONNX 내보내기는
  `schale.scanner.training`에서 제공하며 기본 출력은 사용자 캐시의 `training/equipment`입니다.
- 장비 아이콘이 없으면 기존 공유 HTTP 캐시에서 읽거나 정책에 따라 SchaleDB에서
  받습니다. 오프라인 상태의 누락은 결과의 `unavailable_icons`에 기록합니다.

번들은 개인 환경에서 보관하세요. 공개 배포 여부는 각 게임 자료·학습 데이터·모델의
권리 조건에 따라 별도로 판단해야 합니다. ZIP으로 분리했다는 이유로 재배포가 허용되는
것은 아닙니다. 문자 모델 번들 구성은 [학생 인식 문서](students.md)를 참고하세요.
