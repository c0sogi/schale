# 배포 안내

## 구성

- `schale-0.1.0-py3-none-any.whl`: Python 코드와 감독 HTML. 게임 참조·학습 모델·사용자 캐시는 제외합니다.
- `schale-0.1.0.tar.gz`: wheel을 다시 빌드할 수 있는 소스·문서·테스트.
- 선택형 로컬 문자 모델 ZIP: 최상위에 `model.onnx`, `metadata.json`, `LICENSE-OpenOCR`, `NOTICE.md`를 둡니다. PyPI 게시 대상이 아닙니다.
- `SHA256SUMS.txt`: 전달하는 파일의 체크섬.

자체 코드의 MIT 라이선스는 모델·게임 이미지·상표를 재허가하지 않습니다. `THIRD_PARTY_NOTICES.md`를 함께 전달합니다. 개인 영상, 결과 JSON, 스크린샷, 캐시, 특징 기술자, 학습 모델을 배포 파일에 넣지 않습니다. 로컬 비전 자원 ZIP과 이전 Git 이력 백업은 공개 업로드 대상에서 제외합니다. [자원 구성](assets.md)을 참고하세요.

## 빌드와 확인

```powershell
uv lock --check
uv run pytest -q
uv run --group training pyright
uv build --no-sources
uv run --no-dev python scripts/check_distribution.py dist/schale-0.1.0-py3-none-any.whl dist/schale-0.1.0.tar.gz
uvx twine check --strict dist/schale-0.1.0-py3-none-any.whl dist/schale-0.1.0.tar.gz
```

학생 추출 테스트만 실행할 때는 `uv run --no-default-groups --group test pytest tests/test_students.py tests/test_students_numeric.py tests/test_students_layout.py tests/test_students_inspector.py tests/test_distribution.py -q`를 사용합니다. 기존 `test_parsing.py`는 실제 SchaleDB 접속을 수행합니다. `test_scanner.py` 일부는 별도 예제 스크린샷이 없으면 건너뜁니다.

깨끗한 환경에서 wheel을 설치하고, 소스 저장소가 아닌 폴더에서 다음 사용 흐름을 확인합니다.

1. 기본 설치의 import / `--version` / `--help`와 모델 설치가 비전 의존성 없이 실행되는지 확인.
2. `student-ocr` extra 설치 후 로컬 자원 설치 → 문자 모델 설치 → `doctor` → 실제 영상 추출 → 엄격 내보내기 → 감독 보고서 생성. 자원 미설치 시 명확한 실패도 확인.
3. 서로 다른 작업 폴더에서도 같은 캐시를 사용하는지 확인. 모델 없음·해시 오류·기존 출력 덮어쓰기 거부도 확인.
4. 학생 ID, 19필드, 확정 상태를 독립적인 정답과 대조. 단순 설치 성공을 인식 검증으로 취급하지 않음.
5. sdist에서 wheel 재빌드 및 패키지 파일 목록 대조.

## 공개 업로드

빌드와 설치 검증은 게시와 별개입니다. 저장소 push, 릴리스 생성, PyPI 업로드는 명시적인 게시 요청이 있을 때 수행합니다. 실제 게시 전 패키지명 소유권·버전 중복·인증 및 제3자 자료 배포 조건을 확인해야 합니다. 오래된 `dist/` 파일까지 한꺼번에 업로드하지 않도록 정확한 두 파일을 지정합니다.

빌드 구성 참고: https://docs.astral.sh/uv/concepts/build-backend/

## 툴킷 회귀와 모델 번들

`tests/test_toolkit.py`는 비전 없는 코어 import, 스키마 검증, 사이트 왕복, 데이터 보존, CLI 연결을 검사합니다. 깨끗한 기본 설치에서 `data fetch`와 `collection import → diff → merge → export`, `rewards calculate`를 확인하세요. 기존 실제 영상의 결과는 공통 계정으로 변환해 신규/구 API 내보내기와 독립 정답을 대조합니다.

모델 폴더에 `model.onnx`, `metadata.json`, `LICENSE-OpenOCR`, `NOTICE.md`를 준비한 뒤 다음으로 ZIP을 만듭니다. 기존 ZIP은 덮어쓰지 않습니다.

```powershell
uv run --no-dev python scripts/bundle_student_model.py path/to/model dist/schale-student-numeric-v1.zip
```

CI는 Windows/Linux 및 Python 3.12/3.13에서 개인 영상·모델·라이브 API 없는 회귀 테스트와 빌드를 정의합니다. 워크플로 파일 추가는 원격 CI 성공이나 다중 플랫폼 영상 실행 검증을 뜻하지 않습니다. validation.json에는 이번 배포본에서 실제 수행한 검증과 제한을 구분해 기록하세요.
