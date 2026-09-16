# 개발 안내

Python 3.12+와 uv를 사용합니다. [구조와 확장 계약](docs/architecture.md)을 먼저 확인하세요.

```powershell
# 학습/인벤토리 개발 의존성 없는 회귀 테스트
uv sync --no-default-groups --group test
uv run --no-default-groups --group test pytest tests/test_toolkit.py tests/test_students.py tests/test_students_numeric.py tests/test_students_layout.py tests/test_students_inspector.py tests/test_distribution.py -q

# 기본 개발 환경: 학생 ONNX 추론과 테스트, Torch/EasyOCR 제외
uv sync
uv run pytest -q

# 학습/CNN 비교 실험을 개발할 때만 (CPU Torch 포함)
uv run --group training pytest tests/test_students_benchmark.py -q

# 인벤토리 ONNX 계약 (Torch 불필요)
uv run pytest tests/test_inventory_onnx.py tests/test_scanner.py -q
```

Torch가 없는 환경에서는 학습 전용 `test_students_benchmark.py` 모듈을 건너뜁니다. 로컬 장비 모델이 없으면 해당 모델 통합 검사도 건너뜁니다. 게임 자원 없는 CI는 생성한 도형·배열로 계약을 검증하며 실제 정확도는 별도 평가합니다. 학습 코드를 수정할 때는 반드시 `training` 그룹을 선택해 해당 테스트를 실행하세요. 학습 모듈을 포함하는 프로젝트 전체 타입 검사는 `uv run --group training pyright`로 수행합니다.

단위 테스트는 최소 fixture로 계약을 검증합니다. fixture 추론 모델은 정확도 평가에 쓰지 않습니다. `test_parsing.py`는 SchaleDB 접속 또는 캐시가 필요합니다. `test_scanner.py`의 일부는 별도 예제 이미지가 없으면 건너뜁니다. 실제 정확도는 동의하에 제공된 영상과 독립 정답으로 검증하고 개인 영상·계정 JSON을 저장소/배포본에 넣지 않습니다.

새 기능에는 Python API, 필요한 CLI, 데이터 보존·실패 테스트, 사용 예시를 함께 추가하세요. 기존 `students` 명령과 `Extraction` 형식은 보존합니다. 의존성은 필요한 extra에만 넣고 코어 import가 비전 런타임을 로드하지 않는지 검사합니다. 변경 파일은 Ruff 검사/포맷을 적용하고 프로젝트 전체 Pyright를 확인합니다.

UI 판독 변경은 원본 영상·미사용 촬영 조건 검증이 필요합니다. 기존 입력 통과를 일반화 보증으로 표현하지 않습니다. [배포 절차](docs/releasing.md)의 실제 모델 검증은 CI의 패키지·회귀 검사와 별개입니다.
